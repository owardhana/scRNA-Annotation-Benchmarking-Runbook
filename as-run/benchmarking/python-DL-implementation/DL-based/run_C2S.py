# run_C2S.py
#################################################
# Cell2Sentence (C2S) Function for Python Benchmarking Framework
# Input: Train/test AnnData objects and markers from rank_genes_groups
# Output: Standardized results format for CV framework
#################################################

import subprocess
import tempfile
import os
import pandas as pd
import anndata as ad
import warnings
import pickle
from typing import Dict, List, Any


def run_C2S_function(adata_train: ad.AnnData, adata_test: ad.AnnData, markers: pd.DataFrame) -> Dict[str, Any]:
    """
    Cell2Sentence (C2S) Cell Type Annotation Function

    Purpose: Run Cell2Sentence algorithm using LLM-based cell type prediction
    Inputs:
      - adata_train: Training AnnData object (used for model fine-tuning)
      - adata_test: Test AnnData object to predict
      - markers: Marker genes dataframe from rank_genes_groups() (for interface consistency)
    Outputs: Dict with predictions, true_labels, confidence_scores, cell_ids
    Algorithm: Converts gene expression to cell sentences, fine-tunes Pythia-410M LLM,
               then predicts cell types as natural language strings
    """

    # Default return function for error handling
    def default_return():
        n_test_cells = len(adata_test.obs)
        return {
            'predictions': ['Unknown'] * n_test_cells,
            'true_labels': list(adata_test.obs['Ground_Truth_Celltype']),
            'confidence_scores': [0.0] * n_test_cells,
            'cell_ids': list(adata_test.obs.index)
        }

    # Validate input data
    if 'Ground_Truth_Celltype' not in adata_train.obs.columns:
        warnings.warn("Ground_Truth_Celltype not found in training data")
        return default_return()

    if 'Ground_Truth_Celltype' not in adata_test.obs.columns:
        warnings.warn("Ground_Truth_Celltype not found in test data")
        return default_return()

    # Check conda environment availability (direct directory check)
    conda_env_path = "<WORKSTATION_HOME>/miniconda3/envs/cell2sentence_env"
    if not os.path.isdir(conda_env_path):
        warnings.warn(f"cell2sentence_env conda environment not found at {conda_env_path}. Please create it first.")
        return default_return()

    # Create temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Prepare file paths
            train_h5ad_path = os.path.join(temp_dir, "adata_train.h5ad")
            test_h5ad_path = os.path.join(temp_dir, "adata_test.h5ad")
            results_path = os.path.join(temp_dir, "results.pkl")

            # Get common genes between train and test
            common_genes = list(set(adata_train.var.index) & set(adata_test.var.index))
            if len(common_genes) < 1:
                warnings.warn("Too few common genes between train and test datasets")
                return default_return()

            # Subset both datasets to common genes and ensure consistent ordering
            adata_train_subset = adata_train[:, sorted(common_genes)].copy()
            adata_test_subset = adata_test[:, sorted(common_genes)].copy()

            # Save AnnData objects for inter-process communication
            adata_train_subset.write_h5ad(train_h5ad_path)
            adata_test_subset.write_h5ad(test_h5ad_path)

            # Create Python script for C2S execution
            c2s_script = f'''
import sys
import os
import pickle
import time
import torch
from memory_profiler import memory_usage

try:
    import cell2sentence as cs
    from cell2sentence.tasks import predict_cell_types_of_data
    from transformers import TrainingArguments
    import anndata as ad

    # Load data (NOT measured)
    adata_train = ad.read_h5ad('{train_h5ad_path}')
    adata_test = ad.read_h5ad('{test_h5ad_path}')

    # Rename Ground_Truth_Celltype -> cell_type (C2S expects 'cell_type')
    adata_train.obs['cell_type'] = adata_train.obs['Ground_Truth_Celltype'].astype(str)
    adata_test.obs['cell_type'] = adata_test.obs['Ground_Truth_Celltype'].astype(str)
    print("checkpoint1")
    
    # C2S prompt template for cell_type_prediction requires organism, tissue, sex,
    # batch_condition columns. Add placeholder values if not present.
    _placeholder_cols = {{
        'organism': 'Homo sapiens',
        'tissue': 'unknown',
        'sex': 'unknown',
        'batch_condition': 'unknown',
    }}
    for _col, _val in _placeholder_cols.items():
        if _col not in adata_train.obs.columns:
            adata_train.obs[_col] = _val
        if _col not in adata_test.obs.columns:
            adata_test.obs[_col] = _val
    
    print("checkpoint2")
    
    # Device checkpoint
    if torch.cuda.is_available():
        _gpu_name = torch.cuda.get_device_name(0)
        _gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[C2S] Device: GPU ({{_gpu_name}}, {{_gpu_mem_gb:.1f}} GB VRAM)", flush=True)
    else:
        print("[C2S] Device: CPU (no CUDA GPU detected)", flush=True)

    # --- method execution (measured) ---
    _timing = [0.0]

    print("checkpoint3")
    
    def _run_method():
        if torch.cuda.is_available():
            print("[C2S] Running with GPU acceleration.", flush=True)
            torch.cuda.reset_peak_memory_stats()
        else:
            print("[C2S] Falling back to CPU. No GPU detected.", flush=True)

        _t0 = time.perf_counter()

        # Step 1: Convert training AnnData to Arrow dataset
        print("[C2S] Converting training data to cell sentences...", flush=True)
        _label_cols = ['cell_type', 'organism', 'tissue', 'sex', 'batch_condition']
        train_arrow_ds, train_vocabulary = cs.CSData.adata_to_arrow(
            adata=adata_train,
            random_state=1234,
            sentence_delimiter=' ',
            label_col_names=_label_cols
        )

        # Step 2: Create CSData for training
        train_csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=train_arrow_ds,
            vocabulary=train_vocabulary,
            save_dir='{temp_dir}',
            save_name='c2s_train',
            dataset_backend='arrow'
        )

        # Step 3: Convert test AnnData to Arrow dataset
        print("[C2S] Converting test data to cell sentences...", flush=True)
        test_arrow_ds, test_vocabulary = cs.CSData.adata_to_arrow(
            adata=adata_test,
            random_state=1234,
            sentence_delimiter=' ',
            label_col_names=_label_cols
        )

        # Step 4: Create CSData for test
        test_csdata = cs.CSData.csdata_from_arrow(
            arrow_dataset=test_arrow_ds,
            vocabulary=test_vocabulary,
            save_dir='{temp_dir}',
            save_name='c2s_test',
            dataset_backend='arrow'
        )

        # Step 5: Load pretrained C2S model from HuggingFace
        print("[C2S] Loading pretrained model...", flush=True)
        csmodel = cs.CSModel(
            model_name_or_path='vandijklab/C2S-Pythia-410m-cell-type-prediction',
            save_dir='{temp_dir}',
            save_name='c2s_model'
        )

        # Step 6: Fine-tune on training data
        print("[C2S] Fine-tuning on training data...", flush=True)
        output_dir = os.path.join('{temp_dir}', 'c2s_output')
        os.makedirs(output_dir, exist_ok=True)  # C2S writes data_split_indices_dict.pkl here
        train_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            bf16=True,
            fp16=False,
            lr_scheduler_type='cosine',
            warmup_ratio=0.05,
            logging_steps=50,
            logging_strategy='steps',
            eval_steps=50,
            eval_strategy='steps',
            save_strategy='no',
            report_to='none',
            load_best_model_at_end=False,
        )

        csmodel.fine_tune(
            csdata=train_csdata,
            task='cell_type_prediction',
            train_args=train_args,
            loss_on_response_only=False,
            top_k_genes=200,
            max_eval_samples=500,
        )

        # Step 7: Predict on test data
        print("[C2S] Predicting cell types...", flush=True)
        raw_predictions = predict_cell_types_of_data(
            csdata=test_csdata,
            csmodel=csmodel,
            n_genes=200
        )

        _timing[0] = time.perf_counter() - _t0
        _peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        _peak_vram_mb = _peak_vram_bytes / (1024 * 1024)
        return raw_predictions, _peak_vram_mb

    mem_list, (raw_predictions, peak_vram_mb) = memory_usage(
        (_run_method, [], {{}}), interval=0.1, include_children=True, retval=True)
    peak_system_memory_mb = max(mem_list)
    method_walltime = _timing[0]

    # Post-process predictions: strip trailing periods
    predictions = [str(p).rstrip('.').strip() if p else 'Unknown' for p in raw_predictions]
    true_labels = list(adata_test.obs['Ground_Truth_Celltype'])
    cell_ids = list(adata_test.obs.index)

    results = {{
        'predictions': predictions,
        'true_labels': true_labels,
        'cell_ids': cell_ids,
        'peak_system_memory_mb': peak_system_memory_mb,
        'peak_vram_mb': peak_vram_mb,
        'method_walltime': method_walltime,
    }}

    print(f"Peak system memory: {{peak_system_memory_mb:.2f}} MB, VRAM: {{peak_vram_mb:.2f}} MB, walltime: {{method_walltime:.2f}}s")
    print(f"Predictions: {{len(predictions)}} cells, unique: {{len(set(predictions))}}")

    # Save results
    with open('{results_path}', 'wb') as f:
        pickle.dump(results, f)

except Exception as e:
    print(f"C2S execution failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

            # Execute C2S in conda environment with real-time stdout streaming
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend
            # Ensure CUDA env vars propagate (preserve CUDA_VISIBLE_DEVICES if set)
            if 'CUDA_VISIBLE_DEVICES' not in env:
                env['CUDA_VISIBLE_DEVICES'] = '0'
            # Force Python subprocess to flush stdout immediately (no buffering)
            env['PYTHONUNBUFFERED'] = '1'

            import threading
            process = subprocess.Popen(
                [
                    "<WORKSTATION_HOME>/miniconda3/condabin/conda",
                    "run",
                    "--no-capture-output",  # child writes directly to our pipes (enables real-time streaming)
                    "-n", "cell2sentence_env",
                    "python",
                    "-u",  # unbuffered stdout
                    "-c", c2s_script
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Capture stderr in background thread to avoid deadlock
            stderr_lines = []
            def _read_stderr():
                for line in process.stderr:
                    stderr_lines.append(line)
            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            # Stream stdout line-by-line in real time
            stdout_lines = []
            for line in process.stdout:
                print(line, end='', flush=True)
                stdout_lines.append(line)

            process.wait(timeout=86400)
            stderr_thread.join(timeout=10)
            stderr_output = ''.join(stderr_lines)

            if process.returncode != 0:
                warnings.warn(f"C2S execution failed: {stderr_output}")
                return default_return()

            # Load results from pickle file
            if not os.path.exists(results_path):
                warnings.warn("C2S results file not found")
                return default_return()

            with open(results_path, 'rb') as f:
                results = pickle.load(f)

            cell_predictions = results['predictions']
            true_labels = results['true_labels']
            cell_ids = results['cell_ids']

            # Ensure predictions have correct length
            if len(cell_predictions) != len(adata_test.obs):
                warnings.warn(f"Prediction length mismatch. Expected: {len(adata_test.obs)}, Got: {len(cell_predictions)}")
                return default_return()

            # Convert predictions to strings and handle missing values
            cell_predictions = [str(pred) if pd.notna(pred) else "Unknown" for pred in cell_predictions]

            # C2S provides no native confidence scores; use constant placeholder
            confidence_scores = [0.0 if pred == "Unknown" else 0.8 for pred in cell_predictions]

            print(f"C2S completed successfully:")
            print(f"  - Total predictions: {len(cell_predictions)}")
            print(f"  - Unknown predictions: {sum(1 for p in cell_predictions if p == 'Unknown')}")
            print(f"  - Unique predictions: {len(set(cell_predictions))}")

            # Return standardized format
            return {
                'predictions': cell_predictions,
                'true_labels': true_labels,
                'confidence_scores': confidence_scores,
                'cell_ids': cell_ids,
                'peak_memory_mb':  results.get('peak_system_memory_mb', None),
                'peak_vram_mb':    results.get('peak_vram_mb', None),
                'method_walltime': results.get('method_walltime', None),
            }

        except subprocess.TimeoutExpired:
            warnings.warn("C2S execution timed out after 24 hours")
            return default_return()
        except Exception as e:
            warnings.warn(f"C2S error: {str(e)}")
            return default_return()


# For backward compatibility
run_C2S = run_C2S_function
