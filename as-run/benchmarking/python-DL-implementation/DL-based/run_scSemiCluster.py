# run_scSemiCluster.py
#################################################
# scSemiCluster Function for Python Benchmarking Framework
# Input: Train/test AnnData objects and markers from rank_genes_groups
# Output: Standardized results format for CV framework
#################################################

import subprocess
import tempfile
import os
import pandas as pd
import anndata as ad
import numpy as np
import warnings
import pickle
from typing import Dict, List, Any


def run_scSemiCluster_function(adata_train: ad.AnnData, adata_test: ad.AnnData, markers: pd.DataFrame) -> Dict[str, Any]:
    """
    scSemiCluster Cell Type Annotation Function

    Purpose: Run scSemiCluster algorithm using semi-supervised deep learning
    Inputs:
      - adata_train: Training AnnData object (labeled reference)
      - adata_test: Test AnnData object to predict (unlabeled query)
      - markers: Marker genes dataframe from rank_genes_groups() (for interface consistency)
    Outputs: Dict with predictions, true_labels, confidence_scores, cell_ids
    Algorithm: Uses ZINB autoencoder with semi-supervised softmax classification,
               pairwise cross-entropy constraint, and KMeans clustering loss.
               Annotates test cells via contingency-matrix majority vote mapping.
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

    # Check conda environment availability
    conda_env_path = "<WORKSTATION_HOME>/miniconda3/envs/scSemiCluster_env"
    if not os.path.isdir(conda_env_path):
        warnings.warn(f"scSemiCluster_env conda environment not found at {conda_env_path}. Please create it first.")
        return default_return()

    # Create temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Prepare file paths — use numpy/CSV to avoid anndata version incompatibility
            train_X_path = os.path.join(temp_dir, "train_X.npy")
            test_X_path = os.path.join(temp_dir, "test_X.npy")
            train_obs_path = os.path.join(temp_dir, "train_obs.csv")
            test_obs_path = os.path.join(temp_dir, "test_obs.csv")
            results_path = os.path.join(temp_dir, "results.pkl")

            # Get common genes between train and test
            common_genes = list(set(adata_train.var.index) & set(adata_test.var.index))
            if len(common_genes) < 1:
                warnings.warn("Too few common genes between train and test datasets")
                return default_return()

            # Subset both datasets to common genes and ensure consistent ordering
            sorted_genes = sorted(common_genes)
            adata_train_subset = adata_train[:, sorted_genes].copy()
            adata_test_subset = adata_test[:, sorted_genes].copy()

            # Save as numpy arrays + CSV metadata (avoids anndata version issues)
            import scipy.sparse
            train_X = adata_train_subset.X
            if scipy.sparse.issparse(train_X):
                train_X = train_X.toarray()
            np.save(train_X_path, np.array(train_X, dtype=np.float32))

            test_X = adata_test_subset.X
            if scipy.sparse.issparse(test_X):
                test_X = test_X.toarray()
            np.save(test_X_path, np.array(test_X, dtype=np.float32))

            # Save obs metadata
            adata_train_subset.obs[['Ground_Truth_Celltype']].to_csv(train_obs_path)
            adata_test_subset.obs[['Ground_Truth_Celltype']].to_csv(test_obs_path)

            # Determine project directory (where scSemiCluster_package/ lives)
            project_dir = os.path.dirname(os.path.abspath(__file__))

            # Create Python script for scSemiCluster execution
            scsemi_script = f'''
import sys
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project directory so scSemiCluster_package is importable
sys.path.insert(0, '{project_dir}')

try:
    import numpy as np
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    import time
    import math
    import random
    import tensorflow as tf
    from memory_profiler import memory_usage

    # ---- Load data (numpy/CSV — no anndata file reading needed) ----
    X_train_raw = np.load('{train_X_path}')
    X_test_raw = np.load('{test_X_path}')
    train_obs = pd.read_csv('{train_obs_path}', index_col=0)
    test_obs = pd.read_csv('{test_obs_path}', index_col=0)

    # Reconstruct AnnData objects using local environment's anndata version
    adata_train = ad.AnnData(X_train_raw)
    adata_train.obs = train_obs
    adata_test = ad.AnnData(X_test_raw)
    adata_test.obs = test_obs

    # ---- Method execution (measured) ----
    _timing = [0.0]

    def _run_method():
        if tf.test.is_gpu_available():
            print("[scSemiCluster] Running with GPU acceleration.", flush=True)
        else:
            print("[scSemiCluster] Falling back to CPU. No GPU detected.", flush=True)

        _t0 = time.perf_counter()

        n_train = adata_train.n_obs
        n_test = adata_test.n_obs

        # Extract data matrices (densify if sparse)
        if hasattr(adata_train.X, 'toarray'):
            X_train = adata_train.X.toarray()
        else:
            X_train = np.array(adata_train.X)
        if hasattr(adata_test.X, 'toarray'):
            X_test = adata_test.X.toarray()
        else:
            X_test = np.array(adata_test.X)

        # Ensure non-negative integer-like counts for ZINB model
        X_train = np.ceil(np.abs(X_train)).astype(np.float32)
        X_test = np.ceil(np.abs(X_test)).astype(np.float32)

        # Concatenate train + test
        X_full = np.vstack([X_train, X_test])
        count_X = X_full.copy()

        # Create cellname and batch_label arrays
        cellname_train = np.array(adata_train.obs['Ground_Truth_Celltype'].astype(str))
        cellname_test = np.array(adata_test.obs['Ground_Truth_Celltype'].astype(str))
        cellname = np.concatenate([cellname_train, cellname_test])

        # batch_label: 0 = train (labeled), 1 = test (unlabeled)
        batch_label = np.concatenate([
            np.zeros(n_train, dtype=int),
            np.ones(n_test, dtype=int)
        ])

        # Create AnnData for preprocessing
        adata_full = sc.AnnData(X_full)
        adata_full.obs['celltype'] = cellname
        adata_full.obs['batch'] = batch_label

        # Preprocessing
        from scSemiCluster_package.scSemiCluster_preprocess import normalize
        highly_genes = 1000
        if X_full.shape[1] <= highly_genes:
            highly_genes = None

        adata_proc = normalize(adata_full, highly_genes=highly_genes,
                               size_factors=True, normalize_input=True, logtrans_input=True)

        X_proc = adata_proc.X.astype(np.float32)
        cellname_proc = np.array(adata_proc.obs['celltype'])
        batch_label_proc = np.array(adata_proc.obs['batch']).astype(int)

        # Map surviving cell indices back to original
        surviving_indices = np.array(adata_proc.obs.index, dtype=int)

        # Subset count_X to match processed data
        if highly_genes is not None:
            hvg_indices = np.array(adata_proc.var.index, dtype=int)
            count_X_proc = count_X[np.ix_(surviving_indices, hvg_indices)]
        else:
            gene_indices = np.array(adata_proc.var.index, dtype=int)
            count_X_proc = count_X[np.ix_(surviving_indices, gene_indices)]

        assert X_proc.shape == count_X_proc.shape, \\
            f"Shape mismatch: X_proc {{X_proc.shape}} vs count_X_proc {{count_X_proc.shape}}"

        size_factor = np.array(adata_proc.obs.size_factors).reshape(-1, 1).astype(np.float32)

        # Determine cluster and batch numbers
        cluster_num = len(np.unique(cellname_proc))
        batch_num = len(np.unique(batch_label_proc))
        n_genes = X_proc.shape[1]
        dims = [n_genes, 256, 64, 32]

        # Hyperparameters (paper defaults)
        batch_size = 256
        random_seed = 8888
        alpha = 0.01
        beta = 0.1
        gamma = 0.1
        learning_rate = 1e-4
        gpu_option = "0"

        # Build and train model
        tf.reset_default_graph()
        from scSemiCluster_package.scSemiCluster_model import scSemiCluster
        model = scSemiCluster(
            dataname="benchmark",
            dims=dims,
            batch_num=batch_num,
            cluster_num=cluster_num,
            alpha=alpha, beta=beta, gamma=gamma,
            learning_rate=learning_rate,
            mode="cdec", distrib="ZINB", constraint=True
        )

        annotated_test_accuracy, test_ARI, test_prediction_matrix, latent_repre = \\
            model.train(X_proc, count_X_proc, size_factor, cellname_proc,
                        batch_label_proc, batch_size, random_seed, gpu_option)

        print(f"scSemiCluster annotation accuracy: {{annotated_test_accuracy}}")
        print(f"scSemiCluster test ARI: {{test_ARI}}")

        # Extract predictions for test cells
        # test_prediction_matrix columns: "true label", "true cell type",
        #                                  "cluster label", "annotation cell type"
        predicted_labels = np.array(test_prediction_matrix["annotation cell type"])
        true_labels_from_model = np.array(test_prediction_matrix["true cell type"])

        # Map predictions back to original test cell order
        # surviving_indices tells us which original cells survived filtering
        test_mask_in_surviving = (batch_label_proc != 0)
        surviving_test_original_indices = surviving_indices[test_mask_in_surviving]

        # Build full prediction arrays for ALL original test cells
        predictions = ["Unknown"] * n_test
        confidence_scores = [0.0] * n_test
        true_labels = list(cellname_test)
        cell_ids = list(adata_test.obs.index)

        for i, orig_idx in enumerate(surviving_test_original_indices):
            test_position = orig_idx - n_train  # offset by train size
            if 0 <= test_position < n_test:
                pred = predicted_labels[i]
                predictions[test_position] = pred
                confidence_scores[test_position] = 0.0 if pred == "unassigned" else 1.0

        # Convert "unassigned" to "Unknown" for framework consistency
        predictions = ["Unknown" if p == "unassigned" else p for p in predictions]

        _timing[0] = time.perf_counter() - _t0
        return predictions, confidence_scores, true_labels, cell_ids

    mem_list, (predictions, confidence_scores, true_labels, cell_ids) = memory_usage(
        (_run_method, [], {{}}), interval=0.1, include_children=True, retval=True)
    peak_system_memory_mb = max(mem_list)
    method_walltime = _timing[0]

    # Try to get GPU memory info
    peak_vram_mb = 0.0

    results = {{
        'predictions': list(predictions),
        'confidence_scores': confidence_scores,
        'true_labels': true_labels,
        'cell_ids': cell_ids,
        'peak_system_memory_mb': peak_system_memory_mb,
        'peak_vram_mb': peak_vram_mb,
        'method_walltime': method_walltime,
    }}

    print(f"Peak system memory: {{peak_system_memory_mb:.2f}} MB, walltime: {{method_walltime:.2f}}s")

    # Save results
    with open('{results_path}', 'wb') as f:
        pickle.dump(results, f)

    print(f"Predictions: {{len(predictions)}} cells, unique: {{set(predictions)}}")

except Exception as e:
    print(f"scSemiCluster execution failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

            # Execute scSemiCluster in conda environment
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend

            result = subprocess.run(
                [
                    "<WORKSTATION_HOME>/miniconda3/condabin/conda",
                    "run",
                    "-n", "scSemiCluster_env",
                    "python",
                    "-c", scsemi_script
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour timeout for training
            )

            if result.returncode != 0:
                warnings.warn(f"scSemiCluster execution failed: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return default_return()

            # Print subprocess output
            print(result.stdout)

            # Load results from pickle file
            if not os.path.exists(results_path):
                warnings.warn("scSemiCluster results file not found")
                return default_return()

            with open(results_path, 'rb') as f:
                results = pickle.load(f)

            cell_predictions = results['predictions']
            confidence_scores = results['confidence_scores']
            true_labels = results['true_labels']
            cell_ids = results['cell_ids']

            # Ensure predictions have correct length
            if len(cell_predictions) != len(adata_test.obs):
                warnings.warn(f"Prediction length mismatch. Expected: {len(adata_test.obs)}, Got: {len(cell_predictions)}")
                return default_return()

            # Convert predictions to strings and handle missing values
            cell_predictions = [str(pred) if pd.notna(pred) else "Unknown" for pred in cell_predictions]

            # Process confidence scores
            confidence_scores = [
                0.0 if pred == "Unknown" else (max(0.0, min(1.0, float(conf))) if pd.notna(conf) and isinstance(conf, (int, float)) else 0.5)
                for pred, conf in zip(cell_predictions, confidence_scores)
            ]

            print(f"scSemiCluster completed successfully:")
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
            warnings.warn("scSemiCluster execution timed out after 24 hours")
            return default_return()
        except Exception as e:
            warnings.warn(f"scSemiCluster error: {str(e)}")
            return default_return()


# For backward compatibility
run_scSemiCluster = run_scSemiCluster_function
