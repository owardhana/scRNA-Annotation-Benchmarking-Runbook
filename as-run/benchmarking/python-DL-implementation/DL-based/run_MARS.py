# run_MARS.py
#################################################
# MARS Function for Python Benchmarking Framework
# Input: Train/test AnnData objects and markers from rank_genes_groups
# Output: Standardized results format for CV framework
#################################################

import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
import anndata as ad
import time
import warnings
import pickle
from typing import Dict, List, Any, Optional


def run_MARS_function(adata_train: ad.AnnData, adata_test: ad.AnnData, markers: pd.DataFrame) -> Dict[str, Any]:
    """
    MARS Cell Type Annotation Function

    Purpose: Run MARS (Meta-learning Approach for Recognition in Single-cell data)
    Inputs:
      - adata_train: Training AnnData object (used for meta-learning)
      - adata_test: Test AnnData object to predict
      - markers: Marker genes dataframe from rank_genes_groups() (for interface consistency)
    Outputs: Dict with predictions, true_labels, confidence_scores, cell_ids
    Algorithm: Uses MARS's meta-learning approach to discover and annotate cell types
    Strengths: Discovers novel cell types, transfers knowledge across heterogeneous experiments
    """

    # Default return function for error handling
    def default_return():
        n_test_cells = len(adata_test.obs)
        return {
            'predictions': ['Unknown'] * n_test_cells,
            'true_labels': list(adata_test.obs['Ground_Truth_Celltype']),
            'confidence_scores': [0.0] * n_test_cells,
            'cell_ids': list(adata_test.obs.index),
            'method_walltime': 0.0,
            'peak_memory_mb': None,
            'peak_vram_mb': None,
        }

    # Validate input data
    if 'Ground_Truth_Celltype' not in adata_train.obs.columns:
        warnings.warn("Ground_Truth_Celltype not found in training data")
        return default_return()

    if 'Ground_Truth_Celltype' not in adata_test.obs.columns:
        warnings.warn("Ground_Truth_Celltype not found in test data")
        return default_return()

    # Check conda environment availability (direct directory check)
    conda_env_path = "<WORKSTATION_HOME>/miniconda3/envs/MARS_env"
    if not os.path.isdir(conda_env_path):
        warnings.warn(f"MARS_env conda environment not found at {conda_env_path}. Please create it first.")
        return default_return()

    # Create temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Preparing data for MARS...")

            # Prepare file paths (use npy + json for maximum compatibility)
            train_X_path = os.path.join(temp_dir, "train_X.npy")
            train_meta_path = os.path.join(temp_dir, "train_meta.json")
            test_X_path = os.path.join(temp_dir, "test_X.npy")
            test_meta_path = os.path.join(temp_dir, "test_meta.json")
            results_path = os.path.join(temp_dir, "results.pkl")

            # Get common genes between train and test
            common_genes = list(set(adata_train.var.index) & set(adata_test.var.index))
            if len(common_genes) < 1:
                warnings.warn("Too few common genes between train and test datasets")
                return default_return()

            print(f"Using {len(common_genes)} common genes for MARS analysis")

            # Subset both datasets to common genes and ensure consistent ordering
            adata_train_subset = adata_train[:, sorted(common_genes)].copy()
            adata_test_subset = adata_test[:, sorted(common_genes)].copy()

            # MARS requires cell_type annotation in obs for labeled experiments
            # Ensure it's named 'cell_type' for MARS compatibility
            if 'cell_type' not in adata_train_subset.obs.columns:
                adata_train_subset.obs['cell_type'] = adata_train_subset.obs['Ground_Truth_Celltype']

            # Save data using npy + json for maximum compatibility
            # Avoids h5ad, pickle, and numpy version issues
            import json

            # Get expression matrices as plain numpy arrays
            if hasattr(adata_train_subset.X, 'todense'):
                train_X = np.array(adata_train_subset.X.todense(), dtype=np.float32)
            else:
                train_X = np.array(adata_train_subset.X, dtype=np.float32)

            if hasattr(adata_test_subset.X, 'todense'):
                test_X = np.array(adata_test_subset.X.todense(), dtype=np.float32)
            else:
                test_X = np.array(adata_test_subset.X, dtype=np.float32)

            # Save train data - expression matrix as .npy, metadata as JSON
            np.save(train_X_path, train_X)
            with open(train_meta_path, 'w') as f:
                json.dump({
                    'cell_types': adata_train_subset.obs['cell_type'].astype(str).tolist(),
                    'cell_ids': adata_train_subset.obs.index.astype(str).tolist(),
                    'gene_ids': adata_train_subset.var.index.astype(str).tolist()
                }, f)

            # Save test data
            np.save(test_X_path, test_X)
            with open(test_meta_path, 'w') as f:
                json.dump({
                    'cell_ids': adata_test_subset.obs.index.astype(str).tolist(),
                    'gene_ids': adata_test_subset.var.index.astype(str).tolist()
                }, f)

            print("Running MARS in conda environment...")
            
            # Determine project directory (where MARS_package/ lives)
            project_dir = os.path.dirname(os.path.abspath(__file__))

            # Create Python script for MARS execution
            mars_script = f'''
import tracemalloc
tracemalloc.start()  # Start tracking memory (before imports)

import sys
import os
import time
import json
import pickle
import numpy as np
import argparse
import torch

# Add benchmarking repository root to Python path so MARS_package can be imported
sys.path.insert(0, '{project_dir}')

try:
    from MARS_package.mars import MARS
    from MARS_package.experiment_dataset import ExperimentDataset

    # Load data
    print("Loading training and test data...")
    train_X = np.load('{train_X_path}')
    with open('{train_meta_path}', 'r') as f:
        train_meta = json.load(f)
    train_cell_types = train_meta['cell_types']
    train_cell_ids = train_meta['cell_ids']
    train_gene_ids = train_meta['gene_ids']

    test_X = np.load('{test_X_path}')
    with open('{test_meta_path}', 'r') as f:
        test_meta = json.load(f)
    test_cell_ids = test_meta['cell_ids']
    test_gene_ids = test_meta['gene_ids']

    print(f"Training data: {{train_X.shape[0]}} cells x {{train_X.shape[1]}} genes")
    print(f"Test data: {{test_X.shape[0]}} cells x {{test_X.shape[1]}} genes")
    print(f"Training cell types: {{sorted(set(train_cell_types))}}")

    # Convert cell types to numeric labels
    cell_types = np.unique(train_cell_types)
    cell_type_to_idx = {{ct: idx for idx, ct in enumerate(cell_types)}}
    idx_to_cell_type = {{idx: ct for ct, idx in cell_type_to_idx.items()}}
    y_train = np.array([cell_type_to_idx[ct] for ct in train_cell_types])

    # Create ExperimentDataset objects
    # x: (n_cells, n_genes) numpy array; y: numeric labels (empty list = unlabeled)
    train_dataset = ExperimentDataset(
        x=train_X,
        cells=list(train_cell_ids),
        genes=list(train_gene_ids),
        metadata='train',
        y=y_train
    )
    test_dataset = ExperimentDataset(
        x=test_X,
        cells=list(test_cell_ids),
        genes=list(test_gene_ids),
        metadata='test',
        y=[]  # Test data is unlabeled for MARS
    )

    n_clusters = len(cell_types)
    print(f"Number of clusters: {{n_clusters}}")

    # MARS parameters
    params = argparse.Namespace()
    params.pretrain_batch = None
    params.device = torch.device('cpu')  # Force CPU (RTX 5060 sm_120 incompatible with PyTorch 1.2.0)
    params.epochs = 30
    params.epochs_pretrain = 25
    params.pretrain = True
    params.model_file = '/tmp/mars_model.pt'
    params.learning_rate = 0.001
    params.lr_scheduler_gamma = 0.5
    params.lr_scheduler_step = 20

    print("Initializing MARS model...")
    mars = MARS(
        n_clusters=n_clusters,
        params=params,
        labeled_data=[train_dataset],
        unlabeled_data=test_dataset,
        pretrain_data=train_dataset,  # single ExperimentDataset, not a list
    )

    print("Training MARS model...")
    _t0 = time.perf_counter()
    # evaluation_mode=False: no ground truth needed for test, returns (adata, landmarks)
    adata_result, landmarks = mars.train(evaluation_mode=False)
    method_walltime = time.perf_counter() - _t0
    print(f"MARS training completed in {{method_walltime:.1f}}s")

    # Map MARS cluster IDs to cell type names via centroid matching in embedding space.
    # name_cell_types() uses a Gaussian kernel that collapses to zero in 100D space
    # (curse of dimensionality), so we use nearest-centroid matching instead.
    print("Mapping clusters to cell types via centroid matching...")

    from scipy.spatial.distance import cdist as scipy_cdist

    # Embeddings are in adata_result.obsm['MARS_embedding'] (set by save_result)
    tiss_train = adata_result[adata_result.obs['experiment'] == 'train', :]
    tiss_test  = adata_result[adata_result.obs['experiment'] == 'test',  :]

    tr_embeddings  = tiss_train.obsm['MARS_embedding']      # (n_train, emb_dim)
    test_embeddings = tiss_test.obsm['MARS_embedding']       # (n_test,  emb_dim)
    tr_truth_labels = np.array(list(tiss_train.obs['truth_labels']), dtype=int)
    mars_labels_raw = list(tiss_test.obs['MARS_labels'])
    returned_cell_ids = list(tiss_test.obs_names)

    # Compute per-class training centroids
    unique_tr_labels = np.unique(tr_truth_labels)
    tr_centroid_matrix = np.array([
        tr_embeddings[tr_truth_labels == lbl].mean(axis=0) for lbl in unique_tr_labels
    ])  # (n_classes, emb_dim)

    # Compute per-cluster test centroids
    mars_labels_int = np.array([int(l) for l in mars_labels_raw])
    unique_test_clusters = np.unique(mars_labels_int)
    cluster_to_celltype = {{}}
    for cls in unique_test_clusters:
        centroid = test_embeddings[mars_labels_int == cls].mean(axis=0, keepdims=True)
        dists = scipy_cdist(centroid, tr_centroid_matrix, metric='euclidean')[0]
        nearest_tr_label = unique_tr_labels[np.argmin(dists)]
        cluster_to_celltype[cls] = idx_to_cell_type[int(nearest_tr_label)]

    print(f"Cluster → cell type mapping: {{cluster_to_celltype}}")

    # Build cell_id → prediction dict and re-order to match original test order
    # AnnData's concatenate appends batch categories to the index (e.g. "-test" or "-1").
    # We create a mapping that gracefully matches the original cell ID.
    pred_by_id = {{}}
    for cell_id, lbl in zip(returned_cell_ids, mars_labels_int):
        cell_id_str = str(cell_id)
        ct = cluster_to_celltype.get(lbl, 'Unknown')
        # Store both the exact ID and the stripped version
        pred_by_id[cell_id_str] = ct
        if cell_id_str.endswith('-test'):
            pred_by_id[cell_id_str.rsplit('-test', 1)[0]] = ct
        if cell_id_str.endswith('-1'):
            pred_by_id[cell_id_str.rsplit('-1', 1)[0]] = ct
        if '-' in cell_id_str:
            pred_by_id[cell_id_str.rsplit('-', 1)[0]] = ct
            
    predictions = [pred_by_id.get(str(cid), 'Unknown') for cid in test_cell_ids]

    # Capture peak memory
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_mem / (1024 * 1024)

    results = {{
        'predictions': predictions,
        'confidence': None,  # MARS does not produce per-cell confidence scores
        'true_labels': test_meta['cell_ids'],  # placeholder; overwritten by parent
        'cell_ids': test_cell_ids,
        'peak_memory_mb': peak_memory_mb,
        'method_walltime': method_walltime,
        'peak_vram_mb': 0.0,  # CPU-only run
    }}

    print(f"Peak memory: {{peak_memory_mb:.1f}} MB, walltime: {{method_walltime:.1f}}s")

    with open('{results_path}', 'wb') as f:
        pickle.dump(results, f)

    print(f"Predictions: {{len(predictions)}} cells, unique: {{set(predictions)}}")

except Exception as e:
    print(f"MARS execution failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

            # Execute MARS in conda environment
            result = subprocess.run(
                [
                    "<WORKSTATION_HOME>/miniconda3/condabin/conda",
                    "run",
                    "-n", "MARS_env",
                    "python",
                    "-c", mars_script
                ],
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour (1 day) timeout for meta-learning
            )

            if result.returncode != 0:
                warnings.warn(f"MARS execution failed: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return default_return()

            print("Parsing MARS results...")

            # Load results from pickle file
            if not os.path.exists(results_path):
                warnings.warn("MARS results file not found")
                return default_return()

            with open(results_path, 'rb') as f:
                results = pickle.load(f)

            cell_predictions = results['predictions']
            confidence_from_mars = results.get('confidence', None)
            true_labels = list(adata_test.obs['Ground_Truth_Celltype'])
            cell_ids = results['cell_ids']
            peak_memory_mb = results.get('peak_memory_mb', None)
            method_walltime = results.get('method_walltime', None)
            peak_vram_mb = results.get('peak_vram_mb', None)

            # Ensure predictions have correct length
            if len(cell_predictions) != len(adata_test.obs):
                warnings.warn(f"Prediction length mismatch. Expected: {len(adata_test.obs)}, Got: {len(cell_predictions)}")
                return default_return()

            # Convert predictions to strings and handle missing values
            cell_predictions = [str(pred) if pd.notna(pred) else "Unknown" for pred in cell_predictions]

            # Process confidence scores
            if confidence_from_mars is not None and len(confidence_from_mars) == len(cell_predictions):
                # Use MARS-provided confidence if available
                confidence_scores = []
                for conf in confidence_from_mars:
                    if pd.notna(conf) and isinstance(conf, (int, float)):
                        # Ensure confidence is between 0 and 1
                        conf_val = max(0.0, min(1.0, float(conf)))
                        confidence_scores.append(conf_val)
                    else:
                        confidence_scores.append(0.5)  # Default confidence
            else:
                # Use heuristic if MARS doesn't provide confidence
                confidence_scores = [0.8 if pred != "Unknown" else 0.0
                                   for pred in cell_predictions]

            # Adjust confidence for Unknown predictions
            for i, pred in enumerate(cell_predictions):
                if pred == "Unknown":
                    confidence_scores[i] = 0.0

            print(f"MARS completed successfully:")
            print(f"  - Total predictions: {len(cell_predictions)}")
            print(f"  - Unknown predictions: {sum(1 for p in cell_predictions if p == 'Unknown')}")
            print(f"  - Unique predictions: {set(cell_predictions)}")
            if confidence_scores:
                print(f"  - Confidence range: {min(confidence_scores):.3f} - {max(confidence_scores):.3f}")

            # Return standardized format
            return {
                'predictions': cell_predictions,
                'true_labels': true_labels,
                'confidence_scores': confidence_scores,
                'cell_ids': cell_ids,
                'peak_memory_mb': peak_memory_mb,
                'method_walltime': method_walltime,
                'peak_vram_mb': peak_vram_mb,
            }

        except subprocess.TimeoutExpired:
            warnings.warn("MARS execution timed out after 24 hours")
            return default_return()
        except Exception as e:
            warnings.warn(f"MARS error: {str(e)}")
            return default_return()


# For backward compatibility
run_MARS = run_MARS_function
