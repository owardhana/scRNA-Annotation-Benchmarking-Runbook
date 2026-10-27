# run_scSemiGAN.py
#################################################
# scSemiGAN Function for Python Benchmarking Framework
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


def run_scSemiGAN_function(adata_train: ad.AnnData, adata_test: ad.AnnData, markers: pd.DataFrame) -> Dict[str, Any]:
    """
    scSemiGAN Cell Type Annotation Function

    Purpose: Run scSemiGAN algorithm using a Wasserstein GAN with encoder for
             semi-supervised cell type classification.
    Inputs:
      - adata_train: Training AnnData object (labeled reference)
      - adata_test: Test AnnData object to predict (unlabeled query)
      - markers: Marker genes dataframe (for interface consistency)
    Outputs: Dict with predictions, true_labels, confidence_scores, cell_ids
    Algorithm: Uses WGAN-GP with encoder network. The GAN learns the gene expression
               distribution while the encoder's softmax output classifies cells.
               Supervised loss on labeled cells guides the encoder.
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
    conda_env_path = "<WORKSTATION_HOME>/miniconda3/envs/scSemiGAN_env"
    if not os.path.isdir(conda_env_path):
        warnings.warn(f"scSemiGAN_env conda environment not found at {conda_env_path}. Please create it first.")
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

            # Subset both datasets to common genes
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

            # Determine project directory
            project_dir = os.path.dirname(os.path.abspath(__file__))

            # Create Python script for scSemiGAN execution
            semigan_script = f'''
import sys
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '{project_dir}')

try:
    import numpy as np
    import pandas as pd
    import time
    from sklearn.preprocessing import minmax_scale
    from memory_profiler import memory_usage

    # GPU/CPU: use GPU if available, else CPU (inherits available devices)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import tensorflow as tf

    from scSemiGAN_package.clus_wgan import Discriminator, Generator, Encoder
    from scSemiGAN_package.util import sample_Z

    # ---- Load data (numpy/CSV — no anndata needed) ----
    X_train_raw = np.load('{train_X_path}')
    X_test_raw = np.load('{test_X_path}')
    train_obs = pd.read_csv('{train_obs_path}', index_col=0)
    test_obs = pd.read_csv('{test_obs_path}', index_col=0)

    # ---- Method execution (measured) ----
    _timing = [0.0]

    def _run_method():
        if tf.test.is_gpu_available():
            print("[scSemiGAN] Running with GPU acceleration.", flush=True)
        else:
            print("[scSemiGAN] Falling back to CPU. No GPU detected.", flush=True)

        _t0 = time.perf_counter()

        # --- Data Preparation ---
        # Matrices already loaded as numpy arrays
        X_train = np.abs(X_train_raw.astype(np.float32))
        X_test = np.abs(X_test_raw.astype(np.float32))

        cellname_train = np.array(train_obs['Ground_Truth_Celltype'].astype(str))
        cellname_test = np.array(test_obs['Ground_Truth_Celltype'].astype(str))
        n_train = len(cellname_train)
        n_test = len(cellname_test)

        # Create integer labels for train data
        cell_types = sorted(set(cellname_train))
        type_to_int = {{ct: i for i, ct in enumerate(cell_types)}}
        num_classes = len(cell_types)
        y_train = np.array([type_to_int[ct] for ct in cellname_train], dtype=np.int32)

        print(f"Classes: {{num_classes}}, Train: {{n_train}}, Test: {{n_test}}")

        # Preprocessing: combine for consistent gene filtering/selection
        X_all = np.vstack([X_train, X_test])

        # Filter genes expressed in < 10 cells
        indicator = np.where(X_all > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0)
        gene_mask = sum_gene >= 10
        X_all = X_all[:, gene_mask]

        # Select top 2000 most variable genes
        n_hvg = min(2000, X_all.shape[1])
        var_gene = np.var(X_all, axis=0)
        hvg_indices = np.argsort(var_gene)[-n_hvg:]
        X_all = X_all[:, hvg_indices]
        x_dim = X_all.shape[1]

        # Min-max scale per cell to [0, 1]
        X_all = minmax_scale(X_all, feature_range=(0, 1), axis=1, copy=False)

        X_train_proc = X_all[:n_train]
        X_test_proc = X_all[n_train:]

        print(f"After preprocessing: {{x_dim}} genes")

        # --- Model Setup ---
        dim_gen = 10
        z_dim = dim_gen + num_classes
        batch_size = 64
        beta_cycle_gen = 1.0
        beta_cycle_label = 1.0
        num_batches = 4000

        tf.reset_default_graph()
        tf.set_random_seed(8888)
        np.random.seed(8888)

        d_net = Discriminator(x_dim=x_dim)
        g_net = Generator(z_dim=z_dim, x_dim=x_dim)
        enc_net = Encoder(z_dim=z_dim, dim_gen=dim_gen, x_dim=x_dim)

        # Build graph (adapted from scSemiGAN class)
        x_ph = tf.placeholder(tf.float32, [None, x_dim], name='x')
        z_ph = tf.placeholder(tf.float32, [None, z_dim], name='z')
        x_sup_ph = tf.placeholder(tf.float32, [None, x_dim], name='x_sup')
        x_sup_label_ph = tf.placeholder(tf.int32, [None, num_classes], name='x_sup_label')

        # Generator
        x_gen = g_net(z_ph, reuse=False)
        z_enc_gen, z_enc_label, z_enc_logits = enc_net(x_gen, reuse=False)
        z_infer_gen, z_infer_label, z_infer_logits = enc_net(x_ph)

        # Discriminator
        d_real = d_net(x_ph, reuse=False)
        d_fake = d_net(x_gen)

        # Supervised encoder
        z_infer_gen_sup, z_infer_label_sup, z_infer_logits_sup = enc_net(x_sup_ph)

        z_gen_part = z_ph[:, 0:dim_gen]
        z_hot_part = z_ph[:, dim_gen:]

        # Losses
        g_loss = tf.reduce_mean(d_fake) + \\
                 beta_cycle_gen * tf.reduce_mean(tf.square(z_gen_part - z_enc_gen)) + \\
                 beta_cycle_label * tf.reduce_mean(
                     tf.nn.softmax_cross_entropy_with_logits(logits=z_enc_logits, labels=z_hot_part))
        g_loss_sup = beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=z_infer_logits_sup, labels=tf.cast(x_sup_label_ph, tf.float32)))
        g_loss_total = g_loss + g_loss_sup

        # WGAN-GP
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * x_ph + (1 - epsilon) * x_gen
        d_hat = d_net(x_hat)
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0))
        d_loss = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake) + ddx

        d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            d_loss, var_list=d_net.vars)
        g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            g_loss_total, var_list=[g_net.vars, enc_net.vars])

        # Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        # Prepare supervised data (ALL train data is labeled)
        sup_x = X_train_proc
        sup_label = y_train
        sup_label_onehot = np.eye(num_classes, dtype=np.int32)[sup_label]

        # Combine train + test for discriminator (semi-supervised: D sees all data)
        X_all_proc = np.vstack([X_train_proc, X_test_proc])
        n_all = X_all_proc.shape[0]

        # --- Training Loop ---
        print("Training scSemiGAN (semi-supervised)...")
        for t in range(num_batches):
            # Discriminator steps - samples from ALL data (train + test)
            for _ in range(5):
                idx = np.random.randint(0, n_all, batch_size)
                bx = X_all_proc[idx]
                bz = sample_Z(batch_size, z_dim, num_classes)
                sess.run(d_adam, feed_dict={{x_ph: bx, z_ph: bz}})

            # Generator + Encoder step - supervised loss on labeled train data only
            bz = sample_Z(batch_size, z_dim, num_classes)
            sess.run(g_adam, feed_dict={{
                z_ph: bz,
                x_sup_ph: sup_x,
                x_sup_label_ph: sup_label_onehot
            }})

            if t % 500 == 0:
                print(f"  Step {{t}}/{{num_batches}}")

        print("Training complete.")

        # --- Prediction on test data ---
        test_batch_size = batch_size
        n_test_pts = X_test_proc.shape[0]
        all_labels_prob = np.zeros((n_test_pts, num_classes))

        for b in range(int(np.ceil(n_test_pts / test_batch_size))):
            start = b * test_batch_size
            end = min((b + 1) * test_batch_size, n_test_pts)
            x_batch = X_test_proc[start:end]
            z_gen_out, z_label_out = sess.run(
                [z_infer_gen, z_infer_label],
                feed_dict={{x_ph: x_batch}})
            all_labels_prob[start:end] = z_label_out

        sess.close()

        # Convert softmax outputs to predictions
        pred_indices = np.argmax(all_labels_prob, axis=1)
        confidence = np.max(all_labels_prob, axis=1)

        # Map integer predictions to cell type names
        int_to_type = {{i: ct for ct, i in type_to_int.items()}}
        predictions = [int_to_type.get(int(p), "Unknown") for p in pred_indices]

        true_labels = list(cellname_test)
        cell_ids = list(test_obs.index)
        confidence_scores = confidence.tolist()

        _timing[0] = time.perf_counter() - _t0
        return predictions, confidence_scores, true_labels, cell_ids

    mem_list, (predictions, confidence_scores, true_labels, cell_ids) = memory_usage(
        (_run_method, [], {{}}), interval=0.1, include_children=True, retval=True)
    peak_system_memory_mb = max(mem_list)
    method_walltime = _timing[0]

    results = {{
        'predictions': list(predictions),
        'confidence_scores': confidence_scores,
        'true_labels': true_labels,
        'cell_ids': cell_ids,
        'peak_system_memory_mb': peak_system_memory_mb,
        'peak_vram_mb': 0.0,
        'method_walltime': method_walltime,
    }}

    print(f"Peak system memory: {{peak_system_memory_mb:.2f}} MB, walltime: {{method_walltime:.2f}}s")

    with open('{results_path}', 'wb') as f:
        pickle.dump(results, f)

    print(f"Predictions: {{len(predictions)}} cells, unique: {{set(predictions)}}")

except Exception as e:
    print(f"scSemiGAN execution failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

            # Execute scSemiGAN in conda environment
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'

            result = subprocess.run(
                [
                    "<WORKSTATION_HOME>/miniconda3/condabin/conda",
                    "run",
                    "-n", "scSemiGAN_env",
                    "python",
                    "-c", semigan_script
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour timeout
            )

            if result.returncode != 0:
                warnings.warn(f"scSemiGAN execution failed: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return default_return()

            # Print subprocess output
            print(result.stdout)

            # Load results from pickle file
            if not os.path.exists(results_path):
                warnings.warn("scSemiGAN results file not found")
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

            cell_predictions = [str(pred) if pd.notna(pred) else "Unknown" for pred in cell_predictions]

            confidence_scores = [
                0.0 if pred == "Unknown" else (max(0.0, min(1.0, float(conf))) if pd.notna(conf) and isinstance(conf, (int, float)) else 0.5)
                for pred, conf in zip(cell_predictions, confidence_scores)
            ]

            print(f"scSemiGAN completed successfully:")
            print(f"  - Total predictions: {len(cell_predictions)}")
            print(f"  - Unknown predictions: {sum(1 for p in cell_predictions if p == 'Unknown')}")
            print(f"  - Unique predictions: {len(set(cell_predictions))}")

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
            warnings.warn("scSemiGAN execution timed out after 24 hours")
            return default_return()
        except Exception as e:
            warnings.warn(f"scSemiGAN error: {str(e)}")
            return default_return()


# For backward compatibility
run_scSemiGAN = run_scSemiGAN_function
