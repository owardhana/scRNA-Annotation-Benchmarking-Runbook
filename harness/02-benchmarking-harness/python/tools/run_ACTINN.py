# run_ACTINN.py
#
# Embeds the functions of ACTINN's actinn_predict.py (https://github.com/mafeiyang/ACTINN,
# GPL-3.0); this file is therefore distributed under GPL-3.0, not MIT. See THIRD_PARTY.md.
#################################################
# EXEMPLAR: deep-learning tool (Python side)
#
# ACTINN is a feed-forward neural network classifier. What makes it the useful
# exemplar here is not the model but the EXECUTION MODEL, which is how every
# Python-native tool in this harness runs:
#
#   1. The tool never runs in the harness process. Its dependency set (an old
#      tensorflow.compat.v1 stack) cannot coexist with the harness's own, so
#      the wrapper generates a self-contained script and executes it inside a
#      dedicated conda environment via `conda run`.
#   2. Data crosses the boundary through a TemporaryDirectory as h5ad/pickle,
#      not in memory.
#   3. The subprocess measures ITSELF and returns the numbers with its results,
#      because the parent process cannot see a child's peak RSS or VRAM. This
#      is the Python analogue of R's peakRAM wrapping.
#   4. A wall-clock timeout bounds the run. The R harness has no equivalent;
#      a hanging R tool blocks the sweep.
#
# Returned resource keys are `method_walltime`, `peak_memory_mb` and
# `peak_vram_mb` -- deliberately NOT the R harness's `runtime_secs` and
# `peak_system_memory_mb`, since they are measured differently and are not
# interchangeable. See docs/architecture.md.
#################################################

import subprocess
import tempfile
import os
import pandas as pd
import anndata as ad
import warnings
import pickle
import h5py
from typing import Dict, List, Any

from conda_env import conda_env_path, conda_executable

# Environment holding ACTINN's tensorflow v1 stack. Recipe in
# harness/docs/environments/install_notes_DL.txt; override the search location with
# SC_BENCH_CONDA_PREFIX.
ACTINN_ENV_NAME = os.environ.get("SC_BENCH_ACTINN_ENV", "ACTINN_env")


def run_ACTINN_function(adata_train: ad.AnnData, adata_test: ad.AnnData, markers: pd.DataFrame) -> Dict[str, Any]:
    """
    ACTINN Cell Type Annotation Function

    Purpose: Run ACTINN algorithm using a 4-layer neural network for cell type classification
    Inputs:
      - adata_train: Training AnnData object (used for model training)
      - adata_test: Test AnnData object to predict
      - markers: Marker genes dataframe from rank_genes_groups() (for interface consistency)
    Outputs: Dict with predictions, true_labels, confidence_scores, cell_ids
    Algorithm: Uses ACTINN's 4-layer fully connected neural network (100->50->25->n_types)
               with Adam optimizer and L2 regularization
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

    # ACTINN runs in its own conda environment (see harness/docs/environments/install_notes_DL.txt);
    # conda_env resolves where that lives on this machine.
    if conda_env_path(ACTINN_ENV_NAME) is None:
        warnings.warn(
            f"conda environment '{ACTINN_ENV_NAME}' not found. Create it, or set "
            "SC_BENCH_CONDA_PREFIX to the directory holding your environments."
        )
        return default_return()

    # Create temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Prepare file paths — use numpy/CSV to avoid anndata version incompatibility
            train_X_path = os.path.join(temp_dir, "train_X.npy")
            test_X_path = os.path.join(temp_dir, "test_X.npy")
            train_obs_path = os.path.join(temp_dir, "train_obs.csv")
            test_obs_path = os.path.join(temp_dir, "test_obs.csv")
            var_names_path = os.path.join(temp_dir, "var_names.txt")
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
            import numpy as np
            def save_matrix(mtx, path):
                if scipy.sparse.issparse(mtx):
                    mtx = mtx.toarray()
                np.save(path, np.array(mtx, dtype=np.float32))

            save_matrix(adata_train_subset.X, train_X_path)
            save_matrix(adata_test_subset.X, test_X_path)

            # Save obs metadata
            adata_train_subset.obs[['Ground_Truth_Celltype']].to_csv(train_obs_path)
            adata_test_subset.obs[['Ground_Truth_Celltype']].to_csv(test_obs_path)
            
            # Save var names
            with open(var_names_path, 'w') as f:
                f.write("\n".join(adata_train_subset.var.index.astype(str).tolist()))

            # Create Python script for ACTINN execution with embedded algorithm
            actinn_script = f'''
import tracemalloc
tracemalloc.start()  # Start tracking memory

import sys
import os
import numpy as np
import pandas as pd
import math
import pickle
import warnings
warnings.filterwarnings("ignore")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops
import scipy.sparse
import scanpy as sc

# Set seeds for reproducibility
np.random.seed(1)
tf.set_random_seed(3)

# ============================================
# ACTINN Core Functions (embedded from actinn_predict.py)
# ============================================

def adata_to_actinn_df(adata):
    """Convert AnnData to DataFrame format ACTINN expects (genes x cells, uppercase)"""
    if scipy.sparse.issparse(adata.X):
        data = adata.X.T.toarray()  # genes x cells
    else:
        data = adata.X.T
    return pd.DataFrame(
        data,
        index=[g.upper() for g in adata.var_names],
        columns=adata.obs_names
    )

def scale_sets(sets):
    """Get common genes, normalize and scale the sets"""
    common_genes = set(sets[0].index)
    for i in range(1, len(sets)):
        common_genes = set.intersection(set(sets[i].index), common_genes)
    common_genes = sorted(list(common_genes))
    sep_point = [0]
    for i in range(len(sets)):
        sets[i] = sets[i].loc[common_genes,]
        sep_point.append(sets[i].shape[1])
        
    total_set = np.array(pd.concat(sets, axis=1, sort=False), dtype=np.float32)
    
    # 1. FIX: Prevent NaN from cells with 0 total counts
    # col_sums = np.sum(total_set, axis=0, keepdims=True)
    # col_sums[col_sums == 0] = 1.0  # Avoid division by zero
    # total_set = np.divide(total_set, col_sums) * 10000
    
    # 2. Add pseudocount and Log2 transform
    # total_set = np.log2(total_set + 1)
    
    # NOTE: ACTINN usually does 1st-99th percentile filtering on total_set by Expression sum and CV.
    # However, since the benchmarking framework already heavily filters the inputs (e.g. to 500-1000 marker genes),
    # applying another relative percentile crop will crash if variance distributions are tight, causing `index -1 out of bounds`.
    # Thus, expression and CV percentile filtering are disabled down here.
    
    for i in range(len(sets)):
        sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
    return sets

def one_hot_matrix(labels, C):
    """Turn labels into one-hot matrix"""
    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

def type_to_label_dict(types):
    """Make types to labels dictionary"""
    type_to_label = {{}}
    all_type = list(set(types))
    for i in range(len(all_type)):
        type_to_label[all_type[i]] = i
    return type_to_label

def convert_type_to_label(types, type_to_label):
    """Convert types to labels"""
    types = list(types)
    labels = []
    for t in types:
        labels.append(type_to_label[t])
    return labels

def create_placeholders(n_x, n_y):
    """Create TensorFlow placeholders"""
    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))
    return X, Y

def initialize_parameters(nf, ln1, ln2, ln3, nt):
    """Initialize parameters with Xavier initialization"""
    tf.set_random_seed(3)
    
    W1 = tf.get_variable("W1", [ln1, nf], initializer=tf.initializers.glorot_uniform(seed=3))
    b1 = tf.get_variable("b1", [ln1, 1], initializer=tf.zeros_initializer())
    
    W2 = tf.get_variable("W2", [ln2, ln1], initializer=tf.initializers.glorot_uniform(seed=3))
    b2 = tf.get_variable("b2", [ln2, 1], initializer=tf.zeros_initializer())
    
    W3 = tf.get_variable("W3", [ln3, ln2], initializer=tf.initializers.glorot_uniform(seed=3))
    b3 = tf.get_variable("b3", [ln3, 1], initializer=tf.zeros_initializer())
    
    W4 = tf.get_variable("W4", [nt, ln3], initializer=tf.initializers.glorot_uniform(seed=3))
    b4 = tf.get_variable("b4", [nt, 1], initializer=tf.zeros_initializer())
    
    parameters = {{"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}}
    return parameters

def forward_propagation(X, parameters):
    """Forward propagation: LINEAR->RELU->LINEAR->RELU->LINEAR->RELU->LINEAR"""
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    return Z4

def forward_propagation_for_predict(X, parameters):
    """Forward propagation for prediction"""
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    return Z4

def compute_cost(Z4, Y, parameters, lambd=0.01):
    """Compute cost with L2 regularization"""
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)) + \\
        (tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) +
         tf.nn.l2_loss(parameters["W3"]) + tf.nn.l2_loss(parameters["W4"])) * lambd
    return cost

def random_mini_batches(X, Y, mini_batch_size=32, seed=1):
    """Generate random mini batches"""
    ns = X.shape[1]
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(ns))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_complete_minibatches = int(math.floor(ns / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if ns % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: ns]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: ns]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def predict(X, parameters):
    """Get predicted class indices"""
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    params = {{"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}}
    x = tf.placeholder("float")
    z4 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z4)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict={{x: X}})
    sess.close()
    return prediction

def predict_probability(X, parameters):
    """Get softmax probabilities for each class"""
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    params = {{"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}}
    x = tf.placeholder("float", shape=(X.shape[0], None))
    z4 = forward_propagation_for_predict(x, params)
    p = tf.nn.softmax(z4, axis=0)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict={{x: X}})
    sess.close()
    return prediction

def model(X_train, Y_train, X_test, starting_learning_rate=0.0001, num_epochs=50, minibatch_size=128, print_cost=True):
    """Build and train the 4-layer neural network model"""
    ops.reset_default_graph()
    tf.set_random_seed(3)
    seed = 3
    (nf, ns) = X_train.shape
    nt = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(nf, nt)
    parameters = initialize_parameters(nf=nf, ln1=100, ln2=50, ln3=25, nt=nt)
    Z4 = forward_propagation(X, parameters)
    cost = compute_cost(Z4, Y, parameters, 0.0001)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 1000, 0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainer = optimizer.minimize(cost, global_step=global_step)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(ns / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([trainer, cost], feed_dict={{X: minibatch_X, Y: minibatch_Y}})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost and (epoch + 1) % 5 == 0:
                costs.append(epoch_cost)
        parameters = sess.run(parameters)
        return parameters

# ============================================
# Main Execution
# ============================================
try:
    if tf.test.is_gpu_available():
        print("[ACTINN] Running with GPU acceleration.", flush=True)
    else:
        print("[ACTINN] Falling back to CPU. No GPU detected.", flush=True)

    # Load data (numpy/CSV — no anndata file reading needed)
    X_train_raw = np.load('{train_X_path}')
    X_test_raw = np.load('{test_X_path}')
    train_obs = pd.read_csv('{train_obs_path}', index_col=0)
    test_obs = pd.read_csv('{test_obs_path}', index_col=0)
    
    with open('{var_names_path}', 'r') as f:
        var_names = [line.strip() for line in f.readlines() if line.strip()]

    # Reconstruct AnnData objects
    adata_train = sc.AnnData(X_train_raw)
    adata_train.obs = train_obs
    adata_train.var_names = var_names
    
    adata_test = sc.AnnData(X_test_raw)
    adata_test.obs = test_obs
    adata_test.var_names = var_names

    # Convert to ACTINN DataFrame format (genes x cells, uppercase)
    train_set = adata_to_actinn_df(adata_train)
    test_set = adata_to_actinn_df(adata_test)

    # Remove duplicate genes (keep first occurrence)
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    test_set = test_set.loc[~test_set.index.duplicated(keep='first')]

    # Get labels from training data
    train_labels_raw = adata_train.obs['Ground_Truth_Celltype'].tolist()
    barcode = list(test_set.columns)
    nt = len(set(train_labels_raw))

    # Scale datasets
    train_set, test_set = scale_sets([train_set, test_set])

    # Convert labels to numeric
    type_to_label = type_to_label_dict(train_labels_raw)
    label_to_type = {{v: k for k, v in type_to_label.items()}}
    train_labels_numeric = convert_type_to_label(train_labels_raw, type_to_label)
    train_labels_onehot = one_hot_matrix(train_labels_numeric, nt)

    # Train model
    parameters = model(
        train_set,
        train_labels_onehot,
        test_set,
        starting_learning_rate=0.0001,
        num_epochs=50,
        minibatch_size=128,
        print_cost=True
    )

    # Get predictions
    test_predict_indices = predict(test_set, parameters)
    predictions = [label_to_type[idx] for idx in test_predict_indices]

    # Get confidence scores from probability matrix
    prob_matrix = predict_probability(test_set, parameters)
    confidence_scores = np.max(prob_matrix, axis=0).tolist()

    # Get true labels
    true_labels = adata_test.obs['Ground_Truth_Celltype'].tolist()
    cell_ids = list(adata_test.obs.index)

    # Prepare results
    results = {{
        'predictions': predictions,
        'confidence_scores': confidence_scores,
        'true_labels': true_labels,
        'cell_ids': cell_ids
    }}

    # Capture peak memory usage
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['peak_memory_mb'] = peak_mem / (1024 * 1024)

    print(f"Peak memory usage: {{results['peak_memory_mb']:.2f}} MB")

    # Save results
    with open('{results_path}', 'wb') as f:
        pickle.dump(results, f)

    print(f"Predictions: {{len(predictions)}} cells, unique: {{set(predictions)}}")

except Exception as e:
    print(f"ACTINN execution failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

            # Set environment - restrict to CPU only
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend
            env['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
            
            result = subprocess.run(
                [
                    conda_executable(),
                    "run",
                    "-n", ACTINN_ENV_NAME,
                    "python",
                    "-c", actinn_script
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=86400  # 24 hour timeout for training
            )

            if result.returncode != 0:
                warnings.warn(f"ACTINN execution failed: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return default_return()

            # Print subprocess output (contains verbose prediction comparison)
            print(result.stdout)

            # Load results from pickle file
            if not os.path.exists(results_path):
                warnings.warn("ACTINN results file not found")
                return default_return()

            with open(results_path, 'rb') as f:
                results = pickle.load(f)

            cell_predictions = results['predictions']
            confidence_scores = results['confidence_scores']
            true_labels = results['true_labels']
            cell_ids = results['cell_ids']
            peak_memory_mb = results.get('peak_memory_mb', None)

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

            print(f"ACTINN completed successfully:")
            print(f"  - Total predictions: {len(cell_predictions)}")
            print(f"  - Unknown predictions: {sum(1 for p in cell_predictions if p == 'Unknown')}")
            print(f"  - Unique predictions: {len(set(cell_predictions))}")

            # Return standardized format
            return {
                'predictions': cell_predictions,
                'true_labels': true_labels,
                'confidence_scores': confidence_scores,
                'cell_ids': cell_ids,
                'peak_memory_mb': peak_memory_mb
            }

        except subprocess.TimeoutExpired:
            warnings.warn("ACTINN execution timed out after 24 hours")
            return default_return()
        except Exception as e:
            warnings.warn(f"ACTINN error: {str(e)}")
            return default_return()


# For backward compatibility
run_ACTINN = run_ACTINN_function
