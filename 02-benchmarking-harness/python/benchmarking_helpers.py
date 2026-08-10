"""benchmarking_helpers.py

Helper functions for the Python side of the cell-type annotation benchmark:
metrics, fold construction, per-tool execution, and result aggregation.

This is a port of R/benchmarking_helpers.R and deliberately mirrors its
function names and return shapes, so the two harnesses can be read side by
side. It is NOT identical, and the differences matter when comparing numbers
across languages -- they are catalogued in docs/architecture.md under
"R and Python are not interchangeable". The three that bite hardest:

  * Return keys differ. Python tools report `method_walltime`, `peak_memory_mb`
    and `peak_vram_mb`; R tools report `runtime_secs` and
    `peak_system_memory_mb`. GPU memory has no R counterpart.
  * Python normalises defensively. get_fold_data_cached() sniffs whether the
    matrix looks log-normalised and applies normalize_total + log1p if not.
    The R harness assumes the shared preprocessing already did this.
  * Marker derivation differs by mode. The CV path uses a t-test, the
    single-split path a Wilcoxon test; R uses Seurat's FindAllMarkers
    throughout. There is also no prepare_markers() equivalent here.

Converted from benchmarking_helpers.ipynb. Notebook outputs are not carried
over.
"""

# Import required libraries
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
import scipy.sparse as sp
from collections import defaultdict


def calculate_metrics(predicted: List[str], true_labels: List[str]) -> Dict[str, Any]:
    """
    Calculate Comprehensive Performance Metrics
    
    Purpose: Computes accuracy, precision, recall, F1, macro F1, Cohen's Kappa,
             MCC, unassigned rate, and rare type F1
    Inputs: 
      - predicted: List of predicted cell types
      - true_labels: List of ground truth cell types
    Outputs: Dict with overall_accuracy, macro_f1, cohens_kappa, mcc, unassigned_rate,
             rare_type_f1, rare_type_name, per_class_metrics dataframe
    Key Logic: Handles per-class TP/FP/FN calculations, excludes "Unknown" predictions
    """
    
    # Unassigned rate (compute before numpy conversion)
    unassigned_rate = np.sum(np.array(predicted) == "Unknown") / len(predicted)
    
    # Convert to numpy arrays
    predicted = np.array(predicted)
    true_labels = np.array(true_labels)
    
    # Overall Accuracy
    overall_accuracy = accuracy_score(true_labels, predicted)
    
    # Get unique cell types (exclude Unknown and NaN)
    cell_types = np.unique(np.concatenate([predicted, true_labels]))
    cell_types = cell_types[~pd.isnull(cell_types)]
    cell_types = cell_types[cell_types != "Unknown"]
    
    # Initialize metric arrays
    precision_scores = np.zeros(len(cell_types))
    recall_scores = np.zeros(len(cell_types))
    f1_scores = np.zeros(len(cell_types))
    
    # Calculate per-class metrics
    for i, ct in enumerate(cell_types):
        # TP, FP, FN calculations
        TP = np.sum((predicted == ct) & (true_labels == ct))
        FP = np.sum((predicted == ct) & (true_labels != ct))
        FN = np.sum((predicted != ct) & (true_labels == ct))
        
        # Precision and Recall
        precision_scores[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_scores[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # F1 Score
        if precision_scores[i] + recall_scores[i] > 0:
            f1_scores[i] = 2 * (precision_scores[i] * recall_scores[i]) / (precision_scores[i] + recall_scores[i])
        else:
            f1_scores[i] = 0
    
    # Macro F1 Score (unweighted average)
    macro_f1 = np.mean(f1_scores)
    
    # Cohen's Kappa
    # p_o = observed agreement (overall accuracy, includes Unknown-predicted cells)
    # p_e = expected agreement by chance, summed over filtered cell_types only
    # Unknown predictions reduce p_o and dilute marginal rates in p_e implicitly
    n = len(predicted)
    p_o = overall_accuracy
    p_e = sum(
        (np.sum(predicted == ct) / n) * (np.sum(true_labels == ct) / n)
        for ct in cell_types
    )
    cohens_kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0
    
    # Matthews Correlation Coefficient (multiclass, Gorodkin 2004)
    ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    n_ct = len(cell_types)
    C = np.zeros((n_ct, n_ct))
    for true_ct, pred_ct in zip(true_labels, predicted):
        if true_ct in ct_to_idx and pred_ct in ct_to_idx:
            C[ct_to_idx[true_ct], ct_to_idx[pred_ct]] += 1
    c = np.trace(C)
    s = C.sum()
    t_k = C.sum(axis=1)
    p_k = C.sum(axis=0)
    numerator = c * s - np.dot(t_k, p_k)
    denominator = np.sqrt((s**2 - np.dot(p_k, p_k)) * (s**2 - np.dot(t_k, t_k)))
    mcc = numerator / denominator if denominator > 0 else 0.0
    
    # Rare type F1 (F1 of the least-frequent true cell type)
    rare_idx = np.argmin([np.sum(true_labels == ct) for ct in cell_types])
    rare_type_f1 = f1_scores[rare_idx]
    rare_type_name = cell_types[rare_idx]
    
    # Create per-class metrics dataframe
    per_class_metrics = pd.DataFrame({
        'cell_type': cell_types,
        'precision': precision_scores,
        'recall': recall_scores,
        'f1': f1_scores
    })
    
    return {
        'overall_accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'cohens_kappa': cohens_kappa,
        'mcc': mcc,
        'unassigned_rate': unassigned_rate,
        'rare_type_f1': rare_type_f1,
        'rare_type_name': rare_type_name,
        'per_class_metrics': per_class_metrics
    }


def create_cv_folds(adata: ad.AnnData, k: int = 5, group_by: Optional[str] = None, 
                   stratify_by: str = "Ground_Truth_Celltype", seed: int = 123) -> List[List[str]]:
    """
    Create K-Fold Cross-Validation Splits
    
    Purpose: Creates k folds either grouped by donor or stratified by cell type
    Inputs:
      - adata: AnnData object with metadata
      - k: Number of folds (default 5)
      - group_by: Column for grouped CV (e.g. "donor_id"), None for stratified
      - stratify_by: Column for stratified CV (e.g. "Ground_Truth_Celltype")
      - seed: Random seed for reproducibility
    Outputs: List of k lists, each containing cell IDs for test set of that fold
    Key Logic: Grouped mode keeps donor samples together, stratified mode balances cell types
    """
    
    np.random.seed(seed)
    
    fold_cell_lists = []
    
    if group_by is not None and group_by in adata.obs.columns:
        # --- GROUPED MODE ---
        print(f"Mode: Grouped K-Fold CV by '{group_by}'.")
        
        all_groups = adata.obs[group_by].unique()
        if len(all_groups) < k:
            raise ValueError("Number of groups is less than k.")
        
        # Shuffle groups and assign to folds
        shuffled_groups = np.random.permutation(all_groups)
        group_folds = np.array_split(shuffled_groups, k)
        
        for i in range(k):
            test_groups = group_folds[i]
            # Get cell barcodes belonging to the test groups for this fold
            test_mask = adata.obs[group_by].isin(test_groups)
            fold_cell_lists.append(adata.obs.index[test_mask].tolist())
    
    else:
        # --- STRATIFIED MODE ---
        print(f"Mode: Stratified K-Fold CV by '{stratify_by}'.")
        
        if stratify_by not in adata.obs.columns:
            raise ValueError("Stratification variable not found in metadata.")
        
        # Use scikit-learn StratifiedKFold
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        # Get stratification labels, filtering NaN
        stratify_labels = adata.obs[stratify_by]
        valid_mask = stratify_labels.notna()
        n_dropped = int((~valid_mask).sum())
        if n_dropped > 0:
            print(f"Warning: Dropping {n_dropped} cells with NaN '{stratify_by}'")
        valid_indices = np.where(valid_mask)[0]
        valid_labels = stratify_labels.iloc[valid_indices]
        
        # Create folds
        for train_idx, test_idx in skf.split(valid_indices, valid_labels):
            fold_cell_lists.append(adata.obs.index[valid_indices[test_idx]].tolist())
    
    print("Fold creation complete.")
    
    # Print fold statistics
    for i, fold_cells in enumerate(fold_cell_lists):
        print(f"Fold {i+1}: {len(fold_cells)} cells")
    
    return fold_cell_lists


# Cache system for fold data.
#
# NOTE: the key is the fold index alone, with no dataset identity in it, and
# nothing clears the cache between datasets. A batch loop over several datasets
# must reset it (`fold_cache.clear()`) each iteration or the second dataset
# silently reuses the first dataset's splits and markers. The R harness has the
# same property; see the batch example in R/run_benchmark.R.
fold_cache = {}


# The tool registry, mirroring R/tool_registry.R. It lives here, in the harness
# core, because the execution modes below look tools up in it. benchmarking_main
# populates it IN PLACE (TOOL_REGISTRY[name] = fn) rather than rebinding it, so
# that the entries are visible from this module.
TOOL_REGISTRY: Dict[str, Any] = {}


def get_fold_data_cached(adata: sc.AnnData, folds: List[List[str]], fold_index: int, label_key: str = "Ground_Truth_Celltype") -> Dict[str, Any]:
    """
    Get Fold Data with Caching (Scanpy version)
    
    Purpose:
        Splits data into train/test for a fold and finds marker genes with caching.
    
    Inputs:
        - adata: Full AnnData object
        - folds: List of folds (each fold is a list of cell names)
        - fold_index: Which fold to process (0-based index)
        - label_key: Column in adata.obs indicating true cell type labels
    
    Outputs:
        Dict with keys: 'train', 'test', 'markers', 'fold_info'
    """
    cache_key = f"fold_{fold_index}"
    
    if cache_key not in fold_cache:
        print(f"    Computing fold {fold_index} data and markers...")
        
        # Split data
        test_cells = folds[fold_index]
        train_cells = [c for c in adata.obs_names if c not in test_cells]
        
        adata_train = adata[train_cells].copy()
        adata_test = adata[test_cells].copy()
        
        print(f"    Training: {adata_train.n_obs} cells, Testing: {adata_test.n_obs} cells")
        
        # Check if data is already log-normalized
        # Multiple indicators to determine normalization status:
        def is_log_normalized(adata_obj):
            """
            Check if AnnData is already log-normalized
            Returns: (is_normalized, reason)
            """
            # Check 1: Look for normalization flags in .uns
            if 'log1p' in adata_obj.uns:
                return True, "log1p flag found in .uns"
            
            # Check 2: Examine data range
            # Log-normalized data typically has max values < 15
            # Raw counts typically have max values > 100
            max_val = adata_obj.X.max()
            if max_val < 20:
                return True, f"max value ({max_val:.2f}) suggests log-normalized data"
            
            # Check 3: Check for integer values (raw counts are integers)
            # Sample a subset to check
            if sp.issparse(adata_obj.X):
                sample_data = adata_obj.X.data[:min(10000, len(adata_obj.X.data))]
            else:
                sample_data = adata_obj.X.ravel()[:min(10000, adata_obj.X.size)]
            
            # If more than 90% of values are integers, likely raw counts
            int_fraction = np.mean(sample_data == np.floor(sample_data))
            if int_fraction < 0.1:
                return True, f"only {int_fraction*100:.1f}% integers suggests log-normalized"
            
            # Check 4: If max > 100, likely raw counts
            if max_val > 100:
                return False, f"max value ({max_val:.2f}) suggests raw counts"
            
            # Default: assume not normalized if uncertain
            return False, "uncertain - defaulting to raw counts assumption"
        
        is_normalized, reason = is_log_normalized(adata_train)
        print(f"    Data normalization check: {reason}")
        
        if not is_normalized:
            print("    Log-normalizing training data for marker detection...")
            # Normalize and log-transform (in-place on the copy)
            sc.pp.normalize_total(adata_train, target_sum=1e4)
            sc.pp.log1p(adata_train)
            
            print("    Log-normalizing test data to match training...")
            # Apply same normalization to test data for consistency
            sc.pp.normalize_total(adata_test, target_sum=1e4)
            sc.pp.log1p(adata_test)
        else:
            print("    Data already normalized - skipping normalization step")
        
        # Find markers (expensive - do once per fold)
        sc.tl.rank_genes_groups(
            adata_train, 
            groupby=label_key, 
            method='t-test',   # or 'wilcoxon' / 'logreg' as desired
            pts=True
        )
        
        # Convert results to a DataFrame similar to Seurat output
        groups = adata_train.obs[label_key].unique()
        markers_list = []
        for group in groups:
            scores = adata_train.uns['rank_genes_groups']['scores'][group]
            pvals = adata_train.uns['rank_genes_groups']['pvals'][group]
            logfc = adata_train.uns['rank_genes_groups']['logfoldchanges'][group]
            genes = adata_train.uns['rank_genes_groups']['names'][group]
            df = pd.DataFrame({
                'gene': genes,
                'cluster': group,
                'avg_logFC': logfc,
                'p_val': pvals,
                'score': scores
            })
            markers_list.append(df)
        
        markers = pd.concat(markers_list, ignore_index=True)
        
        print(f"Found {markers.shape[0]} marker genes across {len(groups)} cell types")
        
        # Cache results
        fold_cache[cache_key] = {
            'train': adata_train,
            'test': adata_test,
            'markers': markers,
            'fold_info': {
                'train_cells': adata_train.n_obs,
                'test_cells': adata_test.n_obs,
                'train_cell_types': adata_train.obs[label_key].value_counts(),
                'test_cell_types': adata_test.obs[label_key].value_counts()
            }
        }
    else:
        print(f"    Using cached fold {fold_index} data")
    
    return fold_cache[cache_key]


def aggregate_tool_results(all_predictions: List[str], all_true_labels: List[str], 
                          all_confidence_scores: List[float], all_cell_ids: List[str],
                          fold_metrics: List[Dict], fold_runtimes: List[float],
                          fold_peak_memories: List[Optional[float]],
                          fold_peak_vrams: List[Optional[float]],
                          tool_name: str) -> Dict[str, Any]:
    """
    Aggregate Results from Single Tool Across All Folds
    
    Purpose: Combines CV results for one tool into summary statistics
    Inputs:
      - all_predictions: Combined predictions from all folds
      - all_true_labels: Combined true labels from all folds
      - all_confidence_scores: Combined confidence scores
      - all_cell_ids: Combined cell identifiers
      - fold_metrics: List of per-fold metric results
      - fold_runtimes: List of per-fold runtimes
      - fold_peak_memories: List of per-fold peak memory (MB)
      - tool_name: Name of the tool
    Outputs: Dict with pooled_metrics, fold_variation, runtime_stats, memory_stats, detailed_results
    Key Logic: Provides both pooled analysis (Method B) and fold variation stats (Method A)
    """
    
    # Remove failed folds (None values)
    valid_fold_metrics = [fm for fm in fold_metrics if fm is not None]
    valid_runtimes = [rt for rt in fold_runtimes if not pd.isnull(rt)]
    valid_memories = [mem for mem in fold_peak_memories if mem is not None and not pd.isnull(mem)]
    
    # Pooled metrics (Method B - preferred)
    pooled_metrics = calculate_metrics(all_predictions, all_true_labels)
    pooled_metrics['confusion_matrix'] = confusion_matrix(all_true_labels, all_predictions)
    
    # Fold-wise variation (Method A - for stability assessment)
    if len(valid_fold_metrics) > 0:
        fold_accuracies = [fm['overall_accuracy'] for fm in valid_fold_metrics]
        fold_f1s = [fm['macro_f1'] for fm in valid_fold_metrics]
        fold_kappas = [fm['cohens_kappa'] for fm in valid_fold_metrics]
        fold_mccs = [fm.get('mcc', np.nan) for fm in valid_fold_metrics]
        fold_rare_f1s = [fm.get('rare_type_f1', np.nan) for fm in valid_fold_metrics]
        fold_unassigned = [fm.get('unassigned_rate', np.nan) for fm in valid_fold_metrics]
        
        fold_variation = {
            'accuracy_mean': np.mean(fold_accuracies),
            'accuracy_std': np.std(fold_accuracies, ddof=1) if len(fold_accuracies) > 1 else 0,
            'accuracy_folds': fold_accuracies,
            'accuracy_ci': np.mean(fold_accuracies) + np.array([-1.96, 1.96]) * (np.std(fold_accuracies, ddof=1) / np.sqrt(len(fold_accuracies))) if len(fold_accuracies) > 1 else [np.mean(fold_accuracies), np.mean(fold_accuracies)],
            
            'f1_mean': np.mean(fold_f1s),
            'f1_std': np.std(fold_f1s, ddof=1) if len(fold_f1s) > 1 else 0,
            'f1_folds': fold_f1s,
            'f1_ci': np.mean(fold_f1s) + np.array([-1.96, 1.96]) * (np.std(fold_f1s, ddof=1) / np.sqrt(len(fold_f1s))) if len(fold_f1s) > 1 else [np.mean(fold_f1s), np.mean(fold_f1s)],

            'kappa_mean': np.mean(fold_kappas),
            'kappa_std': np.std(fold_kappas, ddof=1) if len(fold_kappas) > 1 else 0,
            'kappa_folds': fold_kappas,
            'kappa_ci': np.mean(fold_kappas) + np.array([-1.96, 1.96]) * (np.std(fold_kappas, ddof=1) / np.sqrt(len(fold_kappas))) if len(fold_kappas) > 1 else [np.mean(fold_kappas), np.mean(fold_kappas)],
            
            'mcc_mean': np.nanmean(fold_mccs),
            'mcc_std': np.nanstd(fold_mccs, ddof=1) if len(fold_mccs) > 1 else 0,
            'mcc_folds': fold_mccs,
            
            'rare_f1_mean': np.nanmean(fold_rare_f1s),
            'rare_f1_std': np.nanstd(fold_rare_f1s, ddof=1) if len(fold_rare_f1s) > 1 else 0,
            'rare_f1_folds': fold_rare_f1s,
            
            'unassigned_mean': np.nanmean(fold_unassigned),
            'unassigned_std': np.nanstd(fold_unassigned, ddof=1) if len(fold_unassigned) > 1 else 0,
            'unassigned_folds': fold_unassigned
        }
    else:
        fold_variation = {
            'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'accuracy_folds': [],
            'f1_mean': np.nan, 'f1_std': np.nan, 'f1_folds': [],
            'kappa_mean': np.nan, 'kappa_std': np.nan, 'kappa_folds': [],
            'mcc_mean': np.nan, 'mcc_std': np.nan, 'mcc_folds': [],
            'rare_f1_mean': np.nan, 'rare_f1_std': np.nan, 'rare_f1_folds': [],
            'unassigned_mean': np.nan, 'unassigned_std': np.nan, 'unassigned_folds': []
        }
    
    # Runtime statistics
    runtime_stats = {
        'mean_seconds': np.mean(valid_runtimes) if valid_runtimes else np.nan,
        'std_seconds': np.std(valid_runtimes, ddof=1) if len(valid_runtimes) > 1 else 0,
        'total_seconds': np.sum(valid_runtimes) if valid_runtimes else np.nan,
        'fold_runtimes': valid_runtimes
    }
    
    # Memory statistics
    memory_stats = {
        'mean_mb': np.mean(valid_memories) if valid_memories else np.nan,
        'std_mb': np.std(valid_memories, ddof=1) if len(valid_memories) > 1 else 0,
        'max_mb': np.max(valid_memories) if valid_memories else np.nan,
        'min_mb': np.min(valid_memories) if valid_memories else np.nan,
        'fold_memories': valid_memories
    }
    
    # VRAM statistics
    valid_vrams = [v for v in fold_peak_vrams if v is not None and not pd.isnull(v) and v > 0]
    vram_stats = {
        'mean_mb': np.mean(valid_vrams)  if valid_vrams else np.nan,
        'std_mb':  np.std(valid_vrams, ddof=1) if len(valid_vrams) > 1 else 0,
        'max_mb':  np.max(valid_vrams)   if valid_vrams else np.nan,
        'fold_vrams': valid_vrams
    }

    return {
        'tool_name': tool_name,
        'pooled_metrics': pooled_metrics,
        'fold_variation': fold_variation,
        'runtime_stats': runtime_stats,
        'memory_stats': memory_stats,
        'vram_stats': vram_stats,
        'detailed_results': {
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'confidence_scores': all_confidence_scores,
            'cell_ids': all_cell_ids
        },
        'num_successful_folds': len(valid_fold_metrics),
        'num_total_folds': len(fold_metrics)
    }


def run_single_tool_cv(adata: ad.AnnData, tool_function, folds: List[List[str]], tool_name: str) -> Dict[str, Any]:
    """
    Run Cross-Validation for Single Tool
    
    Purpose: Executes one tool across all CV folds with error handling
    Inputs:
      - adata: Full AnnData object
      - tool_function: Function that takes (train, test, markers) and returns predictions
      - folds: Output from create_cv_folds()
      - tool_name: Name for logging/identification
    Outputs: Aggregated results from aggregate_tool_results()
    Key Logic: Runs tool on each fold, accumulates results, handles failures gracefully
    """
    
    print(f"\n=== Running {tool_name} ===")
    
    # Storage for this tool
    all_predictions = []
    all_true_labels = []
    all_confidence_scores = []
    all_cell_ids = []
    fold_metrics = []
    fold_runtimes = []
    fold_peak_memories = []  # Track peak system RAM per fold
    fold_peak_vrams = []     # Track peak GPU VRAM per fold
    
    # Process each fold
    for i in range(len(folds)):
        print(f"  Fold {i+1}/{len(folds)}...")
        
        # Get pre-computed fold data
        fold_data = get_fold_data_cached(adata, folds, i)
        
        # Run tool on this fold
        try:
            fold_result = tool_function(fold_data['train'], fold_data['test'], fold_data['markers'])
            runtime     = fold_result.get('method_walltime', np.nan)
            
            # Validate tool output
            required_keys = ['predictions', 'true_labels', 'confidence_scores', 'cell_ids']
            if not all(key in fold_result for key in required_keys):
                raise ValueError(f"Tool function must return dict with: {required_keys}")
            
            # Accumulate results for pooled analysis
            all_predictions.extend(fold_result['predictions'])
            all_true_labels.extend(fold_result['true_labels'])
            all_confidence_scores.extend(fold_result['confidence_scores'])
            all_cell_ids.extend(fold_result['cell_ids'])
            
            # Store fold-wise metrics
            fold_metrics.append(calculate_metrics(
                predicted=fold_result['predictions'],
                true_labels=fold_result['true_labels']
            ))
            fold_runtimes.append(runtime)
            
            # Collect peak memory and VRAM
            peak_mem_mb  = fold_result.get('peak_memory_mb', None)
            peak_vram_mb = fold_result.get('peak_vram_mb', None)
            fold_peak_memories.append(peak_mem_mb)
            fold_peak_vrams.append(peak_vram_mb)
            
            # Progress message
            memory_str = f", RAM: {peak_mem_mb:.0f} MB" if peak_mem_mb is not None else ""
            vram_str   = f", VRAM: {peak_vram_mb:.0f} MB" if peak_vram_mb else ""
            print(f"    Completed in {runtime:.2f}s (method only), accuracy: {fold_metrics[-1]['overall_accuracy']*100:.2f}%{memory_str}{vram_str}")
            
        except Exception as e:
            warnings.warn(f"Tool {tool_name} failed on fold {i+1}: {str(e)}")
            fold_metrics.append(None)
            fold_runtimes.append(np.nan)
            fold_peak_memories.append(None)
            fold_peak_vrams.append(None)
            print(f"    FAILED: {str(e)}")
    
    # Aggregate results for this tool
    return aggregate_tool_results(all_predictions, all_true_labels, 
                                 all_confidence_scores, all_cell_ids,
                                 fold_metrics, fold_runtimes, fold_peak_memories,
                                 fold_peak_vrams, tool_name)


def aggregate_all_tools(all_tool_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Aggregate Results from All Tools 
    
    Purpose: Creates comparison table and summary statistics across all tools
    Inputs:
      - all_tool_results: Dict of results from run_single_tool_cv() for each tool
    Outputs: Dict with comparison_table, detailed_results, best_tool, summary_stats
    Key Logic: Extracts key metrics from each tool, sorts by kappa, provides rankings
    """
    
    # Create comparison table
    comparison_data = []
    
    # Extract metrics for each tool
    for tool_name, result in all_tool_results.items():
        comparison_data.append({
            'tool': tool_name,
            'pooled_accuracy': result['pooled_metrics']['overall_accuracy'],
            'pooled_macro_f1': result['pooled_metrics']['macro_f1'],
            'pooled_kappa': result['pooled_metrics']['cohens_kappa'],
            'pooled_mcc': result['pooled_metrics']['mcc'],
            'pooled_rare_type_f1': result['pooled_metrics']['rare_type_f1'],
            'pooled_unassigned_rate': result['pooled_metrics']['unassigned_rate'],
            'accuracy_mean': result['fold_variation']['accuracy_mean'],
            'accuracy_std': result['fold_variation']['accuracy_std'],
            'accuracy_ci_lower': result['fold_variation']['accuracy_ci'][0],
            'accuracy_ci_upper': result['fold_variation']['accuracy_ci'][1],
            'f1_mean': result['fold_variation']['f1_mean'],
            'f1_std': result['fold_variation']['f1_std'],
            'f1_ci_lower': result['fold_variation']['f1_ci'][0],
            'f1_ci_upper': result['fold_variation']['f1_ci'][1],
            'kappa_mean': result['fold_variation']['kappa_mean'],
            'kappa_std': result['fold_variation']['kappa_std'],
            'kappa_ci_lower': result['fold_variation']['kappa_ci'][0],
            'kappa_ci_upper': result['fold_variation']['kappa_ci'][1],
            'mcc_mean': result['fold_variation']['mcc_mean'],
            'mcc_std': result['fold_variation']['mcc_std'],
            'rare_f1_mean': result['fold_variation']['rare_f1_mean'],
            'rare_f1_std': result['fold_variation']['rare_f1_std'],
            'unassigned_mean': result['fold_variation']['unassigned_mean'],
            'unassigned_std': result['fold_variation']['unassigned_std'],
            'runtime_mean': result['runtime_stats']['mean_seconds'],
            'runtime_std': result['runtime_stats']['std_seconds'],
            'peak_system_memory_mean_mb': result['memory_stats']['mean_mb'],
            'peak_system_memory_std_mb': result['memory_stats']['std_mb'],
            'peak_system_memory_max_mb': result['memory_stats']['max_mb'],
            'peak_GPU_memory_mean_mb': result['vram_stats']['mean_mb'],
            'peak_GPU_memory_std_mb': result['vram_stats']['std_mb'],
            'peak_GPU_memory_max_mb': result['vram_stats']['max_mb'],
            'successful_folds': result['num_successful_folds']
        })
    
    # Create DataFrame and sort by pooled kappa
    comparison_df = pd.DataFrame(comparison_data)
    if comparison_df.empty:
        warnings.warn(
            "No tool results to aggregate — all tools may have failed or none were "
            f"registered. Tools attempted: {list(all_tool_results.keys())}"
        )
        return {
            'comparison_table': comparison_df,
            'detailed_results': all_tool_results,
            'best_tool': None,
            'summary_stats': {
                'num_tools': 0,
                'best_accuracy': np.nan, 'best_f1': np.nan,
                'best_kappa': np.nan, 'best_mcc': np.nan,
                'fastest_tool': None, 'most_memory_efficient': None
            }
        }
    comparison_df = comparison_df.sort_values('pooled_kappa', ascending=False).reset_index(drop=True)
    
    return {
        'comparison_table': comparison_df,
        'detailed_results': all_tool_results,
        'best_tool': comparison_df.iloc[0]['tool'] if len(comparison_df) > 0 else None,
        'summary_stats': {
            'num_tools': len(comparison_df),
            'best_accuracy': comparison_df['pooled_accuracy'].max() if len(comparison_df) > 0 else np.nan,
            'best_f1': comparison_df['pooled_macro_f1'].max() if len(comparison_df) > 0 else np.nan,
            'best_kappa': comparison_df['pooled_kappa'].max() if len(comparison_df) > 0 else np.nan,
            'best_mcc': comparison_df['pooled_mcc'].max() if len(comparison_df) > 0 else np.nan,
            'fastest_tool': comparison_df.loc[comparison_df['runtime_mean'].idxmin(), 'tool'] if len(comparison_df) > 0 else None,
            'most_memory_efficient': comparison_df.loc[comparison_df['peak_system_memory_mean_mb'].idxmin(), 'tool'] if len(comparison_df) > 0 and not comparison_df['peak_system_memory_mean_mb'].isna().all() else None
        }
    }


def run_all_tools_cv(adata: ad.AnnData, tools_to_run: List[str] = None, 
                    cv_config: Dict = None) -> Dict[str, Dict]:
    """
    Main Benchmarking Execution Function
    
    Purpose: Orchestrates the complete benchmarking workflow
    Inputs:
      - adata: AnnData object with Ground_Truth_Celltype in obs
      - tools_to_run: List of tool names to benchmark (default: TOOLS_TO_RUN)
      - cv_config: Dict with k, group_by, stratify_by, seed (default: CV_CONFIG)
    Outputs: Dict of results from each tool (pass to aggregate_all_tools())
    Workflow:
      1. Creates k-fold splits once (reused for all tools)
      2. Runs each tool across all folds with error handling
      3. Returns aggregated results ready for comparison
    """
    
    if tools_to_run is None:
        tools_to_run = TOOLS_TO_RUN
    if cv_config is None:
        cv_config = CV_CONFIG
    
    print("=== CELL TYPE ANNOTATION BENCHMARKING ===")
    print(f"Dataset: {len(adata.obs)} cells x {len(adata.var)} genes")
    
    if 'Ground_Truth_Celltype' in adata.obs.columns:
        unique_types = adata.obs['Ground_Truth_Celltype'].unique()
        print(f"Cell types: {', '.join(map(str, unique_types))}")
    
    print(f"Tools to run: {', '.join(tools_to_run)}")
    
    # Create folds once, reuse for all tools
    folds = create_cv_folds(adata, 
                           k=cv_config['k'],
                           group_by=cv_config['group_by'], 
                           stratify_by=cv_config['stratify_by'],
                           seed=cv_config['seed'])
    
    # Initialize results storage
    all_tool_results = {}
    
    # Run each tool
    for tool_name in tools_to_run:
        tool_function = TOOL_REGISTRY.get(tool_name)
        if tool_function is None:
            warnings.warn(f"Tool {tool_name} not found in registry")
            continue
        
        # Run CV for this tool
        tool_results = run_single_tool_cv(adata, tool_function, folds, tool_name)
        all_tool_results[tool_name] = tool_results
    
    return all_tool_results


def run_all_tools_single_split(
    adata: ad.AnnData,
    tools_to_run: List[str] = None,
    label_col: str = "Ground_Truth_Celltype",
    seed: int = 123
) -> Dict[str, Dict]:
    """
    Single 80/20 Split Execution — for L9 simulated datasets.

    Purpose: Runs all tools on one stratified 80/20 train/test split.
             Used for L9 benchmark replicates where each dataset is an
             independent Splatter simulation — CV is not needed since
             fold variance is near-zero for these dataset sizes, and
             independent replicates capture between-simulation variance
             more informatively.
    Inputs:
      - adata: AnnData with label_col in obs
      - tools_to_run: list of tool names from TOOL_REGISTRY
      - label_col: obs column containing ground truth labels
      - seed: random seed for reproducibility
    Outputs: Dict in the same structure as run_all_tools_cv() — compatible
             with aggregate_all_tools() without modification.
    """

    if tools_to_run is None:
        tools_to_run = TOOLS_TO_RUN

    print("=== CELL TYPE ANNOTATION BENCHMARKING (Single 80/20 Split) ===")
    print(f"Dataset: {len(adata.obs)} cells x {len(adata.var)} genes")

    if label_col in adata.obs.columns:
        unique_types = adata.obs[label_col].unique()
        print(f"Cell types: {', '.join(map(str, unique_types))}")

    print(f"Tools to run: {', '.join(tools_to_run)}")

    # Stratified 80/20 split using StratifiedKFold(n_splits=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)                                                       
    valid_mask = adata.obs[label_col].notna()                                                                                
    n_dropped = int((~valid_mask).sum())                                                                                     
    if n_dropped > 0:                                                                                                        
        print(f"Warning: Dropping {n_dropped} cells with NaN '{label_col}'")                                                 
    adata = adata[valid_mask].copy()
    all_indices = np.arange(len(adata.obs))                                                                                  
    stratify_labels = adata.obs[label_col]
    train_idx, test_idx = next(skf.split(all_indices, stratify_labels))                                                      
   
    adata_train = adata[train_idx].copy()                                                                                    
    adata_test = adata[test_idx].copy()
    print(f"Train: {adata_train.n_obs} cells, Test: {adata_test.n_obs} cells") 

    # Find marker genes on training data
    try:
        sc.tl.rank_genes_groups(adata_train, groupby=label_col,
                               method='wilcoxon', use_raw=False)
        markers = sc.get.rank_genes_groups_df(adata_train, group=None)
        print(f"Found {len(markers)} marker genes")
    except Exception as e:
        warnings.warn(f"Marker detection failed: {e}")
        markers = pd.DataFrame()

    # Run each tool
    all_tool_results = {}

    for tool_name in tools_to_run:
        tool_function = TOOL_REGISTRY.get(tool_name)
        if tool_function is None:
            warnings.warn(f"Tool {tool_name} not found in registry")
            continue

        print(f"\n=== Running {tool_name} (single split) ===")

        try:
            t0 = time.perf_counter()
            result = tool_function(adata_train, adata_test, markers)
            wall_time = time.perf_counter() - t0

            runtime = result.get('method_walltime', wall_time)
            peak_mem = result.get('peak_memory_mb', None)
            peak_vram = result.get('peak_vram_mb', None)

            predictions = result['predictions']
            true_labels = result['true_labels']
            confidence_scores = result.get('confidence_scores', [0.0] * len(predictions))
            cell_ids = result.get('cell_ids', list(adata_test.obs.index))

            metrics = calculate_metrics(
                predicted=predictions,
                true_labels=true_labels
            )

            # Build result structure matching aggregate_tool_results() output
            pooled_metrics = dict(metrics)
            pooled_metrics['confusion_matrix'] = None

            fold_variation = {
                'accuracy_mean': metrics['overall_accuracy'],
                'accuracy_std': np.nan,
                'accuracy_folds': [metrics['overall_accuracy']],
                'accuracy_ci': [np.nan, np.nan],
                'f1_mean': metrics['macro_f1'],
                'f1_std': np.nan,
                'f1_folds': [metrics['macro_f1']],
                'f1_ci': [np.nan, np.nan],
                'kappa_mean': metrics['cohens_kappa'],
                'kappa_std': np.nan,
                'kappa_folds': [metrics['cohens_kappa']],
                'kappa_ci': [np.nan, np.nan],
                'mcc_mean': metrics['mcc'],
                'mcc_std': np.nan,
                'mcc_folds': [metrics['mcc']],
                'rare_f1_mean': metrics['rare_type_f1'],
                'rare_f1_std': np.nan,
                'rare_f1_folds': [metrics['rare_type_f1']],
                'unassigned_mean': metrics['unassigned_rate'],
                'unassigned_std': np.nan,
                'unassigned_folds': [metrics['unassigned_rate']]
            }

            valid_mem = [peak_mem] if peak_mem is not None else []
            valid_vram = [peak_vram] if peak_vram is not None and peak_vram > 0 else []

            tool_result = {
                'tool_name': tool_name,
                'pooled_metrics': pooled_metrics,
                'fold_variation': fold_variation,
                'runtime_stats': {
                    'mean_seconds': runtime,
                    'std_seconds': np.nan,
                    'total_seconds': runtime,
                    'fold_runtimes': [runtime]
                },
                'memory_stats': {
                    'mean_mb': peak_mem if peak_mem is not None else np.nan,
                    'std_mb': np.nan,
                    'max_mb': peak_mem if peak_mem is not None else np.nan,
                    'min_mb': peak_mem if peak_mem is not None else np.nan,
                    'fold_memories': valid_mem
                },
                'vram_stats': {
                    'mean_mb': peak_vram if peak_vram is not None and peak_vram > 0 else np.nan,
                    'std_mb': np.nan,
                    'max_mb': peak_vram if peak_vram is not None and peak_vram > 0 else np.nan,
                    'fold_vrams': valid_vram
                },
                'detailed_results': {
                    'predictions': predictions,
                    'true_labels': true_labels,
                    'confidence_scores': confidence_scores,
                    'cell_ids': cell_ids
                },
                'num_successful_folds': 1,
                'num_total_folds': 1
            }

            all_tool_results[tool_name] = tool_result

            memory_str = f", RAM: {peak_mem:.0f} MB" if peak_mem is not None else ""
            vram_str = f", VRAM: {peak_vram:.0f} MB" if peak_vram else ""
            print(f"    Completed in {runtime:.2f}s, accuracy: {metrics['overall_accuracy']*100:.2f}%{memory_str}{vram_str}")

        except Exception as e:
            warnings.warn(f"Tool {tool_name} failed: {str(e)}")
            print(f"    FAILED: {str(e)}")

            # Store all-NaN result
            nan_variation = {
                'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'accuracy_folds': [],
                'accuracy_ci': [np.nan, np.nan],
                'f1_mean': np.nan, 'f1_std': np.nan, 'f1_folds': [],
                'f1_ci': [np.nan, np.nan],
                'kappa_mean': np.nan, 'kappa_std': np.nan, 'kappa_folds': [],
                'kappa_ci': [np.nan, np.nan],
                'mcc_mean': np.nan, 'mcc_std': np.nan, 'mcc_folds': [],
                'rare_f1_mean': np.nan, 'rare_f1_std': np.nan, 'rare_f1_folds': [],
                'unassigned_mean': np.nan, 'unassigned_std': np.nan, 'unassigned_folds': []
            }
            all_tool_results[tool_name] = {
                'tool_name': tool_name,
                'pooled_metrics': {
                    'overall_accuracy': np.nan, 'macro_f1': np.nan,
                    'cohens_kappa': np.nan, 'mcc': np.nan,
                    'unassigned_rate': np.nan, 'rare_type_f1': np.nan,
                    'rare_type_name': None,
                    'per_class_metrics': pd.DataFrame(),
                    'confusion_matrix': None
                },
                'fold_variation': nan_variation,
                'runtime_stats': {
                    'mean_seconds': np.nan, 'std_seconds': np.nan,
                    'total_seconds': np.nan, 'fold_runtimes': []
                },
                'memory_stats': {
                    'mean_mb': np.nan, 'std_mb': np.nan,
                    'max_mb': np.nan, 'min_mb': np.nan, 'fold_memories': []
                },
                'vram_stats': {
                    'mean_mb': np.nan, 'std_mb': np.nan,
                    'max_mb': np.nan, 'fold_vrams': []
                },
                'detailed_results': {
                    'predictions': [], 'true_labels': [],
                    'confidence_scores': [], 'cell_ids': []
                },
                'num_successful_folds': 0,
                'num_total_folds': 1
            }

    return all_tool_results


def run_separate_datasets(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    tools_to_run: List[str] = None,
    label_col: str = "Ground_Truth_Celltype"
) -> Dict[str, Dict]:
    """
    Cross-Dataset Generalization Execution (Mode 5)

    Purpose: Trains tools on one dataset and evaluates on a distinct second dataset.
             No gene intersection is performed.
    Inputs:
      - adata_train: Training AnnData object
      - adata_test: Testing AnnData object
      - tools_to_run: list of tool names from TOOL_REGISTRY
      - label_col: obs column containing ground truth labels
    Outputs: Dict in the same structure as run_all_tools_cv()
    """

    if tools_to_run is None:
        tools_to_run = TOOLS_TO_RUN

    print("=== CELL TYPE ANNOTATION BENCHMARKING (Separate Datasets Mode) ===")
    print(f"Training dataset: {len(adata_train.obs)} cells x {len(adata_train.var)} genes")
    print(f"Testing dataset:  {len(adata_test.obs)} cells x {len(adata_test.var)} genes")

    if label_col in adata_train.obs.columns:
        train_types = adata_train.obs[label_col].unique()
        print(f"Train cell types: {', '.join(map(str, train_types))}")
    if label_col in adata_test.obs.columns:
        test_types = adata_test.obs[label_col].unique()
        print(f"Test cell types:  {', '.join(map(str, test_types))}")

    print(f"Tools to run: {', '.join(tools_to_run)}")

    # Find marker genes on training data
    try:
        sc.tl.rank_genes_groups(adata_train, groupby=label_col,
                               method='wilcoxon', use_raw=False)
        markers = sc.get.rank_genes_groups_df(adata_train, group=None)
        print(f"Found {len(markers)} marker genes on training dataset")
    except Exception as e:
        warnings.warn(f"Marker detection failed: {e}")
        markers = pd.DataFrame()

    # Run each tool
    all_tool_results = {}

    for tool_name in tools_to_run:
        tool_function = TOOL_REGISTRY.get(tool_name)
        if tool_function is None:
            warnings.warn(f"Tool {tool_name} not found in registry")
            continue

        print(f"\n=== Running {tool_name} (Separate Datasets Mode) ===")

        try:
            t0 = time.perf_counter()
            result = tool_function(adata_train, adata_test, markers)
            wall_time = time.perf_counter() - t0

            runtime = result.get('method_walltime', wall_time)
            peak_mem = result.get('peak_memory_mb', None)
            peak_vram = result.get('peak_vram_mb', None)

            predictions = result['predictions']
            true_labels = result['true_labels']
            confidence_scores = result.get('confidence_scores', [0.0] * len(predictions))
            cell_ids = result.get('cell_ids', list(adata_test.obs.index))

            metrics = calculate_metrics(
                predicted=predictions,
                true_labels=true_labels
            )

            # Build result structure matching aggregate_tool_results() output
            pooled_metrics = dict(metrics)
            pooled_metrics['confusion_matrix'] = None

            fold_variation = {
                'accuracy_mean': metrics['overall_accuracy'],
                'accuracy_std': np.nan,
                'accuracy_folds': [metrics['overall_accuracy']],
                'accuracy_ci': [np.nan, np.nan],
                'f1_mean': metrics['macro_f1'],
                'f1_std': np.nan,
                'f1_folds': [metrics['macro_f1']],
                'f1_ci': [np.nan, np.nan],
                'kappa_mean': metrics['cohens_kappa'],
                'kappa_std': np.nan,
                'kappa_folds': [metrics['cohens_kappa']],
                'kappa_ci': [np.nan, np.nan],
                'mcc_mean': metrics['mcc'],
                'mcc_std': np.nan,
                'mcc_folds': [metrics['mcc']],
                'rare_f1_mean': metrics['rare_type_f1'],
                'rare_f1_std': np.nan,
                'rare_f1_folds': [metrics['rare_type_f1']],
                'unassigned_mean': metrics['unassigned_rate'],
                'unassigned_std': np.nan,
                'unassigned_folds': [metrics['unassigned_rate']]
            }

            valid_mem = [peak_mem] if peak_mem is not None else []
            valid_vram = [peak_vram] if peak_vram is not None and peak_vram > 0 else []

            tool_result = {
                'tool_name': tool_name,
                'pooled_metrics': pooled_metrics,
                'fold_variation': fold_variation,
                'runtime_stats': {
                    'mean_seconds': runtime,
                    'std_seconds': np.nan,
                    'total_seconds': runtime,
                    'fold_runtimes': [runtime]
                },
                'memory_stats': {
                    'mean_mb': peak_mem if peak_mem is not None else np.nan,
                    'std_mb': np.nan,
                    'max_mb': peak_mem if peak_mem is not None else np.nan,
                    'min_mb': peak_mem if peak_mem is not None else np.nan,
                    'fold_memories': valid_mem
                },
                'vram_stats': {
                    'mean_mb': peak_vram if peak_vram is not None and peak_vram > 0 else np.nan,
                    'std_mb': np.nan,
                    'max_mb': peak_vram if peak_vram is not None and peak_vram > 0 else np.nan,
                    'fold_vrams': valid_vram
                },
                'detailed_results': {
                    'predictions': predictions,
                    'true_labels': true_labels,
                    'confidence_scores': confidence_scores,
                    'cell_ids': cell_ids
                },
                'num_successful_folds': 1,
                'num_total_folds': 1
            }

            all_tool_results[tool_name] = tool_result

            memory_str = f", RAM: {peak_mem:.0f} MB" if peak_mem is not None else ""
            vram_str = f", VRAM: {peak_vram:.0f} MB" if peak_vram else ""
            print(f"    Completed in {runtime:.2f}s, accuracy: {metrics['overall_accuracy']*100:.2f}%{memory_str}{vram_str}")

        except Exception as e:
            warnings.warn(f"Tool {tool_name} failed: {str(e)}")
            print(f"    FAILED: {str(e)}")

            # Store all-NaN result
            nan_variation = {
                'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'accuracy_folds': [],
                'accuracy_ci': [np.nan, np.nan],
                'f1_mean': np.nan, 'f1_std': np.nan, 'f1_folds': [],
                'f1_ci': [np.nan, np.nan],
                'kappa_mean': np.nan, 'kappa_std': np.nan, 'kappa_folds': [],
                'kappa_ci': [np.nan, np.nan],
                'mcc_mean': np.nan, 'mcc_std': np.nan, 'mcc_folds': [],
                'rare_f1_mean': np.nan, 'rare_f1_std': np.nan, 'rare_f1_folds': [],
                'unassigned_mean': np.nan, 'unassigned_std': np.nan, 'unassigned_folds': []
            }
            all_tool_results[tool_name] = {
                'tool_name': tool_name,
                'pooled_metrics': {
                    'overall_accuracy': np.nan, 'macro_f1': np.nan,
                    'cohens_kappa': np.nan, 'mcc': np.nan,
                    'unassigned_rate': np.nan, 'rare_type_f1': np.nan,
                    'rare_type_name': None,
                    'per_class_metrics': pd.DataFrame(),
                    'confusion_matrix': None
                },
                'fold_variation': nan_variation,
                'runtime_stats': {
                    'mean_seconds': np.nan, 'std_seconds': np.nan,
                    'total_seconds': np.nan, 'fold_runtimes': []
                },
                'memory_stats': {
                    'mean_mb': np.nan, 'std_mb': np.nan,
                    'max_mb': np.nan, 'min_mb': np.nan, 'fold_memories': []
                },
                'vram_stats': {
                    'mean_mb': np.nan, 'std_mb': np.nan,
                    'max_mb': np.nan, 'fold_vrams': []
                },
                'detailed_results': {
                    'predictions': [], 'true_labels': [],
                    'confidence_scores': [], 'cell_ids': []
                },
                'num_successful_folds': 0,
                'num_total_folds': 1
            }

    return all_tool_results
