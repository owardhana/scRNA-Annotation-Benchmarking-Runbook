"""TEMPLATE_run_tool.py

Skeleton for a new Python tool wrapper. Copy, rename to run_<TOOL>.py, fill in
the marked sections, and add a row to TOOL_MODULES in benchmarking_main.py.

The contract mirrors the R one (docs/tool-wrapper-contract.md) with AnnData in
place of Seurat and a dict in place of a list. The resource keys differ, and
that difference is deliberate -- see the module docstring of
benchmarking_helpers.py.

Two shapes are possible:

  IN-PROCESS   the tool imports cleanly alongside the harness. Simplest; use
               it when you can.

  SUBPROCESS   the tool needs its own conda environment. Generate a script,
               run it with `conda run`, pass data through a TemporaryDirectory,
               and have the child measure itself. See run_ACTINN.py for a
               complete worked example -- this template shows the in-process
               shape only.
"""

import time
import warnings
from typing import Any, Dict

import anndata as ad
import pandas as pd

# For the subprocess shape, resolve conda through the shared helper rather than
# hardcoding a path:
#     from conda_env import conda_env_path, conda_executable


def run_TOOL_function(adata_train: ad.AnnData,
                      adata_test: ad.AnnData,
                      markers: pd.DataFrame) -> Dict[str, Any]:
    """<TOOL> cell-type annotation.

    Inputs:
        adata_train: labelled training AnnData. Supervised and semi-supervised
            tools use it; tools that only need markers ignore it but still
            accept it, so every wrapper has one signature.
        adata_test:  AnnData to annotate.
        markers:     marker table from sc.tl.rank_genes_groups, reshaped long
            with columns gene, cluster, avg_logFC, p_val, score. Computed once
            per fold by the harness and reused across tools.

    Returns:
        dict with predictions, true_labels, confidence_scores, cell_ids,
        method_walltime, peak_memory_mb, peak_vram_mb.

        predictions, true_labels and cell_ids MUST be the same length and in
        the same order. A tool that returns a subset, or its own ordering, has
        to be realigned here -- a misaligned vector scores as noise instead of
        failing.
    """

    # --- 1. FAILURE SENTINEL ---------------------------------------------
    # Every early exit returns this shape, so a failed tool produces a
    # comparable row rather than a hole in the results table.
    #
    # "Unknown" must be capitalised exactly: calculate_metrics() matches that
    # literal string to compute the unassigned rate and to exclude declined
    # cells from the per-class metrics.
    def default_return() -> Dict[str, Any]:
        n_cells = adata_test.n_obs
        return {
            'predictions':       ["Unknown"] * n_cells,
            'true_labels':       adata_test.obs['Ground_Truth_Celltype'].astype(str).tolist(),
            'confidence_scores': [0.0] * n_cells,
            'cell_ids':          adata_test.obs_names.tolist(),
            'method_walltime':   None,
            'peak_memory_mb':    None,
            'peak_vram_mb':      None,
        }

    for name, adata in (("training", adata_train), ("test", adata_test)):
        if 'Ground_Truth_Celltype' not in adata.obs.columns:
            warnings.warn(f"Ground_Truth_Celltype not found in {name} data")
            return default_return()

    # --- 2. INPUT ADAPTATION ---------------------------------------------
    # Reshape into the tool's native format. Deliberately outside the
    # measurement window: conversion is a cost of the harness, not of the
    # annotation method.
    #
    # Note which layer the tool wants. The harness supplies a log-normalised
    # .X with the raw matrix in .layers['counts']; a tool that models counts
    # must take the layer, not .X.
    tool_input = None  # <- build from adata_train / adata_test / markers

    # --- 3. CORE ANNOTATION CALL -----------------------------------------
    # Time and measure only the annotation itself. For a tool with a training
    # step that means fitting and prediction; for one without, inference alone.
    #
    # For real memory numbers use memory_profiler.memory_usage(...,
    # include_children=True) as run_ACTINN.py does; time.perf_counter alone is
    # enough only when memory is genuinely not of interest.
    start_time = time.perf_counter()
    try:
        results = None  # <- TOOLPACKAGE.annotate(tool_input)
    except Exception as exc:                                   # noqa: BLE001
        warnings.warn(f"TOOL failed: {exc}")
        return default_return()
    method_walltime = time.perf_counter() - start_time

    if results is None:
        return default_return()

    # --- 4. OUTPUT ADAPTATION --------------------------------------------
    predictions = [str(p) for p in results]

    # Confidence is whatever the tool exposes for its own certainty. It is not
    # required to be calibrated or comparable across tools; when a tool exposes
    # nothing, report a constant and say so in a comment.
    confidence_scores = [1.0] * len(predictions)

    if len(predictions) != adata_test.n_obs:
        warnings.warn(
            f"Prediction length mismatch: expected {adata_test.n_obs}, got {len(predictions)}"
        )
        return default_return()

    return {
        'predictions':       predictions,
        'true_labels':       adata_test.obs['Ground_Truth_Celltype'].astype(str).tolist(),
        'confidence_scores': confidence_scores,
        'cell_ids':          adata_test.obs_names.tolist(),
        'method_walltime':   method_walltime,
        'peak_memory_mb':    None,   # populate when measuring memory
        'peak_vram_mb':      None,   # GPU tools only
    }
