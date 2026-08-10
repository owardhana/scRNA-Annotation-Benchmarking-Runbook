"""benchmarking_main.py

Entry point for the Python benchmarking harness -- the AnnData/scanpy
counterpart of R/run_benchmark.R.

Importing this module registers the shipped tool wrappers; it does not start a
run. The example functions at the bottom show how each execution mode is
driven.

    import benchmarking_main as bm
    bm.available_tools()
    bm.example_single_split("Darmanis-Brain-2015_for_use_subset.h5ad")

WHY A SECOND HARNESS
--------------------
Most annotation tools ship as R packages, so the R harness carries the bulk of
the roster. The deep-learning and semi-supervised paradigms are Python-native
and several of their tools cannot be driven from R at all: they want an AnnData
object, a GPU, and frequently a conda environment of their own whose
dependencies conflict with everything else. This harness exists for those.

The consequence is the isolation model. Where the R harness calls tools
in-process, this one launches each tool inside its own conda environment as a
subprocess, passing data through a temporary directory. See tools/run_ACTINN.py.

WHAT IS SHIPPED
---------------
Two exemplar wrappers, one per Python-native paradigm: ACTINN (deep learning)
and scArches (semi-supervised). The registry below lists the full set of tools
the study ran through this harness, so the shape is visible, but the wrapper
files for the other 18 are not included -- see docs/tool-roster.md. The
per-tool import guard makes their absence harmless: a tool whose module cannot
be imported is reported and skipped, not fatal.
"""

import os
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional

import scanpy as sc

from benchmarking_helpers import (  # noqa: F401  (re-exported for convenience)
    TOOL_REGISTRY,
    aggregate_all_tools,
    aggregate_tool_results,
    calculate_metrics,
    create_cv_folds,
    fold_cache,
    get_fold_data_cached,
    run_all_tools_cv,
    run_all_tools_single_split,
    run_separate_datasets,
    run_single_tool_cv,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Neither directory is shipped: this repository contains no data. See
# 01-data-preparation/ for how the .h5ad objects are produced.
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SC_BENCH_DATA_DIR", os.path.join(HARNESS_DIR, "..", "..", "data"))
OUTPUT_DIR = os.environ.get("SC_BENCH_OUTPUT_DIR", os.path.join(HARNESS_DIR, "..", "..", "results"))

sys.path.insert(0, os.path.join(HARNESS_DIR, "tools"))


# --------------------------------------------------------------------------
# Tool registration
# --------------------------------------------------------------------------
# Every tool listed here ran in the study. Each needs its own conda
# environment -- the dependency sets are mutually incompatible -- so a given
# machine will usually have only a few of them installed. _register() therefore
# reports what it could not import and moves on rather than raising.
#
# Only ACTINN and scArches ship as wrapper files in this repository.
TOOL_MODULES = [
    # (registry name,   module,             function)
    ("ACTINN",          "run_ACTINN",       "run_ACTINN_function"),          # deep learning
    ("scArches",        "run_scArches",     "run_scArches_function"),        # semi-supervised
    # --- run in the study; wrappers not shipped, see docs/tool-roster.md ---
    ("scDeepSort",      "run_scDeepSort",   "run_scDeepSort_function"),
    ("scHash",          "run_scHash",       "run_scHash_function"),
    ("scDeepinsight",   "run_scDeepinsight", "run_scDeepinsight_function"),
    ("TOSICA",          "run_TOSICA",       "run_TOSICA_function"),
    ("scMMT",           "run_scMMT",        "run_scMMT_function"),
    ("scnym",           "run_scnym",        "run_scnym_function"),
    ("CIForm",          "run_CIForm",       "run_CIForm_function"),
    ("mtANN",           "run_mtANN",        "run_mtANN_function"),
    ("scTransSort",     "run_scTransSort",  "run_scTransSort_function"),
    ("scBalance",       "run_scBalance",    "run_scBalance_function"),
    ("MARS",            "run_MARS",         "run_MARS_function"),
    ("Cell_BLAST",      "run_Cell_BLAST",   "run_Cell_BLAST_function"),
    ("SCANVI",          "run_SCANVI",       "run_SCANVI_function"),
    ("ItClust",         "run_ItClust",      "run_ItClust_function"),
    ("scGAD",           "run_scGAD",        "run_scGAD_function"),
    ("scSemiCluster",   "run_scSemiCluster", "run_scSemiCluster_function"),
    ("scSemiGAN",       "run_scSemiGAN",    "run_scSemiGAN_function"),
    ("C2S",             "run_C2S",          "run_C2S_function"),
]


def _try_import(module_name: str, func_name: str) -> Optional[Callable]:
    """Import one tool wrapper, returning None if it is unavailable."""
    try:
        module = __import__(module_name, fromlist=[func_name])
        return getattr(module, func_name)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  - {module_name}: unavailable ({exc})")
        return None


def register_tools(verbose: bool = True) -> Dict[str, Callable]:
    """Populate TOOL_REGISTRY with whatever is importable on this machine.

    Mutates TOOL_REGISTRY in place so that the lookups inside
    benchmarking_helpers see the entries.
    """
    if verbose:
        print("Registering Python tools:")

    for tool_name, module_name, func_name in TOOL_MODULES:
        fn = _try_import(module_name, func_name)
        if fn is not None:
            TOOL_REGISTRY[tool_name] = fn
            if verbose:
                print(f"  + {tool_name}")

    if verbose:
        print(f"Registered {len(TOOL_REGISTRY)} of {len(TOOL_MODULES)} tools")
    return TOOL_REGISTRY


def available_tools() -> List[str]:
    """Names that can actually be dispatched on this machine."""
    return sorted(TOOL_REGISTRY)


register_tools()


# --------------------------------------------------------------------------
# Example runs
# --------------------------------------------------------------------------
# One function per execution mode, with the study's arguments as defaults.
# Calling one requires data this repository does not ship.
#
# Only three of the five R modes exist here: the full-dataset LLM and database
# modes are R-only, since no Python-native tool in the roster consumes markers
# that way.

def _save_comparison_table(final_results, label: str) -> str:
    """Write a comparison table with a timestamped filename."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.csv")
    final_results["comparison_table"].to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return out_path


def example_cv(dataset: str = "Darmanis-Brain-2015_for_use_subset.h5ad",
               tools: Optional[List[str]] = None):
    """MODE 1 -- k-fold cross-validation.

    Available but not used for any reported result; the reported within-dataset
    phases use the single split below.
    """
    adata = sc.read_h5ad(os.path.join(DATA_DIR, dataset))

    cv_config = {
        "k": 3,
        "group_by": None,                          # e.g. "donor_id" for grouped CV
        "stratify_by": "Ground_Truth_Celltype",
        "seed": 123,
    }

    results = run_all_tools_cv(adata, tools_to_run=tools or available_tools(),
                               cv_config=cv_config)
    final_results = aggregate_all_tools(results)
    print(final_results["comparison_table"])
    return _save_comparison_table(final_results, "cv")


def example_single_split(dataset: str = "Darmanis-Brain-2015_for_use_subset.h5ad",
                         tools: Optional[List[str]] = None):
    """MODE 2 -- single stratified 80/20 split. The mode behind Phases 1, 2 and 5."""
    adata = sc.read_h5ad(os.path.join(DATA_DIR, dataset))

    results = run_all_tools_single_split(adata, tools_to_run=tools or available_tools())
    final_results = aggregate_all_tools(results)
    print(final_results["comparison_table"])
    return _save_comparison_table(final_results, "single_split")


def example_separate_datasets(train_dataset: str = "Cellbench-10xv2_original_subset.h5ad",
                              test_dataset: str = "Cellbench-CelSeq2_original_subset.h5ad",
                              tools: Optional[List[str]] = None):
    """MODE 5 -- train on one dataset, test on another. Cross-platform transfer, Phase 3."""
    adata_train = sc.read_h5ad(os.path.join(DATA_DIR, train_dataset))
    adata_test = sc.read_h5ad(os.path.join(DATA_DIR, test_dataset))

    results = run_separate_datasets(adata_train, adata_test,
                                    tools_to_run=tools or available_tools())
    final_results = aggregate_all_tools(results)
    print(final_results["comparison_table"])
    return _save_comparison_table(final_results, "cross_dataset")


def example_batch(datasets: Optional[List[str]] = None,
                  tools: Optional[List[str]] = None):
    """Run the single-split mode over several datasets.

    The fold cache is keyed by fold index only, so it must be cleared between
    datasets or the second dataset reuses the first one's splits.
    """
    datasets = datasets or [
        "Darmanis-Brain-2015_for_use_subset.h5ad",
        "He-Skin-2020_for_use_subset.h5ad",
        "Grun-Pancreas-2016_for_use_subset.h5ad",
    ]

    for h5ad_file in datasets:
        dataset_name = os.path.splitext(h5ad_file)[0]
        print(f"\n====== Dataset: {dataset_name} ======")
        try:
            fold_cache.clear()
            adata = sc.read_h5ad(os.path.join(DATA_DIR, h5ad_file))
            results = run_all_tools_single_split(adata, tools_to_run=tools or available_tools())
            final_results = aggregate_all_tools(results)
            print(final_results["comparison_table"])
            _save_comparison_table(final_results, dataset_name)
        except Exception as exc:                              # noqa: BLE001
            print(f"Error for {dataset_name}: {exc}")


if __name__ == "__main__":
    print("Available tools:", available_tools())
