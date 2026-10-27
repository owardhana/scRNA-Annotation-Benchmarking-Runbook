# tool_registry.R
#################################################
# Maps tool names -> wrapper functions, and groups those names into the
# batches the execution modes are run over.
#
# SCOPE: this harness ships one worked wrapper per methodological paradigm,
# not all 63 evaluated configurations. The registry below therefore registers
# the exemplars in `R/tools/`. Every wrapper exactly as the study ran it is in
# `as-run/benchmarking/` at the repository root. The complete roster of what was
# benchmarked — paradigm, package, version, and whether it was reported or
# excluded — is in `harness/docs/tool-roster.md`.
#
# The registry is the single place a tool name becomes callable. Every
# execution mode in `benchmarking_helpers.R` resolves tools the same way:
#
#     tool_function <- TOOL_REGISTRY[[tool_name]]
#     tool_result   <- tool_function(seurat_train, seurat_test, markers)
#
# so a name that is not registered here cannot be run, regardless of whether a
# file for it exists on disk.
#################################################

TOOL_REGISTRY <- list()


#################################################
# 1. ONE TOOL, ONE ENTRY
#################################################
# The common case: a wrapper file defines run_<TOOL>_function() and it is
# registered under its own name.

TOOL_REGISTRY[["scSorter"]] <- run_scSorter_function   # marker-based
TOOL_REGISTRY[["SingleR"]]  <- run_SingleR_function    # similarity-based (reference)
TOOL_REGISTRY[["scLearn"]]  <- run_scLearn_function    # classical ML (R; non-neural metric learning)


#################################################
# 2. ONE TOOL, MANY CONFIGURATIONS
#################################################
# Several "tools" in the roster are really one implementation run under
# different settings. Rather than duplicating a wrapper file per setting — the
# study originally carried 20 near-identical scPred files and 9 near-identical
# mLLMCelltype files — the wrapper takes the setting as an argument and the
# registry generates one entry per configuration.

# --- scPred: one wrapper, one caret model per configuration ---
# scPred's own default is svmRadial; the remaining models are caret method
# strings passed straight through to scPred::trainModel().
SCPRED_MODELS <- c(
  "svmRadial",    # scPred package default, registered as plain "scPred"
  "svmLinear", "svmPoly", "rf", "knn", "lda", "nb", "glm", "glmboost",
  "bayesglm", "earth", "mlp", "nnet", "avNNet", "regLogistic", "multinom",
  "adaboost", "xgbTree"
)

for (scpred_model in SCPRED_MODELS) {
  tool_name <- if (scpred_model == "svmRadial") "scPred" else paste0("scPred_", scpred_model)
  # local() freezes the loop variable into each closure; without it every
  # entry would capture the same (final) value of scpred_model.
  TOOL_REGISTRY[[tool_name]] <- local({
    model <- scpred_model
    function(seurat_train, seurat_test, markers) {
      run_scPred_function(seurat_train, seurat_test, markers, model = model)
    }
  })
}

# --- mLLMCelltype: the model list is an argument ---
# One model ID gives single-model annotation; several switch the wrapper to
# multi-model discussion. The study reports only the consensus arm below.
# `api_key_env` names the environment variables the keys are read from at
# call time; keys are never stored in this repository.
TOOL_REGISTRY[["mLLMCelltype_consensus"]] <- function(seurat_train, seurat_test, markers) {
  run_mLLMCelltype_function(
    seurat_train, seurat_test, markers,
    # The three models of the reported consensus (as-run/benchmarking/LLM-based/run_mLLMCelltype.R)
    models      = c("anthropic/claude-sonnet-4-6", "gpt-5.4", "google/gemini-3.1-pro-preview"),
    api_key_env = c("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
  )
}


#################################################
# 3. ONE TOOL, TWO MARKER SOURCES
#################################################
# Phases 1-3 give marker-based tools a marker table derived from the data
# itself (FindAllMarkers on the training split). Phase 4 instead gives them
# markers looked up in CellMarker2.0 for the dataset's tissue, with no sight of
# the data. Same algorithm, different evidence — so it is one wrapper with a
# `marker_source` argument, registered twice.
#
# The two paths are not cosmetically different: database markers arrive with
# synthetic statistics (avg_log2FC = 1.0, p_val_adj = 0, pct.1 = 1.0), which
# makes the usual statistical filtering meaningless, and CellMarker symbols
# need sanitising before use. See docs/tool-wrapper-contract.md.

TOOL_REGISTRY[["scType"]]          <- function(seurat_train, seurat_test, markers) {
  run_scType_function(seurat_train, seurat_test, markers, marker_source = "de")
}
TOOL_REGISTRY[["scType_database"]] <- function(seurat_train, seurat_test, markers) {
  run_scType_function(seurat_train, seurat_test, markers, marker_source = "database")
}


#################################################
# TOOL GROUP DEFINITIONS
#################################################
# Convenience batches for the execution modes. In the full study these vectors
# held every tool of each paradigm; here they hold the shipped exemplars.
# Commented-out entries in the original recorded run status (broken, too slow)
# — that bookkeeping now lives in docs/tool-roster.md instead.

# scType is cluster-dependent: it is outside the 47-configuration cell-level
# comparison of Phases 1-2 and is reported separately (and, as scType_database,
# in Phase 4). Drop it from marker_based to reproduce the cell-level panel only.
marker_based    <- c("scSorter", "scType")
reference_based <- c("SingleR")                       # the similarity-based family
classicML_based <- c("scPred", "scPred_rf", "scPred_svmLinear", "scLearn")
DL_based        <- character(0)  # R-side DL wrappers (NeuCA) are in as-run/; the DL exemplar is Python (ACTINN)
LLM_based       <- c("mLLMCelltype_consensus")
DB_tools        <- c("scType_database")


#' List every registered tool name
#'
#' Useful as a sanity check before a batch run: a name absent from this vector
#' will fail at dispatch time, not at source time.
available_tools <- function() sort(names(TOOL_REGISTRY))
