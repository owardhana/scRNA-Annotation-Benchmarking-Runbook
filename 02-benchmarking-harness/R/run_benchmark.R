# run_benchmark.R
#################################################
# Entry point for the R benchmarking harness.
#
# Sourcing this file loads the harness and registers the shipped tool
# wrappers. It does not start a run: the example blocks at the bottom are
# functions, so nothing executes until you call one.
#
#   source("run_benchmark.R")
#   available_tools()
#   example_single_split(dataset = "Darmanis-Brain-2015_for_use_subset.rds")
#
# Layout:
#   1. Libraries
#   2. Path resolution and configuration
#   3. Harness modules and tool wrappers
#   4. Example runs, one per execution mode
#
# The five execution modes live in benchmarking_helpers.R:
#   run_all_tools_cv()                     k-fold cross-validation
#   run_all_tools_single_split()           single stratified 80/20 split
#   run_all_tools_full_dataset_llm()       full dataset, LLM tools
#   run_all_tools_full_dataset_database()  full dataset, CellMarker2.0 tools
#   run_separate_datasets()                train on A, test on B
#
# Every mode returns something aggregate_all_tools() can turn into a
# comparison table. See docs/architecture.md for which phase used which mode.
#################################################


#################################################
# 1. LIBRARIES
#################################################
# Only what the harness itself needs. Per-tool dependencies are loaded by the
# wrappers, each behind a requireNamespace() guard, so a missing tool package
# fails that one tool rather than preventing the harness from loading.
library(Seurat)
library(tidyverse)
library(Matrix)
library(caret)
library(SingleCellExperiment)
library(SummarizedExperiment)


#################################################
# 2. PATHS AND CONFIGURATION
#################################################

#' Locate this script's directory
#'
#' Resolves whether the file was source()d, run via Rscript, or neither, so the
#' harness can find its own modules without depending on the working directory.
#' Override with SC_BENCH_HARNESS_DIR if you relocate the files.
harness_dir <- function() {
  from_env <- Sys.getenv("SC_BENCH_HARNESS_DIR")
  if (nzchar(from_env)) return(normalizePath(from_env, mustWork = TRUE))

  # source() records the file being parsed in the calling frame
  for (frame_index in seq_len(sys.nframe())) {
    ofile <- sys.frame(frame_index)$ofile
    if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  }

  # Rscript path/to/run_benchmark.R
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }

  normalizePath(".")
}

HARNESS_DIR <- harness_dir()

# Where your copies of the preprocessed objects live, and where comparison
# tables are written. Neither directory is shipped: this repository contains no
# data. See 01-data-preparation/ for how the .rds objects are produced.
DATA_DIR   <- Sys.getenv("SC_BENCH_DATA_DIR",   file.path(HARNESS_DIR, "..", "..", "data"))
OUTPUT_DIR <- Sys.getenv("SC_BENCH_OUTPUT_DIR", file.path(HARNESS_DIR, "..", "..", "results"))


#################################################
# 3. HARNESS MODULES AND TOOL WRAPPERS
#################################################
source(file.path(HARNESS_DIR, "resource_tracking.R"))

# Cell Ontology label normalisation. Modes 3 and 4 map free-text tool output
# onto the ground-truth vocabulary with it, so it is loaded before the helpers
# that call it rather than re-sourced inside each call.
source(file.path(HARNESS_DIR, "..", "scoring", "normalisation_pipeline.R"))

source(file.path(HARNESS_DIR, "benchmarking_helpers.R"))

# Tool wrappers. In the full study this was ~70 source() lines maintained by
# hand; here the directory is simply enumerated, so adding a wrapper file is
# enough to load it. It still has to be registered to become runnable.
for (tool_file in list.files(file.path(HARNESS_DIR, "tools"),
                             pattern = "^run_.*\\.R$", full.names = TRUE)) {
  source(tool_file)
}

source(file.path(HARNESS_DIR, "tool_registry.R"))


#################################################
# 4. EXAMPLE RUNS
#################################################
# One function per execution mode. These are illustrations of how the harness
# is driven, with the arguments the study used as defaults. They are functions
# rather than commented-out blocks so that they stay syntactically valid, but
# calling one requires data this repository does not ship.

#' Write a comparison table with a timestamped filename
save_comparison_table <- function(final, label) {
  dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(OUTPUT_DIR,
                        sprintf("%s_%s.csv", label, format(Sys.time(), "%Y%m%d_%H%M%S")))
  write.csv(final$comparison_table, out_path, row.names = FALSE)
  message("Wrote ", out_path)
  invisible(out_path)
}


#' MODE 1 - k-fold cross-validation
#'
#' Within-dataset evaluation with per-fold variation statistics. Available in
#' the harness but not used for any reported result; Phases 1, 2 and 5 use the
#' single split below instead.
example_cv <- function(dataset = "Darmanis-Brain-2015_for_use_subset.rds",
                       tools   = reference_based) {
  seurat_obj <- readRDS(file.path(DATA_DIR, dataset))
  Idents(seurat_obj) <- "Ground_Truth_Celltype"

  cv_config <- list(
    k           = 3,
    group_by    = NA,                       # e.g. "donor_id" for grouped CV
    stratify_by = "Ground_Truth_Celltype",
    seed        = 123
  )

  results <- run_all_tools_cv(seurat_obj, tools_to_run = tools, cv_config = cv_config)
  final   <- aggregate_all_tools(results)
  print(final$comparison_table)
  save_comparison_table(final, "cv")
}


#' MODE 2 - single stratified 80/20 split
#'
#' The mode behind Phases 1, 2 and 5. Splits stratified on Ground_Truth_Celltype.
example_single_split <- function(dataset = "Darmanis-Brain-2015_for_use_subset.rds",
                                 tools   = c(marker_based, reference_based),
                                 seed    = 123) {
  seurat_obj <- readRDS(file.path(DATA_DIR, dataset))
  Idents(seurat_obj) <- "Ground_Truth_Celltype"

  # Cell types too rare to appear on both sides of the split break several
  # tools; drop them before splitting rather than mid-run.
  min_cells    <- 3
  type_counts  <- table(seurat_obj$Ground_Truth_Celltype)
  keep_types   <- names(type_counts[type_counts >= min_cells])
  seurat_obj   <- subset(seurat_obj,
                         cells = colnames(seurat_obj)[seurat_obj$Ground_Truth_Celltype %in% keep_types])

  results <- run_all_tools_single_split(seurat_obj, tools_to_run = tools, seed = seed)
  final   <- aggregate_all_tools(results)
  print(final$comparison_table)
  save_comparison_table(final, "single_split")
}


#' MODE 3 - full dataset, LLM tools
#'
#' No train/test split: LLM tools annotate cluster-level marker summaries, so
#' there is nothing to train. Ground_Truth_Cluster supplies a perfect
#' clustering and markers are derived over all cells. Phase 4.
example_full_dataset_llm <- function(dataset      = "Darmanis-Brain-2015_for_use_subset.rds",
                                     tools        = LLM_based,
                                     dataset_name = "Darmanis-Brain-2015") {
  seurat_obj <- readRDS(file.path(DATA_DIR, dataset))
  Idents(seurat_obj) <- "Ground_Truth_Celltype"

  results <- run_all_tools_full_dataset_llm(seurat_obj,
                                            tools_to_run = tools,
                                            dataset_name = dataset_name)
  final <- aggregate_all_tools(results)
  print(final$comparison_table)
  save_comparison_table(final, "llm")
}


#' MODE 4 - full dataset, CellMarker2.0 tools
#'
#' Marker tools given database markers for the dataset's tissue instead of
#' markers derived from the data. `tissue_class` selects the CellMarker2.0
#' subset. Phase 4.
example_full_dataset_database <- function(dataset      = "Darmanis-Brain-2015_for_use_subset.rds",
                                          tools        = DB_tools,
                                          tissue_class = "Brain",
                                          dataset_name = "Darmanis-Brain-2015") {
  seurat_obj <- readRDS(file.path(DATA_DIR, dataset))
  Idents(seurat_obj) <- "Ground_Truth_Celltype"

  results <- run_all_tools_full_dataset_database(seurat_obj,
                                                 tools_to_run = tools,
                                                 tissue_class = tissue_class,
                                                 dataset_name = dataset_name)
  final <- aggregate_all_tools(results)
  print(final$comparison_table)
  save_comparison_table(final, "database")
}


#' MODE 5 - train on one dataset, test on another
#'
#' Cross-platform transfer. Phase 3 ran nine such pairings across three
#' technology blocks (CellBench, Pancreas, PBMCbench).
example_separate_datasets <- function(train_dataset = "Cellbench-10xv2_original_subset.rds",
                                      test_dataset  = "Cellbench-CelSeq2_original_subset.rds",
                                      tools         = reference_based) {
  seurat_train <- readRDS(file.path(DATA_DIR, train_dataset))
  seurat_test  <- readRDS(file.path(DATA_DIR, test_dataset))

  results <- run_separate_datasets(seurat_train, seurat_test, tools_to_run = tools)
  final   <- aggregate_all_tools(results)
  print(final$comparison_table)
  save_comparison_table(final, "cross_dataset")
}


#' Batch example - one mode over several datasets
#'
#' The pattern used for the per-phase sweeps. Two details matter: `fold_cache`
#' is keyed by fold index only, so it must be cleared between datasets or folds
#' from the previous dataset are reused; and large Seurat objects should be
#' dropped and gc() called each iteration, or a long sweep exhausts memory.
example_batch <- function(datasets = c("Darmanis-Brain-2015_for_use_subset.rds",
                                       "He-Skin-2020_for_use_subset.rds",
                                       "Grun-Pancreas-2016_for_use_subset.rds"),
                          tools    = reference_based) {
  for (rds_file in datasets) {
    dataset_name <- tools::file_path_sans_ext(rds_file)
    message("\n====== Dataset: ", dataset_name, " ======")

    tryCatch({
      fold_cache <<- list()   # see note above
      seurat_obj <- readRDS(file.path(DATA_DIR, rds_file))
      Idents(seurat_obj) <- "Ground_Truth_Celltype"

      results <- run_all_tools_single_split(seurat_obj, tools_to_run = tools)
      final   <- aggregate_all_tools(results)
      print(final$comparison_table)
      save_comparison_table(final, dataset_name)

      rm(seurat_obj, results, final)
      gc()
    }, error = function(e) {
      message(sprintf("Error for %s: %s", dataset_name, e$message))
    })
  }
}
