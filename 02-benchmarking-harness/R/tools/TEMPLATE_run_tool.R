# TEMPLATE_run_tool.R
#################################################
# Skeleton for a new tool wrapper. Copy, rename to run_<TOOL>.R, fill in the
# four marked sections, and register the result in tool_registry.R.
#
# A wrapper's job is narrow: adapt the harness's standard inputs into whatever
# the tool wants, run it, and adapt the output back. It should not preprocess,
# should not compute metrics, and should not write files. The harness does all
# three. See docs/tool-wrapper-contract.md for the full contract.
#
# The rule that catches people out: predictions, true_labels and cell_ids must
# be the SAME LENGTH and in the SAME ORDER. Tools that return a subset, or
# their own ordering, must be realigned here — a silently misaligned vector
# scores as noise rather than failing.
#################################################

#' <TOOL> cell-type annotation
#'
#' Inputs:
#'   - seurat_train: labelled training Seurat object. Reference-based and
#'                   supervised tools use it; marker-based tools ignore it but
#'                   still accept it, so every wrapper has one signature.
#'   - seurat_test:  test Seurat object to annotate. Already log-normalised,
#'                   with 2,000 variable features and 50 PCs from the shared
#'                   preprocessing — do not redo that work here.
#'   - markers:      FindAllMarkers() data frame with columns gene, cluster,
#'                   avg_log2FC, p_val_adj, pct.1, pct.2. Computed once per
#'                   dataset by the harness and reused across tools.
#' Outputs: list(predictions, true_labels, confidence_scores, cell_ids,
#'                runtime_secs, peak_system_memory_mb)
run_TOOL_function <- function(seurat_train, seurat_test, markers) {

  # --- 1. DEPENDENCY GUARD ------------------------------------------------
  # Guard rather than a bare library() call: a missing package should fail
  # this tool, not prevent the harness from loading.
  if (!requireNamespace("TOOLPACKAGE", quietly = TRUE)) {
    stop("TOOLPACKAGE not available. Please install it first.")
  }
  library(TOOLPACKAGE)

  # --- 2. FAILURE SENTINEL ------------------------------------------------
  # Every early exit returns this shape, so a tool that fails still produces a
  # comparable row instead of a hole in the results table.
  #
  # "Unknown" must be capitalised exactly: calculate_metrics() matches that
  # literal string to compute the unassigned rate and to exclude declined cells
  # from the per-class metrics. A lowercase "unknown" is scored as if it were a
  # real cell-type prediction.
  default_return <- function(runtime_secs = NA_real_, peak_system_memory_mb = NA_real_) {
    list(
      predictions           = rep("Unknown", ncol(seurat_test)),
      true_labels           = seurat_test$Ground_Truth_Celltype,
      confidence_scores     = rep(0, ncol(seurat_test)),
      cell_ids              = colnames(seurat_test),
      runtime_secs          = runtime_secs,
      peak_system_memory_mb = peak_system_memory_mb
    )
  }

  # --- 3. INPUT ADAPTATION ------------------------------------------------
  # Reshape the standard inputs into the tool's native format. Everything here
  # is outside the measurement window, deliberately: format conversion is a
  # cost of the harness, not of the annotation method.
  #
  # Marker-consuming tools should start from prepare_markers(), which applies
  # the study-wide filter (positive markers, pct.1 >= 0.10, top 20 per cluster
  # by log2FC, no p-value cutoff). It is supplied by the harness; do not
  # re-source benchmarking_helpers.R here.
  filtered_markers <- prepare_markers(markers)
  if (nrow(filtered_markers) == 0) {
    warning("No markers left after filtering")
    return(default_return())
  }

  tool_input <- NULL   # <- reshape filtered_markers / seurat_test as needed

  # --- 4. CORE ANNOTATION CALL --------------------------------------------
  # Wrap ONLY the annotation itself. For a tool with a training step, that
  # means fitting and prediction; for a marker-based tool with no training
  # step, inference alone.
  #
  # Write the call once. The old pattern of an if/else on peakRAM's
  # availability meant maintaining two copies of the same algorithm.
  tracked <- with_resource_tracking({
    tryCatch(
      TOOLPACKAGE::annotate(tool_input),
      error = function(e) {
        warning(paste("TOOL failed:", e$message))
        NULL
      }
    )
  })

  results               <- tracked$value
  runtime_secs          <- tracked$runtime_secs
  peak_system_memory_mb <- tracked$peak_system_memory_mb

  if (is.null(results)) return(default_return(runtime_secs, peak_system_memory_mb))

  # --- 5. OUTPUT ADAPTATION -----------------------------------------------
  cell_predictions  <- as.character(results$labels)

  # Confidence is whatever the tool exposes for its own certainty — a
  # probability, a correlation margin, a score. It is not required to be
  # calibrated or comparable across tools. Tools that expose nothing report a
  # constant; say so in a comment when you do.
  confidence_scores <- as.numeric(results$scores)

  cell_predictions[is.na(cell_predictions)]   <- "Unknown"
  confidence_scores[is.na(confidence_scores)] <- 0

  # Cluster-level tools must expand to cells before returning, e.g.
  #   cell_predictions <- cluster_calls[match(seurat_test$Ground_Truth_Cluster,
  #                                           names(cluster_calls))]
  if (length(cell_predictions) != ncol(seurat_test)) {
    warning(sprintf("Prediction length mismatch: expected %d, got %d",
                    ncol(seurat_test), length(cell_predictions)))
    return(default_return(runtime_secs, peak_system_memory_mb))
  }

  list(
    predictions           = cell_predictions,
    true_labels           = seurat_test$Ground_Truth_Celltype,
    confidence_scores     = confidence_scores,
    cell_ids              = colnames(seurat_test),
    runtime_secs          = runtime_secs,
    peak_system_memory_mb = peak_system_memory_mb
  )
}

# Backward-compatible alias; the registry uses the _function form.
run_TOOL <- run_TOOL_function
