# run_scSorter.R
#################################################
# EXEMPLAR: marker-based tool
#
# scSorter assigns cell types from a marker signature table. It never sees
# labels, so `seurat_train` is unused and present only to satisfy the common
# wrapper signature.
#
# This is the simplest of the shipped exemplars and the best one to read first:
# it shows the whole wrapper contract with almost no tool-specific machinery.
# See docs/tool-wrapper-contract.md for the contract itself.
#################################################

#' scSorter cell-type annotation
#'
#' Inputs:
#'   - seurat_train: training Seurat object (unused; interface consistency only)
#'   - seurat_test:  test Seurat object, already normalised with variable features set
#'   - markers:      FindAllMarkers() data frame (gene, cluster, avg_log2FC, pct.1, ...)
#' Outputs: list(predictions, true_labels, confidence_scores, cell_ids,
#'                runtime_secs, peak_system_memory_mb)
#' Algorithm: reshape markers into scSorter's (Type, Marker, Weight) annotation
#'   table, restrict the expression matrix to marker + variable genes, run scSorter.
run_scSorter_function <- function(seurat_train, seurat_test, markers) {

  if (!requireNamespace("scSorter", quietly = TRUE)) {
    stop("scSorter package not available. Please install it first.")
  }
  library(scSorter)

  # Every early exit returns this, so a failed tool still produces a row with
  # the right shape. "Unknown" (capital U) is the framework sentinel that
  # calculate_metrics() counts as unassigned - see the contract doc.
  default_return <- function() {
    list(
      predictions           = rep("Unknown", ncol(seurat_test)),
      true_labels           = seurat_test$Ground_Truth_Celltype,
      confidence_scores     = rep(0, ncol(seurat_test)),
      cell_ids              = colnames(seurat_test),
      runtime_secs          = NA_real_,
      peak_system_memory_mb = NA_real_
    )
  }

  # --- Marker adaptation -------------------------------------------------
  # prepare_markers() is provided by the harness (benchmarking_helpers.R) and
  # applies the study-wide filter: positive markers only, expressed in >=10% of
  # the target cells, top 20 per cluster by log2FC, no p-value cutoff.
  convert_markers_to_scsorter <- function(marker_df) {
    filtered_markers <- prepare_markers(marker_df)

    # scSorter expects exactly these three columns
    data.frame(
      Type             = filtered_markers$cluster,
      Marker           = filtered_markers$gene,
      Weight           = filtered_markers$avg_log2FC,
      stringsAsFactors = FALSE
    )
  }

  # --- Feature selection -------------------------------------------------
  # The test object arrives already normalised with variable features set by
  # the shared preprocessing, so there is nothing to recompute here.
  topgenes <- VariableFeatures(seurat_test)
  if (length(topgenes) == 0) {
    warning("No variable features found, using all genes")
    topgenes <- rownames(seurat_test)
  }

  expr <- GetAssayData(seurat_test, layer = "data")

  # scSorter is sensitive to sparsely detected genes; keep variable genes seen
  # in more than 10% of cells.
  topgene_filter <- rowSums(as.matrix(expr)[topgenes, ] != 0) > ncol(expr) * 0.1
  topgenes       <- topgenes[topgene_filter]

  anno          <- convert_markers_to_scsorter(markers)
  anno_filtered <- anno[anno$Marker %in% rownames(expr), ]

  if (nrow(anno_filtered) == 0) {
    warning("No marker genes found in expression data")
    return(default_return())
  }

  picked_genes <- unique(c(anno_filtered$Marker, topgenes))
  expr         <- expr[rownames(expr) %in% picked_genes, ]

  # --- Core annotation call ----------------------------------------------
  # Everything above is setup and is deliberately outside the measurement
  # window; only this call is charged to the tool.
  tracked <- with_resource_tracking({
    tryCatch(
      scSorter(expr, anno_filtered),
      error = function(e) {
        warning(paste("scSorter failed:", e$message))
        NULL
      }
    )
  })

  results               <- tracked$value
  runtime_secs          <- tracked$runtime_secs
  peak_system_memory_mb <- tracked$peak_system_memory_mb

  if (is.null(results)) {
    failed <- default_return()
    failed$runtime_secs          <- runtime_secs
    failed$peak_system_memory_mb <- peak_system_memory_mb
    return(failed)
  }

  # --- Extract and standardise -------------------------------------------
  cell_predictions <- results$Pred_Type
  cell_scores      <- results$Scores

  # Confidence is the tool's own score for the assigned type. Tools that
  # expose nothing comparable report a constant instead - the contract asks
  # for a number per cell, not for a calibrated probability.
  if (!is.null(cell_scores)) {
    confidence_scores <- apply(cell_scores, 1, max, na.rm = TRUE)
  } else {
    confidence_scores <- rep(1, length(cell_predictions))
  }

  list(
    predictions           = as.character(cell_predictions),
    true_labels           = seurat_test$Ground_Truth_Celltype,
    confidence_scores     = confidence_scores,
    cell_ids              = colnames(seurat_test),
    runtime_secs          = runtime_secs,
    peak_system_memory_mb = peak_system_memory_mb
  )
}

# Backward-compatible alias; the registry uses the _function form.
run_scSorter <- run_scSorter_function
