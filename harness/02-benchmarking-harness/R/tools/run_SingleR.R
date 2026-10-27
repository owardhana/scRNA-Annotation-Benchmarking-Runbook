# run_SingleR.R
#################################################
# EXEMPLAR: correlation / reference-based tool
#
# SingleR correlates each test cell against a labelled reference. Unlike the
# marker-based exemplar it genuinely needs `seurat_train` — that object *is*
# the reference, and its Ground_Truth_Celltype column supplies the label
# vocabulary the tool can predict from.
#
# It also shows the two things that distinguish this paradigm:
#   - conversion to SingleCellExperiment at the wrapper boundary
#   - confidence taken from a tool-native quantity (delta.max) rather than a
#     probability, which is typical: the contract asks for a per-cell number,
#     not a calibrated one
#################################################

#' SingleR cell-type annotation
#'
#' Inputs:
#'   - seurat_train: labelled reference Seurat object
#'   - seurat_test:  test Seurat object to annotate
#'   - markers:      FindAllMarkers() data frame, used to restrict the gene set
#' Outputs: list(predictions, true_labels, confidence_scores, cell_ids,
#'                runtime_secs, peak_system_memory_mb)
run_SingleR_function <- function(seurat_train, seurat_test, markers) {

  for (pkg in c("SingleR", "SingleCellExperiment")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(pkg, " package not available. Please install it first.")
    }
    library(pkg, character.only = TRUE)
  }

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

  # --- Marker adaptation -------------------------------------------------
  # SingleR takes a named list of gene vectors, one per label, and restricts
  # its correlation to those genes.
  convert_markers_to_singler <- function(marker_df) {
    filtered_markers <- prepare_markers(marker_df)

    if (nrow(filtered_markers) == 0) {
      warning("No markers found after filtering")
      return(NULL)
    }

    lapply(split(filtered_markers$gene, filtered_markers$cluster), unique)
  }

  marker_list <- convert_markers_to_singler(markers)
  if (is.null(marker_list)) return(default_return())

  # --- Input verification ------------------------------------------------
  # The shared preprocessing log-normalises before the harness ever sees the
  # object, so this is a guard rather than a step: it catches an object that
  # was assembled some other way.
  ensure_normalised <- function(seurat_obj, label) {
    has_data <- tryCatch(
      length(GetAssayData(seurat_obj, assay = "RNA", layer = "data")) > 0,
      error = function(e) FALSE
    )
    if (!has_data) {
      warning("No log-normalised layer on ", label, "; normalising in the wrapper")
      seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
    }
    seurat_obj
  }

  seurat_train <- ensure_normalised(seurat_train, "reference")
  seurat_test  <- ensure_normalised(seurat_test,  "test")

  # --- Conversion to the tool's object model -----------------------------
  # Deliberately outside the measurement window: format conversion is a cost
  # of the harness, not of the annotation method.
  to_sce <- function(seurat_obj, label) {
    tryCatch(as.SingleCellExperiment(seurat_obj), error = function(e) {
      warning("Failed to convert ", label, " to SingleCellExperiment: ", e$message)
      NULL
    })
  }

  sce_train <- to_sce(seurat_train, "reference")
  sce_test  <- to_sce(seurat_test,  "test")
  if (is.null(sce_train) || is.null(sce_test)) return(default_return())

  cat(sprintf("Reference: %d genes x %d cells\n", nrow(sce_train), ncol(sce_train)))
  cat(sprintf("Test:      %d genes x %d cells\n", nrow(sce_test),  ncol(sce_test)))
  cat(sprintf("Marker lists: %d labels\n", length(marker_list)))

  # --- Core annotation call ----------------------------------------------
  tracked <- with_resource_tracking({
    tryCatch(
      SingleR(test      = sce_test,
              ref       = sce_train,
              labels    = sce_train$Ground_Truth_Celltype,
              genes     = marker_list,
              de.method = "wilcox"),
      error = function(e) {
        warning(paste("SingleR execution failed:", e$message))
        NULL
      }
    )
  })

  result                <- tracked$value
  runtime_secs          <- tracked$runtime_secs
  peak_system_memory_mb <- tracked$peak_system_memory_mb

  if (is.null(result)) return(default_return(runtime_secs, peak_system_memory_mb))

  # --- Extract and standardise -------------------------------------------
  cell_predictions <- if (!is.null(result$labels)) result$labels else rep("Unknown", ncol(seurat_test))

  # delta.max is the margin between the best and median correlation score:
  # SingleR's own measure of how decisive the call was.
  confidence_scores <- if (!is.null(result$delta.max)) result$delta.max else rep(0, ncol(seurat_test))

  cell_predictions[is.na(cell_predictions)]   <- "Unknown"
  confidence_scores[is.na(confidence_scores)] <- 0

  # The harness requires one prediction per test cell, in test-cell order.
  # Returning a short vector would silently misalign against the truth labels,
  # so fail loudly into the sentinel instead.
  if (length(cell_predictions) != ncol(seurat_test)) {
    warning(sprintf("Prediction length mismatch: expected %d, got %d",
                    ncol(seurat_test), length(cell_predictions)))
    return(default_return(runtime_secs, peak_system_memory_mb))
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
run_SingleR <- run_SingleR_function
