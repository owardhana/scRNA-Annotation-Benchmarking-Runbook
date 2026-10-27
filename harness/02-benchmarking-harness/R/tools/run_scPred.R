# run_scPred.R
#################################################
# EXEMPLAR: classical ML tool, and the one-tool-many-configurations pattern
#
# scPred projects cells into a PCA feature space and trains a caret classifier
# in it. The classifier is a parameter, so the 20 scPred entries in the roster
# are one implementation run 20 times, not 20 tools.
#
# The original study carried one file per classifier — 19 near-identical
# copies, each differing by a single caret method string. Worse, because the
# resource-tracking if/else duplicated the algorithm body, that string appeared
# *twice per file* and the two copies drifted: this wrapper's own ancestor
# passed no model at all in one branch and "svmRadial" in the other. Here the
# classifier is an argument, the body is written once, and tool_registry.R
# generates the 20 registry entries from a vector of method names.
#################################################

#' scPred cell-type annotation
#'
#' Inputs:
#'   - seurat_train: labelled training Seurat object
#'   - seurat_test:  test Seurat object to annotate
#'   - markers:      FindAllMarkers() data frame (unused; scPred does its own
#'                   feature selection in PCA space)
#'   - model:        caret method name. Defaults to scPred's own default,
#'                   "svmRadial". See SCPRED_MODELS in tool_registry.R.
#' Outputs: list(predictions, true_labels, confidence_scores, cell_ids,
#'                runtime_secs, peak_system_memory_mb)
run_scPred_function <- function(seurat_train, seurat_test, markers,
                                model = "svmRadial") {

  if (!requireNamespace("scPred", quietly = TRUE)) {
    stop("scPred package not available. Please install it first.")
  }
  library(scPred)
  library(Seurat)

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

  tryCatch({

    # --- Core annotation call --------------------------------------------
    # This paradigm has a training step, so the measurement window covers
    # feature-space construction, model fitting, and prediction. Marker-based
    # methods have no training step and are charged for inference only.
    tracked <- with_resource_tracking({
      # 1. Build the PCA feature space from the labelled training data
      seurat_train <- getFeatureSpace(seurat_train, "Ground_Truth_Celltype")

      # 2. Fit the classifier in that space
      seurat_train <- trainModel(seurat_train, model = model, allowParallel = TRUE)

      # 3. Project and predict the test cells
      scPredict(seurat_test, seurat_train)
    })

    seurat_test           <- tracked$value
    runtime_secs          <- tracked$runtime_secs
    peak_system_memory_mb <- tracked$peak_system_memory_mb

    # --- Extract and standardise -----------------------------------------
    cell_predictions <- seurat_test$scpred_prediction

    # scPred writes one scpred_<label> probability column per class; the
    # confidence is the winning class probability.
    prob_cols <- grep("scpred_", colnames(seurat_test@meta.data), value = TRUE)
    prob_cols <- setdiff(prob_cols, "scpred_prediction")

    if (length(prob_cols) > 0) {
      prob_matrix       <- seurat_test@meta.data[, prob_cols, drop = FALSE]
      confidence_scores <- apply(prob_matrix, 1, max, na.rm = TRUE)
    } else {
      warning("scPred returned no probability columns; reporting confidence as 1.0")
      confidence_scores <- rep(1.0, length(cell_predictions))
    }

    cell_predictions[is.na(cell_predictions)]   <- "Unknown"
    confidence_scores[is.na(confidence_scores)] <- 0

    list(
      predictions           = as.character(cell_predictions),
      true_labels           = seurat_test$Ground_Truth_Celltype,
      confidence_scores     = confidence_scores,
      cell_ids              = colnames(seurat_test),
      runtime_secs          = runtime_secs,
      peak_system_memory_mb = peak_system_memory_mb
    )

  }, error = function(e) {
    warning(paste0("scPred (", model, ") error: ", e$message))
    default_return()
  })
}

# Backward-compatible alias; the registry uses the _function form.
run_scPred <- run_scPred_function
