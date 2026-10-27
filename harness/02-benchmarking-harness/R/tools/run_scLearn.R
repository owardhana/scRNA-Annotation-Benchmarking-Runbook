# run_scLearn.R
#################################################
# EXEMPLAR: classical-ML tool (R side; non-neural, grouped as classical ML in the paper)
#
# scLearn learns a discriminative metric over selected genes and assigns cells
# by voting, with an automatic rejection threshold — so unlike most classifiers
# it can decline to call a cell.
#
# Two things make this a useful exemplar beyond the paradigm:
#   - it wants RAW COUNTS, not the log-normalised layer most tools take, which
#     is why the contract passes the whole Seurat object rather than a matrix
#   - it returns predictions keyed by its own cell IDs, in its own order and
#     potentially for a subset of cells, so the wrapper has to realign against
#     the test object before returning
#################################################

#' scLearn cell-type annotation
#'
#' Inputs:
#'   - seurat_train: labelled training Seurat object (reference for model learning)
#'   - seurat_test:  test Seurat object to annotate
#'   - markers:      FindAllMarkers() data frame (unused; scLearn selects its
#'                   own features with M3Drop)
#' Outputs: list(predictions, true_labels, confidence_scores, cell_ids,
#'                runtime_secs, peak_system_memory_mb)
run_scLearn_function <- function(seurat_train, seurat_test, markers) {

  for (pkg in c("scLearn", "SingleCellExperiment", "M3Drop")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(pkg, " package not available. Please install it first",
           if (pkg == "scLearn") ": devtools::install_github('bm2-lab/scLearn')" else ".")
    }
    library(pkg, character.only = TRUE)
  }
  library(Seurat)

  default_return <- function() {
    n_test_cells <- ncol(seurat_test)
    list(
      predictions           = as.character(rep("Unknown", n_test_cells)),
      true_labels           = as.character(seurat_test$Ground_Truth_Celltype),
      confidence_scores     = as.numeric(rep(0, n_test_cells)),
      cell_ids              = as.character(colnames(seurat_test)),
      runtime_secs          = NA_real_,
      peak_system_memory_mb = NA_real_
    )
  }

  for (obj in list(list(seurat_train, "training"), list(seurat_test, "test"))) {
    if (!"Ground_Truth_Celltype" %in% colnames(obj[[1]]@meta.data)) {
      warning("Ground_Truth_Celltype not found in ", obj[[2]], " data")
      return(default_return())
    }
  }

  tryCatch({

    # --- Input adaptation ------------------------------------------------
    # scLearn models counts directly, so pull the counts layer rather than the
    # log-normalised one the marker-based tools use.
    get_counts <- function(seurat_obj) {
      counts <- tryCatch(
        GetAssayData(seurat_obj, assay = "RNA", layer = "counts"),
        error = function(e) GetAssayData(seurat_obj, assay = "RNA", layer = "data")
      )
      if (inherits(counts, "sparseMatrix")) counts <- as.matrix(counts)
      counts
    }

    rawcounts_train <- get_counts(seurat_train)
    rawcounts_test  <- get_counts(seurat_test)

    refe_ann        <- as.character(seurat_train$Ground_Truth_Celltype)
    names(refe_ann) <- colnames(seurat_train)

    # scLearn's own QC drops cells, so require a buffer above its minimum or
    # a type can vanish mid-pipeline and break training.
    min_cells_per_type <- 10
    cell_type_counts   <- table(refe_ann)
    valid_cell_types   <- names(cell_type_counts)[cell_type_counts >= min_cells_per_type]

    if (length(valid_cell_types) < 2) {
      warning("Insufficient cell types with enough cells for scLearn training")
      return(default_return())
    }

    valid_cells              <- names(refe_ann)[refe_ann %in% valid_cell_types]
    rawcounts_train_filtered <- rawcounts_train[, valid_cells]
    refe_ann_filtered        <- refe_ann[valid_cells]

    cat(sprintf("Training with %d cells across %d cell types\n",
                length(refe_ann_filtered), length(valid_cell_types)))

    # --- Core annotation pipeline ----------------------------------------
    run_scLearn_pipeline <- function() {
      cat("Performing quality control on training data...\n")
      data_qc <- Cell_qc(rawcounts_train_filtered, refe_ann_filtered, species = "Hs",
                         gene_low = 50, umi_low = 50)

      # Re-check after QC: types that were just above threshold may now be below.
      post_qc_counts  <- table(data_qc$sample_information_cellType)
      surviving_types <- names(post_qc_counts)[post_qc_counts >= min_cells_per_type]
      if (length(surviving_types) < 2) {
        warning("Too few cell types remaining after Cell_qc(); returning Unknown")
        return(NULL)
      }

      keep_mask <- data_qc$sample_information_cellType %in% surviving_types
      data_qc$expression_profile          <- data_qc$expression_profile[, keep_mask]
      data_qc$sample_information_cellType <- data_qc$sample_information_cellType[keep_mask]

      cat("Filtering rare cell types...\n")
      data_type_filtered <- Cell_type_filter(
        data_qc$expression_profile,
        data_qc$sample_information_cellType,
        min_cell_number = min_cells_per_type
      )

      if (length(unique(data_type_filtered$sample_information_cellType)) < 2) {
        warning("Too few cell types remaining after Cell_type_filter(); returning Unknown")
        return(NULL)
      }

      cat("Selecting high variance genes...\n")
      high_varGene_names <- tryCatch({
        Feature_selection_M3Drop(data_type_filtered$expression_profile)
      }, error = function(e) {
        warning("M3Drop feature selection failed, using most variable genes")
        gene_vars <- apply(data_type_filtered$expression_profile, 1, var)
        names(gene_vars)[order(gene_vars, decreasing = TRUE)[1:min(2000, length(gene_vars))]]
      })
      cat(sprintf("Selected %d high variance genes\n", length(high_varGene_names)))

      cat("Training scLearn model...\n")
      scLearn_model_learning_result <- scLearn_model_learning(
        high_varGene_names,
        data_type_filtered$expression_profile,
        data_type_filtered$sample_information_cellType,
        bootstrap_times = 10
      )

      cat("Performing quality control on test data...\n")
      data_qc_query <- Cell_qc(rawcounts_test, species = "Hs", gene_low = 50, umi_low = 50)

      cat("Performing cell type assignment...\n")
      scLearn_cell_assignment(
        scLearn_model_learning_result,
        data_qc_query$expression_profile,
        diff          = 0.05,
        threshold_use = TRUE,
        vote_rate     = 0.6
      )
    }

    tracked <- with_resource_tracking({
      tryCatch(run_scLearn_pipeline(), error = function(e) {
        warning(e$message)
        NULL
      })
    })

    scLearn_predict_result <- tracked$value
    runtime_secs           <- tracked$runtime_secs
    peak_system_memory_mb  <- tracked$peak_system_memory_mb

    if (is.null(scLearn_predict_result)) {
      warning("scLearn pipeline returned NULL - too few cell types after QC/filtering")
      return(default_return())
    }

    # --- Realign to the test object --------------------------------------
    # scLearn's QC can drop test cells, and it returns rows in its own order,
    # so match on barcode. Cells it never reported are dropped from all four
    # vectors together: the harness requires predictions, true_labels and
    # cell_ids to stay the same length and in the same order.
    truth_by_barcode        <- seurat_test@meta.data$Ground_Truth_Celltype
    names(truth_by_barcode) <- rownames(seurat_test@meta.data)

    matched_index <- match(scLearn_predict_result$Query_cell_id, names(truth_by_barcode))
    keep          <- !is.na(matched_index)

    ground_truth <- truth_by_barcode[matched_index[keep]]
    predicted    <- scLearn_predict_result$Predict_cell_type[keep]
    cell_ids     <- scLearn_predict_result$Query_cell_id[keep]

    # scLearn's rejection option reports lowercase "unassigned"; the framework
    # sentinel is capital-U "Unknown" (see docs/tool-wrapper-contract.md).
    predicted[predicted == "unassigned"] <- "Unknown"
    predicted[is.na(predicted)]          <- "Unknown"

    # scLearn exposes no per-cell score. The contract still wants a number, so
    # report a flat value and zero for declined cells - not a probability, and
    # not comparable across tools.
    confidence_scores <- rep(0.8, length(predicted))
    confidence_scores[predicted == "Unknown"] <- 0

    list(
      predictions           = as.character(predicted),
      true_labels           = as.character(ground_truth),
      confidence_scores     = as.numeric(confidence_scores),
      cell_ids              = as.character(cell_ids),
      runtime_secs          = runtime_secs,
      peak_system_memory_mb = peak_system_memory_mb
    )

  }, error = function(e) {
    warning("scLearn error: ", e$message)
    default_return()
  })
}

# Backward-compatible alias; the registry uses the _function form.
run_scLearn <- run_scLearn_function
