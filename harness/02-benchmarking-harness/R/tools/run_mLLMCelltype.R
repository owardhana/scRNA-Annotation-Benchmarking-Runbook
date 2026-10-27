# run_mLLMCelltype.R
#################################################
# EXEMPLAR: LLM-based tool
#
# mLLMCelltype sends each cluster's top marker genes to a language model and
# asks what cell type they indicate. It differs from every other paradigm here
# in three ways that the wrapper has to absorb:
#
#   1. It works at CLUSTER level, not cell level. The wrapper expands one
#      answer per cluster out to one prediction per cell.
#   2. It returns FREE TEXT, not a label from a fixed vocabulary. A model may
#      answer "cytotoxic T lymphocyte" where the ground truth says "CD8+ T
#      cell", or invent something that matches nothing at all. Raw strings are
#      therefore mapped onto the ground-truth vocabulary by
#      scoring/normalisation_pipeline.R before any metric is computed.
#   3. It needs a credential. Keys are read from the environment at call time
#      and never stored in this repository.
#
# Because of (2) the wrapper returns MORE than the standard contract: STRICT
# and LENIENT predictions, the kappa each yields, the gap between them, and a
# hallucination rate. The strict/lenient gap is itself a reported result - it
# measures how much of a model's apparent error is vocabulary rather than
# biology. See docs/tool-wrapper-contract.md, "extended returns".
#
# CONSOLIDATION NOTE: the study's working tree held nine wrapper files - seven
# single-model endpoints, none with a reported result, and two copies of the
# multi-model consensus arm - that differed by a model ID and an
# environment-variable name. Here `models` is an argument: pass one name for
# single-model annotation, several for consensus. tool_registry.R registers the
# reported consensus arm.
#################################################

#' mLLMCelltype cell-type annotation
#'
#' Inputs:
#'   - seurat_train: training Seurat object; supplies the celltype<->cluster map
#'   - seurat_test:  test Seurat object; must carry Ground_Truth_Cluster
#'   - markers:      FindAllMarkers() data frame
#'   - models:       one model ID for single-model annotation, or several for
#'                   multi-model consensus discussion
#'   - api_key_env:  environment variable(s) holding the credential(s)
#'   - tissue_name:  tissue hint passed to the model; "generic" withholds it
#' Outputs: the standard contract plus lenient_predictions, strict_kappa,
#'   lenient_kappa, kappa_gap, hallucination_rate, cluster_mapping_df
run_mLLMCelltype_function <- function(seurat_train, seurat_test, markers,
                                      models      = "gpt-5",
                                      api_key_env = "OPENAI_API_KEY",
                                      tissue_name = "generic") {

  for (pkg in c("mLLMCelltype", "Seurat")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(paste("Package", pkg, "not available. Please install it first."))
    }
  }
  library(mLLMCelltype)
  library(Seurat)

  markers <- prepare_markers(markers)

  default_return <- function() {
    n <- ncol(seurat_test)
    list(
      predictions           = rep("Unknown", n),
      lenient_predictions   = rep("Unknown", n),
      true_labels           = as.character(seurat_test$Ground_Truth_Celltype),
      confidence_scores     = rep(0, n),
      cell_ids              = colnames(seurat_test),
      strict_kappa          = NA_real_,
      lenient_kappa         = NA_real_,
      kappa_gap             = NA_real_,
      hallucination_rate    = NA_real_,
      runtime_secs          = NA_real_,
      peak_system_memory_mb = NA_real_
    )
  }

  # --- Credentials --------------------------------------------------------
  api_keys <- setNames(lapply(api_key_env, Sys.getenv), tolower(sub("_API_KEY$", "", api_key_env)))
  if (all(nchar(unlist(api_keys)) == 0)) {
    warning("mLLMCelltype: no API key found in ", paste(api_key_env, collapse = " / "))
    return(default_return())
  }

  # --- Cluster keying -----------------------------------------------------
  # markers$cluster is keyed differently depending on the execution mode: by
  # cell-type name when FindAllMarkers ran under Ground_Truth_Celltype
  # identities, by cluster ID when it ran under Ground_Truth_Cluster. Detect
  # which, and remap to cluster IDs if needed.
  if (!"Ground_Truth_Cluster" %in% colnames(seurat_test@meta.data)) {
    warning("mLLMCelltype: Ground_Truth_Cluster missing from seurat_test")
    return(default_return())
  }

  mapping_df <- NULL
  if ("Ground_Truth_Cluster" %in% colnames(seurat_train@meta.data)) {
    mapping_df <- unique(seurat_train@meta.data[, c("Ground_Truth_Celltype", "Ground_Truth_Cluster")])

    if (any(as.character(markers$cluster) %in% mapping_df$Ground_Truth_Celltype)) {
      celltype_to_cluster <- setNames(mapping_df$Ground_Truth_Cluster, mapping_df$Ground_Truth_Celltype)
      markers$cluster     <- celltype_to_cluster[as.character(markers$cluster)]

      unmapped <- sum(is.na(markers$cluster))
      if (unmapped > 0) {
        warning("mLLMCelltype: dropping ", unmapped, " marker rows with no cluster mapping")
        markers <- markers[!is.na(markers$cluster), ]
      }
      if (nrow(markers) == 0) {
        warning("mLLMCelltype: no markers remain after cluster remapping")
        return(default_return())
      }
    }
  }

  # --- Core annotation call ----------------------------------------------
  # Only the API call is measured. Runtime here is dominated by network
  # latency and provider queueing, so it is not comparable to the compute-bound
  # runtimes of the local tools - see docs/resource-profiling.md.
  multi_model <- length(models) > 1

  tracked <- with_resource_tracking({
    tryCatch({
      if (multi_model) {
        cat("mLLMCelltype: consensus annotation across", length(models), "models\n")
        mLLMCelltype::interactive_consensus_annotation(
          input                 = markers,
          tissue_name           = tissue_name,
          top_gene_count        = 20,
          models                = models,
          api_keys              = api_keys,
          controversy_threshold = 0.7,
          entropy_threshold     = 1.0,
          max_discussion_rounds = 3,
          consensus_check_model = NULL,
          force_rerun           = TRUE,
          use_cache             = FALSE
        )
      } else {
        cat("mLLMCelltype: single-model annotation with", models, "\n")
        mLLMCelltype::annotate_cell_types(
          input          = markers,
          tissue_name    = tissue_name,
          top_gene_count = 20,
          model          = models,
          api_key        = unlist(api_keys)[[1]]
        )
      }
    }, error = function(e) {
      warning("mLLMCelltype annotation failed: ", e$message)
      NULL
    })
  })

  annotation_results    <- tracked$value
  runtime_secs          <- tracked$runtime_secs
  peak_system_memory_mb <- tracked$peak_system_memory_mb

  if (is.null(annotation_results)) {
    warning("mLLMCelltype: annotation failed - returning default predictions")
    return(default_return())
  }

  # --- Cluster-level raw answers ------------------------------------------
  # Consensus returns a named list; single-model returns an unnamed vector
  # ordered by the sorted unique cluster IDs.
  unique_cluster_ids <- sort(unique(markers$cluster))

  cluster_to_celltype <- if (multi_model) {
    annotation_results$final_annotations
  } else {
    setNames(annotation_results, as.character(unique_cluster_ids))
  }

  llm_raw_df <- data.frame(
    cluster          = as.character(unique_cluster_ids),
    raw_pred         = as.character(cluster_to_celltype[as.character(unique_cluster_ids)]),
    stringsAsFactors = FALSE
  )

  test_cell_df <- data.frame(
    cell_id          = colnames(seurat_test),
    cluster          = as.character(seurat_test$Ground_Truth_Cluster),
    true_label       = as.character(seurat_test$Ground_Truth_Celltype),
    stringsAsFactors = FALSE
  )

  gt_labels <- unique(as.character(seurat_test$Ground_Truth_Celltype))

  # --- Map free text onto the ground-truth vocabulary ---------------------
  # Five-stage cascade: exact -> synonym -> Jaro-Winkler -> Cell Ontology
  # Wu-Palmer -> unmappable. STRICT keeps only exact ontology matches; LENIENT
  # also accepts near relatives. See scoring/normalisation_pipeline.R.
  cl_onto <- tryCatch(load_cell_ontology(), error = function(e) {
    warning("Could not load Cell Ontology: ", e$message)
    NULL
  })

  norm_df <- tryCatch(
    run_normalisation_pipeline(llm_raw_df, gt_labels, cl_onto),
    error = function(e) {
      warning("Normalisation pipeline failed: ", e$message)
      NULL
    }
  )
  if (is.null(norm_df)) return(default_return())

  cat(sprintf("\n=== [mLLMCelltype] Prediction mapping (%d clusters) ===\n", nrow(norm_df)))
  diagnostic_cols <- c("cluster", "raw_pred", "normalised_pred", "strict", "lenient", "mapping_method")
  if (!is.null(mapping_df)) {
    cluster_to_ct <- setNames(as.character(mapping_df$Ground_Truth_Celltype),
                              as.character(mapping_df$Ground_Truth_Cluster))
    norm_df_diag  <- norm_df
    norm_df_diag$gt_celltype <- cluster_to_ct[as.character(norm_df_diag$cluster)]
    print(norm_df_diag[, c("cluster", "gt_celltype", setdiff(diagnostic_cols, "cluster"))], row.names = FALSE)
  } else {
    print(norm_df[, diagnostic_cols], row.names = FALSE)
  }
  cat(sprintf("strict Unknown: %d/%d  |  lenient Unknown: %d/%d\n\n",
              sum(norm_df$strict == "Unknown"), nrow(norm_df),
              sum(norm_df$lenient == "Unknown"), nrow(norm_df)))

  metrics_dual <- tryCatch(
    build_confusion_and_compute_metrics(test_cell_df, norm_df, gt_labels),
    error = function(e) {
      warning("build_confusion_and_compute_metrics failed: ", e$message)
      NULL
    }
  )

  # --- Expand cluster answers to cells ------------------------------------
  cell_strict  <- norm_df$strict[match(test_cell_df$cluster, norm_df$cluster)]
  cell_lenient <- norm_df$lenient[match(test_cell_df$cluster, norm_df$cluster)]
  cell_strict[is.na(cell_strict)]   <- "Unknown"
  cell_lenient[is.na(cell_lenient)] <- "Unknown"

  # Confidence: ontology similarity where the mapping produced one. For
  # consensus runs, fall back to inter-model agreement (normalised entropy);
  # for single-model runs there is nothing to fall back to, so use 0.5.
  wp_scores <- norm_df$wp_sim[match(test_cell_df$cluster, norm_df$cluster)]

  fallback_confidence <- if (multi_model) {
    consensus_details <- if (!is.null(annotation_results$initial_results)) {
      annotation_results$initial_results$consensus_results
    } else {
      warning("mLLMCelltype: initial_results missing - entropy confidence unavailable")
      NULL
    }
    vapply(test_cell_df$cluster, function(cluster_id) {
      if (!is.null(consensus_details) && cluster_id %in% names(consensus_details) &&
          !is.null(consensus_details[[cluster_id]]$entropy)) {
        1 - consensus_details[[cluster_id]]$entropy / log(length(models))
      } else {
        0.5
      }
    }, numeric(1))
  } else {
    rep(0.5, nrow(test_cell_df))
  }

  confidence_scores <- ifelse(is.na(wp_scores), fallback_confidence, as.numeric(wp_scores))

  # --- Per-cluster audit trail, written out by the harness ----------------
  cluster_gt_map <- unique(data.frame(
    cluster               = test_cell_df$cluster,
    ground_truth_celltype = test_cell_df$true_label,
    stringsAsFactors      = FALSE
  ))
  cluster_mapping_df <- merge(norm_df, cluster_gt_map, by = "cluster", all.x = TRUE)
  cluster_mapping_df <- cluster_mapping_df[, c("cluster", "ground_truth_celltype",
                                               "raw_pred", "normalised_pred",
                                               "strict", "lenient",
                                               "mapping_method", "wp_sim")]

  list(
    predictions           = cell_strict,
    lenient_predictions   = cell_lenient,
    true_labels           = as.character(seurat_test$Ground_Truth_Celltype),
    confidence_scores     = as.numeric(confidence_scores),
    cell_ids              = colnames(seurat_test),
    strict_kappa          = if (!is.null(metrics_dual)) metrics_dual$strict$kappa       else NA_real_,
    lenient_kappa         = if (!is.null(metrics_dual)) metrics_dual$lenient$kappa      else NA_real_,
    kappa_gap             = if (!is.null(metrics_dual)) metrics_dual$kappa_gap          else NA_real_,
    hallucination_rate    = if (!is.null(metrics_dual)) metrics_dual$hallucination_rate else NA_real_,
    runtime_secs          = runtime_secs,
    peak_system_memory_mb = peak_system_memory_mb,
    cluster_mapping_df    = cluster_mapping_df
  )
}

# Backward-compatible alias; the registry uses the _function form.
run_mLLMCelltype <- run_mLLMCelltype_function
