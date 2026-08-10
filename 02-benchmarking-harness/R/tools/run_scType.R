# run_scType.R
#################################################
# EXEMPLAR: marker-based tool run against two different marker sources
#
# scType scores each cluster against a marker signature table and takes the
# best-scoring cell type. What it scores against is a parameter:
#
#   marker_source = "de"        markers derived from the data itself
#                               (FindAllMarkers on the training split), filtered
#                               by the study-wide prepare_markers() rule.
#                               Phases 1-3.
#
#   marker_source = "database"  markers looked up in CellMarker2.0 for the
#                               dataset's tissue. The tool never sees the data's
#                               own differential expression. Phase 4.
#
# The study originally kept these as two files in two directories that differed
# by three lines. They are one wrapper here, registered twice (see
# tool_registry.R). The distinction is not cosmetic: database markers arrive
# with SYNTHETIC statistics - avg_log2FC = 1.0, p_val_adj = 0, pct.1 = 1.0 -
# because a curated database has no effect sizes to report. Applying the
# statistical filter to them would be meaningless, so it is skipped.
#
# NOTE: scType is distributed as loose R files rather than a package, so this
# wrapper source()s them at call time. Set SCTYPE_SOURCE_DIR to a directory of
# local copies to run offline and pin the version.
#################################################

#' scType cell-type annotation
#'
#' Inputs:
#'   - seurat_train:  training Seurat object (unused; interface consistency only)
#'   - seurat_test:   test Seurat object to annotate
#'   - markers:       marker table - FindAllMarkers() output, or CellMarker2.0
#'                    rows shaped the same way
#'   - marker_source: "de" or "database"; see above
#' Outputs: list(predictions, true_labels, confidence_scores, cell_ids,
#'                runtime_secs, peak_system_memory_mb)
run_scType_function <- function(seurat_train, seurat_test, markers,
                                marker_source = c("de", "database")) {

  marker_source <- match.arg(marker_source)

  for (pkg in c("dplyr", "Seurat", "HGNChelper", "openxlsx")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("Package ", pkg, " required for scType. Install it with install.packages('", pkg, "')")
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

  # --- Load scType's source files ----------------------------------------
  sctype_base <- Sys.getenv(
    "SCTYPE_SOURCE_DIR",
    "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R"
  )
  tryCatch({
    source(file.path(sctype_base, "gene_sets_prepare.R"))
    source(file.path(sctype_base, "sctype_score_.R"))
    source(file.path(sctype_base, "sctype_wrapper.R"))
  }, error = function(e) {
    stop(paste("Failed to load scType source files:", e$message))
  })

  required_functions <- c("gene_sets_prepare", "sctype_score", "run_sctype")
  missing_functions  <- required_functions[!vapply(required_functions, exists, logical(1))]
  if (length(missing_functions) > 0) {
    stop(paste("Missing scType functions:", paste(missing_functions, collapse = ", ")))
  }

  # --- Marker adaptation --------------------------------------------------
  # The single line that separates the two arms of this tool.
  filtered_markers <- if (marker_source == "de") {
    prepare_markers(markers)     # harness-wide filter: positive, pct.1 >= 0.10, top 20
  } else {
    markers                      # CellMarker2.0 rows pass through unfiltered
  }

  #' Reshape a marker table into scType's spreadsheet format
  #'
  #' scType reads its signatures from an .xlsx with five fixed columns, so the
  #' wrapper writes a temporary one per call.
  convert_to_sctype_excel <- function(filtered_markers) {
    if (nrow(filtered_markers) == 0) {
      warning("No markers found after filtering")
      return(NULL)
    }

    cell_types <- unique(filtered_markers$cluster)
    cat("Building scType signatures for", length(cell_types), "cell types\n")

    sctype_data <- data.frame(
      tissueType      = character(),
      cellName        = character(),
      geneSymbolmore1 = character(),   # positive markers
      geneSymbolmore2 = character(),   # negative markers
      shortName       = character(),
      stringsAsFactors = FALSE
    )

    for (ct in cell_types) {
      pos_markers <- filtered_markers$gene[filtered_markers$cluster == ct]
      if (length(pos_markers) == 0) next

      # Negative markers: strongly-expressed markers of the OTHER types.
      # With database markers every avg_log2FC is the synthetic 1.0, so this
      # yields nothing and scType runs on positive signatures alone.
      neg_markers <- filtered_markers$gene[filtered_markers$cluster != ct &
                                             filtered_markers$avg_log2FC >= 2.0]

      sctype_data <- rbind(sctype_data, data.frame(
        tissueType      = "Custom",   # matches known_tissue_type below
        cellName        = as.character(ct),
        geneSymbolmore1 = paste(pos_markers, collapse = ","),
        geneSymbolmore2 = if (length(neg_markers) > 0) paste(head(neg_markers, 10), collapse = ",") else "",
        shortName       = as.character(ct),
        stringsAsFactors = FALSE
      ))
    }

    sctype_data <- sctype_data[sctype_data$geneSymbolmore1 != "", ]
    if (nrow(sctype_data) == 0) {
      warning("All cell types have empty marker lists")
      return(NULL)
    }
    sctype_data
  }

  sctype_data <- convert_to_sctype_excel(filtered_markers)
  if (is.null(sctype_data)) return(default_return())

  temp_file <- tempfile(pattern = "sctype_markers_", fileext = ".xlsx")
  on.exit(if (file.exists(temp_file)) file.remove(temp_file), add = TRUE)

  tryCatch({
    write.xlsx(sctype_data, temp_file, sheetName = "Sheet1", rowNames = FALSE)
    if (!file.exists(temp_file) || file.size(temp_file) == 0) {
      stop("Failed to create valid Excel file")
    }
    cat("Wrote", nrow(sctype_data), "signatures to", temp_file, "\n")
  }, error = function(e) {
    stop(paste("Failed to create marker file:", e$message))
  })

  # scType scores scaled expression, and takes cluster assignments from
  # seurat_clusters. Ground truth supplies the clustering here, so what is
  # being measured is signature scoring, not the tool's clustering.
  if (!"scale.data" %in% names(seurat_test@assays$RNA@layers) ||
      length(seurat_test@assays$RNA@layers$scale.data) == 0) {
    seurat_test <- ScaleData(seurat_test, features = rownames(seurat_test), verbose = FALSE)
  }
  seurat_test@meta.data[["seurat_clusters"]] <- as.factor(seurat_test@meta.data[["Ground_Truth_Celltype"]])

  # --- Core annotation call ----------------------------------------------
  tracked <- with_resource_tracking({
    tryCatch(
      suppressMessages(suppressWarnings(
        run_sctype(seurat_test,
                   assay              = "RNA",
                   scaled             = TRUE,
                   known_tissue_type  = "Custom",
                   custom_marker_file = temp_file,
                   name               = "sctype_classification")
      )),
      error = function(e) {
        cat("scType wrapper failed:", e$message, "\n")
        NULL
      }
    )
  })

  results               <- tracked$value
  runtime_secs          <- tracked$runtime_secs
  peak_system_memory_mb <- tracked$peak_system_memory_mb

  # --- Fallback: score signatures directly --------------------------------
  # The official wrapper is brittle on non-standard gene symbols (notably the
  # synthetic identifiers Splatter emits). Falling back to sctype_score() keeps
  # simulated datasets measurable. Note this path applies its own marker filter
  # rather than prepare_markers().
  if (is.null(results)) {
    cat("scType wrapper failed; attempting direct sctype_score fallback\n")

    fallback <- tryCatch({
      fallback_markers <- markers[markers$avg_log2FC >= 0.5 &
                                    markers$p_val_adj < 0.05 &
                                    markers$pct.1 >= 0.15, ] %>%
        dplyr::group_by(cluster) %>%
        dplyr::arrange(desc(avg_log2FC)) %>%
        dplyr::slice_head(n = 20) %>%
        dplyr::ungroup()

      gs_positive <- split(as.character(fallback_markers$gene), fallback_markers$cluster)
      gs_positive <- gs_positive[lengths(gs_positive) > 0]
      cat("Created gene sets for", length(gs_positive), "cell types\n")

      # gene_names_to_uppercase = FALSE: case-folding would break the synthetic
      # gene identifiers this path exists to handle.
      scored <- with_resource_tracking({
        sctype_score(
          scRNAseqData            = as.matrix(GetAssayData(seurat_test, assay = "RNA", layer = "scale.data")),
          scaled                  = TRUE,
          gs                      = gs_positive,
          gene_names_to_uppercase = FALSE
        )
      })

      es.max <- scored$value                     # rows = cell types, cols = cells
      align  <- match(colnames(seurat_test), colnames(es.max))

      list(
        predictions           = rownames(es.max)[apply(es.max, 2, which.max)][align],
        confidence_scores     = apply(es.max, 2, max)[align],
        runtime_secs          = scored$runtime_secs,
        peak_system_memory_mb = scored$peak_system_memory_mb
      )
    }, error = function(e) {
      cat("Fallback approach also failed:", e$message, "\n")
      NULL
    })

    if (is.null(fallback)) {
      warning("Both scType wrapper and fallback approach failed")
      return(default_return(runtime_secs, peak_system_memory_mb))
    }

    cell_predictions      <- fallback$predictions
    confidence_scores     <- fallback$confidence_scores
    runtime_secs          <- fallback$runtime_secs
    peak_system_memory_mb <- fallback$peak_system_memory_mb

  } else {
    # --- Extract from the wrapper's output --------------------------------
    cell_predictions <- if ("sctype_classification" %in% colnames(results@meta.data)) {
      results@meta.data$sctype_classification
    } else {
      warning("Could not find scType predictions in results")
      rep("Unknown", ncol(results))
    }

    confidence_scores <- if ("sctype_scores" %in% colnames(results@meta.data)) {
      results@meta.data$sctype_scores
    } else {
      ifelse(cell_predictions == "Unknown", 0, 1)
    }
  }

  # --- Standardise --------------------------------------------------------
  cell_predictions[is.na(cell_predictions) | cell_predictions == ""] <- "Unknown"

  confidence_scores <- as.numeric(confidence_scores)
  confidence_scores[is.na(confidence_scores)] <- 0

  if (length(cell_predictions) != ncol(seurat_test)) {
    warning(sprintf("Prediction length mismatch: expected %d, got %d",
                    ncol(seurat_test), length(cell_predictions)))
    return(default_return(runtime_secs, peak_system_memory_mb))
  }

  cat("scType predictions summary:\n")
  print(table(cell_predictions, useNA = "ifany"))

  list(
    predictions           = as.character(cell_predictions),
    true_labels           = seurat_test$Ground_Truth_Celltype,
    confidence_scores     = confidence_scores,
    cell_ids              = colnames(seurat_test),
    runtime_secs          = runtime_secs,
    peak_system_memory_mb = peak_system_memory_mb
  )
}

# Backward-compatible aliases; the registry uses the _function form with
# marker_source set per entry.
run_scType          <- run_scType_function
run_scType_database <- function(seurat_train, seurat_test, markers) {
  run_scType_function(seurat_train, seurat_test, markers, marker_source = "database")
}
