# preprocess_dataset.R
#################################################
# Stage 1 of data preparation: turn one published dataset into the canonical
# on-disk form the rest of the pipeline consumes.
#
# The study ran 44 candidate datasets through this stage. In the working
# repository that was 44 near-identical scripts differing in two places: how
# the dataset is fetched, and which metadata column holds the cell type. Those
# two facts are data, not code, so they live in datasets.csv and this one
# script is driven by it.
#
# WHAT IT WRITES, per dataset, in the canonical 10x-style quartet:
#
#   <study>/original/   counts.mtx, genes.tsv, barcodes.tsv, metadata.tsv
#   <study>/for_use/    the same four files, after zone downsampling
#
# The `for_use` copy is what the benchmark actually runs on. `original` is
# retained so the downsampling can be inspected or redone.
#
# WHAT IT DELIBERATELY DOES NOT DO: no QC, no normalisation, no feature
# selection, no scaling. Those belong to stage 2 (../loaders/load_seurat.R and
# its Scanpy twin), which applies one identical pipeline to every dataset. Two
# stages, not one, is what makes "the same preprocessing for everything"
# checkable rather than a claim repeated in 44 files.
#
# USAGE
#   Rscript preprocess_dataset.R Darmanis-Brain-2015
#   Rscript preprocess_dataset.R --all
#   Rscript preprocess_dataset.R --phase 2       # just the reported panel
#
# Datasets not distributed through the Bioconductor scRNAseq package have to be
# downloaded by hand first; see the `source` and `source_ref` columns of
# datasets.csv and the worked adapters in examples/.
#################################################

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(SingleCellExperiment)
  library(Matrix)
  library(dplyr)
})

# Reproducibility: the downsampling below is a random stratified draw. Without
# a fixed seed the analysis-ready panels cannot be regenerated - a gap in the
# original scripts, which carried no set.seed() at all.
DEFAULT_SEED <- 123

# Cell-count tiers. These are not arbitrary: 500 / 3,000 / 15,000 are the three
# cell-count levels of the Phase 1 L9 design, so downsampling snaps each real
# dataset onto the same grid the synthetic scenarios were built on.
ZONE_TARGETS   <- c(small = 500, medium = 3000, large = 15000)
ZONE_BOUNDS    <- c(2000, 9000)   # < 2000 -> small; < 9000 -> medium; else large
ZONE_TOLERANCE <- 1.15            # within 15% of target, leave the dataset alone


#' Read the dataset configuration table
read_dataset_config <- function(config_path = "datasets.csv") {
  config <- read.csv(config_path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("study", "phase", "scenario", "source", "source_ref",
                "celltype_column", "gene_id_convention", "notes")
  missing_cols <- setdiff(required, colnames(config))
  if (length(missing_cols) > 0) {
    stop("datasets.csv is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  config
}


#' Load one dataset into a Seurat object
#'
#' The only genuinely per-dataset step. Everything downstream is uniform.
#'
#' Inputs:  - entry: one row of datasets.csv
#'          - data_dir: directory holding hand-downloaded inputs, for the
#'            sources that are not fetched programmatically
#' Outputs: a Seurat object whose meta.data still carries the dataset's own
#'          column names (including entry$celltype_column)
#'
#' The `csv`, `tsv`, `10x-mtx` and `seurat-rds` branches vary in the details of
#' file naming and orientation from dataset to dataset. Three real, complete
#' adapters are in examples/ - start from whichever matches your source format.
load_dataset <- function(entry, data_dir = ".") {
  source_type <- entry$source

  if (source_type == "scRNAseq") {
    # Bioconductor ExperimentHub. The snapshot date in source_ref pins the
    # version, so re-running fetches the same data rather than whatever is
    # current.
    if (!requireNamespace("scRNAseq", quietly = TRUE)) {
      stop("Package scRNAseq required for source type 'scRNAseq'")
    }
    # source_ref is <name>@<snapshot date>, optionally /<sub-dataset path>
    # (e.g. he-organs-2020@2023-12-21/skin, baron-pancreas-2016@2023-12-14/human).
    parts   <- strsplit(entry$source_ref, "@", fixed = TRUE)[[1]]
    version <- sub("/.*$", "", parts[2])
    subpath <- if (grepl("/", parts[2], fixed = TRUE)) sub("^[^/]*/", "", parts[2]) else NA
    sce <- scRNAseq::fetchDataset(parts[1], version, path = subpath, realize.assays = TRUE)

    count_mat <- assay(sce, "counts")
    meta      <- as.data.frame(colData(sce))

    # Some ExperimentHub datasets suffix gene symbols with a locus
    # ("ACTB__chr7"); strip it and collapse the duplicates that creates.
    rownames(count_mat) <- sub("__.*$", "", rownames(count_mat))
    if (anyDuplicated(rownames(count_mat))) {
      count_mat <- rowsum(count_mat, group = rownames(count_mat))
    }

    return(CreateSeuratObject(counts = count_mat, meta.data = meta))
  }

  if (source_type == "seurat-rds") {
    return(UpdateSeuratObject(readRDS(file.path(data_dir, entry$source_ref))))
  }

  if (source_type == "10x-mtx") {
    counts <- as(readMM(file.path(data_dir, "counts.mtx")), "CsparseMatrix")
    genes  <- read.table(file.path(data_dir, "genes.tsv"),    header = FALSE)$V1
    cells  <- read.table(file.path(data_dir, "barcodes.tsv"), header = FALSE)$V1
    rownames(counts) <- make.unique(as.character(genes))
    colnames(counts) <- as.character(cells)

    seurat_obj <- CreateSeuratObject(counts = counts)
    meta_path  <- file.path(data_dir, "metadata.tsv")
    if (file.exists(meta_path)) {
      seurat_obj <- AddMetaData(
        seurat_obj,
        read.table(meta_path, header = TRUE, sep = "\t", row.names = 1)
      )
    }
    return(seurat_obj)
  }

  stop("Source type '", source_type, "' has no generic loader. ",
       "The study's own per-dataset script for it is in ",
       "as-run/preprocessing/real-datasets/ (S5, S8, S9) or as-run/preprocessing/cross-platform/, ",
       "and the processed object is in the Zenodo data bundle under benchmarking/data/.")
}


#' Apply the dataset's gene-identifier convention
#'
#' Foundation models are pre-trained on human gene symbols, so every dataset is
#' brought onto that convention before the shared pipeline runs. Two cases need
#' more than a pass-through.
apply_gene_id_convention <- function(seurat_obj, convention) {
  if (convention == "symbol") {
    return(seurat_obj)
  }

  if (convention == "mouse-uppercase") {
    # Case folding, not orthology mapping. It resolves the large majority of
    # one-to-one mouse/human symbol pairs and leaves genuinely mouse-specific
    # genes unmatched, which is the intended, documented behaviour.
    counts <- GetAssayData(seurat_obj, layer = "counts")
    rownames(counts) <- toupper(rownames(counts))

    # Riken cDNA clone identifiers have no human counterpart and only add noise.
    counts <- counts[!grepl("RIK$", rownames(counts)), , drop = FALSE]

    return(CreateSeuratObject(counts = counts, meta.data = seurat_obj@meta.data))
  }

  if (convention == "ensembl-to-hgnc") {
    for (pkg in c("AnnotationHub", "AnnotationDbi")) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        stop("Package ", pkg, " required for 'ensembl-to-hgnc'")
      }
    }
    # Latest EnsDb for Homo sapiens. Pin the record ID here if you need the
    # mapping to be reproducible across AnnotationHub snapshots.
    hub      <- AnnotationHub::AnnotationHub()
    ensdb_id <- tail(names(AnnotationHub::query(hub, c("EnsDb", "Homo sapiens"))), 1)
    ensdb    <- hub[[ensdb_id]]
    message("Using AnnotationHub record ", ensdb_id)

    counts     <- GetAssayData(seurat_obj, layer = "counts")
    ensembl_id <- sub("\\..*$", "", rownames(counts))   # drop version suffix
    symbols    <- AnnotationDbi::mapIds(
      ensdb, keys = ensembl_id, keytype = "GENEID", column = "SYMBOL"
    )

    mapped <- !is.na(symbols) & nzchar(symbols)
    counts <- counts[mapped, , drop = FALSE]
    rownames(counts) <- make.unique(as.character(symbols[mapped]))

    message("Ensembl -> HGNC: mapped ", sum(mapped), " of ", length(mapped), " genes")
    return(CreateSeuratObject(counts = counts, meta.data = seurat_obj@meta.data))
  }

  stop("Unknown gene_id_convention: ", convention)
}


#' Write the canonical four-file quartet
write_quartet <- function(seurat_obj, out_dir, celltype_column) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  # CsparseMatrix specifically: the R and Python readers disagree on other
  # sparse layouts, and a mismatch here surfaces much later as an empty matrix.
  counts <- as(GetAssayData(seurat_obj, layer = "counts"), "CsparseMatrix")
  writeMM(counts, file = file.path(out_dir, "counts.mtx"))

  write.table(rownames(seurat_obj), file.path(out_dir, "genes.tsv"),
              sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)
  write.table(colnames(seurat_obj), file.path(out_dir, "barcodes.tsv"),
              sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)

  # The label column is renamed to Ground_Truth_Celltype here, once. From this
  # point on nothing downstream needs to know what the source called it.
  metadata <- data.frame(
    barcode               = rownames(seurat_obj@meta.data),
    Ground_Truth_Celltype = seurat_obj@meta.data[[celltype_column]],
    stringsAsFactors      = FALSE
  )
  write.table(metadata, file.path(out_dir, "metadata.tsv"),
              sep = "\t", row.names = FALSE, quote = FALSE)
}


#' Downsample onto the nearest L9 cell-count tier
#'
#' Stratified by cell type, so the composition is preserved and rare types are
#' not wiped out. Datasets already within tolerance of their tier are returned
#' untouched.
downsample_to_zone <- function(seurat_obj, celltype_column, seed = DEFAULT_SEED,
                               target_override = NA_integer_) {
  n_cells <- ncol(seurat_obj)

  target <- if (!is.na(target_override)) {
    target_override
  } else if (n_cells < ZONE_BOUNDS[1]) {
    ZONE_TARGETS[["small"]]
  } else if (n_cells < ZONE_BOUNDS[2]) {
    ZONE_TARGETS[["medium"]]
  } else {
    ZONE_TARGETS[["large"]]
  }

  if (n_cells <= target * ZONE_TOLERANCE) {
    message("Size ", n_cells, " is within ", round((ZONE_TOLERANCE - 1) * 100),
            "% of target ", target, "; keeping all cells.")
    return(seurat_obj)
  }

  message("Size ", n_cells, " exceeds target ", target, "; downsampling.")
  set.seed(seed)

  sampled_barcodes <- seurat_obj@meta.data %>%
    tibble::rownames_to_column("tmp_rn") %>%
    group_by(across(all_of(celltype_column))) %>%
    slice_sample(prop = target / n_cells) %>%
    ungroup() %>%
    pull(tmp_rn)

  subset(seurat_obj, cells = sampled_barcodes)
}


#' Run stage 1 for a single dataset
preprocess_dataset <- function(entry, data_dir = ".", out_root = ".",
                               seed = DEFAULT_SEED) {
  message("\n=== ", entry$study, " ===")

  seurat_obj <- load_dataset(entry, data_dir = data_dir)
  seurat_obj <- apply_gene_id_convention(seurat_obj, entry$gene_id_convention)

  if (!entry$celltype_column %in% colnames(seurat_obj@meta.data)) {
    stop("Column '", entry$celltype_column, "' not found in ", entry$study,
         ". Available: ", paste(colnames(seurat_obj@meta.data), collapse = ", "))
  }

  # Cells with no label cannot be scored, and a tool that predicts one of them
  # correctly would be credited by accident.
  labelled <- !is.na(seurat_obj@meta.data[[entry$celltype_column]]) &
    nzchar(as.character(seurat_obj@meta.data[[entry$celltype_column]]))
  if (any(!labelled)) {
    message("Dropping ", sum(!labelled), " unlabelled cells")
    seurat_obj <- subset(seurat_obj, cells = colnames(seurat_obj)[labelled])
  }

  study_dir <- file.path(out_root, entry$study)
  write_quartet(seurat_obj, file.path(study_dir, "original"), entry$celltype_column)

  # Tabula Muris S5 is pinned to the medium tier rather than the tier its raw
  # size would select; see Methods and the notes column of datasets.csv.
  target_override <- if (grepl("^Tabula Muris-FACS-3k$", entry$study)) 3000L else NA_integer_
  seurat_subset <- downsample_to_zone(seurat_obj, entry$celltype_column,
                                      seed = seed, target_override = target_override)
  write_quartet(seurat_subset, file.path(study_dir, "for_use"), entry$celltype_column)

  message("original: ", ncol(seurat_obj), " cells | for_use: ", ncol(seurat_subset), " cells")
  print(table(seurat_subset@meta.data[[entry$celltype_column]]))

  invisible(study_dir)
}


#################################################
# Command line
#################################################
if (sys.nframe() == 0) {
  args   <- commandArgs(trailingOnly = TRUE)
  config <- read_dataset_config()

  selected <- if (length(args) == 0) {
    stop("Usage: Rscript preprocess_dataset.R <study> | --all | --phase <n>")
  } else if (args[1] == "--all") {
    config
  } else if (args[1] == "--phase") {
    # phase can be a list ("2,3"); match whole entries, so "--phase 2" does not
    # also pick up a hypothetical "12"
    in_phase <- vapply(strsplit(config$phase, ",", fixed = TRUE),
                       function(p) args[2] %in% trimws(p), logical(1))
    config[in_phase, ]
  } else {
    config[config$study == args[1], ]
  }

  if (nrow(selected) == 0) stop("No datasets matched: ", paste(args, collapse = " "))

  for (row_index in seq_len(nrow(selected))) {
    entry <- selected[row_index, ]
    tryCatch(
      preprocess_dataset(entry),
      error = function(e) message("FAILED ", entry$study, ": ", e$message)
    )
  }
}
