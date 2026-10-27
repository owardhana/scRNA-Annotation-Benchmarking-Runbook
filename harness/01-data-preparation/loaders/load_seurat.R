# load_seurat.R
# ============================================================================
# Stage 2 of data preparation, R side: turn the canonical four-file quartet
# into analysis-ready Seurat objects.
#
# This is the twin of loaders/load_anndata.py and is written to mirror it step
# for step. R tools receive a Seurat object and Python tools an AnnData object;
# unless both are built identically, a score difference between an R tool and a
# Python tool could be the preprocessing rather than the method. The Scanpy
# equivalent of each step is named in the comments below.
#
# The pipeline, in the order it must run:
#   1. filter cells   nFeature_RNA >= 200          (sc.pp.filter_cells)
#   2. filter genes   detected in >= 3 cells       (sc.pp.filter_genes)
#                     -- AFTER cell filtering, since genes are judged on the
#                        surviving cells
#   3. FindVariableFeatures  vst, 2000 features    (highly_variable_genes)
#   4. NormalizeData  LogNormalize, scale 10,000   (normalize_total + log1p)
#   5. subset to the 2,000 variable features
#   6. ScaleData      scale.max = 10               (sc.pp.scale)
#   7. RunPCA         50 components                (sc.tl.pca)
#
# Reads:  <study>/{original,for_use}/{counts.mtx,genes.tsv,barcodes.tsv,metadata.tsv}
#         <scenario>/rep<m>/{counts.mtx,genes.txt,cells.txt,metadata.csv}   (L9_Generator.R)
# Writes: <study>/{original,for_use}/<study>_<folder>_subset.rds
#         <scenario>/rep<m>/<scenario>_rep<m>_seurat_subset.rds
#
# These are the object names the study used (and the names in the Zenodo data
# bundle): e.g. Darmanis-Brain-2015_for_use_subset.rds and S1_rep1_seurat_subset.rds.
# For Phase 1, point SC_BENCH_STUDY_DIR at the generator's l9_benchmark_sets/.
#
# Stage 1 (../preprocessing/preprocess_dataset.R) produced those inputs. Every
# per-dataset quirk lives there; this script is uniform by design, which is
# what makes "the same preprocessing for every dataset" checkable.
#
# NOT INCLUDED: the working version also produced a full-gene variant, scaling
# and running PCA over all genes rather than the 2,000 variable features. No
# reported analysis used it.
# ============================================================================

library(Seurat)
library(Matrix)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory holding the per-study folders. Override with SC_BENCH_STUDY_DIR.
base_dir <- Sys.getenv("SC_BENCH_STUDY_DIR", ".")

# ============================================================================
# STRICT DIRECTORY DISCOVERY
# ============================================================================
study_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
target_dirs <- c()
for (study in study_dirs) {
  for (subfolder in c("for_use", "original")) {
    expected_path <- file.path(study, subfolder)
    if (dir.exists(expected_path)) {
      target_dirs <- c(target_dirs, expected_path)
    }
  }
  # L9 synthetic layout: <scenario>/rep<m>/ holding the generator's counts.mtx
  reps <- list.dirs(study, full.names = TRUE, recursive = FALSE)
  reps <- reps[grepl("^rep[0-9]+$", basename(reps)) & file.exists(file.path(reps, "counts.mtx"))]
  target_dirs <- c(target_dirs, reps)
}

target_dirs <- sort(target_dirs)

cat("\n")
cat(rep("=", 70), "\n", sep = "")
cat("  Data Loading to Seurat (Subset Mode - Bulletproof)\n")
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("Base directory : %s\n", base_dir))
cat(sprintf("Datasets found : %d\n\n", length(target_dirs)))

# ============================================================================
# PROCESSING LOOP
# ============================================================================

for (target_dir in target_dirs) {
  
  # Extract names based on the strict folder structure
  parts       <- strsplit(target_dir, .Platform$file.sep)[[1]]
  folder_type <- tail(parts, 1)      # "for_use", "original", or "rep<m>" (L9)
  study_name  <- tail(parts, 2)[1]   # e.g., "MacParland-Liver-2018" or "S1"
  sim_name    <- paste(study_name, folder_type, sep = "_")
  is_l9       <- grepl("^rep[0-9]+$", folder_type)
  
  cat(rep("=", 70), "\n", sep = "")
  cat("Processing:", sim_name, "\n")
  cat("Path:      ", target_dir, "\n")
  
  # ------------------------------------------------------------------------
  # "RESUME" CHECK: Skip if already processed
  # ------------------------------------------------------------------------
  subset_file <- file.path(target_dir, paste0(sim_name, if (is_l9) "_seurat_subset.rds" else "_subset.rds"))

  if (file.exists(subset_file)) {
    cat(sprintf("  -> %s already exists. Skipping...\n", basename(subset_file)))
    next
  }
  
  cat(rep("=", 70), "\n", sep = "")
  
  # ------------------------------------------------------------------------
  # BEGIN TRYCATCH BLOCK
  # ------------------------------------------------------------------------
  tryCatch({
    
    # --- 1. LOAD DATA -------------------------------------------------------
    counts_file <- file.path(target_dir, "counts.mtx")
    if (!file.exists(counts_file)) {
      stop("Missing counts.mtx.")
    }
    
    counts <- readMM(counts_file)
    
    # Force the matrix into Seurat's preferred format BEFORE assigning names
    counts <- as(counts, "CsparseMatrix")
    
    # Load the standard biological outputs (.tsv)
    if (file.exists(file.path(target_dir, "genes.tsv"))) {
      genes    <- read.table(file.path(target_dir, "genes.tsv"), sep="\t", header=FALSE, stringsAsFactors=FALSE)$V1
      cells    <- read.table(file.path(target_dir, "barcodes.tsv"), sep="\t", header=FALSE, stringsAsFactors=FALSE)$V1
      metadata <- read.table(file.path(target_dir, "metadata.tsv"), sep="\t", header=TRUE, stringsAsFactors=FALSE)
      
      # THE UNDERSCORE FIX: Force standard Seurat formatting to prevent "No Cell Overlap"
      clean_barcodes <- gsub("_", "-", as.character(cells))
      clean_barcodes <- make.unique(clean_barcodes)
      
      if("barcode" %in% colnames(metadata)) {
        metadata$barcode <- clean_barcodes
      }
      rownames(metadata) <- clean_barcodes
      colnames(counts)   <- clean_barcodes
      
    } else {
      # Fallback for raw Splatter text files (just in case)
      genes    <- readLines(file.path(target_dir, "genes.txt"))
      cells    <- readLines(file.path(target_dir, "cells.txt"))
      metadata <- read.csv(file.path(target_dir, "metadata.csv"), row.names = 1)
      colnames(counts) <- as.character(cells)
    }
    
    rownames(counts) <- make.unique(as.character(genes))
    
    # Standardize Ground Truth Metadata
    if (!"Ground_Truth_Celltype" %in% colnames(metadata)) {
      if ("cell_type" %in% colnames(metadata)) {
        metadata$Ground_Truth_Celltype <- metadata$cell_type
      } else if ("cell_ontology_class" %in% colnames(metadata)) {
        metadata$Ground_Truth_Celltype <- metadata$cell_ontology_class
      } else {
        stop("Could not find a valid cell type column in metadata.")
      }
    }
    
    unique_types                   <- sort(unique(metadata$Ground_Truth_Celltype))
    type_to_cluster                <- setNames(seq_along(unique_types), unique_types)
    metadata$Ground_Truth_Cluster  <- type_to_cluster[metadata$Ground_Truth_Celltype]
    
    # --- 2. GLOBAL PREPROCESSING --------------------------------------------
    seurat <- CreateSeuratObject(
      counts    = counts,
      meta.data = metadata,
      project   = sim_name
    )
    
    seurat <- subset(seurat, subset = nFeature_RNA >= 200)
    
    counts_matrix <- GetAssayData(seurat, layer = "counts")
    genes_to_keep <- rownames(counts_matrix)[rowSums(counts_matrix > 0) >= 3]
    seurat        <- subset(seurat, features = genes_to_keep)
    
    seurat <- FindVariableFeatures(
      seurat,
      selection.method = "vst",
      nfeatures        = 2000,
      verbose          = FALSE
    )
    
    seurat <- NormalizeData(
      seurat,
      normalization.method = "LogNormalize",
      scale.factor         = 10000,
      verbose              = FALSE
    )
    
    cat(sprintf("  Cells after QC : %d\n", ncol(seurat)))
    cat(sprintf("  Genes after QC : %d\n", nrow(seurat)))
    
    # --- 3. FORK A: SUBSETTED (Standard Benchmark — HVGs only) -------------
    cat("  -> Creating Subsetted version (2000 HVGs)...\n")
    
    seurat_sub <- subset(seurat, features = VariableFeatures(seurat))
    seurat_sub <- ScaleData(seurat_sub, scale.max = 10, verbose = FALSE)
    seurat_sub <- RunPCA(seurat_sub, npcs = 50, verbose = FALSE)
    
    saveRDS(seurat_sub, subset_file)
    
    # --- 4. CLEAN UP --------------------------------------------------------
    rm(seurat, seurat_sub, counts, metadata, counts_matrix)
    gc(verbose = FALSE)
    
    cat(sprintf("  ✓ Saved: %s\n\n", basename(subset_file)))
    
  }, error = function(e) {
    # ------------------------------------------------------------------------
    # ERROR CATCHING
    # ------------------------------------------------------------------------
    message("\n  [!] ERROR processing ", sim_name, ": ", e$message, "\n")
    
    # Force garbage collection on failure to free up RAM before the next iteration
    gc(verbose = FALSE)
  })
}

cat("╔════════════════════════════════════════════════════════════════════╗\n")
cat("║  ✓ All real-world datasets processed                               ║\n")
cat("╚════════════════════════════════════════════════════════════════════╝\n")