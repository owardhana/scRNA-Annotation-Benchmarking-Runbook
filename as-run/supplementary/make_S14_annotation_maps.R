# ============================================================
# Supplementary Fig. 14: annotation maps for the Discussion's
# good and poor examples (V33 Discussion para 57).
#
# Per-cell predictions were never saved in Phase 2, so this script
# re-runs three tools on the original single 80/20 split
# (run_all_tools_single_split, benchmarking_helpers.R:632):
#   good: singleCellNet, CaSTLe   poor: scPred (svmRadial, default)
# on one easy and one hard real dataset:
#   S9 MacParland liver (_original), S3 Nowakowski cortex (_for_use)
# S4 Grun was the first choice for the hard row but its labels are the
# Bioconductor `sample` field (FACS sort gate / library), not cell types.
#
# The split seed (123 or 124) was never recorded, so it is chosen by
# reproducing the published kappa of the deterministic tools
# (singleCellNet seeds itself; CaSTLe's xgboost has no subsampling).
# scPred's caret resampling depends on RNG state, so it gets a looser
# tolerance and the figure labels every panel with the re-run kappa.
#
# Cells with an empty label (92 in Nowakowski) are dropped before the split,
# as in the original runs (test n = 559 reproduces only with them removed).
#
# UMAP: RunUMAP(dims = 1:30) with Seurat's default seed on the FULL object,
# the same call as SF2 (P5/Plots_Analysis.R:2591-2593), so the maps match
# SF2's tiles; the dropped cells are kept in the CSV as split = "excluded".
#
# Outputs
#   figures/_shared/tidy_annotation_maps.csv   one row per (dataset, cell)
#   results/_SI_generated/S14_annotation_maps/
#     kappa_check.csv, predicted_labels.csv, predictions_raw.rds,
#     sessionInfo.txt
# Run with the 4.5-arm64 Rscript (the miniforge R lacks packages).
# ============================================================

suppressPackageStartupMessages({
  library(Seurat); library(caret); library(SingleCellExperiment)
})

THESIS  <- "<THESIS_ROOT>"
BENCH   <- file.path(THESIS, "benchmarking")
P5      <- file.path(THESIS, "results/P5 Real Life Datasets V2")
OUT_DIR <- file.path(THESIS, "results/_SI_generated/S14_annotation_maps")
TIDY    <- file.path(THESIS, "manuscript/publication_manuscript/figures/_shared/tidy_annotation_maps.csv")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Wrappers use paths relative to benchmarking/. Sourcing Benchmarking.R
# itself would start its trial loops (lines 386+), so take only what we need.
setwd(BENCH)
source("benchmarking_helpers.R")
source("classic-ML-based/run_singleCellNet.R")
source("classic-ML-based/run_CaSTLe.R")
source("classic-ML-based/run_scPred.R")

# scPred first: in the S9 run it was the first tool of classicML_based1,
# so its RNG state matches most closely when it runs straight after the split.
TOOLS <- c("scPred", "singleCellNet", "CaSTLe")
TOOL_REGISTRY <- list(
  scPred        = run_scPred_function,
  singleCellNet = run_singleCellNet_function,
  CaSTLe        = run_CaSTLe_function
)
TOL <- c(scPred = 0.02, singleCellNet = 0.005, CaSTLe = 0.005)
SEEDS <- c(123, 124)

DATASETS <- list(
  S9 = list(file = "MacParland-Liver-Broad-2018_original_seurat_subset.rds",
            folder = "S9-MacParland-Liver-Broad_original"),
  S3 = list(file = "Nowakowski-Cortex-2017_for_use_subset.rds",
            folder = "S3-Nowakowski-Cortex-2017_for_use")
)

# Published kappa, read from the same CSVs Plots_Analysis.R loads.
published_kappa <- function(folder) {
  out <- list()
  for (f in list.files(file.path(P5, folder), pattern = "\\.csv$", full.names = TRUE)) {
    d <- read.csv(f, check.names = FALSE)
    d <- d[d$tool %in% TOOLS, c("tool", "kappa_mean", "accuracy_mean")]
    if (nrow(d)) out[[basename(f)]] <- cbind(d, file = basename(f))
  }
  do.call(rbind, out)
}

run_once <- function(obj, seed) {
  res <- run_all_tools_single_split(obj, tools_to_run = TOOLS, seed = seed)
  tab <- data.frame(
    tool   = names(res),
    seed   = seed,
    n_test = vapply(res, function(r) length(r$detailed_results$predictions), 1L),
    acc    = vapply(res, function(r) r$pooled_metrics$overall_accuracy, 1),
    kappa_rerun = vapply(res, function(r) r$pooled_metrics$cohens_kappa, 1),
    row.names = NULL
  )
  list(res = res, tab = tab)
}

checks <- list(); tidy <- list(); labels <- list(); raw <- list()

for (ds in names(DATASETS)) {
  cfg <- DATASETS[[ds]]
  cat(sprintf("\n######## %s  %s ########\n", ds, cfg$file))
  obj_full <- readRDS(file.path("data", cfg$file))
  lab <- obj_full$Ground_Truth_Celltype
  labelled <- colnames(obj_full)[!is.na(lab) & lab != ""]
  cat(sprintf("%d of %d cells have a label\n", length(labelled), ncol(obj_full)))
  obj <- subset(obj_full, cells = labelled)
  Idents(obj) <- "Ground_Truth_Celltype"

  pub <- published_kappa(cfg$folder)
  print(pub)
  pub_k <- setNames(pub$kappa_mean, pub$tool)

  chosen <- NULL
  for (seed in SEEDS) {
    fold_cache <- list()
    run <- run_once(obj, seed)
    run$tab$kappa_published <- pub_k[run$tab$tool]
    run$tab$delta <- run$tab$kappa_rerun - run$tab$kappa_published
    run$tab$within_tol <- abs(run$tab$delta) <= TOL[run$tab$tool]
    run$tab$dataset <- ds
    print(run$tab)
    checks[[paste(ds, seed)]] <- run$tab
    det <- run$tab[run$tab$tool != "scPred", ]
    if (nrow(det) == 2 && all(det$within_tol)) { chosen <- run; break }
  }
  if (is.null(chosen)) {
    write.csv(do.call(rbind, checks), file.path(OUT_DIR, "kappa_check.csv"), row.names = FALSE)
    stop(sprintf("%s: neither seed reproduced the published kappa for singleCellNet and CaSTLe", ds))
  }
  cat(sprintf("%s: seed %d reproduces the published kappa\n", ds, chosen$tab$seed[1]))
  raw[[ds]] <- chosen$res

  # Same UMAP call as SF2, on the full object.
  n_pcs <- min(30L, ncol(Embeddings(obj_full, "pca")))
  obj_full <- RunUMAP(obj_full, dims = seq_len(n_pcs), verbose = FALSE)
  um <- Embeddings(obj_full, "umap")

  test_ids <- chosen$res[[1]]$detailed_results$cell_ids
  df <- data.frame(dataset = ds, cell_id = colnames(obj_full),
                   UMAP_1 = um[, 1], UMAP_2 = um[, 2],
                   truth = as.character(obj_full$Ground_Truth_Celltype),
                   split = ifelse(colnames(obj_full) %in% test_ids, "test",
                                  ifelse(colnames(obj_full) %in% labelled, "train", "excluded")),
                   stringsAsFactors = FALSE)
  for (tool in TOOLS) {
    d <- chosen$res[[tool]]$detailed_results
    # Join check: the wrapper's true labels must match the object's labels
    # for the same barcodes, otherwise predictions would land on wrong cells.
    stopifnot(setequal(d$cell_ids, test_ids),
              all(as.character(d$true_labels) == df$truth[match(d$cell_ids, df$cell_id)]))
    df[[paste0("pred_", tool)]] <- as.character(d$predictions)[match(df$cell_id, d$cell_ids)]
    lt <- as.data.frame(table(prediction = as.character(d$predictions)), stringsAsFactors = FALSE)
    lt$dataset <- ds; lt$tool <- tool
    lt$is_truth_type <- lt$prediction %in% unique(df$truth)
    labels[[paste(ds, tool)]] <- lt
  }
  tidy[[ds]] <- df
  rm(obj, obj_full); gc()
}

write.csv(do.call(rbind, checks), file.path(OUT_DIR, "kappa_check.csv"), row.names = FALSE)
write.csv(do.call(rbind, labels), file.path(OUT_DIR, "predicted_labels.csv"), row.names = FALSE)
write.csv(do.call(rbind, tidy), TIDY, row.names = FALSE)
saveRDS(raw, file.path(OUT_DIR, "predictions_raw.rds"))
writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "sessionInfo.txt"))
cat("\nWrote", TIDY, "\n")
print(do.call(rbind, labels))
