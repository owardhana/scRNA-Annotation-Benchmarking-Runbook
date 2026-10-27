#!/usr/bin/env python3
"""Compute Phase 4 + Phase 5 per-tool mean scores + scatter stats for Results.md."""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats

P7A_SCORE = os.path.join(BENCH_ROOT, "results/P7a Marker - Real Database/with_score")
P7A_RUNTIME = os.path.join(BENCH_ROOT, "results/P7a Marker - Real Database/with_score/runtime_dataset.tsv")
P7B_SCORE = os.path.join(BENCH_ROOT, "results/P7b LLM - Real Internet/with_score")
P7B_RUNTIME = os.path.join(BENCH_ROOT, "results/P7b LLM - Real Internet/with_score/runtime_dataset.tsv")
P8B = os.path.join(BENCH_ROOT, "results/P8b FM - Real/data/real_FM_results.tsv")

# Real profiles (same as P5 script)
REAL_PROFILES = pd.DataFrame({
    "dataset": ["Darmanis-Brain-2015_for_use","Marques-Brain-2016_for_use","Nowakowski-Cortex-2017_for_use",
                "Grun-Pancreas-2016_original","Tabula Muris-FACS-3k_for_use","He-Skin-2020_for_use",
                "Zhao-Immune-Fine-2020_for_use","Zheng-ZhengSort-5cl-2017_original",
                "MacParland-Liver-Broad_original",
                "MacParland-Liver-Broad-2018_original",
                "Zheng-Zhengsort-5cl-2017_original",
                ],
    "log_cells":  np.log10([466, 2993, 3005, 495, 2984, 2990, 14986, 20000, 8444, 8444, 20000]),
    "n_types":    [9, 13, 48, 9, 30, 21, 30, 10, 8, 8, 10],
    "shannon":    [0.8795, 0.9251, 0.8283, 0.9273, 0.9371, 0.5324, 0.8881, 1.0000, 0.7129, 0.7129, 1.0000],
    "knn_purity": [0.7468, 0.5323, 0.1073, 0.1761, 0.7870, 0.9274, 0.7165, 0.6743, 0.9597, 0.9597, 0.6743]
}).drop_duplicates(subset=["dataset"], keep="first")

def scatter_stats(d, x, y):
    d = d.dropna(subset=[x, y])
    d = d[np.isfinite(d[x]) & np.isfinite(d[y])]
    if d.shape[0] < 5: return None
    slope, intercept, r, p_pe, _ = stats.linregress(d[x], d[y])
    sp_r, sp_p = stats.spearmanr(d[x], d[y])
    return {"R2": r**2, "spearman": sp_r, "pearson": r, "p_spearman": sp_p,
            "p_pearson": p_pe, "n": int(d.shape[0]), "slope": slope}

# ---- Phase 4 marker DB ----
def parse_filename_marker(f):
    base = os.path.basename(f).replace(".tsv", "")
    base = base.replace("_database_cluster_map", "")
    # Strip _subset, _for_use_subset etc
    m = re.match(r"(.+?)_subset_(.+)", base)
    if m: return m.group(1), m.group(2)
    return None, None

def parse_filename_llm(f):
    base = os.path.basename(f).replace(".tsv", "")
    base = base.replace("_cluster_map", "")
    m = re.match(r"(.+?)_subset_(.+)", base)
    if m: return m.group(1), m.group(2)
    # Handle seurat variant for MacParland
    return None, None

print("=== PHASE 4 MARKER DB ===")
marker_rows = []
for f in sorted(glob.glob(os.path.join(P7A_SCORE, "*.tsv"))):
    if "runtime" in f: continue
    if " 2.19.28 PM" in f: continue
    dataset, tool = parse_filename_marker(f)
    if dataset is None: continue
    # Normalise seurat alias
    dataset_norm = dataset.replace("_seurat", "").replace("MacParland-Liver-Broad-2018_original", "MacParland-Liver-Broad_original")
    df = pd.read_csv(f, sep="\t")
    # Mean of three scores per row, then mean across rows
    df["mean_score"] = df[["score_1", "score_2", "score_3"]].mean(axis=1, skipna=True)
    mean_score = df["mean_score"].mean(skipna=True)
    marker_rows.append({"dataset": dataset_norm, "tool": tool, "mean_score": mean_score, "n_clusters": df.shape[0]})
marker_df = pd.DataFrame(marker_rows)
# Aggregate per dataset/tool (one file per dataset+tool, but Mac has alias)
marker_df = marker_df.groupby(["dataset","tool"], as_index=False).agg(mean_score=("mean_score","mean"))
marker_df = marker_df.merge(REAL_PROFILES, on="dataset", how="left")
mr = pd.read_csv(P7A_RUNTIME, sep="\t")
mr.columns = [c.strip() for c in mr.columns]
mr["dataset"] = mr["dataset"].str.replace("_seurat", "")
mr["dataset"] = mr["dataset"].replace({"MacParland-Liver-Broad-2018_original": "MacParland-Liver-Broad_original"})
mr["tool"] = mr["tool"].str.replace("_database", "")
marker_df = marker_df.merge(mr.rename(columns={"runtime":"runtime_s"}), on=["dataset","tool"], how="left")
marker_df["log_runtime"] = np.log10(marker_df["runtime_s"].astype(float))
print(marker_df.sort_values(["tool","dataset"]).to_string(index=False))
print("\n-- Per-tool mean score (P7a marker DB) --")
print(marker_df.groupby("tool")["mean_score"].agg(["mean","median","count"]).round(3))
print(f"\n-- Family-level mean score: {marker_df['mean_score'].mean():.3f}")

# Scatter: purity vs score (family-pooled)
print("\n-- Purity → score (marker family, pooled) --")
print(scatter_stats(marker_df, "knn_purity", "mean_score"))
print("-- log_cells → log_runtime (marker family) --")
print(scatter_stats(marker_df, "log_cells", "log_runtime"))
print("-- n_types → log_runtime (marker family) --")
print(scatter_stats(marker_df, "n_types", "log_runtime"))

# ---- Phase 4 LLM ----
print("\n=== PHASE 4 LLM ===")
llm_rows = []
for f in sorted(glob.glob(os.path.join(P7B_SCORE, "*.tsv"))):
    if "runtime" in f: continue
    dataset, tool = parse_filename_llm(f)
    if dataset is None: continue
    dataset_norm = dataset
    df = pd.read_csv(f, sep="\t")
    df["mean_score"] = df[["score_1", "score_2", "score_3"]].mean(axis=1, skipna=True)
    mean_score = df["mean_score"].mean(skipna=True)
    llm_rows.append({"dataset": dataset_norm, "tool": tool, "mean_score": mean_score, "n_clusters": df.shape[0]})
llm_df = pd.DataFrame(llm_rows)
llm_df = llm_df.groupby(["dataset","tool"], as_index=False).agg(mean_score=("mean_score","mean"))
llm_df = llm_df.merge(REAL_PROFILES, on="dataset", how="left")
lr = pd.read_csv(P7B_RUNTIME, sep="\t")
lr.columns = [c.strip() for c in lr.columns]
llm_df = llm_df.merge(lr.rename(columns={"runtime":"runtime_s"}), on=["dataset","tool"], how="left")
llm_df["log_runtime"] = np.log10(llm_df["runtime_s"].astype(float))
print(llm_df.sort_values(["tool","dataset"]).to_string(index=False))
print("\n-- Per-tool mean score (P7b LLM) --")
print(llm_df.groupby("tool")["mean_score"].agg(["mean","median","count"]).round(3))
print(f"\n-- Family-level mean score: {llm_df['mean_score'].mean():.3f}")

print("\n-- Purity → score (LLM family, pooled) --")
print(scatter_stats(llm_df, "knn_purity", "mean_score"))
print("-- log_cells → log_runtime (LLM family) --")
print(scatter_stats(llm_df, "log_cells", "log_runtime"))
print("-- n_types → log_runtime (LLM family) --")
print(scatter_stats(llm_df, "n_types", "log_runtime"))

# ---- Phase 5 FM ----
print("\n=== PHASE 5 FM ===")
fm = pd.read_csv(P8B, sep="\t")
# Map FM 'sample' column to dataset name in REAL_PROFILES
fm["dataset"] = fm["sample"].str.replace("_subset", "")
fm["dataset"] = fm["dataset"].replace({
    "MacParland-Liver-Broad-2018_original": "MacParland-Liver-Broad_original"
})
fm = fm.merge(REAL_PROFILES, on="dataset", how="left")
fm["log_runtime"] = np.log10(fm["runtime_s"].astype(float))
# Filter out kappa = NA / -1 / 0 sentinel (some FM-dataset combos may not have run)
fm_clean = fm[(fm["kappa"].notna()) & (fm["kappa"] > 0)]
print(fm_clean[["model","dataset","kappa","runtime_s","log_runtime","knn_purity","log_cells","n_types"]].sort_values(["model","dataset"]).to_string(index=False))
print("\n-- Per-model mean kappa (P8b FM) --")
print(fm_clean.groupby("model")["kappa"].agg(["mean","median","count"]).round(3))
print(f"\n-- Family-level mean kappa: {fm_clean['kappa'].mean():.3f}")

print("\n-- Purity → κ (FM family, pooled) --")
print(scatter_stats(fm_clean, "knn_purity", "kappa"))
print("-- log_cells → log_runtime (FM family) --")
print(scatter_stats(fm_clean, "log_cells", "log_runtime"))

print("\n-- Per-model FM stats (purity → κ) --")
for model, g in fm_clean.groupby("model"):
    s = scatter_stats(g, "knn_purity", "kappa")
    print(f"  {model}: {s}")
print("\n-- Per-model FM stats (log_cells → log_runtime) --")
for model, g in fm_clean.groupby("model"):
    s = scatter_stats(g, "log_cells", "log_runtime")
    print(f"  {model}: {s}")

# Mean over per-model means (gives equal weight per model rather than per row)
print("\n-- Per-model mean κ, then averaged: --")
per_model_means = fm_clean.groupby("model")["kappa"].mean()
print(per_model_means)
print(f"Mean of model means: {per_model_means.mean():.3f}")

# Compute Phase 4 LLM per-tool scatter as well
print("\n-- Per-tool LLM stats (purity → score) --")
for tool, g in llm_df.groupby("tool"):
    s = scatter_stats(g, "knn_purity", "mean_score")
    print(f"  {tool}: {s}")

# Save tidy outputs
marker_df.to_csv("/tmp/results_phase4_marker.csv", index=False)
llm_df.to_csv("/tmp/results_phase4_llm.csv", index=False)
fm_clean.to_csv("/tmp/results_phase5_fm.csv", index=False)
print("\nDone. /tmp/results_phase4_marker.csv, results_phase4_llm.csv, results_phase5_fm.csv")
