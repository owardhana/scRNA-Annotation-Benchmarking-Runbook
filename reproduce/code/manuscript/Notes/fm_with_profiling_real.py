#!/usr/bin/env python3
"""Re-derive FM dataset-level scatter using Profiling_Real.csv values (what P8b script uses)."""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import pandas as pd, numpy as np
from scipy import stats

fm = pd.read_csv(os.path.join(BENCH_ROOT, "results/P8b FM - Real/data/real_FM_results.tsv"), sep="\t")
fm["scenario"] = fm["sample"].str.replace("_subset","")

prof = pd.read_csv(os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_Real.csv"))

# P8b uses these 9 keys:
fm_scenarios = ["Darmanis-Brain-2015_for_use","Marques-Brain-2016_for_use","Nowakowski-Cortex-2017_for_use",
                "Grun-Pancreas-2016_original","Tabula Muris-FACS-3k_for_use","He-Skin-2020_for_use",
                "Zhao-Immune-Fine-2020_for_use","Zheng-ZhengSort-5cl-2017_original",
                "MacParland-Liver-Broad-2018_original"]
profile_p8b = prof[prof["Sample"].isin(fm_scenarios)].copy()
profile_p8b["log_cells"] = np.log10(profile_p8b["Total_Cells"])
profile_p8b["scenario"] = profile_p8b["Sample"]
profile_p8b = profile_p8b[["scenario","log_cells","Total_CellTypes","Norm_Shannon_Entropy","KNN_Purity"]].rename(
    columns={"Total_CellTypes":"n_types","Norm_Shannon_Entropy":"shannon","KNN_Purity":"knn_purity"})
print("=== Profile values used by P8b script ===")
print(profile_p8b.sort_values("scenario").to_string(index=False))

fm = fm.merge(profile_p8b, on="scenario", how="left")
fm = fm[(fm["kappa"].notna()) & (fm["kappa"] > 0)]
fm["log_runtime"] = np.log10(fm["runtime_s"])

# Dataset-level aggregate (mean across 5 models)
agg = fm.groupby("scenario").agg(
    mean_runtime_raw=("runtime_s","mean"),
    mean_kappa=("kappa","mean"),
    log_cells=("log_cells","first"),
    n_types=("n_types","first"),
    knn_purity=("knn_purity","first"),
).reset_index()
agg["log_mean_runtime"] = np.log10(agg["mean_runtime_raw"])

print("\n=== Per-dataset aggregate (n=9) ===")
print(agg.sort_values("log_cells").to_string(index=False))

def fit(d, x, y):
    d = d.dropna(subset=[x,y])
    slope, intercept, r, p_pe, _ = stats.linregress(d[x], d[y])
    sp_r, sp_p = stats.spearmanr(d[x], d[y])
    return f"n={len(d)} R²={r**2:.4f} Pearson_r={r:.4f} (p={p_pe:.4f}) Spearman_ρ={sp_r:.4f} (p={sp_p:.4f}) slope={slope:.4f} intercept={intercept:.4f}"

print("\n-- FM mean RAW runtime → log10 vs log_cells --")
print(fit(agg, "log_cells", "log_mean_runtime"))

# Now try mean of log_runtime per dataset
fm_with_log = fm.copy()
agg_log = fm_with_log.groupby("scenario").agg(
    mean_log_runtime=("log_runtime","mean"),
    log_cells=("log_cells","first"),
).reset_index()
print("\n-- FM mean of LOG10 runtime vs log_cells --")
print(fit(agg_log, "log_cells", "mean_log_runtime"))

# purity-κ
print("\n-- FM mean κ vs knn_purity --")
print(fit(agg, "knn_purity", "mean_kappa"))

# Now also test: what if there's a single-model R² they're reporting? Earlier showed per-model R²=0.85-0.91.
# Let me try the original (perhaps user computed a different stat — e.g. correlation of log_cell with mean_runtime where they took a different model-set mean).

# Show variations
print("\n=== Pearson r breakdown ===")
print(f"log_cells x log10(mean RAW runtime): {agg[['log_cells','log_mean_runtime']].corr().iloc[0,1]:.4f}")
print(f"log_cells x mean(log_runtime):       {agg_log[['log_cells','mean_log_runtime']].corr().iloc[0,1]:.4f}")

# Try excluding outliers
for excl in ["Grun-Pancreas-2016_original", "Marques-Brain-2016_for_use", "Nowakowski-Cortex-2017_for_use"]:
    sub = agg[agg["scenario"] != excl]
    print(f"Without {excl[:30]:30s}: R²={fit(sub, 'log_cells', 'log_mean_runtime')}")
