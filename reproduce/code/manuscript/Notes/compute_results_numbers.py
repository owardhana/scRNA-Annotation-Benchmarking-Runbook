#!/usr/bin/env python3
"""Compute per-paradigm eta-squared + scatter R^2/Spearman/Pearson for Results.md.
Mirrors the logic of results/P4 and P5 Plots_Analysis.R."""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

P4_DIR = os.path.join(BENCH_ROOT, "results/P4 Synthetic Datasets V1")
P5_DIR = os.path.join(BENCH_ROOT, "results/P5 Real Life Datasets V2")

marker_tools = ["clustifyr_hyper","clustifyr_jaccard","scCATCH","SCINA","ScInfeR","SCSA","scSorter","scType"]
similarity_tools = ["CHETAH","CIPR","scmap_cell","scmap_cluster","Seurat_Transfer_CCA","Seurat_Transfer_PCA","Seurat_Transfer_RPCA","SingleR"]
classicML_tools = ["CaSTLe","CellTypist","scAnnotate","scAnnotatR","scClassify","scibetR","scID",
    "scPred","scPred_adaboost","scPred_avNNet","scPred_bayesglm","scPred_earth",
    "scPred_glm","scPred_glmboost","scPred_knn","scPred_lda","scPred_mlp","scPred_multinom",
    "scPred_nb","scPred_nnet","scPred_regLogistic","scPred_rf","scPred_svmLinear",
    "scPred_svmPoly","scPred_xgbTree","singleCellNet","scLearn"]
DL_tools = ["ACTINN","Cell_BLAST","CIForm","NeuCA","scBalance","scDeepSort","scHash","scMMT","TOSICA"]
semi_tools = ["CALLR","CAMLU","MARS","SCANVI","scArches","scGAD","scnym","scSemiCluster","scSemiGAN"]

cat_map = {}
for t in marker_tools: cat_map[t] = "Marker-Based"
for t in similarity_tools: cat_map[t] = "Similarity-Based"
for t in classicML_tools: cat_map[t] = "Classic ML"
for t in DL_tools: cat_map[t] = "Deep Learning"
for t in semi_tools: cat_map[t] = "Semi-Supervised"

exclude_tools = {"ScInfeR","scPred_adaboost","scAnnotatR","CIPR","scmap_cluster",
                 "clustifyr_hyper","clustifyr_jaccard","SCSA","scCATCH","scType",
                 "scGAD","scDeepSort","CALLR"}

def drop_index_col(df):
    if df.shape[1] > 0 and (df.columns[0] in ("", "X", "Unnamed: 0")):
        df = df.iloc[:, 1:]
    return df

def load_p4():
    rows = []
    for sc in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]:
        sp = os.path.join(P4_DIR, sc)
        if not os.path.isdir(sp): continue
        for r in sorted(os.listdir(sp)):
            rp = os.path.join(sp, r)
            if not os.path.isdir(rp): continue
            for f in glob.glob(os.path.join(rp, "*.csv")):
                try:
                    df = pd.read_csv(f, low_memory=False)
                except Exception:
                    continue
                df = drop_index_col(df)
                df["scenario"] = sc
                df["replicate"] = r
                rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out[~out["tool"].isin(exclude_tools)].copy()
    out["tool_category"] = out["tool"].map(cat_map)
    return out

def load_p5():
    rows = []
    for sc in ["S1-Darmanis-Brain-2015_for_use","S2-Marques-Brain_for_use","S3-Nowakowski-Cortex-2017_for_use",
              "S4-Grun-Pancreas-2016_original","S5-Tabula Muris-FACS-3k_for_use","S6-He-Skin-2020_for_use",
              "S7-Zhao-Immune-Fine-2020_for_use","S8-Zheng-Zhengsort-5cl-2017_original",
              "S9-MacParland-Liver-Broad_original"]:
        sp = os.path.join(P5_DIR, sc)
        if not os.path.isdir(sp): continue
        for f in glob.glob(os.path.join(sp, "*.csv")):
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception:
                continue
            df = drop_index_col(df)
            df["scenario"] = sc
            df["replicate"] = "single"
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out[~out["tool"].isin(exclude_tools)].copy()
    out["tool_category"] = out["tool"].map(cat_map)
    return out

# Profile loaders
synth_raw = pd.read_csv(os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_L9.csv"))
synth_raw = synth_raw.rename(columns={"Scenario": "scenario"})
SYNTH_PROFILES = synth_raw.groupby("scenario").agg(
    log_cells = ("Total_Cells", lambda x: np.log10(x.mean())),
    n_types   = ("Total_CellTypes", lambda x: int(round(x.mean()))),
    shannon   = ("Norm_Shannon_Entropy", "mean"),
    knn_purity= ("KNN_Purity", "mean")
).reset_index()

REAL_PROFILES = pd.DataFrame({
    "scenario": ["S1-Darmanis-Brain-2015_for_use","S2-Marques-Brain_for_use","S3-Nowakowski-Cortex-2017_for_use",
                 "S4-Grun-Pancreas-2016_original","S5-Tabula Muris-FACS-3k_for_use","S6-He-Skin-2020_for_use",
                 "S7-Zhao-Immune-Fine-2020_for_use","S8-Zheng-Zhengsort-5cl-2017_original",
                 "S9-MacParland-Liver-Broad_original"],
    "log_cells":  np.log10([466, 2993, 3005, 495, 2984, 2990, 14986, 20000, 8444]),
    "n_types":    [9, 13, 48, 9, 30, 21, 30, 10, 8],
    "shannon":    [0.8795, 0.9251, 0.8283, 0.9273, 0.9371, 0.5324, 0.8881, 1.0000, 0.7129],
    "knn_purity": [0.7468, 0.5323, 0.1073, 0.1761, 0.7870, 0.9274, 0.7165, 0.6743, 0.9597]
})

def prep(df, profiles):
    df = df.copy()
    mask_zeros = (df.get("kappa_mean", 0) == 0) & (df.get("accuracy_mean", 0) == 0) & (df.get("f1_mean", 0) == 0)
    df.loc[mask_zeros, "kappa_mean"] = np.nan
    df = df[df["tool_category"].notna()].copy()
    # Drop non-numeric columns we don't need; keep core numerics
    numeric_cols = ["kappa_mean", "accuracy_mean", "f1_mean", "mcc_mean",
                    "rare_f1_mean", "runtime_mean", "peak_system_memory_mean_mb"]
    keep = [c for c in numeric_cols if c in df.columns]
    grouped = df.groupby(["scenario", "tool", "tool_category"])[keep].mean(numeric_only=True).reset_index()
    merged = grouped.merge(profiles, on="scenario", how="left")
    merged = merged[merged["kappa_mean"].notna()]
    return merged

def eta_partition(d, response):
    d = d[d[response].notna() & np.isfinite(d[response])].copy()
    if d.shape[0] < 6 or d["log_cells"].nunique() < 2:
        return None
    formula = f"{response} ~ log_cells + n_types + shannon + knn_purity"
    try:
        fit = ols(formula, data=d).fit()
        a = sm.stats.anova_lm(fit, typ=1)
    except Exception as e:
        return None
    ss_total = a["sum_sq"].sum()
    out = []
    for factor in ["log_cells","n_types","shannon","knn_purity"]:
        if factor not in a.index: continue
        out.append({
            "factor": factor,
            "eta_sq": a.loc[factor, "sum_sq"] / ss_total,
            "p_value": a.loc[factor, "PR(>F)"],
            "df": int(a.loc[factor, "df"]),
        })
    out.append({"factor": "Residuals", "eta_sq": a.loc["Residual", "sum_sq"] / ss_total,
                "p_value": np.nan, "df": int(a.loc["Residual", "df"])})
    return pd.DataFrame(out)

def scatter_stats(d, x, y):
    d = d[d[x].notna() & d[y].notna() & np.isfinite(d[x]) & np.isfinite(d[y])].copy()
    if d.shape[0] < 6: return None
    slope, intercept, r, p_pe, se = stats.linregress(d[x], d[y])
    sp_r, sp_p = stats.spearmanr(d[x], d[y])
    return {"R2": r**2, "spearman": sp_r, "pearson": r, "p_spearman": sp_p, "p_pearson": p_pe,
            "n": int(d.shape[0]), "slope": slope, "intercept": intercept}

# ---- Run ----
P4_raw = load_p4()
P5_raw = load_p5()

P4 = prep(P4_raw, SYNTH_PROFILES)
P5 = prep(P5_raw, REAL_PROFILES)

print(f"P4 rows after prep: {len(P4)} | tools: {P4['tool'].nunique()} | paradigms: {sorted(P4['tool_category'].dropna().unique().tolist())}")
print(f"P5 rows after prep: {len(P5)} | tools: {P5['tool'].nunique()} | paradigms: {sorted(P5['tool_category'].dropna().unique().tolist())}")

paradigms = ["Marker-Based", "Similarity-Based", "Classic ML", "Deep Learning", "Semi-Supervised"]

eta_rows = []
def collect_eta(d_all, arm, response):
    for par in paradigms:
        d = d_all[d_all["tool_category"] == par].copy()
        if response == "log_runtime":
            d = d[d["runtime_mean"].notna() & (d["runtime_mean"] > 0)].copy()
            d["log_runtime"] = np.log10(d["runtime_mean"])
        elif response == "log_mem":
            d = d[d["peak_system_memory_mean_mb"].notna() & (d["peak_system_memory_mean_mb"] > 0)].copy()
            d["log_mem"] = np.log10(d["peak_system_memory_mean_mb"])
        r = eta_partition(d, response)
        if r is not None:
            r = r.assign(arm=arm, paradigm=par, response=response)
            eta_rows.append(r)

for arm, d_all in [("synth", P4), ("real", P5)]:
    collect_eta(d_all, arm, "kappa_mean")
    collect_eta(d_all, arm, "log_runtime")
    collect_eta(d_all, arm, "log_mem")

eta_df = pd.concat(eta_rows, ignore_index=True) if eta_rows else pd.DataFrame()

scatter_rows = []
def collect_scatter(d_all, arm, xname, yname, ycomp=None):
    for par in paradigms:
        d = d_all[d_all["tool_category"] == par].copy()
        if yname == "log_runtime":
            d = d[d["runtime_mean"].notna() & (d["runtime_mean"] > 0)].copy()
            d["log_runtime"] = np.log10(d["runtime_mean"])
        elif yname == "log_mem":
            d = d[d["peak_system_memory_mean_mb"].notna() & (d["peak_system_memory_mean_mb"] > 0)].copy()
            d["log_mem"] = np.log10(d["peak_system_memory_mean_mb"])
        s = scatter_stats(d, xname, yname)
        if s is not None:
            s = {**s, "arm": arm, "paradigm": par, "x": xname, "y": yname}
            scatter_rows.append(s)

for arm, d_all in [("synth", P4), ("real", P5)]:
    collect_scatter(d_all, arm, "knn_purity", "kappa_mean")
    collect_scatter(d_all, arm, "log_cells", "log_runtime")
    collect_scatter(d_all, arm, "log_cells", "log_mem")

scatter_df = pd.DataFrame(scatter_rows)

# Pooled purity-kappa for cross-phase synthesis
pooled = []
for arm, d_all in [("synth", P4), ("real", P5)]:
    s = scatter_stats(d_all, "knn_purity", "kappa_mean")
    if s is not None:
        pooled.append({**s, "arm": arm})
pooled_df = pd.DataFrame(pooled)

# Pooled performance-cost null check
pcost = []
for arm, d_all in [("synth", P4), ("real", P5)]:
    d_rt = d_all[d_all["runtime_mean"].notna() & (d_all["runtime_mean"] > 0)].copy()
    d_rt["log_runtime"] = np.log10(d_rt["runtime_mean"])
    s = scatter_stats(d_rt, "kappa_mean", "log_runtime")
    if s is not None:
        pcost.append({**s, "arm": arm, "metric": "kappa_vs_logruntime"})
    d_mem = d_all[d_all["peak_system_memory_mean_mb"].notna() & (d_all["peak_system_memory_mean_mb"] > 0)].copy()
    d_mem["log_mem"] = np.log10(d_mem["peak_system_memory_mean_mb"])
    s = scatter_stats(d_mem, "kappa_mean", "log_mem")
    if s is not None:
        pcost.append({**s, "arm": arm, "metric": "kappa_vs_logmem"})
pcost_df = pd.DataFrame(pcost)

eta_df.to_csv("/tmp/results_eta_squared.csv", index=False)
scatter_df.to_csv("/tmp/results_scatter_stats.csv", index=False)
pooled_df.to_csv("/tmp/results_pooled_purity.csv", index=False)
pcost_df.to_csv("/tmp/results_perf_cost.csv", index=False)

print("\n=== ETA SQUARED (κ, per paradigm, per arm) ===")
e_kappa = eta_df[eta_df["response"] == "kappa_mean"].drop(columns=["response"]).copy()
print(e_kappa.pivot_table(index=["arm","paradigm"], columns="factor", values="eta_sq").round(3))

print("\n=== ETA SQUARED (log_runtime, per paradigm, per arm) ===")
e_rt = eta_df[eta_df["response"] == "log_runtime"].drop(columns=["response"]).copy()
print(e_rt.pivot_table(index=["arm","paradigm"], columns="factor", values="eta_sq").round(3))

print("\n=== ETA SQUARED (log_mem, per paradigm, per arm) ===")
e_mem = eta_df[eta_df["response"] == "log_mem"].drop(columns=["response"]).copy()
print(e_mem.pivot_table(index=["arm","paradigm"], columns="factor", values="eta_sq").round(3))

print("\n=== KAPPA-PURITY SCATTER (per paradigm) ===")
print(scatter_df[(scatter_df["x"]=="knn_purity") & (scatter_df["y"]=="kappa_mean")][["arm","paradigm","R2","spearman","pearson","p_spearman","p_pearson","n"]].round(3))

print("\n=== LOG-RUNTIME ~ LOG-CELLS (per paradigm) ===")
print(scatter_df[(scatter_df["x"]=="log_cells") & (scatter_df["y"]=="log_runtime")][["arm","paradigm","R2","spearman","pearson","p_spearman","p_pearson","n","slope"]].round(3))

print("\n=== LOG-MEM ~ LOG-CELLS (per paradigm) ===")
print(scatter_df[(scatter_df["x"]=="log_cells") & (scatter_df["y"]=="log_mem")][["arm","paradigm","R2","spearman","pearson","p_spearman","p_pearson","n","slope"]].round(3))

print("\n=== POOLED PURITY-KAPPA ===")
print(pooled_df.round(3))

print("\n=== PERFORMANCE-COST NULL CHECK ===")
print(pcost_df.round(3))

print("\nDone. Files: /tmp/results_eta_squared.csv, /tmp/results_scatter_stats.csv, /tmp/results_pooled_purity.csv, /tmp/results_perf_cost.csv")
