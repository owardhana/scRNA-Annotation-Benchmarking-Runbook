#!/usr/bin/env python3
"""Compute Type III partial η² (marginal-after-all-others SS partition) per paradigm.
Type III = SS_factor given all other factors at the same level = order-independent.
For continuous predictors with no interactions, Type II = Type III."""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os, glob, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

P4_DIR = os.path.join(BENCH_ROOT, "results/P4 Synthetic Datasets V1")
P5_DIR = os.path.join(BENCH_ROOT, "results/P5 Real Life Datasets V2")

# Tool category map etc - reuse
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
    if df.shape[1] > 0 and (df.columns[0] in ("", "X", "Unnamed: 0")): return df.iloc[:, 1:]
    return df

P5_SCENARIO_TO_PROFILE = {
    "S1-Darmanis-Brain-2015_for_use":        "Darmanis-Brain-2015_for_use",
    "S2-Marques-Brain_for_use":              "Marques-Brain-2016_for_use",
    "S3-Nowakowski-Cortex-2017_for_use":     "Nowakowski-Cortex-2017_for_use",
    "S4-Grun-Pancreas-2016_original":        "Grun-Pancreas-2016_original",
    "S5-Tabula Muris-FACS-3k_for_use":       "Tabula Muris-FACS-3k_for_use",
    "S6-He-Skin-2020_for_use":               "He-Skin-2020_for_use",
    "S7-Zhao-Immune-Fine-2020_for_use":      "Zhao-Immune-Fine-2020_for_use",
    "S8-Zheng-Zhengsort-5cl-2017_original":  "Zheng-ZhengSort-5cl-2017_original",
    "S9-MacParland-Liver-Broad_original":    "MacParland-Liver-Broad-2018_original",
}
prof_real = pd.read_csv(os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_Real.csv"))
def build_P5_profile():
    rows = []
    for sc, key in P5_SCENARIO_TO_PROFILE.items():
        r = prof_real[prof_real["Sample"] == key].iloc[0]
        rows.append({"scenario": sc, "log_cells": np.log10(r["Total_Cells"]),
                     "n_types": int(r["Total_CellTypes"]), "shannon": r["Norm_Shannon_Entropy"],
                     "knn_purity": r["KNN_Purity"]})
    return pd.DataFrame(rows)
P5_PROFILES = build_P5_profile()

synth_raw = pd.read_csv(os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_L9.csv"))
synth_raw = synth_raw.rename(columns={"Scenario":"scenario"})
P4_PROFILES = synth_raw.groupby("scenario").agg(
    log_cells=("Total_Cells", lambda x: np.log10(x.mean())),
    n_types=("Total_CellTypes", lambda x: int(round(x.mean()))),
    shannon=("Norm_Shannon_Entropy","mean"),
    knn_purity=("KNN_Purity","mean")
).reset_index()

def load_p4():
    rows = []
    for sc in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]:
        sp = os.path.join(P4_DIR, sc)
        for r in sorted(os.listdir(sp)):
            rp = os.path.join(sp, r)
            if not os.path.isdir(rp): continue
            for f in glob.glob(os.path.join(rp, "*.csv")):
                try: df = pd.read_csv(f, low_memory=False)
                except: continue
                df = drop_index_col(df); df["scenario"]=sc; df["replicate"]=r
                rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out[~out["tool"].isin(exclude_tools)].copy()
    out["tool_category"] = out["tool"].map(cat_map)
    return out

def load_p5():
    rows = []
    for sc in P5_SCENARIO_TO_PROFILE.keys():
        sp = os.path.join(P5_DIR, sc)
        for f in glob.glob(os.path.join(sp, "*.csv")):
            try: df = pd.read_csv(f, low_memory=False)
            except: continue
            df = drop_index_col(df); df["scenario"]=sc; df["replicate"]="single"
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out[~out["tool"].isin(exclude_tools)].copy()
    out["tool_category"] = out["tool"].map(cat_map)
    return out

def prep(df, profiles):
    df = df.copy()
    mz = (df.get("kappa_mean",0)==0)&(df.get("accuracy_mean",0)==0)&(df.get("f1_mean",0)==0)
    df.loc[mz,"kappa_mean"] = np.nan
    df = df[df["tool_category"].notna()].copy()
    keep = [c for c in ["kappa_mean","runtime_mean","peak_system_memory_mean_mb"] if c in df.columns]
    g = df.groupby(["scenario","tool","tool_category"])[keep].mean(numeric_only=True).reset_index()
    return g.merge(profiles, on="scenario", how="left").dropna(subset=["kappa_mean"])

P4 = prep(load_p4(), P4_PROFILES)
P5 = prep(load_p5(), P5_PROFILES)

paradigms = ["Marker-Based","Similarity-Based","Classic ML","Deep Learning","Semi-Supervised"]

def eta_typeIII(d, response):
    d = d[d[response].notna() & np.isfinite(d[response])].copy()
    if d.shape[0] < 6 or d["log_cells"].nunique() < 2: return None
    try:
        fit = ols(f"{response} ~ log_cells + n_types + shannon + knn_purity", data=d).fit()
        a3 = sm.stats.anova_lm(fit, typ=3)
        a1 = sm.stats.anova_lm(fit, typ=1)
    except: return None
    ss_total_1 = a1["sum_sq"].sum()
    # Type III: drop Intercept from interpretation; total SS for proportion uses Type I total
    out = []
    for f in ["log_cells","n_types","shannon","knn_purity"]:
        out.append({"factor": f,
                    "eta_I": a1.loc[f, "sum_sq"] / ss_total_1,
                    "eta_III": a3.loc[f, "sum_sq"] / ss_total_1,
                    "p_I": a1.loc[f, "PR(>F)"],
                    "p_III": a3.loc[f, "PR(>F)"]})
    return pd.DataFrame(out)

print("=== η² (κ) per paradigm — Type I vs Type III ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    print(f"\n-- {arm} arm --")
    for par in paradigms:
        d = d_all[d_all["tool_category"] == par]
        r = eta_typeIII(d, "kappa_mean")
        if r is not None:
            print(f"  {par}:")
            for _, row in r.iterrows():
                print(f"    {row.factor:12s}: η²_I={row.eta_I:.3f} (p={row.p_I:.4f})  η²_III={row.eta_III:.3f} (p={row.p_III:.4f})")

# Pooled η² across paradigms
print("\n=== Pooled η² (κ) — Type I vs Type III ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    r = eta_typeIII(d_all, "kappa_mean")
    if r is not None:
        print(f"\n-- {arm} pooled --")
        for _, row in r.iterrows():
            print(f"  {row.factor:12s}: η²_I={row.eta_I:.3f} (p={row.p_I:.4f})  η²_III={row.eta_III:.3f} (p={row.p_III:.4f})")
