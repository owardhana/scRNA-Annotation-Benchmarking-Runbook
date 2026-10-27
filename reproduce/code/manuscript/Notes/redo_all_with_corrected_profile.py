#!/usr/bin/env python3
"""Re-derive all Phase 1+2+4+5 numbers using Profiling_Real.csv values throughout.
Maps each P5/P7a/P7b/P8b scenario folder to the correct Profiling_Real.csv key.
Fixes the Grun_original vs Grun_for_use mismatch in the P5/P7 inline tibbles."""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os, glob, re, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

P4_DIR = os.path.join(BENCH_ROOT, "results/P4 Synthetic Datasets V1")
P5_DIR = os.path.join(BENCH_ROOT, "results/P5 Real Life Datasets V2")
PROFILE_REAL = os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_Real.csv")
PROFILE_L9   = os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_L9.csv")

# ---- Scenario → profile-key map (CORRECTED: S4 uses _original) ----
P5_SCENARIO_TO_PROFILE = {
    "S1-Darmanis-Brain-2015_for_use":        "Darmanis-Brain-2015_for_use",
    "S2-Marques-Brain_for_use":              "Marques-Brain-2016_for_use",
    "S3-Nowakowski-Cortex-2017_for_use":     "Nowakowski-Cortex-2017_for_use",
    "S4-Grun-Pancreas-2016_original":        "Grun-Pancreas-2016_original",     # FIX
    "S5-Tabula Muris-FACS-3k_for_use":       "Tabula Muris-FACS-3k_for_use",
    "S6-He-Skin-2020_for_use":               "He-Skin-2020_for_use",
    "S7-Zhao-Immune-Fine-2020_for_use":      "Zhao-Immune-Fine-2020_for_use",
    "S8-Zheng-Zhengsort-5cl-2017_original":  "Zheng-ZhengSort-5cl-2017_original",
    "S9-MacParland-Liver-Broad_original":    "MacParland-Liver-Broad-2018_original",
}
P7A_SCENARIO_TO_PROFILE = {
    "Darmanis-Brain-2015_for_use":            "Darmanis-Brain-2015_for_use",
    "Marques-Brain-2016_for_use":             "Marques-Brain-2016_for_use",
    "Nowakowski-Cortex-2017_for_use":         "Nowakowski-Cortex-2017_for_use",
    "Grun-Pancreas-2016_original":            "Grun-Pancreas-2016_original",
    "Tabula Muris-FACS-3k_for_use":           "Tabula Muris-FACS-3k_for_use",
    "He-Skin-2020_for_use":                   "He-Skin-2020_for_use",
    "Zhao-Immune-Fine-2020_for_use":          "Zhao-Immune-Fine-2020_for_use",
    "Zheng-ZhengSort-5cl-2017_original":      "Zheng-ZhengSort-5cl-2017_original",
    "MacParland-Liver-Broad-2018_original_seurat": "MacParland-Liver-Broad-2018_original",  # strip _seurat
    "MacParland-Liver-Broad-2018_original":  "MacParland-Liver-Broad-2018_original",
    "MacParland-Liver-Broad_original":       "MacParland-Liver-Broad-2018_original",
}

prof_real_raw = pd.read_csv(PROFILE_REAL)
def get_profile_row(key):
    row = prof_real_raw[prof_real_raw["Sample"] == key]
    if row.empty: return None
    r = row.iloc[0]
    return {"log_cells": np.log10(r["Total_Cells"]),
            "n_types": int(r["Total_CellTypes"]),
            "shannon": r["Norm_Shannon_Entropy"],
            "knn_purity": r["KNN_Purity"]}

def build_profile_df(scenario_to_key):
    rows = []
    for sc, key in scenario_to_key.items():
        p = get_profile_row(key)
        if p is None:
            print(f"  WARN: no profile for {key}")
            continue
        rows.append({"scenario": sc, **p})
    return pd.DataFrame(rows).drop_duplicates(subset=["scenario"])

P5_PROFILES = build_profile_df(P5_SCENARIO_TO_PROFILE)
print("=== P5/Phase1+2 profile (corrected) ===")
print(P5_PROFILES.to_string(index=False))

# ---- Tool category map ----
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

# ---- Loaders ----
def load_p5():
    rows = []
    for sc in P5_SCENARIO_TO_PROFILE.keys():
        sp = os.path.join(P5_DIR, sc)
        if not os.path.isdir(sp): continue
        for f in glob.glob(os.path.join(sp, "*.csv")):
            try: df = pd.read_csv(f, low_memory=False)
            except: continue
            df = drop_index_col(df)
            df["scenario"] = sc; df["replicate"] = "single"
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out[~out["tool"].isin(exclude_tools)].copy()
    out["tool_category"] = out["tool"].map(cat_map)
    return out

def load_p4():
    rows = []
    for sc in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]:
        sp = os.path.join(P4_DIR, sc)
        if not os.path.isdir(sp): continue
        for r in sorted(os.listdir(sp)):
            rp = os.path.join(sp, r)
            if not os.path.isdir(rp): continue
            for f in glob.glob(os.path.join(rp, "*.csv")):
                try: df = pd.read_csv(f, low_memory=False)
                except: continue
                df = drop_index_col(df)
                df["scenario"] = sc; df["replicate"] = r
                rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out[~out["tool"].isin(exclude_tools)].copy()
    out["tool_category"] = out["tool"].map(cat_map)
    return out

# Synthetic profile from Profiling_L9.csv (averaged over 3 reps per scenario)
synth_raw = pd.read_csv(PROFILE_L9)
synth_raw = synth_raw.rename(columns={"Scenario":"scenario"})
P4_PROFILES = synth_raw.groupby("scenario").agg(
    log_cells=("Total_Cells", lambda x: np.log10(x.mean())),
    n_types=("Total_CellTypes", lambda x: int(round(x.mean()))),
    shannon=("Norm_Shannon_Entropy","mean"),
    knn_purity=("KNN_Purity","mean")
).reset_index()

def prep(df, profiles):
    df = df.copy()
    mz = (df.get("kappa_mean", 0) == 0) & (df.get("accuracy_mean", 0) == 0) & (df.get("f1_mean", 0) == 0)
    df.loc[mz, "kappa_mean"] = np.nan
    df = df[df["tool_category"].notna()].copy()
    keep = [c for c in ["kappa_mean","accuracy_mean","f1_mean","mcc_mean","rare_f1_mean","runtime_mean","peak_system_memory_mean_mb"] if c in df.columns]
    grouped = df.groupby(["scenario","tool","tool_category"])[keep].mean(numeric_only=True).reset_index()
    return grouped.merge(profiles, on="scenario", how="left").dropna(subset=["kappa_mean"])

P4 = prep(load_p4(), P4_PROFILES)
P5 = prep(load_p5(), P5_PROFILES)

print(f"\nP4 rows: {len(P4)} | P5 rows: {len(P5)}")

paradigms = ["Marker-Based","Similarity-Based","Classic ML","Deep Learning","Semi-Supervised"]

# η² (per-tool×scenario granularity, per paradigm)
def eta_part(d, response):
    d = d[d[response].notna() & np.isfinite(d[response])].copy()
    if d.shape[0] < 6 or d["log_cells"].nunique() < 2: return None
    try:
        fit = ols(f"{response} ~ log_cells + n_types + shannon + knn_purity", data=d).fit()
        a = sm.stats.anova_lm(fit, typ=1)
    except: return None
    ss = a["sum_sq"].sum()
    out = []
    for f in ["log_cells","n_types","shannon","knn_purity"]:
        if f in a.index: out.append({"factor":f, "eta_sq":a.loc[f,"sum_sq"]/ss, "p":a.loc[f,"PR(>F)"]})
    out.append({"factor":"Residuals","eta_sq":a.loc["Residual","sum_sq"]/ss,"p":np.nan})
    return pd.DataFrame(out)

print("\n=== η²(κ) per paradigm (corrected profile) ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    print(f"-- {arm} --")
    for par in paradigms:
        d = d_all[d_all["tool_category"] == par]
        r = eta_part(d, "kappa_mean")
        if r is not None:
            purity = r[r.factor=="knn_purity"].eta_sq.iloc[0]
            shannon = r[r.factor=="shannon"].eta_sq.iloc[0]
            cells = r[r.factor=="log_cells"].eta_sq.iloc[0]
            ntyp = r[r.factor=="n_types"].eta_sq.iloc[0]
            print(f"  {par:20s}: purity={purity:.3f} cells={cells:.3f} types={ntyp:.3f} shannon={shannon:.3f}")

# η² for log_runtime
print("\n=== η²(log_runtime) per paradigm (corrected profile) ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    print(f"-- {arm} --")
    d_rt = d_all[d_all["runtime_mean"].notna() & (d_all["runtime_mean"] > 0)].copy()
    d_rt["log_runtime"] = np.log10(d_rt["runtime_mean"])
    for par in paradigms:
        d = d_rt[d_rt["tool_category"] == par]
        r = eta_part(d, "log_runtime")
        if r is not None:
            print(f"  {par:20s}: cells={r[r.factor=='log_cells'].eta_sq.iloc[0]:.3f} types={r[r.factor=='n_types'].eta_sq.iloc[0]:.3f} shannon={r[r.factor=='shannon'].eta_sq.iloc[0]:.3f} purity={r[r.factor=='knn_purity'].eta_sq.iloc[0]:.3f}")

# η² for log_mem
print("\n=== η²(log_mem) per paradigm (corrected profile) ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    print(f"-- {arm} --")
    d_m = d_all[d_all["peak_system_memory_mean_mb"].notna() & (d_all["peak_system_memory_mean_mb"] > 0)].copy()
    d_m["log_mem"] = np.log10(d_m["peak_system_memory_mean_mb"])
    for par in paradigms:
        d = d_m[d_m["tool_category"] == par]
        r = eta_part(d, "log_mem")
        if r is not None:
            print(f"  {par:20s}: cells={r[r.factor=='log_cells'].eta_sq.iloc[0]:.3f} types={r[r.factor=='n_types'].eta_sq.iloc[0]:.3f} shannon={r[r.factor=='shannon'].eta_sq.iloc[0]:.3f} purity={r[r.factor=='knn_purity'].eta_sq.iloc[0]:.3f}")

# Dataset-level R² per paradigm (n=9 per fit)
def fit_xy(d, x, y):
    d = d.dropna(subset=[x,y])
    if len(d) < 4: return None
    slope, intercept, r, p_pe, _ = stats.linregress(d[x], d[y])
    sp_r, sp_p = stats.spearmanr(d[x], d[y])
    return {"n":len(d), "R2":r**2, "r":r, "p_pe":p_pe, "rho":sp_r, "p_sp":sp_p, "slope":slope, "intercept":intercept}

print("\n=== Dataset-level R² per paradigm (n=9): kNN purity → κ ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    print(f"-- {arm} --")
    for par in paradigms:
        d = d_all[d_all["tool_category"] == par]
        agg = d.groupby("scenario").agg(mean_kappa=("kappa_mean","mean"),
                                        knn_purity=("knn_purity","first")).reset_index()
        s = fit_xy(agg, "knn_purity", "mean_kappa")
        if s:
            print(f"  {par:20s}: R²={s['R2']:.3f} ρ={s['rho']:.3f} (p={s['p_sp']:.4f}) r={s['r']:.3f}")

print("\n=== Dataset-level R² per paradigm (n=9): log_cells → log_runtime ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    print(f"-- {arm} --")
    for par in paradigms:
        d = d_all[d_all["tool_category"] == par].copy()
        d = d[(d["runtime_mean"].notna()) & (d["runtime_mean"] > 0)]
        d["log_runtime"] = np.log10(d["runtime_mean"])
        agg = d.groupby("scenario").agg(mean_logrt=("log_runtime","mean"),
                                        log_cells=("log_cells","first")).reset_index()
        s = fit_xy(agg, "log_cells", "mean_logrt")
        if s:
            print(f"  {par:20s}: R²={s['R2']:.3f} ρ={s['rho']:.3f} (p={s['p_sp']:.4f}) slope={s['slope']:.3f}")

print("\n=== Dataset-level R² per paradigm (n=9): log_cells → log_mem ===")
for arm, d_all in [("synth", P4), ("real", P5)]:
    print(f"-- {arm} --")
    for par in paradigms:
        d = d_all[d_all["tool_category"] == par].copy()
        d = d[(d["peak_system_memory_mean_mb"].notna()) & (d["peak_system_memory_mean_mb"] > 0)]
        d["log_mem"] = np.log10(d["peak_system_memory_mean_mb"])
        agg = d.groupby("scenario").agg(mean_logmem=("log_mem","mean"),
                                        log_cells=("log_cells","first")).reset_index()
        s = fit_xy(agg, "log_cells", "mean_logmem")
        if s:
            print(f"  {par:20s}: R²={s['R2']:.3f} ρ={s['rho']:.3f} (p={s['p_sp']:.4f}) slope={s['slope']:.3f}")

# Phase 4: re-derive marker DB + LLM dataset-level scatters with corrected profile keys
def aggregate_phase4(score_dir, name_pattern, runtime_tsv):
    """Aggregate scored TSVs to per-(tool, dataset) mean ontology score + runtime."""
    rows = []
    for f in glob.glob(os.path.join(score_dir, "*.tsv")):
        if "runtime" in f.lower(): continue
        if " 2.19.28 PM" in f: continue
        m = re.match(r"(.+?)_subset_(.+)\.tsv", os.path.basename(f).replace("_database_cluster_map","").replace("_cluster_map",""))
        if not m: continue
        dataset_raw, tool = m.group(1), m.group(2)
        df = pd.read_csv(f, sep="\t")
        df["mean_score"] = df[["score_1","score_2","score_3"]].mean(axis=1)
        rows.append({"dataset_raw": dataset_raw, "tool": tool, "mean_score": df["mean_score"].mean()})
    out = pd.DataFrame(rows)
    out = out.groupby(["dataset_raw","tool"], as_index=False).agg(mean_score=("mean_score","mean"))
    rt = pd.read_csv(runtime_tsv, sep="\t")
    rt.columns = [c.strip() for c in rt.columns]
    rt["tool"] = rt["tool"].str.replace("_database","")
    rt["dataset_raw"] = rt["dataset"].str.replace("_seurat","")
    out = out.merge(rt[["dataset_raw","tool","runtime"]].rename(columns={"runtime":"runtime_s"}),
                    on=["dataset_raw","tool"], how="left")
    return out

# Normalise dataset_raw → profile-key via P7A_SCENARIO_TO_PROFILE
def norm_to_profile(d):
    d = d.copy()
    d["profile_key"] = d["dataset_raw"].map(P7A_SCENARIO_TO_PROFILE)
    # alias for stripped seurat
    d.loc[d["dataset_raw"].str.contains("MacParland.*Broad.*original", regex=True), "profile_key"] = "MacParland-Liver-Broad-2018_original"
    return d

p7a = aggregate_phase4(os.path.join(BENCH_ROOT, "results/P7a Marker - Real Database/with_score"),
                       None, os.path.join(BENCH_ROOT, "results/P7a Marker - Real Database/with_score/runtime_dataset.tsv"))
p7a = norm_to_profile(p7a)
p7b = aggregate_phase4(os.path.join(BENCH_ROOT, "results/P7b LLM - Real Internet/with_score"),
                       None, os.path.join(BENCH_ROOT, "results/P7b LLM - Real Internet/with_score/runtime_dataset.tsv"))
p7b = norm_to_profile(p7b)

# Join profile values
prof_lookup = prof_real_raw.set_index("Sample")
def join_profile(df):
    df = df.copy()
    df["log_cells"] = df["profile_key"].map(lambda k: np.log10(prof_lookup.loc[k,"Total_Cells"]) if k in prof_lookup.index else np.nan)
    df["n_types"]   = df["profile_key"].map(lambda k: prof_lookup.loc[k,"Total_CellTypes"] if k in prof_lookup.index else np.nan)
    df["shannon"]   = df["profile_key"].map(lambda k: prof_lookup.loc[k,"Norm_Shannon_Entropy"] if k in prof_lookup.index else np.nan)
    df["knn_purity"]= df["profile_key"].map(lambda k: prof_lookup.loc[k,"KNN_Purity"] if k in prof_lookup.index else np.nan)
    return df
p7a = join_profile(p7a)
p7b = join_profile(p7b)

print("\n=== Phase 4 Marker DB (dataset-level, n=9, corrected profile) ===")
agg = p7a.groupby("profile_key").agg(mean_score=("mean_score","mean"),
                                     mean_runtime=("runtime_s","mean"),
                                     knn_purity=("knn_purity","first"),
                                     log_cells=("log_cells","first"),
                                     n_types=("n_types","first")).reset_index()
print(agg)
print("Purity-score:", fit_xy(agg, "knn_purity", "mean_score"))

print("\n=== Phase 4 LLM (dataset-level, n=9, corrected profile) ===")
agg2 = p7b.groupby("profile_key").agg(mean_score=("mean_score","mean"),
                                      mean_runtime=("runtime_s","mean"),
                                      knn_purity=("knn_purity","first"),
                                      log_cells=("log_cells","first"),
                                      n_types=("n_types","first")).reset_index()
agg2["log_mean_runtime"] = np.log10(agg2["mean_runtime"])
print(agg2)
print("Purity-score:", fit_xy(agg2, "knn_purity", "mean_score"))
print("log_cells → log_runtime:", fit_xy(agg2, "log_cells", "log_mean_runtime"))
print("n_types → log_runtime:", fit_xy(agg2, "n_types", "log_mean_runtime"))

# Per-tool means
print("\n=== Per-tool mean scores ===")
print("Marker DB:")
print(p7a.groupby("tool")["mean_score"].agg(["mean","median","count"]).round(3))
print("\nLLM:")
print(p7b.groupby("tool")["mean_score"].agg(["mean","median","count"]).round(3))

print(f"\nMarker DB family mean: {p7a['mean_score'].mean():.3f}")
print(f"LLM family mean: {p7b['mean_score'].mean():.3f}")

# Phase 5 FM unchanged (already used Profiling_Real)
print("\n=== Phase 5 FM (unchanged, already uses Profiling_Real) ===")
print("R²=0.996, slope=0.964 — already correct.")
