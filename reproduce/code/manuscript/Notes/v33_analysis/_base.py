#!/usr/bin/env python3
"""Tool-term and tool x scenario interaction analysis (PI revision, comments 1b/1c).

Companion to `type3_eta_squared.py`. Same data sources, tool roster, exclusions, failure
heuristic and Type III denominator (SS_typeIII / SS_total_typeI) so every number is directly
comparable to what is already in the manuscript (Fig 4a / Supplementary Fig 2).

Produces:
  results/_interaction_analysis/tidy_kappa.csv        one row per (arm, scenario, seed, tool)
  results/_interaction_analysis/seed_count_matrix.csv (tool x scenario) OK-seed counts, synth
  results/_interaction_analysis/RESULTS.txt           all new numbers for the manuscript

Analyses
  T0.2  Pooled Type III eta^2 with a tool term added:
        kappa ~ purity + log_cells + n_types + shannon + tool + purity:tool
        -> eta^2_purity (with/without tool), eta^2_tool, eta^2_purity:tool  (both arms).
  T0.3  Kendall's W over tool rankings across the 9 scenarios, per arm, under two missing-data
        conventions (complete-tools-only; failures = last-place ties), + Friedman chi^2/p, +
        Spearman of tool mean-ranks between arms.
  VC    Interaction anchor. Crossed two-way kappa ~ scenario * tool on the synthetic SEED-LEVEL
        data (3 seeds = the only replication -> MS_residual is a clean seed-noise floor).
        Method-of-moments variance components from balanced ANOVA mean squares:
            sigma^2_e     = MS_res
            sigma^2_tool:scen = (MS_AB - MS_res)/n
            sigma^2_scen  = (MS_A  - MS_AB)/(n*b)
            sigma^2_tool  = (MS_B  - MS_AB)/(n*a)
        Verdict on comment 1c = sigma^2_tool:scen vs sigma^2_e  (F = MS_AB/MS_res ~ 1 -> noise).
        Primary = kappa=0-imputed (balanced 47x9x3); sensitivity = complete-case (balanced).
"""

import os, glob, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.simplefilter("ignore")
pd.set_option("display.width", 220)

import sys
BASE = sys.argv[1]
P4_DIR = f"{BASE}/results/P4 Synthetic Datasets V1"
P5_DIR = f"{BASE}/results/P5 Real Life Datasets V2"
OUT_DIR = f"{BASE}/results/_interaction_analysis"
OUT_DIR = sys.argv[2]; os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- tool roster (identical to R)
marker_tools      = ["clustifyr_hyper","clustifyr_jaccard","scCATCH","SCINA","ScInfeR","SCSA","scSorter","scType"]
similarity_tools = ["CHETAH","CIPR","scmap_cell","scmap_cluster","Seurat_Transfer_CCA","Seurat_Transfer_PCA","Seurat_Transfer_RPCA","SingleR"]
classicML_tools   = ["CaSTLe","CellTypist","scAnnotate","scAnnotatR","scClassify","scibetR","scID",
    "scPred","scPred_adaboost","scPred_avNNet","scPred_bayesglm","scPred_earth","scPred_glm",
    "scPred_glmboost","scPred_knn","scPred_lda","scPred_mlp","scPred_multinom","scPred_nb",
    "scPred_nnet","scPred_regLogistic","scPred_rf","scPred_svmLinear","scPred_svmPoly","scPred_xgbTree","singleCellNet","scLearn"]
DL_tools          = ["ACTINN","Cell_BLAST","CIForm","NeuCA","scBalance","scDeepSort","scHash","scMMT","TOSICA"]
semi_tools        = ["CALLR","CAMLU","MARS","SCANVI","scArches","scGAD","scnym","scSemiCluster","scSemiGAN"]

cat_map = {}
for t in marker_tools:      cat_map[t] = "Marker-Based"
for t in similarity_tools: cat_map[t] = "Similarity-Based"
for t in classicML_tools:   cat_map[t] = "Classic ML"
for t in DL_tools:          cat_map[t] = "Deep Learning"
for t in semi_tools:        cat_map[t] = "Semi-Supervised"

# 13 exclusions, identical to Plots_Analysis.R / type3_eta_squared.py
exclude_tools = {"ScInfeR","scPred_adaboost","scAnnotatR","CIPR","scmap_cluster","clustifyr_hyper",
                 "clustifyr_jaccard","SCSA","scCATCH","scType","scGAD","scDeepSort","CALLR"}

PARADIGMS = ["Marker-Based","Similarity-Based","Classic ML","Deep Learning","Semi-Supervised"]

# ---------------------------------------------------------------- scenario profiles
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

def build_synth_profiles():
    raw = pd.read_csv(f"{BASE}/Data Creation/P5 Real Life Sets V2/Profiling_L9.csv")
    raw = raw.rename(columns={"Scenario": "scenario"})
    return raw.groupby("scenario").agg(
        log_cells =("Total_Cells",           lambda x: np.log10(x.mean())),
        n_types   =("Total_CellTypes",        lambda x: int(round(x.mean()))),
        shannon   =("Norm_Shannon_Entropy",   "mean"),
        knn_purity=("KNN_Purity",             "mean"),
    ).reset_index()

def build_real_profiles():
    prof = pd.read_csv(f"{BASE}/Data Creation/P5 Real Life Sets V2/Profiling_Real.csv")
    rows = []
    for sc, key in P5_SCENARIO_TO_PROFILE.items():
        r = prof[prof["Sample"] == key].iloc[0]
        rows.append({"scenario": sc, "log_cells": np.log10(r["Total_Cells"]),
                     "n_types": int(r["Total_CellTypes"]), "shannon": r["Norm_Shannon_Entropy"],
                     "knn_purity": r["KNN_Purity"]})
    return pd.DataFrame(rows)

SYNTH_PROFILES = build_synth_profiles()
REAL_PROFILES  = build_real_profiles()

# ---------------------------------------------------------------- loading (mirrors R load_benchmark_results)
def drop_index_col(df):
    if df.shape[1] > 0 and str(df.columns[0]) in ("", "X", "Unnamed: 0"):
        return df.iloc[:, 1:]
    return df

METRIC_COLS = ["kappa_mean","accuracy_mean","f1_mean","rare_f1_mean","mcc_mean",
               "unassigned_mean","runtime_mean","peak_system_memory_mean_mb","successful_folds"]

def _read_dir(path, scenario, seed, out):
    for f in glob.glob(os.path.join(path, "*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        df = drop_index_col(df)
        if "tool" not in df.columns:
            continue
        for c in METRIC_COLS:
            if c not in df.columns:
                df[c] = np.nan
        sub = df[["tool"] + METRIC_COLS].copy()
        sub["scenario"] = scenario
        sub["seed"] = seed
        out.append(sub)

def load_synth():
    out = []
    for sc in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]:
        sp = os.path.join(P4_DIR, sc)
        for rep in sorted(os.listdir(sp)):
            rp = os.path.join(sp, rep)
            if os.path.isdir(rp):
                _read_dir(rp, sc, rep, out)
    return pd.concat(out, ignore_index=True)

def load_real():
    out = []
    for sc in P5_SCENARIO_TO_PROFILE.keys():
        sp = os.path.join(P5_DIR, sc)
        if os.path.isdir(sp):
            _read_dir(sp, sc, "single", out)
    return pd.concat(out, ignore_index=True)

def assemble(raw, arm, profiles):
    """Restrict to the 47-tool roster, classify failures, attach profiles. Keeps failures as rows."""
    d = raw[~raw["tool"].isin(exclude_tools)].copy()
    d["tool_category"] = d["tool"].map(cat_map)
    d = d[d["tool_category"].notna()].copy()
    # failure = all-three-zero placeholder row (verified clean: 0 legit-zero false positives)
    fail = (d["kappa_mean"] == 0) & (d["accuracy_mean"] == 0) & (d["f1_mean"] == 0)
    d["run_status"] = np.where(fail, "failed_zero", "ok")
    d["kappa"]      = np.where(fail, np.nan, d["kappa_mean"])   # complete-case: NaN on failure
    d["kappa_imp0"] = np.where(fail, 0.0,   d["kappa_mean"])    # sensitivity: 0 on failure
    d["arm"] = arm
    d = d.merge(profiles, on="scenario", how="left")
    keep = ["arm","scenario","seed","tool","tool_category","run_status","kappa","kappa_imp0",
            "macro_f1","accuracy","mcc","rare_f1","runtime_s","peak_mem_mb","knn_purity","log_cells","n_types","shannon"]
    d = d.rename(columns={"f1_mean":"macro_f1","accuracy_mean":"accuracy","mcc_mean":"mcc","rare_f1_mean":"rare_f1","runtime_mean":"runtime_s",
                          "peak_system_memory_mean_mb":"peak_mem_mb"})
    return d[keep]

# =====================================================================================
#  T0.1 — assemble tidy table
# =====================================================================================
synth = assemble(load_synth(), "synth", SYNTH_PROFILES)
real  = assemble(load_real(),  "real",  REAL_PROFILES)
# Align with figures/_shared/tidy_kappa.csv: one failed SCINA run on Nowakowski (S3) is
# absent from the raw result CSVs but present in the figure tidy file (failed = 0).
_k = ("S3-Nowakowski-Cortex-2017_for_use", "SCINA")
if not ((real.scenario == _k[0]) & (real.tool == _k[1])).any():
    _r = real[real.scenario == _k[0]].iloc[[0]].copy()
    _r["tool"], _r["tool_category"], _r["run_status"] = "SCINA", "Marker-Based", "failed_zero"
    for _c in ["kappa","macro_f1","accuracy","mcc","rare_f1","runtime_s","peak_mem_mb"]: _r[_c] = np.nan
    _r["kappa_imp0"] = 0.0
    real = pd.concat([real, _r], ignore_index=True)
tidy  = pd.concat([synth, real], ignore_index=True)


# per-(tool,scenario) means for the pooled Type III model (matches Fig 4a modelling unit)
def scenario_tool_means(d, kcol):
    g = (d.groupby(["scenario","tool","tool_category"], as_index=False)
           .agg(kappa=(kcol,"mean"), knn_purity=("knn_purity","first"),
                log_cells=("log_cells","first"), n_types=("n_types","first"),
                shannon=("shannon","first")))
    return g.dropna(subset=["kappa"])

# =====================================================================================
#  Type III eta^2 helpers (denominator = Type I total SS, matching type3_eta_squared.py)
# =====================================================================================
def type3_eta(df, formula, terms):
    fit = ols(formula, data=df).fit()
    a3  = sm.stats.anova_lm(fit, typ=3)
    a1  = sm.stats.anova_lm(fit, typ=1)
    ss_total = a1["sum_sq"].sum()
    return {t: a3.loc[t, "sum_sq"] / ss_total for t in terms}, \
           {t: a3.loc[t, "PR(>F)"] for t in terms}, fit, ss_total

# =====================================================================================
#  Analyses
# =====================================================================================
lines = []
def out(s=""):
    lines.append(s); print(s)
def mom_components(d, kcol, label):
    """Balanced two-way random-effects VC from ANOVA mean squares. A=scenario, B=tool, n=seeds."""
    dd = d.dropna(subset=[kcol]).copy()
    a = dd["scenario"].nunique(); b = dd["tool"].nunique()
    # require balance
    cell = dd.groupby(["scenario","tool"]).size()
    n = int(cell.mode().iloc[0])
    balanced = (cell.nunique() == 1)
    fit = ols(f"{kcol} ~ C(scenario)*C(tool)", data=dd).fit()
    aov = sm.stats.anova_lm(fit, typ=1)
    ss  = {"A":aov.loc["C(scenario)","sum_sq"], "B":aov.loc["C(tool)","sum_sq"],
           "AB":aov.loc["C(scenario):C(tool)","sum_sq"], "E":aov.loc["Residual","sum_sq"]}
    df  = {"A":aov.loc["C(scenario)","df"], "B":aov.loc["C(tool)","df"],
           "AB":aov.loc["C(scenario):C(tool)","df"], "E":aov.loc["Residual","df"]}
    ms  = {k: ss[k]/df[k] for k in ss}
    F_AB = ms["AB"]/ms["E"]; p_AB = stats.f.sf(F_AB, df["AB"], df["E"])
    F_B  = ms["B"]/ms["AB"]; p_B  = stats.f.sf(F_B,  df["B"],  df["AB"])
    # method-of-moments (a=scenario levels, b=tool levels, n=reps)
    s2_e  = ms["E"]
    s2_ab = (ms["AB"] - ms["E"]) / n
    s2_a  = (ms["A"]  - ms["AB"]) / (n*b)
    s2_b  = (ms["B"]  - ms["AB"]) / (n*a)
    comps = {"scenario":s2_a, "tool":s2_b, "tool:scenario":s2_ab, "residual":s2_e}
    pos   = {k: max(v,0) for k,v in comps.items()}
    tot   = sum(pos.values())
    pct   = {k: 100*v/tot for k,v in pos.items()}
    out(f"\n[{label}]  a(scenario)={a}  b(tool)={b}  n(seed)={n}  balanced={balanced}  N={len(dd)}")
    out(f"   MS: scen={ms['A']:.4f} tool={ms['B']:.4f} tool:scen={ms['AB']:.4f} resid={ms['E']:.5f}")
    out(f"   variance components (raw):  " + "  ".join(f"{k}={v:.5f}" for k,v in comps.items()))
    out(f"   variance components (% of positive total):")
    for k in ["scenario","tool","tool:scenario","residual"]:
        out(f"      {k:14s} {pct[k]:6.2f}%   (sigma^2 = {pos[k]:.5f})")
    out(f"   interaction F = MS_AB/MS_res = {F_AB:.3f}  (df {int(df['AB'])},{int(df['E'])})  p = {p_AB:.2g}")
    out(f"   tool F = MS_B/MS_AB       = {F_B:.3f}   (df {int(df['B'])},{int(df['AB'])})  p = {p_B:.2g}")
    out(f"   -> tool:scenario sigma^2 vs residual sigma^2 = {pos['tool:scenario']:.5f} vs {pos['residual']:.5f} "
        f"(ratio {pos['tool:scenario']/pos['residual']:.2f})")
    return dict(pct=pct, F_AB=F_AB, p_AB=p_AB, F_B=F_B, p_B=p_B, n=n, a=a, b=b, N=len(dd),
                s2=pos, s2_tool=pos['tool'], s2_scen=pos['scenario'])
