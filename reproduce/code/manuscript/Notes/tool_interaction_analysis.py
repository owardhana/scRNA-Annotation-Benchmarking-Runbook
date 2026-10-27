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
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os, glob, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.simplefilter("ignore")
pd.set_option("display.width", 220)

BASE = BENCH_ROOT
P4_DIR = f"{BASE}/results/P4 Synthetic Datasets V1"
P5_DIR = f"{BASE}/results/P5 Real Life Datasets V2"
OUT_DIR = f"{BASE}/results/_interaction_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

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
            "macro_f1","rare_f1","runtime_s","peak_mem_mb","knn_purity","log_cells","n_types","shannon"]
    d = d.rename(columns={"f1_mean":"macro_f1","rare_f1_mean":"rare_f1","runtime_mean":"runtime_s",
                          "peak_system_memory_mean_mb":"peak_mem_mb"})
    return d[keep]

# =====================================================================================
#  T0.1 — assemble tidy table
# =====================================================================================
synth = assemble(load_synth(), "synth", SYNTH_PROFILES)
real  = assemble(load_real(),  "real",  REAL_PROFILES)
tidy  = pd.concat([synth, real], ignore_index=True)
tidy.to_csv(f"{OUT_DIR}/tidy_kappa.csv", index=False)

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

out("="*90)
out("TOOL-TERM & INTERACTION ANALYSIS  (comments 1b / 1c)")
out("="*90)

# ---- roster / failure summary
for arm, d in [("synth", synth), ("real", real)]:
    ncell = d.groupby(["scenario","tool"]).ngroups
    nfail = (d["run_status"] == "failed_zero").sum()
    out(f"\n[{arm}] rows={len(d)}  tools={d.tool.nunique()}  scenarios={d.scenario.nunique()}  "
        f"seeds/scen={sorted(d.groupby('scenario').seed.nunique().unique())}  "
        f"(tool,scenario) cells={ncell}  failures(all-zero rows)={nfail}")
    out(f"       tools per paradigm: {d.groupby('tool_category').tool.nunique().to_dict()}")
    fc = d[d.run_status=='failed_zero'].groupby('scenario').tool.nunique()
    if len(fc): out(f"       failures concentrated in: {fc.to_dict()}")

# seed-count matrix (synth, OK seeds)
okm = (synth[synth.run_status=="ok"]
       .pivot_table(index="tool", columns="scenario", values="seed", aggfunc="nunique", fill_value=0))
okm.to_csv(f"{OUT_DIR}/seed_count_matrix.csv")

# -------------------------------------------------------------------------------------
#  T0.2 — pooled Type III eta^2, with vs without the tool term
# -------------------------------------------------------------------------------------
out("\n" + "-"*90)
out("T0.2  POOLED Type III eta^2  (per-(tool,scenario) means; denominator = total Type I SS)")
out("-"*90)
FML_BASE = "kappa ~ knn_purity + log_cells + n_types + shannon"
FML_TOOL = "kappa ~ knn_purity + log_cells + n_types + shannon + C(tool)"
FML_INT  = "kappa ~ knn_purity + log_cells + n_types + shannon + C(tool) + knn_purity:C(tool)"

t02 = {}
for arm, d_arm in [("synth", synth), ("real", real)]:
    g = scenario_tool_means(d_arm, "kappa")          # complete-case means
    n_obs = len(g)
    # (a) baseline 4-proxy model (should reproduce published pooled purity eta^2)
    eb, pb, _, _ = type3_eta(g, FML_BASE, ["knn_purity","log_cells","n_types","shannon"])
    # (b) + tool term
    et, ptt, _, _ = type3_eta(g, FML_TOOL, ["knn_purity","C(tool)"])
    # (c) + tool + purity:tool interaction  (figure bar; df-inflated, not the text anchor)
    ei, pii, fit_i, _ = type3_eta(g, FML_INT, ["knn_purity","C(tool)","knn_purity:C(tool)"])
    t02[arm] = dict(n_obs=n_obs, purity_base=eb["knn_purity"], purity_tool=et["knn_purity"],
                    tool=et["C(tool)"], tool_p=ptt["C(tool)"],
                    purity_int=ei["knn_purity"], toolmain_int=ei["C(tool)"],
                    inter=ei["knn_purity:C(tool)"], inter_p=pii["knn_purity:C(tool)"])
    m = t02[arm]
    out(f"\n[{arm}]  n(tool,scenario means) = {n_obs}")
    out(f"   baseline (no tool): eta^2_purity = {m['purity_base']:.3f}")
    out(f"   +tool term        : eta^2_purity = {m['purity_tool']:.3f}   "
        f"eta^2_tool = {m['tool']:.3f} (p={m['tool_p']:.2g})")
    out(f"   +purity:tool [FIG] : eta^2_purity = {m['purity_int']:.3f}   eta^2_tool = {m['toolmain_int']:.3f}   "
        f"eta^2_purity:tool = {m['inter']:.3f} (p={m['inter_p']:.2g})")
    out(f"   -> cascade on eta^2_purity when tool added: {m['purity_base']:.3f} -> {m['purity_tool']:.3f} "
        f"(delta {m['purity_tool']-m['purity_base']:+.3f})")

# per-paradigm baseline purity eta^2 (sanity vs published 0.58-0.78 synth / 0.29-0.78 real)
out("\n   per-paradigm baseline eta^2_purity (sanity check vs Fig 4a):")
for arm, d_arm in [("synth", synth), ("real", real)]:
    g = scenario_tool_means(d_arm, "kappa")
    vals = {}
    for par in PARADIGMS:
        gp = g[g.tool_category == par]
        if gp.tool.nunique() < 2 or gp["knn_purity"].nunique() < 2 or len(gp) < 6:
            continue
        e, _, _, _ = type3_eta(gp, FML_BASE, ["knn_purity"])
        vals[par] = e["knn_purity"]
    rng = f"{min(vals.values()):.2f}-{max(vals.values()):.2f}" if vals else "n/a"
    out(f"     [{arm}] " + "  ".join(f"{k.split('-')[0][:4]}={v:.3f}" for k,v in vals.items()) + f"   range {rng}")

# -------------------------------------------------------------------------------------
#  T0.3 — Kendall's W (both missing-data conventions) + Friedman + cross-arm Spearman
# -------------------------------------------------------------------------------------
out("\n" + "-"*90)
out("T0.3  Kendall's W over tool rankings across 9 scenarios (tool-level, per arm)")
out("-"*90)

def kendall_w(mat):
    """mat: rows = scenarios (blocks), cols = tools. Higher kappa = better. Returns W, chi2, df, p."""
    N, k = mat.shape
    # rank tools within each scenario (1 = best). average ties.
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r, method="average"), 1, mat)
    Rj = ranks.sum(axis=0)
    S = ((Rj - Rj.mean())**2).sum()
    # tie correction
    T = 0.0
    for i in range(N):
        _, cnt = np.unique(ranks[i], return_counts=True)
        T += (cnt**3 - cnt).sum()
    denom = (N**2 * (k**3 - k) - N*T)
    W = 12*S/denom if denom > 0 else np.nan
    chi2 = N*(k-1)*W
    df = k-1
    p = stats.chi2.sf(chi2, df)
    return W, chi2, df, p, N, k

wide = {}  # arm -> scenario x tool (mean kappa)
for arm, d_arm in [("synth", synth), ("real", real)]:
    g = scenario_tool_means(d_arm, "kappa")
    wide[arm] = g.pivot_table(index="scenario", columns="tool", values="kappa")

t03 = {}
for arm in ["synth","real"]:
    w = wide[arm]
    # convention 1: complete tools only (no NA across scenarios)
    comp = w.dropna(axis=1, how="any")
    W1, c1, df1, p1, N1, k1 = kendall_w(comp.values)
    dropped = w.shape[1] - comp.shape[1]
    # convention 2: all tools, failures scored last-place -> impute worst per scenario (min - eps)
    w2 = w.copy()
    for sc in w2.index:
        row = w2.loc[sc]
        floor = np.nanmin(row.values) - 1e-6 if np.isfinite(np.nanmin(row.values)) else 0.0
        w2.loc[sc] = row.fillna(floor)
    W2, c2, df2, p2, N2, k2 = kendall_w(w2.values)
    t03[arm] = dict(W1=W1,p1=p1,k1=k1,dropped=dropped,W2=W2,p2=p2,k2=k2)
    out(f"\n[{arm}]")
    out(f"   complete-tools ({k1} tools, {dropped} dropped): W = {W1:.3f}  chi2={c1:.1f} df={df1}  p={p1:.2g}")
    out(f"   failures=last-place ({k2} tools)              : W = {W2:.3f}  chi2={c2:.1f} df={df2}  p={p2:.2g}")

# cross-arm Spearman of tool mean-ranks (tools present in both arms, complete)
cs = set(wide["synth"].dropna(axis=1).columns) & set(wide["real"].dropna(axis=1).columns)
cs = sorted(cs)
if len(cs) >= 4:
    mr_s = wide["synth"][cs].apply(lambda c: c).mean(axis=0).rank(ascending=False)
    # mean rank across scenarios per tool
    def mean_rank(w, cols):
        r = np.apply_along_axis(lambda x: stats.rankdata(-x), 1, w[cols].values)
        return pd.Series(r.mean(axis=0), index=cols)
    ms = mean_rank(wide["synth"], cs); mr = mean_rank(wide["real"], cs)
    rho, prho = stats.spearmanr(ms, mr)
    out(f"\n   cross-arm Spearman of tool mean-ranks ({len(cs)} shared complete tools): "
        f"rho = {rho:.3f} (p={prho:.2g})")
    t03["cross_arm_rho"] = rho; t03["cross_arm_p"] = prho; t03["cross_arm_n"] = len(cs)

# -------------------------------------------------------------------------------------
#  VC — seed-level crossed two-way ANOVA (synthetic) -> method-of-moments components
# -------------------------------------------------------------------------------------
out("\n" + "-"*90)
out("VC  Interaction anchor: kappa ~ scenario * tool  (synthetic, seed-level, MoM components)")
out("-"*90)

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

# primary: kappa=0-imputed (balanced 47x9x3)
vc_primary = mom_components(synth, "kappa_imp0", "PRIMARY: kappa=0-imputed (all 47 tools)")
# sensitivity: complete-case = tools fully observed (all 9 scenarios x 3 ok seeds)
fully = (okm == 3).all(axis=1)
comp_tools = list(okm.index[fully])
sens = synth[(synth.tool.isin(comp_tools)) & (synth.run_status=="ok")]
vc_sens = mom_components(sens, "kappa", f"SENSITIVITY: complete-case ({len(comp_tools)} fully-observed tools)")

# =====================================================================================
#  headline summary for prose
# =====================================================================================
out("\n" + "="*90)
out("HEADLINE NUMBERS FOR THE MANUSCRIPT")
out("="*90)
out(f"Pooled eta^2 (synth): purity={t02['synth']['purity_tool']:.3f}  tool={t02['synth']['tool']:.3f}  "
    f"purity:tool={t02['synth']['inter']:.3f}")
out(f"Pooled eta^2 (real) : purity={t02['real']['purity_tool']:.3f}  tool={t02['real']['tool']:.3f}  "
    f"purity:tool={t02['real']['inter']:.3f}")
out(f"VC synth (imputed)  : scenario={vc_primary['pct']['scenario']:.1f}%  tool={vc_primary['pct']['tool']:.1f}%  "
    f"tool:scenario={vc_primary['pct']['tool:scenario']:.1f}%  residual={vc_primary['pct']['residual']:.1f}%  "
    f"(interaction F={vc_primary['F_AB']:.2f}, p={vc_primary['p_AB']:.2g})")
out(f"VC synth (complete) : scenario={vc_sens['pct']['scenario']:.1f}%  tool={vc_sens['pct']['tool']:.1f}%  "
    f"tool:scenario={vc_sens['pct']['tool:scenario']:.1f}%  residual={vc_sens['pct']['residual']:.1f}%  "
    f"(interaction F={vc_sens['F_AB']:.2f}, p={vc_sens['p_AB']:.2g})")
out(f"Kendall's W synth   : complete={t03['synth']['W1']:.3f} (p={t03['synth']['p1']:.2g}, k={t03['synth']['k1']}) | "
    f"fail=last {t03['synth']['W2']:.3f} (p={t03['synth']['p2']:.2g}, k={t03['synth']['k2']})")
out(f"Kendall's W real    : complete={t03['real']['W1']:.3f} (p={t03['real']['p1']:.2g}, k={t03['real']['k1']}) | "
    f"fail=last {t03['real']['W2']:.3f} (p={t03['real']['p2']:.2g}, k={t03['real']['k2']})")
if "cross_arm_rho" in t03:
    out(f"Cross-arm rank rho  : {t03['cross_arm_rho']:.3f} (p={t03['cross_arm_p']:.2g}, n={t03['cross_arm_n']} shared tools)")

with open(f"{OUT_DIR}/RESULTS.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
# also dump the figure-bar values for the eta^2 panel
pd.DataFrame([
    {"arm":arm, "term":term, "eta2":val}
    for arm in ["synth","real"]
    for term,val in [("knn_purity",t02[arm]["purity_int"]),
                     ("tool",t02[arm]["toolmain_int"]),
                     ("purity:tool",t02[arm]["inter"])]
]).to_csv(f"{OUT_DIR}/figure_bar_values.csv", index=False)
print(f"\nWrote {OUT_DIR}/RESULTS.txt, tidy_kappa.csv, seed_count_matrix.csv, figure_bar_values.csv")
