#!/usr/bin/env python3
"""Paradigm-level Kendall's W — companion to `tool_interaction_analysis.py` (T0.3).

Does NOT modify `tool_interaction_analysis.py` or `_interaction_analysis/RESULTS.txt`;
the V14 audit cites that file as the owning source for the verified P28 numbers.
Loader block, tool roster, 13 exclusions and the `kendall_w()` function are copied
verbatim from it so the two levels are computed identically.

Question: the manuscript reports Kendall's W over 45 TOOLS ranked across 9 scenarios
(W = 0.42 synth / 0.57 real, P28). What is the equivalent statistic over the 5 cell-level
PARADIGMS? At paradigm level N=9 blocks vs k=5 items, so the Friedman chi^2 approximation
is far better behaved than at k=45 -- a p-value is defensible (and a permutation test is
reported alongside so no asymptotic argument is needed).

Analyses
  GATE  Reproduce the ten published paradigm mean kappa values (P25) from the
        scenario x paradigm matrix. If these do not come back exactly, the matrix is
        wrong and every W below is meaningless -- the script refuses to continue.
  PW1   Kendall's W over 5 paradigms x 9 scenarios, per arm, RAW (primary) and
        complete-case (comparable to the P28 convention).
  PW2   Same with Marker-Based dropped (k=4). Marker-based is reliably last
        (0.558 synth / 0.512 real); a consistently-last item inflates W mechanically,
        so k=4 tests whether the four competing paradigms actually hold their order.
  PERM  Monte-Carlo permutation null (ranks permuted within scenario) for every W above.
  XARM  Cross-arm Spearman of paradigm mean-ranks (n=5; reported to show it is not usable).

Aggregation order matches Fig 2a: per-(tool, scenario) mean kappa first, then mean over
tools within paradigm. RAW = failed runs count as kappa 0 (the pipeline that reproduces
P25). Complete-case = failed runs dropped.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import os, glob, warnings, sys
import numpy as np, pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
pd.set_option("display.width", 220)

BASE = BENCH_ROOT
P4_DIR = f"{BASE}/results/P4 Synthetic Datasets V1"
P5_DIR = f"{BASE}/results/P5 Real Life Datasets V2"
OUT_DIR = f"{BASE}/results/_interaction_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(20260814)
N_PERM = 100_000

# ---------------------------------------------------------------- tool roster (identical to R)
marker_tools      = ["clustifyr_hyper","clustifyr_jaccard","scCATCH","SCINA","ScInfeR","SCSA","scSorter","scType"]
similarity_tools = ["CHETAH","CIPR","scmap_cell","scmap_cluster","Seurat_Transfer_CCA","Seurat_Transfer_PCA","Seurat_Transfer_RPCA","SingleR"]
classicML_tools   = ["CaSTLe","CellTypist","scAnnotate","scAnnotatR","scClassify","scibetR","scID",
    "scPred","scPred_adaboost","scPred_avNNet","scPred_bayesglm","scPred_earth","scPred_glm",
    "scPred_glmboost","scPred_knn","scPred_lda","scPred_mlp","scPred_multinom","scPred_nb",
    "scPred_nnet","scPred_regLogistic","scPred_rf","scPred_svmLinear","scPred_svmPoly","scPred_xgbTree","singleCellNet"]
DL_tools          = ["ACTINN","Cell_BLAST","CIForm","NeuCA","scBalance","scDeepSort","scHash","scMMT","TOSICA"]
semi_tools        = ["CALLR","CAMLU","MARS","SCANVI","scArches","scGAD","scnym","scSemiCluster","scSemiGAN"]

# V14 regrouping: scLearn is non-neural (Duan et al., Sci. Adv. 6, eabd0855, 2020) and was moved
# Deep Learning -> Classic ML, taking Classic ML 24 -> 25 and Deep Learning 9 -> 8. This is the
# roster the CURRENT manuscript's paradigm means are built on; `tool_interaction_analysis.py`
# predates it and still has scLearn under Deep Learning. See Notes/V14_change_log.md.
classicML_tools.append("scLearn")

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

# ---------------------------------------------------------------- loading (mirrors R load_benchmark_results)
def drop_index_col(df):
    if df.shape[1] > 0 and str(df.columns[0]) in ("", "X", "Unnamed: 0"):
        return df.iloc[:, 1:]
    return df

METRIC_COLS = ["kappa_mean","accuracy_mean","f1_mean"]

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

def assemble(raw, arm):
    d = raw[~raw["tool"].isin(exclude_tools)].copy()
    d["tool_category"] = d["tool"].map(cat_map)
    d = d[d["tool_category"].notna()].copy()
    fail = (d["kappa_mean"] == 0) & (d["accuracy_mean"] == 0) & (d["f1_mean"] == 0)
    d["run_status"] = np.where(fail, "failed_zero", "ok")
    d["kappa_cc"]   = np.where(fail, np.nan, d["kappa_mean"])   # complete-case
    d["kappa_raw"]  = np.where(fail, 0.0,   d["kappa_mean"])    # RAW: failure counts as 0
    d["arm"] = arm
    return d[["arm","scenario","seed","tool","tool_category","run_status","kappa_cc","kappa_raw"]]

synth = assemble(load_synth(), "synth")
real  = assemble(load_real(),  "real")

lines = []
def out(s=""):
    lines.append(s); print(s)

out("="*92)
out("PARADIGM-LEVEL KENDALL'S W  (companion to T0.3 tool-level W)")
out("="*92)
out(f"permutation draws = {N_PERM:,}   seed = 20260814")

# ---------------------------------------------------------------- scenario x paradigm matrices
def tool_level(d, kcol):
    """per-(tool, scenario) mean kappa -- the Fig 2a modelling unit."""
    tl = (d.groupby(["scenario","tool","tool_category"], as_index=False)
            .agg(kappa=(kcol, "mean")))
    return tl.dropna(subset=["kappa"])

def scenario_paradigm_matrix(d, kcol):
    """9 x 5 matrix: within each scenario, a paradigm's value = mean kappa of its tools there."""
    tl = tool_level(d, kcol)
    par_lvl = (tl.groupby(["scenario","tool_category"], as_index=False)
                 .agg(kappa=("kappa","mean"), n_tools=("tool","nunique")))
    mat = par_lvl.pivot(index="scenario", columns="tool_category", values="kappa")
    ntl = par_lvl.pivot(index="scenario", columns="tool_category", values="n_tools")
    return mat[[p for p in PARADIGMS if p in mat.columns]], ntl

def paradigm_grand_mean(d, kcol):
    """P25 convention: mean over ALL per-(tool,scenario) cells (equal weight per cell).

    For balanced paradigms this equals the mean of per-scenario means. It differs only for
    marker-based on the real arm, where SCINA is absent in S3 -- cell-weighted gives the
    published 0.512, scenario-weighted gives 0.485. Verified by the GATE below.
    """
    return tool_level(d, kcol).groupby("tool_category").kappa.mean()

mats = {}
for arm, d in [("synth", synth), ("real", real)]:
    for pipe, kcol in [("RAW","kappa_raw"), ("CC","kappa_cc")]:
        mats[(arm,pipe)], ntl = scenario_paradigm_matrix(d, kcol)
        if pipe == "RAW":
            out(f"\n[{arm}] tools contributing per paradigm (RAW, per scenario):")
            out("   " + "  ".join(f"{p.split('-')[0][:5]}={int(ntl[p].min())}-{int(ntl[p].max())}"
                                  for p in mats[(arm,pipe)].columns))
            nmiss = mats[(arm,pipe)].isna().sum().sum()
            out(f"   NA paradigm cells: {nmiss}  (matrix {mats[(arm,pipe)].shape[0]}x{mats[(arm,pipe)].shape[1]})")

# ---------------------------------------------------------------- GATE: reproduce P25's ten means
out("\n" + "-"*92)
out("GATE  Reproduce P25 published paradigm mean kappa (RAW pipeline)")
out("-"*92)

PUBLISHED = {
    "synth": {"Similarity-Based":0.626, "Classic ML":0.623, "Deep Learning":0.604,
              "Semi-Supervised":0.600, "Marker-Based":0.558},
    "real":  {"Similarity-Based":0.676, "Semi-Supervised":0.675, "Classic ML":0.638,
              "Deep Learning":0.622, "Marker-Based":0.512},
}

gate_ok = True
for arm, d_arm in [("synth", synth), ("real", real)]:
    gm = paradigm_grand_mean(d_arm, "kappa_raw")
    out(f"\n[{arm}]")
    for p in PARADIGMS:
        if p not in gm.index:
            out(f"   {p:<20} MISSING"); gate_ok = False; continue
        got = gm[p]
        want = PUBLISHED[arm][p]
        ok = abs(got - want) < 0.0005
        gate_ok &= ok
        out(f"   {p:<20} computed {got:.4f}   published {want:.3f}   {'OK' if ok else '<<< MISMATCH'}")

if not gate_ok:
    out("\nGATE FAILED — scenario x paradigm matrix does not reproduce the published means.")
    out("Refusing to report Kendall's W. Fix the aggregation before trusting anything below.")
    with open(f"{OUT_DIR}/PARADIGM_LEVEL_W.txt","w") as fh:
        fh.write("\n".join(lines) + "\n")
    sys.exit(1)
out("\nGATE PASSED — all ten paradigm means reproduce.")

# ---------------------------------------------------------------- kendall_w (verbatim from T0.3)
def kendall_w(mat):
    """mat: rows = scenarios (blocks), cols = tools. Higher kappa = better. Returns W, chi2, df, p."""
    N, k = mat.shape
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r, method="average"), 1, mat)
    Rj = ranks.sum(axis=0)
    S = ((Rj - Rj.mean())**2).sum()
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

def perm_p(mat, W_obs, n_perm=N_PERM):
    """Monte-Carlo null: permute ranks independently within each scenario (block)."""
    N, k = mat.shape
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r, method="average"), 1, mat)
    ge = 0
    for _ in range(n_perm):
        perm = np.array([RNG.permutation(ranks[i]) for i in range(N)])
        Rj = perm.sum(axis=0)
        S = ((Rj - Rj.mean())**2).sum()
        T = 0.0
        for i in range(N):
            _, cnt = np.unique(perm[i], return_counts=True)
            T += (cnt**3 - cnt).sum()
        denom = (N**2 * (k**3 - k) - N*T)
        if denom > 0 and 12*S/denom >= W_obs - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)

# ---------------------------------------------------------------- PW1 / PW2
def report(tag, mat):
    W, chi2, df, p, N, k = kendall_w(mat.values)
    pp = perm_p(mat.values, W)
    out(f"   {tag:<34} W = {W:.3f}   chi2 = {chi2:.1f}  df = {df}  p_chi2 = {p:.4g}   "
        f"p_perm = {pp:.4g}   (N={N}, k={k})")
    return dict(W=W, chi2=chi2, df=df, p_chi2=p, p_perm=pp, N=N, k=k)

res = {}
out("\n" + "-"*92)
out("PW1  Kendall's W over 5 paradigms across 9 scenarios")
out("-"*92)
for arm in ["synth","real"]:
    out(f"\n[{arm}]")
    for pipe, label in [("RAW","RAW (primary, matches P25)"), ("CC","complete-case (P28 convention)")]:
        res[(arm,pipe,5)] = report(label, mats[(arm,pipe)])

out("\n" + "-"*92)
out("PW2  Marker-Based dropped (k=4) — does concordance survive without the reliably-last item?")
out("-"*92)
for arm in ["synth","real"]:
    out(f"\n[{arm}]")
    for pipe, label in [("RAW","RAW (primary)"), ("CC","complete-case")]:
        m4 = mats[(arm,pipe)].drop(columns=["Marker-Based"])
        res[(arm,pipe,4)] = report(label, m4)

# ---------------------------------------------------------------- per-scenario rank tables
out("\n" + "-"*92)
out("Per-scenario paradigm ranks (RAW; 1 = best) — the raw material behind W")
out("-"*92)
for arm in ["synth","real"]:
    m = mats[(arm,"RAW")]
    rk = m.apply(lambda r: stats.rankdata(-r.values, method="average"), axis=1, result_type="expand")
    rk.columns = m.columns
    out(f"\n[{arm}]")
    out(rk.to_string(float_format=lambda x: f"{x:.1f}"))
    out("   mean rank: " + "  ".join(f"{c.split('-')[0][:5]}={rk[c].mean():.2f}" for c in rk.columns))

# ---------------------------------------------------------------- XARM
out("\n" + "-"*92)
out("XARM  Cross-arm Spearman of paradigm mean-ranks (n=5)")
out("-"*92)
ms = mats[("synth","RAW")]; mr = mats[("real","RAW")]
common = [c for c in ms.columns if c in mr.columns]
def mean_rank(m, cols):
    r = np.apply_along_axis(lambda x: stats.rankdata(-x), 1, m[cols].values)
    return pd.Series(r.mean(axis=0), index=cols)
a, b = mean_rank(ms, common), mean_rank(mr, common)
rho, prho = stats.spearmanr(a, b)
out(f"   rho = {rho:.3f}  p = {prho:.3g}  (n = {len(common)} paradigms)")
out("   NOTE: n=5 — reported only to show it cannot substitute for the tool-level rho=0.69 (n=40).")

# ---------------------------------------------------------------- headline
out("\n" + "="*92)
out("HEADLINE — paradigm level vs the tool-level numbers currently in P28")
out("="*92)
out(f"  NOTE p_perm floor = {1/(N_PERM+1):.2g}; any value at ~{2/(N_PERM+1):.0e} is resolution-limited")
out( "       and should be quoted as p < 1e-4, not as its literal value. Where p_chi2 and p_perm")
out( "       diverge (real k=5: 3.0e-4 vs <1e-4) the permutation p is the one to report: the")
out( "       chi2 approximation is still imperfect at k=5, N=9.")
out( "  NOTE roster invariance: the V14 scLearn regrouping (Deep Learning -> Classic ML) changes")
out( "       paradigm labels only. Tool-level W is computed on a scenario x tool pivot and the VC")
out( "       model is kappa ~ scenario * tool -- neither uses tool_category -- so P28's 0.42/0.57")
out( "       and the 84/9/4/2 partition are unaffected. Panel is 47 tools either way (2+6+25+8+6).")
out(f"  tool-level (P28, published)   : W = 0.42 synth / 0.57 real   (k=45, N=9; p reported as vacuous)")
for arm in ["synth","real"]:
    r5, r4 = res[(arm,"RAW",5)], res[(arm,"RAW",4)]
    out(f"  paradigm-level {arm:<6} k=5    : W = {r5['W']:.3f}  p_chi2 = {r5['p_chi2']:.3g}  p_perm = {r5['p_perm']:.3g}")
    out(f"  paradigm-level {arm:<6} k=4 (-marker): W = {r4['W']:.3f}  p_chi2 = {r4['p_chi2']:.3g}  p_perm = {r4['p_perm']:.3g}")

with open(f"{OUT_DIR}/PARADIGM_LEVEL_W.txt","w") as fh:
    fh.write("\n".join(lines) + "\n")
out(f"\nwritten: {OUT_DIR}/PARADIGM_LEVEL_W.txt")
