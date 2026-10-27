import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
# Fig 3a sensitivity: family-mean kappa vs kNN purity on real data, Table 2 purities vs
# purities recomputed on the benchmarked cells (S3, S7 labelled-only; S4 post-QC cells).
BASE_PY = os.path.join(BENCH_ROOT, "manuscript/Notes/v33_analysis/_base.py")
exec(open(BASE_PY).read())
from scipy import stats
CORR = {"S3": 0.4252, "S7": 0.8281, "S4": 0.2657}
d = real.copy(); d["sid"] = d.scenario.str.split("-").str[0]
def fits(d, kcol, pur, label):
    rows = []
    for fam, g in d.groupby("tool_category"):
        m = g.groupby("sid")[kcol].mean()
        p = pd.Series({s: pur[s] for s in m.index})
        lr = stats.linregress(p, m); rho = stats.spearmanr(p, m).statistic
        rows.append((fam, round(lr.rvalue**2, 3), round(rho, 3)))
    m = d.groupby("sid")[kcol].mean(); p = pd.Series({s: pur[s] for s in m.index})
    lr = stats.linregress(p, m)
    print(f"[{label} | {kcol}] all-tool R2={lr.rvalue**2:.3f} rho={stats.spearmanr(p, m).statistic:.3f}")
    for r in rows: print("   ", r)
orig = d.groupby("sid").knn_purity.first().to_dict()
corr = {**orig, **CORR}
for kcol in ["kappa", "kappa_imp0"]:
    fits(d, kcol, orig, "Table 2 purity")
    fits(d, kcol, corr, "benchmarked-cell purity")
print("purities used:", {k: (round(orig[k],3), round(corr[k],3)) for k in sorted(orig)})
