import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
# Recompute Discussion para 57 hard-data claims under alternative hard sets.
# Execs manuscript/Notes/v33_analysis/_base.py (same loader, roster, failure rule as q3/q4).
import sys
BASE_PY = os.path.join(BENCH_ROOT, "manuscript/Notes/v33_analysis/_base.py")
exec(open(BASE_PY).read())
def pre(s): return s.split("-")[0]
def hard_stats(d, H, label):
    d = d.copy(); d["sid"] = d.scenario.map(pre)
    h = d[d.sid.isin(H)].groupby("tool").kappa_imp0.mean()
    sp = h[[t for t in h.index if t.startswith("scPred")]]
    within = h[h >= h.max() - 0.1].sort_values(ascending=False)
    print(f"--- {label}  H={H}")
    print(f"  best {h.max():.3f} ({h.idxmax()}); within 0.1 of best: {len(within)}/{len(h)} -> {within.round(3).to_dict()}")
    print(f"  scPred variants: {sp.min():.3f}-{sp.max():.3f} (n={len(sp)}); >0.32: {sp[sp>0.32].round(3).to_dict()}")
    print(f"  top6: {h.sort_values(ascending=False).head(6).round(3).to_dict()}")
    return h
out = {}
out["synth_S3S4"] = hard_stats(synth, ["S3","S4"], "synth (published)")
out["real_S3S4"]  = hard_stats(real,  ["S3","S4"], "real (published)")
out["real_S3"]    = hard_stats(real,  ["S3"],      "real, S3 only")
out["real_S4"]    = hard_stats(real,  ["S4"],      "real, S4 only")
out["synth_S3"]   = hard_stats(synth, ["S3"],      "synth, S3 only")
def topq(a, b, tag):
    h = pd.DataFrame({"synth": out[a], "real": out[b]}).dropna(); q = len(h) / 4
    r = h.rank(ascending=False)
    print(f"top quartile in both [{tag}]:", sorted(h[(r.synth <= q) & (r.real <= q)].index))
topq("synth_S3S4", "real_S3S4", "published: synth S3+S4, real S3+S4")
topq("synth_S3S4", "real_S3",   "synth S3+S4, real S3 only")
topq("synth_S3",   "real_S3",   "synth S3 only, real S3 only")
# dataset-level all-tool mean kappa vs purity, with and without S4
from scipy import stats
d = real.copy(); d["sid"] = d.scenario.map(pre)
g = d.groupby("sid").agg(k=("kappa_imp0","mean"), p=("knn_purity","first"))
for drop in [[], ["S4"]]:
    gg = g.drop(drop); lr = stats.linregress(gg.p, gg.k); rho = stats.spearmanr(gg.p, gg.k)
    print(f"all-tool mean kappa vs purity, real, drop={drop}: R2={lr.rvalue**2:.3f} rho={rho.statistic:.3f} n={len(gg)}")
print("per-dataset all-tool mean kappa (real):", g.k.round(3).to_dict())
