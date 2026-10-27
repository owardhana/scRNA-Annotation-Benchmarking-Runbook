"""V33 revision (ZZ comments #22/#23/#34/#36-38): repeat the dataset-vs-tool analyses for accuracy,
macro F1 and MCC alongside kappa, using the identical roster, exclusions, failure rule and profiles
as tool_interaction_analysis.py. Usage: python v33_multimetric.py <THESIS_BASE> <OUT_DIR>"""
exec(open(__file__.replace("v33_multimetric.py","_base.py")).read())

METRICS = {"kappa":"Cohen's kappa","accuracy":"Accuracy","macro_f1":"Macro F1","mcc":"MCC"}
def prep(d):
    d = d.copy()
    fail = d.run_status == "failed_zero"
    for m in ["accuracy","macro_f1","mcc"]:
        d[m + "_imp0"] = np.where(fail, 0.0, d[m])
        d.loc[fail, m] = np.nan
    return d
synth = prep(synth); real = prep(real)

def means(d, col):
    g = (d.groupby(["scenario","tool","tool_category"], as_index=False)
           .agg(y=(col,"mean"), knn_purity=("knn_purity","first"), log_cells=("log_cells","first"),
                n_types=("n_types","first"), shannon=("shannon","first")))
    return g.dropna(subset=["y"])

out("="*90); out("V33 MULTI-METRIC ROBUSTNESS"); out("="*90)

# 1. variance components (synthetic, seed-level, failures imputed 0) per metric
out("\n[1] Variance components, synthetic seed-level, scenario x tool (failures = 0)")
vc = {}
for m in METRICS:
    col = "kappa_imp0" if m == "kappa" else m + "_imp0"
    d = synth.copy(); d["y"] = d[col]
    r = mom_components(d, "y", f"{m}")
    vc[m] = r["pct"]

# 2. pooled Type III eta2 (purity, tool) per metric and arm
out("\n[2] Pooled Type III eta^2 with tool term (per-(tool,scenario) means, failures = 0 as in manuscript)")
t3 = {}
for arm, d in [("synth", synth), ("real", real)]:
    for m in METRICS:
        g = means(d, m + "_imp0")
        e, p, _, _ = type3_eta(g, "y ~ knn_purity + log_cells + n_types + shannon + C(tool)",
                               ["knn_purity","log_cells","n_types","shannon","C(tool)"])
        t3[(arm,m)] = e
        out(f"   [{arm:5s}] {m:9s} purity={e['knn_purity']:.3f} cells={e['log_cells']:.3f} "
            f"types={e['n_types']:.3f} entropy={e['shannon']:.3f} tool={e['C(tool)']:.3f}")

# 3. tool-rank concordance between kappa and each metric (mean per tool across scenarios)
out("\n[3] Spearman rho of per-tool mean score, kappa vs other metric")
for arm, d in [("synth", synth), ("real", real)]:
    tm = d.groupby("tool")[[m + "_imp0" for m in METRICS]].mean(); tm.columns = list(METRICS)
    for m in ["accuracy","macro_f1","mcc"]:
        x = tm[["kappa", m]].dropna()
        rho, p = stats.spearmanr(x["kappa"], x[m])
        out(f"   [{arm:5s}] kappa vs {m:9s} rho={rho:.3f} (n={len(x)})")
    # family-mean ordering per metric
    fm = d.groupby("tool_category")[list(METRICS)].mean().round(3)
    out(f"   [{arm}] family means:\n" + fm.to_string())

# 4. family-mean vs purity fits per metric (real)
out("\n[4] Family-mean score vs kNN purity, R^2 range across 5 families (dataset level)")
for arm, d in [("synth", synth), ("real", real)]:
    for m in METRICS:
        r2s = []
        for fam, gf in d.groupby("tool_category"):
            fmn = gf.groupby("scenario").agg(y=(m,"mean"), p=("knn_purity","first")).dropna()
            if len(fmn) >= 5:
                r2s.append(stats.linregress(fmn.p, fmn.y).rvalue**2)
        out(f"   [{arm:5s}] {m:9s} R2 {min(r2s):.2f}-{max(r2s):.2f}")

# 5. kappa paradox check: accuracy - kappa gap vs class-distribution entropy (dataset level)
out("\n[5] Prevalence check: per-dataset mean(accuracy - kappa) vs normalized Shannon entropy")
for arm, d in [("synth", synth), ("real", real)]:
    dd = d.dropna(subset=["kappa","accuracy"]).copy(); dd["gap"] = dd.accuracy - dd.kappa
    g = dd.groupby("scenario").agg(gap=("gap","mean"), H=("shannon","first"),
                                   k=("kappa","mean"), p=("knn_purity","first"))
    rho, p = stats.spearmanr(g.H, g.gap)
    out(g.round(3).to_string()); out(f"   [{arm}] Spearman(entropy, gap) rho={rho:.2f} p={p:.2g}")

# 6. low-separability datasets: which tools hold up?
out("\n[6] Tool ranking on low-purity datasets")
prof = {"synth": SYNTH_PROFILES, "real": REAL_PROFILES}
for arm, d in [("synth", synth), ("real", real)]:
    pr = prof[arm].sort_values("knn_purity"); out(f"   [{arm}] purity by scenario:\n" + pr[["scenario","knn_purity"]].round(3).to_string(index=False))
    low = list(pr.scenario[pr.knn_purity < 0.5])
    out(f"   [{arm}] low-purity (<0.5) scenarios: {low}")
    dd = d.copy(); dd["k0"] = dd["kappa_imp0"]
    hard = dd[dd.scenario.isin(low)].groupby(["tool","tool_category"]).k0.mean().sort_values(ascending=False)
    allm = dd.groupby("tool").k0.mean()
    out("   top 8:\n" + hard.head(8).round(3).to_string()); out("   bottom 6:\n" + hard.tail(6).round(3).to_string())
    fam = dd[dd.scenario.isin(low)].groupby("tool_category").k0.mean().round(3)
    out("   family means on low-purity:\n" + fam.to_string())
    hr = hard.reset_index().set_index("tool").k0
    rho, p = stats.spearmanr(hr, allm.loc[hr.index]); out(f"   rank rho(low-purity, all) = {rho:.2f}")

# 7. within-family spread and consistent bottom tools (real, per-tool mean kappa, failures 0)
out("\n[7] Within-family spread of per-tool mean kappa")
for arm, d in [("synth", synth), ("real", real)]:
    tm = d.groupby(["tool_category","tool"]).kappa_imp0.mean()
    for fam, s in tm.groupby(level=0):
        s = s.droplevel(0)
        out(f"   [{arm:5s}] {fam:18s} n={len(s):2d} range {s.min():.3f}-{s.max():.3f} "
            f"(low: {s.idxmin()}, high: {s.idxmax()})")
    out(f"   [{arm}] overall bottom 6:\n" + tm.droplevel(0).sort_values().head(6).round(3).to_string())
    out(f"   [{arm}] overall top 6:\n" + tm.droplevel(0).sort_values(ascending=False).head(6).round(3).to_string())

open(f"{OUT_DIR}/V33_MULTIMETRIC_RESULTS.txt","w").write("\n".join(lines)+"\n")
