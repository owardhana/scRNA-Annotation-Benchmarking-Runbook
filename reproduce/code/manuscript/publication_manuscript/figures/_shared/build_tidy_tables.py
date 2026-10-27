"""
build_tidy_tables.py — rebuild the figure tables in _shared/ from the raw results.

Most of the tables the figure scripts read were first assembled by hand or in
throw-away sessions. This script rebuilds each one from the per-run result files
and checks it against the copy in _shared/, so every figure input has a recorded
derivation. Tables with their own producer are not repeated here:
tidy_crossplatform.csv (build_tidy_crossplatform.py) and tidy_kappa_metrics.csv
(manuscript/Notes/v33_analysis/make_tidy_metrics.py).

  tidy_kappa.csv           tool_interaction_analysis.py output, plus every panel
                           tool x real scenario with no result file added as a failed
                           run (kappa = 0). That is one row: SCINA on S3 (Nowakowski).
  tidy_fm.csv              results/P8b FM - Real/data/real_FM_results.tsv + scenario
  tidy_fm_runtime.csv      tidy_fm + log_cells, n_types of each real scenario
  tidy_db_runtime.csv      results/P7a .../with_score/runtime_dataset.tsv
  tidy_llm_runtime.csv     results/P7b .../with_score/runtime_dataset.tsv + scenario
  tidy_regimes.csv         per (tool, dataset) mean ontology score over clusters,
                           from the P7a/P7b with_score cluster maps
  tidy_marker_derived.csv  full marker family under derived markers (P6a, P6b)
  tidy_cluster_ref.csv     CIPR and scmap_cluster rows of the reference_based tables
  tidy_cluster_ref_tidy.csv  the same, averaged over replicates per scenario
  tidy_gene_overlap.csv    Phase 3 train/test gene Jaccard (Phase3_GeneOverlap.csv)

Transcribed by hand and checked elsewhere, so not rebuilt: l9_design.csv (Table 1),
supp_table4_roster.csv (Supplementary Table 3), s1_panelc.csv.

Run:  python3 build_tidy_tables.py           rebuild in memory and compare (exit 1 on a mismatch)
      python3 build_tidy_tables.py --write   also write the rebuilt tables to _shared/
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SH = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.abspath(os.path.join(SH, *[".."] * 4))
RES = os.path.join(THESIS, "results")
P7A = os.path.join(RES, "P7a Marker - Real Database", "with_score")
P7B = os.path.join(RES, "P7b LLM - Real Internet", "with_score")

SCEN = {  # real dataset object -> scenario
    "Darmanis-Brain-2015": "S1", "Marques-Brain-2016": "S2", "Nowakowski-Cortex-2017": "S3",
    "Grun-Pancreas-2016": "S4", "Tabula Muris-FACS-3k": "S5", "He-Skin-2020": "S6",
    "Zhao-Immune-Fine-2020": "S7", "Zheng-ZhengSort-5cl-2017": "S8", "MacParland-Liver-Broad-2018": "S9",
}


def scen_of(name):
    for k, v in SCEN.items():
        if name.startswith(k):
            return v
    raise KeyError(name)


# ----------------------------------------------------------------------------- tables
def tidy_kappa():
    ia = pd.read_csv(os.path.join(RES, "_interaction_analysis", "tidy_kappa.csv"))
    real = ia[ia.arm == "real"]
    tools, scens = sorted(real.tool.unique()), sorted(real.scenario.unique())
    have = set(zip(real.tool, real.scenario))
    prof = real.groupby("scenario")[["knn_purity", "log_cells", "n_types", "shannon"]].first()
    add = []
    for s in scens:
        for t in tools:
            if (t, s) not in have:
                cat = real.loc[real.tool == t, "tool_category"].iloc[0]
                add.append(dict(arm="real", scenario=s, seed="single", tool=t, tool_category=cat,
                                run_status="failed_zero", kappa=np.nan, kappa_imp0=0.0, macro_f1=0.0,
                                rare_f1=0.0, runtime_s=np.nan, peak_mem_mb=np.nan, **prof.loc[s].to_dict()))
    return pd.concat([ia, pd.DataFrame(add)], ignore_index=True)[ia.columns]


def tidy_fm():
    fm = pd.read_csv(os.path.join(RES, "P8b FM - Real", "data", "real_FM_results.tsv"), sep="\t")
    fm["scenario"] = fm["sample"].map(scen_of)
    return fm


def tidy_fm_runtime():
    fm = tidy_fm().rename(columns={"scenario": "scen"})
    tk = pd.read_csv(os.path.join(SH, "tidy_kappa.csv"))
    prof = tk[tk.arm == "real"].groupby("scenario")[["log_cells", "n_types"]].first()
    prof.index = [s.split("-")[0] for s in prof.index]
    return fm.join(prof, on="scen")


def tidy_db_runtime():
    return pd.read_csv(os.path.join(P7A, "runtime_dataset.tsv"), sep="\t")


def tidy_llm_runtime():
    d = pd.read_csv(os.path.join(P7B, "runtime_dataset.tsv"), sep="\t")
    d.columns = [c.strip() for c in d.columns]
    d["scen"] = d["dataset"].map(scen_of)
    return d


def _score_maps(folder, suffix):
    """{(dataset_object, tool): DataFrame} of scored cluster maps. The .csv copies in
    with_score/ (and the Finder *_dup copies) are the unscored maps written by the
    harness; only files carrying score_1..3 count. If two scored copies exist, keep
    the one with more clusters."""
    best = {}
    for f in sorted(glob.glob(os.path.join(folder, f"*{suffix}*"))):
        base = os.path.basename(f)
        if base.startswith("runtime_dataset"):
            continue
        stem = base.split(suffix)[0]
        ds = next(k for k in SCEN if stem.startswith(k))
        tool = stem[len(ds):].lstrip("_")
        for pre in ("for_use_seurat_subset_", "original_seurat_subset_", "for_use_subset_", "original_subset_"):
            if tool.startswith(pre):
                tool = tool[len(pre):]
        d = pd.read_csv(f, sep="\t" if f.endswith(".tsv") else ",")
        if "score_1" not in d.columns:
            continue
        if (ds, tool) not in best or len(d) > len(best[(ds, tool)]):
            best[(ds, tool)] = d
    return best


def tidy_regimes():
    rows = []
    for family, folder, suffix, strip in [("Database marker", P7A, "_database_cluster_map", ""),
                                          ("LLM", P7B, "_cluster_map", "")]:
        for (ds, tool), d in _score_maps(folder, suffix).items():
            sc = d[["score_1", "score_2", "score_3"]].mean(axis=1)
            rows.append(dict(family=family, tool=tool, scenario=SCEN[ds],
                             n_clusters=len(d), score=round(sc.mean(), 4)))
    return pd.DataFrame(rows)


def _phase_tables(base, pattern):
    """Yield (arm, scenario, seed, table) for every per-run comparison table."""
    for f in sorted(glob.glob(os.path.join(base, "**", pattern), recursive=True)):
        rel = os.path.relpath(f, base).split(os.sep)
        yield rel, pd.read_csv(f)


def tidy_marker_derived():
    rows = []
    for arm, base in [("synth", os.path.join(RES, "P6a Marker - Synthetic", "Data")),
                      ("real", os.path.join(RES, "P6b Marker - Real", "Data"))]:
        for rel, d in _phase_tables(base, "*.csv"):
            name = rel[-1]
            scen = name.split("_")[2] if arm == "synth" else scen_of(name[len("marker_based_"):])
            for _, r in d.iterrows():
                ok = not (r.kappa_mean == 0 and r.accuracy_mean == 0 and r.f1_mean == 0)
                rows.append(dict(arm=arm, scen=scen, tool=r.tool, k=r.kappa_mean if ok else np.nan, ok=ok))
    d = pd.DataFrame(rows)
    g = d.groupby(["arm", "scen", "tool"]).agg(k=("k", "mean"), n_ok=("ok", "sum")).reset_index()
    g["k"] = g.k.fillna(0.0)          # no successful run: a failure, scored kappa = 0
    return g


def tidy_cluster_ref():
    rows = []
    # CIPR and scmap_cluster sit in the reference_based table, or in the single_split
    # re-run table for the scenarios that were re-run (synthetic S3/S4, real S3/S4/S6/S7).
    for arm, base in [("synth", os.path.join(RES, "P4 Synthetic Datasets V1")),
                      ("real", os.path.join(RES, "P5 Real Life Datasets V2"))]:
        for rel, d in _phase_tables(base, "*.csv"):
            if rel[0] in {"extras", "Plots", "plots", "Plots_OrthogonalFactors"} or len(rel) < 2:
                continue
            scen = rel[0].split("-")[0]
            for _, r in d[d.tool.isin(["CIPR", "scmap_cluster"])].iterrows():
                rows.append(dict(arm=arm, scen=scen, tool=r.tool, kappa_mean=r.kappa_mean,
                                 accuracy_mean=r.accuracy_mean, f1_mean=r.f1_mean))
    return pd.DataFrame(rows)


def tidy_cluster_ref_tidy():
    d = tidy_cluster_ref()
    d["ok"] = ~((d.kappa_mean == 0) & (d.accuracy_mean == 0) & (d.f1_mean == 0))
    d["k"] = d.kappa_mean   # failed runs (kappa = accuracy = F1 = 0) stay in the mean at 0; n_ok counts the rest
    return d.groupby(["arm", "scen", "tool"]).agg(k=("k", "mean"), n_ok=("ok", "sum")).reset_index()


TRIALS = {"CB-1": ("Cellbench 10xv2", "Cellbench Cel-Seq2"), "CB-2": ("Cellbench 10xv2", "Cellbench Drop-seq"),
          "CB-3": ("Cellbench Cel-Seq2", "Cellbench 10xv2"), "CB-4": ("Cellbench Drop-seq", "Cellbench 10xv2"),
          "PX-1": ("Baron-Pancreas", "Segerstolpe-Pancreas-2016"), "PX-2": ("Segerstolpe-Pancreas-2016", "Baron-Pancreas"),
          "PBMC-A": ("PBMCBench 10xv2-8cl", "PBMCBench Drop-seq-8cl"), "PBMC-B": ("PBMCBench 10xv2-8cl", "PBMCBench InDrops-8cl"),
          "PBMC-C": ("PBMCBench 10xv2-6cl", "PBMCBench Seq-well-6cl")}  # as-run/benchmarking/Benchmarking.R pairings
BLOCK = {"CB": "CellBench", "PX": "Pancreas", "PBMC": "PBMCbench"}


def tidy_gene_overlap():
    j = pd.read_csv(os.path.join(THESIS, "Data Creation", "P9 Cross Datasets", "Phase3_GeneOverlap.csv"))
    look = {frozenset((r.Dataset_A, r.Dataset_B)): r.Jaccard for r in j.itertuples()}
    return pd.DataFrame([dict(trial=t, block=BLOCK[t.split("-")[0]], jaccard=look[frozenset(p)])
                         for t, p in TRIALS.items()])


TABLES = {
    "tidy_kappa": (tidy_kappa, ["arm", "scenario", "seed", "tool"], ["tool_category"]),
    "tidy_fm": (tidy_fm, ["model", "sample"], []),
    "tidy_fm_runtime": (tidy_fm_runtime, ["model", "sample"], []),
    "tidy_db_runtime": (tidy_db_runtime, ["dataset", "tool"], []),
    "tidy_llm_runtime": (tidy_llm_runtime, ["dataset", "tool"], []),
    "tidy_regimes": (tidy_regimes, ["family", "tool", "scenario"], []),
    "tidy_marker_derived": (tidy_marker_derived, ["arm", "scen", "tool"], []),
    "tidy_cluster_ref": (tidy_cluster_ref, ["arm", "scen", "tool", "kappa_mean"], []),
    "tidy_cluster_ref_tidy": (tidy_cluster_ref_tidy, ["arm", "scen", "tool"], []),
    "tidy_gene_overlap": (tidy_gene_overlap, ["trial"], []),
}


# ----------------------------------------------------------------------------- compare
def compare(new, old, keys, ignore):
    """Same columns, same rows (by keys), numerics equal to 1e-9 relative. Returns problems."""
    probs = []
    if set(new.columns) != set(old.columns):
        return [f"columns differ: new-only {sorted(set(new.columns) - set(old.columns))}, "
                f"deposited-only {sorted(set(old.columns) - set(new.columns))}"]
    if len(new) != len(old):
        probs.append(f"{len(new)} rows rebuilt vs {len(old)} deposited")
    a = new.sort_values(keys).reset_index(drop=True)
    b = old[new.columns].sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        return probs
    for c in new.columns:
        if c in ignore:
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            bad = ~np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            bad = x.astype(str) != y.astype(str)
        if bad.any():
            probs.append(f"column {c}: {int(bad.sum())} rows differ")
    return probs


def main():
    write = "--write" in sys.argv
    failed = 0
    for name, (fn, keys, ignore) in TABLES.items():
        new = fn()
        old = pd.read_csv(os.path.join(SH, f"{name}.csv"))
        probs = compare(new, old, keys, ignore)
        note = f"  (ignored: {', '.join(ignore)})" if ignore else ""
        print(f"{'OK      ' if not probs else 'MISMATCH'} {name:24s} {len(new):5d} rows{note}")
        for p in probs:
            print(f"           {p}")
        failed += bool(probs)
        if write:
            new.to_csv(os.path.join(SH, f"{name}.csv"), index=False)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
