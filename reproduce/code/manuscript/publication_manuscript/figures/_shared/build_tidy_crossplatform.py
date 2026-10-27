"""
build_tidy_crossplatform.py — rebuild tidy_crossplatform.csv from the raw Phase 3 results.

Source : results/P9 Cross Dataset/data/cross_dataset_trial_<TRIAL>_<STAMP>.csv (18 files,
         two runs per trial; every (trial, tool) pair occurs in exactly one file).
Output : tidy_crossplatform.csv, one row per (trial, tool), sorted by trial then tool,
         with the raw columns plus `fam` (catmap.CATMAP) and `block`.

Excluded tools (exact names, nothing else):
  scAnnotatR — excluded from the paper's roster (Plots_Analysis.R exclude_tools).
  CALLR      — not in the 63-configuration roster (degenerate synthetic kappa = 0.018);
               removed from this table on 2026-09-10 (see tidy_crossplatform.csv.bak_with_callr).

ScInfeR is KEPT. The earlier table (before 2026-10-17) silently lacked it, most likely
because the P9 R exclude list's 'scInfeR' entry was matched case-insensitively when the table
was first ported, although in R that entry never matches. ScInfeR has six raw rows (CellBench
CB-1..4 and pancreas PX-1/PX-2) and none for PBMCbench; figure scripts score the missing
PBMC cells kappa = 0 under the paper's failure convention.

Run:  python3 build_tidy_crossplatform.py          (asserts the old rows are reproduced)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SH)
from catmap import CATMAP  # noqa: E402

THESIS = os.path.abspath(os.path.join(SH, *[".."] * 4))
RAW_DIR = os.path.join(THESIS, "results", "P9 Cross Dataset", "data")
OUT = os.path.join(SH, "tidy_crossplatform.csv")
EXCLUDE = {"scAnnotatR", "CALLR"}
BLOCK = {"CB": "CellBench", "PX": "Pancreas", "PBMC": "PBMCbench"}
N_TOOLS = 55  # 8 marker + 8 similarity + 25 classical ML + 8 deep learning + 6 semi-supervised


def load_raw():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW_DIR, "cross_dataset_trial_*.csv"))):
        d = pd.read_csv(f)
        d.insert(0, "trial", os.path.basename(f).split("_")[3])
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    # never more than one row per (trial, tool); average defensively if that ever changes
    if raw.duplicated(["trial", "tool"]).any():
        num = raw.select_dtypes("number").columns
        raw = raw.groupby(["trial", "tool"], as_index=False)[list(num)].mean()
    return raw


def build():
    raw = load_raw()
    t = raw[~raw.tool.isin(EXCLUDE)].copy()
    # failed run (kappa = accuracy = F1 = 0) -> kappa NA, as in Plots_Analysis.R line 291;
    # build_s08.py scores NA as kappa = 0 again on the full grid
    failed = (t.kappa_mean == 0) & (t.accuracy_mean == 0) & (t.f1_mean == 0)
    t.loc[failed, "kappa_mean"] = np.nan
    t["fam"] = t.tool.map(CATMAP)
    if t.fam.isna().any():
        sys.exit(f"tools missing from catmap: {sorted(t.loc[t.fam.isna(), 'tool'].unique())}")
    t["block"] = t.trial.str.split("-").str[0].map(BLOCK)
    t = t.sort_values(["trial", "tool"]).reset_index(drop=True)
    cols = ["trial", "tool"] + [c for c in raw.columns if c not in ("trial", "tool")] + ["fam", "block"]
    return t[cols]


def check_against(new, old_path):
    """Every row of the previous table must come back unchanged; only ScInfeR may be added."""
    old = pd.read_csv(old_path)
    key = ["trial", "tool"]
    added = new.merge(old[key], on=key, how="left", indicator=True)
    added = added[added._merge == "left_only"]
    assert set(added.tool) <= {"ScInfeR"}, f"unexpected new tools: {sorted(set(added.tool))}"
    m = old.merge(new, on=key, suffixes=("_old", "_new"))
    assert len(m) == len(old), "rows of the previous table are missing"
    for c in old.columns:
        if c in key:
            continue
        a, b = m[f"{c}_old"], m[f"{c}_new"]
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            same = np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
        else:
            same = (a.astype(str) == b.astype(str))
        assert same.all(), f"column {c} changed for {int((~same).sum())} rows"
    return len(old), len(added)


if __name__ == "__main__":
    new = build()
    assert new.tool.nunique() == N_TOOLS, f"{new.tool.nunique()} tools, expected {N_TOOLS}"
    baseline = os.environ.get("TIDY_CP_BASELINE")
    if baseline:
        n_old, n_add = check_against(new, baseline)
        print(f"baseline {os.path.basename(baseline)}: {n_old} rows reproduced, {n_add} added")
    new.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(new)} rows, {new.tool.nunique()} tools")
    print(new.groupby("fam").tool.nunique().to_string())
