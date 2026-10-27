"""
panel_counts.py — derive the Figure 1b participation matrix from the tidy result
files, so the panel can never drift from the data it describes.

Every cell is a COUNT OF TOOL CONFIGURATIONS actually present in that arm's tidy
file (scPred's 17 caret variants count as 17, matching the roster and the 63 total).
Nothing here is typed by hand except the family display order.
"""
import os, pandas as pd

SH = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ["Marker-based", "Similarity-based", "Classical ML",
            "Deep learning", "Semi-supervised", "LLM", "Foundation model"]
# tidy-file spellings -> display spelling
NORM = {"Marker-Based": "Marker-based", "Similarity-Based": "Similarity-based",
        "Classic ML": "Classical ML", "Deep Learning": "Deep learning",
        "Semi-Supervised": "Semi-supervised", "LLM": "LLM",
        "Foundation model": "Foundation model", "FM": "Foundation model"}

def _p(n): return os.path.join(SH, n)

def counts():
    m = {f: {} for f in FAMILIES}

    # --- synthetic + real: the 47-configuration cell-level panel
    # NB: tidy_kappa's own tool_category column is the stale pre-V14 split
    # ("Correlation-Based", ML 24 / DL 9). CATMAP is the current authority, and is
    # what Figure 4 uses, so both figures are categorised identically.
    from catmap import CATMAP
    tk = pd.read_csv(_p("tidy_kappa.csv"))
    tk["_f"] = tk.tool.map(CATMAP).map(lambda s: NORM.get(s, s))
    for arm, key in [("synth", "Synthetic"), ("real", "Real")]:
        sub = tk[tk.arm == arm]
        for f, n in sub.groupby("_f").tool.nunique().items():
            if f in m: m[f][key] = int(n)

    # --- cross-platform
    cp = pd.read_csv(_p("tidy_crossplatform.csv"))
    cp["_f"] = cp.tool.map(CATMAP).map(lambda s: NORM.get(s, s))
    for f, n in cp.groupby("_f").tool.nunique().items():
        if f in m: m[f]["Cross-platform"] = int(n)
    # ScInfeR is in the declared pool. Since 2026-10-17 tidy_crossplatform.csv carries its
    # six CellBench/pancreas rows (build_tidy_crossplatform.py), so this guard no longer
    # fires; it stays as a safety net for a table built without ScInfeR.
    if "ScInfeR" not in set(cp.tool):
        m["Marker-based"]["Cross-platform"] = m["Marker-based"].get("Cross-platform", 0) + 1

    # --- database-driven markers + LLM annotators
    reg = pd.read_csv(_p("tidy_regimes.csv"))
    for fam_in, disp in [("Database marker", "Marker-based"), ("LLM", "LLM")]:
        n = reg[reg.family == fam_in].tool.nunique()
        if n: m[disp]["Database / LLM"] = int(n)

    # --- fine-tuned foundation models
    fm = pd.read_csv(_p("tidy_fm.csv"))
    m["Foundation model"]["Fine-tuned FM"] = int(fm.model.nunique())

    # --- roster total per family (distinct configurations anywhere) = the 63
    r = pd.read_csv(_p("supp_table4_roster.csv"))
    r["_f"] = r.paradigm.map(lambda s: {"ClassicalML": "Classical ML",
        "Deeplearning": "Deep learning", "Semi-supervised": "Semi-supervised",
        "Marker": "Marker-based", "Similarity": "Similarity-based",
        "LLM": "LLM", "FM": "Foundation model"}.get(s, s))
    extra = 16  # scPred contributes 17 configurations from one roster row
    for f, n in r.groupby("_f").size().items():
        if f in m:
            m[f]["Total"] = int(n) + (extra if f == "Classical ML" else 0)
    return m

COLUMNS = ["Synthetic", "Real", "Cross-platform", "Database / LLM", "Fine-tuned FM"]

if __name__ == "__main__":
    m = counts()
    hdr = COLUMNS + ["Total"]
    print(f"{'Family':<18}" + "".join(f"{h:>17}" for h in hdr))
    tot = {h: 0 for h in hdr}
    for f in FAMILIES:
        row = "".join(f"{m[f].get(h,'—'):>17}" for h in hdr)
        print(f"{f:<18}{row}")
        for h in hdr: tot[h] += m[f].get(h, 0)
    print(f"{'TOTAL':<18}" + "".join(f"{tot[h]:>17}" for h in hdr))
