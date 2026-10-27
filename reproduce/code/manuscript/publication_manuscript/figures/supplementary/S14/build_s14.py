"""
Supplementary Figure 14 | Annotation maps for representative strong and weak tools.
Nature style, 183 mm wide, 2 rows x 4 maps.

Rows    : S9 MacParland liver (easy, _original) and S3 Nowakowski cortex (hard, _for_use).
          S4 Grun was dropped: its labels are FACS sort gates / libraries, not cell types.
Columns : ground truth, singleCellNet, CaSTLe, scPred (svmRadial): the examples
          named in Discussion para 57.
Data    : ../../_shared/tidy_annotation_maps.csv, one row per (dataset, cell), written by
          results/_SI_generated/make_S14_annotation_maps.R, which re-runs the three tools on
          the original single 80/20 split (per-cell predictions were not kept in Phase 2).
          kappa per panel: results/_SI_generated/S14_annotation_maps/kappa_check.csv
          (re-run value for the seed that reproduces the published kappa).
          kNN purity: "Data Creation/P5 Real Life Sets V2/Profiling_Real.csv".
Encoding: cells of the reference (training) split are a faint background; only test cells
          are coloured. Labels that are not a cell type of the dataset (scPred "unassigned",
          "Unknown", ...) are a hollow FAIL_GREY ring, so they read without colour.
  "label" rows (S9, 8 types): test cells coloured by truth (column 1) or prediction
          (columns 2-4); a type has the same colour in every column.
  "error" rows (S3, 47 types, too many to colour): column 1 colours truth by lineage
          class (Nowakowski et al. 2017 cluster-name prefixes); columns 2-4 mark each test
          cell correct / misclassified / unassigned at the full 47-type resolution.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, re, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SH = os.path.join(HERE, "..", "..", "_shared"); sys.path.insert(0, SH)
THESIS = os.path.abspath(os.path.join(HERE, *[".."] * 5))
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
import matplotlib as mpl
try:
    from setup_style import setup_style
    setup_style(journal="nature", lang="en", constrained_layout=False)
except ImportError:
    pass
import matplotlib.pyplot as plt
import figstyle as FS

FONT = FS.FONT
PT_TICK, PT_LAB, PT_TAG, PT_ROW = FS.PT_TICK, FS.PT_LAB, FS.PT_TAG, FS.PT_ROW
INK, MUTED, LW = FS.INK, FS.MUTED, FS.LW
rc = FS.rc()
rc["font.sans-serif"] = [FONT, "Arial", "DejaVu Sans"]   # Arial is metric-identical
mpl.rcParams.update(rc)

TIDY = os.path.join(SH, "tidy_annotation_maps.csv")
KAPPA = os.path.join(THESIS, "results", "_SI_generated", "S14_annotation_maps", "kappa_check.csv")
PROFILE = os.path.join(THESIS, "Data Creation", "P5 Real Life Sets V2", "Profiling_Real.csv")

ROWS = [dict(sc="S9", name="MacParland liver 2018", level="Easy",
             profile="MacParland-Liver-Broad-2018_original", mode="label"),
        dict(sc="S3", name="Nowakowski cortex 2017", level="Hard",
             profile="Nowakowski-Cortex-2017_for_use", mode="error")]
TOOLS = ["singleCellNet", "CaSTLe", "scPred"]
COL_TITLES = ["Ground truth", "singleCellNet", "CaSTLe", "scPred (svmRadial)"]

# Paul Tol "muted", in Tol's own order (adjacent entries contrast)
PALETTE = ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
           "#882255", "#44AA99", "#999933", "#AA4499"]
UNASSIGNED = "Unassigned"
C_UNASSIGNED = FS.FAIL_GREY
C_TRAIN = "#e6e6e6"
# error rows follow figstyle's convention, cool = good and warm = poor (RdBu); the two
# also differ in lightness so they separate in grayscale
C_CORRECT, C_WRONG = "#92c5de", "#b2182b"
CORRECT, WRONG = "Correct", "Misclassified"
MM = 1 / 25.4

# Nowakowski et al. 2017 cluster names -> lineage group, by name prefix. Order matters:
# nEN before EN, nIN before IN. Display only; scoring uses the 47 labels.
LINEAGE_RULES = [
    (r"^(RG-|tRG$|vRG$|oRG$)", "Radial glia"),
    (r"^IPC-", "Intermediate progenitors"),
    (r"^nEN-", "Newborn excitatory"),
    (r"^EN-", "Excitatory neurons"),
    (r"^MGE-", "MGE progenitors"),
    (r"^nIN", "Newborn interneurons"),
    (r"^IN-", "Interneurons"),
    (r"^(Astrocyte|OPC|Microglia|Endothelial|Mural|Choroid)$", "Glia and non-neural"),
    (r"^(U\d+|Glyc)$", "Unresolved (U1–4, Glyc)"),
]
LINEAGE_ORDER = [g for _, g in LINEAGE_RULES]


def lineage(t):
    for pat, g in LINEAGE_RULES:
        if re.match(pat, t):
            return g
    raise SystemExit(f"no lineage rule for Nowakowski type {t!r}")


def chosen_kappa(kc):
    """Re-run kappa per (dataset, tool) for the seed whose singleCellNet and CaSTLe
    runs both reproduced the published kappa."""
    out = {}
    for ds, d in kc.groupby("dataset"):
        ok = [s for s, g in d.groupby("seed")
              if g.loc[g.tool != "scPred", "within_tol"].astype(str).str.upper().eq("TRUE").all()]
        if not ok:
            raise SystemExit(f"{ds}: no seed reproduced the published kappa")
        g = d[d.seed == ok[0]]
        out[ds] = dict(seed=int(ok[0]), **dict(zip(g.tool, g.kappa_rerun)))
    return out


def r_kappa(pred, true):
    """calculate_metrics() kappa (benchmarking_helpers.R:49): accuracy over all cells,
    chance agreement summed over every label except 'Unknown'."""
    pred, true = np.asarray(pred), np.asarray(true)
    acc = np.mean(pred == true)
    types = [t for t in pd.unique(np.concatenate([pred, true])) if t != "Unknown"]
    pe = sum(np.mean(pred == t) * np.mean(true == t) for t in types)
    return acc, (acc - pe) / (1 - pe) if pe < 1 else 1.0


def square_limits(x, y, pad=0.05, head=0.14):
    """Square window around the cloud, with `head` of the span left empty at the top
    for the corner label."""
    span = max(np.ptp(x), np.ptp(y)) * (1 + 2 * pad) / (1 - head)
    cx = (x.min() + x.max()) / 2
    ylo = (y.min() + y.max()) / 2 - span * (1 - head) / 2
    return (cx - span / 2, cx + span / 2), (ylo, ylo + span)


def build():
    df = pd.read_csv(TIDY)
    kap = chosen_kappa(pd.read_csv(KAPPA))
    prof = pd.read_csv(PROFILE).set_index("Sample")["KNN_Purity"]

    # ---- layout (mm)
    LEFT, TILE, GUT, GAP, LEG = 23.0, 30.0, 2.5, 3.0, 29.0
    HEAD, ROWGAP, BOTTOM = 9.0, 4.0, 2.0
    FW = LEFT + 4 * TILE + 3 * GUT + GAP + LEG
    FH = HEAD + len(ROWS) * TILE + (len(ROWS) - 1) * ROWGAP + BOTTOM
    fig = plt.figure(figsize=(FW * MM, FH * MM)); fig.patch.set_facecolor("white")
    fx, fy = (lambda v: v / FW), (lambda v: 1 - v / FH)        # mm from top-left

    for j, ttl in enumerate(COL_TITLES):
        xc = LEFT + j * (TILE + GUT) + TILE / 2
        fig.text(fx(xc), fy(HEAD - 3.2), ttl, fontsize=PT_LAB, fontweight="bold",
                 ha="center", va="baseline")

    rng = np.random.default_rng(7)
    audit = []
    for i, row in enumerate(ROWS):
        sc, mode = row["sc"], row["mode"]
        d = df[(df.dataset == sc) & (df.split != "excluded")]
        test, train = d[d.split == "test"], d[d.split == "train"]
        types = d.truth.value_counts().index.tolist()          # most frequent first
        if mode == "label":
            if len(types) > len(PALETTE):
                raise SystemExit(f"{sc}: {len(types)} types > {len(PALETTE)} colours")
            cmap = dict(zip(types, PALETTE))
            keys = types
        else:
            lin = {t: lineage(t) for t in types}
            cmap = dict(zip(LINEAGE_ORDER, PALETTE))
            keys = LINEAGE_ORDER
        order = rng.permutation(len(test))                      # no class always on top
        xl, yl = square_limits(d.UMAP_1.values, d.UMAP_2.values)
        s_test = float(np.clip(900 / len(test), 0.6, 4.0))      # pt^2
        s_train = s_test * 0.45
        truth = test.truth.astype(str).values

        ytop = HEAD + i * (TILE + ROWGAP)
        fig.text(fx(1.0), fy(ytop + 4.0), sc, fontsize=PT_TAG, fontweight="bold",
                 ha="left", va="baseline")
        fig.text(fx(1.0), fy(ytop + 8.0), row["name"], fontsize=PT_ROW,
                 ha="left", va="baseline")
        fig.text(fx(1.0), fy(ytop + 11.2),
                 f"{row['level']} · kNN purity {prof[row['profile']]:.2f}",
                 fontsize=PT_ROW, color=MUTED, ha="left", va="baseline")

        for j, col in enumerate(["truth"] + [f"pred_{t}" for t in TOOLS]):
            x0 = LEFT + j * (TILE + GUT)
            ax = fig.add_axes([fx(x0), fy(ytop + TILE), TILE / FW, TILE / FH])
            ax.scatter(train.UMAP_1, train.UMAP_2, s=s_train, c=C_TRAIN, lw=0,
                       rasterized=True, zorder=1)
            raw = test[col].fillna("Unknown").astype(str).values
            known = np.isin(raw, types)
            if mode == "label":
                colours = np.array([cmap[v] if k else C_UNASSIGNED
                                    for v, k in zip(raw, known)])
            elif j == 0:
                colours = np.array([cmap[lin[v]] for v in raw])
            else:
                colours = np.where(~known, C_UNASSIGNED,
                                   np.where(raw == truth, C_CORRECT, C_WRONG))
            t, c, un = test.iloc[order], colours[order], ~known[order]
            ax.scatter(t.UMAP_1[~un], t.UMAP_2[~un], s=s_test, c=list(c[~un]), lw=0,
                       rasterized=True, zorder=2)
            # Unassigned: hollow ring, so it reads without colour (grayscale, CVD)
            ax.scatter(t.UMAP_1[un], t.UMAP_2[un], s=s_test, facecolors="none",
                       edgecolors=C_UNASSIGNED, lw=0.35, rasterized=True, zorder=3)
            ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ["top", "right", "bottom", "left"]:
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set(color="#c8c8c8", linewidth=0.4)
            if j == 0:
                txt = f"{len(test):,} test cells"
                if mode == "error":
                    txt += f"\n{len(types)} types, by class"
            else:
                tool = TOOLS[j - 1]
                acc, k_py = r_kappa(raw, truth)
                k_r = kap[sc][tool]
                frac_un = float(np.mean(~known))
                wrong = known & (raw != truth)
                same_lin = (float(np.mean([lin[p] == lin[q] for p, q in
                                           zip(raw[wrong], truth[wrong])]))
                            if mode == "error" and wrong.any() else np.nan)
                audit.append((sc, tool, kap[sc]["seed"], round(acc, 4), round(k_py, 4),
                              round(k_r, 4), round(frac_un, 3), round(float(wrong.mean()), 3),
                              round(same_lin, 3), sorted(set(raw[~known]))))
                txt = f"κ = {k_r:.3f}"
                if frac_un >= 0.01:
                    txt += f"\n{frac_un:.0%} unassigned"
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=PT_ROW,
                    ha="left", va="top", color=INK, linespacing=1.15, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))

        # ---- per-row legend; error rows get two blocks (lineage for column 1, then
        # the per-cell outcome for columns 2-4) separated by a small gap
        lx = LEFT + 4 * TILE + 3 * GUT + GAP
        tail = [(UNASSIGNED, C_UNASSIGNED), ("Reference split", C_TRAIN)]
        if mode == "label":
            blocks = [[(k, cmap[k]) for k in keys] + tail]
        else:
            blocks = [[(k, cmap[k]) for k in keys],
                      [(CORRECT, C_CORRECT), (WRONG, C_WRONG)] + tail]
        n, BGAP = sum(len(b) for b in blocks), 1.0
        pitch = min(3.0, (TILE - 2.2 - BGAP * (len(blocks) - 1)) / (n - 1))
        yy = ytop + 1.6
        for b in blocks:
            for lbl, c in b:
                hollow = lbl == UNASSIGNED
                fig.patches.append(mpl.patches.Ellipse(
                    (fx(lx + 1.0), fy(yy)), 1.5 / FW, 1.5 / FH, transform=fig.transFigure,
                    fc="none" if hollow else c,
                    ec=c if hollow else ("#9a9a9a" if c == C_TRAIN else "none"),
                    lw=0.6 if hollow else 0.3))
                fig.text(fx(lx + 2.6), fy(yy), lbl, fontsize=PT_ROW, ha="left", va="center",
                         color=MUTED if lbl in (UNASSIGNED, "Reference split") else INK)
                yy += pitch
            yy += BGAP
    return fig, FW, FH, audit


if __name__ == "__main__":
    fig, FW, FH, audit = build()
    out = os.path.join(HERE, "SuppFig_S14")
    try:
        from export_figure import export_figure
        export_figure(fig, out, formats=["pdf", "png"], dpi=600,
                      size_inches=(FW * MM, FH * MM), grayscale_preview=True, tight=False)
    except ImportError:
        for ext in ("pdf", "png"):
            fig.savefig(f"{out}.{ext}", dpi=600)
    print("SuppFig_S14 ->", round(FW, 1), "x", round(FH, 1), "mm")
    print(pd.DataFrame(audit, columns=["dataset", "tool", "seed", "acc_py", "kappa_py",
                                       "kappa_R", "frac_unassigned", "frac_wrong",
                                       "wrong_same_lineage", "non_type_labels"])
          .to_string(index=False))
