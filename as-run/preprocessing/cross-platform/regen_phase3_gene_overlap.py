#!/usr/bin/env python3
"""Regenerate Fig. 5c / phase3_gene_overlap_<block>.png from Phase3_GeneOverlap.csv.

Plot code lifted from make_plots() in Phase3_Profiling.ipynb so the figures stay
visually identical to the rest of the phase-3 set. Reading the saved Jaccard
table avoids re-profiling the raw count matrices, which the notebook does.

Sizing goes through phase3_figstyle: the canvas shrinks so the type is larger on
the page, the height is corrected so the heatmap keeps its area, and the in-cell
numbers take sqrt(K) so they do not collide.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import phase3_figstyle as fs

# These heatmaps take a GENTLER correction than the phase-3 overview. Their row
# labels ("PBMCBench Drop-seq-8cl", "Segerstolpe-Pancreas-2016") are long, and
# at print size they eat the width the cells need -- Fig. 5c places them at only
# 35 mm each, three across. Rendered at 0.90 / 0.80 / 0.70 / 0.55: clean down to
# 0.80, cell values touching by 0.70, unreadable at 0.55.
fs.K = 0.80

BASE = Path(__file__).resolve().parent
PLOT_DIR = BASE / "Plots"
PLOT_DIR.mkdir(exist_ok=True)

jacc_df = pd.read_csv(BASE / "Phase3_GeneOverlap.csv")

sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

for block, sub in jacc_df.groupby("Block"):
    names = sorted(set(sub["Dataset_A"]).union(sub["Dataset_B"]))
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for _, r in sub.iterrows():
        mat.loc[r["Dataset_A"], r["Dataset_B"]] = r["Jaccard"]
        mat.loc[r["Dataset_B"], r["Dataset_A"]] = r["Jaccard"]

    w0, h0 = max(5, 0.7 * len(names)), max(4, 0.7 * len(names))   # as in the notebook
    w1, h1 = fs.scaled_figsize(w0, h0)

    def draw(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(mat.astype(float), annot=True, fmt=".2f",
                    cmap="viridis", vmin=0, vmax=1, ax=ax,
                    annot_kws={"size": fs.annot_size(10)},
                    cbar_kws={"label": "Jaccard of detected genes"})
        ax.set_title(f"Cross-platform gene overlap — {block}",
                     fontname='Helvetica', fontweight='bold', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontname='Helvetica', rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontname='Helvetica', rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_fontname('Helvetica')
        for lab in cbar.ax.get_yticklabels():
            lab.set_fontname('Helvetica')
        plt.tight_layout()
        return fig

    fig = draw((w1, h1))                       # first pass: measure the furniture
    st  = fs.stretch_height(fig, h1)
    plt.close(fig)
    fig = draw((w1, h1 * st))                  # second pass: the real one

    out = PLOT_DIR / f"phase3_gene_overlap_{block.lower()}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fs.report(out.name, w0, h0, w1, h1 * st)
