#!/usr/bin/env python3
"""Regenerate Fig. 5d / phase3_platform_shift_overview.png from Profiling_P9.csv.

Plot code is lifted verbatim from make_plots() in Phase3_Profiling.ipynb so the
figure stays visually identical to the rest of the phase-3 plot set; only the
input numbers change (corrected PBMCbench depth metrics).
Both depth axes are medians: Median_Library_Per_Cell and Median_Genes_Per_Cell.

Sizing goes through phase3_figstyle: the canvas shrinks so the type is larger on
the page, and the height is corrected so the six panels keep their area.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import phase3_figstyle as fs

# Fig. 5d places this at 58 mm, so 10 pt printed at 1.5 pt. The bars are in data
# coordinates and rescale on their own, but the three x labels are the limit:
# at 0.55 "Pancreas" and "PBMCbench" run into each other, at 0.65 they clear.
fs.K = 0.65

BASE = Path(__file__).resolve().parent
PLOT_DIR = BASE / "Plots"
PLOT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "Profiling_P9.csv")

sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

block_order = ["CellBench", "Pancreas", "PBMCbench"]
metric_panel = [
    ("Median_Library_Per_Cell", "Median library per cell\n(UMI or read count)", True),
    ("Dropout_Pct",             "Dropout (% zeros)",                            False),
    ("Median_Genes_Per_Cell",   "Median genes detected per cell",               False),
    ("KNN_Purity",              "kNN purity (class separability)",              False),
    ("Mean_Pi_Score",           "Mean π (DE strength)",                         True),
]

W0, H0 = 15, 9                       # the original canvas
W1, H1 = fs.scaled_figsize(W0, H0)


def draw(figsize):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    legend_dict = {}

    for i, (col, label, logy) in enumerate(metric_panel):
        ax = axes[i]
        sub = df.dropna(subset=[col])
        sns.barplot(data=sub, x="Block", y=col, hue="Platform", order=block_order, ax=ax)
        ax.set_title(label, fontname='Helvetica', fontweight='bold', fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(col, fontname='Helvetica', fontsize=10)
        if logy:
            ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            legend_dict.setdefault(l, h)
        ax.get_legend().remove()

    ax_legend = axes[-1]
    ax_legend.axis("off")
    if legend_dict:
        ax_legend.legend(handles=list(legend_dict.values()), labels=list(legend_dict.keys()),
                         loc="center", title="Platform", title_fontsize=11, fontsize=10,
                         frameon=True)

    plt.tight_layout()
    return fig


fig = draw((W1, H1))                 # first pass: measure what the furniture took
st  = fs.stretch_height(fig, H1)
plt.close(fig)
fig = draw((W1, H1 * st))            # second pass: the one we keep

out = PLOT_DIR / "phase3_platform_shift_overview.png"
plt.savefig(out, dpi=1200, bbox_inches="tight")
plt.close(fig)
fs.report(out.name, W0, H0, W1, H1 * st, placed_mm=58)

# echo the numbers that landed in the figure, for the manuscript table
cols = ["Sample", "Block", "Platform", "Median_Library_Per_Cell", "Dropout_Pct",
        "Median_Genes_Per_Cell", "KNN_Purity", "Mean_Pi_Score"]
print(df[cols].to_string(index=False))
