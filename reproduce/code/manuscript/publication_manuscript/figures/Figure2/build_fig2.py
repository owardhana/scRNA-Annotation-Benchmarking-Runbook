"""
Figure 2 | Dataset structure, not method family, governs annotation accuracy.
Nature style, double column 183 mm. Built via scipilot-figure-skill.

Data   : results/_interaction_analysis/tidy_kappa.csv
Metric : kappa_imp0 (failed runs scored 0) — reproduces the manuscript family means
         to <0.001. Panel a shows raw kappa; failed runs drawn grey.
Family : current post-2026-08-27 map (catmap.py, from P5/Plots_Analysis.R)

Panel order follows reading order: a top-left, then b/c/d down the right column.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, numpy as np, pandas as pd
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))
from setup_style import setup_style
import matplotlib as mpl
info = setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from catmap import CATMAP, FAMILY_ORDER, FAMILY_COLORS
import figstyle as FS

SHORT = {"Marker-Based": "Marker-based", "Similarity-Based": "Similarity-based",
         "Classic ML": "Classical ML", "Deep Learning": "Deep learning",
         "Semi-Supervised": "Semi-supervised"}

FONT = FS.FONT
PT_TICK, PT_LAB, PT_TAG, PT_ROW = FS.PT_TICK, FS.PT_LAB, FS.PT_TAG, 5
INK, MUTED, RULE, LW = FS.INK, FS.MUTED, FS.RULE, FS.LW
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
    "font.size": PT_TICK, "axes.labelsize": PT_LAB,
    "xtick.labelsize": PT_TICK, "ytick.labelsize": PT_TICK, "legend.fontsize": PT_TICK,
    "axes.linewidth": LW, "axes.edgecolor": INK,
    "xtick.major.width": LW, "ytick.major.width": LW,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.dpi": 600, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False, "figure.autolayout": False,
    "xtick.top": False, "ytick.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
})
RAMP = FS.score_ramp()      # red = poor, blue = good, diverging about kappa 0.5

MM = 1/25.4
FW, FH = 183.0, 118.0   # V33: +8 mm for author-name column labels (ZZ #26)
def box(x, y_top, w, h): return [x/FW, (FH - y_top - h)/FH, w/FW, h/FH]

# V33: read the same tidy file as Figure 4 (includes the failed SCINA run on S3)
DATA = os.environ.get("TIDY_KAPPA", os.path.join(HERE, "..", "_shared", "tidy_kappa.csv"))
d = pd.read_csv(DATA)
d["fam"] = d.tool.map(CATMAP); assert d.fam.notna().all()
d["arm_lab"] = d.arm.map({"synth": "Synthetic", "real": "Real"})
d["sid"] = d.scenario.str.extract(r"^(S\d)")[0]
toolmean = d.groupby(["arm_lab", "fam", "tool"]).kappa_imp0.mean().reset_index()

fig = plt.figure(figsize=(FW*MM, FH*MM)); fig.patch.set_facecolor("white")

# ============================================================ a — per-tool heatmap
axa = fig.add_axes(box(31, 8, 47, 81))
real = d[d.arm == "real"]
order = (real.groupby(["fam", "tool"]).kappa_imp0.mean().reset_index()
         .assign(f=lambda x: pd.Categorical(x.fam, FAMILY_ORDER, ordered=True))
         .sort_values(["f", "kappa_imp0"], ascending=[True, False]))
tools = order.tool.tolist(); fam_of = dict(zip(order.tool, order.fam))
mat = real.pivot_table(index="tool", columns="sid", values="kappa", aggfunc="mean").reindex(tools)
nr, nc = mat.shape
im = axa.pcolormesh(np.arange(nc+1)-0.5, np.arange(nr+1)-0.5,
                    np.ma.masked_invalid(mat.values), cmap=RAMP, vmin=0, vmax=1,
                    shading="flat", linewidth=0, rasterized=False)
im.set_edgecolor("none"); axa.set_facecolor(FS.FAIL_GREY)
axa.set_xlim(-0.5, nc-0.5); axa.set_ylim(nr-0.5, -0.5)
AUTHOR = {"S1": "Darmanis", "S2": "Marques", "S3": "Nowakowski", "S4": "Grün",
          "S5": "Tabula Muris", "S6": "He", "S7": "Zhao", "S8": "Zheng", "S9": "MacParland"}
axa.set_xticks(range(nc))
axa.set_xticklabels([f"{AUTHOR[c]} ({c})" for c in mat.columns], rotation=90,
                    ha="center", va="top", fontsize=PT_ROW)
axa.set_yticks(range(nr))
axa.set_yticklabels([t.replace("_", " ") for t in tools], fontsize=PT_ROW)
axa.tick_params(axis="both", which="both", length=0, top=False, right=False)
axa.tick_params(axis="y", pad=10.5); axa.tick_params(axis="x", pad=2)
for sp in axa.spines.values(): sp.set_visible(False)
axa.set_xlabel("Real dataset", fontsize=PT_LAB, labelpad=2)
for i, t in enumerate(tools):
    axa.add_patch(Rectangle((-1.15, i-0.5), 0.45, 1.0, transform=axa.transData,
                            clip_on=False, facecolor=FAMILY_COLORS[fam_of[t]], edgecolor="none"))
for i in range(1, nr):
    if fam_of[tools[i]] != fam_of[tools[i-1]]:
        axa.axhline(i-0.5, color="white", lw=0.9)
        axa.plot([-1.15, -0.70], [i-0.5]*2, color="white", lw=0.9,
                 transform=axa.transData, clip_on=False)
cax = fig.add_axes(box(80.5, 8, 2.0, 26))
cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 0.5, 1])
cb.ax.set_title("κ", fontsize=PT_LAB, pad=3, color=INK)
cb.outline.set_linewidth(LW)
cb.ax.tick_params(labelsize=PT_TICK, length=1.6, width=LW, pad=1.5)

# ============================================================ legend (one tidy row)
axl = fig.add_axes(box(97, 3, 82, 15)); axl.axis("off")
# matplotlib fills legend columns top-to-bottom; reorder so the entries read
# left-to-right in family order
_ORD = [FAMILY_ORDER[i] for i in (0, 3, 1, 4, 2)]
handles = [Line2D([], [], marker="s", linestyle="none", markersize=2.8,
                  markerfacecolor=FAMILY_COLORS[f], markeredgecolor="none")
           for f in _ORD]
leg = axl.legend(handles, [SHORT[f] for f in _ORD], ncol=3, loc="upper left",
                 bbox_to_anchor=(0, 1), frameon=True,
                 title="Family", handletextpad=0.45, columnspacing=1.8, borderaxespad=0,
                 borderpad=0.55, labelspacing=0.55, handlelength=0.9, fontsize=PT_TICK)
leg.get_frame().set(edgecolor=RULE, linewidth=LW, facecolor="white")
leg.get_title().set(fontsize=PT_TICK, fontweight="bold", color=INK)
leg._legend_box.align = "left"
for t in leg.get_texts(): t.set_color(INK)

# ============================================================ b — per-tool means
axb = fig.add_axes(box(97, 21, 84, 27))
rng = np.random.default_rng(11)
x = 0.0; centres = {}
for ai, arm in enumerate(["Synthetic", "Real"]):
    for f in FAMILY_ORDER:
        v = toolmean[(toolmean.arm_lab == arm) & (toolmean.fam == f)].kappa_imp0.values
        axb.scatter(x + rng.uniform(-0.20, 0.20, len(v)), v, s=4.5,
                    facecolor=FAMILY_COLORS[f], edgecolor="white", linewidth=0.2, zorder=3)
        axb.plot([x-0.34, x+0.34], [np.median(v)]*2, color=INK, lw=0.9, zorder=4,
                 solid_capstyle="butt")
        centres.setdefault(arm, []).append(x); x += 1
    if ai == 0: sep = x - 0.2; x += 1.4
axb.axvline(sep, color=RULE, lw=LW)
axb.set_ylim(0.27, 0.93); axb.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axb.set_xticks([]); axb.set_xlim(-0.75, x - 0.35)
axb.set_ylabel("Mean κ per tool", fontsize=PT_LAB, labelpad=2)
axb.text(np.mean(centres["Synthetic"]), 0.935, "Synthetic", ha="center", va="bottom",
         fontsize=PT_TICK, color=INK)
axb.text(np.mean(centres["Real"]), 0.935, "Real", ha="center", va="bottom",
         fontsize=PT_TICK, color=INK)
axb.tick_params(axis="both", which="both", top=False, right=False)

# ============================================================ c — paired scatter
axc = fig.add_axes(box(97, 58, 84, 26))
tm = toolmean.pivot_table(index=["tool", "fam"], columns="arm_lab",
                          values="kappa_imp0").reset_index()
axc.plot([0.30, 0.95], [0.30, 0.95], color=RULE, lw=LW, zorder=1)
for f in FAMILY_ORDER:
    q = tm[tm.fam == f]
    axc.scatter(q["Synthetic"], q["Real"], s=7, facecolor=FAMILY_COLORS[f],
                edgecolor="white", linewidth=0.25, zorder=3)
axc.set_xlim(0.30, 0.95); axc.set_ylim(0.30, 0.95)
axc.set_xticks([0.4, 0.6, 0.8]); axc.set_yticks([0.4, 0.6, 0.8])
axc.tick_params(axis="both", which="both", top=False, right=False)
axc.set_xlabel("Mean κ, synthetic", fontsize=PT_LAB, labelpad=2)
axc.set_ylabel("Mean κ, real", fontsize=PT_LAB, labelpad=2)
from scipy.stats import spearmanr as _sp
_p = tm.dropna(subset=["Synthetic", "Real"])
_rho = _sp(_p["Synthetic"], _p["Real"]).statistic
axc.text(0.97, 0.05, f"ρ = {_rho:.2f}\nn = {len(_p)} tools", transform=axc.transAxes,
         ha="right", va="bottom", fontsize=PT_TICK, color=INK, linespacing=1.35)
print(f"[fig2c] rho={_rho:.4f}  n={len(_p)}")

# ============================================================ d — variance components
axd = fig.add_axes(box(97, 92, 84, 5.0))
comp = [(n, v, FS.VARIANCE_COLORS[n]) for n, v in
        [("Dataset", 83.9), ("Tool × dataset", 9.4), ("Tool", 4.3), ("Replicate", 2.3)]]
left = 0.0
for name, val, col in comp:
    axd.barh([0], [val], left=left, height=1.0, color=col, edgecolor="white", linewidth=0.8)
    left += val
axd.set_xlim(0, 100); axd.set_ylim(-0.5, 0.5); axd.set_yticks([]); axd.set_xticks([])
axd.text(83.9/2, 0, "84%", ha="center", va="center", fontsize=PT_TICK,
         color="white", fontweight="bold")
for sp in axd.spines.values(): sp.set_visible(False)
axk = fig.add_axes(box(97, 98, 84, 8)); axk.axis("off")
for (name, val, col), xx in zip(comp, [0.0, 0.225, 0.545, 0.735]):
    axk.add_patch(Rectangle((xx, 0.60), 0.024, 0.26, transform=axk.transAxes,
                            facecolor=col, edgecolor="none", clip_on=False))
    axk.text(xx+0.033, 0.73, f"{name} {val:.0f}%", transform=axk.transAxes,
             va="center", fontsize=PT_TICK, color=INK)

# ============================================================ panel labels
for s, xmm, ymm in [("a", 3, 7), ("b", 89, 16), ("c", 89, 53), ("d", 89, 87)]:
    fig.text(xmm/FW, (FH-ymm)/FH, s, fontsize=PT_TAG, fontweight="bold",
             va="top", ha="left", color=INK, family=FONT)

if __name__ == "__main__":
    import visual_qa
    visual_qa.render_preview(fig, "_preview_fig2.png", dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig, "Figure2", formats=["pdf", "svg", "png"], dpi=600,
                  size_inches=(FW*MM, FH*MM), grayscale_preview=True, tight=False)
    for arm in ["Synthetic", "Real"]:
        g = toolmean[toolmean.arm_lab == arm].groupby("fam").kappa_imp0
        print(arm, {k: f"n={len(v)} med={v.median():.3f}" for k, v in g})
