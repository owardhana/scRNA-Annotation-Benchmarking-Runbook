"""
Supplementary Figure 1 | Annotation accuracy on the synthetic data.
(a) Per-tool Cohen's kappa across the nine synthetic scenarios — the synthetic
    counterpart of Figure 2a, on the same colour scale so the two can be read together.
(b) Family-mean kappa against kNN purity — the synthetic counterpart of Figure 3a.
Nature style, 183 x 126 mm.

Data   : ../../_shared/tidy_kappa.csv  (arm == "synth")
Metric : (a) raw kappa averaged over the three replicate seeds (42, 123, 999); a tool x
         scenario cell with no valid prediction is drawn grey and scored kappa = 0 in
         every mean, as elsewhere in the study.  (b) kappa_imp0, family means per scenario.
Rows   : the 47 tools entering the cell-level comparison, grouped by family and ordered
         within family by synthetic mean kappa.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE = os.path.dirname(os.path.abspath(__file__))
SH = os.path.join(HERE, "..", "..", "_shared"); sys.path.insert(0, SH)
from setup_style import setup_style
import figstyle as FS
import matplotlib as mpl
info = setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from catmap import CATMAP, FAMILY_ORDER, FAMILY_COLORS

FONT = FS.FONT
PT_TICK, PT_LAB, PT_TAG, PT_ROW, PT_KEY = FS.PT_TICK, FS.PT_LAB, FS.PT_TAG, 5.5, 6
INK, MUTED, RULE, LW = FS.INK, FS.MUTED, FS.RULE, FS.LW
mpl.rcParams.update(FS.rc())
RAMP = FS.score_ramp()

MM = 1/25.4
FW, FH = 183.0, 126.0
def box(x, y_top, w, h): return [x/FW, (FH - y_top - h)/FH, w/FW, h/FH]
def tag(s, x, y): fig.text(x/FW, (FH-y)/FH, s, fontsize=PT_TAG, fontweight="bold",
                           va="top", ha="left", color=INK, family=FONT)

d = pd.read_csv(os.path.join(SH, "tidy_kappa.csv"))
d["fam"] = d.tool.map(CATMAP); assert d.fam.notna().all()
d["sid"] = d.scenario.str.extract(r"^(S\d)")[0]
syn = d[d.arm == "synth"]
SC = [f"S{i}" for i in range(1, 10)]

fig = plt.figure(figsize=(FW*MM, FH*MM)); fig.patch.set_facecolor("white")

# ----------------------------------------------------------------- a : heatmap
order = (syn.groupby(["fam", "tool"]).kappa_imp0.mean().reset_index()
         .assign(f=lambda x: pd.Categorical(x.fam, FAMILY_ORDER, ordered=True))
         .sort_values(["f", "kappa_imp0"], ascending=[True, False]))
tools = order.tool.tolist(); fam_of = dict(zip(order.tool, order.fam))
mat = syn.pivot_table(index="tool", columns="sid", values="kappa",
                      aggfunc="mean").reindex(tools)
nr, nc = mat.shape

axa = fig.add_axes(box(34, 10, 52, 104))
im = axa.pcolormesh(np.arange(nc+1)-0.5, np.arange(nr+1)-0.5,
                    np.ma.masked_invalid(mat.values), cmap=RAMP, vmin=0, vmax=1,
                    shading="flat", linewidth=0, rasterized=False)
im.set_edgecolor("none"); axa.set_facecolor(FS.FAIL_GREY)
axa.set_xlim(-0.5, nc-0.5); axa.set_ylim(nr-0.5, -0.5)
axa.set_xticks(range(nc)); axa.set_xticklabels(mat.columns)
axa.set_yticks(range(nr))
axa.set_yticklabels([t.replace("_", " ") for t in tools], fontsize=PT_ROW)
axa.tick_params(axis="both", which="both", length=0, top=False, right=False)
axa.tick_params(axis="y", pad=13); axa.tick_params(axis="x", pad=2)
for sp in axa.spines.values(): sp.set_visible(False)
axa.set_xlabel("Synthetic scenario", fontsize=PT_LAB, labelpad=2)
for i, t in enumerate(tools):
    axa.add_patch(Rectangle((-0.62, i-0.5), 0.28, 1.0, transform=axa.transData,
                            clip_on=False, facecolor=FAMILY_COLORS[fam_of[t]],
                            edgecolor="none"))
for i in range(1, nr):
    if fam_of[tools[i]] != fam_of[tools[i-1]]:
        axa.axhline(i-0.5, color="white", lw=0.9)
        axa.plot([-0.62, -0.34], [i-0.5]*2, color="white", lw=0.9,
                 transform=axa.transData, clip_on=False)
cax = fig.add_axes(box(89.0, 10, 2.2, 28))
cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 0.5, 1])
cb.ax.set_title("κ", fontsize=PT_LAB, pad=3, color=INK)
cb.outline.set_linewidth(LW)
cb.ax.tick_params(labelsize=PT_TICK, length=1.6, width=LW, pad=1.5)
tag("a", 4, 6)

# ----------------------------------------------------------------- b : purity fit
PUR = syn.groupby("sid").knn_purity.mean().reindex(SC)
FAMK = syn.pivot_table(index="fam", columns="sid", values="kappa_imp0",
                       aggfunc="mean").reindex(FAMILY_ORDER)[SC]
axb = fig.add_axes(box(119, 12, 56, 56))
xs = np.linspace(PUR.min(), PUR.max(), 10); stats = {}
for f in FAMILY_ORDER:
    x = PUR.values; y = FAMK.loc[f].values; c = FAMILY_COLORS[f]
    axb.scatter(x, y, s=8, facecolor=c, edgecolor="white", linewidth=0.3, zorder=3)
    b, a0 = np.polyfit(x, y, 1); axb.plot(xs, a0 + b*xs, color=c, lw=0.8, zorder=2)
    stats[f] = (pearsonr(x, y)[0]**2, spearmanr(x, y).statistic)
axb.set_xlim(-0.02, 1.06); axb.set_ylim(-0.02, 1.06)
axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axb.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
axb.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
axb.set_xlabel("kNN purity", fontsize=PT_LAB, labelpad=2)
axb.set_ylabel("Family-mean Cohen's κ", fontsize=PT_LAB, labelpad=2)
axb.tick_params(axis="both", which="both", top=False, right=False)
tag("b", 100, 6)

# family / R^2 / rho key, under panel (b) — doubles as the colour key for panel (a)
KX, KY, ROW = 108.0, 87.0, 6.4
fig.text(KX/FW, (FH-KY+3.4)/FH, "Family", fontsize=PT_KEY, fontweight="bold",
         va="bottom", ha="left", color=INK)
fig.text((KX+30.0)/FW, (FH-KY+3.4)/FH, "$R^2$", fontsize=PT_KEY, fontweight="bold",
         va="bottom", ha="right", color=INK)
fig.text((KX+38.0)/FW, (FH-KY+3.4)/FH, "ρ", fontsize=PT_KEY, fontweight="bold",
         va="bottom", ha="right", color=INK)
fig.lines.append(plt.Line2D([KX/FW, (KX+38.0)/FW], [(FH-KY+2.6)/FH]*2,
                            transform=fig.transFigure, color=RULE, lw=LW))
for i, f in enumerate(FAMILY_ORDER):
    yy = FH - (KY + i*ROW)
    fig.lines.append(plt.Line2D([KX/FW, (KX+3.4)/FW], [(yy+1.0)/FH]*2,
                     transform=fig.transFigure, color=FAMILY_COLORS[f], lw=1.4,
                     solid_capstyle="butt"))
    fig.text((KX+4.6)/FW, yy/FH, FS.SHORT[f], fontsize=PT_KEY, va="baseline",
             ha="left", color=INK)
    r2, rho = stats[f]
    fig.text((KX+30.0)/FW, yy/FH, f"{r2:.2f}", fontsize=PT_KEY, va="baseline",
             ha="right", color=INK)
    fig.text((KX+38.0)/FW, yy/FH, f"{rho:.2f}", fontsize=PT_KEY, va="baseline",
             ha="right", color=INK)

if __name__ == "__main__":
    import visual_qa
    visual_qa.render_preview(fig, "_preview_s01.png", dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig, "SuppFig_S01", formats=["pdf", "svg", "png"], dpi=600,
                  size_inches=(FW*MM, FH*MM), grayscale_preview=True, tight=False)
    fam = syn.groupby(["fam", "tool"]).kappa_imp0.mean().groupby("fam")
    print("synthetic family means:", {k: f"{v.mean():.3f}" for k, v in fam})
    print("failed cells:", int(mat.isna().sum().sum()), "of", nr*nc)
    print("R2 range: %.2f-%.2f   rho range: %.2f-%.2f" % (
        min(v[0] for v in stats.values()), max(v[0] for v in stats.values()),
        min(v[1] for v in stats.values()), max(v[1] for v in stats.values())))
