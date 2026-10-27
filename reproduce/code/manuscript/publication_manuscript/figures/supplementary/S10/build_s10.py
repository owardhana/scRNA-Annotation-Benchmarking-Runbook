"""
Supplementary Figure 10 | Runtime and peak memory by method family, real arm.
Nature style, 183 x 78 mm, two panels.

Data : ../../_shared/tidy_kappa.csv, arm == "real", run_status == "ok".
Failed runs are excluded here (unlike the kappa analyses): a run that produced no
prediction has no meaningful runtime or memory footprint.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__))
SH=os.path.join(HERE,"..","..","_shared"); sys.path.insert(0, SH)
from setup_style import setup_style
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from catmap import FAMILY_ORDER, FAMILY_COLORS, MARKER, SIMIL, CLASSIC, DL, SEMI

FONT="Liberation Sans"; PT_TICK, PT_LAB, PT_TAG, PT_ROW = 6, 7, 8, 6
INK, MUTED, LW = "#000000", "#4d4d4d", 0.5
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":[FONT,"DejaVu Sans"],
    "font.size":PT_TICK,"axes.labelsize":PT_LAB,"xtick.labelsize":PT_TICK,
    "ytick.labelsize":PT_TICK,"axes.linewidth":LW,"axes.edgecolor":INK,
    "xtick.major.width":LW,"ytick.major.width":LW,
    "xtick.major.size":2.0,"ytick.major.size":0,
    "xtick.direction":"out","ytick.direction":"out",
    "axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "figure.dpi":600,"savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42,
    "svg.fonttype":"none","axes.grid":False,"figure.autolayout":False,
    "xtick.top":False,"ytick.right":False,
    "xtick.minor.visible":False,"ytick.minor.visible":False,
    "xtick.minor.size":0,"ytick.minor.size":0,
})
MM=1/25.4; FW,FH=183.0,68.0
def box_(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
SHORT={"Marker-Based":"Marker-based","Similarity-Based":"Similarity-based",
       "Classic ML":"Classical ML","Deep Learning":"Deep learning",
       "Semi-Supervised":"Semi-supervised"}

FAM_MAP={}
for _f,_l in [("Marker-Based",MARKER),("Similarity-Based",SIMIL),("Classic ML",CLASSIC),
              ("Deep Learning",DL),("Semi-Supervised",SEMI)]:
    for _t in _l: FAM_MAP[_t]=_f
d=pd.read_csv(os.path.join(SH,"tidy_kappa.csv"))
d["fam"]=d.tool.map(FAM_MAP)
r=d[(d.arm=="real")&(d.run_status=="ok")]

fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s,x,y): fig.text(x/FW,(FH-y)/FH,s,fontsize=PT_TAG,fontweight="bold",
                         va="top",ha="left",color=INK,family=FONT)
rng=np.random.default_rng(11)

def panel(ax, col, xlab, ticks, ticklab, show_names):
    allv=r[col].dropna().values; allv=allv[allv>0]
    for i,f in enumerate(FAMILY_ORDER):
        v=r.loc[r.fam==f, col].dropna().values
        v=v[v>0]; y=len(FAMILY_ORDER)-1-i
        c=FAMILY_COLORS[f]
        bp=ax.boxplot([v], positions=[y], vert=False, widths=0.46, whis=(5,95),
                      showfliers=False, patch_artist=True, manage_ticks=False)
        bp["boxes"][0].set(facecolor=c, alpha=0.28, edgecolor=c, linewidth=0.7)
        for k in ("whiskers","caps"):
            for e in bp[k]: e.set(color=c, linewidth=0.7)
        bp["medians"][0].set(color=INK, linewidth=1.0)
        ax.scatter(v, y+rng.uniform(-0.14,0.14,len(v)), s=2.2, facecolor=c,
                   edgecolor="none", alpha=0.55, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(allv.min()/2.2, allv.max()*2.2)
    ax.set_xticks(ticks); ax.set_xticklabels(ticklab)
    ax.set_ylim(-0.62, len(FAMILY_ORDER)-0.38)
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_yticklabels([SHORT[f] for f in FAMILY_ORDER][::-1] if show_names else [""]*5,
                       fontsize=PT_ROW)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel(xlab, fontsize=PT_LAB, labelpad=2)
    ax.tick_params(axis="x", which="both", top=False)

axa=fig.add_axes(box_(26, 10, 58, 50))
panel(axa,"runtime_s","Runtime per tool and dataset (s)",
      [0.1,1,10,100,1000,10000],["0.1","1","10","10²","10³","10⁴"], True)
tag("a",4,5)

axb=fig.add_axes(box_(104, 10, 58, 50))
panel(axb,"peak_mem_mb","Peak memory per tool and dataset (MB)",
      [10,100,1000,10000],["10","100","10³","10⁴"], False)
tag("b",92,5)



if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_s10.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig,"SuppFig_S10",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
    for f in FAMILY_ORDER:
        s=r[r.fam==f]
        print(f"{f:18s} runtime med {s.runtime_s.median():8.1f} s | mem med {s.peak_mem_mb.median():7.1f} MB | n={len(s)}")
