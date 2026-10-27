"""
Supplementary Figure 5 | Type III variance decomposition of Cohen's kappa.  183 x 128 mm
(a, b) pooled across tools, with a tool-identity term.
(c, d) the same decomposition fitted separately within each cell-level family.
Both data types. Failed runs scored kappa = 0 (kappa_imp0), as everywhere in the study.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from setup_style import setup_style
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))
import figstyle as FS
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from _eta_common import *

FONT=FS.FONT; PT_TICK, PT_LAB, PT_TAG, PT_ROW = FS.PT_TICK, FS.PT_LAB, FS.PT_TAG, 5.8
INK, MUTED, RULE, LW = "#000000", "#4d4d4d", "#bfbfbf", 0.5
C_MAIN, C_TOOL = FS.TOL["indigo"], "#7a7a7a"
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
MM=1/25.4
d=load()

# ======================================================= S5 (a,b pooled; c,d per family)
FW,FH=183.0,134.0
def bx(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s_,x,y): fig.text(x/FW,(FH-y)/FH,s_,fontsize=PT_TAG,fontweight="bold",
                          va="top",ha="left",color=INK,family=FONT)

# ---- a, b : pooled across tools, with a tool-identity term -----------------
TERMS=["knn_purity","tool","log_cells","n_types","shannon"]
TLAB={**PLAB,"tool":"Tool identity"}
for arm,ttl,x0,tg in [("synth","Synthetic data",34,"a"),("real","Real data",118,"b")]:
    r=type3(unit(d,arm),"kappa_imp0",with_tool=True)
    ax=fig.add_axes(bx(x0,12,54,32))
    for i,t in enumerate(TERMS):
        v,p=r[t]; y=len(TERMS)-1-i
        ax.barh(y,v,height=0.58,color=C_TOOL if t=="tool" else C_MAIN,alpha=0.9,edgecolor="none")
        ax.text(v+0.016,y,f"{v:.3f}  {stars(p)}",va="center",ha="left",fontsize=PT_ROW,color=INK)
    ax.axvline(r["_residual"][0],color=RULE,lw=0.7,ls=(0,(3,2)),zorder=1)
    ax.text(r["_residual"][0],-0.72,f"residual {r['_residual'][0]:.2f}",ha="center",va="top",
            fontsize=PT_ROW-0.4,color=MUTED)
    ax.set_yticks(range(len(TERMS)))
    ax.set_yticklabels([TLAB[t] for t in TERMS][::-1],fontsize=PT_ROW)
    ax.set_xlim(0,0.92); ax.set_xticks([0,0.25,0.5,0.75]); ax.set_ylim(-0.6,len(TERMS)-0.4)
    ax.set_xlabel("η² of Cohen's κ, pooled",fontsize=PT_LAB,labelpad=2)
    ax.set_title(ttl,fontsize=PT_LAB,pad=3,color=INK)
    ax.tick_params(axis="y",length=0,pad=2); ax.tick_params(axis="x",top=False)
    tag(tg,x0-24,6)

# ---- c, d : the same decomposition fitted inside each family ---------------
res={arm:{f:type3(unit(d,arm,fam=f),"kappa_imp0") for f in FAMILY_ORDER}
     for arm in ["synth","real"]}
ROWS=["knn_purity","log_cells","n_types","shannon"]
for arm,ttl,x0,tg in [("synth","Synthetic data",34,"c"),("real","Real data",118,"d")]:
    ax=fig.add_axes(bx(x0,60,54,56))
    for ri,p_ in enumerate(ROWS):
        base=len(ROWS)-1-ri
        for fi,f in enumerate(FAMILY_ORDER):
            v,pv=res[arm][f][p_]
            y=base+0.30-fi*0.15
            ax.barh(y,v,height=0.135,color=FAMILY_COLORS[f],alpha=0.92,edgecolor="none")
            if stars(pv)!="ns" and v>0.004:
                ax.text(v+0.012,y,stars(pv),va="center",ha="left",fontsize=PT_ROW-1.2,color=MUTED)
        if ri: ax.axhline(base+0.46,color=RULE,lw=0.4)
    ax.set_yticks([len(ROWS)-1-i for i in range(len(ROWS))])
    ax.set_yticklabels([PLAB[p_] for p_ in ROWS],fontsize=PT_ROW)
    ax.set_xlim(0,0.95); ax.set_xticks([0,0.25,0.5,0.75]); ax.set_ylim(-0.55,len(ROWS)-0.45)
    ax.set_xlabel("η² of Cohen's κ, within family",fontsize=PT_LAB,labelpad=2)
    ax.set_title(ttl,fontsize=PT_LAB,pad=3,color=INK)
    ax.tick_params(axis="y",length=0,pad=2); ax.tick_params(axis="x",top=False)
    tag(tg,x0-24,54)

fig.text(34/FW,(FH-126.0)/FH,"Family",fontsize=PT_ROW,fontweight="bold",
         va="baseline",ha="left",color=INK)
for i,f in enumerate(FAMILY_ORDER):
    x=34.0+i*29.0; y=131.0
    fig.lines.append(plt.Line2D([x/FW,(x+3.2)/FW],[(FH-y+0.9)/FH]*2,transform=fig.transFigure,
                     color=FAMILY_COLORS[f],lw=1.4,solid_capstyle="butt"))
    fig.text((x+4.4)/FW,(FH-y)/FH,SHORT[f],fontsize=PT_ROW,va="baseline",ha="left",color=INK)
fig.text(178/FW,(FH-126.0)/FH,"*** p < 0.001   ** p < 0.01   * p < 0.05   ns not significant",
         fontsize=PT_ROW-0.4,va="baseline",ha="right",color=MUTED)

import visual_qa
from export_figure import export_figure
os.chdir(os.path.join(HERE,"S05"))
print(visual_qa.print_report(visual_qa.audit_layout(fig)))
export_figure(fig,"SuppFig_S05",formats=["pdf","svg","png"],dpi=600,
              size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
