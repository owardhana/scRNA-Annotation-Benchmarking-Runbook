"""
Supplementary Figures 9, 13 and 3.
S4   accuracy versus runtime and peak memory, both data types.  183 x 116 mm, 4 panels
S9   Type III eta-squared of log10(runtime), pooled + family.   183 x 94 mm,  4 panels
S10  per-family rare cell-type F1, both data types.                   183 x 72 mm,  2 panels
Data: ../_shared/tidy_kappa.csv. Runtime and memory analyses use run_status == "ok".
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from setup_style import setup_style
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from _eta_common import *

FONT="Liberation Sans"; PT_TICK, PT_LAB, PT_TAG, PT_ROW = 6, 7, 8, 5.8
INK, MUTED, RULE, LW = "#000000", "#4d4d4d", "#bfbfbf", 0.5
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":[FONT,"DejaVu Sans"],
    "font.size":PT_TICK,"axes.labelsize":PT_LAB,"xtick.labelsize":PT_TICK,
    "ytick.labelsize":PT_TICK,"axes.linewidth":LW,"axes.edgecolor":INK,
    "xtick.major.width":LW,"ytick.major.width":LW,
    "xtick.major.size":2.0,"ytick.major.size":2.0,
    "xtick.direction":"out","ytick.direction":"out",
    "axes.spines.top":False,"axes.spines.right":False,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "figure.dpi":600,"savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42,
    "svg.fonttype":"none","axes.grid":False,"figure.autolayout":False,
    "xtick.top":False,"ytick.right":False,
    "xtick.minor.visible":False,"ytick.minor.visible":False,
    "xtick.minor.size":0,"ytick.minor.size":0,
})
MM=1/25.4
import visual_qa
from export_figure import export_figure
d=load()
def famkey(fig,FW,FH,x0,y0,pitch=50.0,ncol=3,titlegap=4.4):
    fig.text(x0/FW,(FH-y0+titlegap)/FH,"Family",fontsize=PT_ROW,fontweight="bold",
             va="baseline",ha="left",color=INK)
    for i,f in enumerate(FAMILY_ORDER):
        col,row=i%ncol,i//ncol; x=x0+col*pitch; y=y0+row*5.0
        fig.lines.append(plt.Line2D([x/FW,(x+3.2)/FW],[(FH-y+0.9)/FH]*2,
            transform=fig.transFigure,color=FAMILY_COLORS[f],lw=1.4,solid_capstyle="butt"))
        fig.text((x+4.4)/FW,(FH-y)/FH,SHORT[f],fontsize=PT_ROW,va="baseline",ha="left",color=INK)

# =============================================== S13 (eta of runtime) (accuracy vs cost)
FW,FH=183.0,126.0
def bx(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s_,x,y): fig.text(x/FW,(FH-y)/FH,s_,fontsize=PT_TAG,fontweight="bold",
                          va="top",ha="left",color=INK,family=FONT)
g=d[d.run_status=="ok"]
panels=[("synth","runtime_s","Runtime (s)","a",18,10),("synth","peak_mem_mb","Peak memory (MB)","b",106,10),
        ("real","runtime_s","Runtime (s)","c",18,66),("real","peak_mem_mb","Peak memory (MB)","d",106,66)]
for arm,col,xlab,tg,x0,y0 in panels:
    ax=fig.add_axes(bx(x0,y0,66,40))
    s=g[g.arm==arm]
    m=s.groupby(["tool","scen","fam"]).agg(k=("kappa_imp0","mean"),v=(col,"mean")).reset_index()
    m=m[m.v>0]
    for f in FAMILY_ORDER:
        t=m[m.fam==f]
        ax.scatter(t.v,t.k,s=2.6,facecolor=FAMILY_COLORS[f],edgecolor="none",alpha=0.55,zorder=2)
    rho=spearmanr(np.log10(m.v),m.k)
    ax.set_xscale("log"); ax.set_xlim(m.v.min()/2.4,m.v.max()*2.4)
    ax.set_ylim(-0.04,1.06); ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0","0.25","0.50","0.75","1.00"])
    ax.set_xlabel(xlab,fontsize=PT_LAB,labelpad=2)
    ax.set_ylabel("Cohen's κ",fontsize=PT_LAB,labelpad=2)
    ax.set_title(("Synthetic data" if arm=="synth" else "Real data"),fontsize=PT_LAB,pad=3,color=INK)
    ax.tick_params(axis="both",which="both",top=False,right=False)
    _p = "p < 0.001" if rho.pvalue < 0.001 else f"p = {rho.pvalue:.3f}"
    ax.text(0.03,0.05,f"Spearman ρ = {rho.statistic:+.2f}  ({_p})",
            transform=ax.transAxes,va="bottom",ha="left",fontsize=PT_TICK,color=MUTED)
    tag(tg,x0-14,y0-5)
famkey(fig,FW,FH,18,121.5,pitch=33.0,ncol=5)
os.chdir(os.path.join(HERE,"S09")); print(visual_qa.print_report(visual_qa.audit_layout(fig)))
export_figure(fig,"SuppFig_S09",formats=["pdf","svg","png"],dpi=600,
              size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)

# =============================================== S9
FW,FH=183.0,100.0
def bx2(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag2(s_,x,y): fig.text(x/FW,(FH-y)/FH,s_,fontsize=PT_TAG,fontweight="bold",
                           va="top",ha="left",color=INK,family=FONT)
gg=d[d.run_status=="ok"].copy(); gg["lrt"]=np.log10(gg.runtime_s.where(gg.runtime_s>0))
ROWS=["log_cells","n_types","shannon","knn_purity"]
for j,(arm,ttl,x0,tg) in enumerate([("synth","Synthetic data",34,"a"),("real","Real data",118,"b")]):
    s=gg[gg.arm==arm].groupby(["tool","scen"]).agg({**{p:"mean" for p in P},"lrt":"mean"}).reset_index()
    r=type3(s,"lrt")
    ax=fig.add_axes(bx2(x0,12,54,26))
    for i,p_ in enumerate(ROWS):
        v,pv=r[p_]; y=len(ROWS)-1-i
        ax.barh(y,v,height=0.56,color="#7a7a7a",alpha=0.9,edgecolor="none")
        ax.text(v+0.014,y,f"{v:.3f}  {stars(pv)}",va="center",ha="left",fontsize=PT_ROW,color=INK)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([PLAB[p_] for p_ in ROWS][::-1],fontsize=PT_ROW)
    ax.set_xlim(0,0.9); ax.set_xticks([0,0.25,0.5,0.75]); ax.set_ylim(-0.6,len(ROWS)-0.4)
    ax.set_xlabel("η² of log₁₀ runtime, pooled",fontsize=PT_LAB,labelpad=2)
    ax.set_title(ttl,fontsize=PT_LAB,pad=3,color=INK)
    ax.tick_params(axis="y",length=0,pad=2); ax.tick_params(axis="x",top=False)
    ax.spines["left"].set_visible(False)
    tag2(tg,x0-24,6)
    axf=fig.add_axes(bx2(x0,54,54,26))
    for ri,p_ in enumerate(ROWS):
        base=len(ROWS)-1-ri
        for fi,f in enumerate(FAMILY_ORDER):
            sf=gg[(gg.arm==arm)&(gg.fam==f)].groupby(["tool","scen"]).agg(
                {**{p:"mean" for p in P},"lrt":"mean"}).reset_index()
            v,pv=type3(sf,"lrt")[p_]
            y=base+0.30-fi*0.15
            axf.barh(y,v,height=0.135,color=FAMILY_COLORS[f],alpha=0.92,edgecolor="none")
            if stars(pv)!="ns" and v>0.004:
                axf.text(v+0.012,y,stars(pv),va="center",ha="left",
                         fontsize=PT_ROW-1.2,color=MUTED)
        if ri: axf.axhline(base+0.46,color=RULE,lw=0.4)
    axf.set_yticks([len(ROWS)-1-i for i in range(len(ROWS))])
    axf.set_yticklabels([PLAB[p_] for p_ in ROWS],fontsize=PT_ROW)
    axf.set_xlim(0,0.9); axf.set_xticks([0,0.25,0.5,0.75]); axf.set_ylim(-0.55,len(ROWS)-0.45)
    axf.set_xlabel("η² of log₁₀ runtime, within family",fontsize=PT_LAB,labelpad=2)
    axf.tick_params(axis="y",length=0,pad=2); axf.tick_params(axis="x",top=False)
    axf.spines["left"].set_visible(False)
    tag2("c" if j==0 else "d",x0-24,48)
famkey(fig,FW,FH,34,94)
fig.text(178/FW,(FH-89.6)/FH,"*** p < 0.001   ** p < 0.01   * p < 0.05   ns not significant",
         fontsize=PT_ROW-0.4,va="baseline",ha="right",color=MUTED)
os.chdir(os.path.join(HERE,"S13")); print(visual_qa.print_report(visual_qa.audit_layout(fig)))
export_figure(fig,"SuppFig_S13",formats=["pdf","svg","png"],dpi=600,
              size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)

# =============================================== S3 (rare cell-type F1)
FW,FH=183.0,68.0
def bx3(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag3(s_,x,y): fig.text(x/FW,(FH-y)/FH,s_,fontsize=PT_TAG,fontweight="bold",
                           va="top",ha="left",color=INK,family=FONT)
rng=np.random.default_rng(5)
for j,(arm,ttl,x0,tg) in enumerate([("synth","Synthetic data",22,"a"),("real","Real data",110,"b")]):
    ax=fig.add_axes(bx3(x0,12,62,40))
    s=d[(d.arm==arm)&d.rare_f1.notna()]
    for i,f in enumerate(FAMILY_ORDER):
        v=s.loc[s.fam==f,"rare_f1"].values
        bp=ax.boxplot([v],positions=[i],widths=0.52,whis=(5,95),showfliers=False,
                      patch_artist=True,manage_ticks=False)
        bp["boxes"][0].set(facecolor=FAMILY_COLORS[f],alpha=0.28,
                           edgecolor=FAMILY_COLORS[f],linewidth=0.7)
        for kk in ("whiskers","caps"):
            for e in bp[kk]: e.set(color=FAMILY_COLORS[f],linewidth=0.7)
        bp["medians"][0].set(color=INK,linewidth=1.0)
        ax.scatter(i+rng.uniform(-0.16,0.16,len(v)),v,s=2.2,facecolor=FAMILY_COLORS[f],
                   edgecolor="none",alpha=0.5,zorder=3)
        ax.text(i,1.045,f"{np.median(v):.2f}",ha="center",va="bottom",fontsize=PT_ROW-0.4,color=MUTED)
    ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[f] for f in FAMILY_ORDER],
                                                fontsize=PT_ROW,rotation=45,ha="right")
    ax.set_ylim(-0.04,1.10); ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0","0.25","0.50","0.75","1.00"])
    ax.set_ylabel("Rare cell-type F1",fontsize=PT_LAB,labelpad=2)
    ax.set_title(ttl,fontsize=PT_LAB,pad=3,color=INK)
    ax.tick_params(axis="both",which="both",top=False,right=False)
    tag3(tg,x0-16,6)

os.chdir(os.path.join(HERE,"S03")); print(visual_qa.print_report(visual_qa.audit_layout(fig)))
export_figure(fig,"SuppFig_S03",formats=["pdf","svg","png"],dpi=600,
              size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
