"""
Supplementary Figure 7 | Full marker family under derived markers, synthetic versus real.
Nature style, 183 x 88 mm, two heatmap panels with one shared colour scale.

Data : ../../_shared/tidy_marker_derived.csv, built from the per-scenario CSVs in
       "results/P6a Marker - Synthetic/Data" and "results/P6b Marker - Real/Data".
       Synthetic values are the mean over three replicate seeds. A tool x scenario cell
       that produced no valid prediction is left white, not scored 0.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__))
SH=os.path.join(HERE,"..","..","_shared"); sys.path.insert(0, SH)
from setup_style import setup_style
import figstyle as FS
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FONT="Liberation Sans"; PT_TICK, PT_LAB, PT_TAG, PT_ROW = 6, 7, 8, 5.8
INK, MUTED, LW = "#000000", "#4d4d4d", 0.5
RAMP=FS.score_ramp()   # red = poor, blue = good
_unused=LinearSegmentedColormap.from_list(
    "blue",["#f5f9fe","#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d2a52"])
del _unused
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":[FONT,"DejaVu Sans"],
    "font.size":PT_TICK,"axes.labelsize":PT_LAB,"xtick.labelsize":PT_TICK,
    "ytick.labelsize":PT_TICK,"axes.linewidth":LW,"axes.edgecolor":INK,
    "figure.dpi":600,"savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42,
    "svg.fonttype":"none","axes.grid":False,"figure.autolayout":False,
    "xtick.top":False,"ytick.right":False,
    "xtick.minor.visible":False,"ytick.minor.visible":False,
    "xtick.minor.size":0,"ytick.minor.size":0,
})
MM=1/25.4; FW,FH=183.0,74.0
def bx(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
ORDER=["scType","SCSA","scCATCH","SCINA","ScInfeR","scSorter","clustifyr_hyper","clustifyr_jaccard"]
LAB={"clustifyr_hyper":"clustifyr hypergeometric","clustifyr_jaccard":"clustifyr Jaccard"}
DAG={"SCINA","scSorter"}
SC=[f"S{i}" for i in range(1,10)]

t=pd.read_csv(os.path.join(SH,"tidy_marker_derived.csv"))
fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s_,x,y): fig.text(x/FW,(FH-y)/FH,s_,fontsize=PT_TAG,fontweight="bold",
                          va="top",ha="left",color=INK,family=FONT)
for j,(arm,ttl,x0,tg) in enumerate([("synth","Synthetic data",44,"a"),("real","Real data",118,"b")]):
    s=t[t.arm==arm]
    piv=s.pivot_table(index="tool",columns="scen",values="k").reindex(ORDER)[SC]
    ok =s.pivot_table(index="tool",columns="scen",values="n_ok").reindex(ORDER)[SC]
    M=np.where((ok.values==0)|np.isnan(ok.values), np.nan, piv.values)
    ax=fig.add_axes(bx(x0,12,46,52))
    im=ax.pcolormesh(np.arange(10)-0.5,np.arange(9)-0.5,np.ma.masked_invalid(M),
                     cmap=RAMP,vmin=0,vmax=1,shading="flat")
    im.set_edgecolor("none"); ax.set_facecolor("white")
    ax.set_xlim(-0.5,8.5); ax.set_ylim(7.5,-0.5)
    ax.set_xticks(range(9)); ax.set_xticklabels(SC,fontsize=PT_ROW)
    ax.set_yticks(range(8))
    ax.set_yticklabels([LAB.get(t_,t_)+(" †" if t_ in DAG else "") for t_ in ORDER],
                       fontsize=PT_ROW) if j==0 else ax.set_yticklabels([])
    ax.tick_params(axis="both",which="both",length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("Scenario" if arm=="synth" else "Real dataset",fontsize=PT_LAB,labelpad=2)
    full=np.where(np.isnan(M),0.0,M)
    ax.set_title(f"{ttl}   family mean κ = {np.mean(full):.3f}",fontsize=PT_LAB,pad=3,color=INK)
    tag(tg,x0-38 if j==0 else x0-14,6)
cax=fig.add_axes(bx(168,16,2.0,26))
cb=fig.colorbar(im,cax=cax,ticks=[0,0.5,1]); cb.outline.set_linewidth(LW)
cb.ax.set_title("κ",fontsize=PT_TICK,pad=3,color=INK)
cb.ax.tick_params(labelsize=PT_TICK,length=1.6,width=LW,pad=1.5)


if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_s07.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig,"SuppFig_S07",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
