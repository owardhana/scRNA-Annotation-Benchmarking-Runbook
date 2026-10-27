"""
Supplementary Figure 6 | Cluster-level reference: CIPR and cluster-level scmap.
Nature style, 130 x 66 mm, two heatmap panels on one shared colour scale.

Data : ../../_shared/tidy_cluster_ref_tidy.csv, extracted from the reference-based
       per-scenario CSVs under "results/P4 Synthetic Datasets V1" and
       "results/P5 Real Life Datasets V2". Synthetic values are the mean over three
       replicate seeds. A cell with no valid prediction is drawn grey and scored kappa = 0
       in the panel mean, matching Supplementary Fig. S9.
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

FONT="Liberation Sans"; PT_TICK, PT_LAB, PT_TAG, PT_ROW = 6, 7, 8, 6
INK, LW = "#000000", 0.5
RAMP=FS.score_ramp()   # red = poor, blue = good
_unused=LinearSegmentedColormap.from_list(
    "blue",["#f5f9fe","#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d2a52"]).copy()
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
MM=1/25.4; FW,FH=130.0,66.0
def bx(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
ORDER=["CIPR","scmap_cluster"]; LAB={"scmap_cluster":"scmap (cluster)","CIPR":"CIPR"}
SC=[f"S{i}" for i in range(1,10)]

t=pd.read_csv(os.path.join(SH,"tidy_cluster_ref_tidy.csv"))
fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s_,x,y): fig.text(x/FW,(FH-y)/FH,s_,fontsize=PT_TAG,fontweight="bold",
                          va="top",ha="left",color=INK,family=FONT)
for j,(arm,ttl,y0,tg,xlab) in enumerate([("synth","Synthetic data",12,"a","Scenario"),
                                         ("real","Real data",42,"b","Real dataset")]):
    s=t[t.arm==arm]
    piv=s.pivot_table(index="tool",columns="scen",values="k").reindex(ORDER)[SC]
    ok =s.pivot_table(index="tool",columns="scen",values="n_ok").reindex(ORDER)[SC]
    M=np.where((ok.values==0)|np.isnan(ok.values), np.nan, piv.values)
    ax=fig.add_axes(bx(30,y0,72,14))
    im=ax.pcolormesh(np.arange(10)-0.5,np.arange(3)-0.5,np.ma.masked_invalid(M),
                     cmap=RAMP,vmin=0,vmax=1,shading="flat")
    im.set_edgecolor("none"); ax.set_facecolor("white")
    ax.set_xlim(-0.5,8.5); ax.set_ylim(1.5,-0.5)
    ax.set_xticks(range(9)); ax.set_xticklabels(SC,fontsize=PT_ROW)
    ax.set_yticks(range(2)); ax.set_yticklabels([LAB[t_] for t_ in ORDER],fontsize=PT_ROW)
    ax.tick_params(axis="both",which="both",length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel(xlab,fontsize=PT_LAB,labelpad=2)
    full=np.where(np.isnan(M),0.0,M)
    ax.set_title(f"{ttl}   mean κ = {np.mean(full):.3f}",fontsize=PT_LAB,pad=3,color=INK)
    tag(tg,4,y0-6)
cax=fig.add_axes(bx(107,18,2.0,34))
cb=fig.colorbar(im,cax=cax,ticks=[0,0.5,1]); cb.outline.set_linewidth(LW)
cb.ax.set_title("κ",fontsize=PT_TICK,pad=3,color=INK)
cb.ax.tick_params(labelsize=PT_TICK,length=1.6,width=LW,pad=1.5)

if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_s06.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig,"SuppFig_S06",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
    for arm in ["synth","real"]:
        s=t[t.arm==arm]
        print(arm, {r.tool: round(r.k,3) for r in s.groupby("tool",as_index=False).k.mean().itertuples()},
              "failures:", int((s.n_ok==0).sum()))
