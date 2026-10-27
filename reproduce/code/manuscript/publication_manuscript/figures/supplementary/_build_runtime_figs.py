"""
Supplementary Figure 12 | Per-tool runtime for the three derivative regimes, one figure.
(a) database-driven marker family  ../../_shared/tidy_db_runtime.csv   (P7a runtime_dataset.tsv)
(b) LLM family                     ../../_shared/tidy_llm_runtime.csv  (P7b runtime_dataset.tsv)
(c) fine-tuned foundation models    ../../_shared/tidy_fm_runtime.csv  (P8b real_FM_results.tsv)
All three share one log runtime axis, so the orders-of-magnitude gap between the regimes
is readable directly off the figure.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__))
SH=os.path.join(HERE,"..","_shared"); sys.path.insert(0, SH)
from setup_style import setup_style
import figstyle as FS
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt

FONT=FS.FONT; PT_TICK, PT_LAB, PT_ROW, PT_TAG = FS.PT_TICK, FS.PT_LAB, 6, FS.PT_TAG
INK, MUTED, LW = "#000000", "#4d4d4d", 0.5
C_DB, C_LLM, C_FM = FS.C_DB, FS.C_LLM, FS.C_FM
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


def panel(ax, series, colour, rng):
    """series: list of (label, values), slowest at top."""
    series=sorted(series, key=lambda kv: np.median(kv[1]))
    for i,(lab,v) in enumerate(series):
        bp=ax.boxplot([v],positions=[i],vert=False,widths=0.48,whis=(0,100),
                      showfliers=False,patch_artist=True,manage_ticks=False)
        bp["boxes"][0].set(facecolor=colour,alpha=0.28,edgecolor=colour,linewidth=0.7)
        for k in ("whiskers","caps"):
            for e in bp[k]: e.set(color=colour,linewidth=0.7)
        bp["medians"][0].set(color=INK,linewidth=1.0)
        ax.scatter(v,i+rng.uniform(-0.15,0.15,len(v)),s=3.2,facecolor=colour,
                   edgecolor="none",alpha=0.6,zorder=3)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([l for l,_ in series],fontsize=PT_ROW)
    ax.set_ylim(-0.62,len(series)-0.38)
    ax.tick_params(axis="y",length=0,pad=2)
    return series

if __name__=="__main__":
    db=pd.read_csv(os.path.join(SH,"tidy_db_runtime.csv"))
    DBLAB={"clustifyr_hyper_database":"clustifyr hypergeometric","clustifyr_jaccard_database":"clustifyr Jaccard"}
    db["lab"]=db.tool.map(lambda t: DBLAB.get(t, t.replace("_database","")))
    llm=pd.read_csv(os.path.join(SH,"tidy_llm_runtime.csv"))
    fm=pd.read_csv(os.path.join(SH,"tidy_fm_runtime.csv"))
    FMLAB={"scFoundation":"scFoundation","geneformer":"Geneformer V2","scGPT":"scGPT",
           "C2S":"C2S †","scBERT":"scBERT"}
    fm["lab"]=fm.model.map(FMLAB)

    BLOCKS=[("a","Public marker database", C_DB,
             [(l,g["runtime"].values) for l,g in db.groupby("lab")]),
            ("b","LLM annotation", C_LLM,
             [(l,g["runtime"].values) for l,g in llm.groupby("tool")]),
            ("c","Fine-tuned foundation models", C_FM,
             [(l,g["runtime_s"].values) for l,g in fm.groupby("lab")])]

    allv=np.concatenate([v for _,_,_,ser in BLOCKS for _,v in ser])
    XLIM=(allv.min()/2.6, allv.max()*2.6)

    FW=120.0; LEFT=44.0; ROW=5.6; TOP=9.0; TITLE=4.6; GAP=6.0; XLAB=9.0
    heights=[len(ser)*ROW for *_ ,ser in BLOCKS]
    FH=TOP+sum(h+TITLE+GAP for h in heights)-GAP+XLAB
    fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
    rng=np.random.default_rng(3)
    y=TOP
    for (tg,ttl,col,ser),h in zip(BLOCKS,heights):
        ax=fig.add_axes([LEFT/FW,(FH-y-TITLE-h)/FH,(FW-LEFT-6)/FW,h/FH])
        ax.set_title(ttl,fontsize=PT_LAB,pad=3,color=INK,loc="left")
        s2=panel(ax,ser,col,rng)
        ax.set_xscale("log"); ax.set_xlim(*XLIM)
        last = (tg=="c")
        ax.tick_params(axis="x",which="both",top=False,
                       labelbottom=last,bottom=last,length=2.0 if last else 0)
        if last: ax.set_xlabel("Runtime per dataset (s)",fontsize=PT_LAB,labelpad=2)
        for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_visible(last)
        fig.text((LEFT-40)/FW,(FH-y)/FH,tg,fontsize=PT_TAG,fontweight="bold",
                 va="top",ha="left",color=INK,family=FONT)
        print(f"== {tg} {ttl}")
        for l,v in s2[::-1]:
            print(f"   {l:26s} n={len(v)} min={np.min(v):8.1f} med={np.median(v):9.1f} max={np.max(v):9.1f}")
        y += TITLE+h+GAP

    os.chdir(os.path.join(HERE,"S12"))
    import visual_qa
    from export_figure import export_figure
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    export_figure(fig,"SuppFig_S12",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
