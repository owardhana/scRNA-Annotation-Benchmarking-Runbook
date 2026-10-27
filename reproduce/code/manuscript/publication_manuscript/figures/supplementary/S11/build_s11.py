"""
Supplementary Figure 11 | Runtime and memory scaling with dataset size, real arm.
Nature style, 183 x 132 mm, four panels.

Data : ../../_shared/tidy_kappa.csv        (a, b - cell-level families, run_status == "ok")
       ../../_shared/tidy_llm_runtime.csv  (c - P7b with_score/runtime_dataset.tsv)
       ../../_shared/tidy_fm_runtime.csv   (d - P8b data/real_FM_results.tsv)

Reported statistics reproduce the manuscript: LLM rho = 0.849, p = 0.0038 on dataset means
(raw runtime), R2 = 0.721 for log10(runtime) ~ cell-type number; foundation-model log-log
slope = 0.964, R2 = 0.996 on dataset means.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__))
SH=os.path.join(HERE,"..","..","_shared"); sys.path.insert(0, SH)
from setup_style import setup_style
import figstyle as FS
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from catmap import FAMILY_ORDER, FAMILY_COLORS, MARKER, SIMIL, CLASSIC, DL, SEMI

FONT="Liberation Sans"; PT_TICK, PT_LAB, PT_TAG, PT_ROW = 6, 7, 8, 6
INK, MUTED, LW = "#000000", "#4d4d4d", 0.5
C_LLM, C_FM = FS.C_LLM, FS.C_FM
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
MM=1/25.4; FW,FH=183.0,136.0
def bx(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]
SHORT={"Marker-Based":"Marker-based","Similarity-Based":"Similarity-based",
       "Classic ML":"Classical ML","Deep Learning":"Deep learning",
       "Semi-Supervised":"Semi-supervised"}

FAM_MAP={}
for _f,_l in [("Marker-Based",MARKER),("Similarity-Based",SIMIL),("Classic ML",CLASSIC),
              ("Deep Learning",DL),("Semi-Supervised",SEMI)]:
    for _t in _l: FAM_MAP[_t]=_f
d=pd.read_csv(os.path.join(SH,"tidy_kappa.csv")); d["fam"]=d.tool.map(FAM_MAP)
d["scen"]=d.scenario.str.split("-").str[0]
r=d[(d.arm=="real")&(d.run_status=="ok")]
meta=d[d.arm=="real"].groupby("scen").agg(log_cells=("log_cells","first"),
                                          n_types=("n_types","first"))
llm=pd.read_csv(os.path.join(SH,"tidy_llm_runtime.csv"))
fm =pd.read_csv(os.path.join(SH,"tidy_fm_runtime.csv"))

fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s,x,y): fig.text(x/FW,(FH-y)/FH,s,fontsize=PT_TAG,fontweight="bold",
                         va="top",ha="left",color=INK,family=FONT)
def note(ax,txt):
    ax.text(0.03,0.965,txt,transform=ax.transAxes,va="top",ha="left",
            fontsize=PT_TICK,color=INK,linespacing=1.35)

# ---- a, b : cell-level families -------------------------------------------
def fam_panel(ax, col, ylab):
    slopes={}
    for f in FAMILY_ORDER:
        s=r[(r.fam==f)]; s=s[s[col]>0]
        x=s.log_cells.values; y=np.log10(s[col].values); c=FAMILY_COLORS[f]
        ax.scatter(x,y,s=2.4,facecolor=c,edgecolor="none",alpha=0.5,zorder=2)
        b,a0=np.polyfit(x,y,1); slopes[f]=b
        xs=np.linspace(x.min(),x.max(),8)
        ax.plot(xs,a0+b*xs,color=c,lw=0.9,zorder=3)
    ax.set_xlim(2.55,4.42); ax.set_xticks([2.7,3.0,3.3,3.6,3.9,4.2])
    ax.set_xticklabels(["500","1k","2k","4k","8k","16k"])
    ax.set_xlabel("Cells in dataset (log scale)",fontsize=PT_LAB,labelpad=2)
    ax.set_ylabel(ylab,fontsize=PT_LAB,labelpad=2)
    ax.tick_params(axis="both",which="both",top=False,right=False)
    return slopes

axa=fig.add_axes(bx(17,10,66,44))
sl_rt=fam_panel(axa,"runtime_s","Runtime per dataset (s, log scale)")
axa.set_yticks([-1,0,1,2,3,4]); axa.set_ylim(-1.4,4.6)
axa.set_yticklabels(["0.1","1","10","100","1,000","10,000"])
tag("a",4,5)

axb=fig.add_axes(bx(104,10,66,44))
sl_mm=fam_panel(axb,"peak_mem_mb","Peak memory (MB, log scale)")
axb.set_yticks([1,2,3,4]); axb.set_ylim(0.5,4.7)
axb.set_yticklabels(["10","100","1,000","10,000"])
tag("b",91,5)

# family key: two rows below (a, b), each entry carrying its two fitted slopes
fig.text(17/FW,(FH-66.0)/FH,"Fitted log–log slope (runtime / peak memory)",
         fontsize=PT_ROW-0.5,va="baseline",ha="left",color=MUTED)
for i,f in enumerate(FAMILY_ORDER):
    col,row = i % 3, i // 3
    x = 17.0 + col*56.0; y = 71.5 + row*5.6
    fig.lines.append(plt.Line2D([x/FW,(x+3.4)/FW],[(FH-y+1.0)/FH]*2,
                     transform=fig.transFigure,color=FAMILY_COLORS[f],lw=1.4,
                     solid_capstyle="butt"))
    fig.text((x+4.6)/FW,(FH-y)/FH,f"{SHORT[f]}   {sl_rt[f]:.2f} / {sl_mm[f]:.2f}",
             fontsize=PT_ROW,va="baseline",ha="left",color=INK)


def panelkey(ax, colour, label):
    """Small in-panel key: a colour dot, a faint dot, and the series name."""
    ax.scatter([0.72],[0.115],s=11,facecolor=colour,edgecolor="white",linewidth=0.4,
               transform=ax.transAxes,clip_on=False,zorder=6)
    ax.scatter([0.72],[0.045],s=3.0,facecolor=colour,edgecolor="none",alpha=0.35,
               transform=ax.transAxes,clip_on=False,zorder=6)
    ax.text(0.755,0.115,label+" (dataset mean)",transform=ax.transAxes,va="center",
            ha="left",fontsize=PT_ROW-0.3,color=INK)
    ax.text(0.755,0.045,"individual runs",transform=ax.transAxes,va="center",
            ha="left",fontsize=PT_ROW-0.3,color=MUTED)

# ---- c : LLM runtime vs cell-type number ----------------------------------
axc=fig.add_axes(bx(17,84,66,44))
llm=llm.join(meta,on="scen")
gm=llm.groupby("scen").agg(rt=("runtime","mean")).join(meta)
axc.scatter(llm.n_types,np.log10(llm["runtime"]),s=3.0,facecolor=C_LLM,
            edgecolor="none",alpha=0.35,zorder=2)
axc.scatter(gm.n_types,np.log10(gm.rt),s=11,facecolor=C_LLM,edgecolor="white",
            linewidth=0.4,zorder=4)
b,a0=np.polyfit(gm.n_types,np.log10(gm.rt),1)
xs=np.linspace(gm.n_types.min(),gm.n_types.max(),8)
axc.plot(xs,a0+b*xs,color=C_LLM,lw=0.9,zorder=3)
rho=spearmanr(gm.n_types,gm.rt); r2=pearsonr(gm.n_types,np.log10(gm.rt))[0]**2
axc.set_xlim(0,53); axc.set_xticks([0,10,20,30,40,50])
axc.set_ylim(0.2,4.0); axc.set_yticks([1,2,3,4])
axc.set_yticklabels(["10","100","1,000","10,000"])
axc.set_xlabel("Cell types in dataset",fontsize=PT_LAB,labelpad=2)
axc.set_ylabel("Runtime per dataset (s, log scale)",fontsize=PT_LAB,labelpad=2)
axc.tick_params(axis="both",which="both",top=False,right=False)
note(axc,f"ρ = {rho.statistic:.2f}  (p = {rho.pvalue:.3f})\n$R^2$ = {r2:.2f}")
panelkey(axc, C_LLM, "LLM")
tag("c",4,79)

# ---- d : foundation-model runtime vs cell count ---------------------------
axd=fig.add_axes(bx(104,84,66,44))
axd.scatter(fm.log_cells,np.log10(fm.runtime_s),s=3.0,facecolor=C_FM,
            edgecolor="none",alpha=0.35,zorder=2)
gf=fm.groupby("scen").agg(rt=("runtime_s","mean")).join(meta)
axd.scatter(gf.log_cells,np.log10(gf.rt),s=11,facecolor=C_FM,edgecolor="white",
            linewidth=0.4,zorder=4)
bf,af=np.polyfit(gf.log_cells,np.log10(gf.rt),1)
xs=np.linspace(gf.log_cells.min(),gf.log_cells.max(),8)
axd.plot(xs,af+bf*xs,color=C_FM,lw=0.9,zorder=3)
r2f=pearsonr(gf.log_cells,np.log10(gf.rt))[0]**2
axd.set_xlim(2.55,4.42); axd.set_xticks([2.7,3.0,3.3,3.6,3.9,4.2])
axd.set_xticklabels(["500","1k","2k","4k","8k","16k"])
axd.set_ylim(1.4,4.6); axd.set_yticks([2,3,4])
axd.set_yticklabels(["100","1,000","10,000"])
axd.set_xlabel("Cells in dataset (log scale)",fontsize=PT_LAB,labelpad=2)
axd.set_ylabel("Runtime per dataset (s, log scale)",fontsize=PT_LAB,labelpad=2)
axd.tick_params(axis="both",which="both",top=False,right=False)
note(axd,f"log–log slope = {bf:.2f}\n$R^2$ = {r2f:.3f}")
panelkey(axd, C_FM, "Foundation model")
tag("d",91,79)

if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_s11.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig,"SuppFig_S11",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
    print("runtime slopes:",{f:round(sl_rt[f],3) for f in FAMILY_ORDER})
    print("memory  slopes:",{f:round(sl_mm[f],3) for f in FAMILY_ORDER})
    print(f"LLM rho={rho.statistic:.3f} p={rho.pvalue:.4f} R2={r2:.3f} | FM slope={bf:.3f} R2={r2f:.3f}")
