"""
Figure 3 | The separability–accuracy relationship holds in every regime tested.
Nature style, double column 183 mm.

Data : _shared/tidy_kappa.csv    (per tool x dataset kappa; five cell-level families)
       _shared/tidy_regimes.csv  (database-marker + LLM ontology scores, per tool x dataset)
       _shared/tidy_fm.csv       (fine-tuned foundation models, per model x dataset)
       kNN purity per real dataset, from tidy_kappa.csv
All three tables reproduce every per-tool mean in the manuscript to <0.001.

v2 (2026-09-07): panel (a) added - family-mean kappa vs kNN purity for the five
cell-level families. Absorbs the orphaned "Fig. 2f" callout. Panels a-c -> b-d,
d,e -> e,f. Top row now reads as four regimes in sequence.

v3 (2026-10-26, journal submission): the "R² =" labels in a-d were mathtext, which set the
superscript 2 at 4.2 pt in Computer Modern - the smallest and the only non-Liberation text
in the figure. They are now Liberation Sans with a 5 pt superscript (r2_text), placed on
the old baselines; nothing else moves. Export is tight with a 0.05 in pad, which reproduces the canvas
the v2 files were published on (508.8 x 258.6 pt).
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"..","_shared"))
from setup_style import setup_style
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import figstyle as FS
FONT=FS.FONT
PT_TICK, PT_LAB, PT_TAG, PT_ROW = FS.PT_TICK, FS.PT_LAB, FS.PT_TAG, FS.PT_ROW
INK, MUTED, RULE, LW = FS.INK, FS.MUTED, FS.RULE, FS.LW
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":[FONT,"DejaVu Sans"],
    "font.size":PT_TICK,"axes.labelsize":PT_LAB,"xtick.labelsize":PT_TICK,
    "ytick.labelsize":PT_TICK,"legend.fontsize":PT_TICK,
    "axes.linewidth":LW,"axes.edgecolor":INK,
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
RAMP=FS.score_ramp()          # red = poor, blue = good, diverging about 0.5
C_DB, C_LLM, C_FM = FS.C_DB, FS.C_LLM, FS.C_FM
from catmap import FAMILY_ORDER, FAMILY_COLORS, MARKER, SIMIL, CLASSIC, DL, SEMI

MM=1/25.4; FW,FH=183.0,100.0
def box(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]

SH=os.path.join(HERE,"..","_shared")
kap=pd.read_csv(os.path.join(SH,"tidy_kappa.csv"))
reg=pd.read_csv(os.path.join(SH,"tidy_regimes.csv"))
fm =pd.read_csv(os.path.join(SH,"tidy_fm.csv"))
PUR={"S1":0.787,"S2":0.546,"S3":0.384,"S4":0.250,"S5":0.803,
     "S6":0.937,"S7":0.738,"S8":0.676,"S9":0.965}
SC=[f"S{i}" for i in range(1,10)]
DB_ORDER=["scType","SCINA","SCSA","scSorter","ScInfeR","scCATCH","clustifyr_hyper","clustifyr_jaccard"]
LLM_ORDER=["mLLMCelltype","CASSIA","GPTCelltype"]
FM_ORDER=["scFoundation","geneformer","scGPT","C2S","scBERT"]
FM_LAB={"scFoundation":"scFoundation","geneformer":"Geneformer V2","scGPT":"scGPT",
        "C2S":"C2S","scBERT":"scBERT"}

fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s,x,y): fig.text(x/FW,(FH-y)/FH,s,fontsize=PT_TAG,fontweight="bold",
                         va="top",ha="left",color=INK,family=FONT)

# ------------------------------------------------ a : cell-level families
FAM_SHORT=FS.SHORT
FAM_MAP={}
for _f,_l in [("Marker-Based",MARKER),("Similarity-Based",SIMIL),("Classic ML",CLASSIC),
              ("Deep Learning",DL),("Semi-Supervised",SEMI)]:
    for _t in _l: FAM_MAP[_t]=_f
kap["scen"]=kap.scenario.str.split("-").str[0]
kap["fam"]=kap.tool.map(FAM_MAP)
kr=kap[kap.arm=="real"]
PUR_R=kr.groupby("scen").knn_purity.mean().reindex(SC)
FAMK=kr.pivot_table(index="fam",columns="scen",values="kappa_imp0",aggfunc="mean").reindex(
        FAMILY_ORDER)[SC]

def frame(ax, ylab, title):
    ax.set_xlim(0.15,1.05); ax.set_ylim(0,1.0)
    ax.set_xticks([0.2,0.5,0.8]); ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0","0.25","0.50","0.75","1.00"])
    ax.set_xlabel("kNN purity", fontsize=PT_LAB, labelpad=2)
    ax.set_ylabel(ylab, fontsize=PT_LAB, labelpad=2)
    ax.set_title(title, fontsize=PT_LAB, pad=3, color=INK)
    ax.tick_params(axis="both",which="both",top=False,right=False)

# "R² = …" in the figure font. Baselines are the v2 mathtext ones: 3.44 pt below
# y = 0.955 of the axes, and 8.02 pt between the R² and ρ lines.
PT_SUP, SUP_RISE, R2_DROP, R2_LEAD = 5.0, 1.9, 3.444, 8.019
def r2_text(ax, value, second=None):
    kw=dict(textcoords="offset points",va="baseline",ha="left",color=INK,family=FONT)
    r=ax.annotate("R",xy=(0.03,0.955),xycoords="axes fraction",xytext=(0,-R2_DROP),
                  fontsize=PT_TICK,style="italic",**kw)
    two=ax.annotate("2",xy=(1,0.955),xycoords=(r,"axes fraction"),
                    xytext=(0.3,-R2_DROP+SUP_RISE),fontsize=PT_SUP,**kw)
    ax.annotate(f" = {value}",xy=(1,0.955),xycoords=(two,"axes fraction"),
                xytext=(0,-R2_DROP),fontsize=PT_TICK,**kw)
    if second:
        ax.annotate(second,xy=(0.03,0.955),xycoords="axes fraction",
                    xytext=(0,-R2_DROP-R2_LEAD),fontsize=PT_TICK,**kw)

axa=fig.add_axes(box(16.5, 11, 31.0, 31))
xs=np.linspace(0.18,1.02,10); r2s=[]
for f in FAMILY_ORDER:
    x=PUR_R.values; y=FAMK.loc[f].values; c=FAMILY_COLORS[f]
    axa.scatter(x,y,s=4.5,facecolor=c,edgecolor="white",linewidth=0.22,zorder=3)
    b,a0=np.polyfit(x,y,1); axa.plot(xs,a0+b*xs,color=c,lw=0.7,zorder=2)
    r2s.append(pearsonr(x,y)[0]**2)
frame(axa,"Family-mean κ","Cell-level families")
FS.family_key(axa, x=0.400, y=0.395, pitch=0.070, swatch=0.070, width=0.635, fontsize=5.0)
r2_text(axa,f"{min(r2s):.2f}–{max(r2s):.2f}")
tag("a",4.5,5)

# ------------------------------------------------ b,c,d : derivative regimes
def scatter(ax, series, colour, ylab, title):
    x=np.array([PUR[s] for s in series.index]); y=series.values
    ax.scatter(x,y,s=9,facecolor=colour,edgecolor="white",linewidth=0.3,zorder=3)
    b,a0=np.polyfit(x,y,1); xs=np.linspace(0.18,1.02,10)
    ax.plot(xs,a0+b*xs,color=colour,lw=0.8,zorder=2)
    rho=spearmanr(x,y); r=pearsonr(x,y)[0]
    frame(ax, ylab, title)
    r2_text(ax,f"{r**2:.2f}",f"ρ = {rho.statistic:.2f}")

db=reg[reg.family=="Database marker"].groupby("scenario").score.mean().reindex(SC)
lm=reg[reg.family=="LLM"].groupby("scenario").score.mean().reindex(SC)
fmm=fm.groupby("scenario").kappa.mean().reindex(SC)
for i,(ser,col,ylab,ttl,tg) in enumerate([
        (db, C_DB,"Ontology score","Public marker database","b"),
        (lm, C_LLM,"Ontology score","LLM annotation","c"),
        (fmm,C_FM,"Cohen's κ","Fine-tuned foundation models","d")]):
    ax=fig.add_axes(box(60.5+i*44.0, 11, 31.0, 31)); scatter(ax,ser,col,ylab,ttl)
    tag(tg, 48.5+i*44.0, 5)

# ------------------------------------------------ e : per-tool heatmap
axd=fig.add_axes(box(40, 55, 64, 33))
piv=reg.pivot_table(index="tool",columns="scenario",values="score").reindex(DB_ORDER+LLM_ORDER)[SC]
nr,nc=piv.shape
im=axd.pcolormesh(np.arange(nc+1)-0.5,np.arange(nr+1)-0.5,np.ma.masked_invalid(piv.values),
                  cmap=RAMP,vmin=0,vmax=1,shading="flat",linewidth=0,rasterized=False)
im.set_edgecolor("none"); axd.set_facecolor(FS.FAIL_GREY)
axd.set_xlim(-0.5,nc-0.5); axd.set_ylim(nr-0.5,-0.5)
axd.set_xticks(range(nc)); axd.set_xticklabels(SC)
axd.set_yticks(range(nr))
axd.set_yticklabels([t.replace("clustifyr_","clustifyr ") for t in piv.index], fontsize=PT_ROW)
axd.tick_params(axis="both",which="both",length=0,top=False,right=False)
axd.tick_params(axis="y",pad=11); axd.tick_params(axis="x",pad=2)
for sp in axd.spines.values(): sp.set_visible(False)
axd.axhline(len(DB_ORDER)-0.5,color="white",lw=1.0)
axd.set_xlabel("Real dataset",fontsize=PT_LAB,labelpad=2)
for lab,col,y0,y1 in [("Marker\ndatabase",C_DB,0,len(DB_ORDER)-1),
                      ("LLM",C_LLM,len(DB_ORDER),nr-1)]:
    axd.plot([-3.15,-3.15],[y0-0.38,y1+0.38],color=col,lw=1.8,
             transform=axd.transData,clip_on=False,solid_capstyle="butt")
    axd.text(-3.62,(y0+y1)/2,lab,transform=axd.transData,clip_on=False,rotation=90,
             ha="center",va="center",fontsize=PT_TICK,color=INK,linespacing=1.1)
caxd=fig.add_axes(box(107,56,1.9,19))
cbd=fig.colorbar(im,cax=caxd,ticks=[0,0.5,1]); cbd.outline.set_linewidth(LW)
cbd.ax.set_title("Score",fontsize=PT_TICK,pad=3,color=INK)
cbd.ax.tick_params(labelsize=PT_TICK,length=1.6,width=LW,pad=1.5)
tag("e",4,50)

# ------------------------------------------------ f : foundation models
axe=fig.add_axes(box(142, 55, 30, 18))
pf=fm.pivot_table(index="model",columns="scenario",values="kappa").reindex(FM_ORDER)[SC]
nr2,nc2=pf.shape
im2=axe.pcolormesh(np.arange(nc2+1)-0.5,np.arange(nr2+1)-0.5,np.ma.masked_invalid(pf.values),
                   cmap=RAMP,vmin=0,vmax=1,shading="flat",linewidth=0,rasterized=False)
im2.set_edgecolor("none"); axe.set_facecolor(FS.FAIL_GREY)
axe.set_xlim(-0.5,nc2-0.5); axe.set_ylim(nr2-0.5,-0.5)
axe.set_xticks(range(nc2)); axe.set_xticklabels(SC)
axe.set_yticks(range(nr2)); axe.set_yticklabels([FM_LAB[m] for m in pf.index],fontsize=PT_ROW)
axe.tick_params(axis="both",which="both",length=0,top=False,right=False)
axe.tick_params(axis="y",pad=10); axe.tick_params(axis="x",pad=2)
for sp in axe.spines.values(): sp.set_visible(False)
axe.set_xlabel("Real dataset",fontsize=PT_LAB,labelpad=2)
axe.plot([-6.35,-6.35],[-0.38,nr2-1+0.38],color=C_FM,lw=1.8,
         transform=axe.transData,clip_on=False,solid_capstyle="butt")
axe.text(-7.9,(nr2-1)/2,"Foundation\nmodel",transform=axe.transData,clip_on=False,
         rotation=90,ha="center",va="center",fontsize=PT_TICK,color=INK,linespacing=1.1)
caxe=fig.add_axes(box(175,56,1.9,14))
cbe=fig.colorbar(im2,cax=caxe,ticks=[0,0.5,1]); cbe.outline.set_linewidth(LW)
cbe.ax.set_title("κ",fontsize=PT_TICK,pad=3,color=INK)
cbe.ax.tick_params(labelsize=PT_TICK,length=1.6,width=LW,pad=1.5)
tag("f",115,50)

if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_fig3.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig,"Figure3",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=True,pad_inches=0.05)
