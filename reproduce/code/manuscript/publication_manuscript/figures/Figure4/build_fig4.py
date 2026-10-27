"""
Figure 4 | Method selection is constrained by available resources, not by achievable accuracy.
Nature style, double column 183 mm. A table-figure, drawn — not typed in slides.

Sources (all recomputed, none transcribed):
  _shared/supp_table4_roster.csv  Supplementary Table 4, extracted from the supp PDF
  tidy_kappa.csv                  real-arm kappa / runtime / memory, 5 cell-level families
  _shared/tidy_regimes.csv        ontology scores, database-marker + LLM
  _shared/tidy_fm.csv             foundation models
  runtime_dataset.tsv (P7a, P7b)  database-marker and LLM runtimes

V33 (ZZ #34): adds mean accuracy, macro F1 and MCC for every kappa-scored family, from
_shared/tidy_kappa_metrics.csv (tidy_kappa.csv + accuracy/MCC; failed runs scored 0, as kappa).

Cohen's kappa and the ontology score are DIFFERENT measurements and are never plotted on a
shared axis — they get two separate range strips.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, numpy as np, pandas as pd
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__))
SH=os.path.join(HERE,"..","_shared"); sys.path.insert(0, SH)
from setup_style import setup_style
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from catmap import CATMAP

import figstyle as FS
FONT=FS.FONT; PT_TICK, PT_LAB, PT_TAG = FS.PT_TICK, FS.PT_LAB, FS.PT_TAG
INK, MUTED, RULE, LW = FS.INK, FS.MUTED, FS.RULE, FS.LW
TRACK = "#eceef0"
mpl.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":[FONT,"DejaVu Sans"],
    "font.size":PT_TICK,"axes.linewidth":LW,"text.color":INK,
    "figure.dpi":600,"savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42,
    "svg.fonttype":"none","figure.autolayout":False,
})
MM=1/25.4; FW, FH = 183.0, 75.0
FAMCOL=FS.FAMILY_COLORS
# Strip marks take the row's family colour, so nothing in this figure uses a
# colour that means "good" or "bad" elsewhere in the paper.

# ---------------------------------------------------------------- data
tk=pd.read_csv(os.environ.get("TIDY_KAPPA", os.path.join(SH,"tidy_kappa_metrics.csv")))
tk["fam"]=tk.tool.map(CATMAP); real=tk[tk.arm=="real"]
tm=real.groupby(["fam","tool"]).kappa_imp0.mean()
cost=real.groupby("fam").agg(rt=("runtime_s","median"), mem=("peak_mem_mb","median"))
reg=pd.read_csv(os.path.join(SH,"tidy_regimes.csv"))
fm =pd.read_csv(os.path.join(SH,"tidy_fm.csv"))
rt_db=pd.read_csv(os.path.join(SH,"tidy_db_runtime.csv"))
rt_llm=pd.read_csv(os.path.join(SH,"tidy_llm_runtime.csv"))
rt_llm.columns=[c.strip() for c in rt_llm.columns]

def band(series): return series.mean(), series.min(), series.max()
K={f:band(tm.loc[f]) for f in ["Marker-Based","Similarity-Based","Classic ML","Deep Learning","Semi-Supervised"]}
KFM=band(fm.groupby("model").kappa.mean())
def band_reg(sub):
    """Mean over datasets (as Fig. 3b and the text); range spans the tools."""
    per_tool=sub.groupby("tool").score.mean()
    return sub.groupby("scenario").score.mean().mean(), per_tool.min(), per_tool.max()
Odb=band_reg(reg[reg.family=="Database marker"])
Ollm=band_reg(reg[reg.family=="LLM"])
# other metrics: mean over tools of the per-tool mean across the nine real datasets
def other(fam):
    t=real[real.fam==fam].groupby("tool")[["accuracy_imp0","macro_f1","mcc_imp0"]].mean()
    return tuple(t.mean())
M={f:other(f) for f in ["Marker-Based","Similarity-Based","Classic ML","Deep Learning","Semi-Supervised"]}
MFM=tuple(fm.groupby("model")[["accuracy","macro_f1","mcc"]].mean().mean())

# family, roster n, needs(ref,mkr,clu,gpu), kappa band, ontology band, runtime s, note
ROWS=[
 ("Marker-based",     8, (0,1,0,0), K["Marker-Based"],   Odb,  rt_db["runtime"].median(),
  "CPU; no reference\nneeded", M["Marker-Based"]),
 ("Similarity-based", 8, (1,0,0,0), K["Similarity-Based"],None, cost.loc["Similarity-Based","rt"],
  "CPU; native to\nSeurat/Scanpy", M["Similarity-Based"]),
 ("Classical ML",    25, (1,0,0,0), K["Classic ML"],     None, cost.loc["Classic ML","rt"],
  "CPU; stable\ndependencies", M["Classic ML"]),
 ("Deep learning",    8, (1,0,0,1), K["Deep Learning"],  None, cost.loc["Deep Learning","rt"],
  "GPU; fast-moving\nDL stack", M["Deep Learning"]),
 ("Semi-supervised",  6, (1,0,0,1), K["Semi-Supervised"],None, cost.loc["Semi-Supervised","rt"],
  "GPU; fast-moving\nDL stack", M["Semi-Supervised"]),
 ("LLM",              3, (0,1,1,0), None,                Ollm, rt_llm["runtime"].median(),
  "API only; no local\ncompute", None),
 ("Foundation model", 5, (1,0,0,1), KFM,                 None, fm.runtime_s.median(),
  "GPU; large model\ndownload", MFM),
]

fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,FW); ax.set_ylim(FH,0); ax.axis("off")
def T(x,y,s,size=PT_TICK,ha="left",va="center",col=INK,weight="normal",rot=0):
    ax.text(x,y,s,fontsize=size,ha=ha,va=va,color=col,fontweight=weight,rotation=rot,
            family=FONT,linespacing=1.15)

XF = 5.0
# The per-family tool/configuration counts moved to Figure 1b, which reports them
# per evaluation setting. Dropping the column here frees ~8 mm; every column right
# of "Family" shifts left by that amount and "Accessibility" gains the slack.
DOTX=[36.0, 44.5, 53.0, 61.5]
KX0, KX1, KVAL = 67.0, 83.0, 89.5
MX = [95.5, 104.5, 113.0]                # accuracy, macro F1, MCC (means)
OX0, OX1, OVAL = 120.0, 136.0, 142.5
XRT   = 149.5
XNOTE = 157.0
RIGHT = 180.0
HY1, HY2, HY3 = 8.0, 12.2, 14.7          # header, sub-label line 1, line 2
AXY, NUMY     = 18.4, 21.4               # score mini-axis and its numbers
Y0, RH        = 23.4, 6.6

# ---- headers
T(XF, HY1, "Family", weight="bold", size=PT_LAB)
T((DOTX[0]+DOTX[3])/2, HY1, "Requires", weight="bold", size=PT_LAB, ha="center")
ax.plot([DOTX[0]-4.0,DOTX[3]+4.0],[HY1+1.7]*2,color=RULE,lw=LW)
for x,lab in zip(DOTX,["Labeled\nreference","Marker\nset","Cluster\nlabels","GPU"]):
    T(x, HY2+1.2, lab, ha="center", col=MUTED)
T((KX0+KX1)/2, HY1, "Cohen's κ", weight="bold", size=PT_LAB, ha="center")
T((OX0+OX1)/2, HY1, "Ontology score", weight="bold", size=PT_LAB, ha="center")
T((KX0+KX1)/2, HY2, "matched inputs", ha="center", col=MUTED)
T((OX0+OX1)/2, HY2, "external inputs", ha="center", col=MUTED)
T((MX[0]+MX[2])/2, HY1, "Other metrics", weight="bold", size=PT_LAB, ha="center")
ax.plot([MX[0]-4.6,MX[2]+3.2],[HY1+1.7]*2,color=RULE,lw=LW)
for x,lab in zip(MX,["Accuracy","Macro\nF1","MCC"]):
    T(x, HY2+1.2, lab, ha="center", col=MUTED)
for x in MX: T(x, NUMY, "mean", ha="center", col=MUTED, size=5.5)
T(XRT, HY1, "Median", weight="bold", size=PT_LAB, ha="center")
T(XRT, HY1+2.6, "runtime", weight="bold", size=PT_LAB, ha="center")
T(XNOTE, HY1, "Accessibility", weight="bold", size=PT_LAB)

# ---- one explicit 0-1 axis per score strip, so the strips are read as scales
for x0,x1,xv in [(KX0,KX1,KVAL),(OX0,OX1,OVAL)]:
    ax.plot([x0,x1],[AXY]*2,color=MUTED,lw=LW)
    for v in (0,0.5,1.0):
        xx=x0+(x1-x0)*v
        ax.plot([xx,xx],[AXY,AXY-1.1],color=MUTED,lw=LW)
        T(xx, NUMY, f"{v:g}", ha="center", col=MUTED, size=5.5)
    T(xv, NUMY, "mean", ha="right", col=MUTED, size=5.5)
ax.plot([XF,RIGHT],[Y0]*2, color=INK, lw=0.7)

for i,(fam,_n,need,kb,ob,rt,note,om) in enumerate(ROWS):   # _n now lives in Fig. 1b
    y=Y0+i*RH+3.4
    ax.add_patch(Rectangle((XF,y-1.3),2.4,2.6,facecolor=FAMCOL[fam],edgecolor="none"))
    T(XF+3.6, y, fam, size=PT_LAB)
    for x,v in zip(DOTX,need):
        if v: ax.add_patch(plt.Circle((x,y),0.95,facecolor=INK,edgecolor="none"))
        else: ax.add_patch(plt.Circle((x,y),0.95,facecolor="none",edgecolor=RULE,lw=0.45))
    col=FAMCOL[fam]
    for bnd,(x0,x1),xv in [(kb,(KX0,KX1),KVAL),(ob,(OX0,OX1),OVAL)]:
        # every row carries the full 0-1 track, so a marker's position is readable
        ax.add_patch(Rectangle((x0,y-0.65),x1-x0,1.3,facecolor=TRACK,edgecolor="none",zorder=1))
        ax.plot([x0+(x1-x0)*0.5]*2,[y-0.65,y+0.65],color="white",lw=0.4,zorder=2)
        if bnd is None:
            T((x0+x1)/2, y, "not evaluated", ha="center", col=MUTED, size=5.0)
            T(xv, y, "—", ha="right", col=RULE, size=5.5); continue
        m,lo,hi=bnd
        f=lambda v: x0+(x1-x0)*v
        ax.plot([f(lo),f(hi)],[y,y],color=col,lw=1.6,alpha=0.55,zorder=3,
                solid_capstyle="butt")
        ax.plot([f(m)],[y],marker="o",ms=2.6,color=col,zorder=4,
                markeredgecolor="white",markeredgewidth=0.3)
        T(xv, y, f"{m:.2f}", size=5.5, col=INK, ha="right")
    T(XRT, y, (f"{rt:.0f} s" if rt<600 else f"{rt/60:.0f} min"), ha="center", size=PT_LAB)
    T(XNOTE, y, note, size=5.5, col=INK)
    for x,v in zip(MX, om if om is not None else [None]*3):
        T(x, y, "—" if v is None else f"{v:.2f}", ha="center", size=5.5,
          col=RULE if v is None else INK)
    if i<len(ROWS)-1:
        ax.plot([XF,RIGHT],[y+RH/2]*2, color="#ececec", lw=0.35)

ax.plot([XF,RIGHT],[Y0+len(ROWS)*RH+1.4]*2, color=INK, lw=0.7)
if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_fig4.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig,"Figure4",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
