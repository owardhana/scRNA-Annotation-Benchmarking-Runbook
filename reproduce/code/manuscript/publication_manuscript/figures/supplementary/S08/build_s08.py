"""
Supplementary Figure 8 | Cross-platform annotation transfer.
Nature style, 183 x 140 mm, four panels.

Data : ../../_shared/tidy_crossplatform.csv, written by ../../_shared/build_tidy_crossplatform.py
       from the 18 per-trial CSVs in "results/P9 Cross Dataset/data" (scAnnotatR and CALLR
       excluded; 55 tool configurations, ScInfeR included);
       ../../_shared/tidy_gene_overlap.csv  (per-trial Jaccard, from
       "Data Creation/P9 Cross Datasets/Phase3_GeneOverlap.csv").
Failure convention matches the rest of the paper: a tool x trial cell with no valid
prediction is scored kappa = 0 on the full grid, not dropped. ScInfeR has no PBMCbench
rows, so its three PBMC cells are 0.
2026-10-17: ScInfeR was missing from the tidy table, so panel (a) showed n = 216/108/162
(54 tools) against the text's 220/110/165. Rebuilt with ScInfeR; previous render kept in
_pre_scinfer/.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
import sys, os, itertools, numpy as np, pandas as pd
SKILL=os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
HERE=os.path.dirname(os.path.abspath(__file__))
SH=os.path.join(HERE,"..","..","_shared"); sys.path.insert(0, SH)
from setup_style import setup_style
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt
from catmap import FAMILY_ORDER, FAMILY_COLORS
import figstyle as FS

FONT=FS.FONT; PT_TICK, PT_LAB, PT_TAG, PT_ROW = FS.PT_TICK, FS.PT_LAB, FS.PT_TAG, 5.6
INK, MUTED, LW = FS.INK, FS.MUTED, FS.LW
BLOCKS=["CellBench","Pancreas","PBMCbench"]
BCOL=FS.BLOCK_COLORS      # three distinct hues, not a gradient
PLAT={"CB-1":"10x→CEL2","CB-2":"10x→Drop","CB-3":"CEL2→10x","CB-4":"Drop→10x",
      "PX-1":"inDrop→SS2","PX-2":"SS2→inDrop","PBMC-A":"10x→Drop",
      "PBMC-B":"10x→inDrops","PBMC-C":"10x→SeqWell"}
TRIALS={"CellBench":["CB-1","CB-2","CB-3","CB-4"],"Pancreas":["PX-1","PX-2"],
        "PBMCbench":["PBMC-A","PBMC-B","PBMC-C"]}
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

c=pd.read_csv(os.path.join(SH,"tidy_crossplatform.csv"))
tools=sorted(c.tool.unique()); trials=sorted(c.trial.unique())
assert len(tools)==55 and len(trials)==9, f"{len(tools)} tools x {len(trials)} trials, expected 55 x 9"
grid=pd.DataFrame(list(itertools.product(tools,trials)),columns=["tool","trial"])
d=grid.merge(c[["tool","trial","kappa_mean","fam"]],how="left",on=["tool","trial"])
d["fam"]=d.groupby("tool").fam.transform(lambda s: s.ffill().bfill())
d["k"]=d.kappa_mean.fillna(0.0)
BLOCK={t:b for b,ts in TRIALS.items() for t in ts}
d["block"]=d.trial.map(BLOCK)

fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s,x,y): fig.text(x/FW,(FH-y)/FH,s,fontsize=PT_TAG,fontweight="bold",
                         va="top",ha="left",color=INK,family=FONT)
rng=np.random.default_rng(7)

def strip(ax, groups, values, colours, jit=0.17, ptsize=2.4, w=0.50):
    for i,(g,v) in enumerate(zip(groups,values)):
        col=colours[i]
        bp=ax.boxplot([v],positions=[i],widths=w,whis=(5,95),showfliers=False,
                      patch_artist=True,manage_ticks=False)
        bp["boxes"][0].set(facecolor=col,alpha=0.30,edgecolor=col,linewidth=0.7)
        for kk in ("whiskers","caps"):
            for e in bp[kk]: e.set(color=col,linewidth=0.7)
        bp["medians"][0].set(color=INK,linewidth=1.0)
        ax.scatter(i+rng.uniform(-jit,jit,len(v)),v,s=ptsize,facecolor=col,
                   edgecolor="none",alpha=0.55,zorder=3)

# ---- a : kappa by block ----------------------------------------------------
axa=fig.add_axes(bx(16,10,60,46))
vals=[d.loc[d.block==b,"k"].values for b in BLOCKS]
strip(axa,BLOCKS,vals,[BCOL[b] for b in BLOCKS])
axa.set_xticks(range(3)); axa.set_xticklabels(BLOCKS,fontsize=PT_ROW)
axa.set_ylim(-0.04,1.06); axa.set_yticks([0,0.25,0.5,0.75,1.0])
axa.set_yticklabels(["0","0.25","0.50","0.75","1.00"])
axa.set_ylabel("Cohen's κ",fontsize=PT_LAB,labelpad=2)
axa.tick_params(axis="both",which="both",top=False,right=False)
for i,b in enumerate(BLOCKS):
    axa.text(i,1.045,f"n = {len(vals[i])}",ha="center",va="bottom",
             fontsize=PT_ROW-0.4,color=MUTED)
tag("a",4,5)

# ---- b : kappa by trial ----------------------------------------------------
axb=fig.add_axes(bx(104,10,64,46))
order=[t for b in BLOCKS for t in TRIALS[b]]
vals=[d.loc[d.trial==t,"k"].values for t in order]
cols=[BCOL[BLOCK[t]] for t in order]
strip(axb,order,vals,cols,jit=0.16,ptsize=1.9,w=0.56)
axb.set_xticks(range(len(order))); axb.set_xticklabels([f"{t}\n{PLAT[t]}" for t in order],fontsize=PT_ROW-0.6,
                    rotation=45,ha="right",linespacing=1.15)
axb.set_ylim(-0.04,1.06); axb.set_yticks([0,0.25,0.5,0.75,1.0])
axb.set_yticklabels(["0","0.25","0.50","0.75","1.00"])
axb.set_ylabel("Cohen's κ",fontsize=PT_LAB,labelpad=2)
axb.tick_params(axis="both",which="both",top=False,right=False)
for b in BLOCKS:
    xs=[order.index(t) for t in TRIALS[b]]
    axb.plot([xs[0]-0.34,xs[-1]+0.34],[1.055]*2,color=BCOL[b],lw=1.3,
             clip_on=False,solid_capstyle="butt")
    axb.text((xs[0]+xs[-1])/2,1.085,b,ha="center",va="bottom",fontsize=PT_ROW-0.4,color=INK)
tag("b",92,5)

# ---- c : easy versus hard biology, per tool --------------------------------
axc=fig.add_axes(bx(16,78,46,46))
cb=d[d.block=="CellBench"].groupby(["tool","fam"]).k.mean()
px=d[d.block=="Pancreas"].groupby(["tool","fam"]).k.mean()
m=pd.concat([cb.rename("cb"),px.rename("px")],axis=1).reset_index()
axc.plot([0,1.02],[0,1.02],color="#bfbfbf",lw=0.6,ls=(0,(3,2)),zorder=1)
for f in FAMILY_ORDER:
    s=m[m.fam==f]
    axc.scatter(s.cb,s.px,s=6,facecolor=FAMILY_COLORS[f],edgecolor="white",
                linewidth=0.25,zorder=3)
axc.set_xlim(-0.04,1.06); axc.set_ylim(-0.04,1.06)
axc.set_xticks([0,0.5,1.0]); axc.set_yticks([0,0.5,1.0])
axc.set_xticklabels(["0","0.50","1.00"]); axc.set_yticklabels(["0","0.50","1.00"])
axc.set_xlabel("Mean κ, CellBench",fontsize=PT_LAB,labelpad=2)
axc.set_ylabel("Mean κ, pancreas",fontsize=PT_LAB,labelpad=2)
axc.tick_params(axis="both",which="both",top=False,right=False)
tag("d",92,73)

axd=fig.add_axes(bx(104,78,64,46))
jj=pd.read_csv(os.path.join(SH,"tidy_gene_overlap.csv")).set_index("trial").reindex(order)
for i,t in enumerate(order):
    axd.scatter([i],[jj.loc[t,"jaccard"]],s=16,facecolor=BCOL[BLOCK[t]],
                edgecolor="white",linewidth=0.4,zorder=3)
    axd.plot([i,i],[0.40,jj.loc[t,"jaccard"]],color=BCOL[BLOCK[t]],lw=0.7,zorder=2)
axd.set_xticks(range(len(order))); axd.set_xticklabels(order,fontsize=PT_ROW,rotation=45,ha="right")
axd.set_xlim(-0.6,len(order)-0.4)
axd.set_ylim(0.40,0.86); axd.set_yticks([0.4,0.5,0.6,0.7,0.8])
axd.set_yticklabels(["0.40","0.50","0.60","0.70","0.80"])
axd.set_ylabel("Detected-gene overlap (Jaccard)",fontsize=PT_LAB,labelpad=2)
axd.tick_params(axis="both",which="both",top=False,right=False)
for b in BLOCKS:
    xs=[order.index(t) for t in TRIALS[b]]
    axd.plot([xs[0]-0.34,xs[-1]+0.34],[0.852]*2,color=BCOL[b],lw=1.3,
             clip_on=False,solid_capstyle="butt")

tag("c",4,73)

# family key, inside panel (c)'s empty lower-right corner
FS.family_key(axc, x=0.470, y=0.415, pitch=0.072, swatch=0.070, width=0.575,
              fontsize=PT_ROW-0.2)

if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_s08.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    # tight crop with a 0.05 in pad reproduces the framing of the V33-embedded render
    # (PDF 166.5 x 130.4 mm, PNG 3934 x 3078 px); grayscale is converted from that PNG
    # because export_figure's own grayscale step re-saves the PNG with a 0.1 in pad.
    export_figure(fig,"SuppFig_S08",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=False,tight=True,pad_inches=0.05)
    from PIL import Image
    Image.open("SuppFig_S08.png").convert("L").save("SuppFig_S08_grayscale.png")
    for b in BLOCKS:
        v=d.loc[d.block==b,"k"]
        print(f"{b:11s} n={len(v):3d} median={v.median():.3f} IQR {v.quantile(.25):.2f}-{v.quantile(.75):.2f}")
