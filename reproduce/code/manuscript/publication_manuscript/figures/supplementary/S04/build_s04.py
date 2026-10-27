"""
Supplementary Figure 4 | Orthogonality of the continuous-proxy framework, synthetic arm.
Nature style, 183 x 72 mm, three panels.

Data : ../../_shared/tidy_kappa.csv  (arm == "synth")
       ../../_shared/l9_design.csv   (the L9(3^4) array, transcribed from
       sc-celltype-benchmark-code/01-data-preparation/synthetic-simulation/L9_Generator.R
       and verified against the data: cell count and cell-type number agree for all nine)

Partitions, stated precisely because they differ between panels:
  (b) purity ~ four categorical factors, Type III, scenario level (n = 9)
  (c) categorical column: kappa ~ four categorical factors, Type III, per (tool, scenario)
      continuous  column: kappa ~ four continuous proxies, Type III, per (tool, scenario)
      Everything in this paper is Type III. Purity takes almost all of the continuous
      column (0.706) because it mediates the other three -- panel (b) is that mediation.
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
C_CON, C_CAT = FS.TOL["indigo"], FS.TOL["sand"]
DIV=FS.score_ramp()   # same ramp as every other heatmap: red = low, blue = high
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
MM=1/25.4; FW,FH=183.0,76.0
def bx(x,ytop,w,h): return [x/FW,(FH-ytop-h)/FH,w/FW,h/FH]

P=["log_cells","n_types","shannon","knn_purity"]
PLAB=["log₁₀ cells","Cell types","Shannon\nentropy","kNN purity"]
k=pd.read_csv(os.path.join(SH,"tidy_kappa.csv")); k["scen"]=k.scenario.str.split("-").str[0]
L9=pd.read_csv(os.path.join(SH,"l9_design.csv")).rename(columns={"n_types":"cat_types"})
s=k[k.arm=="synth"]
sc=s.groupby("scen").agg({**{p:"mean" for p in P},"kappa_imp0":"mean"})
g=s.groupby(["tool","scen"]).agg({**{p:"mean" for p in P},"kappa_imp0":"mean"}).reset_index()
gg=g.merge(L9,on="scen"); sc2=sc.reset_index().merge(L9,on="scen")

def sse(X,y):
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; return r@r
def eta_cat(df,resp,factors):
    y=df[resp].values; cols=[np.ones(len(df))]; spans={}; pos=1
    for f in factors:
        D=pd.get_dummies(df[f],drop_first=True).astype(float).values
        cols.append(D); spans[f]=list(range(pos,pos+D.shape[1])); pos+=D.shape[1]
    X=np.column_stack(cols); sst=((y-y.mean())**2).sum(); full=sse(X,y)
    return {f:(sse(X[:,[0]+[c for c in range(1,X.shape[1]) if c not in spans[f]]],y)-full)/sst
            for f in factors}
FACT=["cell_count","imbalance","cat_types","de_signal"]
FLAB={"cell_count":"Cell count","imbalance":"Imbalance","cat_types":"Cell-type number",
      "de_signal":"DE difficulty"}

fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
def tag(s_,x,y): fig.text(x/FW,(FH-y)/FH,s_,fontsize=PT_TAG,fontweight="bold",
                          va="top",ha="left",color=INK,family=FONT)

# ---- a : correlation heatmap + VIF ---------------------------------------
axa=fig.add_axes(bx(20,12,34,34))
C=sc[P].corr().values
im=axa.pcolormesh(np.arange(5)-0.5,np.arange(5)-0.5,np.ma.masked_invalid(C),
                  cmap=DIV,vmin=-1,vmax=1,shading="flat")
im.set_edgecolor("none")
axa.set_xlim(-0.5,3.5); axa.set_ylim(3.5,-0.5)
axa.set_xticks(range(4)); axa.set_xticklabels(PLAB,fontsize=PT_ROW,rotation=45,ha="right")
axa.set_yticks(range(4)); axa.set_yticklabels(PLAB,fontsize=PT_ROW)
axa.tick_params(axis="both",which="both",length=0,top=False,right=False)
for sp in axa.spines.values(): sp.set_visible(False)
for i in range(4):
    for j in range(4):
        v=C[i,j]
        axa.text(j,i,f"{v:.2f}".replace("0.",".").replace("1.00","1"),ha="center",va="center",
                 fontsize=PT_ROW-0.3,color="white" if abs(v)>0.6 else INK)
cax=fig.add_axes(bx(56,14,1.7,20))
cb=fig.colorbar(im,cax=cax,ticks=[-1,0,1]); cb.outline.set_linewidth(LW)
cb.ax.set_title("r",fontsize=PT_TICK,pad=3,color=INK)
cb.ax.tick_params(labelsize=PT_TICK,length=1.6,width=LW,pad=1.5)
VIF={}
for p in P:
    X=np.column_stack([np.ones(len(sc))]+[sc[q].values for q in P if q!=p]); y=sc[p].values
    VIF[p]=1/(1-(1-sse(X,y)/((y-y.mean())**2).sum()))
fig.text(20/FW,(FH-60)/FH,
  f"max |r| = {np.abs(C[~np.eye(4,dtype=bool)]).max():.2f}   ·   VIF {min(VIF.values()):.2f}–{max(VIF.values()):.2f}",
  fontsize=PT_ROW-0.3,va="top",ha="left",color=MUTED)
tag("a",4,6)

# ---- b : purity mediation -------------------------------------------------
axb=fig.add_axes(bx(80,12,42,34))
bvals=eta_cat(sc2,"knn_purity",FACT)
order=sorted(FACT,key=lambda f:bvals[f])
for i,f in enumerate(order):
    axb.barh(i,bvals[f],height=0.58,color=C_CON,alpha=0.85,edgecolor="none")
    axb.text(bvals[f]+0.018,i,f"{bvals[f]:.3f}",va="center",ha="left",fontsize=PT_ROW,color=INK)
axb.set_yticks(range(4)); axb.set_yticklabels([FLAB[f] for f in order],fontsize=PT_ROW)
axb.set_xlim(0,0.92); axb.set_xticks([0,0.25,0.5,0.75])
axb.set_ylim(-0.6,3.6)
axb.set_xlabel("η² of kNN purity",fontsize=PT_LAB,labelpad=2)
axb.tick_params(axis="y",length=0,pad=2); axb.tick_params(axis="x",top=False)
tag("b",66,6)

# ---- c : categorical versus continuous ------------------------------------
axc=fig.add_axes(bx(140,12,38,34))
cat=eta_cat(gg,"kappa_imp0",FACT)
y3=gg.kappa_imp0.values; X3=np.column_stack([np.ones(len(gg))]+[gg[q].values for q in P])
sst3=((y3-y3.mean())**2).sum(); full3=sse(X3,y3)
con3=[(sse(X3[:,[0]+[j+1 for j in range(4) if j!=i]],y3)-full3)/sst3 for i in range(4)]
pairs=[("Cell count",con3[0],cat["cell_count"]),
       ("Cell-type number",con3[1],cat["cat_types"]),
       ("Imbalance",con3[2],cat["imbalance"]),
       ("DE difficulty / purity",con3[3],cat["de_signal"])]
pairs=sorted(pairs,key=lambda t:max(t[1],t[2]))
for i,(lab,c3,kv) in enumerate(pairs):
    axc.barh(i+0.17,c3,height=0.32,color=C_CON,alpha=0.9,edgecolor="none")
    axc.barh(i-0.17,kv,height=0.32,color=C_CAT,alpha=0.9,edgecolor="none")
axc.set_yticks(range(4)); axc.set_yticklabels([p_[0] for p_ in pairs],fontsize=PT_ROW)
axc.set_xlim(0,0.80); axc.set_xticks([0,0.25,0.5,0.75])
axc.set_ylim(-0.6,3.6)
axc.set_xlabel("η² of Cohen's κ",fontsize=PT_LAB,labelpad=2)
axc.tick_params(axis="y",length=0,pad=2); axc.tick_params(axis="x",top=False)
for i,(lab,col) in enumerate([("continuous proxies",C_CON),
                              ("L9 design factors",C_CAT)]):
    yy=FH-56-i*4.4
    fig.lines.append(plt.Line2D([140/FW,143.4/FW],[(yy+1.0)/FH]*2,transform=fig.transFigure,
                     color=col,lw=1.4,solid_capstyle="butt"))
    fig.text(144.6/FW,yy/FH,lab,fontsize=PT_ROW-0.3,va="baseline",ha="left",color=INK)
tag("c",126,6)

if __name__=="__main__":
    import visual_qa
    visual_qa.render_preview(fig,"_preview_s04.png",dpi=600)
    print(visual_qa.print_report(visual_qa.audit_layout(fig)))
    from export_figure import export_figure
    export_figure(fig,"SuppFig_S04",formats=["pdf","svg","png"],dpi=600,
                  size_inches=(FW*MM,FH*MM),grayscale_preview=True,tight=False)
    print("VIF:",{p:round(v,3) for p,v in VIF.items()})
    print("(b):",{f:round(v,3) for f,v in bvals.items()})
    print("(c) continuous Type III:",[round(v,3) for v in con3])
    print("(c) categorical Type III:",{f:round(v,3) for f,v in cat.items()})
