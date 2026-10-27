"""
Supplementary Figure 2 | Paired synthetic and real UMAP embeddings, nine scenarios.
Nature style, 116 mm wide, two pages: S1-S5 and S6-S9.

Sources: results/P4 Synthetic Datasets V1/Plots/S{n}_rep1_seurat_subset_UMAP.png
         results/P5 Real Life Datasets V2/plots/{dataset}_subset_UMAP.png
Each source is a full ggplot with its own title, axes and per-dataset legend. The point
cloud is located by saturation inside the plot band, left of the legend text, and cropped
out; titles, axes and the 5-48 entry legends are dropped. Each crop is fitted into a square
tile on white, preserving aspect. A source whose legend collapsed the plot panel yields no
crop and leaves a dashed empty tile.
"""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
import sys, os, numpy as np
from PIL import Image
SKILL = os.environ.get("SCIPILOT_SCRIPTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_vendor", "scipilot")
sys.path.insert(0, SKILL)
from setup_style import setup_style
import matplotlib as mpl
info=setup_style(journal="nature", lang="en", constrained_layout=False)
import matplotlib.pyplot as plt

FONT="Liberation Sans"; PT_LAB, PT_TAG, PT_ROW, PT_SUB = 7, 8, 6.4, 5.0
INK, MUTED = "#000000", "#4d4d4d"
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":[FONT,"DejaVu Sans"],
    "figure.dpi":600,"savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42,
    "svg.fonttype":"none","figure.autolayout":False})
MM=1/25.4
P4=os.path.join(BENCH_ROOT, "results/P4 Synthetic Datasets V1/Plots")
P5=os.path.join(BENCH_ROOT, "results/P5 Real Life Datasets V2/Plots")
REAL={"S1":("Darmanis-Brain-2015_for_use_subset_UMAP.png","Darmanis brain 2015"),
      "S2":("Marques-Brain-2016_for_use_subset_UMAP.png","Marques brain 2016"),
      "S3":("Nowakowski-Cortex-2017_for_use_subset_UMAP.png","Nowakowski cortex 2017"),
      "S4":("Grun-Pancreas-2016_original_subset_UMAP.png","Grün pancreas 2016"),
      "S5":("Tabula Muris-FACS-3k_for_use_subset_UMAP.png","Tabula Muris FACS 2018"),
      "S6":("He-Skin-2020_for_use_subset_UMAP.png","He skin 2020"),
      "S7":("Zhao-Immune-Fine-2020_for_use_subset_UMAP.png","Zhao immune 2020"),
      "S8":("Zheng-ZhengSort-5cl-2017_original_subset_UMAP.png","Zheng PBMC 2017"),
      "S9":("MacParland-Liver-Broad-2018_original_seurat_subset_UMAP.png","MacParland liver 2018")}

TILEDIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tiles")

MISSING=[]
def page(scens, fname):
    TILE=42.0; PITCH=45.0; LEFT=27.0; GUT=6.0; HEAD=11.0
    FW=LEFT+TILE+GUT+TILE+6.0
    FH=HEAD+len(scens)*PITCH+2.0
    fig=plt.figure(figsize=(FW*MM,FH*MM)); fig.patch.set_facecolor("white")
    for j,ttl in enumerate(["Synthetic","Real"]):
        x=LEFT+j*(TILE+GUT)+TILE/2
        fig.text(x/FW,(FH-HEAD+3.0)/FH,ttl,fontsize=PT_LAB,fontweight="bold",
                 ha="center",va="bottom",color=INK)
    for i,sc in enumerate(scens):
        ytop=HEAD+i*PITCH
        fig.text(4/FW,(FH-ytop-TILE/2+1.6)/FH,sc,fontsize=PT_LAB,fontweight="bold",
                 ha="left",va="baseline",color=INK)
        fig.text(4/FW,(FH-ytop-TILE/2-2.6)/FH,REAL[sc][1],fontsize=PT_SUB,
                 ha="left",va="baseline",color=MUTED)
        for j in (0,1):
            x=LEFT+j*(TILE+GUT); y=ytop
            ax=fig.add_axes([x/FW,(FH-y-TILE)/FH,TILE/FW,TILE/FH])
            tp=os.path.join(TILEDIR,f"{sc}_{'synth' if j==0 else 'real'}.png")
            img=Image.open(tp) if os.path.exists(tp) else None
            if img is None:
                MISSING.append((sc,"synthetic" if j==0 else "real"))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set(color="#c8c8c8",linewidth=0.4,linestyle=(0,(2,2)))
                continue
            ax.imshow(img,interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set(color="#c8c8c8",linewidth=0.4)
    return fig,FW,FH

if __name__=="__main__":
    from export_figure import export_figure
    for scens,name in [(["S1","S2","S3","S4","S5"],"SuppFig_S02_p1"),
                       (["S6","S7","S8","S9"],"SuppFig_S02_p2")]:
        fig,FW,FH=page(scens,name)
        export_figure(fig,name,formats=["pdf","png"],dpi=600,
                      size_inches=(FW*MM,FH*MM),grayscale_preview=False,tight=False)
        print(name,"->",round(FW,1),"x",round(FH,1),"mm")
    print("unusable source panels:", MISSING or "none")
