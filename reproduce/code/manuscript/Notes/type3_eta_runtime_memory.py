#!/usr/bin/env python3
"""Type III η² for log10(runtime) and log10(peak memory) per paradigm."""
import os  # release: added for BENCH_ROOT / vendored figure helpers
# Release: project root (the unpacked data bundle). Set BENCH_ROOT to override.
BENCH_ROOT = os.environ.get("BENCH_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import os, glob, numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

P4_DIR = os.path.join(BENCH_ROOT, "results/P4 Synthetic Datasets V1")
P5_DIR = os.path.join(BENCH_ROOT, "results/P5 Real Life Datasets V2")
marker_tools = ["clustifyr_hyper","clustifyr_jaccard","scCATCH","SCINA","ScInfeR","SCSA","scSorter","scType"]
similarity_tools = ["CHETAH","CIPR","scmap_cell","scmap_cluster","Seurat_Transfer_CCA","Seurat_Transfer_PCA","Seurat_Transfer_RPCA","SingleR"]
classicML_tools = ["CaSTLe","CellTypist","scAnnotate","scAnnotatR","scClassify","scibetR","scID",
    "scPred","scPred_adaboost","scPred_avNNet","scPred_bayesglm","scPred_earth",
    "scPred_glm","scPred_glmboost","scPred_knn","scPred_lda","scPred_mlp","scPred_multinom",
    "scPred_nb","scPred_nnet","scPred_regLogistic","scPred_rf","scPred_svmLinear",
    "scPred_svmPoly","scPred_xgbTree","singleCellNet","scLearn"]
DL_tools = ["ACTINN","Cell_BLAST","CIForm","NeuCA","scBalance","scDeepSort","scHash","scMMT","TOSICA"]
semi_tools = ["CALLR","CAMLU","MARS","SCANVI","scArches","scGAD","scnym","scSemiCluster","scSemiGAN"]
cat_map = {**{t:"Marker-Based" for t in marker_tools},
           **{t:"Similarity-Based" for t in similarity_tools},
           **{t:"Classic ML" for t in classicML_tools},
           **{t:"Deep Learning" for t in DL_tools},
           **{t:"Semi-Supervised" for t in semi_tools}}
exclude = {"ScInfeR","scPred_adaboost","scAnnotatR","CIPR","scmap_cluster","clustifyr_hyper",
           "clustifyr_jaccard","SCSA","scCATCH","scType","scGAD","scDeepSort","CALLR"}

P5_MAP = {
    "S1-Darmanis-Brain-2015_for_use":"Darmanis-Brain-2015_for_use",
    "S2-Marques-Brain_for_use":"Marques-Brain-2016_for_use",
    "S3-Nowakowski-Cortex-2017_for_use":"Nowakowski-Cortex-2017_for_use",
    "S4-Grun-Pancreas-2016_original":"Grun-Pancreas-2016_original",
    "S5-Tabula Muris-FACS-3k_for_use":"Tabula Muris-FACS-3k_for_use",
    "S6-He-Skin-2020_for_use":"He-Skin-2020_for_use",
    "S7-Zhao-Immune-Fine-2020_for_use":"Zhao-Immune-Fine-2020_for_use",
    "S8-Zheng-Zhengsort-5cl-2017_original":"Zheng-ZhengSort-5cl-2017_original",
    "S9-MacParland-Liver-Broad_original":"MacParland-Liver-Broad-2018_original"
}
pr = pd.read_csv(os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_Real.csv"))
def make_p5_prof():
    rows = []
    for sc, key in P5_MAP.items():
        r = pr[pr["Sample"]==key].iloc[0]
        rows.append({"scenario":sc, "log_cells":np.log10(r["Total_Cells"]),
                     "n_types":int(r["Total_CellTypes"]), "shannon":r["Norm_Shannon_Entropy"],
                     "knn_purity":r["KNN_Purity"]})
    return pd.DataFrame(rows)
P5_PROF = make_p5_prof()

sr = pd.read_csv(os.path.join(BENCH_ROOT, "Data Creation/P5 Real Life Sets V2/Profiling_L9.csv"))
sr = sr.rename(columns={"Scenario":"scenario"})
P4_PROF = sr.groupby("scenario").agg(log_cells=("Total_Cells", lambda x: np.log10(x.mean())),
    n_types=("Total_CellTypes", lambda x: int(round(x.mean()))),
    shannon=("Norm_Shannon_Entropy","mean"), knn_purity=("KNN_Purity","mean")).reset_index()

def drop_idx(df):
    if df.shape[1]>0 and df.columns[0] in ("","X","Unnamed: 0"): return df.iloc[:,1:]
    return df

def load_p4():
    rows=[]
    for sc in ["S1","S2","S3","S4","S5","S6","S7","S8","S9"]:
        sp=os.path.join(P4_DIR,sc)
        for r in sorted(os.listdir(sp)):
            rp=os.path.join(sp,r)
            if not os.path.isdir(rp): continue
            for f in glob.glob(os.path.join(rp,"*.csv")):
                try: df=pd.read_csv(f,low_memory=False)
                except: continue
                df=drop_idx(df); df["scenario"]=sc; df["replicate"]=r
                rows.append(df)
    o=pd.concat(rows, ignore_index=True)
    o=o[~o["tool"].isin(exclude)].copy(); o["tool_category"]=o["tool"].map(cat_map)
    return o

def load_p5():
    rows=[]
    for sc in P5_MAP:
        sp=os.path.join(P5_DIR,sc)
        for f in glob.glob(os.path.join(sp,"*.csv")):
            try: df=pd.read_csv(f,low_memory=False)
            except: continue
            df=drop_idx(df); df["scenario"]=sc; df["replicate"]="single"
            rows.append(df)
    o=pd.concat(rows, ignore_index=True)
    o=o[~o["tool"].isin(exclude)].copy(); o["tool_category"]=o["tool"].map(cat_map)
    return o

def prep(d, prof):
    d=d.copy()
    mz=(d.get("kappa_mean",0)==0)&(d.get("accuracy_mean",0)==0)&(d.get("f1_mean",0)==0)
    d.loc[mz,"kappa_mean"]=np.nan
    d=d[d["tool_category"].notna()].copy()
    keep=[c for c in ["kappa_mean","runtime_mean","peak_system_memory_mean_mb"] if c in d.columns]
    g=d.groupby(["scenario","tool","tool_category"])[keep].mean(numeric_only=True).reset_index()
    return g.merge(prof, on="scenario", how="left").dropna(subset=["kappa_mean"])

P4=prep(load_p4(),P4_PROF); P5=prep(load_p5(),P5_PROF)

paradigms=["Marker-Based","Similarity-Based","Classic ML","Deep Learning","Semi-Supervised"]

def eta3(d, response):
    d=d[d[response].notna() & np.isfinite(d[response])].copy()
    if d.shape[0]<6: return None
    try:
        fit=ols(f"{response} ~ log_cells + n_types + shannon + knn_purity", data=d).fit()
        a3=sm.stats.anova_lm(fit, typ=3)
        a1=sm.stats.anova_lm(fit, typ=1)
    except: return None
    ss=a1["sum_sq"].sum()
    return {f: (a1.loc[f,"sum_sq"]/ss, a3.loc[f,"sum_sq"]/ss, a1.loc[f,"PR(>F)"], a3.loc[f,"PR(>F)"]) for f in ["log_cells","n_types","shannon","knn_purity"]}

print("=== η²(log_runtime) Type I / III per paradigm ===")
for arm,dall in [("synth",P4),("real",P5)]:
    print(f"\n-- {arm} --")
    d_rt=dall[dall["runtime_mean"].notna() & (dall["runtime_mean"]>0)].copy()
    d_rt["log_runtime"]=np.log10(d_rt["runtime_mean"])
    for par in paradigms:
        r=eta3(d_rt[d_rt["tool_category"]==par], "log_runtime")
        if r:
            print(f"  {par}:")
            for f in ["log_cells","n_types","shannon","knn_purity"]:
                e1,e3,p1,p3=r[f]
                print(f"    {f:12s}: η²_I={e1:.3f} (p={p1:.4f})  η²_III={e3:.3f} (p={p3:.4f})")

print("\n=== η²(log_mem) Type I / III per paradigm ===")
for arm,dall in [("synth",P4),("real",P5)]:
    print(f"\n-- {arm} --")
    d_m=dall[dall["peak_system_memory_mean_mb"].notna() & (dall["peak_system_memory_mean_mb"]>0)].copy()
    d_m["log_mem"]=np.log10(d_m["peak_system_memory_mean_mb"])
    for par in paradigms:
        r=eta3(d_m[d_m["tool_category"]==par], "log_mem")
        if r:
            print(f"  {par}:")
            for f in ["log_cells","n_types","shannon","knn_purity"]:
                e1,e3,p1,p3=r[f]
                print(f"    {f:12s}: η²_I={e1:.3f} (p={p1:.4f})  η²_III={e3:.3f} (p={p3:.4f})")
