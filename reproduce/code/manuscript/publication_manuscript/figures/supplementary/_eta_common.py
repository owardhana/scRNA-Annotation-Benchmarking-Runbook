"""Shared Type III eta-squared machinery for Supplementary Figs S2, S10 and S11."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import f as fdist
HERE=os.path.dirname(os.path.abspath(__file__))
SH=os.path.join(HERE,"..","_shared")
sys.path.insert(0, SH)
from catmap import FAMILY_ORDER, FAMILY_COLORS, MARKER, SIMIL, CLASSIC, DL, SEMI

P=["knn_purity","log_cells","n_types","shannon"]
PLAB={"knn_purity":"kNN purity","log_cells":"log₁₀ cell count",
      "n_types":"Cell-type number","shannon":"Shannon entropy"}
SHORT={"Marker-Based":"Marker-based","Similarity-Based":"Similarity-based",
       "Classic ML":"Classical ML","Deep Learning":"Deep learning",
       "Semi-Supervised":"Semi-supervised"}
FAM_MAP={}
for _f,_l in [("Marker-Based",MARKER),("Similarity-Based",SIMIL),("Classic ML",CLASSIC),
              ("Deep Learning",DL),("Semi-Supervised",SEMI)]:
    for _t in _l: FAM_MAP[_t]=_f

def load():
    d=pd.read_csv(os.path.join(SH,"tidy_kappa.csv"))
    d["scen"]=d.scenario.str.split("-").str[0]; d["fam"]=d.tool.map(FAM_MAP)
    return d

def _sse(X,y):
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; return r@r

def type3(g, resp, with_tool=False):
    """Type III eta-squared + p for each continuous predictor (and tool identity)."""
    g=g.dropna(subset=[resp])
    y=g[resp].values; cols=[np.ones(len(g))]+[g[p].values for p in P]; spans={}
    for i,p in enumerate(P): spans[p]=[i+1]
    pos=1+len(P)
    if with_tool:
        D=pd.get_dummies(g.tool,drop_first=True).astype(float).values
        cols.append(D); spans["tool"]=list(range(pos,pos+D.shape[1])); pos+=D.shape[1]
    X=np.column_stack(cols); n,k=X.shape
    sst=((y-y.mean())**2).sum(); full=_sse(X,y); dfres=n-k
    out={}
    for term,idx in spans.items():
        keep=[c for c in range(X.shape[1]) if c not in idx]
        ss=_sse(X[:,keep],y)-full; dfj=len(idx)
        F=(ss/dfj)/(full/dfres) if dfres>0 else np.nan
        out[term]=(ss/sst, 1-fdist.cdf(F,dfj,dfres))
    out["_residual"]=(full/sst, np.nan)
    return out

def stars(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"

def unit(d, arm, resp="kappa_imp0", fam=None):
    s=d[d.arm==arm]
    if fam is not None: s=s[s.fam==fam]
    agg={**{p:"mean" for p in P}, resp:"mean"}
    return s.groupby(["tool","scen"]).agg(agg).reset_index()
