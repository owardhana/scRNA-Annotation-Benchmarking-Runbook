MARKER=["clustifyr_hyper","clustifyr_jaccard","scCATCH","SCINA","ScInfeR","SCSA","scSorter","scType"]
SIMIL=["CHETAH","CIPR","scmap_cell","scmap_cluster","Seurat_Transfer_CCA","Seurat_Transfer_PCA","Seurat_Transfer_RPCA","SingleR"]
CLASSIC=["CaSTLe","CellTypist","scAnnotate","scAnnotatR","scClassify","scibetR","scID","scLearn","scPred","scPred_adaboost","scPred_avNNet","scPred_bayesglm","scPred_earth","scPred_glm","scPred_glmboost","scPred_knn","scPred_lda","scPred_mlp","scPred_multinom","scPred_nb","scPred_nnet","scPred_regLogistic","scPred_rf","scPred_svmLinear","scPred_svmPoly","scPred_xgbTree","singleCellNet"]
DL=["ACTINN","Cell_BLAST","CIForm","NeuCA","scBalance","scDeepSort","scHash","scMMT","TOSICA"]
SEMI=["CALLR","CAMLU","MARS","SCANVI","scArches","scGAD","scnym","scSemiCluster","scSemiGAN"]
CATMAP={}
for lst,name in [(MARKER,"Marker-Based"),(SIMIL,"Similarity-Based"),(CLASSIC,"Classic ML"),(DL,"Deep Learning"),(SEMI,"Semi-Supervised")]:
    for t in lst: CATMAP[t]=name
# Palette and type live in figstyle.py — the single source of truth for every figure.
from figstyle import (FAMILY_ORDER, ALL_FAMILY_ORDER, FAMILY_COLORS, SHORT,
                      FONT, INK, MUTED, RULE, LW, PT_TAG, PT_LAB, PT_TICK, PT_ROW,
                      FAIL_GREY, score_ramp, rc, family_key,
                      VARIANCE_COLORS, BLOCK_COLORS, LEVEL_SHADES, LEVEL_INK,
                      C_DB, C_LLM, C_FM, TOL)
