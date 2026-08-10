# Tool roster

The 63 tool configurations evaluated in the study, and which of them ship as
wrappers here.

This repository is **reference code**: it carries one worked wrapper per
methodological paradigm plus an annotated template, not all 63. This document
is the manifest of what was actually run, so the gap between "evaluated" and
"shipped" is visible rather than implied.

The authoritative version of this table, with citations, mechanisms and
per-phase oracle inputs, is Supplementary Table S4 of the manuscript.

---

## Counts

| Paradigm | Configurations | Wrapper shipped |
|---|--:|---|
| Marker | 8 | `run_scSorter.R`, `run_scType.R` |
| Correlation / reference | 8 | `run_SingleR.R` |
| Classical ML | 24 | `run_scPred.R` (parameterised over 17 configs) |
| Deep learning | 9 | `run_scLearn.R` (R), `run_ACTINN.py` (Python) |
| Semi-supervised | 6 | `run_scArches.py` |
| Foundation model | 5 | — (ran in Colab; out of scope) |
| LLM | 3 | `run_mLLMCelltype.R` (parameterised over 8 endpoints) |
| **Total** | **63** | **8 wrappers + 2 templates** |

"Configurations", not "tools": scPred's 17 classifier settings, scmap's two
prediction modes and clustifyr's two similarity modes each count separately,
because each was run and reported separately.

---

## Marker-based (8)

Consume a marker signature table; ignore `seurat_train`. In Phases 1–3 the
markers are derived from the data (`M`); in Phase 4 they come from CellMarker
2.0 (`DB`). `C` marks tools that also require a cluster assignment.

| Tool | P1–2 | P4 | Wrapper | Notes |
|---|---|---|---|---|
| SCINA | M | DB | — | Cell-level. `rm_overlap=FALSE` in P4: the default strips genes shared between overlapping CellMarker signatures |
| scSorter | M | DB | **shipped** | Cell-level. The simplest complete example of the contract |
| scType | M+C | DB+C | **shipped** | Cluster-dependent. Ships with both marker sources in one wrapper |
| scCATCH | M+C | DB+C | — | Cluster-dependent |
| SCSA | M+C | DB+C | — | Cluster-dependent; Python helper. Also queries its own bundled database alongside CellMarker 2.0 in P4 |
| clustifyr (hypergeometric) | M+C | DB+C | — | Cluster-dependent |
| clustifyr (Jaccard) | M+C | DB+C | — | Cluster-dependent |
| ScInfeR | M+C | DB | — | Cluster-dependent in P1–3, cell-level in P4; needs a UMAP of the test data |

Cluster-dependent tools are excluded from the Phase 1–3 comparison and reported
separately, since supplying ground-truth clusters gives them information the
cell-level tools do not receive.

---

## Correlation / reference-based (8)

Need a labelled reference; mostly ignore `markers`.

| Tool | Wrapper | Notes |
|---|---|---|
| SingleR | **shipped** | Spearman correlation with marker-based fine-tuning; confidence from `delta.max` |
| Seurat label transfer (PCA) | — | Anchor-based; confidence from `prediction.score.max` |
| Seurat label transfer (RPCA) | — | As above, RPCA anchors |
| Seurat label transfer (CCA) | — | As above, CCA anchors |
| scmap (cell) | — | Cell-level nearest neighbour |
| scmap (cluster) | — | Cluster-dependent; reported in Supp. Fig. S5 only |
| scibetR | — | Multinomial with maximum-entropy gene selection |
| CIPR | — | Cluster-dependent; recomputes markers internally on the test set |

---

## Classical ML (24)

Train a classifier on `seurat_train`; ignore `markers`.

**scPred — 17 reported configurations.** One implementation, one caret method
string apart. `run_scPred.R` takes `model =` and `tool_registry.R` generates
the entries from `SCPRED_MODELS`.

| Reported (17) |
|---|
| `svmRadial` (package default, registered as `scPred`), `svmLinear`, `svmPoly`, `rf`, `knn`, `lda`, `nb`, `glm`, `glmboost`, `bayesglm`, `earth`, `mlp`, `nnet`, `avNNet`, `regLogistic`, `multinom`, `xgbTree` |

`SCPRED_MODELS` holds 20 entries: the 17 above plus `adaboost`, `glmnet` and
`mxnetAdam`, which were run but are **not** in the reported rankings — see
[Run but not reported](#run-but-not-reported).

**The other 7:**

| Tool | Wrapper | Notes |
|---|---|---|
| CHETAH | — | Hierarchical classification tree |
| scClassify | — | Weighted kNN ensemble over cell-type trees |
| singleCellNet | — | Random forest on top-scoring-pair features |
| CellTypist | — | Python; R shells out via temp CSV + stdout protocol |
| scAnnotate | — | Python; same shelling pattern |
| CaSTLe | — | XGBoost on discretised expression |
| scID | — | Fisher LDA with cluster-specific signatures |

---

## Deep learning (9)

| Tool | Language | Wrapper | Notes |
|---|---|---|---|
| scLearn | R | **shipped** | DCA with M3Drop feature selection; wants raw counts |
| NeuCA | R | — | Hierarchical feedforward net; vendors its deprecated upstream helpers |
| ACTINN | Python | **shipped** | Multilayer perceptron; the conda-subprocess isolation exemplar |
| Cell BLAST | Python | — | Variational autoencoder with adversarial regularisation |
| CIForm | Python | — | Transformer classifier |
| scBalance | Python | — | Sparse net with adaptive sampling for rare types |
| scHash | Python | — | Deep hashing for cell retrieval |
| scMMT | Python | — | Multi-modal transformer, RNA-only mode |
| TOSICA | Python | — | Multi-head self-attention with pathway interpretability |

---

## Semi-supervised (6)

Structurally distinct: these consume the labelled training partition **and** the
unlabelled test data simultaneously during fitting. That is a property of the
paradigm, not an oracle input.

| Tool | Language | Wrapper |
|---|---|---|
| scArches | Python | **shipped** |
| CALLR | R | — |
| MARS | Python | — |
| scANVI | Python | — |
| scSemiCluster | Python | — |
| scSemiGAN | Python | — |

---

## Foundation models (5) — not in this repository

Phase 5 only. Each was fine-tuned with LoRA on the same stratified 80% training
partition and evaluated on the identical held-out test partitions as Phase 2.

| Model | Backbone | Hardware |
|---|---|---|
| scBERT | Encoder-only transformer, Performer attention (~1M cells) | NVIDIA L4 |
| scGPT | Generative decoder transformer, FlashAttention (~33M cells) | NVIDIA L4 |
| Geneformer V2 | Rank-tokenisation transformer (~104M transcriptomes) | NVIDIA L4 |
| scFoundation | Linear-complexity attention (~100M cells) | NVIDIA L4 |
| C2S (Cell2Sentence) | Autoregressive LM, Pythia-410M backbone | NVIDIA RTX PRO 6000 |

**Why they are absent.** These ran in Google Colab against provider-specific
notebooks with their own environment setup and checkpoint handling, rather than
through either harness in this repository. Including partial notebooks would
misrepresent how they were run. Their absence is a scope decision, not an
omission.

---

## LLM-based (3)

Phase 4 only. Answer per cluster, in free text, from an API. Predictions pass
through `scoring/normalisation_pipeline.R` before scoring.

| Tool | Wrapper | Notes |
|---|---|---|
| mLLMCelltype | **shipped** | Multi-model consensus with uncertainty quantification. The wrapper takes `models =` and covers both single-model and consensus operation |
| GPTCelltype | — | Single-model prompting with per-cluster top DE markers |
| CASSIA | — | Multi-agent pipeline with iterative refinement |

`MLLM_MODELS` in `tool_registry.R` holds the 7 single-model endpoints run in the
study (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro, DeepSeek v3.1 Terminus, Grok 4
Fast, Llama 4 Maverick, Qwen 3 Max) plus a consensus entry. API keys are read
from the environment at call time and never stored here.

---

## Run but not reported

Executed during development and excluded from the reported rankings, for
degenerate performance (kappa ≈ 0), instability, or runtime. They are listed
because their wrappers existed in the working repository and their absence from
the results is a decision, not an oversight.

| Tool / configuration | Reason |
|---|---|
| scPred `adaboost` | kappa ≈ 0 across scenarios; also very slow |
| scPred `glmnet` | unstable |
| scPred `mxnetAdam` | unstable (mxnet backend) |
| Garnett | ground-truth leakage in its fallback path |
| CellAssign | TensorFlow memory not observable by `peakRAM`; unstable |
| scAnnotatR | superseded by other classical ML entries |
| scAnno | degenerate performance |
| CAMLU | designed for novel-type detection rather than closed-set annotation |
| scDeepSort | unstable |
| scGAD | degenerate performance |
| scDeepinsight, scTransSort, mtANN, ItClust, scnym | did not run to completion in a usable configuration |

---

## Adding a wrapper

The exemplars are meant to be extended from. Copy
`R/tools/TEMPLATE_run_tool.R` (or the `.py` twin), follow
[tool-wrapper-contract.md](tool-wrapper-contract.md), and register the result in
`R/tool_registry.R`. `run_benchmark.R` sources every `run_*.R` in `tools/`
automatically, but the registry — not the directory listing — is what makes a
tool dispatchable.
