# Tool roster

The 63 tool configurations evaluated in the study, and which of them ship as
wrappers here.

The harness carries one worked wrapper per methodological paradigm plus an
annotated template, not all 63; every wrapper exactly as the study ran it is in
`as-run/benchmarking/`. This document is the manifest of what was actually run.

Phases 1–2 compare **47** of the 63: the cell-level configurations (2 marker,
6 similarity-based, 25 classical ML, 8 deep learning, 6 semi-supervised). The
six cluster-dependent marker tools and two cluster-dependent similarity tools are
reported separately, and the five foundation models and three LLM tools enter in
Phases 4–5. The reported-panel filter is the `exclude_tools` argument to
`load_benchmark_results()` in each `reproduce/code/results/*/Plots_Analysis.R`.

The authoritative version of this table, with citations, mechanisms and
per-phase oracle inputs, is Supplementary Table 3 of the manuscript.

---

## Counts

| Paradigm | Configurations | Wrapper shipped |
|---|--:|---|
| Marker | 8 | `run_scSorter.R`, `run_scType.R` |
| Similarity-based | 8 | `run_SingleR.R` |
| Classical ML | 25 | `run_scPred.R` (parameterised over 17 configs), `run_scLearn.R` |
| Deep learning | 8 | `run_ACTINN.py` (Python) |
| Semi-supervised | 6 | `run_scArches.py` |
| Foundation model | 5 | — (ran in Colab; notebooks in `as-run/foundation-models/`) |
| LLM | 3 | `run_mLLMCelltype.R` (the reported consensus arm) |
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

## Similarity-based (8)

Need a labelled reference; mostly ignore `markers`.

| Tool | Wrapper | Notes |
|---|---|---|
| SingleR | **shipped** | Spearman correlation with marker-based fine-tuning; confidence from `delta.max` |
| Seurat label transfer (PCA) | — | Anchor-based; confidence from `prediction.score.max` |
| Seurat label transfer (RPCA) | — | As above, RPCA anchors |
| Seurat label transfer (CCA) | — | As above, CCA anchors |
| scmap (cell) | — | Cell-level nearest neighbour |
| scmap (cluster) | — | Cluster-dependent; reported in Supplementary Fig. 6 only |
| CHETAH | — | Hierarchical classification tree traversed by correlation score |
| CIPR | — | Cluster-dependent; recomputes markers internally on the test set |

---

## Classical ML (25)

Train a classifier on `seurat_train`; ignore `markers`.

**scPred — 17 reported configurations.** One implementation, one caret method
string apart. `run_scPred.R` takes `model =` and `tool_registry.R` generates
the entries from `SCPRED_MODELS`.

| Reported (17) |
|---|
| `svmRadial` (package default, registered as `scPred`), `svmLinear`, `svmPoly`, `rf`, `knn`, `lda`, `nb`, `glm`, `glmboost`, `bayesglm`, `earth`, `mlp`, `nnet`, `avNNet`, `regLogistic`, `multinom`, `xgbTree` |

`SCPRED_MODELS` holds 18 entries: the 17 above plus `adaboost`, which was run
but is **not** in the reported rankings — see
[Run but not reported](#run-but-not-reported).

**The other 8:**

| Tool | Wrapper | Notes |
|---|---|---|
| scClassify | — | Weighted kNN ensemble over cell-type trees |
| singleCellNet | — | Random forest on top-scoring-pair features |
| CellTypist | — | Python; R shells out via temp CSV + stdout protocol |
| scAnnotate | — | Python; same shelling pattern |
| CaSTLe | — | XGBoost on discretised expression |
| scID | — | Fisher LDA with cluster-specific signatures |
| scibetR | — | Multinomial with maximum-entropy gene selection |
| scLearn | **shipped** | R; DCA with M3Drop feature selection; wants raw counts |

---

## Deep learning (8)

| Tool | Language | Wrapper | Notes |
|---|---|---|---|
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
| CAMLU | R | — |
| MARS | Python | — |
| scANVI | Python | — |
| scSemiCluster | Python | — |
| scSemiGAN | Python | — |

---

## Foundation models (5)

Phase 5 only. Each was fully fine-tuned on the same stratified 80% training
partition and evaluated on the identical held-out test partitions as Phase 2.

| Model | Backbone | Hardware |
|---|---|---|
| scBERT | Encoder-only transformer, Performer attention (~1M cells) | NVIDIA L4 |
| scGPT | Generative decoder transformer, FlashAttention (~33M cells) | NVIDIA L4 |
| Geneformer V2 | Rank-tokenisation transformer (~104M transcriptomes) | NVIDIA L4 |
| scFoundation | Linear-complexity attention (~50M cells) | NVIDIA L4 |
| C2S (Cell2Sentence) | Autoregressive LM, Pythia-410M backbone | NVIDIA RTX PRO 6000 |

**Not in the harness.** These ran in Google Colab against provider-specific
notebooks with their own environment setup and checkpoint handling, rather than
through either harness. The notebooks are released as run, outputs cleared, in
`as-run/foundation-models/`; results are in the data bundle
(`results/P8a FM - Synthetic/`, `results/P8b FM - Real/`).

---

## LLM-based (3)

Phase 4 only. Answer per cluster, in free text, from an API. Predictions pass
through `scoring/normalisation_pipeline.R` before scoring.

| Tool | Wrapper | Notes |
|---|---|---|
| mLLMCelltype | **shipped** | Multi-model consensus with uncertainty quantification. The wrapper takes `models =` and covers both single-model and consensus operation |
| GPTCelltype | — | Single-model prompting with per-cluster top DE markers (GPT-4) |
| CASSIA | — | Multi-agent pipeline with iterative refinement (GPT-5.4 annotation; Claude Sonnet 4.6 scoring and annotation boost) |

The reported mLLMCelltype result is the three-model consensus of Claude Sonnet 4.6,
GPT-5.4 and Gemini 3.1 Pro preview, the only entry `tool_registry.R` registers for
it. API keys are read from the environment at call time and never stored here.

---

## Run but not reported

Run and excluded from the reported rankings, for degenerate performance
(kappa ≈ 0), instability, runtime, or cluster-level output. Their results are in
the data bundle and their wrappers in `as-run/`, so the exclusion can be checked.

| Tool / configuration | Reason |
|---|---|
| scPred `adaboost` | kappa ≈ 0 across scenarios; also very slow |
| scAnnotatR | superseded by other classical ML entries |
| CALLR | cluster-level output; excluded from the cell-level comparison |
| scDeepSort | unstable |
| scGAD | degenerate performance |

**Tried and dropped.** These were tried during development but have no results
in the data bundle, so their wrappers are not released:

| Tool / configuration | Reason |
|---|---|
| scPred `glmnet` | unstable |
| scPred `mxnetAdam` | unstable (mxnet backend) |
| Garnett | ground-truth leakage in its fallback path |
| CellAssign | TensorFlow memory not observable by `peakRAM`; unstable |
| scAnno | degenerate performance |
| scDeepinsight, scTransSort, mtANN, ItClust, scnym | did not run to completion in a usable configuration |
| scMatch; mLLMCelltype with a single model (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro, DeepSeek v3.1 Terminus, Grok 4 Fast, Llama 4 Maverick, Qwen 3 Max) | no result kept |

---

## Adding a wrapper

The exemplars are meant to be extended from. Copy
`R/tools/TEMPLATE_run_tool.R` (or the `.py` twin), follow
[tool-wrapper-contract.md](tool-wrapper-contract.md), and register the result in
`R/tool_registry.R`. `run_benchmark.R` sources every `run_*.R` in `tools/`
automatically, but the registry — not the directory listing — is what makes a
tool dispatchable.
