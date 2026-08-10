# Architecture and dataflow

How the benchmark is put together: what flows through it, what measures what,
and where the two harnesses agree and disagree.

---

## The problem the architecture solves

Benchmarking 63 cell-type annotation tool configurations is mostly a plumbing
problem. The tools disagree about nearly everything: some want raw counts,
others log-normalised expression or a PCA embedding; some train on labelled
reference cells, others need only a marker list; some are R packages, some are
Python packages with mutually incompatible dependency sets, and three are
network APIs. Left alone, each tool imposes its own preprocessing and its own
notion of what a correct answer looks like, and the comparison measures the
plumbing rather than the methods.

The architecture answers that with two rules:

1. **One preprocessing pipeline, applied once per dataset, before any tool
   runs.** Tools receive an object that has already been filtered, normalised,
   reduced and split. They do not each get to prepare the data their own way.
2. **One wrapper interface, identical across paradigms.** Every tool is reached
   through the same function signature and returns the same shape, so the
   harness never needs to know what kind of tool it is calling.

Everything below is a consequence of those two rules.

---

## Dataflow

```
   published dataset                          Splatter L9 design
   (scRNAseq pkg, GEO, 10x mtx, ...)          (synthetic-simulation/)
            |                                          |
            v                                          |
   [stage 1] preprocess_dataset.R                       |
     per-dataset adapter, driven by datasets.csv        |
     - fetch / read in native format                    |
     - pick the cell-type column -> Ground_Truth_Celltype
     - align gene identifiers to human symbols          |
     - zone downsample (500 / 3,000 / 15,000)           |
            |                                          |
            v                                          v
   +-------------------------------------------------------+
   |  canonical quartet, per dataset                       |
   |    counts.mtx  genes.tsv  barcodes.tsv  metadata.tsv  |
   |    in  original/  and  for_use/                       |
   +-------------------------------------------------------+
            |                                  |
            v                                  v
   [stage 2] load_seurat.R              [stage 2] load_anndata.py
     one identical pipeline               the step-for-step twin
     QC -> HVG -> normalise ->            (same thresholds, same order)
     scale -> PCA
            |                                  |
            v                                  v
      <study>_subset.rds                 <study>_subset.h5ad
            |                                  |
            v                                  v
   +-----------------------+          +-----------------------+
   |   R harness           |          |   Python harness      |
   |   run_benchmark.R     |          |   benchmarking_main.py|
   +-----------------------+          +-----------------------+
            |                                  |
            v                                  v
      split / fold construction          same, via StratifiedKFold
      marker derivation (once)                  |
            |                                  |
            v                                  v
      TOOL_REGISTRY[name]                TOOL_REGISTRY[name]
            |                                  |
            v                                  v
      wrapper: adapt -> run -> adapt     wrapper: subprocess in its
      (in-process)                       own conda environment
            |                                  |
            +----------------+-----------------+
                             v
                  predictions, true_labels,
                  confidence_scores, cell_ids,
                  runtime, peak memory
                             |
                             v
                  calculate_metrics()
                  kappa, macro F1, MCC,
                  rare-type F1, unassigned rate
                             |
                             v
                  aggregate_all_tools()
                  -> comparison_table.csv
```

Free-text output takes one extra step. LLM tools and database-connected marker
tools answer in whatever vocabulary they like, so their predictions pass
through `scoring/normalisation_pipeline.R` before metrics are computed — see
[Free-text output](#free-text-output-and-the-strictlenient-gap) below.

---

## Why preprocessing is two stages

Stage 1 is per-dataset and full of special cases: this dataset stores its
labels in `WGCNAcluster`, that one needs Ensembl IDs mapped to symbols, this
mouse dataset needs case folding. Stage 2 is one pipeline with no branches.

Splitting them is what makes the claim "every dataset had identical
preprocessing" checkable instead of merely asserted. All the variation is
confined to stage 1 and declared in `datasets.csv`; stage 2 has nowhere to hide
a per-dataset exception. It is also why stage 1 writes a plain on-disk format
rather than an R object: the quartet is what lets the R and Python stage-2
loaders start from provably identical input.

The shared pipeline, in the order it runs:

| Step | R (Seurat) | Python (Scanpy) |
|---|---|---|
| 1. filter cells | `nFeature_RNA >= 200` | `filter_cells(min_genes=200)` |
| 2. filter genes | detected in >= 3 cells | `filter_genes(min_cells=3)` |
| 3. variable genes | `FindVariableFeatures(vst, 2000)` | `highly_variable_genes(2000, seurat_v3)` |
| 4. normalise | `NormalizeData(LogNormalize, 1e4)` | `normalize_total(1e4)` + `log1p` |
| 5. subset | to the 2,000 variable features | to the 2,000 HVGs |
| 6. scale | `ScaleData(scale.max = 10)` | `scale(max_value=10)` |
| 7. reduce | `RunPCA(npcs = 50)` | `pca(n_comps=min(50, n_obs-1))` |

Step 2 must follow step 1: genes are judged on the cells that survive, so
reversing the order yields a different gene set. Steps 3 and 4 are ordered as
they are because `flavor='seurat_v3'` expects raw counts.

---

## The wrapper contract in one line

```r
run_<TOOL>_function(seurat_train, seurat_test, markers) -> list(...)
```

Same signature for a marker tool that ignores `seurat_train`, a classifier that
trains on it, and an LLM that sends marker names over the network. The full
specification, including what the returned list must contain and the invariants
the harness enforces, is in [tool-wrapper-contract.md](tool-wrapper-contract.md).

What matters architecturally is that the harness dispatches purely by name:

```r
tool_function <- TOOL_REGISTRY[[tool_name]]
tool_result   <- tool_function(seurat_train, seurat_test, markers)
```

A tool that is not in the registry cannot run, whether or not a file for it
exists. The registry is the manifest.

### Per-paradigm requirements

All seven paradigms share the signature; they differ in what they actually
consume.

| Paradigm | Needs | Ignores | Notes |
|---|---|---|---|
| Marker | `markers`, log-normalised expression | `seurat_train` | Each wrapper reshapes markers into the tool's native signature format |
| Correlation / reference | labelled `seurat_train` | usually `markers` | Converts to SingleCellExperiment or uses Seurat anchors |
| Classical ML | labelled `seurat_train` | `markers` | Has a real training step; timing covers fit + predict |
| Deep learning | labelled `seurat_train`, often raw counts | `markers` | Several are Python-only |
| Semi-supervised | labelled train **and** unlabelled test at fit time | `markers` | All Python |
| Database-connected | CellMarker2.0 markers, cluster assignments | `seurat_train` | Same wrappers as marker tools, different marker source |
| LLM | cluster-level marker names, an API key | expression matrix | Answers per cluster, in free text |
| Foundation model | pre-trained weights, fine-tuning split | `markers` | Ran in Colab; **not in this repository** |

---

## The five execution modes

The harness has five ways of arranging train and test. Which one a phase uses is
a design decision, not a convenience.

| Mode | Function | Arrangement | Used by |
|---|---|---|---|
| 1 | `run_all_tools_cv` | k-fold cross-validation | available; no reported result |
| 2 | `run_all_tools_single_split` | one stratified 80/20 split | **Phases 1, 2, 5** |
| 3 | `run_all_tools_full_dataset_llm` | no split; ground-truth clusters as a perfect clustering | **Phase 4** (LLM) |
| 4 | `run_all_tools_full_dataset_database` | no split; CellMarker2.0 in place of derived markers | **Phase 4** (database) |
| 5 | `run_separate_datasets` | train on dataset A, test on dataset B | **Phase 3** |

Modes 3 and 4 have no split because the tools they serve have nothing to train.
Giving an LLM an 80% training partition would be meaningless — it has already
been trained, on a corpus nobody here controls. What those modes withhold
instead is the reference matrix; what they supply is a marker summary per
cluster.

**Mode 3 and 4 call `tool_function(seurat_obj, seurat_obj, markers)`** — train
and test are the same object. The signature does not change, but its meaning
does, and a wrapper that assumes train and test are disjoint will be wrong in
those modes.

### A keying subtlety in modes 3 and 4

`markers$cluster` is keyed by **cell-type name** when markers were derived under
`Ground_Truth_Celltype` identities (modes 1, 2, 5), and by **cluster ID** when
derived under `Ground_Truth_Cluster` (modes 3, 4). Cluster-consuming wrappers
must detect which they were given. `run_mLLMCelltype.R` shows the detection.

---

## What is measured, and what is deliberately not

`runtime_secs` and `peak_system_memory_mb` cover **the core annotation call
only**: model setup, training where there is a training step, and prediction.

Excluded, deliberately:

- the shared preprocessing, computed once per dataset and reused by every tool
- the single marker derivation per dataset or fold, likewise reused
- object conversion at the wrapper boundary (Seurat -> SingleCellExperiment,
  Seurat -> AnnData)
- package loading and, for tools that shell out, interpreter startup

The rule is that costs paid once per dataset and shared across tools are not
charged to whichever tool happened to run first. Details, and the per-tool
measurement table, are in [resource-profiling.md](resource-profiling.md).

**Runtime is not comparable across paradigms.** An LLM tool's runtime is
network latency and provider queueing; a local tool's is compute. They are
reported in the same column and should not be read as the same quantity.

Two mechanisms produce the numbers:

- **R, in-process**: `with_resource_tracking()` in `R/resource_tracking.R`
  wraps the core call in `peakRAM::peakRAM()`.
- **Python, subprocess**: the child process measures itself with
  `memory_profiler` and returns the figures with its results, because a parent
  cannot observe a child's peak RSS or VRAM. Tools that shell out from R to
  Python use a stdout protocol instead — the helper prints
  `RUNTIME_SECS:` and `PEAK_SYSTEM_MEMORY_MB:` as its last lines and the R side
  scrapes them (`parse_python_resource_output()`).

---

## Metrics

`calculate_metrics()` returns accuracy, macro F1, Cohen's kappa, multiclass MCC
(Gorodkin formulation), rare-type F1, and unassigned rate.

Kappa is the primary metric and `aggregate_all_tools()` sorts by it. The reason
is class imbalance: on a dataset where one cell type is 60% of the cells,
accuracy rewards a tool for predicting that type everywhere, and kappa does not
— chance agreement is the baseline, so kappa 0 means "no better than guessing
at the observed marginals".

**`"Unknown"`, capitalised exactly, is the framework sentinel.** It marks a cell
the tool declined to call. `calculate_metrics()` computes the unassigned rate
from that literal string and excludes those cells from the per-class metrics. A
wrapper that emits lowercase `"unknown"` has its declines scored as if they were
a real cell-type prediction, and reports an unassigned rate of zero.

Rare-type F1 is the F1 of the single least-frequent **true** cell type — a
direct read on whether a tool finds the populations that are usually the reason
for annotating in the first place.

---

## Free-text output and the strict/lenient gap

LLM tools and database-connected marker tools do not predict from a closed
vocabulary. A model may answer "cytotoxic T lymphocyte" where the ground truth
says "CD8+ T cell". Scored naively that is an error, and the resulting number
measures vocabulary rather than biology.

`scoring/normalisation_pipeline.R` maps raw output onto the ground-truth
vocabulary through a five-stage cascade:

1. exact match
2. curated synonym table
3. Jaro-Winkler string similarity (threshold 0.92 on **similarity**)
4. Cell Ontology Wu-Palmer semantic similarity
5. unmappable

It emits two label sets: **strict** (Wu-Palmer = 1.0, exact ontology identity)
and **lenient** (Wu-Palmer >= 0.8, near relatives accepted). Both are scored.
The gap between them is itself a reported result: it separates a tool that
named the wrong cell type from one that named the right cell type differently.

Phase 4 additionally scores agreement on a 0 / 0.5 / 1 ontology-matched scale
averaged over three independent LLM scorers — `scoring/score_cells.py`.

---

## R and Python are not interchangeable

Both harnesses exist because the tool roster spans both ecosystems. They are
built to mirror each other, but they are **not** the same program, and the
differences matter when comparing a number produced by one against a number
produced by the other.

| | R harness | Python harness |
|---|---|---|
| Object | Seurat | AnnData |
| Resource keys | `runtime_secs`, `peak_system_memory_mb` | `method_walltime`, `peak_memory_mb`, `peak_vram_mb` |
| GPU memory | not measured | measured |
| Normalisation | assumes the shared preprocessing did it | sniffs the matrix and re-normalises if it looks raw |
| Marker derivation | `FindAllMarkers` throughout | `rank_genes_groups`, **t-test** in the CV path, **Wilcoxon** in the single-split path |
| Marker filtering | `prepare_markers()` | **no equivalent** |
| 80/20 split | `caret::createDataPartition` | first split of `StratifiedKFold(5)` |
| Tool isolation | in-process | subprocess per tool, own conda environment |
| Timeout | none | yes (`timeout=86400`) |
| Modes | all five | 1, 2 and 5 only |

The consequential ones: Python applies no marker filtering, so a Python tool
consuming markers sees a different marker set than its R counterpart would; and
the two marker-derivation tests are not equivalent. Neither affects the
reported comparisons — the paradigms that Python serves do not consume markers
— but both would matter to anyone extending the Python harness to a
marker-based tool.

`fold_cache` in both harnesses is keyed by fold index alone, with no dataset
identity. A batch loop over several datasets must clear it between datasets or
the second dataset silently reuses the first one's splits and markers. Both
batch examples do this; a new caller must remember to.

---

## What is not in this repository

- **Data.** No datasets, no intermediate results, no figures. `01-data-preparation/`
  regenerates the synthetic data deterministically and documents how the real
  data was obtained.
- **Foundation-model wrappers.** Phase 5 ran in Google Colab against
  provider-specific notebooks; that code is out of scope here.
- **The other 55 tool wrappers.** One exemplar per paradigm is shipped; the full
  roster is in [tool-roster.md](tool-roster.md).
- **Per-phase analysis and figure code.** One representative script is retained
  in `03-analysis-example/`.
- **CellMarker2.0.** The database used by Phase 4 is redistributed by its
  authors; obtain it from the official source.
