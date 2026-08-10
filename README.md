# scRNA-seq cell-type annotation benchmark — reference code

Reference implementation of the benchmarking harness behind a thesis comparing
**63 single-cell RNA-seq cell-type annotation tool configurations** across five
phases, from controlled synthetic simulation to deployment-realistic conditions.

**This is reference code, not the complete analysis codebase.** It exists so a
reader of the manuscript can see how the benchmark is built: the harness
architecture, the tool-wrapper interface, the preprocessing pipeline, and the
dataflow between them. It carries one worked wrapper per methodological
paradigm rather than all 63, and no data.

If you want the design rationale in one document, start with
**[docs/architecture.md](docs/architecture.md)**.

---

## The shape of the problem

Comparing 63 annotation tools is mostly a plumbing problem. They disagree about
nearly everything — raw counts versus log-normalised expression versus a PCA
embedding; labelled reference cells versus a marker list versus a text prompt;
R packages, Python packages with mutually incompatible dependencies, and
network APIs. Left alone, each tool preprocesses the data its own way and the
comparison measures the plumbing.

Two rules hold the benchmark together:

1. **One preprocessing pipeline, applied once per dataset, before any tool
   runs.** Tools receive an object that is already filtered, normalised,
   reduced and split.
2. **One wrapper interface, identical across paradigms.** Every tool is reached
   through the same signature and returns the same shape:

   ```r
   run_<TOOL>_function(seurat_train, seurat_test, markers) -> list(
     predictions, true_labels, confidence_scores, cell_ids,
     runtime_secs, peak_system_memory_mb
   )
   ```

   Same signature for a marker tool that ignores `seurat_train`, a classifier
   that trains on it, and an LLM that sends gene names over the network.

---

## Layout

```
01-data-preparation/     simulation, preprocessing, loaders, dataset profiling
02-benchmarking-harness/ the harness itself — R and Python, side by side
03-analysis-example/     one representative downstream analysis script
docs/                    architecture, wrapper contract, tool roster, runbook
```

### [`01-data-preparation/`](01-data-preparation/)

Preprocessing runs in **two stages**, and the split is the point. Stage 1 is
per-dataset and full of special cases — this dataset keeps its labels in
`WGCNAcluster`, that one needs Ensembl IDs mapped to symbols, this mouse
dataset needs case folding. All of it is declared in `datasets.csv` and driven
by one script. Stage 2 is a single pipeline with no branches, applied
identically to everything, with an R version and a step-for-step Scanpy twin.

Splitting them is what makes "every dataset had identical preprocessing"
checkable rather than merely asserted: stage 2 has nowhere to hide a
per-dataset exception.

Also here: the seeded Splatter generator for the Phase 1 L9(3⁴) design, and the
ten-metric profiling and Borda-count procedure used to map real datasets onto
synthetic scenarios.

### [`02-benchmarking-harness/`](02-benchmarking-harness/)

Two harnesses. The R one carries most of the roster; the Python one exists
because the deep-learning and semi-supervised paradigms are Python-native and
several of those tools cannot be driven from R at all.

They differ in more than language — different resource keys, different marker
derivation, in-process versus conda-subprocess isolation. The differences are
catalogued in [docs/architecture.md](docs/architecture.md#r-and-python-are-not-interchangeable)
rather than papered over.

### [`03-analysis-example/`](03-analysis-example/)

One of the nine per-phase analysis scripts, kept as an example of the
downstream pattern. The other eight are omitted by design.

---

## Manuscript phases

| Phase | What it tests | Execution mode |
|---|---|---|
| 1 | Factorial Splatter simulation, Taguchi L9(3⁴), oracle inputs — main-effect attribution | single 80/20 split |
| 2 | Nine within-platform real datasets — does the Phase 1 structure survive real biology? | single 80/20 split |
| 3 | Cross-platform transfer across three technology blocks | separate datasets |
| 4 | Database-connected marker tools and LLM tools, ontology-aware scoring | full dataset |
| 5 | Fine-tuned transformer foundation models | single 80/20 split |

---

## Deliberately not here

Gaps you will notice, and why they are decisions rather than omissions:

- **All data.** No datasets, intermediate results or figures. Phase 1 data
  regenerates deterministically from the seeded generator; the real datasets are
  publicly available and their sources are recorded in
  `01-data-preparation/preprocessing/datasets.csv`.
- **Foundation-model wrappers.** Phase 5 ran in Google Colab against
  provider-specific notebooks, not through either harness here. Shipping
  partial notebooks would misrepresent how it was run.
- **55 of the 63 tool wrappers.** One exemplar per paradigm plus an annotated
  template; the full roster is in [docs/tool-roster.md](docs/tool-roster.md).
- **Eight of the nine analysis scripts**, and all figure generation.
- **CellMarker 2.0.** Redistributed by its authors; obtain it from the official
  source.

---

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | Dataflow, execution modes, metrics, R/Python divergences |
| [tool-wrapper-contract.md](docs/tool-wrapper-contract.md) | The interface every tool implements |
| [tool-roster.md](docs/tool-roster.md) | All 63 configurations; what ships, what was excluded |
| [marker-filtering.md](docs/marker-filtering.md) | `prepare_markers()` and why there is no p-value cutoff |
| [resource-profiling.md](docs/resource-profiling.md) | What the runtime and memory numbers cover |
| [phase-1-runbook.md](docs/phase-1-runbook.md) | Reproducing Phase 1 on one workstation |
| [environments/](docs/environments/) | R package list and per-tool conda recipes |

---

## Getting started

Reading order, if you want to understand the harness:

1. [docs/architecture.md](docs/architecture.md) — the dataflow diagram
2. [`R/tools/run_scSorter.R`](02-benchmarking-harness/R/tools/run_scSorter.R) —
   the simplest complete wrapper
3. [docs/tool-wrapper-contract.md](docs/tool-wrapper-contract.md) — the rules
   that wrapper is following
4. [`R/benchmarking_helpers.R`](02-benchmarking-harness/R/benchmarking_helpers.R) —
   metrics and the five execution modes

Running something:

```r
setwd("02-benchmarking-harness/R")
source("run_benchmark.R")
available_tools()
```

Sourcing loads the harness and registers the shipped wrappers; it does not
start a run. See [docs/phase-1-runbook.md](docs/phase-1-runbook.md).

---

## License

MIT — see [LICENSE](LICENSE).
