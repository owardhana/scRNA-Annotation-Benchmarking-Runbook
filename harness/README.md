# harness — the benchmarking harness, documented

A clean reimplementation of the study's benchmarking harness, written so a
reader can see how the benchmark is built and so a new tool can be benchmarked
the same way. It carries one worked wrapper per family plus annotated templates.
Every wrapper exactly as the study ran it is in [`../as-run/benchmarking/`](../as-run/benchmarking/),
and the scripts that regenerate the published results are in [`../reproduce/`](../reproduce/).

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
2. **One wrapper interface, identical across families.** Every tool is reached
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
docs/                    architecture, wrapper contract, tool roster, runbook, environments
```

- [`01-data-preparation/`](01-data-preparation/) — the seeded Splatter generator for
  the L9(3⁴) design; the two-stage preprocessing (per-dataset adapters driven by
  `datasets.csv`, which also records accessions and split seeds, then one shared
  pipeline in R and a step-for-step Scanpy twin); the ten-metric profiling and
  Borda-count mapping of real datasets onto synthetic scenarios.
- [`02-benchmarking-harness/`](02-benchmarking-harness/) — two harnesses. The R one
  carries most of the roster; the Python one exists because the deep-learning and
  semi-supervised tools are Python-native. Their differences are catalogued in
  [docs/architecture.md](docs/architecture.md#r-and-python-are-not-interchangeable).

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | Dataflow, execution modes, metrics, R/Python divergences |
| [tool-wrapper-contract.md](docs/tool-wrapper-contract.md) | The interface every tool implements |
| [tool-roster.md](docs/tool-roster.md) | All 63 configurations, the 47 compared in Phases 1–2, what was excluded |
| [marker-filtering.md](docs/marker-filtering.md) | `prepare_markers()` and its rationale |
| [resource-profiling.md](docs/resource-profiling.md) | What the runtime and memory numbers cover |
| [phase-1-runbook.md](docs/phase-1-runbook.md) | Reproducing Phase 1 on one workstation |
| [environments/](docs/environments/) | Install notes; exact versions are in [`../environments/`](../environments/) |

## Getting started

Reading order, if you want to understand the harness:

1. [docs/architecture.md](docs/architecture.md) — the dataflow diagram
2. [`02-benchmarking-harness/R/tools/run_scSorter.R`](02-benchmarking-harness/R/tools/run_scSorter.R) —
   the simplest complete wrapper
3. [docs/tool-wrapper-contract.md](docs/tool-wrapper-contract.md) — the rules
   that wrapper is following
4. [`02-benchmarking-harness/R/benchmarking_helpers.R`](02-benchmarking-harness/R/benchmarking_helpers.R) —
   metrics and the five execution modes

Running something:

```r
setwd("harness/02-benchmarking-harness/R")
source("run_benchmark.R")
available_tools()
```

Sourcing loads the harness and registers the shipped wrappers; it does not
start a run. See [docs/phase-1-runbook.md](docs/phase-1-runbook.md).
