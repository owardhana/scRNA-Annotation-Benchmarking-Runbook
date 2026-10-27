# 02 — Benchmarking harness

The harness itself: two implementations, one contract.

```
R/                 the R harness — most of the tool roster
  benchmarking_helpers.R   metrics, folds, the five execution modes
  resource_tracking.R      runtime and peak-memory measurement
  tool_registry.R          name -> wrapper, including generated configurations
  run_benchmark.R          entry point and mode examples
  tools/                   TEMPLATE + six exemplar wrappers
python/            the Python harness — DL and semi-supervised paradigms
  benchmarking_helpers.py  the ported core
  benchmarking_main.py     entry point, registry, mode examples
  tools/                   TEMPLATE + two exemplar wrappers
scoring/           label normalisation and ontology scoring for free-text output
```

Read [../docs/architecture.md](../docs/architecture.md) first for the dataflow,
then [../docs/tool-wrapper-contract.md](../docs/tool-wrapper-contract.md) for
the interface.

---

## Why there are two harnesses

Most annotation tools ship as R packages, so the R harness carries the bulk of
the roster. But the deep-learning and semi-supervised paradigms are
Python-native, and several of those tools cannot be driven from R at all: they
want an AnnData object, a GPU, and a conda environment whose pinned versions of
torch or tensorflow conflict with everything else installed.

That constraint shapes the difference between them. The R harness calls tools
**in process**. The Python harness launches each tool as a **subprocess in its
own conda environment**, passing data through a temporary directory and letting
the child measure itself — a parent cannot observe a child's peak RSS or VRAM.

They are built to mirror each other but they are not the same program. The
divergences that matter — different resource keys, different marker derivation,
Python normalising defensively where R assumes the shared preprocessing already
did — are catalogued in
[architecture.md](../docs/architecture.md#r-and-python-are-not-interchangeable)
rather than smoothed over.

---

## The five execution modes

Defined in `benchmarking_helpers.R`. Which one a phase uses is a design
decision, not a convenience.

| Mode | Function | Arrangement | Used by |
|---|---|---|---|
| 1 | `run_all_tools_cv` | k-fold cross-validation | available; no reported result |
| 2 | `run_all_tools_single_split` | one stratified 80/20 split | Phases 1, 2, 5 |
| 3 | `run_all_tools_full_dataset_llm` | no split; ground-truth clusters as a perfect clustering | Phase 4 (LLM) |
| 4 | `run_all_tools_full_dataset_database` | no split; CellMarker 2.0 instead of derived markers | Phase 4 (database) |
| 5 | `run_separate_datasets` | train on A, test on B | Phase 3 |

Modes 3 and 4 have no split because the tools they serve have nothing to train.
Giving an LLM an 80% training partition would be meaningless — it has already
been trained, on a corpus nobody here controls. What those modes withhold is
the reference matrix; what they supply is a marker summary per cluster.

In both, the harness calls `tool_function(seurat_obj, seurat_obj, markers)`:
train and test are the same object. The signature does not change but its
meaning does.

Only modes 1, 2 and 5 exist on the Python side; no Python-native tool in the
roster consumes markers the way modes 3 and 4 require.

---

## Exemplar wrappers

One per paradigm, real code, chosen to show a different part of the contract.

| Wrapper | Paradigm | What it demonstrates |
|---|---|---|
| `R/tools/run_scSorter.R` | marker | The whole contract with almost no tool-specific machinery. **Read this first.** |
| `R/tools/run_SingleR.R` | similarity-based (reference) | SCE conversion at the boundary; confidence from a tool-native quantity rather than a probability |
| `R/tools/run_scPred.R` | classical ML | One implementation, many configurations (17 reported) — the registry generates the entries from a vector of caret method names |
| `R/tools/run_scLearn.R` | classical ML (R; non-neural) | A tool that wants raw counts, and returns predictions in its own order for a subset of cells — so the wrapper must realign |
| `R/tools/run_scType.R` | database-connected | One wrapper, two marker sources (`marker_source = "de"` / `"database"`) |
| `R/tools/run_mLLMCelltype.R` | LLM | Cluster-level answers expanded to cells; free text normalised against an ontology; extended return fields |
| `python/tools/run_ACTINN.py` | deep learning | The conda-subprocess isolation pattern in full |
| `python/tools/run_scArches.py` | semi-supervised | Consumes labelled train and unlabelled test simultaneously; reports GPU memory |

Plus `TEMPLATE_run_tool.R` and `TEMPLATE_run_tool.py`, annotated section by
section.

Every other wrapper, exactly as the study ran it, is in `../../as-run/benchmarking/`;
the roster is in [../docs/tool-roster.md](../docs/tool-roster.md). Foundation
models ran in Colab, not through this harness (`../../as-run/foundation-models/`).

### One implementation, many configurations

Several roster entries are one tool run under different settings. The working
repository carried a file per setting — 19 near-identical scPred wrappers and 8
near-identical mLLMCelltype wrappers, each differing by a single string. Worse,
because the resource-tracking code duplicated the algorithm body, that string
appeared *twice per file*, and the copies drifted.

Here the setting is an argument and the registry generates the entries:

```r
for (scpred_model in SCPRED_MODELS) {
  TOOL_REGISTRY[[paste0("scPred_", scpred_model)]] <- local({
    model <- scpred_model
    function(seurat_train, seurat_test, markers) {
      run_scPred_function(seurat_train, seurat_test, markers, model = model)
    }
  })
}
```

---

## `resource_tracking.R`

`with_resource_tracking()` wraps a wrapper's core annotation call and returns
its value alongside runtime and peak memory:

```r
tracked <- with_resource_tracking({ SomeTool::annotate(input) })
tracked$value; tracked$runtime_secs; tracked$peak_system_memory_mb
```

It exists because the alternative was writing each tool's core algorithm twice
— once in a `peakRAM`-available branch and once in a fallback — in every one of
70 wrappers.

The measurement window is deliberately narrow: model setup, training where
there is one, and prediction. It excludes the shared preprocessing, the single
per-dataset marker derivation, object conversion, and interpreter startup —
costs paid once per dataset and reused, which should not be charged to
whichever tool ran first. Details in
[../docs/resource-profiling.md](../docs/resource-profiling.md).

**Runtime is not comparable across paradigms.** An LLM tool's runtime is
network latency; a local tool's is compute. Same column, different quantity.

---

## `scoring/`

LLM tools and database-connected marker tools answer in free text. A model may
say "cytotoxic T lymphocyte" where the ground truth says "CD8+ T cell" — scored
naively that is an error, and the resulting number measures vocabulary rather
than biology.

`normalisation_pipeline.R` maps raw output onto the ground-truth vocabulary
through a five-stage cascade (exact, synonym, Jaro-Winkler, Cell Ontology
Wu-Palmer, unmappable) and emits two label sets: **strict** (exact ontology
identity) and **lenient** (near relatives accepted). Both are scored, and the
gap between them is itself a reported result — it separates a tool that named
the wrong cell type from one that named the right cell type differently.

`score_cells.py` implements the Phase 4 ontology-matched 0 / 0.5 / 1 score,
averaged over three independent LLM scorers.

These compute reported numbers, so they belong to the harness rather than to
the omitted analysis code.

---

## Running it

```r
setwd("R")
source("run_benchmark.R")   # loads the harness; does not start a run
available_tools()

Sys.setenv(SC_BENCH_DATA_DIR = "/path/to/your/data")
example_single_split(dataset = "Darmanis-Brain-2015_for_use_subset.rds")
```

```python
import benchmarking_main as bm
bm.available_tools()
bm.example_single_split("Darmanis-Brain-2015_for_use_subset.h5ad")
```

Both entry points expose one function per execution mode with the study's
arguments as defaults. They are functions rather than commented-out blocks so
they stay syntactically valid, but calling one requires data this repository
does not ship — see [../docs/phase-1-runbook.md](../docs/phase-1-runbook.md).

Paths come from `SC_BENCH_DATA_DIR` and `SC_BENCH_OUTPUT_DIR`, defaulting to
`data/` and `results/` beside the repository. The R harness resolves its own
module directory, so it does not care what the working directory is.
