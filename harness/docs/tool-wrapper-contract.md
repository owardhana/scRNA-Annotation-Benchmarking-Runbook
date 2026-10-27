# The tool wrapper contract

Every tool in the benchmark is reached through a wrapper that satisfies this
contract. It is what lets 63 configurations across seven paradigms be compared
without the harness knowing anything about any of them.

Start from `02-benchmarking-harness/R/tools/TEMPLATE_run_tool.R` (or
`python/tools/TEMPLATE_run_tool.py`), then read `run_scSorter.R` as the
simplest complete example.

---

## Signature

```r
run_<TOOL>_function(seurat_train, seurat_test, markers) -> list(...)
```

Three positional arguments, no defaults, no configuration object. Configurable
tools take extra arguments *after* these three, with defaults, and the registry
supplies them — see `run_scPred.R` (`model =`) and `run_scType.R`
(`marker_source =`).

Python is the same shape with AnnData and a dict:

```python
run_<TOOL>_function(adata_train, adata_test, markers) -> dict
```

---

## Inputs

| Argument | What it is | Guaranteed |
|---|---|---|
| `seurat_train` | labelled training object | has `Ground_Truth_Celltype` |
| `seurat_test` | object to annotate | has `Ground_Truth_Celltype`; in modes 3/4 also `Ground_Truth_Cluster` |
| `markers` | `FindAllMarkers()` data frame | columns `gene`, `cluster`, `avg_log2FC`, `p_val_adj`, `pct.1`, `pct.2` |

Both objects arrive **already preprocessed** — log-normalised, 2,000 variable
features, scaled, 50 PCs. Do not redo that work in a wrapper. A tool whose
published defaults recompute normalisation internally keeps that behaviour, so
it stays faithful to its own documentation, but the wrapper does not add
preprocessing of its own.

`markers` is computed **once per dataset or fold** by the harness and reused
across every tool, so no tool is charged for it and every tool sees the same
evidence.

Three things a wrapper must not assume:

- **That it ignores an argument it does not use.** Marker tools receive
  `seurat_train` and must accept it; the signature is uniform so the harness
  needs no per-tool knowledge.
- **That train and test are disjoint.** In modes 3 and 4 the harness calls
  `tool_function(seurat_obj, seurat_obj, markers)`.
- **That `markers$cluster` holds cell-type names.** It holds cell-type names in
  modes 1, 2 and 5, and cluster IDs in modes 3 and 4. Cluster-consuming
  wrappers must detect which; `run_mLLMCelltype.R` shows how.

---

## Return

```r
list(
  predictions           = character(),  # one per test cell
  true_labels           = character(),  # one per test cell
  confidence_scores     = numeric(),    # one per test cell
  cell_ids              = character(),  # one per test cell
  runtime_secs          = numeric(1),
  peak_system_memory_mb = numeric(1)
)
```

The harness validates that the first four keys are present. It does not
validate their lengths — that is the wrapper's responsibility, and the single
most common way to get a wrapper wrong.

### The length and order invariant

`predictions`, `true_labels` and `cell_ids` **must be the same length and in the
same order**. A tool that returns predictions for a subset of cells, or in its
own ordering, must be realigned inside the wrapper before returning.

This matters because nothing downstream will catch a violation. A misaligned
prediction vector is compared element-wise against the wrong truth labels and
scores as noise — a plausible-looking bad result rather than an error. Two
worked realignments: `run_scLearn.R` (drops cells the tool never reported,
consistently across all four vectors) and `run_mLLMCelltype.R` (expands one
answer per cluster out to one per cell).

Cluster-level tools expand with a `match()` on the cluster column:

```r
cell_predictions <- cluster_calls[match(seurat_test$Ground_Truth_Cluster,
                                        names(cluster_calls))]
```

### `"Unknown"` is the sentinel, and the capital U is load-bearing

`"Unknown"` marks a cell the tool declined to call. `calculate_metrics()`
matches that **exact literal string** to compute the unassigned rate and to
exclude declined cells from the per-class metrics.

A wrapper that emits lowercase `"unknown"` gets two things wrong at once: the
declines are scored as though they were a real cell-type prediction, and the
tool reports an unassigned rate of zero. Neither failure is visible in the
output. Normalise the tool's own rejection label to `"Unknown"` explicitly —
`run_scLearn.R` does this for scLearn's `"unassigned"`.

### Confidence scores

Whatever the tool exposes for its own certainty: a class probability, a
correlation margin, a signature score. It is **not** required to be calibrated
and is **not** comparable across tools — it is recorded, not ranked on.

Tools exposing nothing comparable report a constant, and should say so in a
comment. Convention is `0` for cells predicted `"Unknown"`.

---

## Structure of a wrapper

Five sections, in this order. The template is annotated with each.

**1. Dependency guard.** `requireNamespace()` then `library()`. A guard rather
than a bare `library()` so a missing package fails that tool rather than
preventing the harness from loading.

**2. Failure sentinel.** A local `default_return()` closure producing the
correct shape with all-`"Unknown"` predictions. Every early exit returns it, so
a failed tool yields a comparable row instead of a hole in the results table.
Carry `runtime_secs` and `peak_system_memory_mb` through when the failure
happened *after* the timed call.

**3. Input adaptation.** Reshape into the tool's native format. Marker
consumers start from `prepare_markers()`. This section sits **outside** the
measurement window on purpose: format conversion is a cost of the harness, not
of the annotation method.

**4. Core annotation call.** Wrapped in `with_resource_tracking()`. Write it
once:

```r
tracked <- with_resource_tracking({
  tryCatch(SomeTool::annotate(tool_input), error = function(e) {
    warning(paste("TOOL failed:", e$message)); NULL
  })
})
results               <- tracked$value
runtime_secs          <- tracked$runtime_secs
peak_system_memory_mb <- tracked$peak_system_memory_mb
```

For a tool with a training step the window covers fitting **and** prediction;
for a marker-based tool with no training step, inference alone.

**5. Output adaptation.** Extract, NA-scrub, check the length invariant,
return.

Finally, an alias for readability:

```r
run_<TOOL> <- run_<TOOL>_function
```

---

## Marker filtering

Marker-consuming wrappers call `prepare_markers()`, supplied by the harness:

- positive markers only (`avg_log2FC > 0`)
- expressed in at least 10% of the target cells (`pct.1 >= 0.10`)
- top 20 per cluster by `avg_log2FC`
- **no p-value filter**

The absent p-value filter is deliberate. Under the hard-DE and rare-cell-type
conditions the L9 design deliberately creates, statistical power is low and
adjusted p-values approach 1.0; filtering on them would hand several tools an
empty marker list and score that as tool failure rather than as the intended
difficulty. Full rationale in [marker-filtering.md](marker-filtering.md).

Do not re-source `benchmarking_helpers.R` inside a wrapper. The harness loads it
once; `prepare_markers()` is already in scope.

**Database markers are the exception.** With `marker_source = "database"` the
CellMarker2.0 rows arrive with synthetic statistics — `avg_log2FC = 1.0`,
`p_val_adj = 0`, `pct.1 = 1.0` — because a curated database has no effect sizes
to report. Statistical filtering is meaningless on them and is skipped;
`run_scType.R` shows both paths in one wrapper.

---

## Extended returns

Tools producing free-text labels return more than the six standard keys:

| Key | Meaning |
|---|---|
| `lenient_predictions` | labels under the relaxed ontology threshold |
| `strict_kappa` | kappa under exact ontology identity |
| `lenient_kappa` | kappa accepting near relatives |
| `kappa_gap` | `lenient_kappa - strict_kappa` |
| `hallucination_rate` | fraction of answers mapping to nothing in the ontology |
| `cluster_mapping_df` | per-cluster audit trail, written to CSV by the harness |

The gap separates a tool that named the wrong cell type from one that named the
right cell type differently. See
[architecture.md](architecture.md#free-text-output-and-the-strictlenient-gap).

---

## Registration

A wrapper file is not runnable until it is registered. Add it to
`R/tool_registry.R`:

```r
TOOL_REGISTRY[["scSorter"]] <- run_scSorter_function
```

For a tool with several configurations, generate the entries from a table
rather than duplicating the wrapper — this is what replaced 19 near-identical
scPred files and 8 near-identical mLLMCelltype files:

```r
for (scpred_model in SCPRED_MODELS) {
  TOOL_REGISTRY[[paste0("scPred_", scpred_model)]] <- local({
    model <- scpred_model          # local() freezes the loop variable
    function(seurat_train, seurat_test, markers) {
      run_scPred_function(seurat_train, seurat_test, markers, model = model)
    }
  })
}
```

`run_benchmark.R` sources every `run_*.R` in `tools/` automatically, so a new
file is loaded without editing a source list. It still has to appear in the
registry to be dispatchable — the registry, not the directory, is the manifest.

---

## Python specifics

The contract is the same with two differences.

**Resource keys differ**: `method_walltime`, `peak_memory_mb`, `peak_vram_mb`
instead of `runtime_secs` and `peak_system_memory_mb`. They are measured
differently and are not interchangeable with the R figures.

**Isolation**: most Python tools pin incompatible versions of torch,
tensorflow, scanpy or numpy, so the wrapper does not import the tool. It
generates a self-contained script, runs it in the tool's own conda environment
via `conda run`, passes data through a `TemporaryDirectory`, and lets the child
process measure itself — a parent cannot observe a child's peak RSS or VRAM.
`run_ACTINN.py` is the worked example. Resolve conda through
`tools/conda_env.py` rather than hardcoding a path.
