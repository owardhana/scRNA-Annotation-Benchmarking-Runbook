# 03 — Analysis example

One representative downstream analysis script, kept as an illustration of the
pattern. **Eight others are omitted by design.**

```
synthetic-L9-design/Plots_Analysis.R   the Phase 1 analysis
_cluster_dependent.R                   shared helper, sourced as ../_cluster_dependent.R
```

---

## What this is, and what it is not

This repository is reference code for the benchmarking harness. The harness
produces comparison tables; what happens to those tables afterwards — the
statistics, the figures, the manuscript panels — is analysis, and the full
analysis codebase is not shipped here.

The one script retained is the Phase 1 analysis, because Phase 1 is the design
centrepiece: the factorial Splatter simulation under a Taguchi L9(3⁴)
orthogonal array, where the array's purpose is to make each factor's main
effect estimable independently of the others. Reading it shows what the
harness output is *for* — how a `comparison_table.csv` becomes a claim about
which dataset property governs annotation accuracy.

The other eight (real-data validation, cross-platform transfer,
database/marker tools, LLM ontology scoring, foundation models, and the marker
supplements) follow the same shape against different inputs. They were largely
copies of each other: the flagship two shared roughly three-quarters of their
unique lines, and helper functions were redefined verbatim across files — in a
couple of cases with incompatible signatures. Shipping all nine would add
volume without adding information.

---

## Input

These scripts read comparison tables written by the harness — one row per tool,
sorted by pooled Cohen's kappa, with accuracy, macro F1, kappa, MCC, rare-type
F1, unassigned rate, runtime and peak memory. Nothing here regenerates those
tables; see [`../02-benchmarking-harness/`](../02-benchmarking-harness/).

No result tables or figures are included, so this script will not run as-is.
It is here to be read.

`_cluster_dependent.R` holds the shared logic for separating cluster-dependent
tools from cell-level ones. Tools given ground-truth cluster assignments
receive information the cell-level tools do not, so mixing them in one ranking
would compare unlike things; they are reported separately throughout.
