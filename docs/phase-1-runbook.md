# Runbook: reproducing Phase 1 on a single workstation

Phase 1 is the only phase that can be reproduced end to end without obtaining
any external data: the datasets are simulated, the generator is seeded, and the
whole design regenerates deterministically. Phases 2–5 depend on published
datasets that must be downloaded separately.

This is a reference archive rather than a turnkey pipeline. Expect to install
the tools you want to run and to point the harness at your own directories.

---

## What Phase 1 is

A factorial simulation study. Four factors, three levels each, arranged in a
Taguchi L9(3⁴) orthogonal array — nine scenarios instead of the 81 a full
factorial would need, chosen so that each factor's main effect is estimable
independently of the others.

| Factor | Levels |
|---|---|
| Cell count | 500 / 3,000 / 15,000 |
| Cell types | 5 / 15 / 30 |
| Class imbalance | balanced / mild (5:1) / severe (20:1) |
| DE strength | easy (~4.8× FC) / medium (~2.4×) / hard (~1.5×) |

Nine scenarios × three replicate seeds (42, 123, 999) = 27 datasets.

Every tool receives oracle inputs: a within-dataset labelled reference and
markers derived from the training split. The phase measures how dataset
structure alone governs accuracy, before any real-world confound is introduced.

---

## Requirements

| | |
|---|---|
| R | 4.3 or later |
| Key R packages | `splatter`, `Seurat`, `Matrix`, `dplyr`, `peakRAM`, `caret` |
| Disk | ~4 GB for the 27 simulated datasets |
| Memory | 16 GB is enough for the 500- and 3,000-cell tiers; the 15,000-cell scenarios are the constraint |
| Time | simulation ~1–2 h; a full tool sweep is dominated by whichever tools you install |

Per-tool installation is in [environments/install_notes.R](environments/install_notes.R)
and [environments/install_notes_DL.txt](environments/install_notes_DL.txt).

---

## 1. Generate the synthetic datasets

Sanity-check the simulation parameters before committing to all 27 runs. This
is what caught the original hard-DE level being set too low to produce any
detectable structure:

```bash
cd 01-data-preparation/synthetic-simulation
Rscript L9_Sanity_Check.R
```

Check the report for sparsity in the 70–85% range (the 10x Chromium range),
a visible PCA gradient across DE levels, and realised cell counts matching the
design.

Then generate:

```bash
Rscript L9_Generator.R
```

This writes `l9_benchmark_sets/S<n>/rep<m>/` for n in 1–9 and m in 1–3, each
containing `counts.mtx`, `genes.txt`, `cells.txt`, `metadata.csv`, and a
`config.txt` recording the realised sparsity, median UMI count and per-type cell
counts. `L9_DESIGN_SUMMARY.txt` is written alongside.

The seeds are fixed in `REPLICATE_SEEDS`, so re-running reproduces the same 27
datasets bit for bit. `FIXED_PARAMS` and `DE_LEVELS` in the generator document
why each Splatter parameter takes the value it does.

---

## 2. Build analysis-ready objects

```bash
cd ../loaders
SC_BENCH_STUDY_DIR=../synthetic-simulation/l9_benchmark_sets Rscript load_seurat.R
```

Applies the shared pipeline — QC, 2,000 variable features, log-normalisation,
scaling, 50 PCs — and writes one `.rds` per replicate. For Python tools, run
`load_anndata.py` over the same directory instead; the two are step-for-step
equivalent.

---

## 3. Run the benchmark

```r
setwd("02-benchmarking-harness/R")
source("run_benchmark.R")

available_tools()          # what is installed and registered here

Sys.setenv(SC_BENCH_DATA_DIR   = "/path/to/l9_benchmark_sets")
Sys.setenv(SC_BENCH_OUTPUT_DIR = "/path/to/results")

example_single_split(
  dataset = "S1/rep1/S1_rep1_subset.rds",
  tools   = c(marker_based, reference_based)
)
```

Phase 1 uses **mode 2**, a single stratified 80/20 split on
`Ground_Truth_Celltype`. Not cross-validation: the harness supports k-fold, but
no reported result uses it.

For the full sweep, loop over all 27 replicates with `example_batch()`. Two
things it handles that a naive loop would not: `fold_cache` is keyed by fold
index with no dataset identity, so it must be cleared between datasets; and the
15,000-cell objects need to be dropped and `gc()` called each iteration or a
long sweep exhausts memory.

Only the exemplar wrappers are shipped. To reproduce the full tool panel you
will need to write the remaining wrappers against
[tool-wrapper-contract.md](tool-wrapper-contract.md) — see
[tool-roster.md](tool-roster.md) for what the panel contained.

---

## 4. Output

Each run writes a comparison table to `SC_BENCH_OUTPUT_DIR`:

```
single_split_20260809_143022.csv
```

One row per tool, sorted by pooled Cohen's kappa, with accuracy, macro F1,
kappa, MCC, rare-type F1, unassigned rate, mean runtime, and mean/max peak
memory.

Kappa is the primary metric because the imbalance levels in the design make
accuracy misleading: under severe imbalance a tool that predicts the majority
type everywhere scores well on accuracy and near zero on kappa.

---

## Notes on the other phases

| Phase | What is needed beyond this runbook |
|---|---|
| 2 | The nine published datasets in `01-data-preparation/preprocessing/datasets.csv` (`phase == 2`). Most come from the Bioconductor `scRNAseq` package with pinned snapshot dates; the rest from their original publications. Then stage 1 and stage 2 as above. |
| 3 | The CellBench, Pancreas and PBMCbench blocks; run under mode 5. |
| 4 | CellMarker 2.0, obtained from its official distribution, plus API keys in the environment for the LLM tools. Modes 3 and 4. |
| 5 | Foundation-model checkpoints and a GPU. **Not reproducible from this repository** — Phase 5 ran in Colab; see [tool-roster.md](tool-roster.md#foundation-models-5--not-in-this-repository). |

A caveat for Phases 2–5: the analysis-ready panels were produced by stratified
random downsampling. `preprocess_dataset.R` here seeds that draw, so it is
reproducible going forward, but the original scripts did not — regenerating the
published panels from source will give a different random subset of the same
datasets at the same sizes.
