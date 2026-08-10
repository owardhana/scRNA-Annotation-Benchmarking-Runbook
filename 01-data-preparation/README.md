# 01 — Data preparation

Everything that happens before a tool sees a cell: simulating the Phase 1
design, turning published datasets into a common on-disk form, applying one
shared analysis pipeline, and profiling datasets so real data can be mapped
onto synthetic scenarios.

```
synthetic-simulation/   Splatter L9(3⁴) generator + its sanity check
preprocessing/          stage 1 — per-dataset adapters, driven by datasets.csv
loaders/                stage 2 — the shared pipeline, R and Scanpy twins
profiling/              ten-metric profiles and Borda-count scenario mapping
```

---

## Two stages, and why

Preprocessing is split in two, and the split carries the argument.

**Stage 1 is per-dataset and irreducibly messy.** One dataset stores its labels
in `WGCNAcluster`, another in `inferred.cell.type`, a third in `sample`. One
needs Ensembl IDs mapped to HGNC symbols; two mouse datasets need case folding.
Gene symbols arrive suffixed with locus identifiers and have to be split and
collapsed. There is no version of this that is not full of special cases.

**Stage 2 is one pipeline with no branches**, applied identically to every
dataset, in R and in Python.

Keeping them apart is what makes "every dataset had identical preprocessing"
a checkable claim instead of an assertion repeated in 44 files. All the
variation is confined to stage 1 and declared in a table; stage 2 has nowhere
to put an exception.

It is also why stage 1 writes plain files rather than an R object: the
canonical quartet is what lets the R and Python loaders provably start from the
same input.

---

## `preprocessing/` — stage 1

`preprocess_dataset.R` is driven by `datasets.csv`, which has one row per
dataset and holds the two facts that actually vary:

| Column | |
|---|---|
| `study` | dataset identifier, used as the output directory name |
| `phase` | `2` for the reported panel, `3` for cross-platform, `candidate` for screened-but-not-selected |
| `scenario` | mapped L9 scenario (S1–S9) for the reported panel |
| `source` / `source_ref` | where it comes from; `scRNAseq` entries pin an ExperimentHub snapshot date |
| `celltype_column` | which metadata column holds the ground truth |
| `gene_id_convention` | `symbol`, `mouse-uppercase`, or `ensembl-to-hgnc` |
| `notes` | per-dataset quirks worth knowing |

44 candidate datasets were screened; nine were selected for the reported panel.
Both are in the table, so the selection is visible.

```bash
Rscript preprocess_dataset.R Darmanis-Brain-2015
Rscript preprocess_dataset.R --phase 2      # just the reported panel
Rscript preprocess_dataset.R --all
```

Output, per dataset:

```
<study>/original/   counts.mtx  genes.tsv  barcodes.tsv  metadata.tsv
<study>/for_use/    the same four, after downsampling
```

`for_use` is what the benchmark runs on; `original` is kept so the downsampling
can be inspected.

### Zone downsampling

Datasets are downsampled onto one of three cell-count targets — **500 / 3,000 /
15,000** — by cell-type-stratified sampling, with a 15% tolerance band so a
dataset already near its target is left alone.

Those three numbers are the cell-count levels of the Phase 1 L9 design. The
downsampling snaps real datasets onto the same grid the synthetic scenarios were
built on, which is what makes the Phase 1 to Phase 2 comparison a comparison
rather than an analogy.

The draw is seeded (`DEFAULT_SEED = 123`). The original working scripts were
not, which is why the published `for_use` panels cannot be regenerated exactly
from source — see [phase-1-runbook.md](../docs/phase-1-runbook.md#notes-on-the-other-phases).

### `examples/`

Three complete adapters, kept because the generic loader cannot cover every
input format and a worked example is more useful than a branch:

| File | Shows |
|---|---|
| `Grun-Pancreas-2016_Preprocessing.R` | Bioconductor `scRNAseq` / SCE source; locus-suffix stripping and duplicate-symbol collapse |
| `Zheng68k_Preprocessing.R` | 10x Matrix Market triple; barcode sanitisation and the sparse-format requirement |
| `Segerstolpe-Pancreas-2016_Preprocessing.R` | Cross-platform (Phase 3); cell-type keep-lists and label-vocabulary harmonisation against the paired dataset |

The third is worth reading even if you do not need it. Phase 3 compares
datasets across sequencing platforms, which only works if both sides use the
same cell-type names — so one block harmonises its vocabulary by stripping a
trailing `" cell"` from one dataset's labels to match the other's. Small,
consequential, and easy to miss.

---

## `loaders/` — stage 2

`load_seurat.R` and `load_anndata.py` are step-for-step twins. R tools receive
a Seurat object and Python tools an AnnData object; unless both are built
identically, a score difference between an R tool and a Python tool could be
the preprocessing rather than the method.

| Step | R (Seurat) | Python (Scanpy) |
|---|---|---|
| 1. filter cells | `nFeature_RNA >= 200` | `filter_cells(min_genes=200)` |
| 2. filter genes | detected in >= 3 cells | `filter_genes(min_cells=3)` |
| 3. variable genes | `FindVariableFeatures(vst, 2000)` | `highly_variable_genes(2000, seurat_v3)` |
| 4. normalise | `NormalizeData(LogNormalize, 1e4)` | `normalize_total(1e4)` + `log1p` |
| 5. subset | to the 2,000 variable features | to the 2,000 HVGs |
| 6. scale | `ScaleData(scale.max = 10)` | `scale(max_value=10)` |
| 7. reduce | `RunPCA(npcs = 50)` | `pca(n_comps=min(50, n_obs-1))` |

Two ordering constraints that are easy to get wrong: step 2 must follow step 1,
because genes are judged on the cells that survive; and steps 3 and 4 are in
that order because `flavor='seurat_v3'` expects raw counts.

The loaders also assign `Ground_Truth_Cluster` — a 1-based integer per cell
type, in sorted order so R and Python agree. The full-dataset execution modes
use it as a perfect clustering, standing in for a clustering step the benchmark
is not trying to measure.

**Not included:** the working version also produced a full-gene variant, scaling
and running PCA over all genes rather than the 2,000 HVGs. No reported analysis
used it.

---

## `synthetic-simulation/`

`L9_Generator.R` produces the Phase 1 datasets: a Taguchi L9(3⁴) orthogonal
array over four factors at three levels each — cell count, cell types, class
imbalance, DE strength — giving nine scenarios instead of the 81 a full
factorial would need, arranged so each main effect is estimable independently.
Three replicate seeds (42, 123, 999) give 27 datasets.

The file is self-documenting by design: `FIXED_PARAMS` and `DE_LEVELS` carry a
paragraph of literature justification per parameter, because the credibility of
Phase 1 rests entirely on whether the simulation is calibrated to something
real. It is fully seeded and regenerates deterministically.

`L9_Sanity_Check.R` validates the output before committing to all 27 runs —
sparsity in the 10x range, a visible PCA gradient across DE levels, realised
cell counts matching the design. It earned its place: it is what caught the
hard-DE level originally being set so low that no structure was detectable at
all, and the recalibration is recorded in the generator's comments.

---

## `profiling/`

The machinery behind the Phase 2 claim that a given real dataset "corresponds
to" a given synthetic scenario.

`profile_datasets.py` computes a ten-metric structural profile — size, sequencing
depth, imbalance, separability, DE difficulty, dispersion — on the same axes for
real and synthetic data. Metric order matters: dropout and median UMI are
computed on raw counts, before any normalisation, since normalising first
erases exactly what they measure.

`borda_consensus.py` turns profiles into an assignment. Rather than one
distance calculation, which would make the result an artefact of one arbitrary
choice of metric set and space, the match is computed three ways — full
ten-metric, the four axes the L9 array manipulates, and a PCA space fitted on
real data only — and the three rankings are combined by Borda count. A majority
vote would discard how close each runner-up was; Borda keeps the whole ranking.

Confidence bands (strong / moderate / weak) come from the total Borda score out
of a maximum 27, and are what the manuscript's Phase 2 table reports.

**A limit worth stating**: the procedure ranks, it does not test fit. It assigns
every dataset to its best-scoring scenario whether or not that scenario is a
good match in absolute terms. In the study no candidate matched the L9-4
profile under any scheme, and that slot was filled by a force-assignment made
outside this code and reported as such.
