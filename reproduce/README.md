# reproduce — regenerate the paper's numbers and figures from the deposited data

Everything quoted in the paper is computed from tool-level result tables that
are deposited on Zenodo (the data bundle). This folder holds the scripts that
turn those tables into the published statistics and figures, so checking the
paper does not require re-running any annotation tool.

## Quick start

```bash
# 1. Download the data bundle from Zenodo (doi:10.5281/zenodo.23695089) and unzip it somewhere writable.
#    Check it:  (cd <bundle> && shasum -a 256 -c MANIFEST.sha256)
# 2. Run, from the repository root:
reproduce/run_all.sh <bundle> numbers    # every statistic in the text   (Python, ~5 min)
reproduce/run_all.sh <bundle> figures    # Figures 1-4, Supp. Figs 1-14  (Python, ~5 min)
reproduce/run_all.sh <bundle> analyses   # per-phase R analyses           (R, ~1 h)
```

`run_all.sh` copies `code/` into the bundle and runs each script from its own
folder, because the scripts sit at the same relative paths as the data they read.
One deposited file is rewritten: the `numbers` step regenerates
`manuscript/publication_manuscript/figures/_shared/tidy_kappa_metrics.csv`, so run
the checksum check before running, not after.
It writes a log per step to `<bundle>/reproduce_logs/` and a `SUMMARY.txt`; figures
go to `<bundle>/reproduce_figures/` (Supp. Figs 3, 5, 9, 12, 13 and 14 are written
beside their build scripts under `manuscript/publication_manuscript/figures/supplementary/`).
Set `PYTHON` and `RSCRIPT` to choose interpreters.

The folder names inside the bundle are the working names of the study:
`P4` = Phase 1 (synthetic), `P5` = Phase 2 (real), `P6a/P6b` = derived-marker
supplement, `P7a/P7b` = Phase 4 (database markers / LLM), `P8a/P8b` = Phase 5
(foundation models), `P9` = Phase 3 (cross-platform).

## What each step reproduces

### `numbers`

| Script (`code/manuscript/Notes/…`) | Reproduces |
|---|---|
| `tool_interaction_analysis.py` | Kendall's *W* over tools, variance components (scenario / tool / interaction / residual), pooled η² with a tool term; writes `results/_interaction_analysis/tidy_kappa.csv` |
| `paradigm_level_concordance.py` | The ten paradigm-mean κ values (synthetic and real), with a built-in gate that fails if any differs from the published value; paradigm-level *W* |
| `type3_eta_squared.py` | Per-family Type III η² of κ on the four dataset properties |
| `type3_eta_runtime_memory.py` | Type III η² for log₁₀ runtime and peak memory |
| `redo_all_with_corrected_profile.py` | Dataset-level κ–purity fits (R², ρ) for all phases, using the profiles in `Profiling_Real.csv` |
| `compute_results_numbers.py` | Per-family η², κ–purity scatter statistics and performance–cost correlations |
| `compute_phase4_5_numbers.py` | Phase 4 (database-marker, LLM) and Phase 5 (foundation-model) per-tool scores and purity fits |
| `fm_with_profiling_real.py` | Foundation-model κ versus purity |
| `v33_analysis/v33_multimetric.py` | Supplementary Note 1: the analyses repeated for accuracy, macro F1 and MCC |
| `v33_analysis/make_tidy_metrics.py` | `tidy_kappa_metrics.csv` (accuracy and MCC columns used by Figure 4) |
| `S3_S4_audit/fit_sensitivity.py` | Figure 3a purity fits with purities recomputed on the benchmarked cells |
| `S3_S4_audit/p57_recompute.py` | The Discussion's statements about the two hard scenarios |
| `figures/_shared/build_tidy_crossplatform.py` | `tidy_crossplatform.csv` (Phase 3, Supp. Fig. 8, Supp. Table 2) |
| `figures/_shared/build_tidy_tables.py` | Rebuilds every other table the figures read from the raw results and checks it against the deposited copy |

### `figures`

| Published | Build script (`code/manuscript/publication_manuscript/figures/…`) | Reads (`_shared/`) |
|---|---|---|
| Figure 1 | `Figure1/build_fig1.py` (needs `cairosvg` for the PNG/PDF; the SVG is written first) | `tidy_kappa`, `tidy_regimes`, `tidy_fm`, `supp_table4_roster` |
| Figure 2 | `Figure2/build_fig2.py` | `tidy_kappa` |
| Figure 3 | `Figure3/build_fig3.py` | `tidy_kappa`, `tidy_regimes`, `tidy_fm` |
| Figure 4 | `Figure4/build_fig4.py` | `tidy_kappa_metrics`, `tidy_regimes`, `tidy_fm`, `tidy_db_runtime`, `tidy_llm_runtime`, `supp_table4_roster` |
| Supp. Figs 1, 2, 4, 6, 7, 8, 10, 11, 14 | `supplementary/SNN/build_sNN.py` | see each script's header |
| Supp. Fig. 5 | `supplementary/_build_eta_figs.py` | `tidy_kappa` |
| Supp. Figs 3, 9, 13 | `supplementary/_build_misc_figs.py` | `tidy_kappa` |
| Supp. Fig. 12 | `supplementary/_build_runtime_figs.py` | `tidy_db_runtime`, `tidy_llm_runtime`, `tidy_fm_runtime` |

Supp. Fig. 2 tiles the UMAP images that the R analyses write (`results/P4 …/Plots/`,
`results/P5 …/Plots/`), so run `analyses` first. As the study ran it, the Phase 2
script renders the UMAP for one dataset only; to render all nine, uncomment the
other eight names in `p5_rds_names` near the end of
`code/results/P5 Real Life Datasets V2/Plots_Analysis.R`.

### `analyses`

The per-phase R scripts (`code/results/<phase>/Plots_Analysis.R` and companions).
They produce the exploratory and statistical outputs the paper's figures were
distilled from, the critical-difference and Wilcoxon tests, the LLM same-family
sensitivity table, and the UMAP tiles.

`code/results/P8a FM - Synthetic/Plots_Analysis.R` is included but not run. Its
results (foundation models fine-tuned on the synthetic data) are deposited but not
reported in the paper, and since an edit in June 2026 the script stops on its own
data table (it filters on `accuracy_mean`/`f1_mean`, which that table does not
have).

### Tables

- **Table 1** (the L9 design) is `_shared/l9_design.csv`, the design in
  `as-run/preprocessing/synthetic-L9/L9 Generator.R`.
- **Table 2** (the real-data panel) comes from `Data Creation/P5 Real Life Sets V2/Profiling_Real.csv`
  and the Borda mapping tables beside it (`Mapping_*.csv`). The profiling and
  mapping code is `as-run/preprocessing/real-datasets/Dataset_Profiling_and_Mapping.ipynb`.
  Two cell-type counts in the published table were corrected after profiling: S3
  has 47 types and S7 has 29, where the profile says 48 and 30, because the
  profiling counted unlabelled cells as a type. The analyses (and the `n_types`
  column of `tidy_kappa.csv`) use the profile as deposited.
- **Supp. Table 1** accessions are in `harness/01-data-preparation/preprocessing/datasets.csv`.
- **Supp. Table 2** comes from `Data Creation/P9 Cross Datasets/Profiling_P9.csv`, `Phase3_*.csv`.
- **Supp. Table 3** is the published table; `_shared/supp_table4_roster.csv` was
  extracted from an earlier version of it, and the figures read only its tool and
  family columns (for the per-family totals). Its mechanism text is out of date
  (e.g. GPTCelltype used GPT-4, as the published table says).

## How this was checked

Last run on 27 October 2026, from a `git archive` of this repository and the
unzipped Zenodo data file, on the authors' Mac (Python 3.12, R 4.5.1; versions in
`../environments/`). `run_all.sh … all` completed every step. Two runs on fresh
copies give byte-identical outputs.

- **numbers.** Every script printed output byte-identical to the authors' own run
  against the working tree, and the paradigm-mean gate passes. Every table it
  writes is identical to the authors' copy, with three exceptions. All three come
  from how those copies were saved, not from the code:
  - **`tidy_kappa.csv`.** The authors' copy in `results/_interaction_analysis/`
    predates the August 2026 family regrouping, so its `tool_category` labels differ.
  - **Runtimes.** In `tidy_kappa.csv` (three) and `tidy_kappa_metrics.csv` (two),
    five runtime values differ in the last digit of 17, from floating-point rounding.
  - **`RESULTS.txt` and `V33_MULTIMETRIC_RESULTS.txt`.** The saved `RESULTS.txt`
    predates the same regrouping (scLearn to classical ML, the similarity-based
    family). The saved `V33_MULTIMETRIC_RESULTS.txt` has a section appended by hand
    from a separate re-run.
- **Figure tables.** `build_tidy_tables.py` rebuilds all ten tables from the raw
  results; all ten match the deposited copies. One row in `tidy_kappa.csv` is
  not in any result file: SCINA on the real S3 dataset, which produced no output
  and is scored as a failed run (κ = 0), as every failure is in the paper. The
  deposited `tidy_kappa.csv` also carries a `tool_category` column from before the
  family names were revised; the figure scripts ignore it and use `catmap.py`.
- **analyses.** All eleven R steps ran on the bundle alone, and every table they
  write (weighting sensitivity, Phase 1 vs Phase 2 Wilcoxon tests, LLM same-family
  sensitivity, the Phase 3 category, failure, statistics and top-tool tables) is
  byte-identical to the authors' copy.
- **figures.** Figures 1–4 and Supp. Figs 1, 3–14 carry the same text as the
  published ones (every number, label and tick; compared in the SVG, or the PDF
  for Supp. Fig. 14). Figure 3 and Supp. Fig. 8, last rendered on the same Mac,
  are pixel-identical. The others were last rendered in a Linux environment with
  an earlier version of the figure helpers and differ in canvas cropping and
  anti-aliasing. For the same reason Supp. Fig. 4's colour-bar label −1 comes out
  with a hyphen instead of a minus sign: the vendored helpers set
  `axes.unicode_minus = False`. Some older SVGs store a tick label such as "0.25"
  as separate glyphs; compared character by character, they match. Figure 1's SVG
  is byte-identical to the authors' copy; with
  `cairosvg` 2.9.1 its PDF carries the same text as the published PDF, but on the
  test Mac it was set in Helvetica, because `cairosvg` takes fonts from the
  operating system and Liberation Sans was not installed there (see Requirements).
  Supp. Fig. 2 regenerates fully only after the one-line change described above.

## Requirements

- Python ≥ 3.10 with numpy, pandas, scipy, statsmodels, matplotlib (3.10), pillow,
  seaborn; `cairosvg` for Figure 1's PNG/PDF.
- The Liberation Sans font (any of: installed system-wide; the `.ttf` files in a
  folder named by `LIBERATION_FONTS`; or the R package `fontLiberation`). Figure 1's
  PNG and PDF need it installed system-wide, because `cairosvg` does not use the
  other two routes; without it they fall back to Helvetica (the SVG is unaffected).
- R ≥ 4.3 with tidyverse, ggplot2, patchwork, scales, ggrepel, ggridges, GGally,
  car, PMCMRplus, ComplexHeatmap, circlize, Cairo, extrafont; Seurat for the UMAP tiles.

The figure helpers in `code/manuscript/publication_manuscript/figures/_vendor/scipilot/`
are vendored, unmodified, from scipilot-figure-skill (MIT; commit in `SOURCE.txt`).

## What was changed relative to the working copies

Only paths. Absolute paths to the authors' machine became `BENCH_ROOT`, which
defaults to the bundle root worked out from each script's own location; the
figure scripts' import path for the figure helpers points at the vendored copy.
`MANIFEST.tsv` lists every file with its source and the edits applied.
