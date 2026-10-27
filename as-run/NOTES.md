# as-run — the code exactly as the study executed it

This folder is a frozen copy of the scripts that produced the published results,
taken from the working directories at submission. It exists so that a reader can
see precisely what ran. It is **not** a cleaned pipeline: drivers were edited in
place between runs, outputs were copied between folders by hand, and several
scripts carry dead branches from earlier phases of the work. The clean,
documented reimplementation is `../harness/`; the scripts that regenerate every
published number and figure from the deposited results are `../reproduce/`.

## What was changed on the way in

Nothing that alters behaviour. Every file is listed in `MANIFEST.tsv` with its
source path in the working tree, the sha256 of that source, and the edits applied:

| Edit | Where | Why |
|---|---|---|
| `paths-scrubbed:n` | n occurrences | Personal path prefixes replaced by placeholders: `<THESIS_ROOT>` (the project root on the Mac), `<MAC_HOME>`, `<WORKSTATION_HOME>`, `<WORKSTATION_THESIS_ROOT>` (the Linux GPU workstation). Substitute your own paths to run a script. |
| `outputs-cleared:n` | notebooks; n outputs | Cell outputs removed (they carried local paths and large data dumps); code cells are unchanged. |
| `colab-metadata-cleared:n` | Colab notebooks; n fields | Colab's per-cell `executionInfo` (Google account name and id, timestamps) and `outputId`, and the notebook's `authorship_tag`, removed. The GPU type is kept. |
| `renamed-from:…` | original name | Finder duplicate suffixes dropped, e.g. `run_CIForm (2).py` → `run_CIForm.py`. |
| `non-utf8-bytes-preserved` | | A few R files are not UTF-8; their bytes are copied unchanged. |
| `not-redistributed:no-upstream-licence` | 16 files | Copied from tool repositories that publish no licence, so not included. `fetch_upstream.py` restores them byte for byte (see "Upstream code" below). |
| `added:upstream-licence` | `MARS_package/LICENSE.md` | The MIT licence of the vendored MARS code, added beside it as MIT requires. |

Not copied: data, logs, bundled Python virtual environments (`scsa_env`), the
CellMarker 2.0 table and SCSA's `whole.db` (third-party; see below), editor and
assistant configuration files, compiled Python files, a `.git` folder inside
`MARS_package/`, the pilot phases that are not in the paper, and the wrappers of
tools with no results in the data bundle (see "Wrappers left out" below).

## Layout and where each phase ran

| Folder | Contents | Manuscript phase |
|---|---|---|
| `benchmarking/` | `Benchmarking.R` (driver), `benchmarking_helpers.R` (framework, five execution modes), every R wrapper by family folder, install notes | 1–4 |
| `benchmarking/python-DL-implementation/` | The Python side, as run on the Linux GPU workstation: `benchmarking_main.ipynb` (driver), `benchmarking_helpers.ipynb`, `Install Notes DL.txt`, and in `DL-based/` the 15 wrappers (`run_<tool>.py`) with the tool code they import (`run_CIForm_helper.py`, `MARS_package/`, and the scGAD, scSemiCluster and scSemiGAN packages; the unlicensed ones are restored by `fetch_upstream.py`). `python-DL-implementation2/` holds byte-identical copies of ten of the wrappers | 1–3 |
| `preprocessing/synthetic-L9/` | Splatter L9(3⁴) generator and sanity check, Seurat/AnnData loaders, profiler notebook | 1 |
| `preprocessing/real-datasets/` | One `*_Preprocessing.R` per screened dataset (44 candidates; the 9-dataset panel is listed in `../harness/01-data-preparation/preprocessing/datasets.csv`), the profiling and Borda-mapping notebook, loaders | 2 |
| `preprocessing/cross-platform/` | The 10 Phase 3 arms (CellBench, Pancreas, PBMCbench), profiling notebook, loaders | 3 |
| `scoring/` | `score_cells.py` for the database-marker (P7a) and LLM (P7b) arms: 0 / 0.5 / 1 ontology scoring by three LLM scorers | 4 |
| `supplementary/` | `make_S14_annotation_maps.R`, which re-runs three tools to recover per-cell predictions for Supplementary Fig. 14 | 2 |
| `foundation-models/` | The Google Colab notebooks for scBERT, scGPT, scFoundation, Geneformer V2 and C2S, as run (`benchmarking/colab/` in the working tree) | 5 |
| `upstream/`, `fetch_upstream.py` | Pins and patches for the 16 upstream files that are not redistributed, and the script that restores them | 1–3 |

Internal folder names follow the order the work was done in, not the
manuscript's phases: `P4` = Phase 1 (synthetic), `P5` = Phase 2 (real), `P6a/P6b`
= derived-marker supplement, `P7a/P7b` = Phase 4, `P8a/P8b` = Phase 5, `P9` =
Phase 3. The data bundle uses the same names.

## How the code was driven

- **`Benchmarking.R` is a scratchpad, not a pipeline.** It was edited in place
  for each run, so its final state shows the last configuration used (the
  Phase 3 pairing loop, a Phase 2 single-split block, and the Mode 3/4 calls).
  Run top to bottom it fails: `tools_for_trials` and `scina_rerun_configs` are
  used before they are defined, `classic-ML-Based/` is capitalised wrongly
  (this only works on a case-insensitive file system, as on macOS), and it
  sources six wrappers that are not included (see "Wrappers left out").
- **Outputs were moved by hand.** Each run wrote a timestamped comparison
  table to `recent_run_data/`; these were copied into the per-phase `results/`
  folders that the analysis scripts read. The copied tables are in the data
  bundle.
- **R tools ran on a Mac, Python deep-learning tools on a Linux workstation**,
  each in its own conda environment, called by subprocess from
  `python-DL-implementation/benchmarking_main.ipynb`. Each wrapper names the
  environment it ran in (`<WORKSTATION_HOME>/miniconda3/envs/<name>_env`); the
  recipes are in `Install Notes DL.txt`.
- **Foundation models ran in Google Colab**, one notebook per model. Each reads
  AnnData copies of the panel objects (`h5ad_real/`, written by
  `preprocessing/real-datasets/load_anndata_P5.ipynb`) and the model weights from
  a Google Drive folder, and writes its results back to Drive. The runtime, GPUs
  and checkpoints are in `../environments/colab.txt`; the results are in
  `results/P8b FM - Real/` in the data bundle.
- **C2S exists in two versions.** `run_C2S.py` (dated 20 May 2026) is the
  workstation wrapper (subprocess, `cell2sentence_env`);
  `foundation-models/C2S_FT.ipynb` (last run 27 May 2026) carries an in-process
  rewrite of the same function for Colab. The paper reports C2S as fine-tuned on
  Colab's RTX PRO 6000 (Methods; Supplementary Table 3), which is the notebook.
  Both are included.
- **Split seeds** (Phases 1, 2 and 5, Mode 2, stratified 80/20 with
  `caret::createDataPartition`): 123, except the real-data S3 (Nowakowski),
  which is 124. Seeds were not logged at run time; S3, S4 and S9 were confirmed
  by re-running and matching the published κ. Simulation seeds: 42, 123, 999.

## Known discrepancies in comments

- `benchmarking/LLM-based/run_GPTCelltype.R` says `Model: gpt-5.4` in its header,
  but the call is `gptcelltype(markers, model = "gpt-4")`, and the file predates
  every GPTCelltype result. GPT-4 is what ran, as Supplementary Table 3 states.

## Wrappers left out

A wrapper is included only if its tool has results in the data bundle; tools
that were run but not reported (CALLR, scAnnotatR, scDeepSort, scGAD, scPred
`adaboost`) are included because their results are deposited. Left out, as tools
tried during development with no result kept (`../harness/docs/tool-roster.md`):

- Python (`python-DL-implementation/DL-based/`): `run_ItClust.py`,
  `run_mtANN.py`, `run_scDeepinsight.py`, `run_scnym.py`, and
  `run_scTransSort.py` with `scTransSort_package/`. `benchmarking_main.ipynb`
  imports each wrapper only if it is present, so the notebook is unaffected.
- R: `marker-based/` and `database-based/run_Garnett.R`,
  `marker-based/run_CellAssign.R`, `classic-ML-based/run_scAnno.R`,
  `classic-ML-based/run_scPred_glmnet.R` and `DL-based/run_scPred_mxnetAdam.R`,
  all six sourced by `Benchmarking.R`; and `reference-based/run_scMatch.py`.
- LLM: the seven single-model mLLMCelltype wrappers
  (`run_mLLMCelltype_<model>.R`), and `run_mLLMCelltype_collab.R`, a copy of
  `run_mLLMCelltype.R` (the reported consensus) that `Benchmarking.R` never
  sources.

## Upstream code

Five tools ran from copies of their authors' code, kept beside the wrappers:

| Copy | Upstream (commit) | Licence | Here |
|---|---|---|---|
| `MARS_package/` | snap-stanford/mars (`f6f03ba`) | MIT | Included as run, with `LICENSE.md`. Four files carry small local edits |
| `scGAD_package/` | aimeeyaoyao/scGAD (`3328eda`) | none published | Restored by `fetch_upstream.py`; patches for 4 of 7 files |
| `scSemiCluster_package/` | xuebaliang/scSemiCluster (`075b295`) | none published | Restored; patches for all 3 files |
| `scSemiGAN_package/` | rafa-nadal/scSemiGAN (`e851f7e`) | none published | Restored; patch for `eval_real.py` |
| `run_CIForm_helper.py` (both folders) | zhanglab-wbgcas/CIForm (`fda67c7`), `CIForm_AnnData.py` | none published | Restored; patch of the local edits |

A repository without a licence does not permit redistribution, so those 16 files
are listed in `MANIFEST.tsv` (with the sha256 of the copy that ran) but not
included. Run `python3 as-run/fetch_upstream.py` to download each one at its
pinned commit, apply the local edits from `upstream/patches/`, and check the
result against that sha256; the restored files are git-ignored. The patches
carry only the changed lines. Each package's `__init__.py` was written for the
study and is included.

`run_scGAD.py` and `run_ACTINN.py` also embed portions of upstream code (see
`../THIRD_PARTY.md`).

## Third-party resources not redistributed

- **CellMarker 2.0** human table (Phase 4): `database-based/` read it as
  `Cell_Marker_Human.xlsx - human.tsv`, the "human" sheet of the official
  download exported as tab-separated text.
- **SCSA `whole.db`**, bundled with SCSA (https://github.com/bioinfo-ibms-pumc/SCSA).
- **sc-type** is sourced from its GitHub repository at run time by `run_scType.R`.

## Secrets

LLM wrappers read API keys from environment variables at call time
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`).
No key is stored in any file here.
