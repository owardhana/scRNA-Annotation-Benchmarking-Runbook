# Third-party code in this repository

The MIT licence in [`LICENSE`](LICENSE) covers the authors' own code. The files
below contain code from other projects and remain under their upstream licences.

| Files | Upstream | Licence | How it is included |
|---|---|---|---|
| `as-run/benchmarking/marker-based/SCSA.py`, `as-run/benchmarking/database-based/SCSA.py` | SCSA, https://github.com/bioinfo-ibms-pumc/SCSA | GPL-3.0 | Copied unmodified; the R wrapper calls it as a script |
| `as-run/benchmarking/python-DL-implementation/DL-based/run_ACTINN.py`, `harness/02-benchmarking-harness/python/tools/run_ACTINN.py` | ACTINN, https://github.com/mafeiyang/ACTINN | GPL-3.0 | The functions of ACTINN's `actinn_predict.py` are embedded in the wrapper; these two files are distributed under GPL-3.0 |
| `as-run/benchmarking/DL-based/NeuCA_helpers/` | NeuCA (Bioconductor) | GPL-2 | Vendored and updated for R 4.5 / Bioconductor 3.20 after the package was deprecated |
| `as-run/benchmarking/python-DL-implementation/DL-based/MARS_package/` | MARS, https://github.com/snap-stanford/mars (commit `f6f03ba`) | MIT | Vendored with small local edits to four files; upstream licence in `LICENSE.md` beside it |
| `reproduce/code/manuscript/publication_manuscript/figures/_vendor/scipilot/` | scipilot-figure-skill, https://github.com/Haojae/scipilot-figure-skill | MIT | Copied unmodified; commit in `SOURCE.txt`, licence in `LICENSE` beside it |

Not redistributed, because the upstream repositories publish no licence. The
tools ran from copies of this code; `as-run/fetch_upstream.py` downloads each file
at its pinned commit and re-applies the study's edits (the patches in
`as-run/upstream/patches/` carry only the changed lines), and checks the result
against the copy that ran:

| Files | Upstream (commit) |
|---|---|
| `…/DL-based/scGAD_package/*.py` (except `__init__.py`) | scGAD, https://github.com/aimeeyaoyao/scGAD (`3328eda`) |
| `…/DL-based/scSemiCluster_package/*.py` (except `__init__.py`) | scSemiCluster, https://github.com/xuebaliang/scSemiCluster (`075b295`) |
| `…/DL-based/scSemiGAN_package/*.py` (except `__init__.py`) | scSemiGAN, https://github.com/rafa-nadal/scSemiGAN (`e851f7e`) |
| `…/DL-based/run_CIForm_helper.py`, `…/python-DL-implementation2/run_CIForm_helper.py` | CIForm, https://github.com/zhanglab-wbgcas/CIForm (`fda67c7`), `CIForm_AnnData.py` |

Included, but adapting portions of upstream code that has no licence: the
wrapper `run_scGAD.py` (parts of scGAD's training loop).

`as-run/foundation-models/scGPT_pretrained.ipynb` re-implements scGPT's model
classes to load the released checkpoint, and some of its lines follow
`scgpt/model/model.py` of https://github.com/bowang-lab/scGPT (MIT, Copyright (c)
2022 suber). The C2S, scFoundation and scBERT notebooks install or clone their
model's code at run time (cell2sentence and scFoundation, Apache-2.0; scBERT,
GPL-3.0), and the Geneformer notebook loads the Apache-2.0 Hugging Face release
through `transformers`.

Used but not redistributed:

- **sc-type** (GPL-3.0), sourced from https://github.com/IanevskiAleksandr/sc-type at run time by the scType wrappers.
- **CellMarker 2.0**, and SCSA's bundled `whole.db`. Obtain them from their authors.
- **scBERT and scFoundation model code**, cloned from their repositories at run time by the Colab notebooks, and every **foundation-model checkpoint** (listed in `environments/colab.txt`).

Every other tool is installed from its own distribution (CRAN, Bioconductor,
PyPI, GitHub); see `environments/` and `harness/docs/environments/`.
