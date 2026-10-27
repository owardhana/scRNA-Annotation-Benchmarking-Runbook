# environments — software versions

| File | What it records |
|---|---|
| `R_packages_mac.txt` | Version of every R package the code references, read from the R library (R 4.5.1, macOS arm64) that ran the R tools and analyses. Captured at release; a package may have been updated since its run. |
| `R_sessionInfo_mac_2026-10.txt` | `sessionInfo()` from the October 2026 re-run behind Supplementary Fig. 14 (Seurat 5.5.0, scPred 1.9.2, …). |
| `python_reproduce_mac.txt` | The Python packages `reproduce/` was verified with. |
| `colab.txt` | The Google Colab runtime of the Phase 5 foundation-model notebooks: GPU, Python and the package versions the notebooks installed or printed, and the model checkpoints. |

The per-tool conda environments of the Linux GPU workstation, which ran the
Python deep-learning and semi-supervised tools, are documented by the install
recipes recorded while the study ran:
`../as-run/benchmarking/python-DL-implementation/Install Notes DL.txt` (as
written) and `../harness/docs/environments/install_notes_DL.txt` (corrected so
it runs, and limited to the tools whose wrappers are released). They give the commands that built each environment, with the versions
that were pinned; anything left unpinned was resolved at install time. Each
Python wrapper in `../as-run/benchmarking/python-DL-implementation/DL-based/`
names the conda environment it ran in (`<name>_env`). The R install recipe is
`../harness/docs/environments/install_notes.R`.
