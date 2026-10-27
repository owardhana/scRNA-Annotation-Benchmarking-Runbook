# scRNA-seq cell-type annotation benchmark — code

Code for

> Wardhana O, Zeng Z, Lu X. **Reference dataset structure outweighs method choice
> in single-cell cell-type annotation.**

The study benchmarks 63 cell-type annotation tool configurations from seven
families (marker-based, similarity-based, classical machine learning, deep
learning, semi-supervised, large language model, and fine-tuned foundation
models) in five settings: a Taguchi L9(3⁴) factorial simulation, a nine-dataset
real panel matched to it, cross-platform transfer, database- and LLM-driven
annotation, and foundation-model fine-tuning.

**Data.** The processed input objects and every per-run result table are on
Zenodo, [doi:10.5281/zenodo.23695089](https://doi.org/10.5281/zenodo.23695089). Raw data are at their original
accessions, listed in
[`harness/01-data-preparation/preprocessing/datasets.csv`](harness/01-data-preparation/preprocessing/datasets.csv)
and in Supplementary Table 1 of the paper.

---

## What is here

| Folder | What it is | Use it to |
|---|---|---|
| [`reproduce/`](reproduce/) | The analysis, statistics and figure scripts, with a runner | **Regenerate every number and figure in the paper** from the Zenodo data, without re-running any tool |
| [`as-run/`](as-run/) | Every script exactly as the study executed it: preprocessing, all tool wrappers, the driver, LLM scoring | See precisely what ran |
| [`harness/`](harness/) | A documented reimplementation of the benchmarking harness (R and Python), one worked wrapper per family, design docs | Understand the design, or benchmark a new tool the same way |
| [`environments/`](environments/) | R and Python package versions, the Colab runtime, and the install recipes of the per-tool conda environments | Rebuild the software environment |

## Reproducing the paper

There are two levels, and they come with different guarantees.

**1. From the deposited results (exact).** Download the data bundle from Zenodo
([doi:10.5281/zenodo.23695089](https://doi.org/10.5281/zenodo.23695089)), unzip it, then:

```bash
reproduce/run_all.sh /path/to/bundle numbers
```

```bash
reproduce/run_all.sh /path/to/bundle figures
```

This recomputes the statistics quoted in the text and the main and
supplementary figures from the tool-level result tables. On a fresh copy of the
bundle, every statistic script reproduces the authors' output byte for byte, and
the regenerated figures carry the published values. Details, the limits (Figure 1
needs `cairosvg`; Supplementary Fig. 2 needs a one-line change), the map from each
figure, table and number to its script, and the requirements are in
[`reproduce/README.md`](reproduce/README.md).

**2. Re-running the tools (best effort).** The benchmarked objects are in the
bundle (`benchmarking/data/`), with the 80/20 split seeds in `datasets.csv` (123;
124 for real-data S3). The wrappers that produced the results are in `as-run/`.
Sixteen files of tool code copied from repositories that publish no licence are not
redistributed; `python3 as-run/fetch_upstream.py` downloads them at their pinned
commits and re-applies the study's edits ([`as-run/NOTES.md`](as-run/NOTES.md)). A
re-run depends on the tool versions in `environments/`, a GPU for the deep-learning
and foundation-model tools, and API access for the LLM tools, whose models change
over time. Expect close but not identical numbers. The Phase 1 synthetic data can
be regenerated deterministically: see
[`harness/docs/phase-1-runbook.md`](harness/docs/phase-1-runbook.md).

## Mapping the paper to the code

| Paper | Ran with | Results in the data bundle |
|---|---|---|
| Phase 1 — factorial simulation | `as-run/preprocessing/synthetic-L9/`, `as-run/benchmarking/` | `results/P4 Synthetic Datasets V1/` |
| Phase 2 — real-data panel | `as-run/preprocessing/real-datasets/`, `as-run/benchmarking/` | `results/P5 Real Life Datasets V2/` |
| Phase 3 — cross-platform transfer | `as-run/preprocessing/cross-platform/`, `as-run/benchmarking/` | `results/P9 Cross Dataset/` |
| Phase 4 — database markers and LLMs | `as-run/benchmarking/database-based/`, `LLM-based/`, `as-run/scoring/` | `results/P7a …/`, `results/P7b …/` |
| Phase 5 — foundation models | `as-run/foundation-models/` (Google Colab notebooks) | `results/P8b …/` (reported); `results/P8a …/` (synthetic arm, not reported) |
| Derived-marker supplement | `as-run/benchmarking/marker-based/` | `results/P6a …/`, `results/P6b …/` |

The folder names (`P4`, `P5`, …) are the study's working names, kept so the paths
in the scripts resolve.

## Not included

- **Raw data**, available from the accessions above.
- **CellMarker 2.0**, used by the Phase 4 database tools. It is distributed by its
  authors; obtain it from the official source. SCSA's bundled `whole.db` ships
  with SCSA.
- **API keys.** The LLM tools read them from environment variables; none is stored here.

## Citation

Please cite the article above. To cite the data, use
[doi:10.5281/zenodo.23695089](https://doi.org/10.5281/zenodo.23695089); to cite the code, use
[doi:10.5281/zenodo.23698288](https://doi.org/10.5281/zenodo.23698288). Both DOIs cover all versions and open the
latest one. The metadata are in [`CITATION.cff`](CITATION.cff).

## License

The authors' code: MIT ([`LICENSE`](LICENSE)). A few files carry code from other
projects and keep their upstream licences (SCSA and ACTINN, GPL-3.0; NeuCA,
GPL-2; MARS and the figure helpers, MIT); they are listed in
[`THIRD_PARTY.md`](THIRD_PARTY.md), with the unlicensed tool code that is restored
rather than redistributed.
The data bundle is released under CC BY 4.0.
