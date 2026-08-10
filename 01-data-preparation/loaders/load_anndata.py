"""load_anndata.py

Stage 2 of data preparation, Python side: turn the canonical four-file quartet
into analysis-ready AnnData objects.

This is the Scanpy twin of load_seurat.R and is written to mirror it STEP FOR
STEP. That is the point of having two: R tools receive a Seurat object and
Python tools an AnnData object, and unless both were built the same way, any
difference in benchmark scores between an R tool and a Python tool could be the
preprocessing rather than the method.

The pipeline, in the order it must run:

    1. filter cells   min_genes = 200
    2. filter genes   min_cells = 3          <- AFTER cell filtering, not before
    3. stash raw counts in .layers['counts']
    4. highly variable genes   n_top_genes = 2000, flavor = 'seurat_v3'
    5. normalize_total  target_sum = 1e4
    6. log1p, stashed in .layers['log1p']
    7. subset to the 2,000 HVGs
    8. scale  max_value = 10
    9. PCA    n_comps = min(50, n_obs - 1)

Step order matters. Filtering genes before cells gives a different gene set,
because genes are judged on the surviving cells. Seurat and Scanpy also differ
on where HVG selection sits relative to normalisation: flavor='seurat_v3'
expects raw counts, which is why step 4 precedes step 5 here and why the counts
layer is stashed first.

Reads:  <study>/{original,for_use}/{counts.mtx,genes.tsv,barcodes.tsv,metadata.tsv}
Writes: <study>/{original,for_use}/<study>_<folder>_subset.h5ad

Usage:
    python load_anndata.py                 # every study under the cwd
    python load_anndata.py --base-dir /path/to/studies

A note on what is NOT here: the working version also produced a full-gene
variant (scaling and PCA over all genes rather than the 2,000 HVGs). No
reported analysis used it, so it is omitted.
"""

import argparse
import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread

N_TOP_GENES = 2000
MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3
SCALE_MAX_VALUE = 10
N_PCS = 50


def discover_target_dirs(base_dir: Path):
    """Find every <study>/for_use and <study>/original directory."""
    targets = []
    for study in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        for folder_type in ("for_use", "original"):
            candidate = study / folder_type
            if candidate.is_dir():
                targets.append(candidate)
    return sorted(targets)


def load_quartet(target_dir: Path) -> ad.AnnData:
    """Read the canonical four files into an AnnData."""
    required = ["counts.mtx", "genes.tsv", "barcodes.tsv", "metadata.tsv"]
    missing = [f for f in required if not (target_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {target_dir}")

    # .mtx is written genes x cells by the R side; AnnData wants cells x genes.
    counts = mmread(target_dir / "counts.mtx").T.tocsr()
    genes = pd.read_csv(target_dir / "genes.tsv", sep="\t", header=None)[0].values
    cells = pd.read_csv(target_dir / "barcodes.tsv", sep="\t", header=None)[0].values
    metadata = pd.read_csv(target_dir / "metadata.tsv", sep="\t", index_col=0)

    # Barcodes read back as strings; the index must match or obs silently
    # misaligns against X.
    metadata.index = metadata.index.astype(str)

    if "cell_type" in metadata.columns and "Ground_Truth_Celltype" not in metadata.columns:
        metadata["Ground_Truth_Celltype"] = metadata["cell_type"]
    if "Ground_Truth_Celltype" not in metadata.columns:
        raise ValueError(f"No Ground_Truth_Celltype column in {target_dir}/metadata.tsv")

    # Ground_Truth_Cluster is a 1-based integer ID per cell type, assigned in
    # sorted order so R and Python agree. The full-dataset modes use it as a
    # perfect clustering, standing in for a clustering step the benchmark is
    # not trying to measure.
    unique_types = sorted(metadata["Ground_Truth_Celltype"].dropna().unique())
    type_to_cluster = {t: i + 1 for i, t in enumerate(unique_types)}
    metadata["Ground_Truth_Cluster"] = metadata["Ground_Truth_Celltype"].map(type_to_cluster)

    adata = ad.AnnData(X=counts, obs=metadata)
    adata.var_names = genes
    adata.obs_names = cells
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # Unlabelled cells cannot be scored either way, so drop them before QC
    # rather than carrying them through and filtering at metric time.
    return adata[~adata.obs["Ground_Truth_Celltype"].isna()].copy()


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    """Apply the shared QC / normalisation / HVG / scale / PCA pipeline."""
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(
            f"QC removed everything: {adata.n_obs} cells, {adata.n_vars} genes remain"
        )

    print(f"  Cells after QC : {adata.n_obs}")
    print(f"  Genes after QC : {adata.n_vars}")

    # Tools that model counts read this layer rather than .X.
    adata.layers["counts"] = adata.X.copy()

    # seurat_v3 flavour expects raw counts, hence before normalisation.
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()

    adata_sub = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_sub, max_value=SCALE_MAX_VALUE)

    # PCA cannot ask for more components than cells - 1; small datasets in the
    # 500-cell tier hit this.
    sc.tl.pca(adata_sub, n_comps=min(N_PCS, adata_sub.n_obs - 1))

    return adata_sub


def main(base_dir: Path, overwrite: bool = False) -> None:
    target_dirs = discover_target_dirs(base_dir)
    print(f"Base directory : {base_dir}")
    print(f"Datasets found : {len(target_dirs)}\n")

    for target_dir in target_dirs:
        sim_name = f"{target_dir.parent.name}_{target_dir.name}"
        out_file = target_dir / f"{sim_name}_subset.h5ad"

        # Resume: these runs are long and are routinely restarted.
        if out_file.exists() and not overwrite:
            print(f"{sim_name}: {out_file.name} exists, skipping")
            continue

        print("=" * 70)
        print(f"Processing: {sim_name}")

        try:
            adata = load_quartet(target_dir)
            adata_sub = preprocess(adata)
            adata_sub.write(out_file)
            print(f"  Saved {out_file.name}\n")

            del adata, adata_sub
            gc.collect()
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [!] ERROR processing {sim_name}: {exc}\n")

    print("=" * 70)
    print("Pipeline finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."),
                        help="Directory containing the per-study folders")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild .h5ad files that already exist")
    args = parser.parse_args()
    main(args.base_dir, overwrite=args.overwrite)
