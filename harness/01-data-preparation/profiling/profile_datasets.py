"""profile_datasets.py

Compute the ten-metric structural profile of a dataset.

This is the measurement half of the scenario-mapping procedure. It answers the
question the Phase 2 design rests on: given a synthetic L9 scenario defined by
four deliberately manipulated factors, which real dataset most resembles it?
Answering that requires describing real and synthetic datasets on the same
axes, which is what the profile is.

The ten metrics, grouped by what they index:

    size            Total_Cells, Total_CellTypes, Min_Cluster_Size
    sequencing      Dropout_Pct, Median_UMI_Per_Cell
    imbalance       Norm_Shannon_Entropy
    separability    Silhouette_Score, KNN_Purity
    DE difficulty   Mean_Pi_Score
    dispersion      BCV_Dispersion_Proxy

METRICS_4 is the subset corresponding to the four axes the L9 array actually
manipulates; it is used as a second, deliberately narrower matching scheme.

ORDER OF OPERATIONS MATTERS, and differs from the analysis pipeline:

  * Dropout_Pct and Median_UMI_Per_Cell are computed on RAW COUNTS, before any
    normalisation. Normalising first would erase exactly what they measure.
  * HVG selection uses flavor='seurat' on log-normalised data, not the
    'seurat_v3' used in the loaders. seurat_v3 does not always populate
    dispersions_norm, which BCV_Dispersion_Proxy needs.
  * Geometry metrics are computed on the scaled HVG PCA embedding; DE
    difficulty on the pre-scale log-normalised copy.

Outputs one row per dataset; see borda_consensus.py for how those rows become
a scenario assignment.
"""

import gc
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import entropy
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


# Full reduced set: one representative per construct, no within-group redundancy.
METRICS_10 = [
    "Total_Cells", "Min_Cluster_Size", "Total_CellTypes",
    "Dropout_Pct", "Median_UMI_Per_Cell",
    "Norm_Shannon_Entropy",
    "Silhouette_Score", "KNN_Purity",
    "Mean_Pi_Score", "BCV_Dispersion_Proxy",
]

# Only the four axes directly manipulated by the L9 orthogonal array.
METRICS_4 = [
    "Total_Cells", "Total_CellTypes",
    "Norm_Shannon_Entropy", "Mean_Pi_Score",
]


def calc_knn_purity(adata, label_col="Ground_Truth_Celltype", n_neighbors=15):
    """Mean fraction of a cell's k nearest neighbours sharing its label.

    Uses adata.obsp['distances'] -- the binary k-NN graph with exactly k
    neighbours per cell -- rather than the weighted 'connectivities' graph.
    Connectivities have a variable number of nonzero entries per cell, so the
    denominator would differ cell to cell and the metric would not be
    comparable across datasets.
    """
    assert "X_pca" in adata.obsm, "Run sc.tl.pca() before calc_knn_purity()"
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca", metric="euclidean")

    distances = adata.obsp["distances"]
    labels = adata.obs[label_col].values

    purities = []
    for cell_index in range(adata.n_obs):
        neighbour_index = distances[cell_index, :].nonzero()[1]
        if len(neighbour_index) == 0:
            continue
        purities.append(
            np.sum(labels[neighbour_index] == labels[cell_index]) / len(neighbour_index)
        )

    return float(np.mean(purities)) if purities else np.nan


def profile_dataset(adata_raw, dataset_name, label_col="Ground_Truth_Celltype",
                    n_hvgs=2000, n_pcs=30, n_neighbors=15):
    """Compute the ten-metric profile of one dataset.

    Inputs:
        adata_raw: AnnData with RAW COUNTS in .X and labels in .obs[label_col]
        dataset_name: identifier written to the 'Sample' column
    Returns:
        dict of one row, with keys 'Sample' plus METRICS_10
    """
    result = {"Sample": dataset_name}

    # --- 1. Label-only metrics ------------------------------------------
    cluster_counts = adata_raw.obs[label_col].value_counts()
    n_cells = adata_raw.n_obs
    n_clusters = len(cluster_counts)
    proportions = cluster_counts.values / n_cells

    result["Total_Cells"] = n_cells
    result["Total_CellTypes"] = n_clusters
    result["Min_Cluster_Size"] = int(cluster_counts.min())

    # Normalised so datasets with different numbers of cell types stay
    # comparable: 1.0 is perfectly balanced, 0 is fully concentrated.
    if n_clusters > 1:
        result["Norm_Shannon_Entropy"] = round(
            entropy(proportions, base=np.e) / np.log(n_clusters), 4
        )
    else:
        result["Norm_Shannon_Entropy"] = np.nan

    # --- 2. Raw-count metrics, before any normalisation ------------------
    adata = adata_raw.copy()
    X = adata.X
    if hasattr(X, "toarray"):
        n_zeros = X.shape[0] * X.shape[1] - X.nnz
        library_sizes = np.asarray(X.sum(axis=1)).flatten()
    else:
        n_zeros = int(np.sum(X == 0))
        library_sizes = X.sum(axis=1)

    result["Dropout_Pct"] = round(n_zeros / (adata.n_obs * adata.n_vars) * 100, 2)
    result["Median_UMI_Per_Cell"] = round(float(np.median(library_sizes)), 1)

    # --- 3. Normalise, then HVGs with the 'seurat' flavour ---------------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor="seurat")

    hvg_mask = adata.var["highly_variable"]

    # Biological coefficient of variation proxy: median normalised dispersion
    # over the HVGs. Splatter's bcv.common has no direct empirical counterpart,
    # so this stands in for it on the real-data side.
    result["BCV_Dispersion_Proxy"] = round(
        float(np.median(adata.var.loc[hvg_mask, "dispersions_norm"])), 4
    )

    # --- 4. Geometry, on the scaled HVG PCA embedding --------------------
    adata_de = adata[:, hvg_mask].copy()    # pre-scale copy, for DE below
    adata_geo = adata[:, hvg_mask].copy()

    sc.pp.scale(adata_geo, max_value=10)
    sc.tl.pca(adata_geo, n_comps=min(n_pcs, adata_geo.n_obs - 1, adata_geo.n_vars - 1))

    if n_clusters > 1:
        result["Silhouette_Score"] = round(
            silhouette_score(adata_geo.obsm["X_pca"], adata_geo.obs[label_col].values), 4
        )
        result["KNN_Purity"] = round(
            calc_knn_purity(adata_geo, label_col, n_neighbors), 4
        )
    else:
        result["Silhouette_Score"] = np.nan
        result["KNN_Purity"] = np.nan

    # --- 5. DE difficulty (pi score) -------------------------------------
    # pi = -log10(adjusted p) * |log fold change|, averaged over the top 50
    # genes of each cell type: high when types are separated by both large and
    # confident differences. Groups with fewer than 3 cells are excluded
    # because the Wilcoxon test is undefined for them.
    mean_pi = np.nan
    if n_clusters > 1:
        valid_groups = cluster_counts[cluster_counts >= 3].index.tolist()
        if len(valid_groups) > 1:
            sc.tl.rank_genes_groups(adata_de, label_col, groups=valid_groups,
                                    method="wilcoxon", key_added="de")
            pi_scores = []
            for group in adata_de.uns["de"]["names"].dtype.names:
                pvals = adata_de.uns["de"]["pvals_adj"][group][:50]
                logfc = adata_de.uns["de"]["logfoldchanges"][group][:50]
                pi_scores.extend(-np.log10(pvals + 1e-300) * np.abs(logfc))
            if pi_scores:
                mean_pi = round(float(np.mean(pi_scores)), 4)
    result["Mean_Pi_Score"] = mean_pi

    del adata, adata_de, adata_geo
    gc.collect()
    return result


def profile_many(datasets, label_col="Ground_Truth_Celltype"):
    """Profile a {name: AnnData} mapping into a DataFrame indexed by name."""
    rows = []
    for name, adata in datasets.items():
        print(f"Profiling {name} ...")
        try:
            rows.append(profile_dataset(adata, name, label_col=label_col))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [!] {name}: {exc}")
    return pd.DataFrame(rows).set_index("Sample")
