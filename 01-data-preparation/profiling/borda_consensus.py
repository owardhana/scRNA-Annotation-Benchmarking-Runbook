"""borda_consensus.py

Map each real dataset onto the L9 synthetic scenario it most resembles.

This is the decision half of the scenario-mapping procedure; profile_datasets.py
is the measurement half. Phase 2 asks whether the structure-to-accuracy
relationships found on synthetic data survive on real biology, which only means
something if each real dataset can be said to occupy a particular position in
the design space. That assignment is made here.

WHY THREE SCHEMES AND A VOTE
----------------------------
A single distance calculation would make the assignment an artefact of one
arbitrary choice: which metrics, and in what space. So the match is computed
three ways, each defensible and each with a different bias:

  1. ten-metric   full profile, standardised Euclidean distance.
                  Broadest, but weights redundant axes more heavily.
  2. four-axis    only the axes the L9 array manipulates.
                  Closest to the design's own logic, but ignores everything else.
  3. real-PCA     scaler and PCA fitted on real data only, L9 anchors projected
                  into that space, distance in the components covering 80% of
                  variance. Decorrelates the metrics, at the cost of
                  interpretability.

Fitting the scaler and PCA on the REAL data and projecting the synthetic
anchors into it -- rather than the reverse, or pooling both -- is deliberate:
the scale of the space should be set by real biology, not by whatever range
Splatter happens to produce.

The three rankings are combined by Borda count rather than by taking a
majority. A majority vote discards how close the runner-up was; Borda keeps the
whole ranking, so a dataset that is second-best everywhere beats one that is
first-best once and last twice.

CONFIDENCE
----------
Maximum score is 27 (rank 1 under all three schemes, 9 points each).

    >= 24   strong      rank 1 or 2 under every scheme
    18-23   moderate
    < 18    weak / ambiguous

These are the confidence labels reported in the manuscript's Phase 2 panel
table.

A LIMIT WORTH STATING
---------------------
The procedure assigns every dataset to its best-scoring scenario, whether or
not that scenario is a good match in absolute terms. It ranks; it does not test
fit. In the study no candidate matched the L9-4 profile under any scheme, and
the S4 slot was filled by a force-assignment made outside this code.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCENARIO_COLUMNS = [f"S{i}" for i in range(1, 10)]

STRONG_THRESHOLD = 24
MODERATE_THRESHOLD = 18
PCA_VARIANCE_TARGET = 0.80


def build_l9_anchors(df_l9, scenario_col="Scenario"):
    """Average the replicates of each L9 scenario into a single anchor point.

    Each scenario was simulated three times with different seeds; the anchor is
    their mean, so replicate noise does not drive the assignment.
    """
    return df_l9.groupby(scenario_col).mean(numeric_only=True)


def distance_to_anchors(df_real, df_anchors, metric_cols, scaler=None):
    """Standardised Euclidean distance from each dataset to each anchor.

    The scaler is fitted on the REAL data (see module docstring) and applied to
    both sides. NaNs are filled with real-data medians for the same reason.

    Returns a datasets x scenarios DataFrame of distances.
    """
    real_values = df_real[metric_cols].copy()
    anchor_values = df_anchors[metric_cols].copy()

    medians = real_values.median()
    real_values = real_values.fillna(medians)
    anchor_values = anchor_values.fillna(medians)

    if scaler is None:
        scaler = StandardScaler().fit(real_values.values)

    return pd.DataFrame(
        cdist(scaler.transform(real_values.values),
              scaler.transform(anchor_values.values),
              metric="euclidean"),
        index=df_real.index,
        columns=df_anchors.index,
    )


def distance_in_real_pca_space(df_real, df_anchors, metric_cols,
                               variance_target=PCA_VARIANCE_TARGET):
    """Scheme 3: distance in a PCA space fitted on the real data only."""
    real_values = df_real[metric_cols].copy()
    anchor_values = df_anchors[metric_cols].copy()

    medians = real_values.median()
    real_values = real_values.fillna(medians)
    anchor_values = anchor_values.fillna(medians)

    scaler = StandardScaler().fit(real_values.values)
    real_scaled = scaler.transform(real_values.values)
    anchor_scaled = scaler.transform(anchor_values.values)

    pca = PCA().fit(real_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)

    real_pca = pca.transform(real_scaled)[:, :n_components]
    anchor_pca = pca.transform(anchor_scaled)[:, :n_components]

    return pd.DataFrame(
        cdist(real_pca, anchor_pca, metric="euclidean"),
        index=df_real.index,
        columns=df_anchors.index,
    )


def borda_from_distances(df_distances):
    """Convert a distance matrix into Borda points.

    Rank 1 (nearest) scores n_scenarios points, rank n scores 1. Ties take the
    better rank ('min'), so a tie never costs a scenario points relative to a
    scheme where it would have won outright.
    """
    ranked = df_distances.rank(axis=1, method="min", ascending=True).astype(int)
    return (df_distances.shape[1] + 1) - ranked


def confidence_label(score):
    """Map a total Borda score onto the reported confidence bands."""
    if score >= STRONG_THRESHOLD:
        return "strong"
    if score >= MODERATE_THRESHOLD:
        return "moderate"
    return "weak"


def consensus_mapping(df_real, df_l9, metrics_10, metrics_4, scenario_col="Scenario"):
    """Run all three schemes and aggregate them by Borda count.

    Inputs:
        df_real: profile table for the real datasets, indexed by dataset name
        df_l9:   profile table for the L9 replicates, with a scenario column
        metrics_10, metrics_4: the two metric sets (see profile_datasets.py)

    Returns a DataFrame indexed by dataset with:
        Consensus_Scenario, Consensus_Borda_Score, Runner_Up_Scenario,
        Runner_Up_Score, Score_Gap, Confidence, plus the per-scenario totals.
    """
    anchors = build_l9_anchors(df_l9, scenario_col=scenario_col)

    borda_tables = [
        borda_from_distances(distance_to_anchors(df_real, anchors, metrics_10)),
        borda_from_distances(distance_to_anchors(df_real, anchors, metrics_4)),
        borda_from_distances(distance_in_real_pca_space(df_real, anchors, metrics_10)),
    ]

    borda_total = borda_tables[0]
    for table in borda_tables[1:]:
        borda_total = borda_total.add(table, fill_value=0)
    borda_total.index.name = "Dataset"

    consensus_scenario = borda_total.idxmax(axis=1).rename("Consensus_Scenario")
    consensus_score = borda_total.max(axis=1).rename("Consensus_Borda_Score")

    # The runner-up and the gap to it are the honest reading of how decisive an
    # assignment was: a top score of 24 means little if the runner-up scored 23.
    runner_up_scenario = borda_total.apply(lambda row: row.nlargest(2).index[-1], axis=1)
    runner_up_score = borda_total.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)

    result = pd.concat([consensus_scenario, consensus_score, borda_total], axis=1)
    result["Runner_Up_Scenario"] = runner_up_scenario
    result["Runner_Up_Score"] = runner_up_score
    result["Score_Gap"] = consensus_score - runner_up_score
    result["Confidence"] = consensus_score.apply(confidence_label)

    return result.sort_values(
        ["Consensus_Scenario", "Consensus_Borda_Score", "Score_Gap"],
        ascending=[True, False, False],
    )


def print_consensus(result):
    """Print the consensus table grouped by scenario, best matches first."""
    print(f'{"Score":>6}  {"Gap":>5}  {"Confidence":<10}  {"Runner-up":<8}  Dataset')
    print("-" * 78)

    current_scenario = None
    for dataset, row in result.iterrows():
        if row["Consensus_Scenario"] != current_scenario:
            current_scenario = row["Consensus_Scenario"]
            print(f"\n-- {current_scenario} --")
        print(f"  {row['Consensus_Borda_Score']:>4.0f}   {row['Score_Gap']:>4.0f}  "
              f"{row['Confidence']:<10}  {row['Runner_Up_Scenario']:<8}  {dataset}")

    print("\n" + "-" * 78)
    print(f"strong >= {STRONG_THRESHOLD} | moderate {MODERATE_THRESHOLD}-{STRONG_THRESHOLD - 1} "
          f"| weak < {MODERATE_THRESHOLD}   (max possible 27)")
