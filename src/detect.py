"""
detect.py
==========
Section 6.3 of the project plan: the two-layer detection design.

Layer 1 (graph level, from build_graph.py): Louvain communities -- groups
of accounts structurally connected in ways unrelated shoppers wouldn't be.

Layer 2 (node level, here): per-customer features scored with an
unsupervised Isolation Forest -- appropriate because we deliberately do
not train on the synthetic ground_truth_fraud labels.

Combination: cluster_risk_score = f(node anomaly scores of members,
cluster size, cluster density). This combination -- not either layer
alone -- is the core technical contribution.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

FEATURE_COLS = [
    "n_purchases", "n_returns", "return_ratio",
    "return_velocity_30d", "return_velocity_60d", "return_velocity_90d",
    "max_category_return_ratio", "avg_days_to_return", "min_days_to_return",
    "address_fanin", "payment_fanin",
]


def score_nodes(feat: pd.DataFrame, contamination: float = 0.08,
                 seed: int = 42) -> pd.DataFrame:
    """Layer 2: unsupervised Isolation Forest node-level outlier scoring."""
    feat = feat.copy()
    X = feat[FEATURE_COLS].fillna(0).values
    X = StandardScaler().fit_transform(X)

    model = IsolationForest(contamination=contamination, random_state=seed,
                             n_estimators=300)
    model.fit(X)
    # decision_function: higher = more normal, lower/negative = more anomalous.
    # Flip and min-max scale to [0, 1] so higher = more anomalous ("risk").
    raw = -model.decision_function(X)
    feat["node_anomaly_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return feat


def combine_cluster_risk(feat: pd.DataFrame, membership: dict) -> pd.DataFrame:
    """
    Combine Layer 1 (community structure) with Layer 2 (node anomaly score)
    into a single cluster_risk_score per customer.

    cluster_risk_score = 0.6 * mean(node_anomaly_score of cluster members)
                        + 0.25 * size_factor(cluster)
                        + 0.15 * density_factor(cluster)

    size_factor rewards clusters bigger than a lone customer (fraud rings
    are groups; a singleton "cluster" of one customer with no shared
    address/payment gets no size bonus). density_factor rewards clusters
    where anomaly scores are consistently high rather than one outlier
    dragging the average up.
    """
    feat = feat.copy()
    feat["community"] = feat["customer_id"].map(membership)

    cluster_stats = feat.groupby("community").agg(
        cluster_size=("customer_id", "count"),
        mean_anomaly=("node_anomaly_score", "mean"),
        std_anomaly=("node_anomaly_score", "std"),
    ).fillna(0)

    max_size = max(cluster_stats["cluster_size"].max(), 2)
    cluster_stats["size_factor"] = (cluster_stats["cluster_size"] - 1) / (max_size - 1)
    cluster_stats["size_factor"] = cluster_stats["size_factor"].clip(0, 1)

    # density: low std relative to mean = consistently risky cluster
    cluster_stats["density_factor"] = np.where(
        cluster_stats["mean_anomaly"] > 0,
        1 - (cluster_stats["std_anomaly"] / (cluster_stats["mean_anomaly"] + 1e-9)).clip(0, 1),
        0,
    )

    cluster_stats["cluster_risk_score"] = (
        0.6 * cluster_stats["mean_anomaly"] +
        0.25 * cluster_stats["size_factor"] +
        0.15 * cluster_stats["density_factor"]
    )

    feat = feat.merge(cluster_stats[["cluster_size", "mean_anomaly",
                                      "density_factor", "cluster_risk_score"]],
                       left_on="community", right_index=True, how="left",
                       suffixes=("", "_cluster"))
    return feat


def run(features_path: str = None, graph_path: str = None, out_path: str = None):
    features_path = features_path or os.path.join(DATA_DIR, "features.csv")
    graph_path = graph_path or os.path.join(DATA_DIR, "graph.gpickle")
    out_path = out_path or os.path.join(DATA_DIR, "risk_scores.csv")

    feat = pd.read_csv(features_path)
    with open(graph_path, "rb") as f:
        graph_data = pickle.load(f)
    membership = graph_data["community"]

    feat = score_nodes(feat)
    feat = combine_cluster_risk(feat, membership)

    feat = feat.sort_values("cluster_risk_score", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat.to_csv(out_path, index=False)

    print(f"[detect] scored {len(feat)} customers across "
          f"{feat['community'].nunique()} clusters -> {out_path}")
    return feat


if __name__ == "__main__":
    run()
