"""
explain.py
===========
Section 8 of the project plan: for every flagged cluster, auto-generate a
plain-English reason string driven off which features were the primary
drivers of that cluster's score (highest z-score features). No LLM
required. Detection without explanation is not deployable -- analysts
need to understand *why* an account was flagged before acting on it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.detect import FEATURE_COLS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

FEATURE_LABELS = {
    "n_purchases": "total purchases",
    "n_returns": "total returns",
    "return_ratio": "overall return ratio",
    "return_velocity_30d": "returns within a 30-day window",
    "return_velocity_60d": "returns within a 60-day window",
    "return_velocity_90d": "returns within a 90-day window",
    "max_category_return_ratio": "single-category return ratio",
    "avg_days_to_return": "average days between purchase and return",
    "min_days_to_return": "fastest purchase-to-return turnaround",
    "address_fanin": "distinct accounts sharing one address",
    "payment_fanin": "distinct accounts sharing one payment method",
}

# For these features, LOWER values are the anomalous direction.
LOWER_IS_RISKIER = {"avg_days_to_return", "min_days_to_return"}

RISK_TIER_THRESHOLDS = [(0.75, "High"), (0.5, "Medium"), (0.0, "Low")]


def _risk_tier(score: float) -> str:
    for threshold, label in RISK_TIER_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low"


def _top_driver_features(cluster_df: pd.DataFrame, feature_cols: list,
                          global_stats: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """Rank features by z-score of the cluster's mean vs the global population."""
    drivers = []
    for col in feature_cols:
        cluster_mean = cluster_df[col].mean()
        global_mean = global_stats.loc[col, "mean"]
        global_std = global_stats.loc[col, "std"] or 1e-9
        z = (cluster_mean - global_mean) / global_std
        if col in LOWER_IS_RISKIER:
            z = -z
        drivers.append({"feature": col, "z_score": round(float(z), 2),
                         "cluster_mean": round(float(cluster_mean), 2)})
    drivers.sort(key=lambda d: d["z_score"], reverse=True)
    return drivers[:top_n]


def _reason_sentence(cluster_df: pd.DataFrame, drivers: list[dict]) -> str:
    n = len(cluster_df)
    parts = []
    for d in drivers:
        col, z = d["feature"], d["z_score"]
        if z < 0.5:
            continue
        label = FEATURE_LABELS.get(col, col)
        if col == "address_fanin" and d["cluster_mean"] >= 2:
            parts.append(f"{int(d['cluster_mean'])} accounts share one address")
        elif col == "payment_fanin" and d["cluster_mean"] >= 2:
            parts.append(f"{int(d['cluster_mean'])} accounts share one payment method")
        elif col == "max_category_return_ratio":
            parts.append(f"returned over {d['cluster_mean']:.0%} of purchases in a single category")
        elif col in ("min_days_to_return", "avg_days_to_return"):
            parts.append(f"returns arriving within ~{d['cluster_mean']:.0f} days of purchase")
        elif col.startswith("return_velocity"):
            window = col.split("_")[-1]
            parts.append(f"{d['cluster_mean']:.0f} returns in a {window} window")
        elif col == "return_ratio":
            parts.append(f"an overall return ratio of {d['cluster_mean']:.0%}")

    if not parts:
        return (f"Cluster of {n} account(s) scored above the anomaly threshold, "
                f"but no single feature dominates -- review manually.")

    subject = f"{n} accounts" if n > 1 else "This account"
    joined = "; ".join(parts[:3])
    return f"{subject}: {joined}."


def generate_explanations(risk_scores: pd.DataFrame, top_clusters: int = None) -> list[dict]:
    global_stats = risk_scores[FEATURE_COLS].agg(["mean", "std"]).T

    clusters = []
    for community, cluster_df in risk_scores.groupby("community"):
        risk_score = float(cluster_df["cluster_risk_score"].iloc[0])
        drivers = _top_driver_features(cluster_df, FEATURE_COLS, global_stats)
        reason = _reason_sentence(cluster_df, drivers)
        clusters.append({
            "cluster_id": int(community),
            "cluster_size": int(len(cluster_df)),
            "cluster_risk_score": round(risk_score, 4),
            "risk_tier": _risk_tier(risk_score),
            "member_customer_ids": sorted(cluster_df["customer_id"].tolist()),
            "top_driver_features": drivers,
            "reason": reason,
        })

    clusters.sort(key=lambda c: c["cluster_risk_score"], reverse=True)
    if top_clusters:
        clusters = clusters[:top_clusters]
    return clusters


def run(risk_scores_path: str = None, out_path: str = None):
    risk_scores_path = risk_scores_path or os.path.join(DATA_DIR, "risk_scores.csv")
    out_path = out_path or os.path.join(DATA_DIR, "explained_clusters.json")

    risk_scores = pd.read_csv(risk_scores_path)
    clusters = generate_explanations(risk_scores)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(clusters, f, indent=2)

    n_flagged = sum(1 for c in clusters if c["risk_tier"] in ("High", "Medium"))
    print(f"[explain] generated explanations for {len(clusters)} clusters "
          f"({n_flagged} High/Medium risk) -> {out_path}")
    return clusters


if __name__ == "__main__":
    run()
