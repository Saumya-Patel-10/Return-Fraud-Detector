"""
baseline.py
============
Section 9 of the project plan: the rule-based baseline that must exist
before the graph model, so there's a real comparison point.

Rule: flag any customer with more than 5 returns in a rolling 30-day
window, OR a return ratio (returns / purchases) above 50%.
"""
from __future__ import annotations

import json
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def flag_customers(df: pd.DataFrame, max_returns_30d: int = 5,
                    max_return_ratio: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])

    purchase_counts = df.groupby("customer_id").size().rename("n_purchases")
    return_counts = df.groupby("customer_id")["returned"].sum().rename("n_returns")

    stats = pd.concat([purchase_counts, return_counts], axis=1).fillna(0)
    stats["return_ratio"] = stats["n_returns"] / stats["n_purchases"].replace(0, 1)

    # rolling 30-day return count, per customer, taking the max window seen
    returns_only = df[df["returned"]].copy()
    max_30d = {}
    for cid, g in returns_only.groupby("customer_id"):
        dates = g["purchase_date"].sort_values().reset_index(drop=True)
        best = 0
        for i in range(len(dates)):
            window = dates[(dates >= dates[i]) & (dates <= dates[i] + pd.Timedelta(days=30))]
            best = max(best, len(window))
        max_30d[cid] = best
    stats["max_returns_30d"] = stats.index.map(lambda c: max_30d.get(c, 0))

    stats["baseline_flag"] = (
        (stats["max_returns_30d"] > max_returns_30d) |
        (stats["return_ratio"] > max_return_ratio)
    )
    return stats.reset_index().rename(columns={"index": "customer_id"})


def run(clean_data_path: str = None, ground_truth_path: str = None,
        out_path: str = None):
    clean_data_path = clean_data_path or os.path.join(DATA_DIR, "clean_data.csv")
    ground_truth_path = ground_truth_path or os.path.join(DATA_DIR, "ground_truth_fraud.csv")
    out_path = out_path or os.path.join(DATA_DIR, "baseline_metrics.json")

    df = pd.read_csv(clean_data_path)
    gt = pd.read_csv(ground_truth_path)

    stats = flag_customers(df)
    merged = stats.merge(gt, on="customer_id", how="left")
    merged["ground_truth_fraud"] = merged["ground_truth_fraud"].fillna(False)

    tp = int(((merged["baseline_flag"]) & (merged["ground_truth_fraud"])).sum())
    fp = int(((merged["baseline_flag"]) & (~merged["ground_truth_fraud"])).sum())
    fn = int(((~merged["baseline_flag"]) & (merged["ground_truth_fraud"])).sum())
    tn = int(((~merged["baseline_flag"]) & (~merged["ground_truth_fraud"])).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "model": "rule_based_baseline",
        "rule": "returns_30d > 5 OR return_ratio > 0.5",
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "n_flagged": int(merged["baseline_flag"].sum()),
        "n_customers": len(merged),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[baseline] precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f}")
    print(f"[baseline] wrote {out_path}")
    return metrics, merged


if __name__ == "__main__":
    run()
