"""
evaluate.py
============
Section 9 of the project plan: compute precision/recall/false-positive
rate for the graph model against the held-out ground_truth_fraud column,
and write a metrics_report.md comparing it to the rule-based baseline.

The ground_truth_fraud column is ONLY read here -- never fed into
generate features / detect.py / build_graph.py.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from src.baseline import run as run_baseline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def graph_model_metrics(risk_scores: pd.DataFrame, ground_truth: pd.DataFrame,
                         risk_threshold: float = 0.5) -> dict:
    merged = risk_scores.merge(ground_truth, on="customer_id", how="left")
    merged["ground_truth_fraud"] = merged["ground_truth_fraud"].fillna(False)
    merged["model_flag"] = merged["cluster_risk_score"] >= risk_threshold

    tp = int(((merged["model_flag"]) & (merged["ground_truth_fraud"])).sum())
    fp = int(((merged["model_flag"]) & (~merged["ground_truth_fraud"])).sum())
    fn = int(((~merged["model_flag"]) & (merged["ground_truth_fraud"])).sum())
    tn = int(((~merged["model_flag"]) & (~merged["ground_truth_fraud"])).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "model": "graph_model (Louvain + Isolation Forest)",
        "risk_threshold": risk_threshold,
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "n_flagged": int(merged["model_flag"].sum()),
        "n_customers": len(merged),
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report(baseline: dict, graph: dict, out_path: str) -> None:
    def pct_change(base, new, lower_is_better=False):
        if base == 0:
            return "n/a"
        change = (new - base) / base * 100
        if lower_is_better:
            change = -change
        if abs(change) < 0.05:
            return "0.0% (no change)"
        arrow = "improvement" if change > 0 else "regression"
        return f"{change:+.1f}% ({arrow})"

    lines = [
        "# Metrics Report: Graph Model vs. Rule-Based Baseline",
        "",
        "Computed against the held-out `ground_truth_fraud` column "
        "(injected synthetic labels, never used for training/feature "
        "selection).",
        "",
        "| Metric | Rule-Based Baseline | Graph Model | Change |",
        "|---|---|---|---|",
        f"| Precision | {_fmt_pct(baseline['precision'])} | {_fmt_pct(graph['precision'])} "
        f"| {pct_change(baseline['precision'], graph['precision'])} |",
        f"| Recall | {_fmt_pct(baseline['recall'])} | {_fmt_pct(graph['recall'])} "
        f"| {pct_change(baseline['recall'], graph['recall'])} |",
        f"| False Positive Rate | {_fmt_pct(baseline['false_positive_rate'])} "
        f"| {_fmt_pct(graph['false_positive_rate'])} "
        f"| {pct_change(baseline['false_positive_rate'], graph['false_positive_rate'], lower_is_better=True)} |",
        f"| Flagged customers | {baseline['n_flagged']} | {graph['n_flagged']} | -- |",
        f"| Total customers | {baseline['n_customers']} | {graph['n_customers']} | -- |",
        "",
        "## Rule-Based Baseline",
        f"- Rule: `{baseline['rule']}`",
        f"- TP={baseline['true_positives']} FP={baseline['false_positives']} "
        f"FN={baseline['false_negatives']} TN={baseline['true_negatives']}",
        "",
        "## Graph Model",
        f"- Louvain community detection (Layer 1) + Isolation Forest node "
        f"anomaly scoring (Layer 2), combined into a cluster risk score.",
        f"- Flag threshold: cluster_risk_score >= {graph['risk_threshold']}",
        f"- TP={graph['true_positives']} FP={graph['false_positives']} "
        f"FN={graph['false_negatives']} TN={graph['true_negatives']}",
        "",
        "## Resume Bullet (fill in from the numbers above)",
        f"> Built a graph-based return-fraud detection pipeline "
        f"(NetworkX, scikit-learn, Google Geocoding API) combining "
        f"community detection and anomaly scoring, achieving "
        f"{_fmt_pct(graph['precision'])} precision and {_fmt_pct(graph['recall'])} "
        f"recall on injected fraud patterns -- a "
        f"{pct_change(baseline['false_positive_rate'], graph['false_positive_rate'], lower_is_better=True)} "
        f"change in false positive rate versus a rule-based baseline.",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _summary_path(out_path: str) -> str:
    return os.path.join(os.path.dirname(out_path), "data", "metrics_summary.json")


def run(risk_scores_path: str = None, ground_truth_path: str = None,
        baseline_metrics_path: str = None, out_path: str = None,
        risk_threshold: float = 0.5):
    risk_scores_path = risk_scores_path or os.path.join(DATA_DIR, "risk_scores.csv")
    ground_truth_path = ground_truth_path or os.path.join(DATA_DIR, "ground_truth_fraud.csv")
    baseline_metrics_path = baseline_metrics_path or os.path.join(DATA_DIR, "baseline_metrics.json")
    out_path = out_path or os.path.join(os.path.dirname(DATA_DIR), "metrics_report.md")

    risk_scores = pd.read_csv(risk_scores_path)
    ground_truth = pd.read_csv(ground_truth_path)

    if not os.path.exists(baseline_metrics_path):
        run_baseline()
    with open(baseline_metrics_path) as f:
        baseline = json.load(f)

    graph = graph_model_metrics(risk_scores, ground_truth, risk_threshold)

    write_report(baseline, graph, out_path)

    summary = {"mode": "supervised", "baseline": baseline, "graph": graph}
    with open(_summary_path(out_path), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[evaluate] baseline precision={baseline['precision']:.3f} "
          f"recall={baseline['recall']:.3f} fpr={baseline['false_positive_rate']:.3f}")
    print(f"[evaluate] graph    precision={graph['precision']:.3f} "
          f"recall={graph['recall']:.3f} fpr={graph['false_positive_rate']:.3f}")
    print(f"[evaluate] wrote {out_path}")
    return baseline, graph


def write_unsupervised_report(risk_scores_path: str = None, out_path: str = None,
                               risk_threshold: float = 0.5):
    """
    Used when no ground-truth labels are available (real, unlabeled data).
    No precision/recall is possible without labels, so instead this
    reports how many customers/clusters were flagged at each risk tier,
    and is explicit that these numbers are NOT accuracy metrics.
    """
    risk_scores_path = risk_scores_path or os.path.join(DATA_DIR, "risk_scores.csv")
    out_path = out_path or os.path.join(os.path.dirname(DATA_DIR), "metrics_report.md")

    risk_scores = pd.read_csv(risk_scores_path)
    n_customers = len(risk_scores)
    n_clusters = risk_scores["community"].nunique()

    high = int((risk_scores["cluster_risk_score"] >= 0.75).sum())
    medium = int(((risk_scores["cluster_risk_score"] >= risk_threshold) &
                  (risk_scores["cluster_risk_score"] < 0.75)).sum())
    low = n_customers - high - medium

    lines = [
        "# Metrics Report: Unsupervised Risk Scoring",
        "",
        "No ground-truth fraud labels were provided for this dataset, so "
        "precision/recall cannot be computed. The numbers below describe "
        "how many customers the model flagged at each risk tier -- they "
        "are NOT accuracy metrics. To get precision/recall, re-run with "
        "`--labels your_labels.csv` (see src/data_loader.py for schema).",
        "",
        "| Risk Tier | Customers Flagged | Threshold |",
        "|---|---|---|",
        f"| High | {high} | cluster_risk_score >= 0.75 |",
        f"| Medium | {medium} | cluster_risk_score >= {risk_threshold} |",
        f"| Low | {low} | cluster_risk_score < {risk_threshold} |",
        "",
        f"- Total customers screened: {n_customers}",
        f"- Communities found (Louvain): {n_clusters}",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    summary = {
        "mode": "unsupervised",
        "n_customers": n_customers,
        "n_clusters": n_clusters,
        "high": high, "medium": medium, "low": low,
        "risk_threshold": risk_threshold,
    }
    with open(_summary_path(out_path), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[evaluate] unsupervised mode: {high} high-risk, {medium} medium-risk, "
          f"{low} low-risk customers out of {n_customers}")
    print(f"[evaluate] wrote {out_path}")
    return summary


if __name__ == "__main__":
    run()
