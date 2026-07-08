# Metrics Report: Graph Model vs. Rule-Based Baseline

Computed against the held-out `ground_truth_fraud` column (injected synthetic labels, never used for training/feature selection).

| Metric | Rule-Based Baseline | Graph Model | Change |
|---|---|---|---|
| Precision | 38.7% | 79.0% | +104.2% (improvement) |
| Recall | 11.3% | 46.2% | +308.4% (improvement) |
| False Positive Rate | 2.7% | 1.9% | +31.8% (improvement) |
| Flagged customers | 31 | 62 | -- |
| Total customers | 800 | 800 | -- |

## Rule-Based Baseline
- Rule: `returns_30d > 5 OR return_ratio > 0.5`
- TP=12 FP=19 FN=94 TN=675

## Graph Model
- Louvain community detection (Layer 1) + Isolation Forest node anomaly scoring (Layer 2), combined into a cluster risk score.
- Flag threshold: cluster_risk_score >= 0.5
- TP=49 FP=13 FN=57 TN=681

## Resume Bullet (fill in from the numbers above)
> Built a graph-based return-fraud detection pipeline (NetworkX, scikit-learn, Google Geocoding API) combining community detection and anomaly scoring, achieving 79.0% precision and 46.2% recall on injected fraud patterns -- a +31.8% (improvement) change in false positive rate versus a rule-based baseline.
