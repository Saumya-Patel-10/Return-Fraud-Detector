"""
run_pipeline.py
================
Single entry point (Section 12, "Definition of Done"): runs the whole
pipeline end-to-end from raw data (synthetic OR real) through
explainability and metrics reporting.

    python run_pipeline.py                          # full run, synthetic data
    python run_pipeline.py --skip-data               # reuse existing data/clean_data.csv
    python run_pipeline.py --dashboard               # also launch the Flask dashboard after
    python run_pipeline.py --input real.csv          # run on a REAL transactions CSV
                                                       # (unsupervised: risk scores only,
                                                       # no precision/recall)
    python run_pipeline.py --input real.csv \\
        --labels real_labels.csv                     # real data + known fraud labels
                                                       # (adds precision/recall vs baseline)

See src/data_loader.py for the exact CSV schema --input/--labels expect.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from src import generate_data, build_graph, baseline, features, detect, explain, evaluate, data_loader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GT_PATH = os.path.join(DATA_DIR, "ground_truth_fraud.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-data", action="store_true",
                         help="reuse existing data/clean_data.csv instead of regenerating")
    parser.add_argument("--input", type=str, default=None,
                         help="path to a REAL transactions CSV to run the pipeline on "
                              "instead of synthetic data (see src/data_loader.py for schema)")
    parser.add_argument("--labels", type=str, default=None,
                         help="optional path to a ground-truth labels CSV "
                              "(customer_id, ground_truth_fraud) to pair with --input, "
                              "enabling precision/recall metrics on real data")
    parser.add_argument("--n-customers", type=int, default=800)
    parser.add_argument("--n-transactions", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--dashboard", action="store_true",
                         help="launch the Flask dashboard after the pipeline finishes")
    args = parser.parse_args()

    t0 = time.time()

    # --- Step 1: data ---
    if args.input:
        print(f"\n=== [1/6] Loading real data from {args.input} ===")
        has_ground_truth = data_loader.run(args.input, labels_path=args.labels)
    elif not args.skip_data:
        print("\n=== [1/6] Generating base data + injecting synthetic fraud ===")
        generate_data.run(n_customers=args.n_customers,
                           n_transactions=args.n_transactions, seed=args.seed)
        has_ground_truth = True
    else:
        print("\n=== [1/6] Skipping data generation (--skip-data, reusing existing files) ===")
        has_ground_truth = os.path.exists(GT_PATH)

    print("\n=== [2/6] Building graph + Louvain community detection ===")
    build_graph.run()

    print("\n=== [3/6] Computing rule-based baseline ===")
    if has_ground_truth:
        baseline.run()
    else:
        print("[baseline] skipped -- no ground-truth labels available "
              "(pass --labels to enable precision/recall)")

    print("\n=== [4/6] Feature engineering ===")
    features.run()

    print("\n=== [5/6] Isolation Forest scoring + cluster risk combination ===")
    detect.run()

    print("\n=== [6/6] Explainability + evaluation report ===")
    explain.run()
    if has_ground_truth:
        evaluate.run(risk_threshold=args.risk_threshold)
    else:
        evaluate.write_unsupervised_report(risk_threshold=args.risk_threshold)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s.")
    print("Outputs: data/risk_scores.csv, data/explained_clusters.json, metrics_report.md")
    if not has_ground_truth:
        print("Note: ran in UNSUPERVISED mode (no ground-truth labels) -- "
              "risk scores/clusters are available, but no precision/recall metrics.")

    if args.dashboard:
        print("\nLaunching dashboard at http://127.0.0.1:5050 ...")
        from src.dashboard.app import app
        app.run(debug=False, port=5050)


if __name__ == "__main__":
    sys.exit(main())
