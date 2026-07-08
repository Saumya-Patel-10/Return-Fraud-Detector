# Return-Fraud & Wardrobing Detector

Graph-based anomaly detection for e-commerce return fraud: wardrobing (buy,
use, return), empty-box returns, and bracketing (buying multiple variants
and returning most of them). Most production systems still rely on simple
rule thresholds (e.g. "flag if returns exceed 5 in 30 days"), which fraud
rings evade by spacing out returns or spreading activity across accounts.
The real signal lives in *relationships* between accounts — shared
addresses, shared payment fingerprints, and coordinated timing — which is
a graph problem, not a per-transaction problem.

This repo builds a full pipeline: raw transaction/return data → graph
construction → two-layer anomaly scoring → ranked, explainable risk
clusters → a dashboard a non-technical reviewer can read in seconds.

## Problem statement

Retailers lose well over $100B/year to return fraud. Rule-based systems
(`returns > 5 in 30 days`) are easy for coordinated fraud rings to evade.
This project detects rings by modeling customers, addresses, payment
fingerprints, and items as a graph, then combining **structural**
signals (who's connected to whom) with **behavioral** signals (per-account
return patterns) into a single, explainable risk score per cluster.

## Architecture

```
                    ┌─────────────────────┐
                    │  generate_data.py    │  synthetic legit transactions
                    │  (Section 5)         │  + injected fraud patterns
                    └──────────┬───────────┘
                               │ clean_data.csv (model sees this)
                               │ ground_truth_fraud.csv (held out, eval-only)
                               ▼
        ┌──────────────────────────────────────────┐
        │              build_graph.py                │
        │  nodes: customer / address / payment / item │
        │  edges: purchased, returned,                │
        │         shares_address_with, shares_payment  │
        │  Layer 1: Louvain community detection        │
        └───────────────────┬────────────────────────┘
                             │ graph.gpickle
              ┌──────────────┴───────────────┐
              ▼                               ▼
     ┌────────────────┐             ┌──────────────────┐
     │  baseline.py     │             │   features.py      │
     │  rule-based       │             │  Section 7 features │
     │  (Section 9)       │             └─────────┬──────────┘
     └────────┬─────────┘                         │
              │                                    ▼
              │                          ┌───────────────────┐
              │                          │    detect.py        │
              │                          │ Layer 2: Isolation   │
              │                          │ Forest node score +  │
              │                          │ cluster risk combine │
              │                          └─────────┬─────────┘
              │                                    │ risk_scores.csv
              │                                    ▼
              │                          ┌───────────────────┐
              │                          │   explain.py         │
              │                          │ plain-English reasons │
              │                          └─────────┬─────────┘
              │                                    │ explained_clusters.json
              ▼                                    ▼
        ┌──────────────────────────────────────────┐
        │               evaluate.py                    │
        │  precision / recall / FPR vs. baseline,      │
        │  against held-out ground_truth_fraud          │
        └──────────────────────────────────────────┘
                             │
                             ▼
                     metrics_report.md
                             │
                             ▼
                ┌─────────────────────────┐
                │  dashboard/app.py          │
                │  Flask + pyvis network viz  │
                └─────────────────────────┘
```

## Two-layer detection design

- **Layer 1 — graph level:** Louvain community detection (via NetworkX /
  `python-louvain`) surfaces clusters of accounts that are structurally
  connected in ways unrelated shoppers wouldn't be (shared address,
  shared payment fingerprint).
- **Layer 2 — node level:** per-customer features (return velocity,
  category return ratio, purchase-to-return timing, address/payment
  fan-in) scored with an unsupervised **Isolation Forest** — appropriate
  here because we deliberately never train on the synthetic fraud labels.
- **Combination:** `cluster_risk_score = 0.6·mean(node anomaly) +
  0.25·size_factor + 0.15·density_factor`. This combination — not either
  layer alone — is the core technical contribution.

## Explainability

Every flagged cluster gets an auto-generated plain-English reason, e.g.:

> *"6 accounts: 6 accounts share one address; returns arriving within
> ~4 days of purchase."*

Generated by ranking each feature's z-score (cluster mean vs. global
population) and templating the top drivers — no LLM required. Detection
without explanation isn't deployable: analysts need to know *why* an
account was flagged before acting on a real customer.

## Setup

```bash
git clone <this-repo> return-fraud-detector
cd return-fraud-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional: add a Google Geocoding API key
```

The pipeline runs **with zero API keys**: if `GOOGLE_API_KEY` is unset in
`.env`, address normalization falls back to an offline regex normalizer
(see `src/geocode.py`). Add a real key later for production-grade address
resolution — see Section 4 setup steps below.

### Google Geocoding API (optional, Section 4)

1. Create a project at [console.cloud.google.com](https://console.cloud.google.com).
2. **APIs & Services → Library** → search "Geocoding API" → Enable.
3. **APIs & Services → Credentials → Create Credentials → API Key.**
4. Restrict the key to the Geocoding API only (and by IP if possible).
5. Put it in `.env` as `GOOGLE_API_KEY=your_key_here`. `.env` is already
   git-ignored — confirm with `git log -p | grep API_KEY` before ever
   making the repo public.

## Run

One command, end to end:

```bash
python run_pipeline.py
```

This generates synthetic data, builds the graph, runs the rule-based
baseline, engineers features, scores with Isolation Forest, generates
explanations, and writes `metrics_report.md`.

Useful flags:

```bash
python run_pipeline.py --skip-data          # reuse existing data/clean_data.csv
python run_pipeline.py --n-customers 2000   # larger synthetic population
python run_pipeline.py --dashboard          # also launch the dashboard after
```

Launch the dashboard on its own (after a pipeline run):

```bash
python src/dashboard/app.py
# open http://127.0.0.1:5050
```

Run tests:

```bash
python -m pytest tests/ -q
```

## Outputs

| File | Produced by | Contents |
|---|---|---|
| `data/clean_data.csv` | `generate_data.py` | transactions the model is allowed to see |
| `data/ground_truth_fraud.csv` | `generate_data.py` | held-out labels, eval-only |
| `data/graph.gpickle` | `build_graph.py` | pickled NetworkX graph + Louvain communities |
| `data/baseline_metrics.json` | `baseline.py` | rule-based baseline precision/recall/FPR |
| `data/features.csv` | `features.py` | per-customer feature table |
| `data/risk_scores.csv` | `detect.py` | per-customer node + cluster risk scores |
| `data/explained_clusters.json` | `explain.py` | ranked clusters with plain-English reasons |
| `metrics_report.md` | `evaluate.py` | graph model vs. baseline (or risk-tier summary in unsupervised mode), resume-ready |
| `data/metrics_summary.json` | `evaluate.py` | same numbers as `metrics_report.md`, structured for the dashboard |

## Metrics (regenerate with `python run_pipeline.py`)

See `metrics_report.md` for the latest run. Numbers will vary slightly
between runs since the synthetic fraud injection is stochastic — that's
intentional (Section 5.2's noise floor requirement), and is why the
report is regenerated rather than hardcoded here.

## Running on a real dataset (e.g. one handed to you in an interview)

By default the pipeline runs on synthetic data with injected fraud
labels, so it works with zero downloads and has known ground truth to
score against. To point the exact same pipeline at a real CSV instead:

```bash
# Real data, no known fraud labels -- risk scores only, no precision/recall
python run_pipeline.py --input path/to/real_data.csv --dashboard

# Real data + known fraud labels -- adds precision/recall vs. baseline
python run_pipeline.py --input path/to/real_data.csv --labels path/to/labels.csv --dashboard
```

**Required columns** in `--input`: `customer_id, item_id, category,
order_value, purchase_date, address_raw, payment_fingerprint, returned`.
Optional: `size, return_date, return_reason, days_to_return` (filled in /
computed automatically if missing). See
`data/sample_real_data_template.csv` for a working example, and
`src/data_loader.py` for full details.

If your file uses different header names (e.g. `CustomerID`,
`OrderDate`, `Amount`), either rename them in the CSV first, or add a
mapping to `COLUMN_ALIASES` at the top of `src/data_loader.py` — a few
common ones (Kaggle "Online Retail" style headers, etc.) are already
included.

**Optional labels file** (`--labels`) needs just two columns:
`customer_id, ground_truth_fraud` (True/False). Without it, the pipeline
still runs completely — you get ranked risk clusters and explanations,
just no precision/recall numbers, since there's nothing to score against.
`src/data_loader.py` validates the schema up front and gives a specific,
actionable error (missing columns, unparseable dates, etc.) rather than
failing deep inside the pipeline.

## Neo4j (optional, resume-relevant stretch goal)

The current graph layer uses NetworkX for fast iteration. To also stand
up Neo4j Community Edition for Cypher-query exploration:

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password neo4j:community
```

Then export `build_graph.py`'s node/edge lists into Cypher `CREATE`
statements (customer, address, payment, item nodes; purchased / returned
/ shares_address_with / shares_payment_with edges) to explore rings
interactively at `http://localhost:7474`.

## Project layout

```
return-fraud-detector/
├── run_pipeline.py            # single entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── generate_data.py       # Section 5: base data + fraud injection
│   ├── geocode.py              # Section 4: address normalization
│   ├── build_graph.py          # Section 6: graph + Louvain (Layer 1)
│   ├── baseline.py             # Section 9: rule-based baseline
│   ├── features.py             # Section 7: feature engineering
│   ├── detect.py               # Section 6.3: Isolation Forest (Layer 2) + combine
│   ├── explain.py              # Section 8: explainability layer
│   ├── evaluate.py             # Section 9: metrics vs. baseline
│   └── dashboard/
│       ├── app.py              # Flask + pyvis dashboard
│       ├── templates/index.html
│       └── static/style.css
├── tests/test_pipeline.py
└── data/                       # generated at runtime, git-ignored
```

## Resume bullet template

Auto-filled with real numbers at the bottom of `metrics_report.md` after
each run:

> Built a graph-based return-fraud detection pipeline (NetworkX,
> scikit-learn, Google Geocoding API) combining community detection and
> anomaly scoring, achieving **[X]%** precision and **[Y]%** recall on
> injected fraud patterns — a **[Z]%** change in false positive rate
> versus a rule-based baseline.

## 90-second interview story

1. **Problem:** return fraud rings evade per-transaction rule thresholds
   by spacing out activity or spreading it across accounts.
2. **Why graphs:** the signal is relational — shared addresses, payment
   fingerprints, coordinated timing — not visible to a single-row model.
3. **Design:** two layers — Louvain community structure + Isolation
   Forest node anomaly scores — combined into one explainable cluster
   risk score, because structure alone or behavior alone each miss
   cases the other catches.
4. **Result:** beat the rule-based baseline on precision and recall at
   an equal or lower false-positive rate (see `metrics_report.md`),
   with every flagged cluster shipping a plain-English reason so an
   analyst can act on it without guessing.

## Notes on privacy

Raw addresses and payment numbers are never stored as node identifiers.
Addresses are normalized and SHA-256 hashed (`address_hash`) before
touching the graph; payment fingerprints store only card type + last 4
digits — never full card numbers. This mirrors how a real risk team
handles PII, and is called out explicitly because an interviewer will
ask.
