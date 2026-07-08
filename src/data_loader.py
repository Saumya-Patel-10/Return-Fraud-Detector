"""
data_loader.py
================
Lets the pipeline run on a REAL transactions CSV instead of the synthetic
generator, so the exact same pipeline (graph build -> features ->
detection -> explanation -> dashboard) works whether you're demoing with
synthetic data or a dataset an interviewer hands you on the spot.

Expected CSV schema (same shape generate_data.py produces):

  REQUIRED columns
    customer_id            any string/int id, e.g. "C1001"
    item_id                any string/int id, e.g. "SKU-4471"
    category                e.g. "Electronics"
    order_value             numeric, order amount
    purchase_date            parseable date/datetime
    address_raw              free-text shipping address (used to detect
                              multiple accounts sharing one address)
    payment_fingerprint      any stable per-payment-method id (e.g. last 4
                              digits + card type -- never a full card
                              number)
    returned                 boolean-ish (True/False, 1/0, yes/no)

  OPTIONAL columns (filled in with safe defaults if missing)
    size                      item size, only meaningful for apparel/shoes
    return_date               parseable date/datetime, required if
                               'returned' is True for good features
    return_reason             free text
    days_to_return            numeric; auto-computed from purchase_date /
                               return_date if omitted

  OPTIONAL ground-truth labels file (--labels), used ONLY for computing
  precision/recall on the dashboard -- never fed into the model:
    customer_id, ground_truth_fraud (True/False)

If your real dataset uses different column names, edit COLUMN_ALIASES
below to map them, or rename columns in the CSV before loading.
"""
from __future__ import annotations

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

REQUIRED_COLUMNS = [
    "customer_id", "item_id", "category", "order_value",
    "purchase_date", "address_raw", "payment_fingerprint", "returned",
]
OPTIONAL_COLUMNS = ["size", "return_date", "return_reason", "days_to_return"]

# If your real-world file uses different header names, add mappings here,
# e.g. "CustomerID": "customer_id". Left-hand side = your file's header.
COLUMN_ALIASES = {
    "CustomerID": "customer_id", "customerid": "customer_id",
    "StockCode": "item_id", "ProductID": "item_id", "SKU": "item_id",
    "Category": "category", "ProductCategory": "category",
    "UnitPrice": "order_value", "Amount": "order_value", "Price": "order_value",
    "InvoiceDate": "purchase_date", "OrderDate": "purchase_date", "Date": "purchase_date",
    "Address": "address_raw", "ShippingAddress": "address_raw",
    "PaymentMethod": "payment_fingerprint", "CardFingerprint": "payment_fingerprint",
    "Returned": "returned", "IsReturned": "returned", "return_flag": "returned",
    "ReturnDate": "return_date", "ReturnReason": "return_reason",
    "Size": "size",
}

TRUTHY = {"true", "1", "1.0", "yes", "y", "t"}


class SchemaError(ValueError):
    """Raised when the input CSV doesn't have what the pipeline needs."""


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(TRUTHY)


def load_and_validate(input_path: str) -> pd.DataFrame:
    """
    Read a real transactions CSV, apply column aliasing, validate the
    schema, coerce types, and return a DataFrame in the exact shape the
    rest of the pipeline (build_graph.py, features.py, baseline.py)
    expects. Raises SchemaError with a clear, actionable message if the
    file can't be used as-is.
    """
    if not os.path.exists(input_path):
        raise SchemaError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            "Input CSV is missing required column(s): " + ", ".join(missing) +
            f"\nColumns found: {list(df.columns)}"
            "\nEither rename the columns in your CSV to match, or add an "
            "entry to COLUMN_ALIASES in src/data_loader.py mapping your "
            "header name -> the expected name. Required columns: " +
            ", ".join(REQUIRED_COLUMNS)
        )

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # --- type coercion, with row-level error reporting ---
    df["customer_id"] = df["customer_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    bad_values = pd.to_numeric(df["order_value"], errors="coerce")
    n_bad = bad_values.isna().sum() - df["order_value"].isna().sum()
    if n_bad > 0:
        print(f"[data_loader] WARNING: {n_bad} row(s) had a non-numeric "
              f"order_value and were set to 0.0")
    df["order_value"] = bad_values.fillna(0.0)

    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    n_bad_dates = df["purchase_date"].isna().sum()
    if n_bad_dates > 0:
        print(f"[data_loader] WARNING: {n_bad_dates} row(s) had an "
              f"unparseable purchase_date and were dropped.")
        df = df[df["purchase_date"].notna()]
    df["purchase_date"] = df["purchase_date"].apply(lambda d: d.isoformat())

    df["returned"] = _coerce_bool(df["returned"])

    df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce")
    has_return_date = df["return_date"].notna()
    df.loc[has_return_date, "return_date"] = df.loc[has_return_date, "return_date"].apply(lambda d: d.isoformat())
    df["return_date"] = df["return_date"].where(has_return_date, None)

    # auto-compute days_to_return where possible if not supplied
    need_days = df["days_to_return"].isna() & df["returned"] & df["return_date"].notna()
    if need_days.any():
        pdates = pd.to_datetime(df.loc[need_days, "purchase_date"])
        rdates = pd.to_datetime(df.loc[need_days, "return_date"])
        df.loc[need_days, "days_to_return"] = (rdates - pdates).dt.days

    df["address_raw"] = df["address_raw"].astype(str)
    df["payment_fingerprint"] = df["payment_fingerprint"].astype(str)

    n_customers = df["customer_id"].nunique()
    if n_customers < 5:
        print(f"[data_loader] WARNING: only {n_customers} distinct customers "
              f"found -- graph/community features will be very sparse.")

    ordered = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    df = df[ordered]
    return df


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load an optional ground-truth labels CSV: customer_id, ground_truth_fraud."""
    if not os.path.exists(labels_path):
        raise SchemaError(f"Labels file not found: {labels_path}")
    gt = pd.read_csv(labels_path)
    gt = gt.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in gt.columns})
    if "customer_id" not in gt.columns or "ground_truth_fraud" not in gt.columns:
        raise SchemaError(
            "Labels CSV must have columns: customer_id, ground_truth_fraud. "
            f"Columns found: {list(gt.columns)}"
        )
    gt["customer_id"] = gt["customer_id"].astype(str)
    gt["ground_truth_fraud"] = _coerce_bool(gt["ground_truth_fraud"])
    return gt[["customer_id", "ground_truth_fraud"]]


def run(input_path: str, labels_path: str | None = None, out_dir: str = DATA_DIR) -> bool:
    """
    Validate + load a real dataset, write it to data/clean_data.csv (same
    place generate_data.py writes to, so every downstream step is
    unaffected). Returns True if ground-truth labels were also loaded
    (supervised evaluation mode), False otherwise (unsupervised /
    risk-scores-only mode).
    """
    df = load_and_validate(input_path)
    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, "clean_data.csv")
    df.to_csv(clean_path, index=False)
    print(f"[data_loader] validated and loaded {len(df)} transactions for "
          f"{df['customer_id'].nunique()} customers -> {clean_path}")

    gt_path = os.path.join(out_dir, "ground_truth_fraud.csv")
    if labels_path:
        gt = load_labels(labels_path)
        gt.to_csv(gt_path, index=False)
        print(f"[data_loader] loaded {len(gt)} ground-truth labels -> {gt_path}")
        return True
    else:
        # No labels for real data -- remove any stale ground-truth file
        # from a previous synthetic run so evaluate.py doesn't score
        # this real data against fake old labels.
        if os.path.exists(gt_path):
            os.remove(gt_path)
        print("[data_loader] no --labels file provided -- running in "
              "unsupervised mode (risk scores only, no precision/recall).")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("--labels", default=None)
    args = parser.parse_args()
    run(args.input_path, args.labels)
