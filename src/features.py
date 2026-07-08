"""
features.py
============
Section 7 of the project plan: per-customer feature engineering that
feeds Layer 2 (node-level) anomaly detection.

Features:
  - return velocity: returns per 30/60/90-day window
  - return-to-purchase ratio, broken out by category (bracketing signal)
  - time-between-purchase-and-return (short windows -> wardrobing)
  - address fan-in: distinct customer_ids linked to one address_hash
  - payment fan-in: distinct customer_ids linked to one payment_fingerprint
  - cluster size and density (from Louvain output, added later by detect.py)
"""
from __future__ import annotations

import os

import pandas as pd

from src.geocode import address_hash

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _return_velocity(df: pd.DataFrame, window_days: int) -> pd.Series:
    returns = df[df["returned"]].copy()
    returns["purchase_date"] = pd.to_datetime(returns["purchase_date"])
    counts = {}
    for cid, g in returns.groupby("customer_id"):
        dates = g["purchase_date"].sort_values().reset_index(drop=True)
        best = 0
        for i in range(len(dates)):
            window = dates[(dates >= dates[i]) &
                            (dates <= dates[i] + pd.Timedelta(days=window_days))]
            best = max(best, len(window))
        counts[cid] = best
    return pd.Series(counts, name=f"return_velocity_{window_days}d")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["address_hash"] = df["address_raw"].apply(address_hash)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])

    all_customers = df["customer_id"].unique()
    feat = pd.DataFrame(index=all_customers)
    feat.index.name = "customer_id"

    # --- purchase / return counts ---
    feat["n_purchases"] = df.groupby("customer_id").size()
    feat["n_returns"] = df.groupby("customer_id")["returned"].sum()
    feat["return_ratio"] = (feat["n_returns"] / feat["n_purchases"]).fillna(0)

    # --- return velocity windows ---
    for w in (30, 60, 90):
        feat[f"return_velocity_{w}d"] = _return_velocity(df, w)
    feat = feat.fillna(0)

    # --- category-level return ratio (bracketing signal) ---
    cat_ratio = (
        df.groupby(["customer_id", "category"])["returned"]
        .agg(["sum", "count"])
        .reset_index()
    )
    cat_ratio["ratio"] = cat_ratio["sum"] / cat_ratio["count"]
    max_cat_ratio = cat_ratio.groupby("customer_id")["ratio"].max()
    feat["max_category_return_ratio"] = max_cat_ratio
    feat["max_category_return_ratio"] = feat["max_category_return_ratio"].fillna(0)

    # --- time between purchase and return (wardrobing signal) ---
    days = df.loc[df["returned"], ["customer_id", "days_to_return"]]
    feat["avg_days_to_return"] = days.groupby("customer_id")["days_to_return"].mean()
    feat["min_days_to_return"] = days.groupby("customer_id")["days_to_return"].min()
    feat["avg_days_to_return"] = feat["avg_days_to_return"].fillna(feat["avg_days_to_return"].max())
    feat["min_days_to_return"] = feat["min_days_to_return"].fillna(feat["min_days_to_return"].max())

    # --- address / payment fan-in ---
    addr_fanin = df.groupby("address_hash")["customer_id"].nunique()
    pay_fanin = df.groupby("payment_fingerprint")["customer_id"].nunique()
    cust_addr = df[["customer_id", "address_hash"]].drop_duplicates("customer_id").set_index("customer_id")
    cust_pay = df[["customer_id", "payment_fingerprint"]].drop_duplicates("customer_id").set_index("customer_id")
    feat["address_fanin"] = cust_addr["address_hash"].map(addr_fanin)
    feat["payment_fanin"] = cust_pay["payment_fingerprint"].map(pay_fanin)
    feat[["address_fanin", "payment_fanin"]] = feat[["address_fanin", "payment_fanin"]].fillna(1)

    feat = feat.reset_index()
    return feat


def run(clean_data_path: str = None, out_path: str = None) -> pd.DataFrame:
    clean_data_path = clean_data_path or os.path.join(DATA_DIR, "clean_data.csv")
    out_path = out_path or os.path.join(DATA_DIR, "features.csv")

    df = pd.read_csv(clean_data_path)
    feat = build_features(df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feat.to_csv(out_path, index=False)
    print(f"[features] built {feat.shape[1] - 1} features for {len(feat)} customers -> {out_path}")
    return feat


if __name__ == "__main__":
    run()
