"""
generate_data.py
=================
Section 5 of the project plan: base data + synthetic fraud injection.

This script is self-contained: it synthesizes a realistic "legitimate
e-commerce" transaction log in the shape of a Kaggle "Online Retail"
style dataset (timestamps, item categories, customer IDs, order values),
then injects controlled fraud patterns on top of it:

  - Address collision rings (shared addresses across 3-6 customers)
  - Bracketing (buy S/M/L, return most of them)
  - Wardrobing (buy -> return within an unusually short window,
    concentrated in formalwear/electronics)
  - Abnormal return velocity (customers with return rates far above
    the natural distribution)
  - A realistic noise floor: some legitimate customers also have
    high-but-innocent return rates, so a model that just thresholds
    "returns a lot" cannot separate fraud from noise.

If you have downloaded a real Kaggle "Online Retail" CSV, you can swap
it in as the base population -- see `load_kaggle_base()` below for the
expected column mapping. By default we generate a synthetic base
population so the pipeline runs end-to-end with zero external
downloads.

Outputs (data/):
  - clean_data.csv          the data your model is allowed to see
  - ground_truth_fraud.csv  customer_id -> fraud label (held out,
                             ONLY used later by evaluate.py)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = [
    "Formalwear", "Electronics", "Shoes", "Home & Kitchen",
    "Beauty", "Toys", "Sporting Goods", "Books", "Accessories",
]
RETURN_REASONS = [
    "Wrong size", "Changed my mind", "Item not as described",
    "Defective", "Arrived late", "Better price found",
]
SIZES = ["S", "M", "L"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _fake_address(i: int) -> str:
    street_num = np.random.randint(1, 9999)
    streets = ["Main St", "Oak Ave", "Elm St", "Park Rd", "Cedar Ln",
               "Maple Dr", "Sunset Blvd", "2nd St", "Highland Ave", "River Rd"]
    cities = [("Dallas", "TX"), ("Austin", "TX"), ("Denver", "CO"),
              ("Seattle", "WA"), ("Phoenix", "AZ"), ("Atlanta", "GA"),
              ("Columbus", "OH"), ("Miami", "FL")]
    street = np.random.choice(streets)
    city, state = cities[np.random.randint(0, len(cities))]
    zip_code = 10000 + np.random.randint(0, 89999)
    return f"{street_num} {street}, {city}, {state} {zip_code}"


def _fake_payment_fp() -> str:
    card_types = ["VISA", "MC", "AMEX", "DISC"]
    last4 = np.random.randint(1000, 9999)
    return f"{np.random.choice(card_types)}-{last4}"


def load_kaggle_base(csv_path: str) -> pd.DataFrame:
    """
    Optional: load a real Kaggle 'Online Retail' style CSV instead of the
    synthetic generator. Expected/renamed columns after loading:
      customer_id, item_id, category, order_value, purchase_date
    Adjust the column mapping below to match whatever Kaggle dataset you
    downloaded (e.g. 'Online Retail II', 'Brazilian E-Commerce', etc).
    """
    df = pd.read_csv(csv_path)
    # Example mapping for the classic "Online Retail" Kaggle dataset --
    # edit column names here to match your actual download.
    rename_map = {
        "CustomerID": "customer_id",
        "StockCode": "item_id",
        "Description": "category",
        "UnitPrice": "order_value",
        "InvoiceDate": "purchase_date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def generate_base_population(n_customers: int, n_transactions: int,
                              start_date: datetime, days_span: int) -> pd.DataFrame:
    """Section 5.1: realistic legitimate purchase behavior."""
    customer_ids = [f"C{100000 + i}" for i in range(n_customers)]
    addresses = {cid: _fake_address(i) for i, cid in enumerate(customer_ids)}
    payments = {cid: _fake_payment_fp() for cid in customer_ids}

    rows = []
    for t in range(n_transactions):
        cid = np.random.choice(customer_ids)
        category = np.random.choice(CATEGORIES)
        item_id = f"ITEM-{category[:3].upper()}-{np.random.randint(1, 500)}"
        order_value = round(float(np.random.gamma(3.0, 25.0)), 2)
        purchase_offset = int(np.random.randint(0, days_span))
        purchase_date = start_date + timedelta(days=purchase_offset,
                                                hours=int(np.random.randint(0, 24)))

        # Baseline natural return behavior (~12-18% of orders returned,
        # for ordinary reasons, with a delay spread over a couple weeks).
        returned = np.random.random() < np.random.uniform(0.12, 0.18)
        return_date, return_reason, days_to_return = None, None, None
        if returned:
            delay = int(np.random.randint(2, 21))
            return_date = purchase_date + timedelta(days=delay)
            return_reason = np.random.choice(RETURN_REASONS)
            days_to_return = delay

        rows.append(dict(
            transaction_id=f"T{t:07d}",
            customer_id=cid,
            item_id=item_id,
            category=category,
            size=rng.choice(SIZES) if category in ("Formalwear", "Shoes") else None,
            order_value=order_value,
            purchase_date=purchase_date.isoformat(),
            address_raw=addresses[cid],
            payment_fingerprint=payments[cid],
            returned=returned,
            return_date=return_date.isoformat() if return_date else None,
            return_reason=return_reason,
            days_to_return=days_to_return,
        ))

    df = pd.DataFrame(rows)
    df.attrs["addresses"] = addresses
    df.attrs["payments"] = payments
    return df


def inject_noise_floor(df: pd.DataFrame,
                        n_noisy_customers: int) -> pd.DataFrame:
    """
    Some legitimate customers get elevated-but-innocent return rates
    (e.g. they wear a wide shoe size and return often, or they buy a lot
    and are simply picky). These are NOT fraud, and exist specifically so
    the model has to discriminate rather than just threshold on volume.
    """
    customers = df["customer_id"].unique()
    noisy = np.random.choice(customers, size=min(n_noisy_customers, len(customers)),
                        replace=False)
    mask = df["customer_id"].isin(noisy)
    idx = df[mask].sample(frac=0.5, random_state=int(np.random.randint(0, 1e6))).index
    df.loc[idx, "returned"] = True
    df.loc[idx, "return_reason"] = "Wrong size"
    for i in idx:
        pdate = datetime.fromisoformat(df.loc[i, "purchase_date"])
        delay = int(np.random.randint(3, 18))
        df.loc[i, "return_date"] = (pdate + timedelta(days=delay)).isoformat()
        df.loc[i, "days_to_return"] = delay
    return df


def inject_address_rings(df: pd.DataFrame,
                          n_rings: int, addresses: dict) -> tuple[pd.DataFrame, set]:
    """Section 5.2: duplicate an address across 3-6 distinct customer_ids."""
    all_customers = list(addresses.keys())
    fraud_customers: set[str] = set()
    for r in range(n_rings):
        ring_size = int(np.random.randint(3, 7))
        ring_members = np.random.choice(all_customers, size=ring_size, replace=False)
        shared_address = _fake_address(90000 + r)
        for i, cid in enumerate(ring_members):
            # occasionally introduce a harmless string variant
            # ("St" vs "Street") to make normalization matter.
            addr = shared_address
            if np.random.random() < 0.3:
                addr = addr.replace(" St,", " Street,").replace(" Ave,", " Avenue,")
            df.loc[df["customer_id"] == cid, "address_raw"] = addr
            fraud_customers.add(cid)
    return df, fraud_customers


def inject_bracketing(df: pd.DataFrame,
                       customers: list, start_date: datetime,
                       days_span: int) -> pd.DataFrame:
    """
    Same customer purchases S/M/L of one item; returns 2 of 3 within days.
    """
    new_rows = []
    t_counter = 900000
    for cid in customers:
        item_id = f"ITEM-BRK-{np.random.randint(1, 200)}"
        category = "Formalwear"
        purchase_offset = int(np.random.randint(0, days_span))
        purchase_date = start_date + timedelta(days=purchase_offset)
        keep_size = np.random.choice(SIZES)
        for size in SIZES:
            is_kept = size == keep_size
            returned = not is_kept
            return_date, return_reason, days_to_return = None, None, None
            if returned:
                delay = int(np.random.randint(1, 5))
                return_date = purchase_date + timedelta(days=delay)
                return_reason = "Wrong size"
                days_to_return = delay
            new_rows.append(dict(
                transaction_id=f"T{t_counter:07d}",
                customer_id=cid,
                item_id=item_id,
                category=category,
                size=size,
                order_value=round(float(np.random.gamma(3.0, 30.0)), 2),
                purchase_date=purchase_date.isoformat(),
                address_raw=df.loc[df["customer_id"] == cid, "address_raw"].iloc[0],
                payment_fingerprint=df.loc[df["customer_id"] == cid, "payment_fingerprint"].iloc[0],
                returned=returned,
                return_date=return_date.isoformat() if return_date else None,
                return_reason=return_reason,
                days_to_return=days_to_return,
            ))
            t_counter += 1
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def inject_wardrobing(df: pd.DataFrame,
                       customers: list, start_date: datetime,
                       days_span: int) -> pd.DataFrame:
    """
    Purchase -> return within an unusually short window, concentrated in
    formalwear / electronics (buy for an event, return right after).
    """
    new_rows = []
    t_counter = 950000
    wardrobe_categories = ["Formalwear", "Electronics"]
    for cid in customers:
        n_events = int(np.random.randint(2, 5))
        for _ in range(n_events):
            category = np.random.choice(wardrobe_categories)
            purchase_offset = int(np.random.randint(0, days_span))
            purchase_date = start_date + timedelta(days=purchase_offset)
            delay = int(np.random.randint(1, 3))  # very short window
            return_date = purchase_date + timedelta(days=delay)
            new_rows.append(dict(
                transaction_id=f"T{t_counter:07d}",
                customer_id=cid,
                item_id=f"ITEM-{category[:3].upper()}-{np.random.randint(1, 500)}",
                category=category,
                size=np.random.choice(SIZES) if category == "Formalwear" else None,
                order_value=round(float(np.random.gamma(4.0, 40.0)), 2),
                purchase_date=purchase_date.isoformat(),
                address_raw=df.loc[df["customer_id"] == cid, "address_raw"].iloc[0],
                payment_fingerprint=df.loc[df["customer_id"] == cid, "payment_fingerprint"].iloc[0],
                returned=True,
                return_date=return_date.isoformat(),
                return_reason="Changed my mind",
                days_to_return=delay,
            ))
            t_counter += 1
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def inject_return_velocity(df: pd.DataFrame,
                            customers: list, start_date: datetime,
                            days_span: int) -> pd.DataFrame:
    """Customers with return rates far above the dataset's natural distribution."""
    new_rows = []
    t_counter = 970000
    for cid in customers:
        n_extra = int(np.random.randint(8, 15))
        for _ in range(n_extra):
            category = np.random.choice(CATEGORIES)
            purchase_offset = int(np.random.randint(0, days_span))
            purchase_date = start_date + timedelta(days=purchase_offset)
            delay = int(np.random.randint(1, 10))
            return_date = purchase_date + timedelta(days=delay)
            new_rows.append(dict(
                transaction_id=f"T{t_counter:07d}",
                customer_id=cid,
                item_id=f"ITEM-{category[:3].upper()}-{np.random.randint(1, 500)}",
                category=category,
                size=np.random.choice(SIZES) if category in ("Formalwear", "Shoes") else None,
                order_value=round(float(np.random.gamma(2.5, 20.0)), 2),
                purchase_date=purchase_date.isoformat(),
                address_raw=df.loc[df["customer_id"] == cid, "address_raw"].iloc[0],
                payment_fingerprint=df.loc[df["customer_id"] == cid, "payment_fingerprint"].iloc[0],
                returned=True,
                return_date=return_date.isoformat(),
                return_reason=np.random.choice(RETURN_REASONS),
                days_to_return=delay,
            ))
            t_counter += 1
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def run(n_customers=800, n_transactions=12000, n_rings=10,
        n_bracketing=25, n_wardrobing=20, n_velocity=15,
        n_noisy=40, seed=42, out_dir=DATA_DIR):
    _rng(seed)
    start_date = datetime(2024, 1, 1)
    days_span = 270

    df = generate_base_population(n_customers, n_transactions, start_date,
                                   days_span)
    addresses = df.attrs["addresses"]
    all_customers = list(addresses.keys())

    df = inject_noise_floor(df, n_noisy)

    df, ring_fraud = inject_address_rings(df, n_rings, addresses)

    remaining = [c for c in all_customers if c not in ring_fraud]
    bracket_customers = list(np.random.choice(remaining, size=n_bracketing, replace=False))
    remaining = [c for c in remaining if c not in bracket_customers]
    wardrobe_customers = list(np.random.choice(remaining, size=n_wardrobing, replace=False))
    remaining = [c for c in remaining if c not in wardrobe_customers]
    velocity_customers = list(np.random.choice(remaining, size=n_velocity, replace=False))

    # Give roughly half of the address-ring members bracketing/wardrobing
    # behavior too (coordinated rings), so shared-address alone isn't the
    # sole signal -- it has to combine with behavior, as the plan intends.
    ring_list = list(ring_fraud)
    ring_bracketers = ring_list[: max(1, len(ring_list) // 2)]

    df = inject_bracketing(df, bracket_customers + ring_bracketers,
                            start_date, days_span)
    df = inject_wardrobing(df, wardrobe_customers, start_date, days_span)
    df = inject_return_velocity(df, velocity_customers, start_date, days_span)

    fraud_customers = set(ring_fraud) | set(bracket_customers) | \
        set(wardrobe_customers) | set(velocity_customers)

    df = df.sort_values(["customer_id", "purchase_date"]).reset_index(drop=True)

    ground_truth = pd.DataFrame({
        "customer_id": all_customers,
        "ground_truth_fraud": [c in fraud_customers for c in all_customers],
    })
    fraud_type_map = {}
    for c in ring_fraud:
        fraud_type_map[c] = "address_ring"
    for c in bracket_customers + ring_bracketers:
        fraud_type_map[c] = fraud_type_map.get(c, "bracketing")
    for c in wardrobe_customers:
        fraud_type_map[c] = "wardrobing"
    for c in velocity_customers:
        fraud_type_map[c] = "return_velocity"
    ground_truth["fraud_type"] = ground_truth["customer_id"].map(fraud_type_map).fillna("none")

    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, "clean_data.csv")
    gt_path = os.path.join(out_dir, "ground_truth_fraud.csv")
    df.to_csv(clean_path, index=False)
    ground_truth.to_csv(gt_path, index=False)

    print(f"[generate_data] wrote {len(df)} transactions -> {clean_path}")
    print(f"[generate_data] wrote {len(ground_truth)} customer labels -> {gt_path}")
    print(f"[generate_data] injected fraud customers: {len(fraud_customers)} "
          f"({len(fraud_customers) / len(all_customers):.1%} of population)")
    return df, ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=800)
    parser.add_argument("--n-transactions", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n_customers=args.n_customers, n_transactions=args.n_transactions,
        seed=args.seed)
