"""
Lightweight sanity tests -- not exhaustive, but enough to catch a broken
pipeline before a demo. Run with: pytest tests/ (or: python -m pytest tests/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.geocode import address_hash
from src.generate_data import run as generate_run
from src.build_graph import build_graph, customer_projection, detect_communities
from src.features import build_features
from src.detect import score_nodes, combine_cluster_risk, FEATURE_COLS
from src.data_loader import load_and_validate, load_labels, SchemaError, REQUIRED_COLUMNS


def test_address_hash_normalizes_variants():
    a = address_hash("123 Main St, Dallas, TX 75201")
    b = address_hash("123 Main Street, Dallas, TX 75201")
    assert a == b


def test_generate_data_has_ground_truth_and_fraud(tmp_path):
    df, gt = generate_run(n_customers=100, n_transactions=1200,
                           n_rings=3, n_bracketing=5, n_wardrobing=4,
                           n_velocity=3, n_noisy=8, seed=1, out_dir=str(tmp_path))
    assert "ground_truth_fraud" not in df.columns
    assert gt["ground_truth_fraud"].sum() > 0
    assert set(gt["customer_id"]) == set(df["customer_id"].unique())


def test_graph_build_and_communities(tmp_path):
    df, gt = generate_run(n_customers=100, n_transactions=1200,
                           n_rings=3, n_bracketing=5, n_wardrobing=4,
                           n_velocity=3, n_noisy=8, seed=1, out_dir=str(tmp_path))
    G = build_graph(df)
    assert G.number_of_nodes() > 0
    H = customer_projection(G)
    membership = detect_communities(H)
    assert len(membership) == H.number_of_nodes()
    # at least one non-singleton cluster should exist given injected rings
    from collections import Counter
    sizes = Counter(membership.values())
    assert max(sizes.values()) >= 3


def test_features_and_scoring(tmp_path):
    df, gt = generate_run(n_customers=100, n_transactions=1200,
                           n_rings=3, n_bracketing=5, n_wardrobing=4,
                           n_velocity=3, n_noisy=8, seed=1, out_dir=str(tmp_path))
    feat = build_features(df)
    for col in FEATURE_COLS:
        assert col in feat.columns
    scored = score_nodes(feat)
    assert scored["node_anomaly_score"].between(0, 1).all()

    G = build_graph(df)
    H = customer_projection(G)
    membership = detect_communities(H)
    combined = combine_cluster_risk(scored, membership)
    assert "cluster_risk_score" in combined.columns
    assert combined["cluster_risk_score"].between(0, 1.01).all()


def test_data_loader_accepts_real_style_csv(tmp_path):
    """A real CSV with the required columns, plus aliased headers, should
    load cleanly and end up in the exact schema the rest of the pipeline
    expects."""
    import pandas as pd
    real = pd.DataFrame({
        "CustomerID": ["A1", "A1", "A2"],
        "ProductID": ["I1", "I2", "I1"],
        "category": ["Electronics", "Shoes", "Electronics"],
        "Amount": [100.0, 50.0, 75.0],
        "OrderDate": ["2025-01-01", "2025-01-05", "2025-01-02"],
        "ShippingAddress": ["1 Main St", "1 Main St", "2 Oak Ave"],
        "PaymentMethod": ["VISA-1111", "VISA-1111", "MC-2222"],
        "IsReturned": [True, False, False],
        "ReturnDate": ["2025-01-03", None, None],
    })
    csv_path = tmp_path / "real.csv"
    real.to_csv(csv_path, index=False)

    df = load_and_validate(str(csv_path))
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert df["returned"].dtype == bool
    assert df.loc[df["customer_id"] == "A1", "days_to_return"].iloc[0] == 2


def test_data_loader_raises_clear_error_on_missing_columns(tmp_path):
    import pandas as pd
    bad = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    csv_path = tmp_path / "bad.csv"
    bad.to_csv(csv_path, index=False)

    try:
        load_and_validate(str(csv_path))
        assert False, "expected SchemaError"
    except SchemaError as e:
        assert "customer_id" in str(e)


def test_data_loader_labels_schema(tmp_path):
    import pandas as pd
    labels = pd.DataFrame({"customer_id": ["A1", "A2"], "ground_truth_fraud": [True, False]})
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    gt = load_labels(str(path))
    assert gt["ground_truth_fraud"].dtype == bool
    assert set(gt["customer_id"]) == {"A1", "A2"}
