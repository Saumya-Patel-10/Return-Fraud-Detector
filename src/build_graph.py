"""
build_graph.py
===============
Section 6 of the project plan: graph architecture and Layer 1 (graph-level)
detection.

Nodes:
  - customer_id
  - address_hash          (from geocode.normalize_address / hashing)
  - payment_fingerprint   (last 4 + card type only)
  - item_id / category

Edges:
  - purchased            customer -> item, weighted by frequency/value
  - returned              customer -> item, with return_reason / days_to_return
  - shares_address_with   customer <-> customer (derived)
  - shares_payment_with   customer <-> customer (derived)

Layer 1: Louvain community detection surfaces clusters of accounts that
are structurally connected in ways unrelated shoppers wouldn't be.
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities

from src.geocode import address_hash

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    df = df.copy()
    df["address_hash"] = df["address_raw"].apply(address_hash)

    # --- nodes ---
    for cid in df["customer_id"].unique():
        G.add_node(cid, node_type="customer")
    for ah in df["address_hash"].unique():
        G.add_node(ah, node_type="address")
    for pf in df["payment_fingerprint"].unique():
        G.add_node(pf, node_type="payment")
    for item_id, category in df[["item_id", "category"]].drop_duplicates().values:
        G.add_node(item_id, node_type="item", category=category)

    # --- purchased / returned edges + customer-address/payment links ---
    for row in df.itertuples(index=False):
        G.add_edge(row.customer_id, row.item_id, key="purchased",
                    edge_type="purchased", value=row.order_value,
                    date=row.purchase_date)
        if row.returned:
            G.add_edge(row.customer_id, row.item_id, key="returned",
                        edge_type="returned", return_reason=row.return_reason,
                        days_to_return=row.days_to_return,
                        date=row.return_date)
        G.add_edge(row.customer_id, row.address_hash, key="has_address",
                    edge_type="has_address")
        G.add_edge(row.customer_id, row.payment_fingerprint, key="has_payment",
                    edge_type="has_payment")

    # --- derived customer<->customer edges ---
    _add_shared_edges(G, df, "address_hash", "shares_address_with")
    _add_shared_edges(G, df, "payment_fingerprint", "shares_payment_with")

    return G


def _add_shared_edges(G: nx.MultiDiGraph, df: pd.DataFrame, group_col: str,
                       edge_type: str) -> None:
    groups: dict[str, set] = defaultdict(set)
    for cid, val in df[["customer_id", group_col]].drop_duplicates().values:
        groups[val].add(cid)
    for members in groups.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            G.add_edge(a, b, key=edge_type, edge_type=edge_type)
            G.add_edge(b, a, key=edge_type, edge_type=edge_type)


def customer_projection(G: nx.MultiDiGraph) -> nx.Graph:
    """
    Undirected customer-only graph used for community detection: an edge
    exists between two customers if they share an address or a payment
    fingerprint (the relational fraud-ring signal from Section 1).
    """
    H = nx.Graph()
    customers = [n for n, d in G.nodes(data=True) if d.get("node_type") == "customer"]
    H.add_nodes_from(customers)
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") in ("shares_address_with", "shares_payment_with"):
            if H.has_edge(u, v):
                H[u][v]["weight"] += 1
            else:
                H.add_edge(u, v, weight=1)
    return H


def detect_communities(H: nx.Graph, seed: int = 42) -> dict[str, int]:
    """Louvain community detection (Section 6.3, Layer 1)."""
    if H.number_of_edges() == 0:
        return {n: i for i, n in enumerate(H.nodes())}
    communities = louvain_communities(H, weight="weight", seed=seed)
    membership = {}
    for cluster_id, members in enumerate(communities):
        for m in members:
            membership[m] = cluster_id
    # customers with no shared-address/payment edges get their own
    # singleton cluster id, offset past the real clusters
    next_id = len(communities)
    for n in H.nodes():
        if n not in membership:
            membership[n] = next_id
            next_id += 1
    return membership


def run(clean_data_path: str = None, out_path: str = None):
    clean_data_path = clean_data_path or os.path.join(DATA_DIR, "clean_data.csv")
    out_path = out_path or os.path.join(DATA_DIR, "graph.gpickle")

    df = pd.read_csv(clean_data_path)
    G = build_graph(df)
    H = customer_projection(G)
    membership = detect_communities(H)

    nx.set_node_attributes(G, {c: membership.get(c) for c in membership}, "community")

    with open(out_path, "wb") as f:
        pickle.dump({"graph": G, "customer_projection": H, "community": membership}, f)

    n_clusters = len(set(membership.values()))
    print(f"[build_graph] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[build_graph] customer projection: {H.number_of_nodes()} nodes, "
          f"{H.number_of_edges()} edges")
    print(f"[build_graph] Louvain found {n_clusters} communities -> {out_path}")
    return G, H, membership


if __name__ == "__main__":
    run()
