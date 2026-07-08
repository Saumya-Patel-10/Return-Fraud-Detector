"""
dashboard/app.py
=================
Flask + pyvis dashboard (Section 10 of the plan). Shows ranked risk
clusters with plain-English explanations and an interactive network
graph for the selected cluster, so a non-technical reviewer can
immediately see which clusters are risky and why.

Run with:  python run_pipeline.py --dashboard-only
       or: python src/dashboard/app.py
"""
from __future__ import annotations

import json
import os
import pickle

import pandas as pd
from flask import Flask, render_template, jsonify, abort
from pyvis.network import Network

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR)


def _load_data():
    with open(os.path.join(DATA_DIR, "explained_clusters.json")) as f:
        clusters = json.load(f)
    risk_scores = pd.read_csv(os.path.join(DATA_DIR, "risk_scores.csv"))
    with open(os.path.join(DATA_DIR, "graph.gpickle"), "rb") as f:
        graph_data = pickle.load(f)
    summary_path = os.path.join(DATA_DIR, "metrics_summary.json")
    metrics = {"mode": "unknown"}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            metrics = json.load(f)
    return clusters, risk_scores, graph_data, metrics


@app.route("/")
def index():
    clusters, risk_scores, _, metrics = _load_data()
    n_flagged = sum(1 for c in clusters if c["risk_tier"] in ("High", "Medium"))
    return render_template("index.html", clusters=clusters,
                            n_customers=len(risk_scores),
                            n_flagged=n_flagged,
                            metrics=metrics)


@app.route("/api/clusters")
def api_clusters():
    clusters, _, _, _ = _load_data()
    return jsonify(clusters)


@app.route("/api/cluster/<int:cluster_id>/graph")
def cluster_graph(cluster_id: int):
    clusters, _, graph_data, _ = _load_data()
    cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
    if cluster is None:
        abort(404)

    G = graph_data["graph"]
    members = set(cluster["member_customer_ids"])

    # subgraph: cluster members + their directly-connected item/address/payment nodes
    keep_nodes = set(members)
    for m in members:
        if m in G:
            keep_nodes.update(G.neighbors(m))
    sub = G.subgraph(keep_nodes)

    net = Network(height="500px", width="100%", bgcolor="#111827",
                   font_color="#e5e7eb", directed=True)
    color_map = {
        "customer": "#f87171", "address": "#60a5fa",
        "payment": "#fbbf24", "item": "#34d399",
    }
    for node, data in sub.nodes(data=True):
        ntype = data.get("node_type", "unknown")
        is_flagged_member = node in members
        net.add_node(str(node), label=str(node)[:14],
                     color=color_map.get(ntype, "#9ca3af"),
                     size=25 if is_flagged_member else 12,
                     title=f"{ntype}: {node}")
    for u, v, data in sub.edges(data=True):
        net.add_edge(str(u), str(v), title=data.get("edge_type", ""))

    net.set_options("""
    {
      "physics": {"stabilization": {"iterations": 150}},
      "edges": {"color": {"color": "#4b5563"}, "smooth": false}
    }
    """)
    html = net.generate_html(notebook=False)
    return html


if __name__ == "__main__":
    app.run(debug=True, port=5050)
