"""EDA: why LMC succeeds on HydroGraphNet (KNN, pure 2D) but fails on
UrbanFlood (physical mesh, 2D+1D coupling).

Reads raw mesh/edge data directly (bypasses the dataset classes — HG's was
removed when the repo refocused on UrbanFlood) and the existing training
logs. Outputs CSVs, PNGs, and the seed data for eda_lmc_vs_dataset_report.md.

Run from project root or this file's directory:
    python eda_lmc_vs_dataset.py
Outputs go to ./eda_results/.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

HG_DATA_DIR = "/home/woody/iwi5/iwi5416h/hydrographnet/data"
HG_TRAIN_LOG = "/home/woody/iwi5/iwi5416h/hydrographnet/outputs_lc_v7_run1_rep1/train.log"
HG_KNN_K = 4
HG_PREFIX = "M80"
HG_USED_NODE_COUNT = 4787  # M80_I.txt / M80_CU.txt line count (trimmed active set)

UF_DATA_DIR = "/home/woody/iwi5/iwi5416h/urbanflood/data"
UF_LMC_TRAIN_LOG = "/home/woody/iwi5/iwi5416h/urbanflood/outputs_lc_gt_conn_1d/train.log"
UF_MODEL_NAMES = ["Model_1", "Model_2"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_results")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# HydroGraphNet structural metrics (KNN graph reconstructed from coords)
# -----------------------------------------------------------------------
def load_hg_coords() -> np.ndarray:
    """Active-set HG coordinates: first HG_USED_NODE_COUNT of M80_CA/CE."""
    xs = np.loadtxt(os.path.join(HG_DATA_DIR, f"{HG_PREFIX}_CA.txt"))
    ys = np.loadtxt(os.path.join(HG_DATA_DIR, f"{HG_PREFIX}_CE.txt"))
    coords = np.column_stack([xs, ys])[:HG_USED_NODE_COUNT]
    return coords


def hg_knn_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build KNN k=4 edges (excluding self). Returns (edge_index [2,E], lengths [E])."""
    nn = NearestNeighbors(n_neighbors=HG_KNN_K + 1).fit(coords)
    distances, indices = nn.kneighbors(coords)
    # drop self-edge (col 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    n_nodes = coords.shape[0]
    src = np.repeat(np.arange(n_nodes), HG_KNN_K)
    dst = indices.flatten()
    lengths = distances.flatten()
    # Symmetrise (KNN-from-A may not be KNN-from-B). Dedup undirected pairs.
    pair_min = np.minimum(src, dst)
    pair_max = np.maximum(src, dst)
    undirected = np.unique(np.column_stack([pair_min, pair_max]), axis=0)
    # Recompute lengths from coords for dedup'd undirected edges.
    u_lengths = np.linalg.norm(coords[undirected[:, 0]] - coords[undirected[:, 1]], axis=1)
    edge_index_directed = np.column_stack([src, dst])
    return edge_index_directed, lengths, undirected, u_lengths


def hg_metrics() -> dict:
    coords = load_hg_coords()
    edge_idx_dir, lengths_dir, edge_idx_und, lengths_und = hg_knn_edges(coords)

    # degree from the undirected (deduplicated) graph
    degrees = np.zeros(coords.shape[0], dtype=int)
    for u, v in edge_idx_und:
        degrees[u] += 1
        degrees[v] += 1

    # boundary nodes via convex hull on the coordinate cloud
    hull = ConvexHull(coords)
    boundary_set = set(int(i) for i in hull.vertices)

    return {
        "name": "HydroGraphNet (M80 KNN k=4)",
        "n_nodes": int(coords.shape[0]),
        "n_edges_directed": int(edge_idx_dir.shape[0]),
        "n_edges_undirected": int(edge_idx_und.shape[0]),
        "degrees": degrees,
        "edge_lengths": lengths_und,
        "edge_len_mean": float(lengths_und.mean()),
        "edge_len_std": float(lengths_und.std()),
        "edge_len_cov": float(lengths_und.std() / lengths_und.mean()),
        "edge_len_min": float(lengths_und.min()),
        "edge_len_max": float(lengths_und.max()),
        "boundary_node_count": len(boundary_set),
        "boundary_node_frac": len(boundary_set) / coords.shape[0],
        "n_1d_nodes": 0,
        "n_connections": 0,
        "graph_type": "KNN k=4 over (x,y) coords",
    }


# -----------------------------------------------------------------------
# UrbanFlood structural metrics (read CSVs directly)
# -----------------------------------------------------------------------
def uf_metrics(model_name: str) -> dict:
    base = os.path.join(UF_DATA_DIR, model_name, "train")
    nodes_2d = pd.read_csv(os.path.join(base, "2d_nodes_static.csv"))
    edges_2d_idx = pd.read_csv(os.path.join(base, "2d_edge_index.csv"))
    edges_2d_static = pd.read_csv(os.path.join(base, "2d_edges_static.csv"))
    conn = pd.read_csv(os.path.join(base, "1d2d_connections.csv"))
    nodes_1d = pd.read_csv(os.path.join(base, "1d_nodes_static.csv"))

    coords = nodes_2d[["position_x", "position_y"]].values
    src = edges_2d_idx["from_node"].values
    dst = edges_2d_idx["to_node"].values

    pair_min = np.minimum(src, dst)
    pair_max = np.maximum(src, dst)
    pairs = np.column_stack([pair_min, pair_max])
    undirected = np.unique(pairs, axis=0)

    # degree counts for the undirected 2D graph
    deg = Counter()
    for u, v in undirected:
        deg[int(u)] += 1
        deg[int(v)] += 1
    degrees = np.array([deg.get(i, 0) for i in range(coords.shape[0])])

    edge_lengths = edges_2d_static["length"].values.astype(float)
    # undirected edge lengths — pick one per undirected pair
    # (use the directed CSV order; dedup by pair_min/pair_max)
    seen = set()
    und_lengths = []
    for s, d, L in zip(src, dst, edge_lengths):
        key = (int(min(s, d)), int(max(s, d)))
        if key not in seen:
            seen.add(key)
            und_lengths.append(float(L))
    und_lengths = np.array(und_lengths)

    hull = ConvexHull(coords)
    boundary_set = set(int(i) for i in hull.vertices)
    inlet_set = set(int(i) for i in conn["node_2d"].values)
    bc_union = boundary_set | inlet_set

    return {
        "name": f"UrbanFlood {model_name}",
        "n_nodes": int(coords.shape[0]),
        "n_edges_directed": int(len(src)),
        "n_edges_undirected": int(undirected.shape[0]),
        "degrees": degrees,
        "edge_lengths": und_lengths,
        "edge_len_mean": float(und_lengths.mean()),
        "edge_len_std": float(und_lengths.std()),
        "edge_len_cov": float(und_lengths.std() / und_lengths.mean()),
        "edge_len_min": float(und_lengths.min()),
        "edge_len_max": float(und_lengths.max()),
        "boundary_node_count": len(boundary_set),
        "boundary_node_frac": len(boundary_set) / coords.shape[0],
        "inlet_node_count": len(inlet_set),
        "bc_union_count": len(bc_union),
        "bc_union_frac": len(bc_union) / coords.shape[0],
        "n_1d_nodes": int(nodes_1d.shape[0]),
        "n_connections": int(conn.shape[0]),
        "graph_type": "Physical 2D mesh + 1D coupling",
    }


# -----------------------------------------------------------------------
# Log mining: per-epoch loss components
# -----------------------------------------------------------------------
EPOCH_LINE_HG = re.compile(
    r"Epoch (\d+) completed\. Train Loss: ([\d.eE+-]+) \| Val Loss: ([\d.eE+-]+)"
)
COMP_LINE = re.compile(
    r"Components: loss_one: ([\d.eE+-]+) \| loss_stability: ([\d.eE+-]+).*?local_physics_loss: ([\d.eE+-]+)"
)
EPOCH_LINE_UF = re.compile(
    r"Epoch (\d+) — Avg Loss: ([\d.eE+-]+) \| loss_one: ([\d.eE+-]+) \| loss_stability: ([\d.eE+-]+) \| edge_loss: ([\d.eE+-]+) \| local_physics_loss: ([\d.eE+-]+)"
)


def parse_hg_log(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        text = f.read()
    # HG log has Epoch X completed line, then a Components line
    for ep_m in EPOCH_LINE_HG.finditer(text):
        epoch = int(ep_m.group(1))
        train_loss = float(ep_m.group(2))
        val_loss = float(ep_m.group(3))
        # find the Components line right after
        after = text[ep_m.end():]
        comp_m = COMP_LINE.search(after[:1000])
        if not comp_m:
            continue
        loss_one = float(comp_m.group(1))
        loss_stab = float(comp_m.group(2))
        lmc = float(comp_m.group(3))
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "prediction_mse": loss_one + loss_stab,
            "local_physics_loss": lmc,
        })
    return pd.DataFrame(rows).drop_duplicates(subset="epoch").reset_index(drop=True)


def parse_uf_log(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            m = EPOCH_LINE_UF.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            rows.append({
                "epoch": epoch,
                "avg_loss": float(m.group(2)),
                "prediction_mse": float(m.group(3)) + float(m.group(4)),
                "edge_loss": float(m.group(5)),
                "local_physics_loss": float(m.group(6)),
            })
    return pd.DataFrame(rows).drop_duplicates(subset="epoch").reset_index(drop=True)


def loss_correlation(df: pd.DataFrame, warmup_epochs: int) -> dict:
    sub = df[df["epoch"] >= warmup_epochs]
    if len(sub) < 3:
        return {"n": int(len(sub)), "pearson_r": float("nan"), "warmup_skipped": warmup_epochs}
    r = float(np.corrcoef(sub["prediction_mse"], sub["local_physics_loss"])[0, 1])
    return {"n": int(len(sub)), "pearson_r": r, "warmup_skipped": warmup_epochs}


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
def plot_degree_distribution(metrics_list: list[dict], out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in metrics_list:
        bins = np.arange(0, max(m["degrees"].max(), 10) + 2) - 0.5
        ax.hist(m["degrees"], bins=bins, alpha=0.55, label=m["name"], density=True)
    ax.set_xlabel("Node degree (undirected 2D graph)")
    ax.set_ylabel("Fraction of nodes")
    ax.set_title("Node degree distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_edge_length_distribution(metrics_list: list[dict], out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in metrics_list:
        ax.hist(m["edge_lengths"], bins=60, alpha=0.55, label=f"{m['name']} (CoV={m['edge_len_cov']:.2f})", density=True)
    ax.set_xlabel("Edge length (model units)")
    ax.set_ylabel("Density")
    ax.set_title("Edge length distribution")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_trajectory(df: pd.DataFrame, title: str, out_path: str, warmup: int):
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(df["epoch"], df["prediction_mse"], "-o", color="C0", label="Prediction MSE (loss_one+stab)", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Prediction MSE", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["local_physics_loss"], "-s", color="C3", label="LMC loss", markersize=4)
    ax2.set_ylabel("LMC loss", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_yscale("log")

    if warmup > 0:
        ax1.axvline(warmup - 0.5, color="grey", linestyle="--", alpha=0.5, label=f"end of warmup (ep {warmup})")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("[EDA] Computing structural metrics …")
    hg = hg_metrics()
    uf_m1 = uf_metrics("Model_1")
    try:
        uf_m2 = uf_metrics("Model_2")
        uf_list = [uf_m1, uf_m2]
    except Exception as exc:
        print(f"  [warn] Model_2 metric load failed: {exc}")
        uf_list = [uf_m1]

    metrics_all = [hg] + uf_list

    print("[EDA] Writing structural summary CSV …")
    summary_rows = []
    for m in metrics_all:
        summary_rows.append({
            "dataset": m["name"],
            "graph_type": m["graph_type"],
            "n_2d_nodes": m["n_nodes"],
            "n_1d_nodes": m.get("n_1d_nodes", 0),
            "n_connections": m.get("n_connections", 0),
            "n_undirected_edges": m["n_edges_undirected"],
            "degree_mean": float(m["degrees"].mean()),
            "degree_std": float(m["degrees"].std()),
            "degree_min": int(m["degrees"].min()),
            "degree_max": int(m["degrees"].max()),
            "edge_len_mean": m["edge_len_mean"],
            "edge_len_std": m["edge_len_std"],
            "edge_len_cov": m["edge_len_cov"],
            "edge_len_min": m["edge_len_min"],
            "edge_len_max": m["edge_len_max"],
            "boundary_node_count": m["boundary_node_count"],
            "boundary_node_frac": m["boundary_node_frac"],
            "inlet_node_count": m.get("inlet_node_count", 0),
            "bc_union_frac": m.get("bc_union_frac", m["boundary_node_frac"]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "structural_summary.csv"), index=False)
    print(summary_df.to_string(index=False))

    print("\n[EDA] Plots: degree + edge-length distributions …")
    plot_degree_distribution(metrics_all, os.path.join(OUT_DIR, "degree_distribution.png"))
    plot_edge_length_distribution(metrics_all, os.path.join(OUT_DIR, "edge_length_distribution.png"))

    print("\n[EDA] Parsing training logs …")
    log_results = {}
    if os.path.exists(HG_TRAIN_LOG):
        hg_df = parse_hg_log(HG_TRAIN_LOG)
        hg_df.to_csv(os.path.join(OUT_DIR, "loss_trajectory_hydrographnet.csv"), index=False)
        hg_corr = loss_correlation(hg_df, warmup_epochs=5)
        log_results["HydroGraphNet"] = {"path": HG_TRAIN_LOG, "rows": len(hg_df), "corr": hg_corr}
        plot_loss_trajectory(hg_df, "HydroGraphNet — prediction MSE vs LMC loss",
                             os.path.join(OUT_DIR, "loss_trajectory_hydrographnet.png"), warmup=5)
        print(f"  HG epochs: {len(hg_df)}, Pearson r(post-warmup) = {hg_corr['pearson_r']:.3f}")
    else:
        print(f"  [warn] HG train.log not found at {HG_TRAIN_LOG}")

    if os.path.exists(UF_LMC_TRAIN_LOG):
        uf_df = parse_uf_log(UF_LMC_TRAIN_LOG)
        uf_df.to_csv(os.path.join(OUT_DIR, "loss_trajectory_urbanflood.csv"), index=False)
        uf_corr = loss_correlation(uf_df, warmup_epochs=5)
        log_results["UrbanFlood (Model_1, v3 Exp 1)"] = {"path": UF_LMC_TRAIN_LOG, "rows": len(uf_df), "corr": uf_corr}
        plot_loss_trajectory(uf_df, "UrbanFlood Model_1 v3 Exp 1 — prediction MSE vs LMC loss",
                             os.path.join(OUT_DIR, "loss_trajectory_urbanflood.png"), warmup=5)
        print(f"  UF epochs: {len(uf_df)}, Pearson r(post-warmup) = {uf_corr['pearson_r']:.3f}")
    else:
        print(f"  [warn] UF train.log not found at {UF_LMC_TRAIN_LOG}")

    with open(os.path.join(OUT_DIR, "log_correlation_summary.json"), "w") as f:
        json.dump(log_results, f, indent=2, default=str)

    print(f"\n[EDA] Done. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
