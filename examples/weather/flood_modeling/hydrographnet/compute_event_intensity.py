"""Compute per-event total rainfall (inches) for UrbanFlood Model_1 / Model_2.

Reads each event's 2d_nodes_dynamic_all.csv. Per the dataset loader
(hydrographnet_dataset.py:1340) rainfall is spatially uniform per timestep —
we take node_idx==0 and sum across timesteps.

Output: event_intensity.csv with columns
    model_name, split, event_id, num_timesteps, total_inches, bin
where bin uses the proposal §3.5 cuts: 0-2, 2-4, 4-6, 6-8, 8-10 inches.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


BIN_EDGES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, np.inf]
BIN_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]


def event_total_inches(event_dir: str) -> tuple[int, float]:
    path = os.path.join(event_dir, "2d_nodes_dynamic_all.csv")
    df = pd.read_csv(path, usecols=["timestep", "node_idx", "rainfall"])
    n0 = df[df.node_idx == 0].sort_values("timestep")
    return int(n0.shape[0]), float(n0["rainfall"].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/woody/iwi5/iwi5416h/urbanflood/data")
    p.add_argument("--models", nargs="+", default=["Model_1"])
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: alongside this script).")
    args = p.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "event_intensity.csv")

    rows = []
    for model in args.models:
        for split in ("train", "test"):
            split_dir = os.path.join(args.data_root, model, split)
            if not os.path.isdir(split_dir):
                continue
            entries = sorted(
                [d for d in os.listdir(split_dir) if d.startswith("event_")],
                key=lambda x: int(re.search(r"(\d+)", x).group(1)),
            )
            for ev in entries:
                ev_dir = os.path.join(split_dir, ev)
                n_t, total = event_total_inches(ev_dir)
                ev_id = int(re.search(r"(\d+)", ev).group(1))
                bin_label = BIN_LABELS[int(np.digitize(total, BIN_EDGES) - 1)]
                rows.append({
                    "model_name": model,
                    "split": split,
                    "event_id": ev_id,
                    "num_timesteps": n_t,
                    "total_inches": round(total, 4),
                    "bin": bin_label,
                })

    df = pd.DataFrame(rows).sort_values(["model_name", "event_id"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")

    for model in args.models:
        sub = df[df.model_name == model]
        print(f"\n=== {model} ({len(sub)} events) ===")
        bin_counts_all = sub.groupby("bin").size().reindex(BIN_LABELS, fill_value=0)
        bin_counts_train = sub[sub.split == "train"].groupby("bin").size().reindex(BIN_LABELS, fill_value=0)
        bin_counts_test = sub[sub.split == "test"].groupby("bin").size().reindex(BIN_LABELS, fill_value=0)
        table = pd.DataFrame({
            "train": bin_counts_train, "test": bin_counts_test, "total": bin_counts_all
        })
        print(table.to_string())
        print(f"total_inches  min={sub.total_inches.min():.3f}  max={sub.total_inches.max():.3f}  "
              f"mean={sub.total_inches.mean():.3f}  median={sub.total_inches.median():.3f}")


if __name__ == "__main__":
    sys.exit(main())
