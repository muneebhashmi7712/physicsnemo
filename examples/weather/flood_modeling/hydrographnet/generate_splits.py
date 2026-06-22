"""Generate per-experiment symlinked data dirs for UrbanFlood v8 interpolation.

For each experiment (mode × seed), creates::

    <out_root>/<exp_name>/<model>/train/   <- symlinks to selected train events
    <out_root>/<exp_name>/<model>/test/    <- symlinks to selected test events
    <out_root>/<exp_name>/splits.json      <- reproducible record

Static CSVs (graph topology, identical across all events) are symlinked into
both train/ and test/. Event subdirs are symlinked as a whole. Each experiment
gets its own dyn_norm_stats.json by virtue of having its own train/ folder.

Modes
-----
stratified80   -- 80/20 stratified split by rainfall bin (Phase 1A).
holdout        -- entire bin held out as test set (Phase 1B). Requires
                  --holdout-bin: one of "0-2", "2-4", "4-6", "6-8", "8-10", or a
                  comma-separated set ("6-8,8-10") to hold out several bins together
                  (e.g. a combined ">=6" high-intensity holdout).

Bin "8-10" has only 2 events in Model_1 and is never drawn into Model_1's stratified
test set, so for Model_1 it is held out combined with 6-8 (--holdout-bin "6-8,8-10").
For Model_2 (9 events in 8-10) it can be held out on its own (--holdout-bin "8-10").
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import pandas as pd


STATIC_CSV_NAMES = [
    "1d2d_connections.csv",
    "1d_edge_index.csv",
    "1d_edges_static.csv",
    "1d_nodes_static.csv",
    "2d_edge_index.csv",
    "2d_edges_static.csv",
    "2d_nodes_static.csv",
]

BIN_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-10"]


def _symlink(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def _find_event_source(data_root: str, model: str, event_id: int) -> str:
    """Return absolute path to the original event dir (either train/ or test/)."""
    for split in ("train", "test"):
        candidate = os.path.join(data_root, model, split, f"event_{event_id}")
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"event_{event_id} not found under {data_root}/{model}/{{train,test}}"
    )


def _find_static_source(data_root: str, model: str) -> str:
    """Static CSVs are identical in train/ and test/; return whichever exists."""
    for split in ("train", "test"):
        d = os.path.join(data_root, model, split)
        if os.path.isfile(os.path.join(d, STATIC_CSV_NAMES[0])):
            return d
    raise FileNotFoundError(f"No static CSVs found under {data_root}/{model}")


def stratified_split(df_model: pd.DataFrame, seed: int, test_frac: float = 0.2):
    """Stratified random split by bin. Bins with <2 events go fully to train."""
    rng = random.Random(seed)
    train_ids, test_ids = [], []
    for b in BIN_LABELS:
        ids = df_model[df_model["bin"] == b]["event_id"].tolist()
        rng.shuffle(ids)
        n_test = round(len(ids) * test_frac)
        if len(ids) - n_test < 1:
            # Tiny bin — keep all in train.
            train_ids.extend(ids)
            continue
        test_ids.extend(ids[:n_test])
        train_ids.extend(ids[n_test:])
    return sorted(train_ids), sorted(test_ids)


def random_split(df_model: pd.DataFrame, seed: int, test_frac: float = 0.2):
    """Plain random 80/20 split over all eligible events (bin-agnostic).

    Used by v9 (time-less peak-depth prediction): the task is no longer about
    interpolating across rainfall bins, so no stratification or bin holdout is
    needed — just measure how well the emulator predicts on unseen events.
    """
    rng = random.Random(seed)
    ids = df_model["event_id"].tolist()
    rng.shuffle(ids)
    n_test = max(1, round(len(ids) * test_frac))
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]
    return sorted(train_ids), sorted(test_ids)


def holdout_split(df_model: pd.DataFrame, holdout_bin: str):
    """Held-out bin(s) -> test; all other bins -> train.

    `holdout_bin` may be a single label ("4-6") or a comma-separated set
    ("6-8,8-10") to hold out several bins together as one combined test set.
    """
    bins = [b.strip() for b in holdout_bin.split(",") if b.strip()]
    bad = [b for b in bins if b not in BIN_LABELS]
    if bad:
        raise ValueError(f"holdout_bin entries {bad} not in {BIN_LABELS}")
    test_ids = df_model[df_model["bin"].isin(bins)]["event_id"].tolist()
    train_ids = df_model[~df_model["bin"].isin(bins)]["event_id"].tolist()
    return sorted(train_ids), sorted(test_ids)


def materialize(
    exp_name: str,
    out_root: str,
    data_root: str,
    model: str,
    train_ids: list[int],
    test_ids: list[int],
    mode: str,
    seed: int,
    holdout_bin: str | None,
    df_model: pd.DataFrame,
) -> str:
    exp_root = os.path.join(out_root, exp_name)
    os.makedirs(exp_root, exist_ok=True)
    static_src = _find_static_source(data_root, model)

    for split_name, ev_ids in (("train", train_ids), ("test", test_ids)):
        split_dir = os.path.join(exp_root, model, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Symlink static CSVs.
        for csv in STATIC_CSV_NAMES:
            src = os.path.join(static_src, csv)
            if os.path.isfile(src):
                _symlink(src, os.path.join(split_dir, csv))

        # Symlink selected event dirs.
        for ev in ev_ids:
            src = _find_event_source(data_root, model, ev)
            _symlink(src, os.path.join(split_dir, f"event_{ev}"))

    # Per-bin counts for sanity.
    bin_counts_train = (
        df_model[df_model.event_id.isin(train_ids)]
        .groupby("bin").size().reindex(BIN_LABELS, fill_value=0).to_dict()
    )
    bin_counts_test = (
        df_model[df_model.event_id.isin(test_ids)]
        .groupby("bin").size().reindex(BIN_LABELS, fill_value=0).to_dict()
    )

    splits_meta = {
        "exp_name": exp_name,
        "model_name": model,
        "mode": mode,
        "seed": seed,
        "holdout_bin": holdout_bin,
        "num_train": len(train_ids),
        "num_test": len(test_ids),
        "train_event_ids": train_ids,
        "test_event_ids": test_ids,
        "bin_counts_train": bin_counts_train,
        "bin_counts_test": bin_counts_test,
    }
    meta_path = os.path.join(exp_root, "splits.json")
    with open(meta_path, "w") as f:
        json.dump(splits_meta, f, indent=2)
    return meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intensity-csv",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "event_intensity.csv"))
    ap.add_argument("--data-root", default="/home/woody/iwi5/iwi5416h/urbanflood/data")
    ap.add_argument("--out-root", default="/home/woody/iwi5/iwi5416h/urbanflood/data_v8",
                    help="Where to create per-experiment symlinked trees.")
    ap.add_argument("--model", default="Model_1")
    ap.add_argument("--mode", choices=["stratified80", "holdout", "random80"], required=True)
    ap.add_argument("--holdout-bin", default=None,
                    help='Holdout mode: one of "0-2","2-4","4-6","6-8","8-10", or a '
                         'comma-separated set like "6-8,8-10" for a combined holdout.')
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exp-name", required=True,
                    help="Used as subdir name. Convention: p1a_s0, p1b_h46_s0, ...")
    args = ap.parse_args()

    df = pd.read_csv(args.intensity_csv)
    df_model = df[df.model_name == args.model].reset_index(drop=True)
    if df_model.empty:
        sys.exit(f"No rows for model={args.model} in {args.intensity_csv}")

    # Only the original train/ events have valid water_level/water_volume
    # ground truth — the original test/ events are blinded (all NaN dynamic
    # targets) and serve as inference-only holdouts. They cannot be used as
    # training data, and there is no GT to compare against if they end up in
    # the held-out test split. Restrict the eligible pool accordingly.
    df_model = df_model[df_model["split"] == "train"].reset_index(drop=True)
    if df_model.empty:
        sys.exit(f"No usable events (split=='train' rows) for model={args.model}")

    if args.mode == "stratified80":
        train_ids, test_ids = stratified_split(df_model, args.seed)
    elif args.mode == "random80":
        train_ids, test_ids = random_split(df_model, args.seed)
    else:
        if args.holdout_bin is None:
            sys.exit("--holdout-bin required for mode=holdout")
        train_ids, test_ids = holdout_split(df_model, args.holdout_bin)

    # Record a clean combined label in splits.json ("6-8,8-10" -> "6-8+8-10").
    holdout_label = (args.holdout_bin.replace(",", "+")
                     if args.mode == "holdout" else None)
    meta_path = materialize(
        args.exp_name, args.out_root, args.data_root, args.model,
        train_ids, test_ids, args.mode, args.seed, holdout_label, df_model,
    )
    print(f"Wrote split metadata -> {meta_path}")
    print(f"  train={len(train_ids)} test={len(test_ids)}")
    print(f"  train_dir = {os.path.join(args.out_root, args.exp_name, args.model, 'train')}")


if __name__ == "__main__":
    main()
