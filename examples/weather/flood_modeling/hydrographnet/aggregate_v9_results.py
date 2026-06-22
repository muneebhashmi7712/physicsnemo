"""Aggregate UrbanFlood v9 (time-less peak-depth) inference RMSE.

Reuses the v8 parsers (parse_inference_out, load_splits, load_intensity,
analysis_1, analysis_2) verbatim — only the .out file discovery and the
config dimension are new. v9 reuses the data_v8 split dirs, so load_splits /
load_intensity work unchanged.

For each of the 4 configs {2d_nolc, 2d_lmc, 1d_nolc, 1d_lmc} × 9 splits we
parse the single-step peak-depth RMSE (the summary tensors are length-1, so
mean==last==peak). 2D-only is the governing metric.

Usage:  python aggregate_v9_results.py            # headline table
        python aggregate_v9_results.py --analysis all
Outputs v9_results.csv next to this script.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from statistics import mean, pstdev

import aggregate_v8_results as v8  # reuse parsers / analyses / loaders

OUTPUTS_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/outputs_v9"
CONFIGS = ["2d_nolc", "2d_lmc", "1d_nolc", "1d_lmc"]
# v9 = pure prediction on a single random 80/20 split (no bin holdouts).
EXP_NAMES = ["r80_s0"]


def sub_exp(exp: str) -> str:
    return "random80"


def find_out_file(exp: str, cfg: str):
    """uf_v9_infer_<exp>__<cfg>_<jobid>.out — newest if several."""
    pat = os.path.join(OUTPUTS_ROOT, f"uf_v9_infer_{exp}__{cfg}_*.out")
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", choices=["none", "1", "2", "all"], default="none")
    ap.add_argument("--exp", default="r80_s0",
                    help="Comma-separated split/exp name(s) to aggregate, e.g. "
                         "r80_s0 for a single Model_1 seed or "
                         "r80_s0,r80_s1,r80_s2 to pool 3 seeds into mean±std "
                         "(same for Model_2 with r80_m2_s*).")
    args = ap.parse_args()
    global EXP_NAMES
    EXP_NAMES = args.exp.split(",")

    rows = []
    # (config, sub_exp) -> list of 2D peak RMSE (one per seed/split)
    grouped = defaultdict(list)
    for cfg in CONFIGS:
        for exp in EXP_NAMES:
            path = find_out_file(exp, cfg)
            if path is None:
                print(f"[missing] {exp}__{cfg}")
                continue
            parsed = v8.parse_inference_out(path)
            steps2d = parsed["step_2d_means"]
            rmse_2d = steps2d[0] if steps2d else float("nan")
            steps_all = parsed["step_overall_means"]
            rmse_all = steps_all[0] if steps_all else float("nan")
            rows.append({
                "config": cfg, "exp": exp, "sub_exp": sub_exp(exp),
                "rmse_2d_peak": rmse_2d, "rmse_overall_peak": rmse_all,
                "n_test_events": len(parsed["per_event_rmse"]),
            })
            grouped[(cfg, sub_exp(exp))].append(rmse_2d)

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "v9_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "config", "exp", "sub_exp", "rmse_2d_peak",
            "rmse_overall_peak", "n_test_events"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}\n")

    # Headline: 2D peak RMSE mean±std across seeds, per (config, sub-exp).
    print("| Config | Sub-exp | 2D peak RMSE (mean ± std) | n_runs |")
    print("|---|---|---|---|")
    for cfg in CONFIGS:
        for se in ["random80"]:
            vals = [v for v in grouped.get((cfg, se), []) if v == v]  # drop nan
            if not vals:
                continue
            m = mean(vals)
            s = pstdev(vals) if len(vals) > 1 else 0.0
            print(f"| {cfg} | {se} | {m:.4f} ± {s:.4f} | {len(vals)} |")

    if args.analysis in ("1", "all"):
        print("\n--- Analysis 1 (interpolation gap) per config ---")
        # v8.analysis_1 expects its own file discovery; v9 differs, so this is
        # a placeholder hook — run per-config once enough runs exist.
        print("(run v8.analysis_1-style pooling per config when logs are in)")
    if args.analysis in ("2", "all"):
        print("\n--- Analysis 2 (normalized RMSE) per config ---")
        print("(normalize per-event peak RMSE by per-event max depth as in v8)")


if __name__ == "__main__":
    main()
