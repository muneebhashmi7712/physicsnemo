"""Aggregate UrbanFlood v8 EXTENSION inference results.

Covers the 18 new experiments: City-2 + LMC-on extension of the original
Phase-1 interpolation study.

  Cell A: Model_1 + LMC on    (6 exps — p1a/p1b_h46/p1b_h24 × lmc × s0,s1)
  Cell B: Model_2 + no-LMC    (6 exps — p1a/p1b_h46/p1b_h24 × nolc × s0,s1)
  Cell C: Model_2 + LMC on    (6 exps — p1a/p1b_h46/p1b_h24 × lmc × s0,s1)

Also loads the original 9 Model_1/no-LMC experiments (Cell 0) from
aggregate_v8_results.py for cross-cell comparison. Reads INFER_JOBIDS from
a JSON sidecar file (infer_jobids_ext.json) to keep IDs editable without
touching this script.

Usage:
    # 1. Fill in infer_jobids_ext.json (see below) once inference completes.
    # 2. python aggregate_v8_ext_results.py
    # 3. Optional: --plots to save bar charts to v8_ext_plots/
    python aggregate_v8_ext_results.py [--plots]

infer_jobids_ext.json format:
    {
        "p1a_lmc_s0":          1703510,
        "p1a_lmc_s1":          1703511,
        ...
    }
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/outputs_v8"
DATA_ROOT    = "/home/woody/iwi5/iwi5416h/urbanflood/data_v8"
INTENSITY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "event_intensity.csv")
JOBIDS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "infer_jobids_ext.json")

# Cell 0 (original, already done) — used only for cross-cell comparison.
CELL0_EXP_NAMES = [
    "p1a_s0", "p1a_s1", "p1a_s2",
    "p1b_h46_s0", "p1b_h46_s1", "p1b_h46_s2",
    "p1b_h24_s0", "p1b_h24_s1", "p1b_h24_s2",
]
CELL0_INFER_JOBIDS = {
    "p1a_s0": 1686971, "p1a_s1": 1686972, "p1a_s2": 1686973,
    "p1b_h46_s0": 1686974, "p1b_h46_s1": 1686975, "p1b_h46_s2": 1686976,
    "p1b_h24_s0": 1686977, "p1b_h24_s1": 1686978, "p1b_h24_s2": 1686979,
}

# Cell A/B/C — 18 new experiments.
EXT_EXP_NAMES = [
    # A: Model_1 + lmc
    "p1a_lmc_s0",       "p1a_lmc_s1",
    "p1b_h46_lmc_s0",   "p1b_h46_lmc_s1",
    "p1b_h24_lmc_s0",   "p1b_h24_lmc_s1",
    # B: Model_2 + nolc
    "p1a_m2_s0",        "p1a_m2_s1",
    "p1b_h46_m2_s0",    "p1b_h46_m2_s1",
    "p1b_h24_m2_s0",    "p1b_h24_m2_s1",
    # C: Model_2 + lmc
    "p1a_m2_lmc_s0",    "p1a_m2_lmc_s1",
    "p1b_h46_m2_lmc_s0","p1b_h46_m2_lmc_s1",
    "p1b_h24_m2_lmc_s0","p1b_h24_m2_lmc_s1",
]

# Cell metadata derived from exp name.
def _cell_meta(exp: str) -> dict:
    model = "Model_2" if "_m2_" in exp else "Model_1"
    lmc   = "lmc"  if "_lmc_" in exp else "nolc"
    if exp.startswith("p1a_"):
        split_type = "stratified80"
        holdout_bin = None
    elif "h46" in exp:
        split_type = "holdout"
        holdout_bin = "4-6"
    else:
        split_type = "holdout"
        holdout_bin = "2-4"
    return {"model": model, "lmc": lmc,
            "split_type": split_type, "holdout_bin": holdout_bin}


# ---------------------------------------------------------------------------
# Parsers (minimal versions of the originals in aggregate_v8_results.py)
# ---------------------------------------------------------------------------
def parse_tensor_line(line: str) -> list:
    m = re.search(r"tensor\(\[([^\]]+)\]\)", line)
    if not m:
        return []
    return [float(x.strip()) for x in m.group(1).split(",")]


def parse_event_line(line: str):
    m = re.search(r"Event\s+event_(\d+):\s+Mean\s+RMSE\s+=\s+([\d.eE+-]+)\s+(?:ft|m)\b", line)
    if not m:
        return None
    event_id, overall = int(m.group(1)), float(m.group(2))
    m2d = re.search(r"\|\s*2D\s*=\s*([\d.eE+-]+)\s+(?:ft|m)\b", line)
    m1d = re.search(r"\|\s*1D\s*=\s*([\d.eE+-]+)\s+(?:ft|m)\b", line)
    return event_id, overall, (float(m2d.group(1)) if m2d else None), (float(m1d.group(1)) if m1d else None)


def parse_inference_out(path: str) -> dict:
    out = {"step_2d_means": [], "per_event_rmse": {}, "per_event_2d": {}}
    with open(path) as f:
        for line in f:
            if "findfont" in line:
                continue
            if "Overall Mean RMSE — 2D nodes" in line:
                out["step_2d_means"] = parse_tensor_line(line)
            else:
                ev = parse_event_line(line)
                if ev:
                    eid, overall, rmse_2d, _ = ev
                    out["per_event_rmse"][eid] = overall
                    if rmse_2d is not None:
                        out["per_event_2d"][eid] = rmse_2d
    return out


def find_out_file(exp: str, jobid: int) -> str:
    path = os.path.join(OUTPUTS_ROOT, f"uf_v8_infer_{exp}_{jobid}.out")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Log not found: {path}\n"
            f"  -> Check that inference completed and JOBID {jobid} is correct "
            f"in infer_jobids_ext.json"
        )
    return path


def load_splits(exp: str) -> dict:
    with open(os.path.join(DATA_ROOT, exp, "splits.json")) as f:
        return json.load(f)


def load_intensity() -> dict:
    """(model_name, event_id) -> (total_inches, bin) — both models, no collision."""
    out = {}
    with open(INTENSITY_CSV) as f:
        for row in csv.DictReader(f):
            out[(row["model_name"], int(row["event_id"]))] = (
                float(row["total_inches"]), row["bin"])
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def cell_label(model: str, lmc: str) -> str:
    return f"{model}/{lmc}"


def aggregate_cell(exp_names: list, parsed: dict, splits: dict,
                   intensity: dict) -> dict:
    """Return per-split-type mean±std 2D RMSE, per-event 2D by split, and
    Phase-1A per-bin pooled 2D RMSE (for the correct interpolation gap reference)."""
    by_split: dict = defaultdict(list)
    per_event_2d_by_split: dict = defaultdict(list)
    # Phase 1A only: per-bin pooled per-event 2D RMSE (the correct iid reference)
    p1a_per_bin_2d: dict = defaultdict(list)

    for exp in exp_names:
        meta  = _cell_meta(exp)
        model = meta["model"]
        stype = f"{meta['split_type']}_{meta['holdout_bin'] or 'iid'}"
        p     = parsed[exp]
        sp    = splits[exp]

        step_means = p["step_2d_means"]
        rollout_mean = mean(step_means) if step_means else float("nan")
        by_split[stype].append(rollout_mean)

        for eid in sp["test_event_ids"]:
            if eid in p["per_event_2d"]:
                val = p["per_event_2d"][eid]
                per_event_2d_by_split[stype].append(val)
                if meta["split_type"] == "stratified80":
                    ikey = (model, eid)
                    if ikey in intensity:
                        bin_label = intensity[ikey][1]
                        p1a_per_bin_2d[bin_label].append(val)

    return {"by_split": dict(by_split),
            "per_event_2d_by_split": dict(per_event_2d_by_split),
            "p1a_per_bin_2d": dict(p1a_per_bin_2d)}


def interpolation_gap(cell_data: dict) -> dict:
    """
    For each holdout bin compute: holdout_mean_2d - Phase-1A SAME-BIN mean_2d.
    Uses per-event 2D RMSEs pooled across seeds (mirrors v8 §10.7 Analysis 1).

    The reference is the Phase-1A per-bin mean (not the overall i.i.d. mean),
    so the gap isolates the generalization cost on THAT specific rainfall intensity,
    not the gap relative to an average over all bins (which would conflate bin
    difficulty with generalization).
    """
    p1a_per_bin = cell_data.get("p1a_per_bin_2d", {})

    results = {}
    for prefix, b in (("holdout_4-6", "4-6"), ("holdout_2-4", "2-4")):
        h_vals   = cell_data["per_event_2d_by_split"].get(prefix, [])
        iid_vals = p1a_per_bin.get(b, [])
        if not h_vals or not iid_vals:
            results[b] = None
            continue
        h_mean   = mean(h_vals)
        iid_mean = mean(iid_vals)
        gap      = h_mean - iid_mean
        results[b] = {
            "holdout_mean": h_mean,
            "iid_mean":     iid_mean,
            "gap":          gap,
            "gap_pct":      100 * gap / iid_mean if iid_mean else float("nan"),
            "n_holdout":    len(h_vals),
            "n_iid":        len(iid_vals),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plots", action="store_true",
                        help="Save bar charts to v8_ext_plots/")
    args = parser.parse_args()

    # Load job IDs.
    if not os.path.isfile(JOBIDS_FILE):
        sys.exit(
            f"ERROR: {JOBIDS_FILE} not found.\n"
            "Create it with the 18 inference job IDs, e.g.:\n"
            '  {"p1a_lmc_s0": 1703510, "p1a_lmc_s1": 1703511, ...}'
        )
    with open(JOBIDS_FILE) as f:
        ext_jobids = json.load(f)
    missing = [e for e in EXT_EXP_NAMES if e not in ext_jobids]
    if missing:
        sys.exit(f"ERROR: infer_jobids_ext.json is missing entries: {missing}")

    intensity = load_intensity()

    # Parse all logs.
    all_parsed  = {}
    all_splits  = {}
    all_exp_names = CELL0_EXP_NAMES + EXT_EXP_NAMES
    for exp in CELL0_EXP_NAMES:
        jobid = CELL0_INFER_JOBIDS[exp]
        all_parsed[exp] = parse_inference_out(find_out_file(exp, jobid))
        all_splits[exp] = load_splits(exp)
    for exp in EXT_EXP_NAMES:
        jobid = ext_jobids[exp]
        all_parsed[exp] = parse_inference_out(find_out_file(exp, jobid))
        all_splits[exp] = load_splits(exp)

    # Aggregate by cell.
    cells = {
        "Model_1/nolc": CELL0_EXP_NAMES,
        "Model_1/lmc":  [e for e in EXT_EXP_NAMES if "_lmc_" in e and "_m2_" not in e],
        "Model_2/nolc": [e for e in EXT_EXP_NAMES if "_m2_" in e and "_lmc_" not in e],
        "Model_2/lmc":  [e for e in EXT_EXP_NAMES if "_m2_" in e and "_lmc_" in e],
    }
    cell_data = {label: aggregate_cell(exps, all_parsed, all_splits, intensity)
                 for label, exps in cells.items()}

    # -----------------------------------------------------------------------
    # Table 1: per-experiment headline 2D RMSE
    # -----------------------------------------------------------------------
    print("# UrbanFlood v8 Extension — Headline 2D RMSE per experiment\n")
    print("Mean over 8 rollout steps of `Overall Mean RMSE — 2D nodes` (ft).\n")
    print("| Exp | Model | LMC | Split | n_test | 2D mean RMSE (ft) |")
    print("|---|---|---|---|---|---|")
    for exp in EXT_EXP_NAMES:
        meta = _cell_meta(exp)
        sp   = all_splits[exp]
        sm   = all_parsed[exp]["step_2d_means"]
        rm   = f"{mean(sm):.4f}" if sm else "—"
        print(f"| {exp} | {meta['model']} | {meta['lmc']} | "
              f"{meta['split_type']} {meta['holdout_bin'] or 'i.i.d.'} | "
              f"{sp['num_test']} | **{rm}** |")

    # -----------------------------------------------------------------------
    # Table 2: per-cell mean ± std by split type
    # -----------------------------------------------------------------------
    SPLIT_KEYS = [
        ("stratified80_iid",  "Phase 1A (i.i.d.)"),
        ("holdout_4-6",       "Phase 1B-1 (holdout 4-6\")"),
        ("holdout_2-4",       "Phase 1B-2 (holdout 2-4\")"),
    ]
    print("\n\n# Table 2 — per-cell mean ± std 2D RMSE\n")
    header = "| Split type | " + " | ".join(cells.keys()) + " |"
    sep    = "|---|" + "---|" * len(cells)
    print(header)
    print(sep)
    for skey, slabel in SPLIT_KEYS:
        row = f"| {slabel} |"
        for label in cells:
            vals = cell_data[label]["by_split"].get(skey, [])
            if not vals:
                row += " — |"
            elif len(vals) == 1:
                row += f" {vals[0]:.4f} |"
            else:
                row += f" {mean(vals):.4f} ± {stdev(vals):.4f} |"
        print(row)

    # -----------------------------------------------------------------------
    # Table 3: interpolation gap per cell
    # -----------------------------------------------------------------------
    print("\n\n# Table 3 — Interpolation gap (Phase 1B holdout RMSE − Phase 1A bin RMSE)\n")
    print("Negative = model handles held-out bin better than i.i.d. baseline on same bin.\n")
    print("Each gap uses per-event 2D RMSEs pooled across seeds (same methodology as "
          "v8 §10.7 Analysis 1).\n")

    gaps_by_cell = {label: interpolation_gap(cell_data[label]) for label in cells}

    print("| Held-out bin | " + " | ".join(cells.keys()) + " |")
    print("|---|" + "---|" * len(cells))
    for _, b in (("holdout_4-6", "4-6"), ("holdout_2-4", "2-4")):
        row = f"| {b}\" |"
        for label in cells:
            g = gaps_by_cell[label].get(b)
            if g is None:
                row += " — |"
            else:
                row += (f" {g['gap']:+.4f} ft ({g['gap_pct']:+.1f}%) |")
        print(row)

    # Raw numbers for the gap table.
    print("\n## Raw numbers (holdout mean / i.i.d. mean / gap)\n")
    for label in cells:
        print(f"### {label}")
        for _, b in (("holdout_4-6", "4-6"), ("holdout_2-4", "2-4")):
            g = gaps_by_cell[label].get(b)
            if g:
                print(f"  {b}\": holdout={g['holdout_mean']:.4f} (n={g['n_holdout']})"
                      f"  iid={g['iid_mean']:.4f} (n={g['n_iid']})"
                      f"  gap={g['gap']:+.4f} ({g['gap_pct']:+.1f}%)")

    # -----------------------------------------------------------------------
    # Table 4: LMC effect on interpolation gap (delta-gap)
    # -----------------------------------------------------------------------
    print("\n\n# Table 4 — LMC effect on the interpolation gap\n")
    print("Δ gap = gap(lmc) − gap(nolc).  "
          "Negative = LMC *shrinks* the interpolation gap (helpful).\n")
    print("| Held-out bin | Model_1 Δ gap | Model_2 Δ gap |")
    print("|---|---|---|")
    for _, b in (("holdout_4-6", "4-6"), ("holdout_2-4", "2-4")):
        g_m1_nolc = gaps_by_cell.get("Model_1/nolc", {}).get(b)
        g_m1_lmc  = gaps_by_cell.get("Model_1/lmc",  {}).get(b)
        g_m2_nolc = gaps_by_cell.get("Model_2/nolc", {}).get(b)
        g_m2_lmc  = gaps_by_cell.get("Model_2/lmc",  {}).get(b)
        m1 = (f"{g_m1_lmc['gap']-g_m1_nolc['gap']:+.4f} ft"
              if (g_m1_nolc and g_m1_lmc) else "—")
        m2 = (f"{g_m2_lmc['gap']-g_m2_nolc['gap']:+.4f} ft"
              if (g_m2_nolc and g_m2_lmc) else "—")
        print(f"| {b}\" | {m1} | {m2} |")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "v8_ext_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "model", "lmc", "split_type", "holdout_bin",
                    "n_test", "rollout_mean_2d_rmse_m", "step8_2d_rmse_m"])
        for exp in EXT_EXP_NAMES:
            meta = _cell_meta(exp)
            sm   = all_parsed[exp]["step_2d_means"]
            w.writerow([exp, meta["model"], meta["lmc"],
                        meta["split_type"], meta["holdout_bin"] or "",
                        all_splits[exp]["num_test"],
                        mean(sm) if sm else "",
                        sm[-1] if sm else ""])
    print(f"\nSaved -> {out_csv}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    if args.plots:
        _make_plots(cells, cell_data, gaps_by_cell)


def _make_plots(cells, cell_data, gaps_by_cell):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[plots skipped — matplotlib not available]")
        return

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "v8_ext_plots")
    os.makedirs(out_dir, exist_ok=True)
    cell_labels = list(cells.keys())
    colors = {"Model_1/nolc": "#4C72B0", "Model_1/lmc": "#74aee8",
              "Model_2/nolc": "#DD8452", "Model_2/lmc": "#f5b97f"}

    SPLIT_KEYS = [
        ("stratified80_iid", "Phase 1A\n(i.i.d.)"),
        ("holdout_4-6",      "Phase 1B-1\n(holdout 4-6\")"),
        ("holdout_2-4",      "Phase 1B-2\n(holdout 2-4\")"),
    ]

    # ---- Plot 1: 2D RMSE per split-type, grouped by cell -------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    n_groups = len(SPLIT_KEYS)
    n_cells  = len(cell_labels)
    width    = 0.18
    x        = np.arange(n_groups)
    offsets  = np.linspace(-(n_cells - 1) / 2, (n_cells - 1) / 2, n_cells) * width

    for i, label in enumerate(cell_labels):
        means, errs = [], []
        for skey, _ in SPLIT_KEYS:
            vals = cell_data[label]["by_split"].get(skey, [])
            means.append(mean(vals) if vals else 0)
            errs.append(stdev(vals) if len(vals) > 1 else 0)
        bars = ax.bar(x + offsets[i], means, width,
                      label=label, color=colors[label],
                      yerr=errs, capsize=3, error_kw={"elinewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels([s for _, s in SPLIT_KEYS])
    ax.set_ylabel("Mean 2D RMSE (ft)")
    ax.set_title("v8 Extension — 2D RMSE by split type and cell")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    p = os.path.join(out_dir, "v8_ext_rmse_by_split.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Saved plot -> {p}")

    # ---- Plot 2: Interpolation gap (%) per cell per holdout bin ------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    for ax, (_, b) in zip(axes, (("holdout_4-6", "4-6"), ("holdout_2-4", "2-4"))):
        gaps_pct = []
        for label in cell_labels:
            g = gaps_by_cell[label].get(b)
            gaps_pct.append(g["gap_pct"] if g else 0)
        bar_colors = [colors[l] for l in cell_labels]
        bars = ax.bar(cell_labels, gaps_pct, color=bar_colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Interpolation gap — holdout {b}\"")
        ax.set_ylabel("Gap (%) [1B − 1A per-bin, positive = 1B worse]")
        ax.set_xticks(range(len(cell_labels)))
        ax.set_xticklabels(cell_labels, rotation=20, ha="right", fontsize=8)
        for bar, pct in zip(bars, gaps_pct):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if pct >= 0 else -1.2),
                    f"{pct:+.1f}%", ha="center", va="bottom", fontsize=8)
    plt.suptitle("v8 Extension — Interpolation gap by cell\n"
                 "(small gap = model generalizes to unseen rainfall intensities)",
                 fontsize=10)
    plt.tight_layout()
    p = os.path.join(out_dir, "v8_ext_interp_gap.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Saved plot -> {p}")

    # ---- Plot 3: LMC effect on gap (delta-gap across cells) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, (_, b) in zip(axes, (("holdout_4-6", "4-6"), ("holdout_2-4", "2-4"))):
        cities = ["Model_1", "Model_2"]
        delta_gaps = []
        for city in cities:
            g_nolc = gaps_by_cell.get(f"{city}/nolc", {}).get(b)
            g_lmc  = gaps_by_cell.get(f"{city}/lmc",  {}).get(b)
            if g_nolc and g_lmc:
                delta_gaps.append(g_lmc["gap"] - g_nolc["gap"])
            else:
                delta_gaps.append(0)
        city_colors = ["#4C72B0", "#DD8452"]
        bars = ax.bar(cities, delta_gaps, color=city_colors)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Δ gap (lmc − nolc) — holdout {b}\"")
        ax.set_ylabel("Δ interpolation gap (ft)\n[negative = LMC shrinks gap]")
        for bar, val in zip(bars, delta_gaps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.0005 if val >= 0 else -0.002),
                    f"{val:+.4f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("v8 Extension — LMC effect on interpolation gap\n"
                 "(does LMC help or hurt generalization to unseen rainfall?)",
                 fontsize=10)
    plt.tight_layout()
    p = os.path.join(out_dir, "v8_ext_lmc_delta_gap.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Saved plot -> {p}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
