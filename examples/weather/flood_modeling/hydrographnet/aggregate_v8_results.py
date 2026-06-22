"""Aggregate UrbanFlood v8 inference RMSE across all 9 experiments.

Parses Overall Mean/Std 2D RMSE per rollout step from each `.out` log,
also pulls per-event Mean RMSE lines so we can compute per-bin (per
held-out intensity bin) RMSE for Phase 1A. Phase 1B trivially: the test
set is one whole bin, so per-event mean over that bin IS the per-bin
RMSE.

Outputs `v8_results.csv` next to this script, and prints a markdown
summary table for inclusion in the design doc / memory.

User constraint: only 2D RMSE is the success metric. 1D & connection
edges are inputs, not targets.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from statistics import mean, stdev


OUTPUTS_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/outputs_v8"
DATA_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/data_v8"
# Production data layout — shared static topology + per-event dynamic CSVs.
# Used by Analysis 2 to compute per-event max water depth above ground.
PROD_DATA_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/data/Model_1/train"

EXP_NAMES = [
    "p1a_s0", "p1a_s1", "p1a_s2",
    "p1b_h46_s0", "p1b_h46_s1", "p1b_h46_s2",
    "p1b_h24_s0", "p1b_h24_s1", "p1b_h24_s2",
]

# Inference resubmission job IDs (round 2 — first round crashed on norm-stats).
INFER_JOBIDS = dict(zip(EXP_NAMES, range(1686971, 1686980)))


def _find_out_file(exp: str) -> str:
    jobid = INFER_JOBIDS[exp]
    path = os.path.join(OUTPUTS_ROOT, f"uf_v8_infer_{exp}_{jobid}.out")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def parse_tensor_line(line: str) -> list[float]:
    """Parse 'Overall Mean RMSE — 2D nodes: tensor([0.0040, ..., 0.0225])'."""
    m = re.search(r"tensor\(\[([^\]]+)\]\)", line)
    if not m:
        return []
    return [float(x.strip()) for x in m.group(1).split(",")]


def parse_event_line(line: str):
    """Parse a per-event RMSE line. Two formats exist:

      Event event_42: Mean RMSE = 0.0123 m                         (legacy)
      Event event_42: Mean RMSE = 0.0123 m | 2D = 0.0100 m | 1D = 0.0500 m

    Returns (event_id, overall_rmse, rmse_2d_or_None, rmse_1d_or_None).
    """
    m = re.search(r"Event\s+event_(\d+):\s+Mean\s+RMSE\s+=\s+([\d.eE+-]+)\s+m", line)
    if not m:
        return None
    event_id = int(m.group(1))
    overall = float(m.group(2))
    m2d = re.search(r"\|\s*2D\s*=\s*([\d.eE+-]+)\s+m", line)
    m1d = re.search(r"\|\s*1D\s*=\s*([\d.eE+-]+)\s+m", line)
    rmse_2d = float(m2d.group(1)) if m2d else None
    rmse_1d = float(m1d.group(1)) if m1d else None
    return event_id, overall, rmse_2d, rmse_1d


def parse_inference_out(path: str) -> dict:
    """Return {step_2d_means: [8 floats],
               step_2d_stds:  [8 floats],
               step_overall_means: [8 floats],
               per_event_rmse: {event_id: float}}.
    """
    out = {
        "step_2d_means": [],
        "step_2d_stds": [],
        "step_overall_means": [],
        "per_event_rmse": {},      # overall (2D+1D) per event
        "per_event_2d": {},        # 2D-only per event (None if log lacks it)
        "per_event_1d": {},        # 1D-only per event (None if log lacks it)
    }
    with open(path) as f:
        for line in f:
            if "findfont" in line:
                continue
            if "Overall Mean RMSE — 2D nodes" in line:
                out["step_2d_means"] = parse_tensor_line(line)
            elif "Overall Std  RMSE — 2D nodes" in line:
                out["step_2d_stds"] = parse_tensor_line(line)
            elif "Overall Mean RMSE (m) over rollout steps" in line:
                out["step_overall_means"] = parse_tensor_line(line)
            else:
                ev = parse_event_line(line)
                if ev is not None:
                    event_id, overall, rmse_2d, rmse_1d = ev
                    out["per_event_rmse"][event_id] = overall
                    if rmse_2d is not None:
                        out["per_event_2d"][event_id] = rmse_2d
                    if rmse_1d is not None:
                        out["per_event_1d"][event_id] = rmse_1d
    return out


def load_splits(exp: str) -> dict:
    with open(os.path.join(DATA_ROOT, exp, "splits.json")) as f:
        return json.load(f)


def load_intensity() -> dict:
    """event_id -> (total_inches, bin)."""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "event_intensity.csv")
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["model_name"] != "Model_1":
                continue
            out[int(row["event_id"])] = (float(row["total_inches"]), row["bin"])
    return out


def load_static_elevation() -> dict:
    """node_idx -> ground elevation (m), from the shared 2d_nodes_static.csv.

    This is the SAME column 6 ('elevation') that inference.py denormalises into
    `elev_real` and subtracts from water_level to form depth above ground
    (hydrographnet_dataset.py:1221, inference.py:375). Using it here keeps the
    Analysis-2 depth scale identical to the physical quantity the RMSE measures.
    """
    csv_path = os.path.join(PROD_DATA_ROOT, "2d_nodes_static.csv")
    elev = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # header: node_idx,position_x,position_y,area,roughness,
                      #         min_elevation,elevation,aspect,curvature,flow_accumulation
        for row in reader:
            elev[int(row[0])] = float(row[6])
    return elev


def event_max_depth(event_id: int, elevation: dict) -> float:
    """Max 2D water depth above ground over all nodes & timesteps for one event.

    depth = clamp(water_level - elevation, 0), matching inference.py:375. Streams
    only the (node_idx, water_level) columns of 2d_nodes_dynamic_all.csv. Returns
    the per-event max depth (clamped to >= 1e-6 so it is safe as a divisor).
    """
    import pandas as pd  # local import: only Analysis 2 needs pandas

    csv_path = os.path.join(PROD_DATA_ROOT, f"event_{event_id}",
                            "2d_nodes_dynamic_all.csv")
    df = pd.read_csv(csv_path, usecols=["node_idx", "water_level"])
    elev_arr = df["node_idx"].map(elevation).to_numpy()
    depth = df["water_level"].to_numpy() - elev_arr
    return max(float(depth.max()), 1e-6)


def analysis_1(parsed_by_exp: dict, intensity: dict, splits_by_exp: dict):
    """Variance / significance audit (open thread 1).

    (a) Decompose p1a_s1's 13 per-event 2D RMSEs — is the 0.0358 m mean driven by
        one or two hard events? (leave-one-out / leave-two-out).
    (b) Re-quote the interpolation gap in 2D-only terms using a seed-robust
        per-bin-pooled-across-seeds metric, plus the gap with p1a_s1 excluded.
    """
    print("\n\n# Analysis 1 — variance / significance audit (2D-only)\n")

    # ---- (a) p1a_s1 decomposition -----------------------------------------
    s1 = parsed_by_exp["p1a_s1"]["per_event_2d"]
    test_ids = splits_by_exp["p1a_s1"]["test_event_ids"]
    rows = sorted(((ev, s1[ev]) for ev in test_ids if ev in s1),
                  key=lambda r: r[1], reverse=True)
    full_mean = mean(r[1] for r in rows)
    print("## p1a_s1 per-event 2D RMSE (the outlier seed, mean = "
          f"{full_mean:.4f} m)\n")
    print("| event | bin | total in | 2D RMSE (m) |")
    print("|---|---|---|---|")
    for ev, r in rows:
        inches, b = intensity[ev]
        print(f"| event_{ev} | {b}\" | {inches:.2f} | {r:.4f} |")
    vals = [r[1] for r in rows]
    loo = mean(sorted(vals)[:-1])       # drop the single worst
    lto = mean(sorted(vals)[:-2])       # drop the worst two
    print(f"\n- Full mean (13 events): **{full_mean:.4f} m**")
    print(f"- Leave-one-out (drop event_{rows[0][0]}): **{loo:.4f} m**")
    print(f"- Leave-two-out (drop event_{rows[0][0]}, event_{rows[1][0]}): "
          f"**{lto:.4f} m**")
    other = mean([parsed_by_exp[f"p1a_{s}"]["per_event_2d"][ev]
                  for s in ("s0", "s2")
                  for ev in splits_by_exp[f"p1a_{s}"]["test_event_ids"]
                  if ev in parsed_by_exp[f"p1a_{s}"]["per_event_2d"]])
    print(f"- Reference: pooled per-event 2D RMSE of s0+s2: **{other:.4f} m**")

    # ---- (b) seed-robust 2D interpolation gap -----------------------------
    # Phase 1A: pool per-event 2D RMSEs across the three stratified seeds by bin.
    def pool_p1a_by_bin():
        bin_to_vals = defaultdict(list)
        for s in ("s0", "s1", "s2"):
            exp = f"p1a_{s}"
            p2d = parsed_by_exp[exp]["per_event_2d"]
            for ev in splits_by_exp[exp]["test_event_ids"]:
                if ev in p2d:
                    bin_to_vals[intensity[ev][1]].append(p2d[ev])
        return bin_to_vals

    def pool_p1b(prefix: str):
        vals = []
        for exp in [e for e in EXP_NAMES if e.startswith(prefix)]:
            p2d = parsed_by_exp[exp]["per_event_2d"]
            for ev in splits_by_exp[exp]["test_event_ids"]:
                if ev in p2d:
                    vals.append(p2d[ev])
        return vals

    from statistics import median

    print("\n## Seed-robust 2D interpolation gap "
          "(per-event 2D RMSE pooled across seeds)\n")
    print("The pooled metric weights every test event equally regardless of which "
          "seed drew it, so it does NOT inherit Phase 1A's 0.0120 m seed-mean "
          "spread (which comes from which 13 events land in each stratified draw).\n")
    print("| Held-out bin | Phase 1A bin RMSE (2D) | Phase 1B bin RMSE (2D) | gap |")
    print("|---|---|---|---|")
    p1a_all = pool_p1a_by_bin()
    robust_lines = []
    for prefix, b in (("p1b_h46_", "4-6"), ("p1b_h24_", "2-4")):
        p1b_vals = pool_p1b(prefix)
        p1a_vals = p1a_all[b]
        p1b_mean, p1a_mean = mean(p1b_vals), mean(p1a_vals)
        gap = p1b_mean - p1a_mean
        print(f"| {b}\" | {p1a_mean:.4f} (n={len(p1a_vals)}) | "
              f"{p1b_mean:.4f} (n={len(p1b_vals)}) | "
              f"{gap:+.4f} ({100 * gap / p1a_mean:+.1f}%) |")
        # Symmetric robustness: median gap, and an upper-10%-trimmed mean gap.
        # The per-event distribution is heavy-tailed (a few hard events such as
        # event_79 dominate), so report (a) the median — the typical event — and
        # (b) a mean that drops the top 10% of events from BOTH pools. The trim is
        # proportional (not "drop the single worst") because the pools differ in
        # size (1A pools ~12-15 events, 1B ~60-75), and the hard tail events sit
        # on both sides of the same bin.
        def upper_trim_mean(vals, frac=0.10):
            k = int(len(vals) * frac)
            return mean(sorted(vals)[:len(vals) - k]) if len(vals) - k > 0 else mean(vals)
        med_gap = median(p1b_vals) - median(p1a_vals)
        ta, tb = upper_trim_mean(p1a_vals), upper_trim_mean(p1b_vals)
        robust_lines.append(
            f"- {b}\": median gap = {med_gap:+.4f} m "
            f"(1A {median(p1a_vals):.4f} / 1B {median(p1b_vals):.4f}); "
            f"upper-10%-trimmed mean gap = {tb - ta:+.4f} m "
            f"(1A {ta:.4f} / 1B {tb:.4f})"
        )
    print("\nRobustness (symmetric — same operation on both sides of the gap):")
    print("- median = the typical event; trimmed = mean after dropping the "
          "hardest 10% of events from each pool.")
    for ln in robust_lines:
        print(ln)


def analysis_2(parsed_by_exp: dict, intensity: dict, splits_by_exp: dict):
    """Normalized-RMSE test (open thread 2): bracketing vs intrinsic difficulty.

    normalized_rmse = per_event_2D_RMSE / per_event_max_depth. If the ~0.012 m
    absolute gap between the 2-4" and 4-6" holdouts closes under normalization,
    the 'small-storm events are intrinsically harder targets' story wins over the
    'bracketing symmetry' story.
    """
    print("\n\n# Analysis 2 — normalized RMSE (bracketing vs intrinsic difficulty)\n")
    elevation = load_static_elevation()

    def per_bin_normalized(prefix: str):
        """For a holdout sub-experiment, per held-out event: mean 2D RMSE across
        the 3 (deterministic) seeds, max depth, normalized RMSE."""
        exps = [e for e in EXP_NAMES if e.startswith(prefix)]
        # held-out test set is deterministic across seeds; take it from the first.
        test_ids = splits_by_exp[exps[0]]["test_event_ids"]
        rows = []
        for ev in test_ids:
            seed_vals = [parsed_by_exp[e]["per_event_2d"][ev]
                         for e in exps if ev in parsed_by_exp[e]["per_event_2d"]]
            if not seed_vals:
                continue
            rmse_2d = mean(seed_vals)
            md = event_max_depth(ev, elevation)
            rows.append((ev, rmse_2d, md, rmse_2d / md))
        return rows

    bins = {}
    for prefix, b in (("p1b_h46_", "4-6"), ("p1b_h24_", "2-4")):
        rows = per_bin_normalized(prefix)
        bins[b] = rows
        print(f"## Holdout {b}\" — per held-out event "
              f"(2D RMSE = mean across 3 seeds)\n")
        print("| event | total in | 2D RMSE (m) | max depth (m) | norm RMSE |")
        print("|---|---|---|---|---|")
        for ev, rmse_2d, md, nrm in sorted(rows, key=lambda r: r[0]):
            print(f"| event_{ev} | {intensity[ev][0]:.2f} | {rmse_2d:.4f} | "
                  f"{md:.3f} | {nrm:.4f} |")
        print()

    print("## Summary — does the gap close under normalization?\n")
    print("| Holdout bin | mean 2D RMSE (m) | mean max depth (m) | mean norm RMSE |")
    print("|---|---|---|---|")
    summary = {}
    for b in ("4-6", "2-4"):
        rows = bins[b]
        abs_rmse = mean(r[1] for r in rows)
        md = mean(r[2] for r in rows)
        nrm = mean(r[3] for r in rows)
        summary[b] = (abs_rmse, md, nrm)
        print(f"| {b}\" | {abs_rmse:.4f} | {md:.3f} | {nrm:.4f} |")

    abs_gap = summary["2-4"][0] - summary["4-6"][0]
    nrm_gap = summary["2-4"][2] - summary["4-6"][2]
    abs_ratio = summary["2-4"][0] / summary["4-6"][0]
    nrm_ratio = summary["2-4"][2] / summary["4-6"][2]
    print(f"\n- Absolute gap (2-4\" − 4-6\"): **{abs_gap:+.4f} m** "
          f"({abs_ratio:.2f}× harder)")
    print(f"- Normalized gap (2-4\" − 4-6\"): **{nrm_gap:+.4f}** "
          f"({nrm_ratio:.2f}× harder)")
    print("\nVerdict: if the normalized ratio collapses toward ~1.0× the headline "
          "should flip to 'small-storm events are intrinsically harder targets'; "
          "if the 2-4\" holdout stays markedly worse even per-unit-depth, the "
          "bracketing-symmetry story holds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", choices=["0", "1", "2", "all"], default="all",
        help="0 = original headline tables only; 1/2 = that follow-up analysis; "
             "all = headline + analyses 1 and 2 (default).",
    )
    args = parser.parse_args()

    intensity = load_intensity()

    # Parse every inference log + split once; reuse for the headline tables and
    # the follow-up analyses below.
    parsed_by_exp = {exp: parse_inference_out(_find_out_file(exp)) for exp in EXP_NAMES}
    splits_by_exp = {exp: load_splits(exp) for exp in EXP_NAMES}

    all_rows = []
    summary = {}

    for exp in EXP_NAMES:
        parsed = parsed_by_exp[exp]
        splits = splits_by_exp[exp]
        mode = splits["mode"]
        holdout = splits["holdout_bin"]
        test_ids = splits["test_event_ids"]

        # Headline: mean over rollout steps of "Overall Mean RMSE — 2D nodes".
        step_means = parsed["step_2d_means"]
        rollout_mean = mean(step_means) if step_means else float("nan")
        step8 = step_means[-1] if step_means else float("nan")

        # Per-bin breakdown using per-event RMSEs (note: this is overall RMSE
        # not 2D-only; inference.py only prints overall per event, not 2D-only
        # per event. Headline numbers above are 2D-only.).
        bin_to_rmse = defaultdict(list)
        for ev_id in test_ids:
            rmse = parsed["per_event_rmse"].get(ev_id)
            if rmse is None:
                continue
            _, bin_label = intensity[ev_id]
            bin_to_rmse[bin_label].append(rmse)

        per_bin = {b: (mean(v), len(v)) for b, v in bin_to_rmse.items()}

        all_rows.append({
            "exp": exp,
            "mode": mode,
            "holdout_bin": holdout,
            "n_test": len(test_ids),
            "rollout_mean_2d_rmse_m": rollout_mean,
            "step8_2d_rmse_m": step8,
            "step_means_2d": step_means,
        })
        summary[exp] = {
            "rollout_mean": rollout_mean,
            "step8": step8,
            "per_bin_overall": per_bin,
        }

    if args.analysis in ("0", "all"):
        # Print headline markdown.
        print("# UrbanFlood v8 — 2D RMSE per experiment")
        print()
        print("Headline = mean across the 8 rollout steps (5..40 min) of "
              "`Overall Mean RMSE — 2D nodes`.")
        print()
        print("| Exp | mode | holdout bin | n_test | 2D mean RMSE (m) | step-8 2D RMSE (m) |")
        print("|---|---|---|---|---|---|")
        for row in all_rows:
            print(f"| {row['exp']} | {row['mode']} | "
                  f"{row['holdout_bin'] or '—'} | {row['n_test']} | "
                  f"**{row['rollout_mean_2d_rmse_m']:.4f}** | "
                  f"{row['step8_2d_rmse_m']:.4f} |")

        # Aggregate by sub-experiment (mean ± std across 3 seeds).
        print()
        print("## Per-sub-experiment mean ± std across 3 seeds")
        print()
        print("| Sub-exp | 2D mean RMSE (m) | step-8 2D RMSE (m) |")
        print("|---|---|---|")
        for prefix, label in (("p1a_", "Phase 1A — stratified 80/20"),
                              ("p1b_h46_", "Phase 1B-1 — holdout 4-6\""),
                              ("p1b_h24_", "Phase 1B-2 — holdout 2-4\"")):
            rs = [r for r in all_rows if r["exp"].startswith(prefix)]
            rolls = [r["rollout_mean_2d_rmse_m"] for r in rs]
            s8 = [r["step8_2d_rmse_m"] for r in rs]
            print(f"| {label} | {mean(rolls):.4f} ± {stdev(rolls):.4f} | "
                  f"{mean(s8):.4f} ± {stdev(s8):.4f} |")

        # Save CSV.
        out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "v8_results.csv")
        with open(out_csv, "w") as f:
            w = csv.writer(f)
            w.writerow(["exp", "mode", "holdout_bin", "n_test",
                        "rollout_mean_2d_rmse_m", "step8_2d_rmse_m"]
                       + [f"step{i+1}_2d_rmse_m" for i in range(8)])
            for row in all_rows:
                w.writerow([row["exp"], row["mode"], row["holdout_bin"] or "",
                            row["n_test"], row["rollout_mean_2d_rmse_m"],
                            row["step8_2d_rmse_m"]] + row["step_means_2d"])
        print(f"\nSaved per-experiment table -> {out_csv}")

        # Phase 1A per-bin (using per-event overall RMSEs as a diagnostic — not 2D-only).
        print("\n## Phase 1A per-bin (overall RMSE per event, n events in bracket)")
        p1a_bin_to_rmses = defaultdict(list)
        for exp in ("p1a_s0", "p1a_s1", "p1a_s2"):
            for b, (rmse, n) in summary[exp]["per_bin_overall"].items():
                p1a_bin_to_rmses[b].extend([rmse] * n)
        print("| Bin | n events (pooled across 3 seeds) | mean overall RMSE (m) |")
        print("|---|---|---|")
        for b in ("0-2", "2-4", "4-6", "6-8", "8-10"):
            if b in p1a_bin_to_rmses:
                v = p1a_bin_to_rmses[b]
                print(f"| {b}\" | {len(v)} | {mean(v):.4f} |")

        # Interpolation gap (Phase 1B vs Phase 1A on the same bin).
        print("\n## Interpolation gap")
        print("Comparing held-out-bin RMSE under bracketing (Phase 1B) vs the "
              "same bin's RMSE under stratified split (Phase 1A).")
        print()
        print("| Held-out bin | Phase 1A bin RMSE (overall) | Phase 1B mean RMSE (overall) | gap (1B − 1A) |")
        print("|---|---|---|---|")
        for prefix, b in (("p1b_h46_", "4-6"), ("p1b_h24_", "2-4")):
            # Phase 1B "overall" per-event mean (averaged across all test events of
            # all 3 seeds — same test set since holdout is deterministic).
            rs = []
            for exp in [e for e in EXP_NAMES if e.startswith(prefix)]:
                rs.extend(parsed_by_exp[exp]["per_event_rmse"].values())
            p1b_mean = mean(rs)
            # Phase 1A: same bin's per-event RMSE.
            p1a_vals = p1a_bin_to_rmses.get(b, [])
            p1a_mean = mean(p1a_vals) if p1a_vals else float("nan")
            gap = p1b_mean - p1a_mean
            print(f"| {b}\" | {p1a_mean:.4f} | {p1b_mean:.4f} | "
                  f"{gap:+.4f} ({100 * gap / p1a_mean:+.1f}%) |")

    if args.analysis in ("1", "all"):
        analysis_1(parsed_by_exp, intensity, splits_by_exp)
    if args.analysis in ("2", "all"):
        analysis_2(parsed_by_exp, intensity, splits_by_exp)


if __name__ == "__main__":
    main()
