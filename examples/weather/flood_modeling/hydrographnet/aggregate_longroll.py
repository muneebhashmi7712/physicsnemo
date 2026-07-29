#!/usr/bin/env python3
"""Aggregate long-rollout LMC vs no-LMC results on fully held-out rainfall bins.

Reads the per-event metrics.json written by inference.py under
/home/woody/iwi5/iwi5416h/urbanflood/longroll/<BUCKET>__<arm>/ and emits:

  results_urbanflood_longroll_per_event.csv   per-event rows (both arms, both horizons)
  summary_longroll_lmc.csv                    bucket-level LMC vs nolc + relative diff

Each event's ``rmse_2d_series`` (per-step 2D RMSE) lets any shorter horizon be
recovered as a prefix of the long rollout, so the common horizon and each
bucket's own maximum come from the same inference run.

Context: the standard 8-step metric scores only timesteps 0-9, during which the
water field is completely static for 35-50 % of events. The ``frozen_window``
column carries that flag forward so it stays visible here.
"""

import argparse
import csv
import json
import math
import os

RES_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/longroll"
DATA_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/data"
INTENSITY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "event_intensity.csv")

# bucket -> (held-out bin label, model)
BUCKETS = {
    "p1b_h02":     ("0-2",  "Model_1"),
    "p1b_h24":     ("2-4",  "Model_1"),
    "p1b_h46":     ("4-6",  "Model_1"),
    "p1b_hge6":    (">=6",  "Model_1"),
    "p1b_h02_m2":  ("0-2",  "Model_2"),
    "p1b_h24_m2":  ("2-4",  "Model_2"),
    "p1b_h46_m2":  ("4-6",  "Model_2"),
    "p1b_h68_m2":  ("6-8",  "Model_2"),
    "p1b_h810_m2": ("8-10", "Model_2"),
}
ARMS = ("nolc", "lmc")
DT_MIN = 5  # minutes per rollout step


def load_intensity():
    """event_intensity.csv -> {(model, event_id): (rain_bin, total_inches, T)}."""
    out = {}
    with open(INTENSITY_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "train":
                continue
            out[(r["model_name"], int(r["event_id"]))] = (
                r["bin"], float(r["total_inches"]), int(r["num_timesteps"])
            )
    return out


def in_window_rain(model, event_id, n_steps=10):
    """Total rainfall over the first ``n_steps`` timesteps (inches).

    Rain is spatially uniform, so node_idx 0 carries the series. Returns None if
    the event CSV is unreadable.
    """
    path = os.path.join(DATA_ROOT, model, "train", f"event_{event_id}",
                        "2d_nodes_dynamic_all.csv")
    if not os.path.exists(path):
        return None
    total = 0.0
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            if int(row[0]) >= n_steps:
                break
            if row[1] == "0":
                total += float(row[2])
    return total


def prefix_rmse(series, horizon):
    """Mean 2D RMSE over the first ``horizon`` rollout steps."""
    vals = [v for v in series[:horizon] if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-horizon", type=int, default=92,
                    help="steps for the cross-bucket comparable horizon")
    ap.add_argument("--outdir", default="/home/hpc/iwi5/iwi5416h/hydrographnet/longroll_lmc")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    intensity = load_intensity()
    rain_cache = {}
    per_event_rows = []
    # (bucket, horizon) -> {arm: [rmse, ...]} keyed by event for pairing
    paired = {}

    for bucket, (bin_label, model) in BUCKETS.items():
        loaded = {}
        for arm in ARMS:
            mpath = os.path.join(RES_ROOT, f"{bucket}__{arm}", "metrics.json")
            if not os.path.exists(mpath):
                print(f"  MISSING {mpath}")
                continue
            with open(mpath) as fh:
                loaded[arm] = json.load(fh)
        if len(loaded) < 2:
            print(f"skip {bucket}: have arms {sorted(loaded)}")
            continue

        rollouts = {a: int(loaded[a]["config"]["rollout_length"]) for a in loaded}
        if len(set(rollouts.values())) != 1:
            print(f"WARNING {bucket}: arms ran different rollouts {rollouts}")
        max_h = min(rollouts.values())
        horizons = {"common": min(args.common_horizon, max_h), "max": max_h}

        for arm, blob in loaded.items():
            for ev_str, rec in blob["per_hydrograph"].items():
                # inference.py keys these by the dataset's event_id, which is
                # the directory name ("event_24"), not a bare integer.
                ev = int(str(ev_str).replace("event_", ""))
                series = rec.get("rmse_2d_series")
                if not series:
                    print(f"  {bucket}/{arm}/event_{ev}: no rmse_2d_series "
                          f"(inference.py predates the patch)")
                    continue
                rbin, inches, n_t = intensity.get((model, ev), ("?", float("nan"), -1))
                key = (model, ev)
                if key not in rain_cache:
                    rain_cache[key] = in_window_rain(model, ev)
                rain10 = rain_cache[key]

                for hname, h in horizons.items():
                    rmse = prefix_rmse(series, h)
                    full = (hname == "max")
                    per_event_rows.append({
                        "model": model,
                        "holdout_bin": bin_label,
                        "bucket": bucket,
                        "arm": arm,
                        "seed": 0,
                        "horizon_kind": hname,
                        "horizon_steps": h,
                        "horizon_min": h * DT_MIN,
                        "event_id": ev,
                        "rain_bin": rbin,
                        "total_inches": inches,
                        "event_timesteps": n_t,
                        "rain_first10_in": (
                            "" if rain10 is None else f"{rain10:.4f}"),
                        "frozen_window": (
                            "" if rain10 is None else int(rain10 == 0.0)),
                        "rmse_2d_m": rmse,
                        "rmse_2d_mm": rmse * 1000.0,
                        # Metrics below are computed by inference.py over the
                        # FULL rollout only; they cannot be re-derived for a
                        # prefix, so they are blank on the common-horizon rows.
                        "nse": rec["mean_nse"] if full else "",
                        "csi_005": rec["mean_csi_005"] if full else "",
                        "csi_030": rec["mean_csi_030"] if full else "",
                        "scalefree_err_pct": rec["scalefree_err_pct"] if full else "",
                        "rmse_over_sigma": rec["rmse_over_sigma"] if full else "",
                    })
                    paired.setdefault((bucket, hname), {}).setdefault(arm, {})[ev] = rmse

    if not per_event_rows:
        print("No results found — did the jobs finish?")
        return

    ev_csv = os.path.join(args.outdir, "results_urbanflood_longroll_per_event.csv")
    with open(ev_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_event_rows[0].keys()))
        w.writeheader()
        w.writerows(per_event_rows)
    print(f"wrote {ev_csv}  ({len(per_event_rows)} rows)")

    # ---- bucket-level summary: LMC vs nolc on the SAME events ----
    summary = []
    for (bucket, hname), arms in sorted(paired.items()):
        if set(arms) != set(ARMS):
            continue
        common_ev = sorted(set(arms["lmc"]) & set(arms["nolc"]))
        if not common_ev:
            continue
        bin_label, model = BUCKETS[bucket]
        lmc = [arms["lmc"][e] for e in common_ev]
        nolc = [arms["nolc"][e] for e in common_ev]
        m_l, m_n = sum(lmc) / len(lmc), sum(nolc) / len(nolc)
        n_frozen = sum(
            1 for e in common_ev
            if rain_cache.get((model, e)) == 0.0
        )
        h = next(r["horizon_steps"] for r in per_event_rows
                 if r["bucket"] == bucket and r["horizon_kind"] == hname)
        summary.append({
            "model": model,
            "holdout_bin": bin_label,
            "bucket": bucket,
            "horizon_kind": hname,
            "horizon_steps": h,
            "horizon_hours": round(h * DT_MIN / 60.0, 2),
            "n_events": len(common_ev),
            "n_frozen_8step": n_frozen,
            "rmse_2d_nolc_m": round(m_n, 6),
            "rmse_2d_lmc_m": round(m_l, 6),
            "abs_diff_m": round(m_l - m_n, 6),
            "rel_diff_pct": round(100.0 * (m_l - m_n) / m_n, 2) if m_n else float("nan"),
            "lmc_better_on": sum(1 for a, b in zip(lmc, nolc) if a < b),
        })

    sum_csv = os.path.join(args.outdir, "summary_longroll_lmc.csv")
    with open(sum_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {sum_csv}  ({len(summary)} rows)\n")

    hdr = (f"{'model':>8} {'bin':>5} {'horizon':>12} {'n':>3} {'froz':>4} "
           f"{'nolc':>8} {'lmc':>8} {'rel %':>8} {'lmc<nolc':>9}")
    print(hdr); print("-" * len(hdr))
    for r in summary:
        if r["horizon_kind"] != "common":
            continue
        print(f"{r['model']:>8} {r['holdout_bin']:>5} "
              f"{str(r['horizon_steps'])+'st':>12} {r['n_events']:3d} "
              f"{r['n_frozen_8step']:4d} {r['rmse_2d_nolc_m']:8.4f} "
              f"{r['rmse_2d_lmc_m']:8.4f} {r['rel_diff_pct']:+8.2f} "
              f"{str(r['lmc_better_on'])+'/'+str(r['n_events']):>9}")


if __name__ == "__main__":
    main()
