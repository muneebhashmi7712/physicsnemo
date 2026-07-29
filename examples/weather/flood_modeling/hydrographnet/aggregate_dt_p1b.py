#!/usr/bin/env python3
"""dt / 20-min re-evaluation on the HELD-OUT-BIN (p1b) design — figI's twin.

Why this exists
---------------
`aggregate_dt_fullevent.py` scores the **p1a** split, where every rainfall bin is
present in training (City 1 train = 7/16/20/12 across bins) and the test set is a
13-event stratified sample.  It therefore cannot say anything about the regime the
model never saw — its own CONCLUSIONS.md records exactly that limitation.

figI (`holdout_fullevent/figI_lmc_rel_city{1,2}.png`) is built on **p1b**: one whole
rainfall bin is deleted from training and the model is tested only on that bin.  This
script scores the coarse (20-min) p1b runs alongside the existing fine (5-min) ones so
the only difference between the two figures is temporal resolution.

Design differences from the p1a aggregator
------------------------------------------
* In p1b a bucket IS a bin (`p1b_h46` -> 4-6"), so the "bins within a bucket" analysis
  of the p1a script is degenerate.  We emit per-event rows and a per-bucket summary and
  leave the narrative to the plotting layer.
* Coarse is available at BOTH seeds here (p1a had coarse at seed 0 only), so the
  fine/coarse pairing is checked at each seed rather than only at seed 0.

Everything about the time alignment is inherited unchanged from
`aggregate_dt_fullevent.py` -- BLOCK=4, N_WARMUP=2, FINE_OFFSET=6, and
`coarse [i0, i0+K) == fine [4*i0+6, 4*i0+6+4K)` with the remainder-absorbing final
coarse block dropped.  `matched_full` is the headline horizon: the coarse warm-up
consumes 40 min of physical time against the fine 10 min, so `own_full` is NOT
physically matched and must not be compared across resolutions.

Usage:  python aggregate_dt_p1b.py
"""
import argparse
import os

import aggregate_dt_fullevent as dtf
from aggregate_fullevent import _write_csv, _mean

# ---------------------------------------------------------------- p1b config
# bucket -> model.  A bucket is one held-out bin.
BUCKETS = {
    "p1b_h02": "Model_1",
    "p1b_h24": "Model_1",
    "p1b_h46": "Model_1",
    "p1b_hge6": "Model_1",
    "p1b_h02_m2": "Model_2",
    "p1b_h24_m2": "Model_2",
    "p1b_h46_m2": "Model_2",
    "p1b_h68_m2": "Model_2",
    "p1b_h810_m2": "Model_2",
}

# bucket suffix -> the bin label figI uses on its x-axis
BIN_OF_BUCKET = {
    "p1b_h02": "0-2", "p1b_h24": "2-4", "p1b_h46": "4-6", "p1b_hge6": ">=6",
    "p1b_h02_m2": "0-2", "p1b_h24_m2": "2-4", "p1b_h46_m2": "4-6",
    "p1b_h68_m2": "6-8", "p1b_h810_m2": "8-10",
}

RUNS = [("fine", 0), ("fine", 1), ("coarse", 0), ("coarse", 1)]
ARMS = ("nolc", "lmc")

OUTDIR = "/home/hpc/iwi5/iwi5416h/hydrographnet/dt_fullevent_p1b"

# Rebind the module globals the inherited helpers close over by name.
dtf.BUCKETS = BUCKETS
dtf.RUNS = RUNS


# ---------------------------------------------------------------- extra checks
def extra_checks(store, rows):
    """p1b-specific integrity, on top of dtf.check_integrity."""
    problems = []

    # (A) fine and coarse must be the same events at EVERY seed (p1a only checked s0)
    for seed in sorted({s for _, s in RUNS}):
        for bucket in BUCKETS:
            kf, kc = ("fine", seed, bucket, "nolc"), ("coarse", seed, bucket, "nolc")
            if kf in store and kc in store:
                a, b = sorted(store[kf]), sorted(store[kc])
                if a != b:
                    problems.append(
                        f"[A] fine/coarse event mismatch s{seed} {bucket}: {a} vs {b}")

    # (B) the design guard: in p1b every test event must lie in its own held-out bin.
    # City 1's top bucket is an open-ended ">=6", which legitimately merges the 6-8
    # and 8-10 rain bins, so it is checked as a membership rather than an equality.
    GE6_OK = {"6-8", "8-10"}
    for r in rows:
        ok = (r["rain_bin"] in GE6_OK if r["holdout_bin"] == ">=6"
              else r["holdout_bin"] == r["rain_bin"])
        if not ok:
            problems.append(
                f"[B] {r['bucket']} event_{r['event_id']}: holdout_bin="
                f"{r['holdout_bin']} but rain_bin={r['rain_bin']} — split built wrong")
            break

    # (C) matched_full must exist for every (res, seed, bucket, arm, event)
    have = {(r["resolution"], r["seed"], r["bucket"], r["arm"], r["event_id"])
            for r in rows if r["horizon_kind"] == "matched_full"}
    allev = {(r["resolution"], r["seed"], r["bucket"], r["arm"], r["event_id"])
             for r in rows}
    if allev - have:
        miss = sorted(allev - have)[:5]
        problems.append(f"[C] {len(allev - have)} rows have no matched_full, e.g. {miss}")

    # (D) matched windows must cover the identical physical span across resolutions
    span = {}
    for r in rows:
        if r["horizon_kind"] != "matched_full":
            continue
        span.setdefault((r["seed"], r["bucket"], r["arm"], r["event_id"]), {})[
            r["resolution"]] = float(r["horizon_hours"])
    for k, v in span.items():
        if "fine" in v and "coarse" in v and abs(v["fine"] - v["coarse"]) > 1e-6:
            problems.append(f"[D] {k}: matched span differs "
                            f"fine={v['fine']}h coarse={v['coarse']}h")
            break
    return problems


def build_summary(rows):
    """Per (model, bucket, resolution, seed): LMC-vs-noLMC at each horizon."""
    idx = {}
    for r in rows:
        k = (r["resolution"], r["seed"], r["bucket"], r["horizon_kind"], r["arm"])
        idx.setdefault(k, {})[r["event_id"]] = r
    out = []
    for (res, seed, bucket, kind, arm), evs in sorted(idx.items()):
        if arm != "nolc":
            continue
        lmc = idx.get((res, seed, bucket, kind, "lmc"), {})
        common = sorted(set(evs) & set(lmc))
        if not common:
            continue
        pct, wins = [], 0
        for ev in common:
            a = float(evs[ev]["rmse_2d_m"])
            b = float(lmc[ev]["rmse_2d_m"])
            if a > 0:
                pct.append(100.0 * (b / a - 1.0))
                wins += int(b < a)
        out.append({
            "model": BUCKETS[bucket], "bucket": bucket,
            "holdout_bin": BIN_OF_BUCKET[bucket], "resolution": res, "seed": seed,
            "horizon_kind": kind, "n_events": len(common),
            "rmse_nolc_mean": round(_mean([float(evs[e]["rmse_2d_m"])
                                           for e in common]), 6),
            "rmse_lmc_mean": round(_mean([float(lmc[e]["rmse_2d_m"])
                                          for e in common]), 6),
            "rel_pct_mean": round(_mean(pct), 2) if pct else "",
            "rel_pct_median": round(sorted(pct)[len(pct) // 2], 2) if pct else "",
            "lmc_wins": wins,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-root", default=dtf.RES_ROOT)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    intensity = dtf.load_intensity()
    peak_rain = dtf.load_peak_rain()
    store = dtf.load_runs(args.res_root)
    if not store:
        raise SystemExit(f"no runs found under {args.res_root}")
    print(f"loaded {len(store)} (res, seed, bucket, arm) runs")

    problems = dtf.check_integrity(store, intensity)

    kev, i0_onset, K_common, K_onset, fine_only_K = dtf.build_windows(store)
    rows = dtf.build_per_event(store, intensity, peak_rain, kev, i0_onset,
                               K_common, K_onset, fine_only_K)

    # attach the figI-compatible bin label
    for r in rows:
        r["holdout_bin"] = BIN_OF_BUCKET[r["bucket"]]

    problems += extra_checks(store, rows)
    for p in problems:
        print("INTEGRITY:", p)

    summary = build_summary(rows)
    _write_csv(os.path.join(args.outdir, "per_event_dt_p1b.csv"), rows)
    _write_csv(os.path.join(args.outdir, "summary_dt_p1b.csv"), summary)

    print(f"\nK_common={K_common} coarse steps, K_onset={K_onset} coarse steps")
    print(f"wrote {len(rows)} per-event rows -> {args.outdir}/per_event_dt_p1b.csv")

    print("\n=== LMC effect by bucket (matched_full, mean rel%) ===")
    print(f"{'bucket':14s} {'bin':5s} {'n':>3s} | "
          f"{'fine s0':>9s} {'fine s1':>9s} | {'crse s0':>9s} {'crse s1':>9s}")
    for bucket in BUCKETS:
        cells = {}
        for s in summary:
            if s["bucket"] == bucket and s["horizon_kind"] == "matched_full":
                cells[(s["resolution"], s["seed"])] = s
        if not cells:
            continue
        n = max(c["n_events"] for c in cells.values())
        f = lambda r, sd: (f"{cells[(r, sd)]['rel_pct_mean']:+9.1f}"
                           if (r, sd) in cells else f"{'--':>9s}")
        print(f"{bucket:14s} {BIN_OF_BUCKET[bucket]:5s} {n:3d} | "
              f"{f('fine', 0)} {f('fine', 1)} | {f('coarse', 0)} {f('coarse', 1)}")

    if problems:
        print(f"\n!! {len(problems)} integrity problem(s) — see INTEGRITY lines above")


if __name__ == "__main__":
    main()
