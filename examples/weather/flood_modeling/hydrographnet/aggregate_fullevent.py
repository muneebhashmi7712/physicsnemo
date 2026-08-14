#!/usr/bin/env python3
"""Consolidated FULL-EVENT held-out-bin analysis (supersedes the 8-step
`holdout_uf_scalefree/` files and the bin-min `longroll_lmc/`).

Reads the per-event metrics.json written by the patched inference.py under
/home/woody/iwi5/iwi5416h/urbanflood/fullevent/<BUCKET>__<arm>/, where each
event carries STEP-ALIGNED per-step series for every metric:
    rmse_2d_series, nse_2d_series (NaN on near-dry steps), csi005_2d_series,
    csi030_2d_series, rmse_over_sigma_series, sd_gt_series
so any common horizon H is recoverable as a prefix of the full rollout — for
ALL metrics, not just RMSE.

Answers the three study questions on real flooding (not the frozen 8-step window):
  Q1  RMSE for each UNSEEN bin, at (a) a common horizon (global min held-out
      event length) for strict cross-bin fairness, and (b) each event's OWN full
      horizon.
  Q2  LMC vs no-LMC per bucket: relative RMSE diff, Spearman vs bin midpoint and
      vs total_inches magnitude, paired Wilcoxon.
  Q3  Does error itself track the bin? All metrics (RMSE, NSE, CSI, scalefree,
      rmse_over_sigma) at the common horizon. NSE is scale-free ⇒ cross-bin
      comparable.
  QA  rollout_rain_in = rainfall actually delivered over the full rollout;
      asserted ≈ total_inches (bin ↔ delivered-rain audit).

Outputs (default --outdir /home/hpc/iwi5/iwi5416h/hydrographnet/holdout_fullevent):
  per_event_fullevent.csv     one row per (bucket, arm, event, horizon_kind)
  summary_fullevent.csv       bucket-level LMC vs nolc, per horizon
  trends_fullevent.md         Q2/Q3 trend stats + Q1 table + QA summary
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

try:
    from scipy import stats as _stats
except Exception:  # scipy optional; Wilcoxon/Spearman then reported as NaN
    _stats = None

RES_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/fullevent"
DATA_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/data"
INTENSITY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "event_intensity.csv")
ONSET_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "event_onset.csv")  # per-event rain onset (5% of peak)
N_WARMUP = 2   # n_time_steps; the rollout starts after this input window
DT_MIN = 5     # minutes per step

# bucket -> (held-out bin label, model, bin midpoint for trend tests)
BUCKETS = {
    "p1b_h02":     ("0-2",  "Model_1", 1),
    "p1b_h24":     ("2-4",  "Model_1", 3),
    "p1b_h46":     ("4-6",  "Model_1", 5),
    "p1b_hge6":    (">=6",  "Model_1", 8),
    "p1b_h02_m2":  ("0-2",  "Model_2", 1),
    "p1b_h24_m2":  ("2-4",  "Model_2", 3),
    "p1b_h46_m2":  ("4-6",  "Model_2", 5),
    "p1b_h68_m2":  ("6-8",  "Model_2", 7),
    "p1b_h810_m2": ("8-10", "Model_2", 9),
}
ARMS = ("nolc", "lmc")


# ---------------------------------------------------------------- helpers
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


def load_onset():
    """event_onset.csv -> {(model, event_id): onset_timestep} (absolute step of
    the first timestep whose rainfall >= 5% of the event's peak)."""
    out = {}
    if not os.path.exists(ONSET_CSV):
        return out
    with open(ONSET_CSV) as fh:
        for r in csv.DictReader(fh):
            out[(r["model_name"], int(r["event_id"]))] = int(r["onset_timestep"])
    return out


def head_rain(model, event_id, upto=10):
    """Node-0 rain (inches) over the first `upto` timesteps, split at N_WARMUP.
    Reads only the head of the CSV (breaks at timestep >= upto), so it is cheap.
    Returns (warm, rain10): warm = steps [0:N_WARMUP] (the input window, NOT
    rolled out); rain10 = steps [0:upto] (the OLD 8-step scored window).
    rollout rain is then total_inches - warm (total_inches sums the whole event).
    Rain is spatially uniform, so node_idx 0 carries the series.
    """
    path = os.path.join(DATA_ROOT, model, "train", f"event_{event_id}",
                        "2d_nodes_dynamic_all.csv")
    if not os.path.exists(path):
        return None, None
    warm, rain10 = 0.0, 0.0
    with open(path) as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            t = int(row[0])
            if t >= upto:
                break
            if row[1] != "0":
                continue
            v = float(row[2])
            rain10 += v
            if t < N_WARMUP:
                warm += v
    return warm, rain10


def _finite(xs):
    return [v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]


def window_mean(series, start, length):
    """Mean of the finite values in series[start : start+length]."""
    vals = _finite(series[start:start + length])
    return sum(vals) / len(vals) if vals else float("nan")


def prefix_mean(series, h):
    return window_mean(series, 0, h)


def scalefree_from_nse(nse_mean):
    if nse_mean is None or (isinstance(nse_mean, float) and math.isnan(nse_mean)):
        return float("nan")
    return 100.0 * math.sqrt(max(0.0, 1.0 - nse_mean))


def spearman(x, y):
    if _stats is None or len(x) < 3:
        return float("nan"), float("nan")
    rho, p = _stats.spearmanr(x, y)
    return float(rho), float(p)


def wilcoxon(diffs):
    d = [v for v in diffs if v != 0 and not math.isnan(v)]
    if _stats is None or len(d) < 3:
        return float("nan")
    try:
        return float(_stats.wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-horizon", type=int, default=None,
                    help="Cross-bucket comparable horizon (steps). Default = the "
                         "global minimum held-out event length across all events.")
    ap.add_argument("--outdir",
                    default="/home/hpc/iwi5/iwi5416h/hydrographnet/holdout_fullevent")
    ap.add_argument("--res-root", default=RES_ROOT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    intensity = load_intensity()
    onset = load_onset()

    # ---- pass 1: load every event's series, find the global common horizons ----
    # store[(bucket, arm)] = {event_id: record};  onset_idx_map = per-event window start
    store = defaultdict(dict)
    onset_idx_map = {}
    min_len = None
    K_onset = None   # global min post-onset window supported by every event
    for bucket, (bin_label, model, _mid) in BUCKETS.items():
        for arm in ARMS:
            mpath = os.path.join(args.res_root, f"{bucket}__{arm}", "metrics.json")
            if not os.path.exists(mpath):
                print(f"  MISSING {mpath}")
                continue
            with open(mpath) as fh:
                blob = json.load(fh)
            for ev_str, rec in blob.get("per_hydrograph", {}).items():
                ev = int(str(ev_str).replace("event_", ""))
                series = rec.get("rmse_2d_series")
                if not series:
                    print(f"  {bucket}/{arm}/event_{ev}: no rmse_2d_series")
                    continue
                store[(bucket, arm)][ev] = rec
                L = len(series)
                min_len = L if min_len is None else min(min_len, L)
                # series index 0 == absolute event step N_WARMUP, so the onset's
                # series index is (onset - N_WARMUP), floored at 0.
                oidx = max(0, onset.get((model, ev), 0) - N_WARMUP)
                onset_idx_map[(model, ev)] = oidx
                post = L - oidx
                K_onset = post if K_onset is None else min(K_onset, post)

    if not store:
        print("No results found — did the full-event jobs finish?")
        return

    H_common = args.common_horizon or min_len
    if K_onset is None:
        K_onset = H_common
    print(f"Global common horizon = {H_common} steps ({H_common*DT_MIN/60:.1f} h); "
          f"min event length = {min_len}; onset-aligned window K = {K_onset} steps "
          f"({K_onset*DT_MIN/60:.1f} h)")

    rain_cache = {}
    per_event_rows = []
    # paired[(bucket, hkind)][arm][event] = {metric: value}
    paired = defaultdict(lambda: defaultdict(dict))

    for bucket, (bin_label, model, _mid) in BUCKETS.items():
        for arm in ARMS:
            recs = store.get((bucket, arm), {})
            for ev, rec in recs.items():
                rbin, inches, n_t = intensity.get((model, ev), ("?", float("nan"), -1))
                if (model, ev) not in rain_cache:
                    rain_cache[(model, ev)] = head_rain(model, ev)
                warm_rain, rain10 = rain_cache[(model, ev)]
                # rain delivered over the rollout = whole-event total minus the
                # warm-up window (which is input, not rolled out).
                roll_rain = (None if (warm_rain is None or inches != inches)
                             else max(0.0, inches - warm_rain))
                rmse_s = rec["rmse_2d_series"]
                nse_s = rec.get("nse_2d_series", [])
                c05_s = rec.get("csi005_2d_series", [])
                c30_s = rec.get("csi030_2d_series", [])
                ros_s = rec.get("rmse_over_sigma_series", [])
                full_h = len(rmse_s)
                # 8-step frozen flag: was the OLD scored window (steps 0..9) rain-free?
                frozen8 = "" if rain10 is None else int(rain10 == 0.0)
                oidx = onset_idx_map.get((model, ev), 0)

                # Three (start, length) windows:
                #   common — first H_common steps (equal wall-clock; high bins may
                #            still be pre-onset here)
                #   onset  — K_onset steps starting at each event's rain onset
                #            (equal storm progress; the fair cross-bin severity view)
                #   full   — the whole event (within-bin comparisons only)
                horizons = (
                    ("common", 0,    min(H_common, full_h)),
                    ("onset",  oidx, min(K_onset, full_h - oidx)),
                    ("full",   0,    full_h),
                )
                for hkind, start, length in horizons:
                    nse_h = window_mean(nse_s, start, length) if nse_s else float("nan")
                    row = {
                        "model": model, "holdout_bin": bin_label, "bucket": bucket,
                        "arm": arm, "seed": 0, "event_id": ev,
                        "rain_bin": rbin, "total_inches": inches,
                        "event_timesteps": n_t,
                        "onset_timestep": onset.get((model, ev), ""),
                        "rollout_rain_in": ("" if roll_rain is None
                                            else round(roll_rain, 4)),
                        "rain_consistency_pct": ("" if not roll_rain or not inches
                                                 else round(100.0 * roll_rain / inches, 1)),
                        "frozen_8step": frozen8,
                        "horizon_kind": hkind, "horizon_start": start,
                        "horizon_steps": length,
                        "horizon_hours": round(length * DT_MIN / 60.0, 2),
                        "rmse_2d_m": window_mean(rmse_s, start, length),
                        "nse": nse_h,
                        "scalefree_err_pct": scalefree_from_nse(nse_h),
                        "csi_005": window_mean(c05_s, start, length) if c05_s else float("nan"),
                        "csi_030": window_mean(c30_s, start, length) if c30_s else float("nan"),
                        "rmse_over_sigma": window_mean(ros_s, start, length) if ros_s else float("nan"),
                    }
                    per_event_rows.append(row)
                    paired[(bucket, hkind)][arm][ev] = row

    _write_csv(os.path.join(args.outdir, "per_event_fullevent.csv"), per_event_rows)

    # ---- per-ROLLOUT-STEP detail (for plotting error vs rollout step) ----
    _write_per_step(os.path.join(args.outdir, "per_step_fullevent.csv"),
                    store, intensity, onset)
    _write_per_step_binmean(os.path.join(args.outdir, "per_step_binmean_fullevent.csv"),
                            store)

    # ---- bucket-level summary (paired lmc vs nolc on common events) ----
    summary = _build_summary(paired, rain_cache)
    _write_csv(os.path.join(args.outdir, "summary_fullevent.csv"), summary)

    _write_trends(os.path.join(args.outdir, "trends_fullevent.md"),
                  summary, per_event_rows, H_common, rain_cache)
    _print_headline(summary, H_common)


def _build_summary(paired, rain_cache):
    summary = []
    for (bucket, hkind), arms in sorted(paired.items()):
        if set(arms) != set(ARMS):
            continue
        common_ev = sorted(set(arms["lmc"]) & set(arms["nolc"]))
        if not common_ev:
            continue
        bin_label, model, _mid = BUCKETS[bucket]

        def col(arm, key):
            return [arms[arm][e][key] for e in common_ev]

        rl, rn = col("lmc", "rmse_2d_m"), col("nolc", "rmse_2d_m")
        m_l, m_n = sum(rl) / len(rl), sum(rn) / len(rn)
        h = arms["nolc"][common_ev[0]]["horizon_steps"]
        summary.append({
            "model": model, "holdout_bin": bin_label, "bucket": bucket,
            "bin_mid": _mid, "horizon_kind": hkind, "horizon_steps": h,
            "horizon_hours": round(h * DT_MIN / 60.0, 2),
            "n_events": len(common_ev),
            "rmse_nolc_m": round(m_n, 6), "rmse_lmc_m": round(m_l, 6),
            "rel_diff_pct": round(100.0 * (m_l - m_n) / m_n, 2) if m_n else float("nan"),
            "lmc_better_on": sum(1 for a, b in zip(rl, rn) if a < b),
            "wilcoxon_p": round(wilcoxon([a - b for a, b in zip(rl, rn)]), 4),
            "nse_nolc": round(_mean(col("nolc", "nse")), 4),
            "nse_lmc": round(_mean(col("lmc", "nse")), 4),
            "csi005_nolc": round(_mean(col("nolc", "csi_005")), 4),
            "csi005_lmc": round(_mean(col("lmc", "csi_005")), 4),
            "csi030_nolc": round(_mean(col("nolc", "csi_030")), 4),
            "csi030_lmc": round(_mean(col("lmc", "csi_030")), 4),
            "ros_nolc": round(_mean(col("nolc", "rmse_over_sigma")), 4),
            "ros_lmc": round(_mean(col("lmc", "rmse_over_sigma")), 4),
        })
    return summary


def _mean(xs):
    v = _finite(xs)
    return sum(v) / len(v) if v else float("nan")


def _r(v, nd=6):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return round(v, nd)


def _write_per_step(path, store, intensity, onset):
    """Long-format per-ROLLOUT-STEP detail: one row per (bucket, arm, event, step),
    so you can plot how each metric evolves over the rollout. `step` is the series
    index (0 == absolute event timestep N_WARMUP=2); `abs_timestep` and `time_h` give
    physical time from the event start; `onset_rel_step` = steps since rain onset.
    `nse` is blank on near-dry (unconditioned) steps."""
    keys = ["model", "holdout_bin", "bucket", "arm", "event_id", "rain_bin",
            "total_inches", "step", "abs_timestep", "time_h", "onset_rel_step",
            "rmse_2d_m", "nse", "csi_005", "csi_030", "rmse_over_sigma", "sd_gt"]
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for bucket, (bin_label, model, _mid) in BUCKETS.items():
            for arm in ARMS:
                for ev, rec in sorted(store.get((bucket, arm), {}).items()):
                    rbin, inches, _T = intensity.get((model, ev), ("", "", -1))
                    oidx = max(0, onset.get((model, ev), 0) - N_WARMUP)
                    rmse_s = rec.get("rmse_2d_series", [])
                    nse_s = rec.get("nse_2d_series", [])
                    c05_s = rec.get("csi005_2d_series", [])
                    c30_s = rec.get("csi030_2d_series", [])
                    ros_s = rec.get("rmse_over_sigma_series", [])
                    sd_s = rec.get("sd_gt_series", [])
                    g = lambda s, t: (s[t] if t < len(s) else float("nan"))
                    for t in range(len(rmse_s)):
                        w.writerow([model, bin_label, bucket, arm, ev, rbin, inches,
                                    t, t + N_WARMUP,
                                    round((t + N_WARMUP) * DT_MIN / 60.0, 4), t - oidx,
                                    _r(g(rmse_s, t)), _r(g(nse_s, t)), _r(g(c05_s, t)),
                                    _r(g(c30_s, t)), _r(g(ros_s, t)), _r(g(sd_s, t))])
                        n += 1
    print(f"wrote {path}  ({n} rows)")


def _write_per_step_binmean(path, store):
    """Bin x arm mean of each metric at each step index — one plottable curve per
    (model, held-out bin, arm). At step t the mean is over the events in that bin
    whose rollout reaches t, so `n_events` shrinks as t grows (a bin holds events of
    different lengths); filter on n_events when plotting the tail."""
    acc = defaultdict(lambda: [0.0, 0.0, 0, 0, 0.0])  # sum_rmse, sum_nse, n_nse, n, sum_ros
    for bucket, (bin_label, model, _mid) in BUCKETS.items():
        for arm in ARMS:
            for ev, rec in store.get((bucket, arm), {}).items():
                rmse_s = rec.get("rmse_2d_series", [])
                nse_s = rec.get("nse_2d_series", [])
                ros_s = rec.get("rmse_over_sigma_series", [])
                for t in range(len(rmse_s)):
                    a = acc[(model, bin_label, arm, t)]
                    a[0] += rmse_s[t]; a[3] += 1
                    if t < len(ros_s):
                        a[4] += ros_s[t]
                    if t < len(nse_s) and not (isinstance(nse_s[t], float)
                                               and math.isnan(nse_s[t])):
                        a[1] += nse_s[t]; a[2] += 1
    rows = []
    for (model, bin_label, arm, t), a in sorted(acc.items()):
        rows.append({"model": model, "holdout_bin": bin_label, "arm": arm, "step": t,
                     "time_h": round((t + N_WARMUP) * DT_MIN / 60.0, 4),
                     "n_events": a[3],
                     "mean_rmse_2d_m": _r(a[0] / a[3]),
                     "mean_nse": (_r(a[1] / a[2]) if a[2] else ""),
                     "mean_rmse_over_sigma": _r(a[4] / a[3])})
    _write_csv(path, rows)


def _write_csv(path, rows):
    if not rows:
        print(f"  (no rows for {path})")
        return
    # union of keys preserves first-row order then appends any extras
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(r.get(k), float) and math.isnan(r.get(k)))
                            else r.get(k, "")) for k in keys})
    print(f"wrote {path}  ({len(rows)} rows)")


def _write_trends(path, summary, per_event_rows, H_common, rain_cache):
    by = lambda kind: [s for s in summary if s["horizon_kind"] == kind]
    ons, full, com = by("onset"), by("full"), by("common")
    K = ons[0]["horizon_steps"] if ons else H_common

    lines = ["# Full-event held-out-bin analysis — Q1/Q2/Q3\n",
             f"**Headline view = ONSET-ALIGNED**: every bin scored over the same "
             f"{K} steps ({K*DT_MIN/60:.1f} h) of ACTIVE storm measured from each "
             f"event's own rain onset (5% of peak) — the fair cross-bin severity "
             f"comparison. `common` ({H_common}-step equal wall-clock) and `full` "
             f"(each event's whole length) are in the CSVs; `full` is used for the "
             f"within-bin LMC penalty since it is apples-to-apples inside a bin.\n"]

    # Q1 — onset-aligned
    lines.append("\n## Q1 — RMSE for each UNSEEN bin (onset-aligned, no-LMC arm)\n")
    lines.append("| model | bin | n | RMSE nolc (ft) | NSE nolc | scalefree% | CSI0.05 |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in ons:
        lines.append(f"| {s['model']} | {s['holdout_bin']} | {s['n_events']} | "
                     f"{s['rmse_nolc_m']:.4f} | {s['nse_nolc']:.3f} | "
                     f"{scalefree_from_nse(s['nse_nolc']):.1f} | {s['csi005_nolc']:.3f} |")

    # Q2 — onset + full
    lines.append("\n## Q2 — LMC vs no-LMC per bin (+ = LMC worse)\n")
    lines.append("| model | bin | n | rel% ONSET | Wilcoxon | rel% FULL | Wilcoxon | lmc<nolc full |")
    lines.append("|---|---|---|---|---|---|---|---|")
    fmap = {(s["model"], s["holdout_bin"]): s for s in full}
    for s in ons:
        f = fmap.get((s["model"], s["holdout_bin"]), {})
        lines.append(f"| {s['model']} | {s['holdout_bin']} | {s['n_events']} | "
                     f"{s['rel_diff_pct']:+.1f} | {s['wilcoxon_p']:.3f} | "
                     f"{f.get('rel_diff_pct', float('nan')):+.1f} | "
                     f"{f.get('wilcoxon_p', float('nan')):.3f} | "
                     f"{f.get('lmc_better_on','?')}/{s['n_events']} |")
    for label, view in (("onset", ons), ("full", full)):
        for model in ("Model_1", "Model_2", "pooled"):
            rows = view if model == "pooled" else [s for s in view if s["model"] == model]
            if len(rows) >= 3:
                rho, p = spearman([s["bin_mid"] for s in rows],
                                  [s["rel_diff_pct"] for s in rows])
                lines.append(f"\n- Spearman(LMC rel-diff, bin midpoint) [{label}, {model}] = "
                             f"{rho:+.2f} (p={p:.3f}, n={len(rows)})")

    # Q3 — onset-aligned, all metrics
    lines.append("\n## Q3 — error vs bin, all metrics (onset-aligned, no-LMC arm)\n")
    lines.append("| model | bin | RMSE | NSE | scalefree% | CSI0.05 | CSI0.30 | RMSE/σ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in ons:
        lines.append(f"| {s['model']} | {s['holdout_bin']} | {s['rmse_nolc_m']:.4f} | "
                     f"{s['nse_nolc']:.3f} | {scalefree_from_nse(s['nse_nolc']):.1f} | "
                     f"{s['csi005_nolc']:.3f} | {s['csi030_nolc']:.3f} | {s['ros_nolc']:.3f} |")
    for metric, key, asc in (("RMSE", "rmse_nolc_m", True), ("NSE", "nse_nolc", False)):
        for model in ("Model_1", "Model_2"):
            rows = [s for s in ons if s["model"] == model]
            if len(rows) >= 3:
                rho, p = spearman([s["bin_mid"] for s in rows], [s[key] for s in rows])
                lines.append(f"\n- Spearman({metric} nolc, bin midpoint) [{model}] = "
                             f"{rho:+.2f} (p={p:.3f}, n={len(rows)}) "
                             f"{'(higher=harder)' if not asc else ''}")

    # QA
    consist = [r["rain_consistency_pct"] for r in per_event_rows
               if r["horizon_kind"] == "full" and isinstance(r["rain_consistency_pct"], (int, float))]
    if consist:
        lines.append("\n## QA — rollout rainfall vs total_inches (bin consistency)\n")
        lines.append(f"- rain delivered over full rollout = "
                     f"{min(consist):.1f}–{max(consist):.1f}% of total_inches "
                     f"(mean {sum(consist)/len(consist):.1f}%). "
                     "Shortfall = the 2 warm-up steps not rolled out (input window).")

    lines.append("\n_Single seed; 9 buckets. At `full` horizon bins span different "
                 "durations, so cross-bin absolute error there is confounded — the "
                 "onset-aligned view removes that. Long-horizon NSE can go very "
                 "negative for BOTH arms ⇒ a stability/divergence result, not accuracy._\n")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


def _print_headline(summary, H_common):
    ons = [s for s in summary if s["horizon_kind"] == "onset"]
    K = ons[0]["horizon_steps"] if ons else H_common
    print(f"\n{'model':>8} {'bin':>5} {'n':>3} {'nolc':>8} {'lmc':>8} "
          f"{'rel%':>7} {'wilcox':>7} {'NSEnolc':>8}  (onset {K}st)")
    for s in ons:
        print(f"{s['model']:>8} {s['holdout_bin']:>5} {s['n_events']:3d} "
              f"{s['rmse_nolc_m']:8.4f} {s['rmse_lmc_m']:8.4f} "
              f"{s['rel_diff_pct']:+7.2f} {s['wilcoxon_p']:7.3f} {s['nse_nolc']:8.3f}")


if __name__ == "__main__":
    main()
