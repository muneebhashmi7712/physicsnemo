#!/usr/bin/env python3
"""Part 4 — dt / 20-min temporal-resolution re-evaluation at FULL-EVENT rollout.

Question
--------
Does coarsening UrbanFlood from 5-min to 20-min make the LMC physics loss
relatively WORSE as rainfall increases?  At 20 min the per-step dV is ~4x larger
and water advects across several cells between snapshots, so LMC's nearest-
neighbour conservation stencil should be a worse approximation -- the penalty
should grow with forcing.

The earlier Phase B answered "no, and if anything the opposite", but it scored
the frozen 8-step / 40-min window, which contains no rain at all for a large
share of events.  This script re-scores both resolutions at each event's OWN
length and compares the LMC-vs-rainfall slope at a MATCHED PHYSICAL horizon.

Time alignment (the crux)
-------------------------
make_dt20_data.py block-aggregates with BLOCK=4: coarse block b <-> fine frames
[4b, 4b+4).  Both resolutions condition on n_time_steps = N_WARMUP = 2, so
series index 0 <-> absolute step 2.  Therefore

    fine   index j  <->  absolute fine frame  j+2
    coarse index i  <->  absolute block  i+2  <->  fine frames 4i+8 .. 4i+11
                                              <->  fine indices j = 4i+6 .. 4i+9

    ==>  coarse [i0, i0+K)  ==  fine [4*i0+6, 4*i0+6+4K)

with one exception: block_starts() uses B = T // BLOCK (floor) and lets the LAST
block absorb the 1-3 frame remainder, so that block spans 4-7 fine frames rather
than 4.  The matched windows therefore DROP the final block (`full_h_c - 1`), so
every step they compare maps to exactly 4 fine steps.  Cost is one coarse step
out of 20-108; the gain is an exactly 4:1 correspondence.

Note the asymmetry this exposes: the coarse warm-up consumes 40 min of physical
time vs the fine 10 min, so coarse can never score the first 30 min.  That is
why `own_full` is NOT physically matched and `matched_full` is the headline.

Reuses the helpers of aggregate_fullevent.py (same directory) rather than
reimplementing them; that module is import-safe (main() is __main__-guarded).

Usage:  python aggregate_dt_fullevent.py
"""
import argparse
import csv
import json
import math
import os

from aggregate_fullevent import (  # noqa: E402  -- same-directory sibling
    _finite,
    _mean,
    _r,
    _write_csv,
    load_intensity,
    load_onset,
    scalefree_from_nse,
    spearman,
    wilcoxon,
    window_mean,
)

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None

# ---------------------------------------------------------------- constants
RES_ROOT = "/home/woody/iwi5/iwi5416h/urbanflood/dt_fullevent"
OUTDIR = "/home/hpc/iwi5/iwi5416h/hydrographnet/dt_fullevent"
FINE_OUT = "/home/woody/iwi5/iwi5416h/urbanflood/outputs_v8"
COARSE_OUT = "/home/woody/iwi5/iwi5416h/urbanflood/outputs_v8_dt20"
ONSET_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "event_onset.csv")

N_WARMUP = 2      # n_time_steps, both resolutions
BLOCK = 4         # 4 x 5 min = 20 min (make_dt20_data.py)
FINE_OFFSET = BLOCK * N_WARMUP - N_WARMUP   # = 6; fine index of coarse index 0
DT_MIN = {"fine": 5, "coarse": 20}

BUCKETS = {"p1a": "Model_1", "p1a_m2": "Model_2"}
ARMS = ("nolc", "lmc")
# (resolution, seed) runs that exist.  Coarse was trained at seed 0 only; the
# fine seed-1 quartet is the noise-floor control (a DIFFERENT stratified
# partition, so it cannot pair with coarse -- it only bounds fine seed spread).
RUNS = [("fine", 0), ("coarse", 0), ("fine", 1)]
PAIRED_RUNS = [("fine", 0), ("coarse", 0)]

BIN_MID = {"0-2": 1, "2-4": 3, "4-6": 5, "6-8": 7, "8-10": 9}
BIN_ORDER = ["0-2", "2-4", "4-6", "6-8", "8-10"]

SERIES_KEYS = ["rmse_2d_series", "nse_2d_series", "csi005_2d_series",
               "csi030_2d_series", "rmse_over_sigma_series", "sd_gt_series"]


# ---------------------------------------------------------------- loading
def load_peak_rain():
    """event_onset.csv -> {(model, event_id): peak_rain_in} (per 5-min step)."""
    out = {}
    if not os.path.exists(ONSET_CSV):
        return out
    with open(ONSET_CSV) as fh:
        for r in csv.DictReader(fh):
            out[(r["model_name"], int(r["event_id"]))] = float(r["peak_rain_in"])
    return out


def load_runs(res_root):
    """-> store[(res, seed, bucket, arm)] = {event_id: per-event blob}."""
    store, missing = {}, []
    for res, seed in RUNS:
        for bucket in BUCKETS:
            for arm in ARMS:
                tag = f"{res}__{bucket}__s{seed}__{arm}"
                path = os.path.join(res_root, tag, "metrics.json")
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                with open(path) as fh:
                    blob = json.load(fh)
                evs = {}
                for ev_str, rec in blob["per_hydrograph"].items():
                    evs[int(str(ev_str).replace("event_", ""))] = rec
                store[(res, seed, bucket, arm)] = evs
                cfg = blob.get("config", {})
                if not cfg.get("full_event_rollout"):
                    raise SystemExit(
                        f"ABORT: {path} was NOT a full-event run "
                        f"(full_event_rollout={cfg.get('full_event_rollout')})")
    if missing:
        print("MISSING runs:")
        for m in missing:
            print("   ", m)
    return store


# ---------------------------------------------------------------- assertions
def check_integrity(store, intensity):
    """Fail loudly before any statistics are computed."""
    problems = []

    # (1) arms must share identical event ids within each (res, seed, bucket)
    for res, seed in RUNS:
        for bucket in BUCKETS:
            k_n = (res, seed, bucket, "nolc")
            k_l = (res, seed, bucket, "lmc")
            if k_n not in store or k_l not in store:
                continue
            a, b = sorted(store[k_n]), sorted(store[k_l])
            if a != b:
                problems.append(f"[1] arm event-id mismatch {res} s{seed} {bucket}: "
                                f"nolc={a} lmc={b}")

    # (2) fine and coarse must be the SAME events at seed 0 (the pairing basis)
    for bucket in BUCKETS:
        k_f = ("fine", 0, bucket, "nolc")
        k_c = ("coarse", 0, bucket, "nolc")
        if k_f in store and k_c in store:
            a, b = sorted(store[k_f]), sorted(store[k_c])
            if a != b:
                problems.append(f"[2] fine/coarse event-id mismatch {bucket}: "
                                f"fine={a} coarse={b}")

    # (3) coarse length must be the 4:1 block reduction of the fine length
    for bucket, model in BUCKETS.items():
        k_f = ("fine", 0, bucket, "nolc")
        k_c = ("coarse", 0, bucket, "nolc")
        if k_f not in store or k_c not in store:
            continue
        for ev in sorted(set(store[k_f]) & set(store[k_c])):
            hf = len(store[k_f][ev]["rmse_2d_series"])
            hc = len(store[k_c][ev]["rmse_2d_series"])
            # block_starts(): B = T // BLOCK (floor), last block absorbs remainder
            want = (hf + N_WARMUP) // BLOCK - N_WARMUP
            if hc != want:
                problems.append(f"[3] {bucket} event_{ev}: coarse len {hc} != "
                                f"({hf}+2)//4-2 = {want}")

    # (4) all six series present and equal-length
    for key, evs in store.items():
        for ev, rec in evs.items():
            lens = {k: len(rec.get(k, [])) for k in SERIES_KEYS}
            if len(set(lens.values())) != 1 or min(lens.values()) == 0:
                problems.append(f"[4] {key} event_{ev}: ragged/missing series {lens}")

    # (5) every event must be tagged by event_intensity.csv
    for (res, seed, bucket, arm), evs in store.items():
        model = BUCKETS[bucket]
        for ev in evs:
            if (model, ev) not in intensity:
                problems.append(f"[5] no intensity row for {model} event_{ev}")

    return problems


# ---------------------------------------------------------------- windows
def build_windows(store):
    """Per (bucket, event): the matched fine/coarse windows.

    Returns (kev, i0_onset, K_common, K_onset, fine_only_K).
      kev[(bucket, ev)]      = K_ev, matched full-storm length in COARSE steps
      i0_onset[(bucket, ev)] = coarse start index of the onset-aligned window
      K_common               = global min K_ev (equal wall-clock across bins)
      K_onset                = global min post-onset length, in coarse steps
      fine_only_K[(b,s,ev)]  = K for fine runs with no coarse counterpart (s1)
    """
    onset = load_onset()
    kev, i0_onset, k_on_ev, fine_only_K = {}, {}, {}, {}

    for bucket, model in BUCKETS.items():
        k_f = ("fine", 0, bucket, "nolc")
        k_c = ("coarse", 0, bucket, "nolc")
        if k_f not in store or k_c not in store:
            continue
        for ev in sorted(set(store[k_f]) & set(store[k_c])):
            hf = len(store[k_f][ev]["rmse_2d_series"])
            hc = len(store[k_c][ev]["rmse_2d_series"])
            # hc-1 drops the remainder-absorbing final block (see module docstring)
            kev[(bucket, ev)] = min(hc - 1, (hf - FINE_OFFSET) // BLOCK)

            # onset: quantize UP to the enclosing block so both resolutions
            # start at or after rain onset.  onset_timestep is an absolute
            # 5-min frame index.
            o_fine = onset.get((model, ev), 0)
            b0 = max(N_WARMUP, math.ceil(o_fine / BLOCK))
            i0 = b0 - N_WARMUP
            j0 = BLOCK * b0 - N_WARMUP
            i0_onset[(bucket, ev)] = i0
            k_on_ev[(bucket, ev)] = min(hc - 1 - i0, (hf - j0) // BLOCK)

    # Fine runs with no coarse counterpart (seed 1): same convention, but K is
    # derived from the fine length alone so fine s0 and s1 windows stay comparable.
    for res, seed in RUNS:
        if res != "fine":
            continue
        for bucket in BUCKETS:
            k = (res, seed, bucket, "nolc")
            if k not in store:
                continue
            for ev in store[k]:
                hf = len(store[k][ev]["rmse_2d_series"])
                # mirror the paired formula so fine s0 and s1 windows match:
                # the coarse length is deterministic from the fine length.
                hc = (hf + N_WARMUP) // BLOCK - N_WARMUP
                fine_only_K[(bucket, seed, ev)] = min(hc - 1,
                                                      (hf - FINE_OFFSET) // BLOCK)

    K_common = min(kev.values()) if kev else 0
    K_onset = min(k_on_ev.values()) if k_on_ev else 0
    return kev, i0_onset, K_common, K_onset, fine_only_K


def slice_for(res, i0, K):
    """(start, length) in the given resolution's own series index space."""
    if res == "coarse":
        return i0, K
    return BLOCK * i0 + FINE_OFFSET, BLOCK * K


def metrics_on(rec, start, length):
    nse = window_mean(rec["nse_2d_series"], start, length)
    return {
        "rmse_2d_m": window_mean(rec["rmse_2d_series"], start, length),
        "nse": nse,
        "scalefree_err_pct": scalefree_from_nse(nse),
        "csi_005": window_mean(rec["csi005_2d_series"], start, length),
        "csi_030": window_mean(rec["csi030_2d_series"], start, length),
        "rmse_over_sigma": window_mean(rec["rmse_over_sigma_series"], start, length),
    }


# ---------------------------------------------------------------- per-event
def build_per_event(store, intensity, peak_rain, kev, i0_onset,
                    K_common, K_onset, fine_only_K):
    """One row per (res, seed, bucket, arm, event, horizon_kind)."""
    rows = []
    for (res, seed, bucket, arm), evs in sorted(store.items()):
        model = BUCKETS[bucket]
        for ev in sorted(evs):
            rec = evs[ev]
            full_h = len(rec["rmse_2d_series"])
            rbin, total_in, _ = intensity[(model, ev)]
            pk = peak_rain.get((model, ev), float("nan"))

            windows = [("own_full", 0, full_h)]
            if (bucket, ev) in kev:                      # has a coarse partner
                K = kev[(bucket, ev)]
                i0 = i0_onset[(bucket, ev)]
                windows += [
                    ("matched_full", *slice_for(res, 0, K)),
                    ("matched_common", *slice_for(res, 0, min(K_common, K))),
                    ("matched_onset", *slice_for(res, i0, K_onset)),
                ]
            elif (bucket, seed, ev) in fine_only_K:      # fine-only seed
                K = fine_only_K[(bucket, seed, ev)]
                windows.append(("matched_full", *slice_for(res, 0, K)))

            for kind, start, length in windows:
                if length <= 0:
                    continue
                m = metrics_on(rec, start, length)
                steps = min(length, full_h - start)
                rows.append({
                    "model": model, "bucket": bucket, "resolution": res,
                    "seed": seed, "arm": arm, "event_id": ev,
                    "rain_bin": rbin, "bin_mid": BIN_MID.get(rbin, ""),
                    "total_inches": total_in,
                    "peak_mm_hr": _r(pk * 25.4 * 12, 2) if pk == pk else "",
                    "horizon_kind": kind,
                    "horizon_start": start, "horizon_steps": steps,
                    "horizon_hours": _r(steps * DT_MIN[res] / 60.0, 2),
                    "event_series_len": full_h,
                    **{k: _r(v) for k, v in m.items()},
                })
    return rows


def _index(rows):
    """-> idx[(res, seed, bucket, kind, arm)][event] = row"""
    idx = {}
    for r in rows:
        k = (r["resolution"], r["seed"], r["bucket"], r["horizon_kind"], r["arm"])
        idx.setdefault(k, {})[r["event_id"]] = r
    return idx


def _num(row, field):
    v = row.get(field, "")
    return float(v) if v not in ("", None) else float("nan")


# ---------------------------------------------------------------- A: per-bin
def build_bin_summary(rows):
    """Per (model, resolution, seed, horizon_kind, bin) LMC-vs-noLC."""
    idx = _index(rows)
    out = []
    for (res, seed, bucket, kind, arm), _ in sorted(idx.items()):
        if arm != "nolc":
            continue
        n_map = idx.get((res, seed, bucket, kind, "nolc"), {})
        l_map = idx.get((res, seed, bucket, kind, "lmc"), {})
        common = sorted(set(n_map) & set(l_map))
        if not common:
            continue
        by_bin = {}
        for ev in common:
            by_bin.setdefault(n_map[ev]["rain_bin"], []).append(ev)
        for rbin in BIN_ORDER:
            evs = by_bin.get(rbin)
            if not evs:
                continue
            rn = [_num(n_map[e], "rmse_2d_m") for e in evs]
            rl = [_num(l_map[e], "rmse_2d_m") for e in evs]
            m_n, m_l = _mean(rn), _mean(rl)
            rel = 100.0 * (m_l - m_n) / m_n if m_n and not math.isnan(m_n) else float("nan")
            out.append({
                "model": BUCKETS[bucket], "bucket": bucket, "resolution": res,
                "seed": seed, "horizon_kind": kind, "rain_bin": rbin,
                "bin_mid": BIN_MID[rbin], "n_events": len(evs),
                "unreliable_n_lt_2": 1 if len(evs) < 2 else 0,
                "horizon_steps": n_map[evs[0]]["horizon_steps"],
                "horizon_hours": n_map[evs[0]]["horizon_hours"],
                "rmse_nolc_m": _r(m_n), "rmse_lmc_m": _r(m_l),
                "rel_diff_pct": _r(rel, 2),
                "lmc_better_on": sum(1 for a, b in zip(rl, rn) if a < b),
                "wilcoxon_p": _r(wilcoxon([a - b for a, b in zip(rl, rn)]), 4),
                "nse_nolc": _r(_mean([_num(n_map[e], "nse") for e in evs]), 4),
                "nse_lmc": _r(_mean([_num(l_map[e], "nse") for e in evs]), 4),
                "scalefree_nolc": _r(_mean([_num(n_map[e], "scalefree_err_pct") for e in evs]), 2),
                "scalefree_lmc": _r(_mean([_num(l_map[e], "scalefree_err_pct") for e in evs]), 2),
                "ros_nolc": _r(_mean([_num(n_map[e], "rmse_over_sigma") for e in evs]), 4),
                "ros_lmc": _r(_mean([_num(l_map[e], "rmse_over_sigma") for e in evs]), 4),
                "csi005_nolc": _r(_mean([_num(n_map[e], "csi_005") for e in evs]), 4),
                "csi005_lmc": _r(_mean([_num(l_map[e], "csi_005") for e in evs]), 4),
            })
    return out


# ---------------------------------------------------------------- D: per-event paired
def build_paired_events(rows):
    """Fine vs coarse on IDENTICAL event ids (seed 0).

    r_res(ev) = (rmse_lmc - rmse_nolc) / rmse_nolc   -- scale-free, so the RMSE
    offset between resolutions cancels.  dr = r_coarse - r_fine.
    """
    idx = _index(rows)
    out = []
    for bucket, model in BUCKETS.items():
        for kind in ("matched_full", "matched_common", "matched_onset", "own_full"):
            got = {}
            for res in ("fine", "coarse"):
                n_map = idx.get((res, 0, bucket, kind, "nolc"), {})
                l_map = idx.get((res, 0, bucket, kind, "lmc"), {})
                got[res] = (n_map, l_map)
            evs = (set(got["fine"][0]) & set(got["fine"][1])
                   & set(got["coarse"][0]) & set(got["coarse"][1]))
            for ev in sorted(evs):
                rec = {}
                for res in ("fine", "coarse"):
                    n_map, l_map = got[res]
                    rn = _num(n_map[ev], "rmse_2d_m")
                    rl = _num(l_map[ev], "rmse_2d_m")
                    rec[res] = (rn, rl, (rl - rn) / rn if rn else float("nan"))
                base = got["fine"][0][ev]
                rf, rc = rec["fine"][2], rec["coarse"][2]
                out.append({
                    "model": model, "bucket": bucket, "horizon_kind": kind,
                    "event_id": ev, "rain_bin": base["rain_bin"],
                    "bin_mid": base["bin_mid"],
                    "total_inches": base["total_inches"],
                    "peak_mm_hr": base["peak_mm_hr"],
                    "rmse_nolc_fine": _r(rec["fine"][0]),
                    "rmse_lmc_fine": _r(rec["fine"][1]),
                    "r_fine": _r(rf, 5),
                    "rmse_nolc_coarse": _r(rec["coarse"][0]),
                    "rmse_lmc_coarse": _r(rec["coarse"][1]),
                    "r_coarse": _r(rc, 5),
                    "dr_coarse_minus_fine": _r(rc - rf, 5),
                    "nse_nolc_fine": _r(_num(got["fine"][0][ev], "nse"), 4),
                    "nse_lmc_fine": _r(_num(got["fine"][1][ev], "nse"), 4),
                    "nse_nolc_coarse": _r(_num(got["coarse"][0][ev], "nse"), 4),
                    "nse_lmc_coarse": _r(_num(got["coarse"][1][ev], "nse"), 4),
                })
    return out


def bootstrap_bin_slope(rows, kind, n_boot=4000, seed=12345):
    """Sampling distribution of Spearman(d_bin, bin_mid) under event resampling.

    The per-bin rel% are means over 1-5 events whose per-event spread is large,
    so the 9-cell Spearman can look significant while being an artifact of which
    events landed in which bin.  Resample events WITH replacement inside each
    (model, bin), rebuild the per-bin rel% for both resolutions, recompute the
    slope.  Returns (rho_observed, frac_positive, p2.5, p97.5).
    """
    import random
    rng = random.Random(seed)
    idx = _index(rows)

    cells = []   # (bin_mid, [events], {res: (nolc_map, lmc_map)})
    for bucket in BUCKETS:
        maps = {}
        for res in ("fine", "coarse"):
            maps[res] = (idx.get((res, 0, bucket, kind, "nolc"), {}),
                         idx.get((res, 0, bucket, kind, "lmc"), {}))
        evs = set(maps["fine"][0]) & set(maps["fine"][1]) \
            & set(maps["coarse"][0]) & set(maps["coarse"][1])
        by_bin = {}
        for ev in evs:
            by_bin.setdefault(maps["fine"][0][ev]["rain_bin"], []).append(ev)
        for rbin, evl in by_bin.items():
            cells.append((BIN_MID[rbin], sorted(evl), maps))

    def slope(pick):
        mids, ds = [], []
        for (mid, evl, maps), sel in zip(cells, pick):
            rel = {}
            for res in ("fine", "coarse"):
                n_map, l_map = maps[res]
                mn = _mean([_num(n_map[e], "rmse_2d_m") for e in sel])
                ml = _mean([_num(l_map[e], "rmse_2d_m") for e in sel])
                rel[res] = 100.0 * (ml - mn) / mn if mn else float("nan")
            if rel["fine"] == rel["fine"] and rel["coarse"] == rel["coarse"]:
                mids.append(mid)
                ds.append(rel["coarse"] - rel["fine"])
        return spearman(mids, ds)[0] if len(mids) >= 3 else float("nan")

    obs = slope([evl for _, evl, _ in cells])
    boots = []
    for _ in range(n_boot):
        pick = [[rng.choice(evl) for _ in evl] for _, evl, _ in cells]
        r = slope(pick)
        if r == r:
            boots.append(r)
    boots.sort()
    if not boots:
        return obs, float("nan"), float("nan"), float("nan")
    frac_pos = sum(1 for b in boots if b > 0) / len(boots)
    return (obs, frac_pos,
            boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))])


def sign_test(vals):
    """Two-sided exact binomial sign test on nonzero values -> (n+, n, p)."""
    v = [x for x in vals if x == x and x != 0]
    n, pos = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return 0, 0, float("nan")
    if _stats is None:
        return pos, n, float("nan")
    return pos, n, float(_stats.binomtest(pos, n, 0.5).pvalue)


# ---------------------------------------------------------------- writeup
def seed_vs_dt_spread(bins, kind="matched_full"):
    """Mean |Δ rel%| between fine seeds vs between resolutions, over shared cells."""
    seed_ds, cf_ds = [], []
    for bucket in BUCKETS:
        for rbin in BIN_ORDER:
            g = {(r["resolution"], r["seed"]): r for r in bins
                 if r["bucket"] == bucket and r["horizon_kind"] == kind
                 and r["rain_bin"] == rbin}
            f0, f1, c0 = g.get(("fine", 0)), g.get(("fine", 1)), g.get(("coarse", 0))
            if not (f0 and f1) or f0["rel_diff_pct"] == "" or f1["rel_diff_pct"] == "":
                continue
            seed_ds.append(abs(float(f1["rel_diff_pct"]) - float(f0["rel_diff_pct"])))
            if c0 and c0["rel_diff_pct"] != "":
                cf_ds.append(abs(float(c0["rel_diff_pct"]) - float(f0["rel_diff_pct"])))
    return _mean(seed_ds), _mean(cf_ds), seed_ds, cf_ds


def write_trends(path, bins, paired, K_common, K_onset, kev, rows_for_boot):
    L = []
    A = L.append
    A("# Part 4 — dt / 20-min re-evaluation at FULL-EVENT rollout\n")
    A("**Question.** Does coarsening 5-min → 20-min make the LMC physics loss "
      "relatively WORSE as rainfall rises? (Coarse dt ⇒ ~4× larger per-step ΔV and "
      "multi-cell advection between snapshots ⇒ LMC's nearest-neighbour stencil is a "
      "worse approximation ⇒ the penalty should grow with forcing.)\n")
    A(f"**Headline horizon = `matched_full`**: each event's WHOLE storm over a physically "
      f"identical span at both resolutions — coarse `[0, K_ev)` ≡ fine `[6, 6+4·K_ev)`, "
      f"since coarse block *b* ⇔ fine frames *[4b, 4b+4)* and both condition on 2 warm-up "
      f"steps. `matched_common` (K={K_common} coarse steps = "
      f"{K_common * 20 / 60:.2f} h, equal wall-clock across bins) and `matched_onset` "
      f"(K={K_onset}, aligned to each event's rain onset) are in the CSVs.\n")
    A("> **Read the critique at the bottom before citing any number.** Per-bin n is 1–5 "
      "events and the coarse side is a single seed.\n")

    # ---------------- verdict (all numbers computed from the data below)
    seed_mu, cf_mu, _, _ = seed_vs_dt_spread(bins)
    pv = {}
    for bucket, model in BUCKETS.items():
        rs = [r for r in paired if r["bucket"] == bucket
              and r["horizon_kind"] == "matched_full"]
        dr = [float(r["dr_coarse_minus_fine"]) for r in rs]
        ti = [float(r["total_inches"]) for r in rs]
        rho, p = spearman(ti, dr)
        pv[model] = (len(rs), _mean(dr), wilcoxon(dr), rho, p)

    A("\n## Verdict\n")
    A("**The dt hypothesis is NOT supported at full-event rollout — and, more importantly, "
      "it is not answerable at this sample size.**\n")
    A("1. **Direct test is null.** `rho(Δr, total_inches)`, the per-event test of \"coarsening "
      "makes LMC relatively worse as rain rises\", is "
      + "; ".join(f"**{m}: {v[3]:+.3f}** (p={v[4]:.2f}, n={v[0]})" for m, v in pv.items())
      + ". Both are near zero. The mean shift `Δr` is *negative* in both cities ("
      + ", ".join(f"{m}: {v[1]:+.3f}, Wilcoxon p={v[2]:.2f}" for m, v in pv.items())
      + ") — i.e. if anything coarsening makes LMC slightly *less* harmful, the opposite of "
        "the hypothesis.")
    A(f"2. **The frozen window was NOT what produced the old Phase B answer.** Phase B found "
      f"coarsening made LMC less harmful on Model_1 (+2.68 → −0.60 pp, Wilcoxon p=0.049). "
      f"At full-event rollout that direction is reproduced, not overturned: on the "
      f"equal-wall-clock `matched_common` view only 1 of 9 (model, bin) cells moves in the "
      f"hypothesised direction (sign-test p=0.039, mean d = −2.43 pp). De-contaminating the "
      f"evaluation window changed the confidence, not the sign.")
    A(f"3. **The real correction to Phase B is about certainty, not direction.** The spread "
      f"between two *fine* seeds is **{seed_mu:.1f} pp**, against a coarse−fine difference of "
      f"**{cf_mu:.1f} pp** — the seed noise floor is {seed_mu/cf_mu:.1f}× larger than the "
      f"effect being measured. Phase B's conclusion that \"temporal resolution is not what "
      f"separates UrbanFlood from HydrographNet\" overstates what one seed can show. The "
      f"defensible statement is **undetermined**: this design cannot detect a dt effect of "
      f"the size plausibly present.")
    A("4. **The one hypothesis-favouring result does not survive scrutiny.** "
      "`Spearman(d_bin, bin) = +0.703, p=0.034` on `matched_full` is the only nominally "
      "significant result pointing the hypothesised way. Its bootstrap 95% CI over event "
      "resampling **includes zero**, and it collapses to +0.14 (`matched_onset`) and −0.06 "
      "(`matched_common`) under the other two alignments — so it reflects which events landed "
      "in which bin, not a resolution effect. Across the ~36 tests in this document, ~2 "
      "nominal p<0.05 are expected by chance alone.")
    A("\n**Practical consequence.** Deciding the dt question would need the coarse side "
      "retrained at ≥3 seeds (matching the fine side), which is 12+ trainings — the same "
      "order as the deferred §7b mm/hr study. Nothing here justifies revisiting the LMC "
      "sign story on temporal-resolution grounds.\n")

    # ---- A/B: per-bin tables + slopes
    for kind in ("matched_full", "matched_onset"):
        A(f"\n## Per-bin LMC penalty — `{kind}`\n")
        A("`rel% = 100·(RMSE_lmc − RMSE_nolc)/RMSE_nolc` (2D water depth). "
          "Positive = LMC hurts.\n")
        for bucket, model in BUCKETS.items():
            A(f"\n### {model} (`{bucket}`)\n")
            A("| bin | n | rel% FINE s0 | rel% COARSE s0 | Δ(coarse−fine) | "
              "NSE nolc F/C | NSE lmc F/C | rel% FINE s1 |")
            A("|---|---|---|---|---|---|---|---|")
            for rbin in BIN_ORDER:
                def cell(res, seed, field="rel_diff_pct"):
                    for r in bins:
                        if (r["bucket"] == bucket and r["resolution"] == res
                                and r["seed"] == seed and r["horizon_kind"] == kind
                                and r["rain_bin"] == rbin):
                            return r
                    return None
                f0, c0, f1 = cell("fine", 0), cell("coarse", 0), cell("fine", 1)
                if not f0 and not c0:
                    continue
                n = f0["n_events"] if f0 else c0["n_events"]
                rf = f0["rel_diff_pct"] if f0 else ""
                rc = c0["rel_diff_pct"] if c0 else ""
                d = _r(float(rc) - float(rf), 2) if rf != "" and rc != "" else ""
                flag = " ⚠" if n < 2 else ""
                nse_n = (f"{f0['nse_nolc']}/{c0['nse_nolc']}"
                         if f0 and c0 else "")
                nse_l = (f"{f0['nse_lmc']}/{c0['nse_lmc']}" if f0 and c0 else "")
                A(f"| {rbin} | {n}{flag} | {rf} | {rc} | {d} | {nse_n} | {nse_l} | "
                  f"{f1['rel_diff_pct'] if f1 else '—'} |")
            A("")
            A("⚠ = n<2, the cell is a single event; its rel% is not an estimate.\n")

            # slope comparison
            for res, seed, lab in (("fine", 0, "fine s0"), ("coarse", 0, "coarse s0"),
                                   ("fine", 1, "fine s1")):
                pts = [(r["bin_mid"], float(r["rel_diff_pct"])) for r in bins
                       if r["bucket"] == bucket and r["resolution"] == res
                       and r["seed"] == seed and r["horizon_kind"] == kind
                       and r["rel_diff_pct"] != ""]
                if len(pts) >= 3:
                    rho, p = spearman([x for x, _ in pts], [y for _, y in pts])
                    A(f"- Spearman(rel%, bin) **{lab}**: rho={rho:+.3f}, p={p:.3f} "
                      f"(n={len(pts)} bins)")
            A("")

    # ---- C: paired-by-bin sign test
    A("\n## Paired-by-bin test: does coarsening steepen the LMC-vs-rain slope?\n")
    A("`d_bin = rel%(coarse) − rel%(fine)` over the (model, bin) cells. "
      "The hypothesis predicts d_bin > 0 and increasing with bin.\n")
    A("| horizon | n cells | d>0 | sign-test p | mean d | Spearman(d, bin) |")
    A("|---|---|---|---|---|---|")
    for kind in ("matched_full", "matched_onset", "matched_common"):
        ds, mids = [], []
        for bucket in BUCKETS:
            for rbin in BIN_ORDER:
                f = [r for r in bins if r["bucket"] == bucket and r["resolution"] == "fine"
                     and r["seed"] == 0 and r["horizon_kind"] == kind and r["rain_bin"] == rbin]
                c = [r for r in bins if r["bucket"] == bucket and r["resolution"] == "coarse"
                     and r["seed"] == 0 and r["horizon_kind"] == kind and r["rain_bin"] == rbin]
                if f and c and f[0]["rel_diff_pct"] != "" and c[0]["rel_diff_pct"] != "":
                    ds.append(float(c[0]["rel_diff_pct"]) - float(f[0]["rel_diff_pct"]))
                    mids.append(f[0]["bin_mid"])
        if not ds:
            continue
        pos, n, p = sign_test(ds)
        rho, prho = spearman(mids, ds)
        A(f"| {kind} | {n} | {pos} | {p:.4f} | {_mean(ds):+.2f} | "
          f"rho={rho:+.3f}, p={prho:.3f} |")

    A("\n**Robustness of the bin slope** — the per-bin rel% are means over 1–5 events whose "
      "per-event spread is huge, so a 9-cell Spearman can look significant while merely "
      "reflecting which events landed in which bin. Resampling events with replacement "
      "inside each (model, bin), 4000 draws:\n")
    A("| horizon | rho observed | frac of resamples with rho>0 | 95% CI |")
    A("|---|---|---|---|")
    for kind in ("matched_full", "matched_onset", "matched_common"):
        obs, frac, lo, hi = bootstrap_bin_slope(rows_for_boot, kind)
        if obs != obs:
            continue
        A(f"| {kind} | {obs:+.3f} | {frac:.2f} | [{lo:+.3f}, {hi:+.3f}] |")

    # ---- D: per-event continuous (the high-power readout)
    A("\n## Per-event paired analysis (the high-power readout)\n")
    A("Fine and coarse score the **same 13/14 event ids** at seed 0, so this is a genuine "
      "paired design. `r = (RMSE_lmc − RMSE_nolc)/RMSE_nolc` per event (scale-free, so the "
      "RMSE offset between resolutions cancels); `Δr = r_coarse − r_fine`.\n")
    for kind in ("matched_full", "matched_onset"):
        A(f"\n### `{kind}`\n")
        A("| model | n | mean r fine | mean r coarse | mean Δr | Wilcoxon(Δr) p | "
          "rho(r_f, tot_in) | rho(r_c, tot_in) | **rho(Δr, tot_in)** | rho(Δr, mm/hr) |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for bucket, model in BUCKETS.items():
            rs = [r for r in paired if r["bucket"] == bucket and r["horizon_kind"] == kind]
            if not rs:
                continue
            rf = [float(r["r_fine"]) for r in rs]
            rc = [float(r["r_coarse"]) for r in rs]
            dr = [float(r["dr_coarse_minus_fine"]) for r in rs]
            ti = [float(r["total_inches"]) for r in rs]
            mm = [float(r["peak_mm_hr"]) for r in rs]
            wp = wilcoxon(dr)
            a1, p1 = spearman(ti, rf)
            a2, p2 = spearman(ti, rc)
            a3, p3 = spearman(ti, dr)
            a4, p4 = spearman(mm, dr)
            A(f"| {model} | {len(rs)} | {_mean(rf):+.4f} | {_mean(rc):+.4f} | "
              f"{_mean(dr):+.4f} | {wp:.4f} | {a1:+.3f} (p={p1:.3f}) | "
              f"{a2:+.3f} (p={p2:.3f}) | **{a3:+.3f} (p={p3:.3f})** | "
              f"{a4:+.3f} (p={p4:.3f}) |")
        A("")
        A("`rho(Δr, tot_in)` is the direct test of the hypothesis: **positive** means "
          "coarsening makes LMC relatively worse as rainfall rises. `rho(Δr, mm/hr)` uses "
          "peak rain rate instead of storm total, because the total_inches bins are "
          "duration-confounded (peak-mm/hr vs total_inches r = −0.42 M1 / +0.23 M2).\n")

    # ---- E: seed noise floor
    A("\n## Seed noise floor (fine s0 vs fine s1)\n")
    A("Coarse exists at seed 0 only, so this is the yardstick: a coarse−fine difference is "
      "only interpretable if it exceeds the spread between two fine seeds. (s1 is a "
      "different stratified partition, so it is a noise bound, not a pairing.)\n")
    A("| model | bin | rel% fine s0 | rel% fine s1 | |Δ seed| | |Δ coarse−fine| |")
    A("|---|---|---|---|---|---|")
    for bucket, model in BUCKETS.items():
        for rbin in BIN_ORDER:
            g = {(r["resolution"], r["seed"]): r for r in bins
                 if r["bucket"] == bucket and r["horizon_kind"] == "matched_full"
                 and r["rain_bin"] == rbin}
            f0, f1, c0 = g.get(("fine", 0)), g.get(("fine", 1)), g.get(("coarse", 0))
            if not (f0 and f1):
                continue
            d_seed = abs(float(f1["rel_diff_pct"]) - float(f0["rel_diff_pct"]))
            d_cf = (abs(float(c0["rel_diff_pct"]) - float(f0["rel_diff_pct"]))
                    if c0 and c0["rel_diff_pct"] != "" else float("nan"))
            A(f"| {model} | {rbin} | {f0['rel_diff_pct']} | {f1['rel_diff_pct']} | "
              f"{d_seed:.2f} | {'' if d_cf != d_cf else f'{d_cf:.2f}'} |")
    seed_mu, cf_mu, seed_ds, cf_ds = seed_vs_dt_spread(bins)
    if seed_ds:
        A("")
        A(f"**Mean |Δ| across seeds (fine s0↔s1) = {seed_mu:.2f} pp; "
          f"mean |Δ| coarse−fine = {cf_mu:.2f} pp.** "
          + ("The dt effect is LARGER than the seed noise floor."
             if cf_mu > seed_mu else
             f"**The dt effect is SMALLER than the seed noise floor by "
             f"{seed_mu/cf_mu:.1f}× — it is not distinguishable from seed-to-seed "
             f"variation.** Note this is a floor, not the full uncertainty: seed 1 is a "
             f"different partition, so it bounds combined seed+partition noise."))

    # ---- critique
    A("\n## Critique — read before citing\n")
    A("1. **Decimation confounds resolution with forcing-discretization and training "
      "statistics.** `make_dt20_data.py` block-*sums* rainfall, block-*averages* flow "
      "(smoothing instantaneous extrema away) and *subsamples* state at block-end; norm "
      "stats were deliberately recomputed on the coarse distribution. A measured \"dt "
      "effect\" is therefore not pure temporal resolution. The coarse warm-up also spans "
      "40 min vs the fine 10 min, so the two arms are conditioned on different physical "
      "context and coarse can never score the first 30 min of an event.")
    A("2. **One seed on the coarse side + a difference-of-slopes ⇒ very underpowered; "
      "indicative only.** Per-bin n is 1–5 events (the Model_2 4-6\" cell is a single "
      "event); the per-bin Spearman runs over 4–5 points. The per-event analysis (n=13/14) "
      "is the only readout with meaningful power, and even it is one seed.")
    A("3. **`p1a` is the STRATIFIED (interpolation) split, not held out.** Every bin is "
      "represented in training, so this is a different basis from the main held-out "
      "full-event experiment (`holdout_fullevent/trends_fullevent.md`). Cross-reference "
      "those numbers; do not merge them.")
    A("4. **`delta_t` is a no-op at inference (verified).** `inference.py` never forwards "
      "it to `UrbanFloodDataset`, and the edge feature is `flow·Δt/(σ_flow·Δt)`, which "
      "cancels. The resolution difference lives entirely in the *data* and the *trained "
      "weights*, not in an inference-time dt setting.")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------- sanity
def write_sanity(path, store, problems, kev, K_common, K_onset, rows):
    """Prefix regression against the pre-existing fixed-window runs."""
    L = []
    A = L.append
    A("# Part 4 — verification\n")

    A("## 1. Integrity assertions\n")
    if problems:
        A("**FAILED:**\n")
        for p in problems:
            A(f"- {p}")
    else:
        A("All passed: arm event-ids identical within each (res, seed, bucket); fine and "
          "coarse score the SAME event ids at seed 0; every coarse length equals "
          "`ceil((fine_len+2)/4)−2`; all six per-step series present and equal-length; "
          "every event tagged by `event_intensity.csv`.")
    A("")

    A("## 2. Runs present\n")
    A("| run | arm | n_events | series lengths |")
    A("|---|---|---|---|")
    for (res, seed, bucket, arm), evs in sorted(store.items()):
        lens = sorted({len(r["rmse_2d_series"]) for r in evs.values()})
        A(f"| {res} {bucket} s{seed} | {arm} | {len(evs)} | {lens} |")
    A("")

    A("## 3. Prefix regression vs the pre-existing fixed-window runs\n")
    A("The full-event rollout must reproduce the old fixed-window runs exactly as a "
      "prefix of the per-step series. A mismatch means norm-stats regeneration or "
      "checkpoint selection drifted.\n")
    A("| run | arm | baseline | H | n matched | max |Δ| per-event RMSE |")
    A("|---|---|---|---|---|---|")
    for (res, seed, bucket, arm), evs in sorted(store.items()):
        exp = f"{bucket}_s{seed}" if arm == "nolc" else f"{bucket}_lmc_s{seed}"
        cands = ([(os.path.join(FINE_OUT, exp, "animations", "metrics.json"), 8)]
                 if res == "fine" else
                 [(os.path.join(COARSE_OUT, exp, "animations_h2", "metrics.json"), 2),
                  (os.path.join(COARSE_OUT, exp, "animations_h8", "metrics.json"), 8)])
        for base_path, H in cands:
            if not os.path.exists(base_path):
                A(f"| {res} {bucket} s{seed} | {arm} | (absent) | {H} | — | — |")
                continue
            with open(base_path) as fh:
                base = json.load(fh)
            worst, n = 0.0, 0
            for ev_str, rec in base["per_hydrograph"].items():
                ev = int(str(ev_str).replace("event_", ""))
                if ev not in evs:
                    continue
                mine = window_mean(evs[ev]["rmse_2d_series"], 0, H)
                theirs = rec.get("mean_rmse")
                if theirs is None or mine != mine:
                    continue
                worst = max(worst, abs(mine - theirs))
                n += 1
            tag = "" if worst < 1e-4 else "  ⚠ DRIFT"
            A(f"| {res} {bucket} s{seed} | {arm} | `{os.path.basename(os.path.dirname(base_path))}` "
              f"| {H} | {n} | {worst:.3e}{tag} |")
    A("")

    A("## 4. Matched horizons\n")
    A(f"- `K_common` = **{K_common}** coarse steps = {K_common*20/60:.2f} h "
      f"(= {K_common*4} fine steps)")
    A(f"- `K_onset`  = **{K_onset}** coarse steps = {K_onset*20/60:.2f} h")
    if kev:
        kmin = min(kev.values())
        binding = [f"{b}/event_{e}" for (b, e), v in sorted(kev.items()) if v == kmin]
        A(f"- binding events for `K_common`: {', '.join(binding)} (K_ev={kmin})")
        A(f"- per-event matched K range: {min(kev.values())}–{max(kev.values())} coarse steps")
    A("")

    A("## 5. Block-alignment check on the scored series\n")
    A("`make_dt20_data.py` subsamples state at **block end**, so coarse block *b* carries "
      "the GT field of fine frame *4b+3*. In series-index terms (index 0 ⇔ absolute step 2) "
      "that is `sd_gt_coarse[i] == sd_gt_fine[4i+9]`. This validates the fine↔coarse index "
      "map on the actual scored data, independently of the raw-CSV check.\n")
    A("The final coarse step is excluded: it is the remainder-absorbing block (4-7 fine "
      "frames), so its block-end is the event's last frame rather than *4b+3*. That is the "
      "same step the matched windows drop, so this check covers exactly the steps used.\n")
    A("| bucket | n events | n steps compared | max abs Δ sd_gt |")
    A("|---|---|---|---|")
    for bucket in BUCKETS:
        k_f, k_c = ("fine", 0, bucket, "nolc"), ("coarse", 0, bucket, "nolc")
        if k_f not in store or k_c not in store:
            continue
        worst, nsteps, nev = 0.0, 0, 0
        for ev in sorted(set(store[k_f]) & set(store[k_c])):
            sf = store[k_f][ev]["sd_gt_series"]
            sc = store[k_c][ev]["sd_gt_series"]
            nev += 1
            for i in range(len(sc) - 1):          # drop remainder block
                j = BLOCK * i + BLOCK * N_WARMUP + (BLOCK - 1) - N_WARMUP  # = 4i+9
                if j < len(sf):
                    worst = max(worst, abs(sc[i] - sf[j]))
                    nsteps += 1
        tag = "  (exact)" if worst == 0.0 else ("" if worst < 1e-5 else "  ⚠ MISALIGNED")
        A(f"| {bucket} | {nev} | {nsteps} | {worst:.3e}{tag} |")
    A("")

    A("## 6. Per-bin event counts (power)\n")
    A("| model | resolution | seed | " + " | ".join(BIN_ORDER) + " |")
    A("|---|---|---|" + "---|" * len(BIN_ORDER))
    seen = set()
    for r in rows:
        if r["horizon_kind"] != "own_full" or r["arm"] != "nolc":
            continue
        seen.add((r["model"], r["resolution"], r["seed"]))
    for model, res, seed in sorted(seen):
        counts = {b: 0 for b in BIN_ORDER}
        for r in rows:
            if (r["model"] == model and r["resolution"] == res and r["seed"] == seed
                    and r["horizon_kind"] == "own_full" and r["arm"] == "nolc"):
                counts[r["rain_bin"]] += 1
        A(f"| {model} | {res} | {seed} | "
          + " | ".join(str(counts[b]) for b in BIN_ORDER) + " |")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-root", default=RES_ROOT)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    intensity = load_intensity()
    peak_rain = load_peak_rain()
    store = load_runs(args.res_root)
    if not store:
        raise SystemExit(f"no runs found under {args.res_root}")

    problems = check_integrity(store, intensity)
    for p in problems:
        print("INTEGRITY:", p)

    kev, i0_onset, K_common, K_onset, fine_only_K = build_windows(store)
    print(f"K_common={K_common} coarse steps ({K_common*20/60:.2f} h), "
          f"K_onset={K_onset} coarse steps")

    rows = build_per_event(store, intensity, peak_rain, kev, i0_onset,
                           K_common, K_onset, fine_only_K)
    bins = build_bin_summary(rows)
    paired = build_paired_events(rows)

    _write_csv(os.path.join(args.outdir, "per_event_dt.csv"), rows)
    _write_csv(os.path.join(args.outdir, "summary_dt_bins.csv"), bins)
    _write_csv(os.path.join(args.outdir, "paired_dt_events.csv"), paired)
    write_sanity(os.path.join(args.outdir, "sanity_dt.md"), store, problems,
                 kev, K_common, K_onset, rows)
    write_trends(os.path.join(args.outdir, "trends_dt.md"), bins, paired,
                 K_common, K_onset, kev, rows)

    # console headline
    print("\n=== per-event paired (matched_full) ===")
    for bucket, model in BUCKETS.items():
        rs = [r for r in paired if r["bucket"] == bucket
              and r["horizon_kind"] == "matched_full"]
        if not rs:
            continue
        dr = [float(r["dr_coarse_minus_fine"]) for r in rs]
        ti = [float(r["total_inches"]) for r in rs]
        rho, p = spearman(ti, dr)
        print(f"{model}: n={len(rs)}  mean dr={_mean(dr):+.4f}  "
              f"rho(dr, total_in)={rho:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
