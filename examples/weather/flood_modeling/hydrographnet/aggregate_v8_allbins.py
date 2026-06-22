"""aggregate_v8_allbins.py — matched-event ("fair") interpolation gap.

Implements the reviewer (professor) fix to the v8 interpolation gap. The original
gap compared a holdout bin's RMSE over the WHOLE bin (Phase 1B) against the same
bin's RMSE over only the SUBSET of events the stratified split happened to test
(Phase 1A) — different event sets, so the gap conflated "held the bin out" with
"tested different events".

Fix: compute a second holdout RMSE restricted to EXACTLY the events the matching
Phase 1A seed tested in that bin (per-seed paired). The reference (Phase 1A side)
is unchanged — it is the same per-bin pooled per-event 2D RMSE the ext study used
(M1 4-6" = 0.0178 n=15, 2-4" = 0.0310 n=12). Only the 1B side changes: instead of
all bin events (M1 4-6" n=75) it uses the matched subset (n=15). The full-bin
holdout RMSE (old number) is still reported alongside for context.

Covers every cell (Model_1/Model_2 × nolc/lmc) and every bin:
  Model_1: 0-2, 2-4, 4-6, ≥6 (=6-8+8-10)
  Model_2: 0-2, 2-4, 4-6, 6-8, 8-10

2-4"/4-6" need zero new compute (existing logs). The other bins are filled in once
their holdout jobs (submit_v8_allbins.sh) finish and IDs land in
infer_jobids_allbins.json.

Success metric is 2D-only RMSE throughout (per feedback_urbanflood_2d_only_metric).

Usage:
    python aggregate_v8_allbins.py
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, stdev

# Reuse the validated parsers / loaders from the ext aggregator.
from aggregate_v8_ext_results import (
    parse_inference_out,
    find_out_file,
    load_splits,
    load_intensity,
    CELL0_INFER_JOBIDS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
JOBIDS_EXT     = os.path.join(HERE, "infer_jobids_ext.json")
JOBIDS_ALLBINS = os.path.join(HERE, "infer_jobids_allbins.json")

# ---------------------------------------------------------------------------
# Cell / bin / experiment-name structure
# ---------------------------------------------------------------------------
MODEL_SFX = {"Model_1": "", "Model_2": "_m2"}
LMC_SFX   = {"nolc": "", "lmc": "_lmc"}

# bincode -> (display label, set of intensity-bin labels it covers)
BINCODES = {
    "h02":  ("0-2",  {"0-2"}),
    "h24":  ("2-4",  {"2-4"}),
    "h46":  ("4-6",  {"4-6"}),
    "h68":  ("6-8",  {"6-8"}),
    "h810": ("8-10", {"8-10"}),
    "hge6": ("≥6",   {"6-8", "8-10"}),
}

# Bins each city actually holds out (Model_1 combines ≥6; Model_2 splits 6-8/8-10).
CITY_BINCODES = {
    "Model_1": ["h02", "h24", "h46", "hge6"],
    "Model_2": ["h02", "h24", "h46", "h68", "h810"],
}

CELLS = [
    ("Model_1", "nolc"),
    ("Model_1", "lmc"),
    ("Model_2", "nolc"),
    ("Model_2", "lmc"),
]

# Try these seeds per cell; missing exps (no jobid / no log / no split) are skipped,
# so Model_1/nolc uses s0,s1,s2 for 2-4"/4-6" but only s0,s1 for the new bins.
SEEDS = [0, 1, 2]

# Display order for the headline matrix.
BIN_DISPLAY_ORDER = ["0-2", "2-4", "4-6", "6-8", "8-10", "≥6"]


def cell_label(model, lmc):
    return f"{model}/{lmc}"


def p1a_exp(model, lmc, seed):
    return f"p1a{MODEL_SFX[model]}{LMC_SFX[lmc]}_s{seed}"


def p1b_exp(model, lmc, bincode, seed):
    return f"p1b_{bincode}{MODEL_SFX[model]}{LMC_SFX[lmc]}_s{seed}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_jobids() -> dict:
    """Merge the three job-id sources: CELL0 (hardcoded), ext (json), allbins (json)."""
    jobids = dict(CELL0_INFER_JOBIDS)
    with open(JOBIDS_EXT) as f:
        jobids.update(json.load(f))
    if os.path.isfile(JOBIDS_ALLBINS):
        with open(JOBIDS_ALLBINS) as f:
            jobids.update(json.load(f))
    return jobids


def try_load(exp: str, jobids: dict):
    """Return {'parsed':..., 'splits':...} or None if the exp is not available yet."""
    jid = jobids.get(exp)
    if jid is None:
        return None
    try:
        path = find_out_file(exp, jid)
    except FileNotFoundError:
        return None
    return {"parsed": parse_inference_out(path), "splits": load_splits(exp)}


# ---------------------------------------------------------------------------
# Matched-event gap
# ---------------------------------------------------------------------------
def matched_gap(model, lmc, bincode, loaded, intensity):
    """Per-seed paired matched-event gap for one (cell, bin).

    Returns None if neither phase is available yet (e.g. new bin, untrained).
    """
    label, bins = BINCODES[bincode]
    pooled_1a, pooled_1b = [], []   # matched (seed, event) per-event 2D RMSE
    full_1b = []                    # ALL holdout-bin events (old methodology)
    per_seed_gap = []
    seeds_used = []

    for seed in SEEDS:
        a = loaded.get(p1a_exp(model, lmc, seed))
        b = loaded.get(p1b_exp(model, lmc, bincode, seed))
        if a is None or b is None:
            continue
        a2d = a["parsed"]["per_event_2d"]
        b2d = b["parsed"]["per_event_2d"]

        # Phase 1A events that fall in this bin (set) and have a 2D RMSE.
        matched = [ev for ev in a["splits"]["test_event_ids"]
                   if ev in a2d and intensity.get((model, ev), (None, None))[1] in bins]
        if not matched:
            continue

        seed_1a, seed_1b = [], []
        for ev in matched:
            # Holdout tests the WHOLE bin, so every matched 1A event must be in the
            # 1B log too — identical event set on both sides (fail loudly otherwise).
            assert ev in b2d, (
                f"matched-set mismatch: event_{ev} tested by {p1a_exp(model, lmc, seed)} "
                f"but absent from {p1b_exp(model, lmc, bincode, seed)} log")
            seed_1a.append(a2d[ev])
            seed_1b.append(b2d[ev])

        pooled_1a.extend(seed_1a)
        pooled_1b.extend(seed_1b)
        per_seed_gap.append(mean(seed_1b) - mean(seed_1a))
        seeds_used.append(seed)

        # Full bin = every event in the holdout test set (== whole bin).
        full_1b.extend(b2d[ev] for ev in b["splits"]["test_event_ids"] if ev in b2d)

    if not pooled_1a:
        return None

    ref     = mean(pooled_1a)
    matched_mean = mean(pooled_1b)
    full    = mean(full_1b) if full_1b else float("nan")
    return {
        "label": label,
        "n_matched": len(pooled_1a),
        "n_full": len(full_1b),
        "ref_1a": ref,
        "matched_1b": matched_mean,
        "full_1b": full,
        "matched_gap": matched_mean - ref,
        "matched_gap_pct": 100 * (matched_mean - ref) / ref if ref else float("nan"),
        "full_gap": full - ref,
        "full_gap_pct": 100 * (full - ref) / ref if ref else float("nan"),
        "per_seed_gap": per_seed_gap,
        "seeds_used": seeds_used,
    }


def fmt_gap(g):
    if g is None:
        return "—"
    return f"{g['matched_gap']:+.4f} ({g['matched_gap_pct']:+.1f}%)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="store_true",
                    help="Also write v8_allbins_results.csv next to this script.")
    args = ap.parse_args()

    jobids = load_all_jobids()
    intensity = load_intensity()

    # Lazily load every exp that could be referenced.
    loaded = {}
    for model, lmc in CELLS:
        for seed in SEEDS:
            for exp in [p1a_exp(model, lmc, seed)] + \
                       [p1b_exp(model, lmc, bc, seed) for bc in CITY_BINCODES[model]]:
                if exp not in loaded:
                    loaded[exp] = try_load(exp, jobids)

    # Compute matched gaps: results[(model,lmc)][bincode] = gap dict or None.
    results = {}
    for model, lmc in CELLS:
        results[(model, lmc)] = {
            bc: matched_gap(model, lmc, bc, loaded, intensity)
            for bc in CITY_BINCODES[model]
        }

    # -----------------------------------------------------------------------
    # Headline: matched-event gap matrix (bin × cell)
    # -----------------------------------------------------------------------
    print("# v8 all-bins — MATCHED-event interpolation gap (2D RMSE)\n")
    print("Gap = (Phase 1B holdout RMSE on the SAME events Phase 1A tested) "
          "− (Phase 1A same-event RMSE). Identical event set on both sides; the only "
          "difference is whether the bin was in training. Negative = holdout does better.\n")
    header = "| Held-out bin | " + " | ".join(cell_label(m, l) for m, l in CELLS) + " |"
    print(header)
    print("|---|" + "---|" * len(CELLS))
    # Build label -> bincode per cell for display lookup.
    for blabel in BIN_DISPLAY_ORDER:
        row_cells = []
        any_present = False
        for model, lmc in CELLS:
            bc = next((c for c in CITY_BINCODES[model] if BINCODES[c][0] == blabel), None)
            g = results[(model, lmc)].get(bc) if bc else None
            if g is not None:
                any_present = True
            row_cells.append(fmt_gap(g))
        if any_present:
            print(f"| {blabel}\" | " + " | ".join(row_cells) + " |")

    # -----------------------------------------------------------------------
    # Comparison: matched gap vs old full-bin gap (the unfair one)
    # -----------------------------------------------------------------------
    print("\n\n# Matched gap vs full-bin gap (the fairness correction)\n")
    print("Shows how much the gap changes once the holdout RMSE is restricted to the "
          "stratified-tested events. 'full' = old methodology (whole bin).\n")
    for model, lmc in CELLS:
        print(f"### {cell_label(model, lmc)}")
        print("| Bin | 1A ref (n) | matched 1B (n) | matched gap | full 1B (n) | full gap |")
        print("|---|---|---|---|---|---|")
        for bc in CITY_BINCODES[model]:
            g = results[(model, lmc)].get(bc)
            if g is None:
                print(f"| {BINCODES[bc][0]}\" | — | — | — | — | — |")
                continue
            print(f"| {g['label']}\" | {g['ref_1a']:.4f} (n={g['n_matched']}) | "
                  f"{g['matched_1b']:.4f} (n={g['n_matched']}) | "
                  f"{g['matched_gap']:+.4f} ({g['matched_gap_pct']:+.1f}%) | "
                  f"{g['full_1b']:.4f} (n={g['n_full']}) | "
                  f"{g['full_gap']:+.4f} ({g['full_gap_pct']:+.1f}%) |")
        print()

    # -----------------------------------------------------------------------
    # Per-seed gap spread (robustness)
    # -----------------------------------------------------------------------
    print("\n# Per-seed matched-gap spread\n")
    print("| Cell | Bin | per-seed gaps | mean ± std |")
    print("|---|---|---|---|")
    for model, lmc in CELLS:
        for bc in CITY_BINCODES[model]:
            g = results[(model, lmc)].get(bc)
            if g is None or not g["per_seed_gap"]:
                continue
            gaps = g["per_seed_gap"]
            spread = (f"{mean(gaps):+.4f} ± {stdev(gaps):.4f}"
                      if len(gaps) > 1 else f"{gaps[0]:+.4f} (1 seed)")
            seeds = ",".join(f"s{s}={v:+.4f}" for s, v in zip(g["seeds_used"], gaps))
            print(f"| {cell_label(model, lmc)} | {g['label']}\" | {seeds} | {spread} |")

    # -----------------------------------------------------------------------
    # Sanity check vs published numbers
    # -----------------------------------------------------------------------
    print("\n# Sanity check — 1A reference reproduces the published per-bin numbers\n")
    g46 = results[("Model_1", "nolc")].get("h46")
    g24 = results[("Model_1", "nolc")].get("h24")
    for g, expect, blab in ((g46, 0.0178, "4-6"), (g24, 0.0310, "2-4")):
        if g is None:
            print(f"- Model_1/nolc {blab}\": NOT AVAILABLE")
            continue
        ok = abs(g["ref_1a"] - expect) < 5e-4
        print(f"- Model_1/nolc {blab}\" 1A ref = {g['ref_1a']:.4f} (n={g['n_matched']}); "
              f"expected ≈ {expect:.4f} → {'OK' if ok else 'MISMATCH'}")

    if args.csv:
        out_csv = os.path.join(HERE, "v8_allbins_results.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "lmc", "bin", "n_matched", "ref_1a", "matched_1b",
                        "matched_gap", "matched_gap_pct", "n_full", "full_1b",
                        "full_gap", "full_gap_pct", "n_seeds"])
            for model, lmc in CELLS:
                for bc in CITY_BINCODES[model]:
                    g = results[(model, lmc)].get(bc)
                    if g is None:
                        continue
                    w.writerow([model, lmc, g["label"], g["n_matched"], g["ref_1a"],
                                g["matched_1b"], g["matched_gap"], g["matched_gap_pct"],
                                g["n_full"], g["full_1b"], g["full_gap"],
                                g["full_gap_pct"], len(g["seeds_used"])])
        print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
