"""Regenerate the 5 publication-quality figures for UrbanFlood v8 Phase 1.

One invocation -> 5 figures, each saved as PNG (300 dpi, for slides) and PDF
(for paper-quality print) under
`/home/woody/iwi5/iwi5416h/urbanflood/outputs_v8/plots/`.

All inputs already exist on disk (no GPU, no retraining):
  * v8_results.csv          -- per-step 2D RMSE per experiment
  * event_intensity.csv     -- per-event total inches + intensity bin
  * outputs_v8/uf_v8_infer_<exp>_<jobid>.out -- per-event RMSE (overall & 2D)
  * data_v8/<exp>/splits.json -- per-experiment train/test event IDs + bin counts

Governing rule: 2D RMSE is the only success metric. Every per-event quantity
plotted here is the **2D-only** RMSE (the `| 2D = X m` field of each event line),
which keeps all five figures on one consistent, rule-compliant metric.

Reuses parsers/constants from aggregate_v8_results.py; adds one parser for the
per-event 2D field (the aggregator's parse_event_line only reads the leading
overall value).
"""

import csv
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import aggregate_v8_results as agg


# ----------------------------------------------------------------------------
# Constants / config
# ----------------------------------------------------------------------------
PLOTS_DIR = os.path.join(agg.OUTPUTS_ROOT, "plots")
V8_RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "v8_results.csv")
V2_NOLC_REF = 0.0195  # full-pool v2-nolc 2D RMSE reference (m)

# Sub-experiment grouping: prefix -> (label, color, linestyle, marker).
SUBEXP = {
    "p1a_": dict(label="Phase 1A — stratified 80/20",
                 color="#1f77b4", ls="-", marker="o"),
    "p1b_h46_": dict(label="Phase 1B-1 — holdout 4-6\"",
                     color="#d62728", ls="--", marker="s"),
    "p1b_h24_": dict(label="Phase 1B-2 — holdout 2-4\"",
                     color="#2ca02c", ls=":", marker="^"),
}
SUBEXP_ORDER = ["p1a_", "p1b_h46_", "p1b_h24_"]

BINS = ["0-2", "2-4", "4-6", "6-8", "8-10"]
BIN_EDGES = [0, 2, 4, 6, 8, 10]

# Per-event 2D-only RMSE parser (aggregator only captures the overall value).
_EVENT_2D_RE = re.compile(
    r"Event\s+event_(\d+):.*?\|\s*2D\s*=\s*([\d.eE+-]+)\s+m")


# ----------------------------------------------------------------------------
# Parsing / data assembly
# ----------------------------------------------------------------------------
def parse_event_2d(path: str) -> dict:
    """Return {event_id: 2D-only RMSE (m)} parsed from an inference .out log."""
    out = {}
    with open(path) as f:
        for line in f:
            if "findfont" in line:
                continue
            m = _EVENT_2D_RE.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def load_step_curves() -> list:
    """Per-experiment per-step 2D RMSE rows from v8_results.csv."""
    rows = []
    with open(V8_RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            r["steps"] = [float(r[f"step{i}_2d_rmse_m"]) for i in range(1, 9)]
            rows.append(r)
    return rows


def subexp_of(exp: str) -> str:
    for prefix in SUBEXP_ORDER:
        if exp.startswith(prefix):
            return prefix
    raise ValueError(exp)


def assemble_per_event() -> dict:
    """exp -> {event_id: 2D RMSE}. Asserts count == n_test for each exp."""
    intensity = agg.load_intensity()
    per_exp = {}
    for exp in agg.EXP_NAMES:
        ev2d = parse_event_2d(agg._find_out_file(exp))
        splits = agg.load_splits(exp)
        test_ids = splits["test_event_ids"]
        n_test = splits["num_test"]
        assert len(ev2d) == n_test, (
            f"{exp}: parsed {len(ev2d)} per-event 2D lines, expected "
            f"n_test={n_test}")
        # Sanity: every parsed event should be a known test event with intensity.
        for ev in ev2d:
            assert ev in intensity, f"{exp}: event {ev} missing from intensity csv"
            assert ev in test_ids, f"{exp}: event {ev} not in test_event_ids"
        per_exp[exp] = ev2d
    return per_exp


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def setup_style():
    """Serif fonts with graceful fallback; never Times (avoids findfont noise)."""
    available = {f.name for f in fm.fontManager.ttflist}
    serif_pref = [f for f in ("CMU Serif", "Computer Modern Roman",
                              "DejaVu Serif") if f in available]
    if not serif_pref:
        serif_pref = ["DejaVu Serif"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": serif_pref,
        "mathtext.fontset": "cm",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
    })
    print(f"[style] serif fonts: {serif_pref}")


def save(fig, name: str):
    png = os.path.join(PLOTS_DIR, f"{name}.png")
    pdf = os.path.join(PLOTS_DIR, f"{name}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png}\n[saved] {pdf}")


# ----------------------------------------------------------------------------
# Figure 1 -- rollout curves
# ----------------------------------------------------------------------------
def fig_rollout_curves(step_rows):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    steps = list(range(1, 9))
    for r in step_rows:
        s = SUBEXP[subexp_of(r["exp"])]
        ax.plot(steps, r["steps"], color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=4, lw=1.3, alpha=0.55)
    ax.axhline(V2_NOLC_REF, color="black", ls="-.", lw=1.4, alpha=0.8)

    handles = [Line2D([0], [0], color=SUBEXP[p]["color"], ls=SUBEXP[p]["ls"],
                      marker=SUBEXP[p]["marker"], lw=1.8, label=SUBEXP[p]["label"])
               for p in SUBEXP_ORDER]
    handles.append(Line2D([0], [0], color="black", ls="-.", lw=1.4,
                          label=f"v2-nolc full-pool ref = {V2_NOLC_REF:.4f} m"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    ax.set_xticks(steps)
    ax.set_xticklabels([f"{i}\n(t+{5*i}min)" for i in steps])
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("2D RMSE (m)")
    ax.set_title("Per-step 2D RMSE over the 8-step rollout (all 9 runs)")
    ax.set_ylim(bottom=0)
    save(fig, "rollout_curves")


# ----------------------------------------------------------------------------
# Figure 2 -- interpolation gap (the RQ1 headline)
# ----------------------------------------------------------------------------
def fig_interpolation_gap(per_event):
    intensity = agg.load_intensity()

    # Phase 1A: pool per-event 2D across the 3 seeds, by bin.
    p1a_bin = defaultdict(list)
    for exp in ("p1a_s0", "p1a_s1", "p1a_s2"):
        for ev, rmse in per_event[exp].items():
            p1a_bin[intensity[ev][1]].append(rmse)

    groups = [("4-6", "p1b_h46_"), ("2-4", "p1b_h24_")]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    width = 0.38
    xs = range(len(groups))

    print("\n## Figure 2 -- interpolation gap (2D-only)")
    print("| Held-out bin | Phase 1A 2D RMSE | Phase 1B 2D RMSE (mean±std) | gap |")
    print("|---|---|---|---|")

    bar_tops = []
    for i, (b, prefix) in enumerate(groups):
        a_val = mean(p1a_bin[b])
        # Phase 1B: per-seed mean over the whole held-out test set, then across seeds.
        seed_means = [mean(per_event[e].values())
                      for e in agg.EXP_NAMES if e.startswith(prefix)]
        b_val, b_std = mean(seed_means), stdev(seed_means)
        gap = b_val - a_val
        pct = 100 * gap / a_val

        ax.bar(i - width / 2, a_val, width, color="#1f77b4",
               label="Phase 1A (on this bin)" if i == 0 else None,
               edgecolor="black", lw=0.6)
        ax.bar(i + width / 2, b_val, width, yerr=b_std, capsize=5,
               color="#d62728" if b == "4-6" else "#2ca02c",
               label="Phase 1B (held out)" if i == 0 else None,
               edgecolor="black", lw=0.6)

        bar_tops.append((i, max(a_val, b_val + b_std), gap, pct))
        print(f"| {b}\" | {a_val:.4f} | {b_val:.4f} ± {b_std:.4f} | "
              f"{gap:+.4f} m ({pct:+.1f}%) |")

    # Headroom first, then place each gap label a fixed clearance above its bars.
    ymax = max(t for _, t, _, _ in bar_tops)
    ax.set_ylim(0, ymax * 1.32)
    for i, top, gap, pct in bar_tops:
        ax.annotate(f"gap {gap:+.4f} m ({pct:+.1f}%)",
                    xy=(i, top), xytext=(i, top + 0.04 * ymax),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{b}\" held out" for b, _ in groups])
    ax.set_xlabel("Held-out rainfall-intensity bin")
    ax.set_ylabel("2D RMSE (m)")
    ax.set_title("Interpolation gap: held-out bin vs.\nsame bin under stratified split",
                 fontsize=13)
    ax.legend(loc="upper left", framealpha=0.9)
    save(fig, "interpolation_gap")


# ----------------------------------------------------------------------------
# Figure 3 -- per-event 2D RMSE vs total rainfall
# ----------------------------------------------------------------------------
def fig_per_event_vs_intensity(per_event):
    intensity = agg.load_intensity()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Held-out bins shaded as vertical bands.
    ax.axvspan(2, 4, color="gray", alpha=0.15, zorder=0)
    ax.axvspan(4, 6, color="lightblue", alpha=0.30, zorder=0)

    for prefix in SUBEXP_ORDER:
        s = SUBEXP[prefix]
        xs, ys = [], []
        for exp in [e for e in agg.EXP_NAMES if e.startswith(prefix)]:
            for ev, rmse in per_event[exp].items():
                xs.append(intensity[ev][0])
                ys.append(rmse)
        ax.scatter(xs, ys, c=s["color"], marker=s["marker"], s=34,
                   alpha=0.7, edgecolors="white", linewidths=0.4,
                   label=s["label"], zorder=3)

    band_handles = [Patch(facecolor="gray", alpha=0.15, label="2-4\" held out"),
                    Patch(facecolor="lightblue", alpha=0.30,
                          label="4-6\" held out")]
    leg1 = ax.legend(loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=band_handles, loc="upper right", framealpha=0.9)

    ax.set_xlabel("Total event rainfall (inches)")
    ax.set_ylabel("Per-event 2D RMSE (m)")
    ax.set_title("Per-event 2D RMSE vs rainfall intensity")
    ax.set_ylim(bottom=0)
    save(fig, "per_event_rmse_vs_intensity")


# ----------------------------------------------------------------------------
# Figure 4 -- intensity distribution of the 68 train-pool events
# ----------------------------------------------------------------------------
def fig_intensity_distribution():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "event_intensity.csv")
    counts = defaultdict(int)
    for r in csv.DictReader(open(csv_path)):
        if r["model_name"] == "Model_1" and r["split"] == "train":
            counts[r["bin"]] += 1
    total = sum(counts.values())
    assert total == 68, f"expected 68 train-pool events, got {total}"

    centers = [(BIN_EDGES[i] + BIN_EDGES[i + 1]) / 2 for i in range(len(BINS))]
    vals = [counts[b] for b in BINS]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(centers, vals, width=1.9, edgecolor="black", lw=0.8,
                  color="#9ecae1")
    # Highlight the two Phase 1B holdout bins.
    for b, bar in zip(BINS, bars):
        if b == "4-6":      # 1B-1
            bar.set_color("#d62728")
            bar.set_alpha(0.75)
        elif b == "2-4":    # 1B-2
            bar.set_facecolor("#2ca02c")
            bar.set_alpha(0.55)
            bar.set_hatch("//")

    for c, v in zip(centers, vals):
        ax.annotate(str(v), xy=(c, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    legend_handles = [
        Patch(facecolor="#d62728", alpha=0.75, edgecolor="black",
              label="4-6\" — Phase 1B-1 holdout"),
        Patch(facecolor="#2ca02c", alpha=0.55, hatch="//", edgecolor="black",
              label="2-4\" — Phase 1B-2 holdout"),
        Patch(facecolor="#9ecae1", edgecolor="black",
              label="other bins (always in train)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    ax.set_xticks(BIN_EDGES)
    ax.set_xlabel("Total event rainfall (inches)")
    ax.set_ylabel("Number of events")
    ax.set_title(f"Model_1 training-pool intensity distribution (n={total})")
    ax.set_ylim(0, max(vals) * 1.18)
    save(fig, "intensity_distribution")


# ----------------------------------------------------------------------------
# Figure 5 -- Phase 1A per-seed per-bin variance story
# ----------------------------------------------------------------------------
def fig_p1a_seed_comparison(per_event):
    intensity = agg.load_intensity()
    seeds = ["p1a_s0", "p1a_s1", "p1a_s2"]
    seed_colors = {"p1a_s0": "#4c78a8", "p1a_s1": "#e45756", "p1a_s2": "#54a24b"}

    # (seed, bin) -> list of per-event 2D RMSE.
    cell = {sd: defaultdict(list) for sd in seeds}
    for sd in seeds:
        for ev, rmse in per_event[sd].items():
            cell[sd][intensity[ev][1]].append(rmse)

    # Drop bins with no Phase 1A test events in any seed (e.g. 8-10": always train).
    bins = [b for b in BINS if any(cell[sd][b] for sd in seeds)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.26
    x = range(len(bins))

    for j, sd in enumerate(seeds):
        offs = (j - 1) * width
        means = [mean(cell[sd][b]) if cell[sd][b] else 0.0 for b in bins]
        ns = [len(cell[sd][b]) for b in bins]
        bars = ax.bar([i + offs for i in x], means, width,
                      color=seed_colors[sd], edgecolor="black", lw=0.6,
                      label=f"seed {sd[-1]}")
        for bar, n, m in zip(bars, ns, means):
            if n == 0:
                continue
            ax.annotate(f"n={n}", xy=(bar.get_x() + bar.get_width() / 2, m),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}\"" for b in bins])
    ax.set_xlabel("Rainfall-intensity bin")
    ax.set_ylabel("Phase 1A per-event 2D RMSE (m)")
    ax.set_title("Phase 1A per-bin 2D RMSE by seed "
                 "(seed s1 draws an unusually hard test set)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)
    save(fig, "p1a_seed_comparison")


# ----------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    step_rows = load_step_curves()
    per_event = assemble_per_event()
    print(f"[ok] parsed per-event 2D RMSE for {len(per_event)} experiments")

    fig_rollout_curves(step_rows)
    fig_interpolation_gap(per_event)
    fig_per_event_vs_intensity(per_event)
    fig_intensity_distribution()
    fig_p1a_seed_comparison(per_event)

    print(f"\nAll figures written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
