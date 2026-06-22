"""
Comprehensive plots for UrbanFlood LMC experiment progression.
Covers: mask_inlets → restrict_to_2d → v3 GT conn → v4 antisym → v5 multi-step → v7 loss-side → City-2.
Run from: physicsnemo/examples/weather/flood_modeling/hydrographnet/
Output:  experiment_plots/
"""

import os, re, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

BASE = "/home/woody/iwi5/iwi5416h/urbanflood"
OUT  = os.path.join(os.path.dirname(__file__), "experiment_plots")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────
EXPS = {
    # label : (dir, infer_file, color, linestyle, group, has_lmc)
    "v2-nolc\n(baseline)":         ("outputs_1d_v2_nolc",             "infer_1595490.out",  "black",    "-",  "all",  False),
    "mask_inlets\n(0.0216)":        ("outputs_lc_inlet_mask_1d",       "infer_1598376.out",  "#e67e22",  "--", "diag", True),
    "restrict_to_2d\n(H2, 0.0234)": ("outputs_lc_h2_1d",              "infer_1598375.out",  "#e74c3c",  "-.", "diag", True),
    "GT conn ΔQ\n(Exp1, 0.0207)":   ("outputs_lc_gt_conn_1d",         "infer_1599307.out",  "#2980b9",  "-",  "v3",   True),
    "GT conn+dyn\n(Exp2, 0.0217)":  ("outputs_lc_gt_conn_dynedge_1d", "infer_1599308.out",  "#85c1e9",  "--", "v3",   True),
    "Antisym\n(Exp3, 0.0231)":      ("outputs_lc_antisym_1d",         "infer_1600072.out",  "#8e44ad",  "-",  "v4",   True),
    "Antisym+Concat\n(Exp4, 0.0235)":("outputs_lc_antisym_concat_1d","infer_1600073.out",  "#d2b4de",  "--", "v4",   True),
    "Multi-step LMC\n(Exp5, 0.0249)":("outputs_lc_v5_multi_lmc",     "infer_1600855.out",  "#27ae60",  "-",  "v5",   True),
    "Multi-step nolc\n(Exp6, 0.0269)":("outputs_lc_v5_multi_nolc",   "infer_1602526.out",  "#a9dfbf",  "--", "v5",   False),
    "Per-type wt\n(Exp7, 0.0223)":  ("outputs_lc_v7_pertype_1d",      "infer_1645616.out",  "#c0392b",  "-",  "v7",   True),
    "Restrict 2D\n(Exp8, 0.0225)":  ("outputs_lc_v7_2donly_1d",       "infer_1645618.out",  "#f1948a",  "--", "v7",   True),
    "City-2 LMC\n(0.0140)":         ("outputs_lc_gt_conn_1d_city2",   "infer_1664051.out",  "#1abc9c",  "-",  "c2",   True),
    "City-2 nolc\n(0.0149)":        ("outputs_1d_v2_nolc_city2",      "infer_1664052.out",  "#76d7c4",  "--", "c2",   False),
}

STEPS = list(range(1, 9))

# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────
def parse_infer_2d(path):
    """Return per-step 2D RMSE as a list of 8 floats (extract from tensor brackets)."""
    bracket_re = re.compile(r"\[([^\]]+)\]")
    with open(path) as f:
        for line in f:
            if "Overall Mean RMSE — 2D nodes" in line:
                m = bracket_re.search(line)
                if m:
                    return [float(x) for x in re.findall(r"[\d.]+(?:e[+-]?\d+)?", m.group(1))]
    # fallback: overall (no 2D breakdown in very old files)
    with open(path) as f:
        for line in f:
            if "Overall Mean RMSE (m) over rollout steps" in line:
                m = bracket_re.search(line)
                if m:
                    return [float(x) for x in re.findall(r"[\d.]+(?:e[+-]?\d+)?", m.group(1))]
    return None

def parse_train_log(path):
    """Return dict of lists: epoch, pred_mse, edge_loss, lmc_loss.
    pred_mse = loss_one + loss_stability  (or pred_loss for multi-step).
    """
    epochs, pred_mse, edge_loss, lmc_loss = [], [], [], []
    ep_re = re.compile(r"Epoch (\d+) — Avg Loss")
    with open(path) as f:
        for line in f:
            if not ep_re.search(line):
                continue
            ep = int(ep_re.search(line).group(1))
            # prediction MSE
            lo = re.search(r"loss_one:\s*([\d.e+\-]+)", line)
            ls = re.search(r"loss_stability:\s*([\d.e+\-]+)", line)
            pl = re.search(r"pred_loss:\s*([\d.e+\-]+)", line)
            el = re.search(r"edge_loss:\s*([\d.e+\-]+)", line)
            ll = re.search(r"local_physics_loss:\s*([\d.e+\-]+)", line)
            if pl:
                mse = float(pl.group(1))
            elif lo and ls:
                mse = float(lo.group(1)) + float(ls.group(1))
            else:
                continue
            epochs.append(ep)
            pred_mse.append(mse)
            edge_loss.append(float(el.group(1)) if el else float("nan"))
            lmc_loss.append(float(ll.group(1)) if ll else float("nan"))
    return {
        "epoch": np.array(epochs),
        "pred_mse": np.array(pred_mse),
        "edge_loss": np.array(edge_loss),
        "lmc_loss": np.array(lmc_loss),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────
rmse_data = {}   # label -> [8 floats]
train_data = {}  # label -> dict

for label, (d, infer_f, color, ls, group, has_lmc) in EXPS.items():
    infer_path = os.path.join(BASE, d, infer_f)
    train_path = os.path.join(BASE, d, "train.log")
    if os.path.exists(infer_path):
        rmse_data[label] = parse_infer_2d(infer_path)
    if os.path.exists(train_path):
        train_data[label] = parse_train_log(train_path)

# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_LABEL = "v2-nolc\n(baseline)"
BASELINE_COLOR = "black"

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Rollout curves: ALL Model_1 experiments (one panel)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(0.0195, color="black", lw=1.5, ls="--", label="v2-nolc target 0.0195 m")

for label, (d, infer_f, color, ls, group, has_lmc) in EXPS.items():
    if "city2" in d or label == BASELINE_LABEL:
        continue
    if label not in rmse_data or rmse_data[label] is None:
        continue
    lw = 2.5 if label == BASELINE_LABEL else 1.5
    ax.plot(STEPS, rmse_data[label], color=color, ls=ls, lw=lw,
            marker="o", ms=4, label=label.replace("\n", " "))

if BASELINE_LABEL in rmse_data:
    ax.plot(STEPS, rmse_data[BASELINE_LABEL], color="black", ls="-", lw=2.5,
            marker="o", ms=5, label="v2-nolc (baseline)")

ax.set_xlabel("Rollout step", fontsize=12)
ax.set_ylabel("2D RMSE (m)", fontsize=12)
ax.set_title("Model_1: 2D rollout RMSE across all LMC experiments", fontsize=13)
ax.legend(fontsize=7.5, ncol=3, loc="upper left")
ax.set_xticks(STEPS)
ax.grid(True, alpha=0.3)
save(fig, "fig1_all_rollout_model1.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Rollout curves: grouped by experiment bundle (4 panels)
# ─────────────────────────────────────────────────────────────────────────────
GROUP_DEFS = [
    ("Diagnostics: mask_inlets & restrict_to_2d", ["diag"]),
    ("v3: GT connection ΔQ fix", ["v3"]),
    ("v4: Antisymmetry & concat-trick", ["v4"]),
    ("v5: Multi-step rollout training", ["v5"]),
    ("v7: Loss-mean levers (per-type, restrict_to_2d)", ["v7"]),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for idx, (title, groups) in enumerate(GROUP_DEFS):
    ax = axes[idx]
    # baseline always present
    if BASELINE_LABEL in rmse_data:
        ax.plot(STEPS, rmse_data[BASELINE_LABEL], color="black", ls="-", lw=2,
                marker="o", ms=4, label="v2-nolc baseline")
    ax.axhline(0.0195, color="black", lw=1, ls=":", alpha=0.5)
    for label, (d, infer_f, color, ls, group, has_lmc) in EXPS.items():
        if group not in groups or label not in rmse_data:
            continue
        ax.plot(STEPS, rmse_data[label], color=color, ls=ls, lw=1.8,
                marker="o", ms=4, label=label.replace("\n", " "))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Rollout step", fontsize=9)
    ax.set_ylabel("2D RMSE (m)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(STEPS)
    ax.grid(True, alpha=0.3)

# City-2 in last panel
ax = axes[5]
for label, (d, infer_f, color, ls, group, has_lmc) in EXPS.items():
    if group != "c2" or label not in rmse_data:
        continue
    ax.plot(STEPS, rmse_data[label], color=color, ls=ls, lw=1.8,
            marker="o", ms=4, label=label.replace("\n", " "))
ax.set_title("City-2 (Model_2): LMC flips sign", fontsize=10)
ax.set_xlabel("Rollout step", fontsize=9)
ax.set_ylabel("2D RMSE (m)", fontsize=9)
ax.legend(fontsize=8, loc="upper left")
ax.set_xticks(STEPS)
ax.grid(True, alpha=0.3)

fig.suptitle("2D Rollout RMSE by Experiment Bundle — UrbanFlood", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig2_grouped_rollout_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Training loss curves: prediction MSE over epochs
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

TRAIN_GROUPS = [
    ("Diagnostics", ["diag"]),
    ("v3: GT conn fix", ["v3"]),
    ("v4: Antisymmetry", ["v4"]),
    ("v5: Multi-step", ["v5"]),
    ("v7: Loss-mean levers", ["v7"]),
    ("City-2 (Model_2)", ["c2"]),
]
for idx, (title, groups) in enumerate(TRAIN_GROUPS):
    ax = axes[idx]
    # baseline
    if BASELINE_LABEL in train_data:
        td = train_data[BASELINE_LABEL]
        ax.semilogy(td["epoch"], td["pred_mse"], color="black", lw=2, label="v2-nolc")
    for label, (d, infer_f, color, ls, group, has_lmc) in EXPS.items():
        if group not in groups or label not in train_data:
            continue
        td = train_data[label]
        ax.semilogy(td["epoch"], td["pred_mse"], color=color, ls=ls, lw=1.5,
                    label=label.replace("\n", " "))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Pred MSE (log scale)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

fig.suptitle("Training Prediction MSE over Epochs", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "fig3_training_pred_mse.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — LMC loss vs Pred MSE over training (the Pearson r story)
# Shows WHERE the conflict happens epoch-by-epoch
# ─────────────────────────────────────────────────────────────────────────────
LMC_RUN_LABELS = [
    "GT conn ΔQ\n(Exp1, 0.0207)",
    "Antisym\n(Exp3, 0.0231)",
    "Multi-step LMC\n(Exp5, 0.0249)",
    "Per-type wt\n(Exp7, 0.0223)",
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()
pearson_results = {}

WARMUP = 5  # epochs

for idx, label in enumerate(LMC_RUN_LABELS):
    ax = axes[idx]
    if label not in train_data:
        ax.set_visible(False)
        continue
    td = train_data[label]
    ep = td["epoch"]
    mse = td["pred_mse"]
    lmc = td["lmc_loss"]

    # Normalise both to [0,1] for visual overlay
    def norm01(x): return (x - x.min()) / (x.max() - x.min() + 1e-30)

    ax.plot(ep, norm01(mse), color="#2980b9", lw=2, label="Pred MSE (norm)")
    ax.plot(ep, norm01(lmc), color="#e74c3c", lw=2, ls="--", label="LMC loss (norm)")

    # Mark warmup
    ax.axvspan(0, WARMUP, alpha=0.08, color="grey", label=f"Warmup (ep 0-{WARMUP})")

    # Pearson r post-warmup
    mask = ep >= WARMUP
    if mask.sum() >= 3:
        r, pval = pearsonr(mse[mask], lmc[mask])
        pearson_results[label] = (r, pval, int(mask.sum()))
        ax.set_title(f"{label.replace(chr(10), ' ')}\nr(MSE, LMC) = {r:+.3f} post-warmup (n={mask.sum()})",
                     fontsize=9)
    else:
        ax.set_title(label.replace("\n", " "), fontsize=9)

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Normalised value", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("LMC Loss vs Prediction MSE During Training\n(normalised to [0,1] for overlay; r measured post-warmup)",
             fontsize=12)
fig.tight_layout()
save(fig, "fig4_lmc_vs_mse_correlation.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Bar chart: final 2D RMSE across all Model_1 experiments
# ─────────────────────────────────────────────────────────────────────────────
bar_labels = []
bar_vals   = []
bar_colors = []

ordered_labels = [
    "v2-nolc\n(baseline)",
    "mask_inlets\n(0.0216)",
    "restrict_to_2d\n(H2, 0.0234)",
    "GT conn ΔQ\n(Exp1, 0.0207)",
    "GT conn+dyn\n(Exp2, 0.0217)",
    "Antisym\n(Exp3, 0.0231)",
    "Antisym+Concat\n(Exp4, 0.0235)",
    "Multi-step LMC\n(Exp5, 0.0249)",
    "Multi-step nolc\n(Exp6, 0.0269)",
    "Per-type wt\n(Exp7, 0.0223)",
    "Restrict 2D\n(Exp8, 0.0225)",
]

for label in ordered_labels:
    if label in rmse_data and rmse_data[label] is not None:
        bar_labels.append(label.replace("\n", "\n"))
        bar_vals.append(np.mean(rmse_data[label]))
        bar_colors.append(EXPS[label][2])

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(bar_labels))
bars = ax.bar(x, bar_vals, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.65)
ax.axhline(0.0195, color="black", lw=1.8, ls="--", label="v2-nolc target 0.0195 m", zorder=5)

for bar, val in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel("Mean 2D RMSE — 8-step rollout (m)", fontsize=10)
ax.set_title("Model_1: Final 2D RMSE across all experiments (lower = better)", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0.015, 0.032)
ax.grid(True, alpha=0.25, axis="y")
save(fig, "fig5_bar_final_rmse.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — City-2 comparison: per-step + LMC benefit accumulates with horizon
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: per-step 2D RMSE, Model_1 vs Model_2, LMC vs nolc
ax = axes[0]
city1_lmc_label  = "GT conn ΔQ\n(Exp1, 0.0207)"
city1_nolc_label = "v2-nolc\n(baseline)"
city2_lmc_label  = "City-2 LMC\n(0.0140)"
city2_nolc_label = "City-2 nolc\n(0.0149)"

for lbl, color, ls, marker in [
    (city1_nolc_label, "black",   "-",  "o"),
    (city1_lmc_label,  "#2980b9", "--", "s"),
    (city2_nolc_label, "#76d7c4", "-",  "o"),
    (city2_lmc_label,  "#1abc9c", "--", "s"),
]:
    if lbl in rmse_data and rmse_data[lbl] is not None:
        ax.plot(STEPS, rmse_data[lbl], color=color, ls=ls, lw=2, marker=marker, ms=5,
                label=lbl.replace("\n", " "))

ax.set_xlabel("Rollout step", fontsize=11)
ax.set_ylabel("2D RMSE (m)", fontsize=11)
ax.set_title("Model_1 vs Model_2: LMC sign flips\nwith coupling density", fontsize=11)
ax.legend(fontsize=8)
ax.set_xticks(STEPS)
ax.grid(True, alpha=0.3)

# Right: LMC benefit (nolc - lmc) per step for both cities
ax = axes[1]
if city2_nolc_label in rmse_data and city2_lmc_label in rmse_data:
    c2_benefit = np.array(rmse_data[city2_nolc_label]) - np.array(rmse_data[city2_lmc_label])
    ax.plot(STEPS, c2_benefit*1000, color="#1abc9c", lw=2, marker="o", ms=5,
            label="City-2: nolc − lmc (positive = LMC helps)")
if city1_nolc_label in rmse_data and city1_lmc_label in rmse_data:
    c1_benefit = np.array(rmse_data[city1_nolc_label]) - np.array(rmse_data[city1_lmc_label])
    ax.plot(STEPS, c1_benefit*1000, color="#2980b9", lw=2, marker="s", ms=5, ls="--",
            label="City-1: nolc − lmc (negative = LMC hurts)")

ax.axhline(0, color="black", lw=1, ls=":")
ax.set_xlabel("Rollout step", fontsize=11)
ax.set_ylabel("RMSE benefit of LMC (mm)", fontsize=11)
ax.set_title("LMC benefit accumulates with rollout horizon\n(City-2 only)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xticks(STEPS)
ax.grid(True, alpha=0.3)

fig.suptitle("City-2: LMC flips to beneficial — coupling density governs LMC sign", fontsize=12)
fig.tight_layout()
save(fig, "fig6_city2_lmc_benefit.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Edge loss + LMC loss together over training (shows warmup ramp)
# For Exp1 (gt_conn): shows full loss breakdown
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

target_labels = [
    ("GT conn ΔQ\n(Exp1, 0.0207)", axes[0]),
    ("v2-nolc\n(baseline)",        axes[1]),
]
for label, ax in target_labels:
    if label not in train_data:
        ax.set_visible(False); continue
    td = train_data[label]
    ep = td["epoch"]
    ax.semilogy(ep, td["pred_mse"],  color="#2980b9", lw=2, label="Pred MSE (loss_one+stab)")
    if not np.all(np.isnan(td["edge_loss"])):
        ax.semilogy(ep, td["edge_loss"], color="#e67e22", lw=1.5, ls="--", label="Edge loss")
    if not np.all(np.isnan(td["lmc_loss"])):
        ax.semilogy(ep, td["lmc_loss"],  color="#e74c3c", lw=1.5, ls="-.", label="LMC loss")
    ax.axvspan(0, WARMUP, alpha=0.08, color="grey", label="LMC warmup")
    ax.set_title(label.replace("\n", " "), fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss (log scale)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

fig.suptitle("Loss component breakdown during training", fontsize=12)
fig.tight_layout()
save(fig, "fig7_loss_breakdown.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG_V5 — v5 Multi-step rollout: "Training metrics by epoch" breakdown
# Matches the 4-curve / curriculum-boundaries / annotated style
# ─────────────────────────────────────────────────────────────────────────────
v5_lmc_key  = "Multi-step LMC\n(Exp5, 0.0249)"
v5_nolc_key = "Multi-step nolc\n(Exp6, 0.0269)"

if v5_lmc_key in train_data and v5_nolc_key in train_data:
    td_lmc  = train_data[v5_lmc_key]
    td_nolc = train_data[v5_nolc_key]
    ep_l    = td_lmc["epoch"]
    ep_n    = td_nolc["epoch"]

    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.plot(ep_l, td_lmc["pred_mse"],  "-o", color="#1f77b4", ms=3.5, lw=1.6, label="LMC — Pred MSE")
    ax.plot(ep_l, td_lmc["edge_loss"], "-o", color="#ff7f0e", ms=3.5, lw=1.6, label="LMC — Edge loss")
    ax.plot(ep_l, td_lmc["lmc_loss"],  "-o", color="#2ca02c", ms=3.5, lw=1.6, label="LMC — LMC loss")
    ax.plot(ep_n, td_nolc["pred_mse"], "-o", color="#d62728", ms=3.5, lw=1.6, label="nolc — Pred MSE")
    ax.set_yscale("log")

    # Warmup boundary (dotted) + curriculum transitions (dashed)
    ax.axvline(WARMUP, color="#5b9bd5", lw=1.3, ls="--", alpha=0.75)
    for cur_ep in [12, 25, 37]:
        ax.axvline(cur_ep, color="#5b9bd5", lw=1.0, ls="--", alpha=0.5)

    # Curriculum labels at top using mixed (data-x, axes-y) transform
    ax.text(WARMUP + 0.5, 0.97, "warmup done", fontsize=8, color="#444",
            rotation=90, va="top", ha="left",
            transform=ax.get_xaxis_transform())
    for cur_ep, cur_lbl in [(12, "O=1→2"), (25, "O=2→4"), (37, "O=4→8")]:
        ax.text(cur_ep + 0.5, 0.97, cur_lbl, fontsize=7.5, color="#5b9bd5",
                rotation=90, va="top", ha="left",
                transform=ax.get_xaxis_transform())

    # LMC minimum — lowest point in the fully-warmed O=1 phase (ep WARMUP–11)
    o1_mask = (ep_l >= WARMUP) & (ep_l < 12)
    lmc_o1  = np.where(o1_mask, td_lmc["lmc_loss"], np.inf)
    min_idx = int(np.argmin(lmc_o1))
    min_ep  = int(ep_l[min_idx])
    min_val = float(td_lmc["lmc_loss"][min_idx])
    ax.annotate("LMC minimum",
                xy=(min_ep, min_val),
                xytext=(min_ep - 4, min_val * 3.5),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.9,
                                connectionstyle="arc3,rad=0.15"))

    # LMC loss rises — label the first post-switch climb (ep 13 onward)
    rise_ep  = 15
    rise_idx = np.where(ep_l == rise_ep)[0]
    if len(rise_idx):
        rise_val = float(td_lmc["lmc_loss"][rise_idx[0]])
        ax.annotate("LMC loss rises",
                    xy=(rise_ep, rise_val),
                    xytext=(rise_ep + 4, rise_val * 2.8),
                    fontsize=8, color="#333",
                    arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss / MSE (log scale)", fontsize=11)
    ax.set_title("Training metrics by epoch", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlim(-1, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "fig_v5_multistep_training_metrics.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print Pearson r table + key insights
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PEARSON r(pred_MSE, LMC_loss) post-warmup — key diagnostic")
print("="*70)
print(f"{'Experiment':<40} {'r':>8} {'p-val':>10} {'n':>5}  Interpretation")
print("-"*70)

ALL_LMC_TRAIN = {
    "mask_inlets\n(0.0216)":         "mask_inlets",
    "restrict_to_2d\n(H2, 0.0234)":  "restrict_2d",
    "GT conn ΔQ\n(Exp1, 0.0207)":    "gt_conn_exp1",
    "Antisym\n(Exp3, 0.0231)":       "antisym",
    "Multi-step LMC\n(Exp5, 0.0249)":"multi_lmc",
    "Per-type wt\n(Exp7, 0.0223)":   "pertype",
    "Restrict 2D\n(Exp8, 0.0225)":   "restrict2d_v7",
}

for label in EXPS:
    if not EXPS[label][5]:  # not LMC
        continue
    if label not in train_data:
        continue
    td = train_data[label]
    ep, mse, lmc = td["epoch"], td["pred_mse"], td["lmc_loss"]
    if np.all(np.isnan(lmc)):
        continue
    mask = ep >= WARMUP
    if mask.sum() < 3:
        continue
    r, pval = pearsonr(mse[mask], lmc[mask])
    interp = "LMC helps (aligned)" if r > 0.9 else ("Conflict — LMC hurts" if r < 0 else "Weak alignment")
    name = label.replace("\n", " ")
    print(f"  {name:<38} {r:+.3f}  {pval:.2e}  {mask.sum():>3}  {interp}")

print()
print("="*70)
print("KEY INSIGHTS from rollout curves")
print("="*70)

# Compute interesting numbers
if "GT conn ΔQ\n(Exp1, 0.0207)" in rmse_data and BASELINE_LABEL in rmse_data:
    exp1 = np.array(rmse_data["GT conn ΔQ\n(Exp1, 0.0207)"])
    base = np.array(rmse_data[BASELINE_LABEL])
    gap_s1 = (exp1[0] - base[0]) * 1000
    gap_s8 = (exp1[7] - base[7]) * 1000
    print(f"  Exp1 vs baseline gap: step1={gap_s1:+.1f}mm, step8={gap_s8:+.1f}mm  (gap GROWS with rollout)")

if "City-2 LMC\n(0.0140)" in rmse_data and "City-2 nolc\n(0.0149)" in rmse_data:
    c2l = np.array(rmse_data["City-2 LMC\n(0.0140)"])
    c2n = np.array(rmse_data["City-2 nolc\n(0.0149)"])
    diffs = (c2n - c2l) * 1000
    print(f"  City-2 LMC benefit by step: {[f'{d:+.1f}' for d in diffs]}")
    print(f"    → Benefit accumulates: step1={diffs[0]:+.1f}mm → step8={diffs[7]:+.1f}mm")

if "Multi-step LMC\n(Exp5, 0.0249)" in rmse_data and "Multi-step nolc\n(Exp6, 0.0269)" in rmse_data:
    ml = np.mean(rmse_data["Multi-step LMC\n(Exp5, 0.0249)"])
    mn = np.mean(rmse_data["Multi-step nolc\n(Exp6, 0.0269)"])
    print(f"  Multi-step: LMC={ml:.4f}, nolc={mn:.4f}  → LMC wins by {(mn-ml)*1000:.1f}mm in multi-step world")
    b = np.mean(rmse_data[BASELINE_LABEL])
    print(f"    But single-step nolc={b:.4f} still beats both!")

print()
print(f"All figures saved to: {OUT}/")
