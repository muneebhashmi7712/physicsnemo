"""
Generate a markdown report + plots from a val_breakdown.csv produced by
analyze_val_breakdown.py.

Outputs (alongside the CSV):
  - validation_loss_breakdown_v1.md  (markdown report)
  - val_breakdown_components.png     (stacked components vs epoch)
  - val_breakdown_components_log.png (same, log y-axis)
  - val_breakdown_rollout.png        (val RMSE@K vs epoch for each K)
  - val_breakdown_rollout_vs_inf.png (overlay against existing inference RMSE
                                       per epoch from epoch_sweep, if available)

Usage:
  python make_val_breakdown_report.py \
    --csv /path/to/run/val_breakdown.csv \
    --run-name "v8 R1 (β=0.1, w=0.05, 50ep)" \
    --best-train-epoch 19 \
    --inf-rmse-json /path/to/run/epoch_sweep/inf_rmse_per_epoch.json   # optional

The --inf-rmse-json file (if provided) should be JSON of the form
  {"5": 0.118, "10": 0.115, "15": 0.111, "19": 0.118, ...}
mapping training-epoch (string) → average inference RMSE (float). When omitted,
the per-epoch inference RMSE overlay is skipped.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v) if v != "" else float("nan")
                except ValueError:
                    row[k] = v
            rows.append(row)
    rows.sort(key=lambda x: x["epoch"])
    return rows


def find_K_columns(fieldnames):
    """Returns a sorted list of K values present in the CSV header."""
    Ks = []
    for f in fieldnames:
        if f.startswith("rmse_K") and not f.endswith("_cum"):
            try:
                Ks.append(int(f.replace("rmse_K", "")))
            except ValueError:
                pass
    return sorted(Ks)


def plot_components(rows, out_path, log_y=False):
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [r["val_total"] for r in rows], "k-", lw=2.5, label="val_total")
    ax.plot(epochs, [r["loss_one"] for r in rows], "C0-", label="loss_one (1-step MSE)")
    ax.plot(epochs, [r["loss_stability"] for r in rows], "C1-", label="loss_stability (pushforward MSE)")
    ax.plot(epochs, [r["local_phys_weighted"] for r in rows], "C2-", label="local_phys (weighted)")
    ax.plot(epochs, [r["edge_reg_weighted"] for r in rows], "C3-", label="edge_reg (weighted)")
    ax.plot(epochs, [r["local_phys_raw"] for r in rows], "C2:", alpha=0.6, label="local_phys (RAW, unweighted)")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss component value")
    ax.set_title("Validation loss components vs epoch" + (" (log y)" if log_y else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rollout(rows, K_list, out_path):
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis([i / max(1, len(K_list) - 1) for i in range(len(K_list))])
    for K, c in zip(K_list, colors):
        col = f"rmse_K{K}"
        ax.plot(epochs, [r[col] for r in rows], "-", color=c, label=f"val_RMSE @ K={K}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE on val hydrographs (m)")
    ax.set_title("Multi-step rollout RMSE on val hydrographs vs epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rollout_vs_inf(rows, K_list, inf_rmse_per_epoch, out_path):
    epochs = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    K_max = max(K_list)
    ax1.plot(epochs, [r[f"rmse_K{K_max}"] for r in rows], "C0-",
             label=f"val_RMSE @ K={K_max} (val hydrographs)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("RMSE on val hydrographs (m)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    inf_eps = sorted(int(k) for k in inf_rmse_per_epoch)
    inf_vals = [inf_rmse_per_epoch[str(e)] for e in inf_eps]
    ax2.plot(inf_eps, inf_vals, "C3o-", label="test inference RMSE (45-step, H476–H500)")
    ax2.set_ylabel("RMSE on test hydrographs (m)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    ax1.set_title("val_RMSE@K=max vs test inference RMSE per training epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fmt(v, n=4):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    return f"{v:.{n}e}"


def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def build_report(csv_path, run_name, best_train_epoch, inf_rmse_per_epoch=None, out_md=None):
    rows = read_csv(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    K_list = find_K_columns(rows[0].keys())
    out_dir = Path(csv_path).parent
    if out_md is None:
        out_md = out_dir / "validation_loss_breakdown_v1.md"

    plot_components(rows, out_dir / "val_breakdown_components.png", log_y=False)
    plot_components(rows, out_dir / "val_breakdown_components_log.png", log_y=True)
    plot_rollout(rows, K_list, out_dir / "val_breakdown_rollout.png")
    if inf_rmse_per_epoch:
        plot_rollout_vs_inf(rows, K_list, inf_rmse_per_epoch, out_dir / "val_breakdown_rollout_vs_inf.png")

    n_epochs = len(rows)
    first_ep = int(rows[0]["epoch"])
    last_ep = int(rows[-1]["epoch"])

    # Component breakdown table
    comp_rows = []
    for r in rows:
        ep = int(r["epoch"])
        comp_rows.append([
            ep,
            fmt(r["val_total"]),
            fmt(r["loss_one"]),
            fmt(r["loss_stability"]),
            fmt(r["local_phys_raw"]),
            fmt(r["local_phys_weighted"]),
            fmt(r["edge_reg_weighted"]),
            f"{r['eff_weight']:.4f}",
            fmt(r["reconstruction_residual"], 2),
        ])

    # Rollout table
    rollout_rows = []
    for r in rows:
        ep = int(r["epoch"])
        rollout_rows.append([ep] + [fmt(r[f"rmse_K{K}"], 4) for K in K_list]
                            + [fmt(r[f"rmse_K{K}_cum"], 4) for K in K_list])

    # Find best by various criteria
    def best_by(col, minimize=True):
        valid = [r for r in rows if not math.isnan(r.get(col, float("nan")))]
        if not valid:
            return None, None
        chosen = min(valid, key=lambda r: r[col]) if minimize else max(valid, key=lambda r: r[col])
        return int(chosen["epoch"]), chosen[col]

    best_epochs = []
    for col in ["val_total", "loss_one", "loss_stability", "local_phys_raw"] + [f"rmse_K{K}" for K in K_list]:
        ep, v = best_by(col)
        best_epochs.append([col, ep, fmt(v, 4) if v is not None else "—"])

    md = []
    md.append(f"# Validation Loss Breakdown — {run_name}\n")
    md.append(f"Source CSV: `{csv_path}`")
    md.append(f"Epochs analyzed: {first_ep} → {last_ep} ({n_epochs} checkpoints), "
              f"K_list = {K_list}\n")
    md.append("## Plots\n")
    md.append(f"- ![Components vs epoch](val_breakdown_components.png)")
    md.append(f"- ![Components vs epoch (log y)](val_breakdown_components_log.png)")
    md.append(f"- ![Multi-step rollout RMSE vs epoch](val_breakdown_rollout.png)")
    if inf_rmse_per_epoch:
        md.append(f"- ![val_RMSE@K vs test inference RMSE](val_breakdown_rollout_vs_inf.png)")
    md.append("")

    md.append("## Best epoch by each metric\n")
    md.append(md_table(["metric", "best epoch", "best value"], best_epochs))

    md.append("\n## Per-epoch component breakdown\n")
    md.append(md_table(
        ["ep", "val_total", "loss_one", "loss_stab", "local_phys_raw",
         "local_phys_weighted", "edge_reg_weighted", "eff_w", "recon_resid"],
        comp_rows,
    ))

    md.append("\n## Per-epoch K-step rollout RMSE (val hydrographs)\n")
    K_hdr = [f"K={K}" for K in K_list] + [f"K={K} (cum)" for K in K_list]
    md.append(md_table(["ep"] + K_hdr, rollout_rows))

    md.append("\n## Reconstruction sanity check\n")
    md.append(
        "`reconstruction_residual = val_total - (loss_one + loss_stability + "
        "local_phys_weighted + edge_reg_weighted)`. Should be ≈ 0 (< 1e-6). "
        "Any non-trivial residual means the validate() decomposition is missing a term."
    )

    md.append("\n## Hypothesis tracker\n")
    md.append(
        f"With best_train_epoch = {best_train_epoch} (the inference-best from epoch_sweep), "
        "compare against best epochs above:\n"
    )
    md.append(
        "- **H1 (stability decoupling):** if `loss_one` keeps falling but `loss_stability` "
        "plateaus or rises after epoch ~20, the decoupling is between 1-step and 1-step-pushforward. "
        "This points toward extending pushforward in training or selecting checkpoints by `loss_stability`."
    )
    md.append(
        "- **H2 (conservation drift):** if `local_phys_raw` rises while `local_phys_weighted` stays small, "
        "the model is drifting from conservation under weighted shadowing. Fix: use unweighted residual "
        "(or large weight) as a checkpoint criterion."
    )
    md.append(
        "- **H3 (multi-step val gap):** if `rmse_K1` keeps falling but `rmse_K20`/`rmse_K45` start rising "
        "after epoch ~20, the val loss is structurally insufficient — switch checkpoint metric to "
        "`rmse_K20` (this is the highest-leverage fix)."
    )
    md.append(
        "- **H4 (val/test catchment shift):** if all components AND all `rmse_K*` keep falling on val, "
        "the gap is between val (H426–H475) and test (H476–H500). Fix is structural: hold out a catchment in val."
    )

    out_md = Path(out_md)
    out_md.write_text("\n".join(md))
    print(f"Report written: {out_md}")
    print(f"Plots in: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--best-train-epoch", type=int, default=-1,
                   help="Epoch at which inference RMSE is best (from epoch_sweep)")
    p.add_argument("--inf-rmse-json", default=None,
                   help="Optional JSON dict {epoch_str: avg_inf_rmse}")
    p.add_argument("--out-md", default=None)
    args = p.parse_args()

    inf = None
    if args.inf_rmse_json and os.path.exists(args.inf_rmse_json):
        with open(args.inf_rmse_json) as f:
            inf = json.load(f)

    build_report(args.csv, args.run_name, args.best_train_epoch, inf, args.out_md)


if __name__ == "__main__":
    main()
