"""
Per-checkpoint validation breakdown analysis.

For each saved per-epoch checkpoint of a training run:
  1. Re-runs validate() with per-component breakdown
     (loss_one, loss_stability, local_phys raw + weighted, edge_reg weighted, total).
  2. Performs K-step autoregressive rollout on the val hydrographs and records
     RMSE@K for K in K_list.

Output: a CSV with one row per epoch.

Usage (with Hydra overrides matching the original training run):

  python analyze_val_breakdown.py \
    data_dir=$TMPDIR/data \
    test_dir=$TMPDIR/data \
    use_local_physics_loss=true \
    local_physics_loss_weight=0.05 \
    use_edge_smoothness=false \
    local_loss_warmup_epochs=5 \
    use_boundary_masking=true \
    use_huber_residual=true \
    huber_beta=0.1 \
    graph_type=knn \
    node_loss_type=mse \
    epochs=50 \
    ckpt_path=/path/to/run/checkpoints \
    +analysis_run_dir=/path/to/run \
    +analysis_csv_out=/path/to/run/val_breakdown.csv \
    +analysis_first_epoch=0 \
    +analysis_last_epoch=49 \
    +analysis_K_list="[1,5,10,20,45]"

The +analysis_* fields are extra config keys (the leading "+" is Hydra syntax for
adding keys not in the base config).
"""

import csv
import math
import os
import sys
import time

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Reuse training-time setup.
from train import MGNTrainer
from physicsnemo.utils import load_checkpoint
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from metrics import compute_rmse


def build_rollout_val_dataset(cfg: DictConfig, rollout_length: int):
    """Val hydrographs (val.txt) wrapped in test-mode framing so each item gives
    (graph, rollout_data_dict) — rollout_data has inflow/precip/wd_gt/vol_gt for
    the next `rollout_length` steps. Mirrors what inference.py uses on test.txt.
    """
    return HydroGraphDataset(
        data_dir=cfg.data_dir,
        prefix="M80",
        n_time_steps=cfg.n_time_steps,
        hydrograph_ids_file="val.txt",
        split="test",
        rollout_length=rollout_length,
        return_physics=False,
        graph_type=cfg.get("graph_type", "knn"),
        make_bidirectional=cfg.get("use_local_physics_loss", False),
    )


@torch.no_grad()
def rollout_per_step_rmse(model, rollout_dataset, device, n_time_steps, K_max):
    """For each hydrograph in `rollout_dataset`, autoregressively predict K_max
    steps and return per-step RMSE on denormalized water depth. Returns a tensor
    of shape (num_hydrographs, K_max). Mirrors inference.py's rollout loop.
    """
    eps = 1e-8
    wd_mean = rollout_dataset.dynamic_stats["water_depth"]["mean"]
    wd_std = rollout_dataset.dynamic_stats["water_depth"]["std"]
    vol_mean = rollout_dataset.dynamic_stats["volume"]["mean"]
    vol_std = rollout_dataset.dynamic_stats["volume"]["std"]
    inflow_mean = rollout_dataset.dynamic_stats["inflow_hydrograph"]["mean"]
    inflow_std = rollout_dataset.dynamic_stats["inflow_hydrograph"]["std"]
    precip_mean = rollout_dataset.dynamic_stats["precipitation"]["mean"]
    precip_std = rollout_dataset.dynamic_stats["precipitation"]["std"]

    per_hydro_rmse = []
    for idx in range(len(rollout_dataset)):
        g, rollout_data = rollout_dataset[idx]
        g = g.to(device)
        edge_features = g.edge_attr.to(device)
        X_iter = g.x.to(device).clone()
        num_nodes = X_iter.size(0)

        inflow_seq = rollout_data["inflow"].to(device)
        precip_seq = rollout_data["precipitation"].to(device)
        wd_gt_seq = rollout_data["water_depth_gt"].to(device)

        rmses = []
        for t in range(K_max):
            static_part = X_iter[:, :12]
            water_depth_window = X_iter[:, 12 : 12 + n_time_steps]
            volume_window = X_iter[:, 12 + n_time_steps : 12 + 2 * n_time_steps]
            X_input = torch.cat([static_part, water_depth_window, volume_window], dim=1)

            out = model(X_input, edge_features, g)
            pred = out[0] if isinstance(out, tuple) else out
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]

            water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)
            new_flow = inflow_seq[t].unsqueeze(0).expand(num_nodes, 1)
            new_precip = precip_seq[t].unsqueeze(0).expand(num_nodes, 1)
            static_part_updated = static_part.clone()
            static_part_updated[:, 10:12] = torch.cat([new_flow, new_precip], dim=1)
            X_iter = torch.cat([static_part_updated, water_depth_updated, volume_updated], dim=1)

            pred_wd_real = new_wd.squeeze(1) * (wd_std + eps) + wd_mean
            gt_wd_real = wd_gt_seq[t] * (wd_std + eps) + wd_mean
            rmses.append(compute_rmse(pred_wd_real, gt_wd_real))

        per_hydro_rmse.append(rmses)

    return torch.tensor(per_hydro_rmse)  # (num_hydrographs, K_max)


def aggregate_rmse_at_K(per_hydro_rmse: torch.Tensor, K_list):
    """Returns dict { f"rmse_K{K}": per-step RMSE at step K (1-indexed) averaged over
    hydrographs, and f"rmse_K{K}_cum": mean RMSE over steps 1..K }.
    """
    out = {}
    for K in K_list:
        idx = K - 1  # 1-indexed step
        if idx < 0 or idx >= per_hydro_rmse.shape[1]:
            out[f"rmse_K{K}"] = float("nan")
            out[f"rmse_K{K}_cum"] = float("nan")
            continue
        out[f"rmse_K{K}"] = float(per_hydro_rmse[:, idx].mean())
        out[f"rmse_K{K}_cum"] = float(per_hydro_rmse[:, : idx + 1].mean())
    return out


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="val_breakdown_analysis",
        group="analysis",
        mode="disabled",
    )
    logger = PythonLogger("analysis")
    rzl = RankZeroLoggingWrapper(logger, DistributedManager())

    run_dir = cfg.get("analysis_run_dir")
    csv_out = cfg.get("analysis_csv_out", os.path.join(run_dir, "val_breakdown.csv"))
    first_epoch = int(cfg.get("analysis_first_epoch", 0))
    last_epoch = int(cfg.get("analysis_last_epoch", cfg.epochs - 1))
    K_list = list(cfg.get("analysis_K_list", [1, 5, 10, 20, 45]))
    K_max = max(K_list)

    rzl.info(f"Analysis run_dir: {run_dir}")
    rzl.info(f"CSV out: {csv_out}")
    rzl.info(f"Epoch range: {first_epoch}..{last_epoch}, K_list: {K_list} (K_max={K_max})")

    # Build trainer (sets up model, val_dataloader, normalization, etc.)
    trainer = MGNTrainer(cfg, rzl)
    device = trainer.dist.device

    # Build a separate rollout-mode val dataset (test-mode framing on val.txt).
    rollout_dataset = build_rollout_val_dataset(cfg, rollout_length=K_max)
    rzl.info(f"Rollout val dataset: {len(rollout_dataset)} hydrographs, K_max={K_max}")

    ckpt_dir = to_absolute_path(cfg.ckpt_path)
    rzl.info(f"Loading checkpoints from: {ckpt_dir}")

    fieldnames = [
        "epoch", "val_total", "loss_one", "loss_stability", "mse_total",
        "local_phys_raw", "local_phys_weighted",
        "edge_reg_raw", "edge_reg_weighted",
        "eff_weight", "reconstructed_total", "reconstruction_residual",
    ] + [f"rmse_K{K}" for K in K_list] + [f"rmse_K{K}_cum" for K in K_list]

    csv_exists = os.path.exists(csv_out)
    f = open(csv_out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()

    for epoch in range(first_epoch, last_epoch + 1):
        t0 = time.time()
        # Load weights for this epoch via physicsnemo loader (handles .mdlus).
        loaded_epoch = load_checkpoint(
            ckpt_dir,
            models=trainer.model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=epoch,
            device=device,
        )
        if int(loaded_epoch) != epoch:
            rzl.info(f"WARN: requested epoch {epoch}, loader returned {loaded_epoch}; skipping")
            continue

        # Set current_epoch so warmup/decay produces the same eff_weight as training.
        trainer.current_epoch = epoch

        # Phase A: per-component validation.
        val = trainer.validate()

        # Sanity check: total ≈ loss_one + loss_stability + local_phys_weighted + edge_reg_weighted
        recon = val["loss_one"] + val["loss_stability"] + val["local_phys_weighted"] + val["edge_reg_weighted"]
        residual = val["total"] - recon

        # Phase B: K-step rollout RMSE on val hydrographs.
        trainer.model.eval()
        per_hydro_rmse = rollout_per_step_rmse(
            trainer.model, rollout_dataset, device, cfg.n_time_steps, K_max
        )
        rmse_metrics = aggregate_rmse_at_K(per_hydro_rmse, K_list)

        row = {
            "epoch": epoch,
            "val_total": val["total"],
            "loss_one": val["loss_one"],
            "loss_stability": val["loss_stability"],
            "mse_total": val["mse"],
            "local_phys_raw": val["local_phys_raw"],
            "local_phys_weighted": val["local_phys_weighted"],
            "edge_reg_raw": val["edge_reg_raw"],
            "edge_reg_weighted": val["edge_reg_weighted"],
            "eff_weight": val["eff_weight"],
            "reconstructed_total": recon,
            "reconstruction_residual": residual,
            **rmse_metrics,
        }
        writer.writerow(row)
        f.flush()
        dt = time.time() - t0
        rzl.info(
            f"ep{epoch:02d} | val_total={val['total']:.4e} | "
            f"loss_one={val['loss_one']:.4e} loss_stab={val['loss_stability']:.4e} "
            f"local_w={val['local_phys_weighted']:.4e} edge_w={val['edge_reg_weighted']:.4e} "
            f"recon_resid={residual:+.2e} | "
            f"K1={rmse_metrics['rmse_K1']:.4f} K20={rmse_metrics.get('rmse_K20', float('nan')):.4f} "
            f"K45={rmse_metrics.get('rmse_K45', float('nan')):.4f} | {dt:.1f}s"
        )

    f.close()
    rzl.info(f"Done. CSV at: {csv_out}")


if __name__ == "__main__":
    main()
