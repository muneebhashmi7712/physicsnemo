# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Epoch sweep / checkpoint-selection script for UrbanFlood LMC bundle v6.

For a given run's checkpoint directory, evaluates the test rollout at each
epoch index in `sweep_epochs` and writes one CSV row per epoch with the
mean rollout RMSE on 2D, 1D, and overall nodes.

Reuses inference.py's rollout loop verbatim, just without animations and
with the checkpoint index pinned via load_checkpoint(epoch=...).

v6 success metric is 2D RMSE only. 1D and total are reported alongside
for diagnostics but do not drive the decision.
"""

import csv
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from physicsnemo.utils import load_checkpoint
from physicsnemo.datapipes.gnn.hydrographnet_dataset import UrbanFloodDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from model import HydroGraphKANWithEdgeDecoder


def build_model(cfg, device):
    use_local_physics_loss = cfg.get("use_local_physics_loss", False)
    model_args = dict(
        input_dim_nodes=cfg.get("num_input_features", 16),
        input_dim_edges=cfg.get("num_edge_features", 3),
        output_dim=cfg.get("num_output_features", 2),
        processor_size=cfg.get("processor_size", 5),
        hidden_dim_processor=cfg.get("hidden_dim_processor", 64),
        hidden_dim_node_encoder=cfg.get("hidden_dim_node_encoder", 64),
        hidden_dim_edge_encoder=cfg.get("hidden_dim_edge_encoder", 64),
        hidden_dim_node_decoder=cfg.get("hidden_dim_node_decoder", 64),
        num_layers_node_processor=cfg.get("num_layers_node_processor", 1),
        num_layers_edge_processor=cfg.get("num_layers_edge_processor", 1),
        num_layers_edge_encoder=cfg.get("num_layers_edge_encoder", 1),
        num_layers_node_decoder=cfg.get("num_layers_node_decoder", 1),
        num_harmonics=cfg.get("num_harmonics", 5),
    )
    if use_local_physics_loss:
        model = HydroGraphKANWithEdgeDecoder(
            **model_args,
            concat_endpoints=bool(cfg.get("edge_decoder_concat_endpoints", False)),
        )
    else:
        model = MeshGraphKAN(**model_args)
    return model.to(device)


@torch.no_grad()
def evaluate_rollout(model, test_dataset, cfg, device):
    """Run full test-set rollout, return (mean_total, mean_2d, mean_1d) in metres."""
    rollout_length = cfg.get("num_test_time_steps", 8)
    n_time_steps = cfg.get("n_time_steps", 2)
    use_1d = cfg.get("use_1d", False)
    edge_q_prev_as_input = bool(cfg.get("edge_q_prev_as_input", False))

    wd_stats = test_dataset.dynamic_stats["water_depth"]
    wd_mean = wd_stats["mean"]
    wd_std = wd_stats["std"]
    if use_1d:
        wd_mean_2d = wd_stats["2d"]["mean"]
        wd_std_2d = wd_stats["2d"]["std"]
        wd_mean_1d = wd_stats["1d"]["mean"]
        wd_std_1d = wd_stats["1d"]["std"]
    elev_mean_raw = test_dataset.static_stats["elevation"]["mean"]
    elev_std_raw = test_dataset.static_stats["elevation"]["std"]
    elev_mean = elev_mean_raw[0] if isinstance(elev_mean_raw, list) else elev_mean_raw
    elev_std = elev_std_raw[0] if isinstance(elev_std_raw, list) else elev_std_raw
    epsilon = 1e-8

    all_rmse_all, all_rmse_2d, all_rmse_1d = [], [], []

    for idx in range(len(test_dataset)):
        g, rollout_data = test_dataset[idx]
        g = g.to(device)
        edge_features = g.edge_attr.to(device)
        X_current = g.x.to(device)
        num_nodes = X_current.size(0)
        node_type = g.node_type.to(device) if hasattr(g, "node_type") else None

        rmse_list, rmse_2d_list, rmse_1d_list = [], [], []
        inflow_seq = rollout_data["inflow"].to(device)
        precip_seq = rollout_data["precipitation"].to(device)
        wd_gt_seq = rollout_data["water_depth_gt"].to(device)

        elev_norm = X_current[:, 3]
        elev_real = elev_norm * (elev_std + epsilon) + elev_mean

        if use_1d and node_type is not None:
            is_1d = node_type == 1
            wd_mean_per_node = torch.where(
                is_1d,
                torch.tensor(wd_mean_1d, device=device, dtype=X_current.dtype),
                torch.tensor(wd_mean_2d, device=device, dtype=X_current.dtype),
            )
            wd_std_per_node = torch.where(
                is_1d,
                torch.tensor(wd_std_1d, device=device, dtype=X_current.dtype),
                torch.tensor(wd_std_2d, device=device, dtype=X_current.dtype),
            )
        else:
            wd_mean_per_node = wd_mean
            wd_std_per_node = wd_std

        n_static = X_current.size(1) - 2 * n_time_steps
        X_iter = X_current.clone()

        for t in range(rollout_length):
            static_part = X_iter[:, :n_static]
            water_depth_window = X_iter[:, n_static : n_static + n_time_steps]
            volume_window = X_iter[:, n_static + n_time_steps : n_static + 2 * n_time_steps]

            X_input = torch.cat([static_part, water_depth_window, volume_window], dim=1)
            out = model(X_input, edge_features, g)
            pred = out[0] if isinstance(out, tuple) else out
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]

            if edge_q_prev_as_input and isinstance(out, tuple) and len(out) > 1:
                edge_pred = out[1].squeeze(-1)
                edge_features = edge_features.clone()
                edge_features[:, -1] = edge_features[:, -1] + edge_pred

            water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)
            new_flow = inflow_seq[t].unsqueeze(0).expand(num_nodes, 1)
            new_precip = precip_seq[t].unsqueeze(0).expand(num_nodes, 1)
            static_part_updated = static_part.clone()
            static_part_updated[:, 10:12] = torch.cat([new_flow, new_precip], dim=1)
            X_iter = torch.cat(
                [static_part_updated, water_depth_updated, volume_updated], dim=1
            )

            pred_wl = new_wd.squeeze(1) * (wd_std_per_node + epsilon) + wd_mean_per_node
            gt_wl = wd_gt_seq[t] * (wd_std_per_node + epsilon) + wd_mean_per_node
            pred_depth = torch.clamp(pred_wl - elev_real, min=0.0)
            gt_depth = torch.clamp(gt_wl - elev_real, min=0.0)

            rmse = torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2)).item()
            rmse_list.append(rmse)

            if node_type is not None:
                m2 = node_type == 0
                m1 = node_type == 1
                if m2.any():
                    rmse_2d_list.append(
                        torch.sqrt(torch.mean((pred_depth[m2] - gt_depth[m2]) ** 2)).item()
                    )
                if m1.any():
                    rmse_1d_list.append(
                        torch.sqrt(torch.mean((pred_depth[m1] - gt_depth[m1]) ** 2)).item()
                    )

        all_rmse_all.append(sum(rmse_list) / len(rmse_list))
        if rmse_2d_list:
            all_rmse_2d.append(sum(rmse_2d_list) / len(rmse_2d_list))
        if rmse_1d_list:
            all_rmse_1d.append(sum(rmse_1d_list) / len(rmse_1d_list))

    mean_total = sum(all_rmse_all) / len(all_rmse_all)
    mean_2d = sum(all_rmse_2d) / len(all_rmse_2d) if all_rmse_2d else float("nan")
    mean_1d = sum(all_rmse_1d) / len(all_rmse_1d) if all_rmse_1d else float("nan")
    return mean_total, mean_2d, mean_1d


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    ckpt_path = cfg.get("ckpt_path")
    sweep_epochs_str = str(cfg.get("sweep_epochs", "4,9,14,19,24,29,34,39,44,49"))
    sweep_epochs = [int(e) for e in sweep_epochs_str.split(",") if e.strip() != ""]
    out_csv = cfg.get("sweep_csv", "sweep_results.csv")

    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    print(f"Sweeping {len(sweep_epochs)} epochs: {sweep_epochs}")
    print(f"Writing CSV to: {out_csv}")

    test_dataset = UrbanFloodDataset(
        data_dir=cfg.get("test_dir", cfg.get("data_dir")),
        model_name=cfg.get("model_name", "Model_1"),
        split="test",
        n_time_steps=cfg.get("n_time_steps", 2),
        rollout_length=cfg.get("num_test_time_steps", 8),
        return_physics=False,
        use_1d=cfg.get("use_1d", False),
        edge_q_prev_as_input=bool(cfg.get("edge_q_prev_as_input", False)),
        lmc_antisymmetric=bool(cfg.get("lmc_antisymmetric", False)),
    )
    print(f"Loaded test dataset with {len(test_dataset)} events.")

    model = build_model(cfg, device)

    rows = []
    for ep in sweep_epochs:
        epoch_loaded = load_checkpoint(
            to_absolute_path(ckpt_path),
            models=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=ep,
            device=device,
        )
        model.eval()
        mean_total, mean_2d, mean_1d = evaluate_rollout(
            model, test_dataset, cfg, device
        )
        print(
            f"[epoch {ep:>2d}] 2D_mean={mean_2d:.4f} m | "
            f"1D_mean={mean_1d:.4f} m | total_mean={mean_total:.4f} m "
            f"(loaded={epoch_loaded})"
        )
        rows.append(
            dict(epoch=ep, mean_2d=mean_2d, mean_1d=mean_1d, mean_total=mean_total)
        )

    out_csv_abs = to_absolute_path(out_csv)
    os.makedirs(os.path.dirname(out_csv_abs) or ".", exist_ok=True)
    with open(out_csv_abs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "mean_2d", "mean_1d", "mean_total"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote sweep CSV to {out_csv_abs}")

    best = min(rows, key=lambda r: r["mean_2d"])
    print(
        f"\nBest 2D epoch = {best['epoch']} "
        f"(2D={best['mean_2d']:.4f} m | 1D={best['mean_1d']:.4f} m | "
        f"total={best['mean_total']:.4f} m)"
    )
    print("Target to beat: 2D_mean <= 0.0195 m (v2-nolc baseline)")


if __name__ == "__main__":
    main()
