# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
rollout_script.py

A standalone script that uses Hydra to load the shared configuration,
instantiates the test dataset and the trained MeshGraphKAN model, loads the checkpoint,
and performs an iterative rollout for each test hydrograph sample.
For each sample, a fancy four-panel animation is generated that shows:
  1. Prediction (node colors represent predicted actual water depth)
  2. Ground Truth (node colors represent actual water depth)
  3. Absolute Error (difference between prediction and ground truth)
  4. RMSE curve over time (updated with each rollout step)

The model checkpoint is loaded using the provided load_checkpoint utility.
"""

import os
import json
import math
import torch
import hydra
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Import the load_checkpoint utility from Modulus Launch.
from physicsnemo.utils import load_checkpoint

# Import the dataset and model.
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from model import HydroGraphKANWithEdgeDecoder
from metrics import compute_rmse, compute_nse, compute_csi, compute_mass_balance_error

# For converting PyG graph to networkx.
from torch_geometric.utils import to_networkx


def create_animation(
    rollout_predictions,
    ground_truth,
    initial_graph,
    rmse_list,
    output_path,
    time_per_step=20 / 60,
):
    """
    Create a four-panel animation for one hydrograph rollout.

    Parameters:
      rollout_predictions: list of predicted actual water depth tensors (each shape: [num_nodes])
      ground_truth: list of ground truth water depth tensors (each shape: [num_nodes])
      initial_graph: the initial PyG graph sample (used for node positions and edges)
      rmse_list: list of RMSE values computed at each rollout step
      output_path: file path to save the animation (e.g. a GIF file)
      time_per_step: simulation time (in hours) corresponding to each rollout step.
    """
    # Set professional style.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    # Create figure and extra axes for colorbars.
    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    cax1 = fig.add_axes([0.05, 0.53, 0.02, 0.35])
    cax2 = fig.add_axes([0.95, 0.53, 0.02, 0.35])
    cax3 = fig.add_axes([0.05, 0.1, 0.02, 0.35])

    num_frames = len(rollout_predictions)
    # Use the first two columns of node features for positions.
    init_node_feats = initial_graph.x
    pos = {
        i: (init_node_feats[i, 0].item(), init_node_feats[i, 1].item())
        for i in range(init_node_feats.shape[0])
    }

    # Compute global color scaling based on both predictions and ground truth.
    all_vals = torch.cat(rollout_predictions + ground_truth)
    vmin_global = all_vals.min().item()
    vmax_global = all_vals.max().item()

    def update(frame):
        for ax in axes.flat:
            ax.clear()
        current_time = (frame + 1) * time_per_step

        # Panel 1: Prediction.
        pred_vals = rollout_predictions[frame].cpu().numpy()
        # Ensure the graph is on CPU before converting.
        g_pred = to_networkx(initial_graph)
        g_pred = g_pred.to_undirected()
        nodes_pred = nx.draw_networkx_nodes(
            g_pred,
            pos,
            node_color=pred_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 0],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_pred, pos, alpha=0.5, ax=axes[0, 0])
        axes[0, 0].set_title(f"Time {current_time:.2f} Hours - Prediction", fontsize=24)
        fig.colorbar(nodes_pred, cax=cax1)

        # Panel 2: Ground Truth.
        gt_vals = ground_truth[frame].cpu().numpy()
        g_gt = to_networkx(initial_graph)
        g_gt = g_gt.to_undirected()
        nodes_gt = nx.draw_networkx_nodes(
            g_gt,
            pos,
            node_color=gt_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 1],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_gt, pos, alpha=0.5, ax=axes[0, 1])
        axes[0, 1].set_title(
            f"Time {current_time:.2f} Hours - Ground Truth", fontsize=24
        )
        fig.colorbar(nodes_gt, cax=cax2)

        # Panel 3: Absolute Error.
        abs_error = torch.abs(rollout_predictions[frame] - ground_truth[frame])
        abs_vals = abs_error.cpu().numpy()
        g_error = to_networkx(initial_graph.cpu())
        g_error = g_error.to_undirected()
        nodes_error = nx.draw_networkx_nodes(
            g_error,
            pos,
            node_color=abs_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[1, 0],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_error, pos, alpha=0.5, ax=axes[1, 0])
        axes[1, 0].set_title(
            f"Time {current_time:.2f} Hours - Absolute Error", fontsize=24
        )
        fig.colorbar(nodes_error, cax=cax3)

        # Panel 4: RMSE Curve.
        times = [(i + 1) * time_per_step for i in range(frame + 1)]
        axes[1, 1].plot(
            times,
            rmse_list[: frame + 1],
            label="Water Depth RMSE",
            color="b",
            linewidth=3,
        )
        axes[1, 1].set_title("RMSE Over Time", fontsize=24)
        axes[1, 1].set_xlabel("Time (Hours)", fontsize=24)
        axes[1, 1].set_ylabel("RMSE", fontsize=24)
        axes[1, 1].legend(fontsize=20)
        axes[1, 1].grid(True)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(output_path, writer="pillow", fps=2)
    plt.close(fig)
    print(f"Animation saved to {output_path}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function that loads the configuration, instantiates the test dataset and model,
    loads the checkpoint using load_checkpoint, performs iterative rollout, and generates animations.
    """
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    rollout_length = cfg.get(
        "num_test_time_steps", 10
    )  # Rollout length (number of future steps)
    n_time_steps = cfg.get("n_time_steps", 2)
    prefix = cfg.get("prefix", "M80")
    data_dir = cfg.get("test_dir")
    test_ids_file = cfg.get("test_ids_file", "test.txt")
    ckpt_path = cfg.get("ckpt_path")
    ckpt_epoch = cfg.get("ckpt_epoch", None)  # specific epoch index to load; None = latest
    anim_output_dir = cfg.get("animation_output_dir", "animations")
    os.makedirs(anim_output_dir, exist_ok=True)

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Instantiate the test dataset.
    test_dataset = HydroGraphDataset(
        data_dir=data_dir,
        prefix=prefix,
        n_time_steps=n_time_steps,
        hydrograph_ids_file=test_ids_file,
        split="test",
        rollout_length=rollout_length,
        return_physics=False,
        graph_type=cfg.get("graph_type", "knn"),
        make_bidirectional=cfg.get("use_local_physics_loss", False),
    )
    print(f"Loaded test dataset with {len(test_dataset)} hydrographs.")

    # Extract denormalization statistics for physical-unit metrics.
    wd_mean = test_dataset.dynamic_stats["water_depth"]["mean"]
    wd_std = test_dataset.dynamic_stats["water_depth"]["std"]
    vol_mean = test_dataset.dynamic_stats["volume"]["mean"]
    vol_std = test_dataset.dynamic_stats["volume"]["std"]
    eps = 1e-8

    # Instantiate the model.
    num_input_features = cfg.get("num_input_features", 16)
    num_edge_features = cfg.get("num_edge_features", 3)
    num_output_features = cfg.get("num_output_features", 2)
    use_local = cfg.get("use_local_physics_loss", False)
    _lp_cfg = cfg.get("lowpass", {}) or {}
    model_args = dict(
        input_dim_nodes=num_input_features,
        input_dim_edges=num_edge_features,
        output_dim=num_output_features,
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
        lowpass_enabled=bool(_lp_cfg.get("enabled", False)),
        lowpass_iterations=int(_lp_cfg.get("iterations", 1)),
        lowpass_alpha_init=float(_lp_cfg.get("alpha_init", 0.2)),
        lowpass_learnable_alpha=bool(_lp_cfg.get("learnable_alpha", False)),
        lowpass_alpha_per_channel=bool(_lp_cfg.get("alpha_per_channel", False)),
    )
    if use_local:
        model = HydroGraphKANWithEdgeDecoder(**model_args)
    else:
        base_args = {k: v for k, v in model_args.items() if not k.startswith("lowpass_")}
        model = MeshGraphKAN(**base_args)
    model.to(device)

    # Load model checkpoint.
    # If ckpt_epoch is specified, load that exact epoch directly (epoch sweep mode).
    # Otherwise prefer best_checkpoint.pt, then fall back to the latest checkpoint.
    if ckpt_epoch is not None:
        print(f"Loading checkpoint for epoch {ckpt_epoch} (epoch sweep mode).")
        epoch_loaded = load_checkpoint(
            to_absolute_path(ckpt_path),
            models=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=ckpt_epoch,
            device=device,
        )
        print(f"Checkpoint loaded from epoch {epoch_loaded}")
    else:
        best_ckpt_file = os.path.join(to_absolute_path(ckpt_path), "best_checkpoint.pt")
        if os.path.exists(best_ckpt_file):
            best_ckpt = torch.load(best_ckpt_file, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            epoch_loaded = best_ckpt.get("epoch", 0)
            val_loss = best_ckpt.get("val_loss", float("nan"))
            print(f"Loaded best checkpoint from epoch {epoch_loaded} (val_loss={val_loss:.4e})")
        else:
            print("No best_checkpoint.pt found, falling back to latest checkpoint.")
            epoch_loaded = load_checkpoint(
                to_absolute_path(ckpt_path),
                models=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=device,
            )
            print(f"Checkpoint loaded from epoch {epoch_loaded}")
    model.eval()

    # Metric collection across all hydrographs.
    all_rmse_all = []
    all_nse_all = []
    all_csi_005_all = []
    all_csi_030_all = []
    all_mbe_all = []
    sample_ids = []

    # Loop over each test hydrograph.
    for idx in range(len(test_dataset)):
        g, rollout_data = test_dataset[idx]
        g = g.to(device)
        edge_features = g.edge_attr.to(device)
        X_current = g.x.to(device)  # Expected shape: [num_nodes, 16]
        num_nodes = X_current.size(0)

        rollout_preds = []  # Predicted water depth (denormalized) for animation.
        ground_truth_list = []  # Ground truth water depth (denormalized) for animation.
        rmse_list = []
        nse_list = []
        csi_005_list = []
        csi_030_list = []
        mbe_list = []

        # Rollout data tensors.
        inflow_seq = rollout_data["inflow"].to(device)
        precip_seq = rollout_data["precipitation"].to(device)
        wd_gt_seq = rollout_data["water_depth_gt"].to(device)
        vol_gt_seq = rollout_data["volume_gt"].to(device)

        X_iter = X_current.clone()

        for t in range(rollout_length):
            # Split into static and dynamic parts.
            static_part = X_iter[
                :, :12
            ]  # columns 0-11: static features (including flow/precip)
            water_depth_window = X_iter[
                :, 12 : 12 + n_time_steps
            ]  # e.g., columns 12-13 for n_time_steps=2
            volume_window = X_iter[
                :, 12 + n_time_steps : 12 + 2 * n_time_steps
            ]  # e.g., columns 14-15

            # Use the full dynamic window as input.
            X_input = torch.cat(
                [static_part, water_depth_window, volume_window], dim=1
            )  # shape remains 16

            # Predict the differences (delta).
            out = model(X_input, edge_features, g)
            pred = out[0] if isinstance(out, tuple) else out  # shape: (num_nodes, 2)
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]

            # Update dynamic window: drop the oldest time step and append the new prediction.
            water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)

            # Update static part: since inflow_seq and precip_seq are 1D,
            # we unsqueeze and expand them to shape (num_nodes, 1).
            new_flow = inflow_seq[t].unsqueeze(0).expand(num_nodes, 1)
            new_precip = precip_seq[t].unsqueeze(0).expand(num_nodes, 1)
            static_part_updated = static_part.clone()
            static_part_updated[:, 10:12] = torch.cat([new_flow, new_precip], dim=1)

            # Form updated X_iter.
            X_iter = torch.cat(
                [static_part_updated, water_depth_updated, volume_updated], dim=1
            )

            # Denormalize to physical units for metrics.
            pred_wd_real = new_wd.squeeze(1) * (wd_std + eps) + wd_mean
            gt_wd_real = wd_gt_seq[t] * (wd_std + eps) + wd_mean
            pred_vol_real = new_vol.squeeze(1) * (vol_std + eps) + vol_mean
            gt_vol_real = vol_gt_seq[t] * (vol_std + eps) + vol_mean

            # Store denormalized values for animation.
            rollout_preds.append(pred_wd_real.detach().cpu())
            ground_truth_list.append(gt_wd_real.detach().cpu())

            # Compute all metrics on denormalized values.
            rmse_list.append(compute_rmse(pred_wd_real, gt_wd_real))
            nse_list.append(compute_nse(pred_wd_real, gt_wd_real))
            csi_005_list.append(compute_csi(pred_wd_real, gt_wd_real, 0.05))
            csi_030_list.append(compute_csi(pred_wd_real, gt_wd_real, 0.30))
            mbe_list.append(compute_mass_balance_error(pred_vol_real, gt_vol_real))

        # Aggregate per-hydrograph.
        all_rmse_all.append(rmse_list)
        all_nse_all.append(nse_list)
        all_csi_005_all.append(csi_005_list)
        all_csi_030_all.append(csi_030_list)
        all_mbe_all.append(mbe_list)

        sample_id = test_dataset.dynamic_data[idx].get("hydro_id", idx)
        sample_ids.append(str(sample_id))

        mean_rmse = sum(rmse_list) / len(rmse_list)
        mean_nse = sum(nse_list) / len(nse_list)
        valid_csi_005 = [x for x in csi_005_list if not math.isnan(x)]
        valid_csi_030 = [x for x in csi_030_list if not math.isnan(x)]
        mean_csi_005 = sum(valid_csi_005) / len(valid_csi_005) if valid_csi_005 else float("nan")
        mean_csi_030 = sum(valid_csi_030) / len(valid_csi_030) if valid_csi_030 else float("nan")
        mean_mbe = sum(mbe_list) / len(mbe_list)

        print(
            f"Hydrograph {sample_id}: "
            f"RMSE={mean_rmse:.4f}m | NSE={mean_nse:.4f} | "
            f"CSI@0.05={mean_csi_005:.4f} | CSI@0.3={mean_csi_030:.4f} | "
            f"MBE={mean_mbe:.6f}"
        )

        # anim_filename = os.path.join(anim_output_dir, f"animation_{sample_id}.gif")
        # create_animation(rollout_preds, ground_truth_list, g, rmse_list, anim_filename)

    # ---- Overall metrics aggregation ----
    all_rmse_tensor = torch.tensor(all_rmse_all)
    all_nse_tensor = torch.tensor(all_nse_all)
    all_csi_005_tensor = torch.tensor(all_csi_005_all)
    all_csi_030_tensor = torch.tensor(all_csi_030_all)
    all_mbe_tensor = torch.tensor(all_mbe_all)

    # Per-step averages across hydrographs.
    mean_rmse_per_step = all_rmse_tensor.mean(dim=0)
    std_rmse_per_step = all_rmse_tensor.std(dim=0)
    mean_nse_per_step = all_nse_tensor.mean(dim=0)
    mean_csi_005_per_step = torch.nanmean(all_csi_005_tensor, dim=0)
    mean_csi_030_per_step = torch.nanmean(all_csi_030_tensor, dim=0)
    mean_mbe_per_step = all_mbe_tensor.mean(dim=0)

    print("\n===== Overall Metrics (mean +/- std across hydrographs) =====")
    print(f"RMSE:     {all_rmse_tensor.mean():.4f} +/- {all_rmse_tensor.std():.4f} m")
    print(f"NSE:      {all_nse_tensor.mean():.4f} +/- {all_nse_tensor.std():.4f}")
    print(f"CSI@0.05: {torch.nanmean(all_csi_005_tensor):.4f} +/- {all_csi_005_tensor[~all_csi_005_tensor.isnan()].std():.4f}")
    print(f"CSI@0.3:  {torch.nanmean(all_csi_030_tensor):.4f} +/- {all_csi_030_tensor[~all_csi_030_tensor.isnan()].std():.4f}")
    print(f"MBE:      {all_mbe_tensor.mean():.6f} +/- {all_mbe_tensor.std():.6f}")

    print("\nMean RMSE per rollout step:", mean_rmse_per_step.tolist())
    print("Std RMSE per rollout step:", std_rmse_per_step.tolist())

    # Save all metrics to JSON.
    metrics_output = {
        "config": {
            "use_local_physics_loss": use_local,
            "rollout_length": rollout_length,
            "num_hydrographs": len(test_dataset),
            "ckpt_path": str(ckpt_path),
            "epoch_loaded": epoch_loaded,
        },
        "per_step": {
            "rmse_mean": mean_rmse_per_step.tolist(),
            "rmse_std": std_rmse_per_step.tolist(),
            "nse_mean": mean_nse_per_step.tolist(),
            "csi_005_mean": mean_csi_005_per_step.tolist(),
            "csi_030_mean": mean_csi_030_per_step.tolist(),
            "mbe_mean": mean_mbe_per_step.tolist(),
        },
        "per_hydrograph": {
            sample_ids[i]: {
                "mean_rmse": float(all_rmse_tensor[i].mean()),
                "mean_nse": float(all_nse_tensor[i].mean()),
                "mean_csi_005": float(torch.nanmean(all_csi_005_tensor[i])),
                "mean_csi_030": float(torch.nanmean(all_csi_030_tensor[i])),
                "mean_mbe": float(all_mbe_tensor[i].mean()),
            }
            for i in range(len(test_dataset))
        },
        "overall": {
            "rmse_mean": float(all_rmse_tensor.mean()),
            "rmse_std": float(all_rmse_tensor.std()),
            "nse_mean": float(all_nse_tensor.mean()),
            "nse_std": float(all_nse_tensor.std()),
            "csi_005_mean": float(torch.nanmean(all_csi_005_tensor)),
            "csi_030_mean": float(torch.nanmean(all_csi_030_tensor)),
            "mbe_mean": float(all_mbe_tensor.mean()),
        },
    }
    metrics_path = os.path.join(anim_output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Plot RMSE curve with error bands.
    timesteps = [(i + 1) * (20 / 60) for i in range(rollout_length)]
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rmse_per_step.numpy(), label="Mean RMSE", linewidth=3)
    plt.fill_between(
        timesteps,
        (mean_rmse_per_step - std_rmse_per_step).numpy(),
        (mean_rmse_per_step + std_rmse_per_step).numpy(),
        alpha=0.3,
        label="\u00b1 Std",
    )
    plt.xlabel("Time (Hours)", fontsize=20)
    plt.ylabel("RMSE (m)", fontsize=20)
    plt.title("Overall RMSE Curve Over Rollout", fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(True)
    plot_path = os.path.join(anim_output_dir, "rmse_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"RMSE plot saved to {plot_path}")


if __name__ == "__main__":
    main()
