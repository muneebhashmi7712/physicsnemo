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
Inference / rollout script for HydroGraphNet on UrbanFlood dataset.
Identical to inference.py except for dataset import/instantiation and timestep scaling.

For each test event, generates a four-panel animation:
  1. Prediction (node colors = predicted water depth above ground)
  2. Ground Truth (node colors = actual water depth above ground)
  3. Absolute Error
  4. RMSE curve over time (in meters)
"""

import os
import numpy as np
import torch
import hydra
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from physicsnemo.utils import load_checkpoint

from physicsnemo.datapipes.gnn.hydrographnet_dataset import UrbanFloodDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from model import HydroGraphKANWithEdgeDecoder

from torch_geometric.utils import to_networkx


def create_animation(
    rollout_predictions,
    ground_truth,
    initial_graph,
    rmse_list,
    output_path,
    time_per_step=5 / 60,  # 5 minutes per step for UrbanFlood
):
    """
    Create a four-panel animation for one event rollout.

    Parameters:
      rollout_predictions: list of predicted water depth above ground tensors (meters)
      ground_truth: list of ground truth water depth above ground tensors (meters)
      initial_graph: the initial PyG graph sample (used for node positions and edges)
      rmse_list: list of RMSE values in meters computed at each rollout step
      output_path: file path to save the animation (e.g. a GIF file)
      time_per_step: simulation time (in hours) corresponding to each rollout step.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    cax1 = fig.add_axes([0.05, 0.53, 0.02, 0.35])
    cax2 = fig.add_axes([0.95, 0.53, 0.02, 0.35])
    cax3 = fig.add_axes([0.05, 0.1, 0.02, 0.35])

    num_frames = len(rollout_predictions)
    init_node_feats = initial_graph.x
    pos = {
        i: (init_node_feats[i, 0].item(), init_node_feats[i, 1].item())
        for i in range(init_node_feats.shape[0])
    }

    # Depth color scale: 0 to max depth across pred + GT.
    all_vals = torch.cat(rollout_predictions + ground_truth)
    vmin_depth = 0.0
    vmax_depth = max(all_vals.max().item(), 1e-6)

    # Error color scale: 0 to max absolute error.
    all_errors = torch.cat(
        [torch.abs(p - g) for p, g in zip(rollout_predictions, ground_truth)]
    )
    vmax_error = max(all_errors.max().item(), 1e-6)

    def update(frame):
        for ax in axes.flat:
            ax.clear()
        current_time = (frame + 1) * time_per_step

        # Panel 1: Prediction.
        pred_vals = rollout_predictions[frame].cpu().numpy()
        g_pred = to_networkx(initial_graph)
        g_pred = g_pred.to_undirected()
        nodes_pred = nx.draw_networkx_nodes(
            g_pred,
            pos,
            node_color=pred_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 0],
            vmin=vmin_depth,
            vmax=vmax_depth,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_pred, pos, alpha=0.5, ax=axes[0, 0])
        axes[0, 0].set_title(f"Time {current_time:.2f} Hours - Prediction", fontsize=24)
        fig.colorbar(nodes_pred, cax=cax1, label="Depth (m)")

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
            vmin=vmin_depth,
            vmax=vmax_depth,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_gt, pos, alpha=0.5, ax=axes[0, 1])
        axes[0, 1].set_title(
            f"Time {current_time:.2f} Hours - Ground Truth", fontsize=24
        )
        fig.colorbar(nodes_gt, cax=cax2, label="Depth (m)")

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
            cmap=plt.cm.Reds,
            ax=axes[1, 0],
            vmin=0.0,
            vmax=vmax_error,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_error, pos, alpha=0.5, ax=axes[1, 0])
        axes[1, 0].set_title(
            f"Time {current_time:.2f} Hours - Absolute Error", fontsize=24
        )
        fig.colorbar(nodes_error, cax=cax3, label="Error (m)")

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
        axes[1, 1].set_ylabel("RMSE (m)", fontsize=24)
        axes[1, 1].legend(fontsize=20)
        axes[1, 1].grid(True)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(output_path, writer="pillow", fps=2)
    plt.close(fig)
    print(f"Animation saved to {output_path}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function: loads config, instantiates test dataset and model,
    loads checkpoint, performs rollout, and generates animations.
    """
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    rollout_length = cfg.get("num_test_time_steps", 44)
    n_time_steps = cfg.get("n_time_steps", 2)
    model_name = cfg.get("model_name", "Model_1")
    data_dir = cfg.get("test_dir", cfg.get("data_dir"))
    ckpt_path = cfg.get("ckpt_path")
    anim_output_dir = cfg.get("animation_output_dir", "animations")
    os.makedirs(anim_output_dir, exist_ok=True)

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Instantiate the test dataset.
    use_1d = cfg.get("use_1d", False)
    static_prediction = bool(cfg.get("static_prediction", False))
    edge_q_prev_as_input = bool(cfg.get("edge_q_prev_as_input", False))
    test_dataset = UrbanFloodDataset(
        data_dir=data_dir,
        model_name=model_name,
        split="test",
        n_time_steps=n_time_steps,
        rollout_length=rollout_length,
        return_physics=False,
        use_1d=use_1d,
        edge_q_prev_as_input=bool(cfg.get("edge_q_prev_as_input", False)),
        lmc_antisymmetric=bool(cfg.get("lmc_antisymmetric", False)),
        static_prediction=static_prediction,
    )
    print(f"Loaded test dataset with {len(test_dataset)} events.")

    # Normalization stats for denormalization to physical units.
    # v2: when use_1d, water_depth stats are split per node-type, so we build
    # a per-node tensor at evaluation time (see below). For the 2D-only path
    # the legacy scalar mean/std are used directly.
    wd_stats = test_dataset.dynamic_stats["water_depth"]
    wd_mean = wd_stats["mean"]   # legacy scalar (== 2D mean when use_1d)
    wd_std  = wd_stats["std"]    # legacy scalar (== 2D std  when use_1d)
    if use_1d:
        wd_mean_2d = wd_stats["2d"]["mean"]
        wd_std_2d  = wd_stats["2d"]["std"]
        wd_mean_1d = wd_stats["1d"]["mean"]
        wd_std_1d  = wd_stats["1d"]["std"]
    elev_mean_raw = test_dataset.static_stats["elevation"]["mean"]
    elev_std_raw = test_dataset.static_stats["elevation"]["std"]
    # Static stats may be stored as single-element lists.
    elev_mean = elev_mean_raw[0] if isinstance(elev_mean_raw, list) else elev_mean_raw
    elev_std = elev_std_raw[0] if isinstance(elev_std_raw, list) else elev_std_raw
    epsilon = 1e-8

    # Instantiate the model.
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
    model.to(device)

    # Load model checkpoint.
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

    # ------------------------------------------------------------------
    # v9: time-less peak-depth prediction — one forward pass per event.
    # Emits per-event + summary lines in the exact v8 log format so the
    # existing aggregator parses it unchanged (summary as 1-element tensors).
    # ------------------------------------------------------------------
    if static_prediction:
        pd_stats = test_dataset.dynamic_stats["peak_depth"]
        ev_2d, ev_1d, ev_all = [], [], []
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                g, meta = test_dataset[idx]
                g = g.to(device)
                node_type = g.node_type.to(device) if hasattr(g, "node_type") else None
                out = model(g.x.to(device), g.edge_attr.to(device), g)
                pred = out[0] if isinstance(out, tuple) else out  # (N, 1)
                pred = pred.squeeze(-1)

                if use_1d and node_type is not None:
                    is_1d = (node_type == 1)
                    mean_pn = torch.where(
                        is_1d,
                        torch.tensor(pd_stats["1d"]["mean"], device=device, dtype=pred.dtype),
                        torch.tensor(pd_stats["2d"]["mean"], device=device, dtype=pred.dtype),
                    )
                    std_pn = torch.where(
                        is_1d,
                        torch.tensor(pd_stats["1d"]["std"], device=device, dtype=pred.dtype),
                        torch.tensor(pd_stats["2d"]["std"], device=device, dtype=pred.dtype),
                    )
                else:
                    mean_pn = pd_stats["mean"]
                    std_pn = pd_stats["std"]

                pred_phys = torch.clamp(pred * (std_pn + epsilon) + mean_pn, min=0.0)
                gt_phys = meta["peak_depth_phys"].to(device)

                rmse = torch.sqrt(torch.mean((pred_phys - gt_phys) ** 2)).item()
                ev_all.append(rmse)
                sample_id = meta["event_id"]
                if node_type is not None:
                    m2 = node_type == 0
                    m1 = node_type == 1
                    r2 = torch.sqrt(torch.mean((pred_phys[m2] - gt_phys[m2]) ** 2)).item()
                    r1 = torch.sqrt(torch.mean((pred_phys[m1] - gt_phys[m1]) ** 2)).item()
                    ev_2d.append(r2)
                    ev_1d.append(r1)
                    print(
                        f"Event {sample_id}: Mean RMSE = {rmse:.4f} m | "
                        f"2D = {r2:.4f} m | 1D = {r1:.4f} m"
                    )
                else:
                    print(f"Event {sample_id}: Mean RMSE = {rmse:.4f} m")

        # Summary as 1-element tensors (peak-depth has no rollout dimension).
        mean_all = sum(ev_all) / len(ev_all)
        print(
            "Overall Mean RMSE (m) over rollout steps:",
            torch.tensor([mean_all]),
        )
        print("Overall Std RMSE (m) over rollout steps:", torch.tensor([float(np.std(ev_all))]))
        if ev_2d:
            print("Overall Mean RMSE — 2D nodes:", torch.tensor([sum(ev_2d) / len(ev_2d)]))
            print("Overall Std  RMSE — 2D nodes:", torch.tensor([float(np.std(ev_2d))]))
            print("Overall Mean RMSE — 1D nodes:", torch.tensor([sum(ev_1d) / len(ev_1d)]))
            print("Overall Std  RMSE — 1D nodes:", torch.tensor([float(np.std(ev_1d))]))
        else:
            # 2D-only run: surface line is the same as overall.
            print("Overall Mean RMSE — 2D nodes:", torch.tensor([mean_all]))
            print("Overall Std  RMSE — 2D nodes:", torch.tensor([float(np.std(ev_all))]))
        print(f"Static peak-depth inference done over {len(ev_all)} events.")
        return

    all_rmse_all = []
    all_rmse_2d = []   # per-event RMSE on 2D nodes only (use_1d case)
    all_rmse_1d = []   # per-event RMSE on 1D nodes only (use_1d case)

    # Loop over each test event.
    for idx in range(len(test_dataset)):
        g, rollout_data = test_dataset[idx]
        g = g.to(device)
        edge_features = g.edge_attr.to(device)
        X_current = g.x.to(device)
        num_nodes = X_current.size(0)
        node_type = (
            g.node_type.to(device) if hasattr(g, "node_type") else None
        )  # 0 = 2D, 1 = 1D

        rollout_preds = []
        ground_truth_list = []
        rmse_list = []
        rmse_2d_list = []
        rmse_1d_list = []

        inflow_seq = rollout_data["inflow"].to(device)
        precip_seq = rollout_data["precipitation"].to(device)
        wd_gt_seq = rollout_data["water_depth_gt"].to(device)

        # Denormalize "elevation" column (column 3) for surface depth computation.
        # In use_1d schema, this column is ground_elev for 2D rows and
        # surface_elev_1d for 1D rows — both standardised with the SAME stats
        # so denormalising recovers each node's relevant surface elevation.
        elev_norm = X_current[:, 3]
        elev_real = elev_norm * (elev_std + epsilon) + elev_mean

        # v2: build per-node water-depth (mean, std) tensors. For 2D-only
        # runs these collapse to scalar broadcasts of the legacy values.
        if use_1d and node_type is not None:
            is_1d = (node_type == 1)
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
            wd_std_per_node  = wd_std

        # Determine the static block width once. The dynamic block (water_depth
        # window + volume window) is always at the END of the row, so:
        n_static = X_current.size(1) - 2 * n_time_steps

        X_iter = X_current.clone()

        for t in range(rollout_length):
            static_part = X_iter[:, :n_static]
            water_depth_window = X_iter[:, n_static : n_static + n_time_steps]
            volume_window = X_iter[:, n_static + n_time_steps : n_static + 2 * n_time_steps]

            X_input = torch.cat(
                [static_part, water_depth_window, volume_window], dim=1
            )

            # Predict the differences (delta water_level and delta volume).
            out = model(X_input, edge_features, g)
            pred = out[0] if isinstance(out, tuple) else out  # shape: (num_nodes, 2)
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]

            # Change B.3: autoregressive Q rollout. The 11th edge_attr column
            # holds normalised Q_prev; update it with the predicted ΔQ so the
            # next step sees Q_pred[t] = Q_prev[t-1] + ΔQ_pred (DUALFloodGNN
            # autoregressive flow rollout, doc §9.5 option 1). The first-step
            # value comes from GT at the last warmup step (dataset B.2).
            if edge_q_prev_as_input and isinstance(out, tuple) and len(out) > 1:
                edge_pred = out[1].squeeze(-1)  # [E]
                edge_features = edge_features.clone()
                edge_features[:, -1] = edge_features[:, -1] + edge_pred

            # Update dynamic window.
            water_depth_updated = torch.cat(
                [water_depth_window[:, 1:], new_wd], dim=1
            )
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)

            # Update static part: inflow at col 10, precip at col 11
            # (positions are constant in both schemas; static block is just wider).
            new_flow = inflow_seq[t].unsqueeze(0).expand(num_nodes, 1)
            new_precip = precip_seq[t].unsqueeze(0).expand(num_nodes, 1)
            static_part_updated = static_part.clone()
            static_part_updated[:, 10:12] = torch.cat([new_flow, new_precip], dim=1)

            X_iter = torch.cat(
                [static_part_updated, water_depth_updated, volume_updated], dim=1
            )

            # Denormalize to physical water level and compute depth above the
            # relevant surface (ground for 2D, manhole rim for 1D). Per-node
            # mean/std handles per-type stats under use_1d.
            pred_wl = new_wd.squeeze(1) * (wd_std_per_node + epsilon) + wd_mean_per_node
            gt_wl   = wd_gt_seq[t]      * (wd_std_per_node + epsilon) + wd_mean_per_node
            pred_depth = torch.clamp(pred_wl - elev_real, min=0.0)
            gt_depth = torch.clamp(gt_wl - elev_real, min=0.0)

            rollout_preds.append(pred_depth.detach().cpu())
            ground_truth_list.append(gt_depth.detach().cpu())

            rmse = torch.sqrt(
                torch.mean((pred_depth - gt_depth) ** 2)
            ).item()
            rmse_list.append(rmse)

            if node_type is not None:
                m2 = node_type == 0
                m1 = node_type == 1
                if m2.any():
                    rmse_2d_list.append(
                        torch.sqrt(
                            torch.mean((pred_depth[m2] - gt_depth[m2]) ** 2)
                        ).item()
                    )
                if m1.any():
                    rmse_1d_list.append(
                        torch.sqrt(
                            torch.mean((pred_depth[m1] - gt_depth[m1]) ** 2)
                        ).item()
                    )

        all_rmse_all.append(rmse_list)
        if rmse_2d_list:
            all_rmse_2d.append(rmse_2d_list)
        if rmse_1d_list:
            all_rmse_1d.append(rmse_1d_list)
        mean_rmse_sample = sum(rmse_list) / len(rmse_list)
        sample_id = test_dataset.dynamic_data[idx].get("event_id", idx)
        if rmse_2d_list and rmse_1d_list:
            print(
                f"Event {sample_id}: Mean RMSE = {mean_rmse_sample:.4f} m | "
                f"2D = {sum(rmse_2d_list)/len(rmse_2d_list):.4f} m | "
                f"1D = {sum(rmse_1d_list)/len(rmse_1d_list):.4f} m"
            )
        else:
            print(f"Event {sample_id}: Mean RMSE = {mean_rmse_sample:.4f} m")

        anim_filename = os.path.join(anim_output_dir, f"animation_{sample_id}.gif")
        create_animation(rollout_preds, ground_truth_list, g, rmse_list, anim_filename)

    all_rmse_tensor = torch.tensor(all_rmse_all)
    overall_mean_rmse = torch.mean(all_rmse_tensor, dim=0)
    overall_std_rmse = torch.std(all_rmse_tensor, dim=0)
    print("Overall Mean RMSE (m) over rollout steps:", overall_mean_rmse)
    print("Overall Std RMSE (m) over rollout steps:", overall_std_rmse)

    if all_rmse_2d and all_rmse_1d:
        m2 = torch.tensor(all_rmse_2d)
        m1 = torch.tensor(all_rmse_1d)
        print("Overall Mean RMSE — 2D nodes:", torch.mean(m2, dim=0))
        print("Overall Std  RMSE — 2D nodes:", torch.std(m2, dim=0))
        print("Overall Mean RMSE — 1D nodes:", torch.mean(m1, dim=0))
        print("Overall Std  RMSE — 1D nodes:", torch.std(m1, dim=0))

    # 5 minutes per step for UrbanFlood.
    timesteps = [(i + 1) * (5 / 60) for i in range(rollout_length)]
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, overall_mean_rmse.numpy(), label="Mean RMSE", linewidth=3)
    plt.fill_between(
        timesteps,
        (overall_mean_rmse - overall_std_rmse).numpy(),
        (overall_mean_rmse + overall_std_rmse).numpy(),
        alpha=0.3,
        label="± Std",
    )
    plt.xlabel("Time (Hours)", fontsize=20)
    plt.ylabel("RMSE (m)", fontsize=20)
    plt.title("Overall RMSE Curve Over Rollout", fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
