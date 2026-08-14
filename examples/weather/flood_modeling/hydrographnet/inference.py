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
  4. RMSE curve over time (in feet)
"""

import os
import json
import math
import numpy as np
import torch
import hydra
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from physicsnemo.utils import load_checkpoint
from metrics import (
    compute_csi,
    compute_mass_balance_error,
    compute_mass_balance_residual,
    compute_nse,
)

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
      rollout_predictions: list of predicted water depth above ground tensors (feet)
      ground_truth: list of ground truth water depth above ground tensors (feet)
      initial_graph: the initial PyG graph sample (used for node positions and edges)
      rmse_list: list of RMSE values in feet computed at each rollout step
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
        fig.colorbar(nodes_pred, cax=cax1, label="Depth (ft)")

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
        fig.colorbar(nodes_gt, cax=cax2, label="Depth (ft)")

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
        fig.colorbar(nodes_error, cax=cax3, label="Error (ft)")

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
        axes[1, 1].set_ylabel("RMSE (ft)", fontsize=24)
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
    # Long-rollout runs render one GIF frame per step at 3000x3000 px, which
    # dominates runtime once rollout_length leaves the 8-step regime. Default
    # stays True so existing invocations are unchanged.
    save_animations = bool(cfg.get("save_animations", True))
    # Full-event rollout: roll each test event to its OWN length (not a single
    # global num_test_time_steps). Needed so every event's real flood window is
    # scored — the fixed 8-step window froze out the higher rainfall bins.
    full_event = bool(cfg.get("full_event_rollout", False))

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Instantiate the test dataset.
    use_1d = cfg.get("use_1d", False)
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
        full_event_rollout=full_event,
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
    vol_stats = test_dataset.dynamic_stats["volume"]
    vol_mean = vol_stats["mean"]
    vol_std = vol_stats["std"]
    if use_1d:
        vol_mean_2d = vol_stats["2d"]["mean"]
        vol_std_2d = vol_stats["2d"]["std"]
        vol_mean_1d = vol_stats["1d"]["mean"]
        vol_std_1d = vol_stats["1d"]["std"]
    precip_stats = test_dataset.dynamic_stats["precipitation"]
    precip_mean = precip_stats["mean"]
    precip_std = precip_stats["std"]
    delta_t = float(cfg.get("delta_t", 300.0))
    area_sum_2d_ft2 = float(np.sum(test_dataset.static_data["area_denorm"]))
    ft3_to_m3 = 0.028316846592
    ft6_to_m6 = ft3_to_m3 ** 2
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
    # The autoregressive rollout below is not wrapped in no_grad, so the graph
    # was accumulating across every step — harmless at 8 steps, OOM past ~40 on
    # a 10 GB card. Nothing in this script backprops, so disable grad globally.
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------------
    # v9: time-less peak-depth prediction — one forward pass per event.
    # Emits per-event + summary lines in the exact v8 log format so the
    # existing aggregator parses it unchanged (summary as 1-element tensors).
    # ------------------------------------------------------------------

    all_rmse_all = []
    all_rmse_2d = []   # per-event RMSE on 2D nodes only (use_1d case)
    all_rmse_1d = []   # per-event RMSE on 1D nodes only (use_1d case)

    # Phase A (scale-free re-analysis): per-event 2D-only NSE / CSI / scale-free
    # error, mirroring the HydrographNet combo-holdout metrics.json so the two
    # datasets are compared with identical definitions. 2D is the governing
    # field for UrbanFlood (1D is an input, not a target).
    tau_wet = float(cfg.get("scalefree_tau", 0.01))  # min GT std (ft) for NSE conditioning
    per_hydrograph = {}

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
        # Phase A: per-step 2D-only scale-free diagnostics (conditioned on wet GT).
        nse_2d_list = []           # per-step NSE on conditioned steps only
        csi005_2d_list = []
        csi030_2d_list = []
        rmse_over_sigma_list = []  # companion global RSR = RMSE_2d / max(std_gt, tau)
        nse_2d_series = []         # step-aligned NSE (NaN on unconditioned steps)
        sd_gt_2d_list = []         # per-step GT spatial std (ft)
        mbe_ft6_list = []
        mbe_m6_list = []
        mbe_gt_ft6_list = []
        mbe_gt_m6_list = []
        mass_residual_ft3_list = []
        mass_residual_gt_ft3_list = []
        balance_depth_mm_list = []
        balance_depth_gt_mm_list = []

        inflow_seq = rollout_data["inflow"].to(device)
        precip_seq = rollout_data["precipitation"].to(device)
        drainage_outflow_seq = rollout_data["drainage_outflow"].to(device)
        wd_gt_seq = rollout_data["water_depth_gt"].to(device)
        vol_gt_seq = rollout_data["volume_gt"].to(device)
        # Per-event rollout length = returned GT sequence length. In full-event
        # mode this is the event's own length; otherwise the fixed global window.
        n_steps_ev = wd_gt_seq.shape[0]

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
            vol_mean_per_node = torch.where(
                is_1d,
                torch.tensor(vol_mean_1d, device=device, dtype=torch.float64),
                torch.tensor(vol_mean_2d, device=device, dtype=torch.float64),
            )
            vol_std_per_node = torch.where(
                is_1d,
                torch.tensor(vol_std_1d, device=device, dtype=torch.float64),
                torch.tensor(vol_std_2d, device=device, dtype=torch.float64),
            )
        else:
            wd_mean_per_node = wd_mean
            wd_std_per_node  = wd_std
            vol_mean_per_node = torch.full(
                (num_nodes,), float(vol_mean), device=device, dtype=torch.float64
            )
            vol_std_per_node = torch.full(
                (num_nodes,), float(vol_std), device=device, dtype=torch.float64
            )
        mbe_mask = (
            node_type == 0
            if node_type is not None
            else torch.ones(num_nodes, device=device, dtype=torch.bool)
        )

        # Determine the static block width once. The dynamic block (water_depth
        # window + volume window) is always at the END of the row, so:
        n_static = X_current.size(1) - 2 * n_time_steps

        X_iter = X_current.clone()

        for t in range(n_steps_ev):
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

            # HydroGraphNet paper Eq. 33 on the governing 2D surface domain.
            # UF rainfall is already an interval depth in inches, so convert it
            # to an interval-average source rate. inlet_flow_1d is the signed
            # drainage discharge leaving the 2D surface. Passing the same
            # interval-average rate at both endpoints makes Eq. 33's trapezoid
            # integral equal the source volume represented by this stored step.
            previous_vol_norm = volume_window[:, -1]
            gt_previous_vol_norm = (
                previous_vol_norm if t == 0 else vol_gt_seq[t - 1]
            )
            pred_vol_real = (
                new_vol.squeeze(1).to(torch.float64)
                * (vol_std_per_node + epsilon)
                + vol_mean_per_node
            )
            previous_pred_vol_real = (
                previous_vol_norm.to(torch.float64)
                * (vol_std_per_node + epsilon)
                + vol_mean_per_node
            )
            gt_vol_real = (
                vol_gt_seq[t].to(torch.float64)
                * (vol_std_per_node + epsilon)
                + vol_mean_per_node
            )
            previous_gt_vol_real = (
                gt_previous_vol_norm.to(torch.float64)
                * (vol_std_per_node + epsilon)
                + vol_mean_per_node
            )
            precip_in = (
                precip_seq[t].to(torch.float64) * (precip_std + epsilon)
                + precip_mean
            )
            rainfall_rate_ft3_s = (
                precip_in * area_sum_2d_ft2 / 12.0 / delta_t
            )
            net_source_rate_ft3_s = (
                rainfall_rate_ft3_s - drainage_outflow_seq[t]
            )
            mass_residual_ft3 = compute_mass_balance_residual(
                pred_vol_real[mbe_mask],
                previous_pred_vol_real[mbe_mask],
                net_source_rate_ft3_s,
                net_source_rate_ft3_s,
                delta_t,
            )
            mbe_ft6 = compute_mass_balance_error(
                pred_vol_real[mbe_mask],
                previous_pred_vol_real[mbe_mask],
                net_source_rate_ft3_s,
                net_source_rate_ft3_s,
                delta_t,
            )
            mass_residual_gt_ft3 = compute_mass_balance_residual(
                gt_vol_real[mbe_mask],
                previous_gt_vol_real[mbe_mask],
                net_source_rate_ft3_s,
                net_source_rate_ft3_s,
                delta_t,
            )
            mbe_gt_ft6 = compute_mass_balance_error(
                gt_vol_real[mbe_mask],
                previous_gt_vol_real[mbe_mask],
                net_source_rate_ft3_s,
                net_source_rate_ft3_s,
                delta_t,
            )
            mbe_ft6_list.append(mbe_ft6)
            mbe_m6_list.append(mbe_ft6 * ft6_to_m6)
            mbe_gt_ft6_list.append(mbe_gt_ft6)
            mbe_gt_m6_list.append(mbe_gt_ft6 * ft6_to_m6)
            mass_residual_ft3_list.append(mass_residual_ft3)
            mass_residual_gt_ft3_list.append(mass_residual_gt_ft3)
            balance_depth_mm_list.append(
                abs(mass_residual_ft3) / area_sum_2d_ft2 * 304.8
            )
            balance_depth_gt_mm_list.append(
                abs(mass_residual_gt_ft3) / area_sum_2d_ft2 * 304.8
            )

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
                    p2, g2 = pred_depth[m2], gt_depth[m2]
                    rmse_2d = torch.sqrt(torch.mean((p2 - g2) ** 2)).item()
                    rmse_2d_list.append(rmse_2d)
                    # Scale-free diagnostics on the 2D field. Population std of GT
                    # gates NSE: when the observed field is near-flat (dry / early
                    # steps) NSE is ill-posed (sqrt(1-NSE) blows up), so only count
                    # steps whose GT std >= tau; always carry the robust companion.
                    sd_gt = torch.std(g2, unbiased=False).item()
                    sd_gt_2d_list.append(sd_gt)
                    csi005_2d_list.append(compute_csi(p2, g2, 0.05))
                    csi030_2d_list.append(compute_csi(p2, g2, 0.30))
                    rmse_over_sigma_list.append(rmse_2d / max(sd_gt, tau_wet))
                    if sd_gt >= tau_wet:
                        _nse_step = compute_nse(p2, g2)
                        nse_2d_list.append(_nse_step)
                        nse_2d_series.append(_nse_step)
                    else:
                        # keep the series step-aligned; unconditioned = NaN
                        nse_2d_series.append(float("nan"))
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
                f"Event {sample_id}: Mean RMSE = {mean_rmse_sample:.4f} ft | "
                f"2D = {sum(rmse_2d_list)/len(rmse_2d_list):.4f} ft | "
                f"1D = {sum(rmse_1d_list)/len(rmse_1d_list):.4f} ft"
            )
        else:
            print(f"Event {sample_id}: Mean RMSE = {mean_rmse_sample:.4f} ft")
        print(
            f"Event {sample_id}: Eq33 MBE={sum(mbe_m6_list)/len(mbe_m6_list):.6e} m^6 "
            f"| GT floor={sum(mbe_gt_m6_list)/len(mbe_gt_m6_list):.6e} m^6"
        )

        # Phase A: record per-event 2D-only scale-free metrics for metrics.json.
        def _nanmean(xs):
            vals = [v for v in xs if not math.isnan(v)]
            return sum(vals) / len(vals) if vals else float("nan")

        mean_rmse_2d = (
            sum(rmse_2d_list) / len(rmse_2d_list) if rmse_2d_list else float("nan")
        )
        if nse_2d_list:
            mean_nse_2d = sum(nse_2d_list) / len(nse_2d_list)
            scalefree_err = math.sqrt(max(0.0, 1.0 - mean_nse_2d))
            scalefree_defined = True
        else:
            mean_nse_2d = float("nan")
            scalefree_err = float("nan")
            scalefree_defined = False
        mean_ros = (
            sum(rmse_over_sigma_list) / len(rmse_over_sigma_list)
            if rmse_over_sigma_list else float("nan")
        )
        per_hydrograph[str(sample_id)] = {
            "mean_rmse": mean_rmse_2d,            # 2D-only, feet (governing field)
            "mean_rmse_allnodes": mean_rmse_sample,
            "mean_nse": mean_nse_2d,
            "mean_csi_005": _nanmean(csi005_2d_list),
            "mean_csi_030": _nanmean(csi030_2d_list),
            "mean_mbe": sum(mbe_ft6_list) / len(mbe_ft6_list),
            "mean_mbe_m6": sum(mbe_m6_list) / len(mbe_m6_list),
            "mean_mbe_gt": sum(mbe_gt_ft6_list) / len(mbe_gt_ft6_list),
            "mean_mbe_gt_m6": sum(mbe_gt_m6_list) / len(mbe_gt_m6_list),
            "mean_mass_residual_ft3": (
                sum(mass_residual_ft3_list) / len(mass_residual_ft3_list)
            ),
            "mean_mass_residual_gt_ft3": (
                sum(mass_residual_gt_ft3_list) / len(mass_residual_gt_ft3_list)
            ),
            "mean_abs_balance_depth_mm": (
                sum(balance_depth_mm_list) / len(balance_depth_mm_list)
            ),
            "mean_abs_balance_depth_gt_mm": (
                sum(balance_depth_gt_mm_list) / len(balance_depth_gt_mm_list)
            ),
            "scalefree_err": scalefree_err,
            "scalefree_err_pct": (100.0 * scalefree_err
                                  if scalefree_defined else float("nan")),
            "rmse_over_sigma": mean_ros,
            "n_timesteps": len(rmse_2d_list),
            "n_conditioned": len(nse_2d_list),
            "scalefree_defined": scalefree_defined,
            # Per-step 2D series so ANY shorter horizon can be recovered as a
            # prefix of a long rollout without re-running inference. All are
            # step-aligned (same length = n_timesteps); nse is NaN on
            # unconditioned (near-dry) steps.
            "rmse_2d_series": list(rmse_2d_list),
            "nse_2d_series": list(nse_2d_series),
            "csi005_2d_series": list(csi005_2d_list),
            "csi030_2d_series": list(csi030_2d_list),
            "rmse_over_sigma_series": list(rmse_over_sigma_list),
            "mbe_ft6_series": list(mbe_ft6_list),
            "mbe_m6_series": list(mbe_m6_list),
            "mbe_gt_ft6_series": list(mbe_gt_ft6_list),
            "mbe_gt_m6_series": list(mbe_gt_m6_list),
            "mass_residual_ft3_series": list(mass_residual_ft3_list),
            "mass_residual_gt_ft3_series": list(mass_residual_gt_ft3_list),
            "abs_balance_depth_mm_series": list(balance_depth_mm_list),
            "abs_balance_depth_gt_mm_series": list(balance_depth_gt_mm_list),
            "sd_gt_series": list(sd_gt_2d_list),
        }

        if save_animations:
            anim_filename = os.path.join(anim_output_dir, f"animation_{sample_id}.gif")
            create_animation(rollout_preds, ground_truth_list, g, rmse_list, anim_filename)

    # Full-event mode gives each event its own length, so stacking the per-step
    # RMSE lists (torch.tensor over ragged rows) is undefined — guard it. The
    # per-event metrics.json below is the authoritative output either way.
    _lengths = {len(r) for r in all_rmse_all}
    _stackable = bool(all_rmse_all) and len(_lengths) == 1
    overall_mean_rmse = overall_std_rmse = None
    if _stackable:
        all_rmse_tensor = torch.tensor(all_rmse_all)
        overall_mean_rmse = torch.mean(all_rmse_tensor, dim=0)
        overall_std_rmse = torch.std(all_rmse_tensor, dim=0)
        print("Overall Mean RMSE (ft) over rollout steps:", overall_mean_rmse)
        print("Overall Std RMSE (ft) over rollout steps:", overall_std_rmse)
        if all_rmse_2d and all_rmse_1d and len({len(r) for r in all_rmse_2d}) == 1:
            m2 = torch.tensor(all_rmse_2d)
            m1 = torch.tensor(all_rmse_1d)
            print("Overall Mean RMSE — 2D nodes:", torch.mean(m2, dim=0))
            print("Overall Std  RMSE — 2D nodes:", torch.std(m2, dim=0))
            print("Overall Mean RMSE — 1D nodes:", torch.mean(m1, dim=0))
            print("Overall Std  RMSE — 1D nodes:", torch.std(m1, dim=0))
    else:
        print(f"Variable-length rollout (lengths {sorted(_lengths)}) — skipping "
              f"stacked overall-RMSE curve; per-event metrics.json is authoritative.")

    # Phase A: write per-event scale-free metrics.json next to the GIFs, mirroring
    # the HydrographNet combo-holdout metrics.json (per_hydrograph + overall block).
    def _mean_defined(key):
        vals = [
            v[key] for v in per_hydrograph.values()
            if isinstance(v[key], float) and not math.isnan(v[key])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    metrics_output = {
        "config": {
            "ckpt_path": str(ckpt_path),
            "epoch_loaded": int(epoch_loaded) if epoch_loaded is not None else None,
            "rollout_length": int(rollout_length),
            "full_event_rollout": full_event,
            "model_name": model_name,
            "use_1d": bool(use_1d),
            "use_local_physics_loss": bool(use_local_physics_loss),
            "scalefree_tau": tau_wet,
            "n_events": len(per_hydrograph),
            "mbe_definition": "HydroGraphNet Eq. 33 squared global continuity residual",
            "mbe_volume_scope": "2D surface nodes",
            "mbe_native_unit": "ft^6",
            "mbe_si_unit": "m^6",
            "mbe_delta_t_seconds": delta_t,
            "mbe_source_discretization": (
                "per-step rain depth and interval-average drainage discharge"
            ),
            "mbe_source_assumption": (
                "zero unobserved infiltration and boundary surface outflow"
            ),
        },
        "overall": {
            "rmse_2d_mean": _mean_defined("mean_rmse"),
            "nse_2d_mean": _mean_defined("mean_nse"),
            "csi_005_mean": _mean_defined("mean_csi_005"),
            "csi_030_mean": _mean_defined("mean_csi_030"),
            "mbe_mean": _mean_defined("mean_mbe"),
            "mbe_mean_m6": _mean_defined("mean_mbe_m6"),
            "mbe_gt_mean": _mean_defined("mean_mbe_gt"),
            "mbe_gt_mean_m6": _mean_defined("mean_mbe_gt_m6"),
            "mass_residual_mean_ft3": _mean_defined("mean_mass_residual_ft3"),
            "mass_residual_gt_mean_ft3": _mean_defined(
                "mean_mass_residual_gt_ft3"
            ),
            "abs_balance_depth_mean_mm": _mean_defined(
                "mean_abs_balance_depth_mm"
            ),
            "abs_balance_depth_gt_mean_mm": _mean_defined(
                "mean_abs_balance_depth_gt_mm"
            ),
            "scalefree_err_pct_mean": _mean_defined("scalefree_err_pct"),
            "rmse_over_sigma_mean": _mean_defined("rmse_over_sigma"),
            "n_scalefree_undefined": sum(
                1 for v in per_hydrograph.values() if not v["scalefree_defined"]
            ),
        },
        "per_hydrograph": per_hydrograph,
    }
    metrics_path = os.path.join(anim_output_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics_output, fh, indent=2)
    print(
        f"Wrote per-event scale-free metrics to {metrics_path} "
        f"({len(per_hydrograph)} events, "
        f"{metrics_output['overall']['n_scalefree_undefined']} scale-free undefined)"
    )

    # 5 minutes per step for UrbanFlood. Only meaningful when every event shares
    # one rollout length (guarded); skipped for variable-length full-event runs.
    if _stackable and overall_mean_rmse is not None:
        timesteps = [(i + 1) * (5 / 60) for i in range(len(overall_mean_rmse))]
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
        plt.ylabel("RMSE (ft)", fontsize=20)
        plt.title("Overall RMSE Curve Over Rollout", fontsize=24)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
