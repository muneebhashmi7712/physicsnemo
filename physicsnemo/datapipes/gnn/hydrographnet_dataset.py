# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
UrbanFloodDataset module

Adapts the HydroGraphDataset interface for the UrbanFlood pluvial urban flood dataset.
Uses 2D surface nodes only (Phase 1 baseline). Data is loaded from CSV files with
explicit mesh connectivity instead of k-NN graph construction.

The dataset supports two modes:
    - Training: Each sample is a sliding window sample.
    - Testing: Each sample corresponds to an entire event with rollout data.
"""

import json
import logging
import math
import os
import re
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from physicsnemo.core.version_check import OptionalImport

pyg = OptionalImport("torch_geometric")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

STATIC_NORM_STATS_FILE = "static_norm_stats.json"
DYNAMIC_NORM_STATS_FILE = "dynamic_norm_stats.json"


class UrbanFloodDataset(Dataset):
    """
    Dataset for UrbanFlood 2D surface nodes.

    Mirrors the HydroGraphDataset interface but loads UrbanFlood CSV data with
    explicit mesh connectivity.

    Args:
        name: Dataset name identifier.
        data_dir: Root directory containing Model_1/ and Model_2/ subdirectories.
        model_name: Which city model to use ("Model_1" or "Model_2").
        split: "train" or "test".
        n_time_steps: Number of time steps in the sliding window.
        noise_std: Standard deviation for added noise.
        noise_type: Type of noise to apply.
        num_samples: Maximum number of events to load.
        rollout_length: Number of rollout steps (test mode only).
        return_physics: Whether to include physics data in output.
    """

    def __init__(
        self,
        name: str = "urbanflood_dataset",
        data_dir: str = "",
        model_name: str = "Model_1",
        split: str = "train",
        n_time_steps: int = 2,
        noise_std: float = 0.01,
        noise_type: str = "none",
        num_samples: int = 500,
        rollout_length: Optional[int] = None,
        return_physics: bool = False,
        delta_t: float = 300.0,
        use_1d: bool = False,
        compute_boundary_mask: bool = False,
        mask_inlets_in_lc: bool = False,
        edge_q_prev_as_input: bool = False,
        lmc_antisymmetric: bool = False,
        train_rollout_length: int = 1,
        static_prediction: bool = False,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")

        self.data_dir = str(data_dir)
        self.model_name = model_name
        self.split = split
        self.split_dir = os.path.join(self.data_dir, model_name, split)
        self.n_time_steps = n_time_steps
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.num_samples = num_samples
        self.rollout_length = rollout_length if rollout_length is not None else 0
        self.return_physics = return_physics
        self.delta_t = delta_t
        self.use_1d = use_1d
        self.compute_boundary_mask = compute_boundary_mask
        self.mask_inlets_in_lc = mask_inlets_in_lc
        self.edge_q_prev_as_input = edge_q_prev_as_input
        self.lmc_antisymmetric = lmc_antisymmetric
        self.train_rollout_length = max(1, int(train_rollout_length))
        # v9: time-less peak-depth prediction. When True, each event becomes a
        # single graph (one forward pass, no rollout/window); node input gets a
        # per-event total-rainfall scalar instead of the per-step precip/window,
        # and the target is per-node peak depth above ground.
        self.static_prediction = static_prediction

        self.static_data = {}
        self.dynamic_data = []
        self.sample_index = []
        self.event_ids = []
        self.static_stats = {}
        self.dynamic_stats = {}
        self.num_fwd_edges = 0          # 2D forward edge count
        self.num_1d_fwd_edges = 0       # 1D pipe forward edge count
        self.num_conn_edges = 0         # 1D-2D connection forward edge count
        self.num_2d_nodes = 0
        self.num_1d_nodes = 0

        self.process()

    def process(self) -> None:
        """Load static/dynamic data, build graph connectivity, compute normalization stats."""
        # --- Load static node data ---
        if self.split == "train":
            (
                xy_coords,
                area,
                area_denorm,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                self.static_stats,
            ) = self.load_static_data(self.split_dir, norm_stats_static=None)
            # Note: when use_1d, static stats are saved again after 1D loading
            # below so depth_1d / base_area_1d are persisted with the 2D keys.
            if not self.use_1d:
                self.save_norm_stats(self.static_stats, STATIC_NORM_STATS_FILE)
        else:
            self.static_stats = self.load_norm_stats(STATIC_NORM_STATS_FILE)
            (
                xy_coords,
                area,
                area_denorm,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                _,
            ) = self.load_static_data(
                self.split_dir, norm_stats_static=self.static_stats
            )

        num_nodes = xy_coords.shape[0]
        self.num_2d_nodes = num_nodes

        # --- Load edge connectivity and features ---
        edge_index, edge_features = self.load_edge_data(self.split_dir)

        self.static_data = {
            "xy_coords": xy_coords,
            "area": area,
            "area_denorm": area_denorm,
            "elevation": elevation,
            "slope": slope,
            "aspect": aspect,
            "curvature": curvature,
            "manning": manning,
            "flow_accum": flow_accum,
            "infiltration": infiltration,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

        # --- 1D system: nodes, pipes, 1D<->2D connections ---
        if self.use_1d:
            (
                xy_1d,
                depth_1d,
                invert_elev_1d,
                surface_elev_1d,
                base_area_1d,
                base_area_1d_denorm,
                self.static_stats,
            ) = self.load_1d_static_data(self.split_dir, self.static_stats)

            self.num_1d_nodes = xy_1d.shape[0]

            # 1D pipe edges (reindex node ids by +num_2d_nodes).
            edge_index_1d, edge_feat_1d = self.load_1d_edge_data(self.split_dir)
            edge_index_1d = edge_index_1d + self.num_2d_nodes  # all 1D node ids

            # 1D <-> 2D connection edges (node_2d unchanged, node_1d shifted).
            edge_index_conn, conn_node_1d_local = self.load_1d2d_connections(
                self.split_dir
            )

            # Bidirectionalise pipe and connection edges (mirrors 2D handling).
            edge_index_1d_bi, edge_feat_1d_bi = self._bidirect(
                edge_index_1d, edge_feat_1d, negate_first_two=True
            )
            edge_feat_conn = self._connection_edge_features(
                edge_index_conn, xy_coords, xy_1d
            )
            edge_index_conn_bi, edge_feat_conn_bi = self._bidirect(
                edge_index_conn, edge_feat_conn, negate_first_two=True
            )

            # Build edge_type one-hot (3 cols: 2D mesh, 1D pipe, connection).
            E_2d_bi = edge_index.shape[1]
            E_1d_bi = edge_index_1d_bi.shape[1]
            E_conn_bi = edge_index_conn_bi.shape[1]
            n_2d_feat = edge_features.shape[1]              # 3 (rel_x, rel_y, length)
            n_1d_feat = edge_feat_1d_bi.shape[1]            # 7 (+ diam, shape, rough, slope)
            n_conn_feat = edge_feat_conn_bi.shape[1]        # 3

            common = 3                                        # rel_x, rel_y, length
            pipe_extra = n_1d_feat - common                   # 4 cols only on 1D pipes

            # Pad 2D and connection edges to match 1D feature width.
            edge_features_padded = np.hstack(
                [edge_features, np.zeros((E_2d_bi, pipe_extra), dtype=edge_features.dtype)]
            )
            edge_feat_conn_padded = np.hstack(
                [edge_feat_conn_bi, np.zeros((E_conn_bi, pipe_extra), dtype=edge_features.dtype)]
            )

            # One-hot type appended to all edges.
            type_2d = np.tile(np.array([1.0, 0.0, 0.0]), (E_2d_bi, 1))
            type_1d = np.tile(np.array([0.0, 1.0, 0.0]), (E_1d_bi, 1))
            type_cn = np.tile(np.array([0.0, 0.0, 1.0]), (E_conn_bi, 1))

            edge_features_2d = np.hstack([edge_features_padded, type_2d])
            edge_features_1d = np.hstack([edge_feat_1d_bi, type_1d])
            edge_features_cn = np.hstack([edge_feat_conn_padded, type_cn])

            edge_index_all = np.concatenate(
                [edge_index, edge_index_1d_bi, edge_index_conn_bi], axis=1
            )
            edge_features_all = np.vstack(
                [edge_features_2d, edge_features_1d, edge_features_cn]
            )

            self.static_data.update(
                {
                    "xy_1d": xy_1d,
                    "depth_1d": depth_1d,
                    "invert_elev_1d": invert_elev_1d,
                    "surface_elev_1d": surface_elev_1d,
                    "base_area_1d": base_area_1d,
                    "base_area_1d_denorm": base_area_1d_denorm,
                    "edge_index": edge_index_all,
                    "edge_features": edge_features_all,
                    "edge_index_2d": edge_index,            # for source-term scatter slicing
                    "edge_index_1d": edge_index_1d_bi,
                    "edge_index_conn": edge_index_conn_bi,
                    "conn_node_1d_local": conn_node_1d_local,  # (E_conn_fwd,)
                }
            )

            self.num_conn_edges = edge_index_conn.shape[1]      # forward count

            # Forward-edge mask for DUALFloodGNN-style antisymmetric LMC.
            # Layout of edge_index_all: each block is [forward | reverse],
            # so within each block the first half is forward.
            #   2D  forward: [0 : E_fwd_2d]
            #   2D  reverse: [E_fwd_2d : 2*E_fwd_2d]
            #   1D  forward: [2*E_fwd_2d : 2*E_fwd_2d + E_fwd_1d]
            #   1D  reverse: [... + E_fwd_1d : 2*(E_fwd_2d + E_fwd_1d)]
            #   conn fwd  : [2*(E_fwd_2d + E_fwd_1d) : ... + E_fwd_conn]
            #   conn rev  : [... + E_fwd_conn : 2*(E_fwd_2d + E_fwd_1d + E_fwd_conn)]
            E_fwd_2d = self.num_fwd_edges
            E_fwd_1d = self.num_1d_fwd_edges
            E_fwd_cn = self.num_conn_edges
            E_total  = 2 * (E_fwd_2d + E_fwd_1d + E_fwd_cn)
            is_fwd = np.zeros(E_total, dtype=bool)
            is_fwd[0 : E_fwd_2d] = True
            is_fwd[2 * E_fwd_2d : 2 * E_fwd_2d + E_fwd_1d] = True
            is_fwd[2 * (E_fwd_2d + E_fwd_1d) :
                   2 * (E_fwd_2d + E_fwd_1d) + E_fwd_cn] = True
            self.is_forward_edge_mask = is_fwd

            # Persist combined 2D + 1D static stats for the test split to load.
            if self.split == "train":
                self.save_norm_stats(self.static_stats, STATIC_NORM_STATS_FILE)

        # --- Convex-hull boundary mask (ported from White-River v7-R1).
        # Computed on the 2D xy_coords. For unified-graph (use_1d=True) we
        # extend with False for every 1D row so the mask is shape [N_total].
        # Optionally OR with the 2D-side inlet nodes (mask_inlets_in_lc).
        if self.compute_boundary_mask:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xy_coords)
            n_total = num_nodes + (self.num_1d_nodes if self.use_1d else 0)
            bmask_np = np.zeros(n_total, dtype=bool)
            bmask_np[hull.vertices] = True
            n_inlets_added = 0
            if self.mask_inlets_in_lc and self.use_1d:
                # 2D-side connection nodes = row-0 of edge_index_conn (forward),
                # which are stored as raw 2D node ids by load_1d2d_connections.
                inlet_2d_ids = self.static_data["edge_index_conn"][0, : self.num_conn_edges]
                inlet_2d_ids = np.asarray(inlet_2d_ids, dtype=np.int64)
                inlet_2d_ids = inlet_2d_ids[inlet_2d_ids < num_nodes]  # safety
                pre = bmask_np.sum()
                bmask_np[inlet_2d_ids] = True
                n_inlets_added = int(bmask_np.sum() - pre)
            self.static_data["boundary_mask"] = bmask_np
            logger.info(
                f"Boundary mask: {int(bmask_np.sum())}/{n_total} nodes "
                f"(convex hull = {len(hull.vertices)}, "
                f"+inlets_added = {n_inlets_added})"
            )

        # --- Discover events ---
        all_entries = os.listdir(self.split_dir)
        event_dirs = [d for d in all_entries if d.startswith("event_")]
        event_dirs.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))
        self.event_ids = event_dirs[: self.num_samples]
        logger.info(
            f"Found {len(event_dirs)} events in {self.split_dir}, "
            f"using {len(self.event_ids)}."
        )

        # --- Load dynamic data per event ---
        temp_dynamic_data = []
        water_depth_list = []
        volume_list = []
        precipitation_list = []
        inflow_list = []
        event_total_rain = []  # v9: per-event total inches (one scalar per event)

        # v9: physical elevation (per node) for peak-depth-above-ground targets.
        # static_stats["elevation"] is populated for both splits (train computes
        # it; test loads it), so denormalising recovers physical ground elevation.
        if self.static_prediction:
            elev_2d_phys = self.denormalize(
                self.static_data["elevation"],
                self.static_stats["elevation"]["mean"],
                self.static_stats["elevation"]["std"],
            ).reshape(-1)  # (N_2d,)
            if self.use_1d:
                surf_1d_phys = self.denormalize(
                    self.static_data["surface_elev_1d"],
                    self.static_stats["elevation"]["mean"],
                    self.static_stats["elevation"]["std"],
                ).reshape(-1)  # (N_1d,)

        for event_id in self.event_ids:
            event_dir = os.path.join(self.split_dir, event_id)
            water_level, inflow, volume, rainfall = self.load_dynamic_data(
                event_dir, num_nodes
            )
            edge_flow = self.load_edge_dynamic_data(
                event_dir, self.num_fwd_edges
            )

            sample = {
                "water_depth": water_level,
                "inflow_hydrograph": inflow,
                "volume": volume,
                "precipitation": rainfall,
                "edge_flow": edge_flow,
                "event_id": event_id,
            }

            if self.static_prediction:
                # rainfall column is already per-step inches, so the sum is the
                # event total inches (matches compute_event_intensity.py).
                sample["event_total_rain"] = float(rainfall.sum())
                sample["n_timesteps"] = int(water_level.shape[0])
                # Per-node peak depth above ground over the whole event.
                peak_2d = np.clip(
                    water_level - elev_2d_phys[None, :], 0.0, None
                ).max(axis=0)  # (N_2d,)
                sample["peak_depth_phys"] = peak_2d

            if self.use_1d:
                wl_1d, inlet_flow_1d, vol_1d = self.load_1d_dynamic_data(
                    event_dir,
                    self.num_1d_nodes,
                    self.static_data["base_area_1d_denorm"],
                )
                edge_flow_1d = self.load_1d_edge_dynamic_data(
                    event_dir, self.num_1d_fwd_edges
                )
                # Concatenate 2D + 1D water_level and volume along the node axis.
                sample["water_depth"] = np.concatenate([water_level, wl_1d], axis=1)
                sample["volume"] = np.concatenate([volume, vol_1d], axis=1)
                sample["edge_flow_1d"] = edge_flow_1d
                sample["inlet_flow_1d"] = inlet_flow_1d  # used as connection-edge GT
                if self.static_prediction:
                    peak_1d = np.clip(
                        wl_1d - surf_1d_phys[None, :], 0.0, None
                    ).max(axis=0)  # (N_1d,)
                    sample["peak_depth_phys"] = np.concatenate(
                        [sample["peak_depth_phys"], peak_1d]
                    )

            temp_dynamic_data.append(sample)
            if self.static_prediction:
                event_total_rain.append(sample["event_total_rain"])
            water_depth_list.append(sample["water_depth"].flatten())
            volume_list.append(sample["volume"].flatten())
            precipitation_list.append(rainfall.flatten())
            inflow_list.append(inflow.flatten())

        # --- Compute or load dynamic normalization stats ---
        # v2: when use_1d, water_depth and volume stats are split per node-type
        # (2D vs 1D) under "2d" / "1d" sub-keys. Legacy "mean" / "std" keys are
        # kept for backward-compat and point at the 2D stats. Edge-flow stats
        # are also split per edge-type (2d pipe / 1d pipe / connection).
        N_2d = self.num_2d_nodes
        N_1d = self.num_1d_nodes
        if self.split == "train":
            self.dynamic_stats = {}
            # Note: each item in water_depth_list / volume_list is shape (T, N).
            # We split along the node axis BEFORE flattening so stats are
            # computed only over their own node type.
            if self.use_1d:
                wd_2d_all = np.concatenate(
                    [a.reshape(-1, N_2d + N_1d)[:, :N_2d].flatten()
                     for a in water_depth_list]
                )
                wd_1d_all = np.concatenate(
                    [a.reshape(-1, N_2d + N_1d)[:, N_2d:].flatten()
                     for a in water_depth_list]
                )
                vol_2d_all = np.concatenate(
                    [a.reshape(-1, N_2d + N_1d)[:, :N_2d].flatten()
                     for a in volume_list]
                )
                vol_1d_all = np.concatenate(
                    [a.reshape(-1, N_2d + N_1d)[:, N_2d:].flatten()
                     for a in volume_list]
                )
                wd_2d_stats = {"mean": float(np.mean(wd_2d_all)),
                               "std":  float(np.std(wd_2d_all))}
                wd_1d_stats = {"mean": float(np.mean(wd_1d_all)),
                               "std":  float(np.std(wd_1d_all))}
                vol_2d_stats = {"mean": float(np.mean(vol_2d_all)),
                                "std":  float(np.std(vol_2d_all))}
                vol_1d_stats = {"mean": float(np.mean(vol_1d_all)),
                                "std":  float(np.std(vol_1d_all))}
                self.dynamic_stats["water_depth"] = {
                    "2d": wd_2d_stats, "1d": wd_1d_stats,
                    "mean": wd_2d_stats["mean"], "std": wd_2d_stats["std"],
                }
                self.dynamic_stats["volume"] = {
                    "2d": vol_2d_stats, "1d": vol_1d_stats,
                    "mean": vol_2d_stats["mean"], "std": vol_2d_stats["std"],
                }
                logger.info(
                    f"v2 per-type stats — 2D vol mean/std = "
                    f"{vol_2d_stats['mean']:.2f}/{vol_2d_stats['std']:.2f}; "
                    f"1D vol mean/std = "
                    f"{vol_1d_stats['mean']:.2f}/{vol_1d_stats['std']:.2f}"
                )
            else:
                water_depth_all = np.concatenate(water_depth_list)
                self.dynamic_stats["water_depth"] = {
                    "mean": float(np.mean(water_depth_all)),
                    "std":  float(np.std(water_depth_all)),
                }
                volume_all = np.concatenate(volume_list)
                self.dynamic_stats["volume"] = {
                    "mean": float(np.mean(volume_all)),
                    "std":  float(np.std(volume_all)),
                }
            precipitation_all = np.concatenate(precipitation_list)
            self.dynamic_stats["precipitation"] = {
                "mean": float(np.mean(precipitation_all)),
                "std": float(np.std(precipitation_all)),
            }
            inflow_all = np.concatenate(inflow_list)
            self.dynamic_stats["inflow_hydrograph"] = {
                "mean": float(np.mean(inflow_all)),
                "std": float(np.std(inflow_all)),
            }

            # v2: per-edge-type flow std (no mean — flows are signed). These
            # are scales for raw m³/s flows, so the conversion to "vol per
            # step" units is `flow_phys * delta_t / sigma_phys` where
            # sigma_phys is the correct edge-type std.
            if self.use_1d:
                ef_2d_all = np.concatenate(
                    [d["edge_flow"].flatten() for d in temp_dynamic_data]
                )
                ef_1d_all = np.concatenate(
                    [d["edge_flow_1d"].flatten() for d in temp_dynamic_data]
                )
                # Connection-edge GT flow comes from inlet_flow_1d at the 1D
                # end of each connection. Pool over (time × connections).
                conn_local = self.static_data["conn_node_1d_local"]
                conn_flow_all = np.concatenate(
                    [d["inlet_flow_1d"][:, conn_local].flatten()
                     for d in temp_dynamic_data]
                )
                self.dynamic_stats["edge_flow_2d"] = {
                    "std": float(np.std(ef_2d_all))
                }
                self.dynamic_stats["edge_flow_1d"] = {
                    "std": float(np.std(ef_1d_all))
                }
                self.dynamic_stats["edge_flow_conn"] = {
                    "std": float(np.std(conn_flow_all))
                }
                logger.info(
                    f"v2 per-edge-type flow std (raw m³/s) — "
                    f"2D = {self.dynamic_stats['edge_flow_2d']['std']:.4f}; "
                    f"1D = {self.dynamic_stats['edge_flow_1d']['std']:.4f}; "
                    f"conn = {self.dynamic_stats['edge_flow_conn']['std']:.4f}"
                )

            # v9: per-event rain-scalar stats + per-node peak-depth target stats
            # (train-only; test loads them below). Following the per-type
            # convention so the 2D-only metric stays governing under use_1d.
            if self.static_prediction:
                self.dynamic_stats["event_rain_scalar"] = {
                    "mean": float(np.mean(event_total_rain)),
                    "std": float(np.std(event_total_rain)),
                }
                if self.use_1d:
                    pk2 = np.concatenate(
                        [s["peak_depth_phys"][:N_2d] for s in temp_dynamic_data]
                    )
                    pk1 = np.concatenate(
                        [s["peak_depth_phys"][N_2d:] for s in temp_dynamic_data]
                    )
                    pk2_stats = {"mean": float(np.mean(pk2)), "std": float(np.std(pk2))}
                    pk1_stats = {"mean": float(np.mean(pk1)), "std": float(np.std(pk1))}
                    self.dynamic_stats["peak_depth"] = {
                        "2d": pk2_stats, "1d": pk1_stats,
                        "mean": pk2_stats["mean"], "std": pk2_stats["std"],
                    }
                else:
                    pk = np.concatenate(
                        [s["peak_depth_phys"] for s in temp_dynamic_data]
                    )
                    self.dynamic_stats["peak_depth"] = {
                        "mean": float(np.mean(pk)), "std": float(np.std(pk)),
                    }
                logger.info(
                    f"v9 static stats — rain scalar mean/std = "
                    f"{self.dynamic_stats['event_rain_scalar']['mean']:.3f}/"
                    f"{self.dynamic_stats['event_rain_scalar']['std']:.3f}; "
                    f"peak depth mean/std = "
                    f"{self.dynamic_stats['peak_depth']['mean']:.4f}/"
                    f"{self.dynamic_stats['peak_depth']['std']:.4f}"
                )

            self.save_norm_stats(self.dynamic_stats, DYNAMIC_NORM_STATS_FILE)
        else:
            self.dynamic_stats = self.load_norm_stats(DYNAMIC_NORM_STATS_FILE)

        # --- Normalize dynamic data ---
        self.dynamic_data = []
        for dyn in temp_dynamic_data:
            if self.use_1d:
                vol_2d_stats = self.dynamic_stats["volume"]["2d"]
                vol_1d_stats = self.dynamic_stats["volume"]["1d"]
                wd_2d_stats  = self.dynamic_stats["water_depth"]["2d"]
                wd_1d_stats  = self.dynamic_stats["water_depth"]["1d"]
                vol_norm = np.concatenate([
                    self.normalize(dyn["volume"][:, :N_2d],
                                   vol_2d_stats["mean"], vol_2d_stats["std"]),
                    self.normalize(dyn["volume"][:, N_2d:],
                                   vol_1d_stats["mean"], vol_1d_stats["std"]),
                ], axis=1)
                wd_norm = np.concatenate([
                    self.normalize(dyn["water_depth"][:, :N_2d],
                                   wd_2d_stats["mean"], wd_2d_stats["std"]),
                    self.normalize(dyn["water_depth"][:, N_2d:],
                                   wd_1d_stats["mean"], wd_1d_stats["std"]),
                ], axis=1)
            else:
                vol_norm = self.normalize(
                    dyn["volume"],
                    self.dynamic_stats["volume"]["mean"],
                    self.dynamic_stats["volume"]["std"],
                )
                wd_norm = self.normalize(
                    dyn["water_depth"],
                    self.dynamic_stats["water_depth"]["mean"],
                    self.dynamic_stats["water_depth"]["std"],
                )
            dyn_std = {
                "water_depth": wd_norm,
                "volume": vol_norm,
                "precipitation": self.normalize(
                    dyn["precipitation"],
                    self.dynamic_stats["precipitation"]["mean"],
                    self.dynamic_stats["precipitation"]["std"],
                ),
                "inflow_hydrograph": self.normalize(
                    dyn["inflow_hydrograph"],
                    self.dynamic_stats["inflow_hydrograph"]["mean"],
                    self.dynamic_stats["inflow_hydrograph"]["std"],
                ),
                "edge_flow": dyn["edge_flow"],  # raw m³/s, converted at __getitem__ time
                "event_id": dyn["event_id"],
            }
            if self.use_1d:
                # raw m³/s for 1D pipe edges and inlet flows; converted at __getitem__ time
                dyn_std["edge_flow_1d"] = dyn["edge_flow_1d"]
                dyn_std["inlet_flow_1d"] = dyn["inlet_flow_1d"]
            if self.static_prediction:
                dyn_std["peak_depth_phys"] = dyn["peak_depth_phys"]
                dyn_std["event_total_rain"] = dyn["event_total_rain"]
                dyn_std["n_timesteps"] = dyn["n_timesteps"]
            self.dynamic_data.append(dyn_std)

        # --- Build sample indices ---
        if self.static_prediction:
            # One graph per event for BOTH splits — no time window, no rollout.
            self.length = len(self.dynamic_data)
            logger.info(
                f"Static prediction: {self.length} event-level samples "
                f"({self.split})."
            )
        elif self.split == "train":
            # For multi-step training rollout (DUALFloodGNN regime), we need
            # `train_rollout_length` consecutive future timesteps after the
            # input window. With pushforward the input window already needs
            # n_time_steps+1 timesteps; the rollout needs O additional ones.
            for h_idx, dyn in enumerate(self.dynamic_data):
                T = dyn["water_depth"].shape[0]
                if self.noise_type == "pushforward":
                    max_t = T - self.n_time_steps - self.train_rollout_length
                else:
                    max_t = T - self.n_time_steps - (self.train_rollout_length - 1)
                if max_t <= 0:
                    raise ValueError(
                        f"Event {dyn.get('event_id', h_idx)} has {T} timesteps, "
                        f"too few for n_time_steps={self.n_time_steps} + "
                        f"train_rollout_length={self.train_rollout_length}."
                    )
                for t in range(max_t):
                    self.sample_index.append((h_idx, t))
            self.length = len(self.sample_index)
            logger.info(
                f"Training samples: {self.length} "
                f"(train_rollout_length={self.train_rollout_length})"
            )
        elif self.split == "test":
            for dyn in self.dynamic_data:
                T = dyn["water_depth"].shape[0]
                if T < self.n_time_steps + self.rollout_length:
                    raise ValueError(
                        f"Event {dyn['event_id']} has {T} timesteps, needs at least "
                        f"{self.n_time_steps + self.rollout_length}."
                    )
            self.length = len(self.dynamic_data)
            logger.info(f"Test samples: {self.length}")

    def _unified_static_blocks(self):
        """Build node static arrays of length N = N_2d + N_1d for the unified
        graph. 2D-only static columns get zeros for 1D rows; the shared
        elevation column gets surface_elev_1d for 1D rows; the trailing
        ``extra_static`` block carries 1D-only quantities + a node_type bit
        (0 = 2D, 1 = 1D)."""
        sd = self.static_data
        n_2d = self.num_2d_nodes
        n_1d = self.num_1d_nodes
        zeros = lambda c: np.zeros((n_1d, c), dtype=sd["xy_coords"].dtype)
        zeros_2d = lambda c: np.zeros((n_2d, c), dtype=sd["xy_coords"].dtype)

        xy = np.vstack([sd["xy_coords"], sd["xy_1d"]])
        area = np.vstack([sd["area"], zeros(1)])
        # Shared elevation column: 2D ground elev, 1D surface elev (same scale).
        elevation = np.vstack([sd["elevation"], sd["surface_elev_1d"]])
        slope = np.vstack([sd["slope"], zeros(1)])
        aspect = np.vstack([sd["aspect"], zeros(1)])
        curvature = np.vstack([sd["curvature"], zeros(1)])
        manning = np.vstack([sd["manning"], zeros(1)])
        flow_accum = np.vstack([sd["flow_accum"], zeros(1)])
        infiltration = np.vstack([sd["infiltration"], zeros(1)])

        # Extra static block: depth_1d, invert_elev_1d, base_area_1d, node_type
        depth_1d_full = np.vstack([zeros_2d(1), sd["depth_1d"]])
        invert_1d_full = np.vstack([zeros_2d(1), sd["invert_elev_1d"]])
        base_area_full = np.vstack([zeros_2d(1), sd["base_area_1d"]])
        node_type = np.vstack(
            [np.zeros((n_2d, 1)), np.ones((n_1d, 1))]
        ).astype(sd["xy_coords"].dtype)
        extra = np.hstack([depth_1d_full, invert_1d_full, base_area_full, node_type])

        return (
            xy, area, elevation, slope, aspect, curvature,
            manning, flow_accum, infiltration, extra,
        )

    def _static_blocks(self):
        """Return the base static node arrays for the active schema."""
        sd = self.static_data
        if self.use_1d:
            return self._unified_static_blocks()
        return (
            sd["xy_coords"], sd["area"], sd["elevation"],
            sd["slope"], sd["aspect"], sd["curvature"],
            sd["manning"], sd["flow_accum"], sd["infiltration"], None,
        )

    def _attach_steady_physics(self, g, dyn):
        """Populate steady-state mass-conservation tensors on the graph (v9).

        Mirrors the per-edge sigma / per-node V_std plumbing of the
        autoregressive path, but with edge_q_prev=0 (the edge decoder
        predicts the steady flow Q directly) and a rainfall-derived source
        term S_phys = inches_m * area / T_steps (m³/step). At the storm peak
        dV/dt≈0, so steady continuity is net_flow + S = 0.
        """
        sd = self.static_data
        num_nodes_s = (
            self.num_2d_nodes + self.num_1d_nodes if self.use_1d
            else sd["xy_coords"].shape[0]
        )
        if self.use_1d:
            V_std_2d = self.dynamic_stats["volume"]["2d"]["std"]
            V_std_1d = self.dynamic_stats["volume"]["1d"]["std"]
            sigma_2d_phys = self.dynamic_stats["edge_flow_2d"]["std"] * self.delta_t
            sigma_1d_phys = self.dynamic_stats["edge_flow_1d"]["std"] * self.delta_t
            sigma_cn_phys = self.dynamic_stats["edge_flow_conn"]["std"] * self.delta_t
            sigma_per_edge = np.concatenate([
                np.full(2 * self.num_fwd_edges,    sigma_2d_phys, dtype=np.float64),
                np.full(2 * self.num_1d_fwd_edges, sigma_1d_phys, dtype=np.float64),
                np.full(2 * self.num_conn_edges,   sigma_cn_phys, dtype=np.float64),
            ])
            V_std_per_node = np.concatenate([
                np.full(self.num_2d_nodes, V_std_2d, dtype=np.float64),
                np.full(self.num_1d_nodes, V_std_1d, dtype=np.float64),
            ])
            area_node = np.concatenate([
                sd["area_denorm"].reshape(-1),
                sd["base_area_1d_denorm"].reshape(-1),
            ])
        else:
            V_std_shared = self.dynamic_stats["volume"]["std"]
            sigma_per_edge = np.full(
                2 * self.num_fwd_edges, V_std_shared, dtype=np.float64
            )
            V_std_per_node = np.full(num_nodes_s, V_std_shared, dtype=np.float64)
            area_node = sd["area_denorm"].reshape(-1)

        # Source term: total rain volume per node, spread over the event steps.
        inches_m = float(dyn["event_total_rain"]) * 0.0254
        T_steps = max(1, int(dyn["n_timesteps"]))
        S_phys = inches_m * area_node / T_steps                  # (N,) m³/step

        E_total = sigma_per_edge.shape[0]
        g.edge_q_prev = torch.zeros(E_total, dtype=torch.float)
        g.Q_sigma_per_edge_phys = torch.tensor(sigma_per_edge, dtype=torch.float)
        g.V_std_per_node = torch.tensor(V_std_per_node, dtype=torch.float)
        g.source_term = torch.tensor(S_phys / V_std_per_node, dtype=torch.float)
        if self.use_1d and self.lmc_antisymmetric and hasattr(
            self, "is_forward_edge_mask"
        ):
            g.is_forward_edge = torch.from_numpy(self.is_forward_edge_mask).bool()

    def _getitem_static(self, idx: int):
        """v9 time-less peak-depth sample: one graph per event."""
        sd = self.static_data
        dyn = self.dynamic_data[idx]
        (
            xy, area, elev, slope, aspect, curv,
            manning, flow_accum, infilt, extra,
        ) = self._static_blocks()

        num_nodes = xy.shape[0]
        rs = self.dynamic_stats["event_rain_scalar"]
        rain_norm = (dyn["event_total_rain"] - rs["mean"]) / (rs["std"] + 1e-8)
        rain_col = np.full((num_nodes, 1), rain_norm, dtype=xy.dtype)

        blocks = [xy, area, elev, slope, aspect, curv, manning, flow_accum, infilt]
        if extra is not None:
            blocks.append(extra)
        blocks.append(rain_col)
        node_features = np.hstack(blocks)

        peak_phys = dyn["peak_depth_phys"]
        if self.use_1d:
            pd2 = self.dynamic_stats["peak_depth"]["2d"]
            pd1 = self.dynamic_stats["peak_depth"]["1d"]
            peak_norm = np.concatenate([
                (peak_phys[: self.num_2d_nodes] - pd2["mean"]) / (pd2["std"] + 1e-8),
                (peak_phys[self.num_2d_nodes:] - pd1["mean"]) / (pd1["std"] + 1e-8),
            ])
        else:
            pds = self.dynamic_stats["peak_depth"]
            peak_norm = (peak_phys - pds["mean"]) / (pds["std"] + 1e-8)

        src, dst = sd["edge_index"]
        edges = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).long()
        g = pyg.data.Data(edge_index=edges)
        g.edge_attr = torch.tensor(sd["edge_features"], dtype=torch.float)
        g.x = torch.tensor(node_features, dtype=torch.float)
        g.y = torch.tensor(peak_norm, dtype=torch.float).unsqueeze(-1)  # (N, 1)

        if self.use_1d:
            g.node_type = torch.tensor(
                np.concatenate([
                    np.zeros(self.num_2d_nodes, dtype=np.int64),
                    np.ones(self.num_1d_nodes, dtype=np.int64),
                ]),
                dtype=torch.long,
            )
        if "boundary_mask" in sd:
            g.boundary_mask = torch.tensor(sd["boundary_mask"], dtype=torch.bool)

        if self.return_physics and self.num_fwd_edges > 0:
            self._attach_steady_physics(g, dyn)

        if self.split == "test":
            meta = {
                "event_id": dyn["event_id"],
                "peak_depth_phys": torch.tensor(peak_phys, dtype=torch.float),
            }
            return g, meta
        if self.return_physics:
            # Per-node physics lives on `g`; the dict is only for collate's
            # (graph, physics_data) contract.
            return g, {}
        return g

    def __getitem__(self, idx: int):
        """Retrieve a graph sample."""
        if self.static_prediction:
            return self._getitem_static(idx)
        sd = self.static_data
        if self.split != "test":
            # --- Training mode: sliding window ---
            hydro_idx, t_idx = self.sample_index[idx]
            dyn = self.dynamic_data[hydro_idx]

            end_index = (
                t_idx + self.n_time_steps + 1
                if self.noise_type == "pushforward"
                else t_idx + self.n_time_steps
            )

            if self.use_1d:
                (
                    xy, area, elev, slope, aspect, curv,
                    manning, flow_accum, infilt, extra,
                ) = self._unified_static_blocks()
            else:
                xy, area, elev, slope, aspect, curv = (
                    sd["xy_coords"], sd["area"], sd["elevation"],
                    sd["slope"], sd["aspect"], sd["curvature"],
                )
                manning, flow_accum, infilt = (
                    sd["manning"], sd["flow_accum"], sd["infiltration"],
                )
                extra = None

            node_features, future_flow, future_precip = self.create_node_features(
                xy, area, elev, slope, aspect, curv,
                manning, flow_accum, infilt,
                dyn["water_depth"][t_idx:end_index, :],
                dyn["volume"][t_idx:end_index, :],
                dyn["precipitation"],
                t_idx,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
                extra_static=extra,
            )
            target_time = t_idx + self.n_time_steps
            prev_time = target_time - 1
            target_depth = (
                dyn["water_depth"][target_time, :] - dyn["water_depth"][prev_time, :]
            )
            target_volume = (
                dyn["volume"][target_time, :] - dyn["volume"][prev_time, :]
            )
            target = np.stack([target_depth, target_volume], axis=1)  # (N, 2)

            src, dst = sd["edge_index"]
            edges = torch.stack(
                [torch.tensor(src), torch.tensor(dst)], dim=0
            ).long()
            g = pyg.data.Data(edge_index=edges)
            g.edge_attr = torch.tensor(sd["edge_features"], dtype=torch.float)
            g.x = torch.tensor(node_features, dtype=torch.float)
            g.y = torch.tensor(target, dtype=torch.float)

            # Multi-step rollout node targets — must be available even when
            # LMC is OFF (nolc baseline still rolls out O steps and needs
            # per-step y to compute pred_loss). Edge / source-term rollout
            # tensors live inside the `return_physics` block below since they
            # are only consumed by the LMC + edge supervision losses.
            O = self.train_rollout_length
            if O > 1:
                num_nodes_g = g.y.shape[0]
                y_seq = np.zeros((num_nodes_g, O, 2), dtype=np.float32)
                y_seq[:, 0, 0] = target_depth
                y_seq[:, 0, 1] = target_volume
                for o in range(1, O):
                    tt = target_time + o
                    pt = prev_time + o
                    y_seq[:, o, 0] = (
                        dyn["water_depth"][tt, :] - dyn["water_depth"][pt, :]
                    )
                    y_seq[:, o, 1] = (
                        dyn["volume"][tt, :] - dyn["volume"][pt, :]
                    )
                g.y_rollout = torch.tensor(y_seq, dtype=torch.float)

            if self.use_1d:
                # 0 for 2D rows, 1 for 1D rows. Used for per-type RMSE / loss masking.
                g.node_type = torch.tensor(
                    np.concatenate(
                        [
                            np.zeros(self.num_2d_nodes, dtype=np.int64),
                            np.ones(self.num_1d_nodes, dtype=np.int64),
                        ]
                    ),
                    dtype=torch.long,
                )

            if "boundary_mask" in sd:
                g.boundary_mask = torch.tensor(sd["boundary_mask"], dtype=torch.bool)

            # Edge flow targets: delta-Q and Q_prev. v2 normalises per-edge by
            # the edge-type's flow std and per-node by the node-type's volume
            # std, so the conservation residual can be formed in physical
            # units (m³) and then made dimensionless by dividing by the
            # per-node volume std.
            if self.return_physics and self.num_fwd_edges > 0:
                num_nodes_s = (
                    self.num_2d_nodes + self.num_1d_nodes
                    if self.use_1d
                    else sd["xy_coords"].shape[0]
                )

                # `edge_index_all` is built per-type, each block already
                # bidirectional: [2D_fwd|2D_rev | 1D_fwd|1D_rev | conn_fwd|conn_rev].
                # sigma / flow tensors must follow the SAME per-type-bi layout
                # so positions align with edge_index_all.
                if self.use_1d:
                    V_std_2d = self.dynamic_stats["volume"]["2d"]["std"]
                    V_std_1d = self.dynamic_stats["volume"]["1d"]["std"]
                    sigma_2d_phys  = self.dynamic_stats["edge_flow_2d"]["std"] * self.delta_t
                    sigma_1d_phys  = self.dynamic_stats["edge_flow_1d"]["std"] * self.delta_t
                    sigma_cn_phys  = self.dynamic_stats["edge_flow_conn"]["std"] * self.delta_t
                    sigma_per_edge = np.concatenate([
                        np.full(2 * self.num_fwd_edges,    sigma_2d_phys, dtype=np.float64),
                        np.full(2 * self.num_1d_fwd_edges, sigma_1d_phys, dtype=np.float64),
                        np.full(2 * self.num_conn_edges,   sigma_cn_phys, dtype=np.float64),
                    ])
                    V_std_per_node = np.concatenate([
                        np.full(self.num_2d_nodes, V_std_2d, dtype=np.float64),
                        np.full(self.num_1d_nodes, V_std_1d, dtype=np.float64),
                    ])
                    flow_2d_prev = dyn["edge_flow"][prev_time, :]
                    flow_2d_targ = dyn["edge_flow"][target_time, :]
                    flow_1d_prev = dyn["edge_flow_1d"][prev_time, :]
                    flow_1d_targ = dyn["edge_flow_1d"][target_time, :]
                    conn_local = sd["conn_node_1d_local"]
                    flow_conn_prev = dyn["inlet_flow_1d"][prev_time, conn_local]
                    flow_conn_targ = dyn["inlet_flow_1d"][target_time, conn_local]
                    flow_prev_bi = np.concatenate([
                        flow_2d_prev,   -flow_2d_prev,
                        flow_1d_prev,   -flow_1d_prev,
                        flow_conn_prev, -flow_conn_prev,
                    ])
                    flow_targ_bi = np.concatenate([
                        flow_2d_targ,   -flow_2d_targ,
                        flow_1d_targ,   -flow_1d_targ,
                        flow_conn_targ, -flow_conn_targ,
                    ])
                else:
                    # Backward-compat: single shared volume std applied to
                    # every edge and every node. The math reduces to the
                    # v1/v5 formulation bit-identically.
                    V_std_shared = self.dynamic_stats["volume"]["std"]
                    sigma_per_edge = np.full(
                        2 * self.num_fwd_edges, V_std_shared, dtype=np.float64
                    )
                    V_std_per_node = np.full(
                        num_nodes_s, V_std_shared, dtype=np.float64
                    )
                    flow_prev_fwd = dyn["edge_flow"][prev_time, :]
                    flow_targ_fwd = dyn["edge_flow"][target_time, :]
                    flow_prev_bi  = np.concatenate([flow_prev_fwd, -flow_prev_fwd])
                    flow_targ_bi  = np.concatenate([flow_targ_fwd, -flow_targ_fwd])

                # Q in physical units (m³ per step) and normalised by per-edge sigma.
                Q_prev_phys   = flow_prev_bi * self.delta_t
                Q_target_phys = flow_targ_bi * self.delta_t
                Q_prev_norm   = Q_prev_phys / sigma_per_edge
                delta_Q_norm  = (Q_target_phys - Q_prev_phys) / sigma_per_edge

                g.edge_y = torch.tensor(delta_Q_norm, dtype=torch.float).unsqueeze(-1)
                g.edge_q_prev = torch.tensor(Q_prev_norm, dtype=torch.float)
                g.Q_sigma_per_edge_phys = torch.tensor(
                    sigma_per_edge, dtype=torch.float
                )
                g.V_std_per_node = torch.tensor(V_std_per_node, dtype=torch.float)

                # Optionally expose GT Q_prev to the model as the 11th edge_attr
                # column (Change B of urbanflood_lc_bundle_v3). Gated so default
                # runs keep the 10-col edge_attr produced above unchanged.
                if self.edge_q_prev_as_input:
                    g.edge_attr = torch.cat(
                        [g.edge_attr, g.edge_q_prev.unsqueeze(-1)], dim=1
                    )

                # Forward-edge mask for DUALFloodGNN-style antisymmetric LMC
                # (Exp 3 / Exp 4 of urbanflood_lc_bundle_v3). When set, utils.py
                # computes the LMC residual using forward edges only and scatter
                # +Q to dst / -Q to src, removing the empirical /2.0.
                if self.use_1d and self.lmc_antisymmetric and hasattr(
                    self, "is_forward_edge_mask"
                ):
                    g.is_forward_edge = torch.from_numpy(
                        self.is_forward_edge_mask
                    ).bool()

                # Ground-truth per-node source term, in V_std-per-node units.
                # ΔV_phys[i] = ΔV_norm[i] * V_std_per_node[i]. Both 2D and 1D
                # nodes have ΔV_norm = (V_norm[t] − V_norm[t-1])[i] because
                # the per-type mean cancels in the difference.
                delta_V_norm_node = (
                    dyn["volume"][target_time, :] - dyn["volume"][prev_time, :]
                )
                delta_V_phys = delta_V_norm_node * V_std_per_node
                Q_in_phys = np.zeros(num_nodes_s)
                np.add.at(Q_in_phys, dst, Q_target_phys)
                Q_out_phys = np.zeros(num_nodes_s)
                np.add.at(Q_out_phys, src, Q_target_phys)
                net_flow_phys = (Q_in_phys - Q_out_phys) / 2.0
                S_phys = delta_V_phys - net_flow_phys              # m³
                g.source_term = torch.tensor(
                    S_phys / V_std_per_node, dtype=torch.float
                )

                # ---- Multi-step rollout targets (LMC bundle v5) -------------
                # Edge + source-term rollout tensors (LMC + edge supervision
                # only). `g.y_rollout` is built earlier outside the
                # return_physics block so it is available for the nolc
                # baseline too. Layout is EDGE/NODE-first so PyG batches
                # dim 0 per default:
                #   edge_y_rollout       : [E, O, 1]  edge ΔQ per rollout step
                #   source_term_rollout  : [N, O]     per-node S per rollout step
                # Step o=0 of each matches the single-step field above.
                O = self.train_rollout_length
                if O > 1:
                    E_total = len(flow_targ_bi)
                    edge_y_seq = np.zeros((E_total, O, 1), dtype=np.float32)
                    src_seq = np.zeros((num_nodes_s, O), dtype=np.float32)

                    # Step 0 already computed above — just stash it.
                    edge_y_seq[:, 0, 0] = delta_Q_norm
                    src_seq[:, 0] = S_phys / V_std_per_node

                    for o in range(1, O):
                        tt = target_time + o
                        pt = prev_time + o
                        # Edge delta-Q for step o (mirror per-type bidirectional layout)
                        if self.use_1d:
                            f2_p = dyn["edge_flow"][pt, :]
                            f2_t = dyn["edge_flow"][tt, :]
                            f1_p = dyn["edge_flow_1d"][pt, :]
                            f1_t = dyn["edge_flow_1d"][tt, :]
                            fc_p = dyn["inlet_flow_1d"][pt, conn_local]
                            fc_t = dyn["inlet_flow_1d"][tt, conn_local]
                            fp_bi_o = np.concatenate([
                                f2_p, -f2_p, f1_p, -f1_p, fc_p, -fc_p,
                            ])
                            ft_bi_o = np.concatenate([
                                f2_t, -f2_t, f1_t, -f1_t, fc_t, -fc_t,
                            ])
                        else:
                            f2_p = dyn["edge_flow"][pt, :]
                            f2_t = dyn["edge_flow"][tt, :]
                            fp_bi_o = np.concatenate([f2_p, -f2_p])
                            ft_bi_o = np.concatenate([f2_t, -f2_t])

                        Qp_phys_o = fp_bi_o * self.delta_t
                        Qt_phys_o = ft_bi_o * self.delta_t
                        edge_y_seq[:, o, 0] = (
                            (Qt_phys_o - Qp_phys_o) / sigma_per_edge
                        )

                        # Source term for step o (matches step-0 formula)
                        dV_norm_o = (
                            dyn["volume"][tt, :] - dyn["volume"][pt, :]
                        )
                        dV_phys_o = dV_norm_o * V_std_per_node
                        Qin_o  = np.zeros(num_nodes_s)
                        np.add.at(Qin_o, dst, Qt_phys_o)
                        Qout_o = np.zeros(num_nodes_s)
                        np.add.at(Qout_o, src, Qt_phys_o)
                        nf_o = (Qin_o - Qout_o) / 2.0
                        S_phys_o = dV_phys_o - nf_o
                        src_seq[:, o] = S_phys_o / V_std_per_node

                    g.y_rollout = torch.tensor(y_seq, dtype=torch.float)
                    g.edge_y_rollout = torch.tensor(edge_y_seq, dtype=torch.float)
                    g.source_term_rollout = torch.tensor(src_seq, dtype=torch.float)

            need_physics = self.return_physics or (self.noise_type == "pushforward")
            if need_physics:
                past_volume = float(np.sum(dyn["volume"][prev_time, :]))
                future_volume = (
                    float(np.sum(dyn["volume"][target_time + 1, :]))
                    if (target_time + 1 < dyn["volume"].shape[0])
                    else float(np.sum(dyn["volume"][target_time, :]))
                )
                avg_inflow_norm = float(
                    (
                        dyn["inflow_hydrograph"][prev_time]
                        + dyn["inflow_hydrograph"][target_time]
                    )
                    / 2
                )
                avg_precip_norm = float(
                    (
                        dyn["precipitation"][prev_time]
                        + dyn["precipitation"][target_time]
                    )
                    / 2
                )
                denorm_avg_inflow = (
                    avg_inflow_norm * self.dynamic_stats["inflow_hydrograph"]["std"]
                    + self.dynamic_stats["inflow_hydrograph"]["mean"]
                )
                denorm_avg_precip = (
                    avg_precip_norm * self.dynamic_stats["precipitation"]["std"]
                    + self.dynamic_stats["precipitation"]["mean"]
                )

                if (target_time + 1) < dyn["inflow_hydrograph"].shape[0]:
                    next_inflow_norm = dyn["inflow_hydrograph"][target_time + 1]
                    next_precip_norm = dyn["precipitation"][target_time + 1]
                else:
                    next_inflow_norm = dyn["inflow_hydrograph"][target_time]
                    next_precip_norm = dyn["precipitation"][target_time]
                denorm_next_inflow = (
                    next_inflow_norm * self.dynamic_stats["inflow_hydrograph"]["std"]
                    + self.dynamic_stats["inflow_hydrograph"]["mean"]
                )
                denorm_next_precip = (
                    next_precip_norm * self.dynamic_stats["precipitation"]["std"]
                    + self.dynamic_stats["precipitation"]["mean"]
                )

                full_physics_data = {
                    "flow_future": float(
                        future_flow
                        * self.dynamic_stats["inflow_hydrograph"]["std"]
                        + self.dynamic_stats["inflow_hydrograph"]["mean"]
                    ),
                    "precip_future": float(
                        future_precip * self.dynamic_stats["precipitation"]["std"]
                        + self.dynamic_stats["precipitation"]["mean"]
                    ),
                    "past_volume": past_volume,
                    "future_volume": future_volume,
                    "avg_inflow": denorm_avg_inflow,
                    "avg_precipitation": denorm_avg_precip,
                    "next_inflow": denorm_next_inflow,
                    "next_precip": denorm_next_precip,
                    "volume_mean": float(self.dynamic_stats["volume"]["mean"]),
                    "volume_std": float(self.dynamic_stats["volume"]["std"]),
                    "inflow_mean": float(
                        self.dynamic_stats["inflow_hydrograph"]["mean"]
                    ),
                    "inflow_std": float(
                        self.dynamic_stats["inflow_hydrograph"]["std"]
                    ),
                    "precip_mean": float(
                        self.dynamic_stats["precipitation"]["mean"]
                    ),
                    "precip_std": float(self.dynamic_stats["precipitation"]["std"]),
                    "num_nodes": float(sd["xy_coords"].shape[0]),
                    "area_sum": float(np.sum(sd["area_denorm"])),
                    "area_mean": float(self.static_stats["area"]["mean"][0]),
                    "area_std": float(self.static_stats["area"]["std"][0]),
                    "infiltration_mean": float(self.static_stats["infiltration"]["mean"][0]),
                    "infiltration_std": float(self.static_stats["infiltration"]["std"][0]),
                    "infiltration_area_sum": float(
                        np.sum(
                            self.denormalize(
                                sd["infiltration"],
                                self.static_stats["infiltration"]["mean"],
                                self.static_stats["infiltration"]["std"],
                            )
                            * sd["area_denorm"]
                        )
                    )
                    / 100.0,
                }
                if not self.return_physics and self.noise_type == "pushforward":
                    physics_data = {
                        "flow_future": full_physics_data["flow_future"],
                        "precip_future": full_physics_data["precip_future"],
                        "next_inflow": full_physics_data["next_inflow"],
                        "next_precip": full_physics_data["next_precip"],
                    }
                else:
                    physics_data = full_physics_data
                return g, physics_data
            else:
                return g
        else:
            # --- Test mode: full event with rollout data ---
            dyn = self.dynamic_data[idx]
            if self.use_1d:
                (
                    xy, area, elev, slope, aspect, curv,
                    manning, flow_accum, infilt, extra,
                ) = self._unified_static_blocks()
            else:
                xy, area, elev, slope, aspect, curv = (
                    sd["xy_coords"], sd["area"], sd["elevation"],
                    sd["slope"], sd["aspect"], sd["curvature"],
                )
                manning, flow_accum, infilt = (
                    sd["manning"], sd["flow_accum"], sd["infiltration"],
                )
                extra = None
            node_features, _, _ = self.create_node_features(
                xy, area, elev, slope, aspect, curv,
                manning, flow_accum, infilt,
                dyn["water_depth"][0 : self.n_time_steps, :],
                dyn["volume"][0 : self.n_time_steps, :],
                dyn["precipitation"],
                0,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
                extra_static=extra,
            )
            src, dst = sd["edge_index"]
            edges = torch.stack(
                [torch.tensor(src), torch.tensor(dst)], dim=0
            ).long()
            g = pyg.data.Data(edge_index=edges)
            g.edge_attr = torch.tensor(sd["edge_features"], dtype=torch.float)
            g.x = torch.tensor(node_features, dtype=torch.float)

            # Change B.2: when edge_q_prev_as_input is on, append GT Q_prev
            # (normalised per-edge previous flow at the last warmup step) as
            # the 11th edge_attr column. Mirrors the train-branch ordering at
            # lines 686-695 so per-edge alignment with edge_index is preserved.
            # Inference.py is responsible for autoregressively refreshing this
            # column across rollout steps (B.3).
            if self.use_1d and self.edge_q_prev_as_input:
                prev_time_test = self.n_time_steps - 1
                sigma_2d_phys  = self.dynamic_stats["edge_flow_2d"]["std"] * self.delta_t
                sigma_1d_phys  = self.dynamic_stats["edge_flow_1d"]["std"] * self.delta_t
                sigma_cn_phys  = self.dynamic_stats["edge_flow_conn"]["std"] * self.delta_t
                sigma_per_edge = np.concatenate([
                    np.full(2 * self.num_fwd_edges,    sigma_2d_phys, dtype=np.float64),
                    np.full(2 * self.num_1d_fwd_edges, sigma_1d_phys, dtype=np.float64),
                    np.full(2 * self.num_conn_edges,   sigma_cn_phys, dtype=np.float64),
                ])
                flow_2d_prev   = dyn["edge_flow"][prev_time_test, :]
                flow_1d_prev   = dyn["edge_flow_1d"][prev_time_test, :]
                conn_local     = sd["conn_node_1d_local"]
                flow_conn_prev = dyn["inlet_flow_1d"][prev_time_test, conn_local]
                flow_prev_bi   = np.concatenate([
                    flow_2d_prev,   -flow_2d_prev,
                    flow_1d_prev,   -flow_1d_prev,
                    flow_conn_prev, -flow_conn_prev,
                ])
                Q_prev_norm0 = (flow_prev_bi * self.delta_t) / sigma_per_edge
                q_prev_col   = torch.tensor(
                    Q_prev_norm0, dtype=torch.float
                ).unsqueeze(-1)
                g.edge_attr  = torch.cat([g.edge_attr, q_prev_col], dim=1)
                g.edge_q_prev = torch.tensor(Q_prev_norm0, dtype=torch.float)

            # Forward-edge mask for antisymmetric LMC (Exp 3/4). LMC isn't
            # evaluated during inference but inference.py may carry the
            # attribute through for consistency / future use.
            if self.use_1d and self.lmc_antisymmetric and hasattr(
                self, "is_forward_edge_mask"
            ):
                g.is_forward_edge = torch.from_numpy(
                    self.is_forward_edge_mask
                ).bool()

            if self.use_1d:
                g.node_type = torch.tensor(
                    np.concatenate(
                        [
                            np.zeros(self.num_2d_nodes, dtype=np.int64),
                            np.ones(self.num_1d_nodes, dtype=np.int64),
                        ]
                    ),
                    dtype=torch.long,
                )
            if "boundary_mask" in sd:
                g.boundary_mask = torch.tensor(sd["boundary_mask"], dtype=torch.bool)
            rollout_data = {
                "inflow": torch.tensor(
                    dyn["inflow_hydrograph"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "precipitation": torch.tensor(
                    dyn["precipitation"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "water_depth_gt": torch.tensor(
                    dyn["water_depth"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "volume_gt": torch.tensor(
                    dyn["volume"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
            }
            return g, rollout_data

    def __len__(self) -> int:
        return self.length

    # ------------------------------------------------------------------
    # Static helpers (identical to HydroGraphDataset)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(data, mean, std, epsilon=1e-8):
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return (data - mean) / (std + epsilon)

    @staticmethod
    def denormalize(data, mean, std, epsilon=1e-8):
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return data * (std + epsilon) + mean

    def apply_noise_to_feature(self, data, noise_type, noise_std):
        if noise_type in ["none", "pushforward"]:
            return data
        T, num_nodes = data.shape
        if noise_type == "only_last":
            noise = np.random.normal(0, noise_std, size=(1, num_nodes))
            data_modified = data.copy()
            data_modified[-1] += noise[0]
            return data_modified
        elif noise_type == "correlated":
            noise = np.random.normal(0, noise_std, size=(1, num_nodes))
            return data + noise
        elif noise_type == "uncorrelated":
            noise = np.random.normal(0, noise_std, size=(T, num_nodes))
            return data + noise
        elif noise_type == "random_walk":
            noise_increments = np.random.normal(
                0, noise_std / math.sqrt(T), size=(T, num_nodes)
            )
            return data + np.cumsum(noise_increments, axis=0)
        else:
            logger.warning(f"Unknown noise_type={noise_type}, skipping noise.")
            return data

    # ------------------------------------------------------------------
    # Norm stats I/O (save to split_dir instead of data_dir)
    # ------------------------------------------------------------------

    def save_norm_stats(self, stats, filename):
        filepath = os.path.join(self.split_dir, filename)
        with open(filepath, "w") as f:
            json.dump(stats, f)
        logger.info(f"Saved norm stats to {filepath}")

    def load_norm_stats(self, filename):
        # Test split loads stats from the *train* directory.
        train_dir = os.path.join(self.data_dir, self.model_name, "train")
        filepath = os.path.join(train_dir, filename)
        with open(filepath, "r") as f:
            stats = json.load(f)
        logger.info(f"Loaded norm stats from {filepath}")
        return stats

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_static_data(self, split_dir, norm_stats_static=None):
        """Load and standardize static 2D node data from CSV."""
        epsilon = 1e-8
        stats = norm_stats_static if norm_stats_static is not None else {}

        def standardize(data, key):
            if key in stats:
                mean_val = np.array(stats[key]["mean"])
                std_val = np.array(stats[key]["std"])
            else:
                mean_val = np.mean(data, axis=0)
                std_val = np.std(data, axis=0)
                stats[key] = {"mean": mean_val.tolist(), "std": std_val.tolist()}
            return (data - mean_val) / (std_val + epsilon)

        csv_path = os.path.join(split_dir, "2d_nodes_static.csv")
        # Columns: node_idx, position_x, position_y, area, roughness,
        #          min_elevation, elevation, aspect, curvature, flow_accumulation
        raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, filling_values=np.nan)
        num_nodes = raw.shape[0]

        # Extract columns (skip node_idx at col 0)
        pos_x = raw[:, 1]
        pos_y = raw[:, 2]
        area_raw = raw[:, 3]
        roughness = raw[:, 4]
        min_elev = raw[:, 5]
        elev = raw[:, 6]
        aspect_raw = raw[:, 7]
        curvature_raw = raw[:, 8]
        flow_accum_raw = raw[:, 9]

        # Fill NaN min_elevation with elevation
        nan_mask = np.isnan(min_elev)
        if np.any(nan_mask):
            logger.info(
                f"Filling {nan_mask.sum()} NaN min_elevation values with elevation."
            )
            min_elev[nan_mask] = elev[nan_mask]

        # Build feature arrays matching HydroGraphNet layout
        xy_coords = np.column_stack([pos_x, pos_y])
        xy_coords = standardize(xy_coords, "xy_coords")

        area_denorm = area_raw.reshape(-1, 1)
        area = standardize(area_denorm.copy(), "area")

        elevation = standardize(elev.reshape(-1, 1), "elevation")

        # Slope: not available per-node in 2D mesh, pad with zeros
        slope_raw = np.zeros((num_nodes, 1))
        slope = standardize(slope_raw, "slope")

        aspect = standardize(aspect_raw.reshape(-1, 1), "aspect")
        curvature = standardize(curvature_raw.reshape(-1, 1), "curvature")
        manning = standardize(roughness.reshape(-1, 1), "manning")
        flow_accum = standardize(flow_accum_raw.reshape(-1, 1), "flow_accum")
        infiltration = standardize(min_elev.reshape(-1, 1), "infiltration")

        return (
            xy_coords,
            area,
            area_denorm,
            elevation,
            slope,
            aspect,
            curvature,
            manning,
            flow_accum,
            infiltration,
            stats,
        )

    def load_edge_data(self, split_dir):
        """Load explicit mesh connectivity and edge features, make bidirectional."""
        # Edge index: edge_idx, from_node, to_node
        edge_index_path = os.path.join(split_dir, "2d_edge_index.csv")
        edge_idx_raw = np.genfromtxt(
            edge_index_path, delimiter=",", skip_header=1, dtype=int
        )
        src_fwd = edge_idx_raw[:, 1]
        dst_fwd = edge_idx_raw[:, 2]

        # Edge features: edge_idx, relative_position_x, relative_position_y,
        #                face_length, length, slope
        edges_static_path = os.path.join(split_dir, "2d_edges_static.csv")
        edges_raw = np.genfromtxt(edges_static_path, delimiter=",", skip_header=1)
        rel_x_fwd = edges_raw[:, 1]
        rel_y_fwd = edges_raw[:, 2]
        length_fwd = edges_raw[:, 4]  # column 4 = length

        # Make bidirectional: forward + reverse
        src_bi = np.concatenate([src_fwd, dst_fwd])
        dst_bi = np.concatenate([dst_fwd, src_fwd])
        edge_index = np.array([src_bi, dst_bi])

        # Reverse edges: negate relative positions, keep same length
        rel_x_bi = np.concatenate([rel_x_fwd, -rel_x_fwd])
        rel_y_bi = np.concatenate([rel_y_fwd, -rel_y_fwd])
        length_bi = np.concatenate([length_fwd, length_fwd])

        # Normalize edge features (z-score, same as HydroGraphDataset.create_edge_features)
        epsilon = 1e-8
        rel_coords = np.column_stack([rel_x_bi, rel_y_bi])
        rel_coords = (rel_coords - np.mean(rel_coords, axis=0)) / (
            np.std(rel_coords, axis=0) + epsilon
        )
        length_norm = (length_bi - np.mean(length_bi)) / (np.std(length_bi) + epsilon)

        edge_features = np.hstack([rel_coords, length_norm[:, None]])

        self.num_fwd_edges = len(src_fwd)
        logger.info(
            f"Loaded {len(src_fwd)} edges, made bidirectional: {edge_index.shape[1]} edges."
        )
        return edge_index, edge_features

    def load_edge_dynamic_data(self, event_dir, num_edges):
        """Load dynamic 2D edge flow data from a single event CSV.

        Returns:
            edge_flow: shape (T, num_edges) — flow rate per forward edge per timestep (m³/s).
        """
        csv_path = os.path.join(event_dir, "2d_edges_dynamic_all.csv")
        # Columns: timestep, edge_idx, flow, velocity
        raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

        timestep_col = raw[:, 0].astype(int)
        flow_col = raw[:, 2]

        timesteps = np.unique(timestep_col)
        T = len(timesteps)

        # Reshape to (T, num_edges) — data ordered by (timestep, edge_idx)
        edge_flow = flow_col.reshape(T, num_edges)
        return edge_flow

    def load_dynamic_data(self, event_dir, num_nodes):
        """Load dynamic 2D node data from a single event CSV.

        Returns:
            Tuple of (water_level, inflow, volume, rainfall) where:
                water_level: shape (T, num_nodes)
                inflow: shape (T,) - all zeros for pluvial system
                volume: shape (T, num_nodes)
                rainfall: shape (T,) - uniform per timestep
        """
        csv_path = os.path.join(event_dir, "2d_nodes_dynamic_all.csv")
        # Columns: timestep, node_idx, rainfall, water_level, water_volume
        raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

        timestep_col = raw[:, 0].astype(int)
        node_idx_col = raw[:, 1].astype(int)
        rainfall_col = raw[:, 2]
        water_level_col = raw[:, 3]
        water_volume_col = raw[:, 4]

        # Determine dimensions
        timesteps = np.unique(timestep_col)
        T = len(timesteps)

        # Reshape to (T, num_nodes) - data is ordered by (timestep, node_idx)
        water_level = water_level_col.reshape(T, num_nodes)
        volume = water_volume_col.reshape(T, num_nodes)

        # Rainfall is uniform per timestep — take from first node of each timestep
        rainfall = rainfall_col.reshape(T, num_nodes)[:, 0]

        # No river inflow in pluvial system
        inflow = np.zeros(T, dtype=np.float64)

        return water_level, inflow, volume, rainfall

    def create_node_features(
        self,
        xy_coords,
        area,
        elevation,
        slope,
        aspect,
        curvature,
        manning,
        flow_accum,
        infiltration,
        water_depth,
        volume,
        precipitation_data,
        time_step,
        n_time_steps,
        inflow_hydrograph,
        extra_static=None,
    ):
        """Create node features.

        Layout (when ``extra_static`` is None — original 16-dim 2D-only schema):
            [xy(2), area, elev, slope, aspect, curv, manning, flow_accum, infilt,
             inflow, precip, water_depth_window(n), volume_window(n)]

        Layout (when ``extra_static`` is provided — unified 1D+2D schema):
            [...same first 12 base static dims...,
             extra_static cols (e.g. depth_1d, invert_elev_1d, base_area_1d, node_type),
             water_depth_window(n), volume_window(n)]

        Dynamic columns stay at the END of the row in both schemas so callers that
        slice ``X[:, n_static:]`` still get the dynamic block by computing
        ``n_static = X.shape[1] - 2 * n_time_steps``.
        """
        if self.noise_type not in ["none", "pushforward"]:
            window_slice = slice(time_step, time_step + n_time_steps)
            water_depth[window_slice, :] = self.apply_noise_to_feature(
                water_depth[window_slice, :], self.noise_type, self.noise_std
            )
            volume[window_slice, :] = self.apply_noise_to_feature(
                volume[window_slice, :], self.noise_type, self.noise_std
            )
        num_nodes = xy_coords.shape[0]
        flow_hydrograph_current_step = np.full(
            (num_nodes, 1), inflow_hydrograph[time_step]
        )
        precip_current_step = np.full((num_nodes, 1), precipitation_data[time_step])
        blocks = [
            xy_coords,           # 0-1
            area,                 # 2
            elevation,            # 3
            slope,                # 4
            aspect,               # 5
            curvature,            # 6
            manning,              # 7
            flow_accum,           # 8
            infiltration,         # 9
            flow_hydrograph_current_step,  # 10
            precip_current_step,           # 11
        ]
        if extra_static is not None:
            blocks.append(extra_static)
        blocks.extend([water_depth.T, volume.T])
        features = np.hstack(blocks)
        future_inflow = inflow_hydrograph[time_step + n_time_steps]
        future_precip = precipitation_data[time_step + n_time_steps]
        return features, future_inflow, future_precip

    # ------------------------------------------------------------------
    # 1D system loaders (only used when self.use_1d == True)
    # ------------------------------------------------------------------

    def load_1d_static_data(self, split_dir, stats):
        """Load static 1D node data and standardise.

        Standardisation policy: shared scales (xy, elevation) reuse the 2D stats
        already in ``stats`` so the model sees a single distribution for those
        columns. 1D-only quantities (depth, base_area) get their own stats keys.

        Returns (xy_1d, depth_1d, invert_elev_1d, surface_elev_1d, base_area_1d,
                 base_area_1d_denorm, stats).
        """
        epsilon = 1e-8

        def standardize_with_key(data, key):
            """Z-score using stats[key]; create stats[key] from this data if absent."""
            if key in stats:
                m = np.array(stats[key]["mean"])
                s = np.array(stats[key]["std"])
            else:
                m = np.mean(data, axis=0)
                s = np.std(data, axis=0)
                stats[key] = {"mean": m.tolist(), "std": s.tolist()}
            return (data - m) / (s + epsilon)

        def standardize_shared(data, shared_key):
            """Z-score using an existing stats[shared_key] entry (created by 2D pass)."""
            m = np.array(stats[shared_key]["mean"])
            s = np.array(stats[shared_key]["std"])
            return (data - m) / (s + epsilon)

        csv_path = os.path.join(split_dir, "1d_nodes_static.csv")
        # node_idx, position_x, position_y, depth, invert_elevation,
        # surface_elevation, base_area
        raw = np.genfromtxt(
            csv_path, delimiter=",", skip_header=1, filling_values=np.nan
        )
        pos_x = raw[:, 1]
        pos_y = raw[:, 2]
        depth = raw[:, 3].reshape(-1, 1)
        invert_elev = raw[:, 4].reshape(-1, 1)
        surface_elev = raw[:, 5].reshape(-1, 1)
        base_area_denorm = raw[:, 6].reshape(-1, 1)

        xy_1d_raw = np.column_stack([pos_x, pos_y])
        xy_1d = standardize_shared(xy_1d_raw, "xy_coords")           # share with 2D
        invert_elev_n = standardize_shared(invert_elev, "elevation")  # share scale
        surface_elev_n = standardize_shared(surface_elev, "elevation")
        depth_n = standardize_with_key(depth, "depth_1d")
        base_area_n = standardize_with_key(base_area_denorm.copy(), "base_area_1d")

        return (
            xy_1d,
            depth_n,
            invert_elev_n,
            surface_elev_n,
            base_area_n,
            base_area_denorm,
            stats,
        )

    def load_1d_edge_data(self, split_dir):
        """Load 1D pipe connectivity and per-pipe static features (forward only)."""
        epsilon = 1e-8
        ei_path = os.path.join(split_dir, "1d_edge_index.csv")
        edge_idx_raw = np.genfromtxt(ei_path, delimiter=",", skip_header=1, dtype=int)
        src_fwd = edge_idx_raw[:, 1]
        dst_fwd = edge_idx_raw[:, 2]

        es_path = os.path.join(split_dir, "1d_edges_static.csv")
        # edge_idx, relative_position_x, relative_position_y, length, diameter,
        # shape, roughness, slope
        edges_raw = np.genfromtxt(es_path, delimiter=",", skip_header=1)
        rel_x = edges_raw[:, 1]
        rel_y = edges_raw[:, 2]
        length = edges_raw[:, 3]
        diameter = edges_raw[:, 4]
        shape = edges_raw[:, 5]
        roughness = edges_raw[:, 6]
        slope = edges_raw[:, 7]

        # Z-score within the 1D pipe set (these features have no 2D analog).
        rel_coords = np.column_stack([rel_x, rel_y])
        rel_coords = (rel_coords - np.mean(rel_coords, axis=0)) / (
            np.std(rel_coords, axis=0) + epsilon
        )
        length_n = (length - np.mean(length)) / (np.std(length) + epsilon)
        diameter_n = (diameter - np.mean(diameter)) / (np.std(diameter) + epsilon)
        # 'shape' is a categorical code; leave as-is (single value usually).
        roughness_n = (roughness - np.mean(roughness)) / (np.std(roughness) + epsilon)
        slope_n = (slope - np.mean(slope)) / (np.std(slope) + epsilon)

        edge_features = np.column_stack(
            [
                rel_coords[:, 0],
                rel_coords[:, 1],
                length_n,
                diameter_n,
                shape,
                roughness_n,
                slope_n,
            ]
        )
        edge_index = np.array([src_fwd, dst_fwd])

        self.num_1d_fwd_edges = len(src_fwd)
        logger.info(f"Loaded {len(src_fwd)} 1D pipe edges (forward).")
        return edge_index, edge_features

    def load_1d2d_connections(self, split_dir):
        """Load 1D<->2D connection edge index.

        Returns:
            edge_index:    (2, E_conn) forward only; row 0 = 2D node ids,
                           row 1 = shifted 1D node ids (local + num_2d_nodes).
            node_1d_local: (E_conn,)   local 1D node id per connection — used
                           at runtime to look up per-connection inlet_flow GT.
        """
        path = os.path.join(split_dir, "1d2d_connections.csv")
        raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=int)
        node_1d_local = raw[:, 1]
        node_2d = raw[:, 2]
        # Convention: forward edge = 2D -> 1D (water entering inlet).
        edge_index = np.array([node_2d, node_1d_local + self.num_2d_nodes])
        logger.info(f"Loaded {edge_index.shape[1]} 1D<->2D connection edges (forward).")
        return edge_index, node_1d_local

    def load_1d_dynamic_data(self, event_dir, num_1d_nodes, base_area_1d_denorm):
        """Load dynamic 1D node data and synthesise per-node volume.

        Volume = water_level * base_area (cylindrical-manhole identity, matches
        what hydraulic solvers like SWMM use internally for junction storage).

        Returns:
            water_level: (T, N_1d) — m
            inlet_flow:  (T, N_1d) — m^3/s
            volume:      (T, N_1d) — m^3 (synthesised)
        """
        path = os.path.join(event_dir, "1d_nodes_dynamic_all.csv")
        # timestep, node_idx, water_level, inlet_flow
        raw = np.genfromtxt(path, delimiter=",", skip_header=1)
        T = len(np.unique(raw[:, 0].astype(int)))
        water_level = raw[:, 2].reshape(T, num_1d_nodes)
        inlet_flow = raw[:, 3].reshape(T, num_1d_nodes)
        # base_area_1d_denorm is (N_1d, 1) — broadcast across T.
        volume = water_level * base_area_1d_denorm.reshape(1, num_1d_nodes)
        return water_level, inlet_flow, volume

    def load_1d_edge_dynamic_data(self, event_dir, num_1d_edges):
        """Load dynamic 1D pipe flow data. Returns (T, num_1d_edges) m^3/s."""
        path = os.path.join(event_dir, "1d_edges_dynamic_all.csv")
        # timestep, edge_idx, flow, velocity
        raw = np.genfromtxt(path, delimiter=",", skip_header=1)
        T = len(np.unique(raw[:, 0].astype(int)))
        return raw[:, 2].reshape(T, num_1d_edges)

    # ------------------------------------------------------------------
    # Edge construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bidirect(edge_index_fwd, edge_features_fwd, negate_first_two=True):
        """Mirror forward edges into reverse edges; optionally negate the first
        two feature columns (relative-position vector flips sign)."""
        src, dst = edge_index_fwd
        edge_index_bi = np.array(
            [np.concatenate([src, dst]), np.concatenate([dst, src])]
        )
        if negate_first_two and edge_features_fwd.shape[1] >= 2:
            rev = edge_features_fwd.copy()
            rev[:, 0] = -rev[:, 0]
            rev[:, 1] = -rev[:, 1]
        else:
            rev = edge_features_fwd
        edge_features_bi = np.vstack([edge_features_fwd, rev])
        return edge_index_bi, edge_features_bi

    @staticmethod
    def _connection_edge_features(edge_index_conn_fwd, xy_2d_norm, xy_1d_norm):
        """Build a 3-col [rel_x, rel_y, length] feature row per forward connection
        edge. Uses standardised xy positions so values share scale with 2D edges.
        Indices in edge_index_conn_fwd: row 0 = 2D node ids, row 1 = shifted 1D ids.
        """
        eps = 1e-8
        n_2d = xy_2d_norm.shape[0]
        e2d = edge_index_conn_fwd[0]
        e1d_shifted = edge_index_conn_fwd[1]
        e1d = e1d_shifted - n_2d
        diff = xy_1d_norm[e1d] - xy_2d_norm[e2d]   # standardised-space displacement
        length = np.sqrt(np.sum(diff ** 2, axis=1))
        length_n = (length - np.mean(length)) / (np.std(length) + eps)
        return np.column_stack([diff[:, 0], diff[:, 1], length_n])
