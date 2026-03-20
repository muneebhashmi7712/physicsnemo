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

        self.static_data = {}
        self.dynamic_data = []
        self.sample_index = []
        self.event_ids = []
        self.static_stats = {}
        self.dynamic_stats = {}

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

        for event_id in self.event_ids:
            event_dir = os.path.join(self.split_dir, event_id)
            water_level, inflow, volume, rainfall = self.load_dynamic_data(
                event_dir, num_nodes
            )
            temp_dynamic_data.append(
                {
                    "water_depth": water_level,
                    "inflow_hydrograph": inflow,
                    "volume": volume,
                    "precipitation": rainfall,
                    "event_id": event_id,
                }
            )
            water_depth_list.append(water_level.flatten())
            volume_list.append(volume.flatten())
            precipitation_list.append(rainfall.flatten())
            inflow_list.append(inflow.flatten())

        # --- Compute or load dynamic normalization stats ---
        if self.split == "train":
            self.dynamic_stats = {}
            water_depth_all = np.concatenate(water_depth_list)
            self.dynamic_stats["water_depth"] = {
                "mean": float(np.mean(water_depth_all)),
                "std": float(np.std(water_depth_all)),
            }
            volume_all = np.concatenate(volume_list)
            self.dynamic_stats["volume"] = {
                "mean": float(np.mean(volume_all)),
                "std": float(np.std(volume_all)),
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
            self.save_norm_stats(self.dynamic_stats, DYNAMIC_NORM_STATS_FILE)
        else:
            self.dynamic_stats = self.load_norm_stats(DYNAMIC_NORM_STATS_FILE)

        # --- Normalize dynamic data ---
        self.dynamic_data = []
        for dyn in temp_dynamic_data:
            dyn_std = {
                "water_depth": self.normalize(
                    dyn["water_depth"],
                    self.dynamic_stats["water_depth"]["mean"],
                    self.dynamic_stats["water_depth"]["std"],
                ),
                "volume": self.normalize(
                    dyn["volume"],
                    self.dynamic_stats["volume"]["mean"],
                    self.dynamic_stats["volume"]["std"],
                ),
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
                "event_id": dyn["event_id"],
            }
            self.dynamic_data.append(dyn_std)

        # --- Build sample indices ---
        if self.split == "train":
            for h_idx, dyn in enumerate(self.dynamic_data):
                T = dyn["water_depth"].shape[0]
                if self.noise_type == "pushforward":
                    max_t = T - self.n_time_steps - 1
                else:
                    max_t = T - self.n_time_steps
                for t in range(max_t):
                    self.sample_index.append((h_idx, t))
            self.length = len(self.sample_index)
            logger.info(f"Training samples: {self.length}")
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

    def __getitem__(self, idx: int):
        """Retrieve a graph sample."""
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

            node_features, future_flow, future_precip = self.create_node_features(
                sd["xy_coords"],
                sd["area"],
                sd["elevation"],
                sd["slope"],
                sd["aspect"],
                sd["curvature"],
                sd["manning"],
                sd["flow_accum"],
                sd["infiltration"],
                dyn["water_depth"][t_idx:end_index, :],
                dyn["volume"][t_idx:end_index, :],
                dyn["precipitation"],
                t_idx,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
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
            node_features, _, _ = self.create_node_features(
                sd["xy_coords"],
                sd["area"],
                sd["elevation"],
                sd["slope"],
                sd["aspect"],
                sd["curvature"],
                sd["manning"],
                sd["flow_accum"],
                sd["infiltration"],
                dyn["water_depth"][0 : self.n_time_steps, :],
                dyn["volume"][0 : self.n_time_steps, :],
                dyn["precipitation"],
                0,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
            )
            src, dst = sd["edge_index"]
            edges = torch.stack(
                [torch.tensor(src), torch.tensor(dst)], dim=0
            ).long()
            g = pyg.data.Data(edge_index=edges)
            g.edge_attr = torch.tensor(sd["edge_features"], dtype=torch.float)
            g.x = torch.tensor(node_features, dtype=torch.float)
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

        logger.info(
            f"Loaded {len(src_fwd)} edges, made bidirectional: {edge_index.shape[1]} edges."
        )
        return edge_index, edge_features

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
    ):
        """Create 16-dim node features. Identical logic to HydroGraphDataset."""
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
        features = np.hstack(
            [
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
                water_depth.T,        # 12-(12+n_time_steps-1)
                volume.T,            # (12+n_time_steps)-(12+2*n_time_steps-1)
            ]
        )
        future_inflow = inflow_hydrograph[time_step + n_time_steps]
        future_precip = precipitation_data[time_step + n_time_steps]
        return features, future_inflow, future_precip
