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
Training script for HydroGraphNet on the UrbanFlood dataset (Phase 1: 2D-only baseline).
Identical to train.py except for dataset import/instantiation and config name.
"""

import os
import time

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric as pyg
import wandb

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from torch_geometric.loader import DataLoader as PyGDataLoader

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.datapipes.gnn.hydrographnet_dataset import UrbanFloodDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from model import HydroGraphKANWithEdgeDecoder
from utils import (
    compute_physics_loss,
    compute_local_conservation_loss,
    compute_edge_flow_loss,
)


# Custom collate function that checks if each item is a tuple (graph, physics_data) or a plain graph.
def collate_fn(batch):
    if isinstance(batch[0], tuple):
        graphs, physics_list = zip(*batch)
        batched_graph = pyg.data.from_data_list(graphs)
        physics_data = {}
        # For each key, build a tensor by stacking the scalar values from each sample.
        for key in physics_list[0].keys():
            physics_data[key] = torch.tensor(
                [d[key] for d in physics_list], dtype=torch.float
            )
        return batched_graph, physics_data
    else:
        return pyg.data.from_data_list(batch)


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        # Ensure distributed manager is initialized.
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = cfg.amp
        self.noise_type = cfg.noise_type

        # Physics loss settings.
        self.use_physics_loss = cfg.get("use_physics_loss", False)
        self.delta_t = cfg.get("delta_t", 300.0)
        self.physics_loss_weight = cfg.get("physics_loss_weight", 0.0)

        # Local conservation loss settings.
        self.use_local_physics_loss = cfg.get("use_local_physics_loss", False)
        self.local_physics_loss_weight = cfg.get("local_physics_loss_weight", 0.1)
        self.edge_loss_weight = cfg.get("edge_loss_weight", 1.0)
        # Bundle knobs ported from White-River v7-R1.
        self.local_loss_warmup_epochs = int(cfg.get("local_loss_warmup_epochs", 0))
        self.local_loss_smooth_l1_beta = float(cfg.get("local_loss_smooth_l1_beta", 0.0))
        self.lc_apply_boundary_mask = bool(cfg.get("compute_boundary_mask", False))
        self.lc_restrict_to_2d = bool(cfg.get("restrict_lc_to_2d", False))
        self.lc_node_type_weighting = str(cfg.get("lmc_node_type_weighting", "none"))
        # Set per-epoch from the main loop. Linear ramp [0, 1] over the
        # warmup window. With warmup_epochs=0 this is always 1.0 (no ramp).
        self.lc_eff_weight = self.local_physics_loss_weight

        # 1D coupling toggle.
        self.use_1d = cfg.get("use_1d", False)
        self.n_time_steps = cfg.n_time_steps
        # The dynamic block in graph.x has 2 * window_size columns (one half
        # water_depth, the other volume). With pushforward noise the dataset
        # supplies one extra timestep so window_size = n_time_steps + 1.
        self.dynamic_window = self.n_time_steps + (
            1 if cfg.noise_type == "pushforward" else 0
        )

        # ---- LMC bundle v5: multi-step rollout training (DUALFloodGNN) -----
        # `rollout_curriculum` is a list of (epoch_start, O) pairs, e.g.
        # [[0, 1], [12, 2], [25, 4], [37, 8]]. The trainer derives the maximum
        # O across the schedule and asks the dataset for that many consecutive
        # GT steps per sample. The current `curriculum_O` is set per-epoch by
        # the main loop and controls how many rollout steps the train loop
        # actually unrolls. When the max O across the schedule is 1, the
        # legacy pushforward branch is used and behavior is unchanged.
        self.rollout_curriculum = list(cfg.get("rollout_curriculum", [[0, 1]]))
        self.train_rollout_length = max(int(o) for _, o in self.rollout_curriculum)
        self.curriculum_O = 1  # set per-epoch by main loop

        # Set activation function.
        mlp_act = "relu"
        if cfg.recompute_activation:
            rank_zero_logger.info(
                "Setting MLP activation to SiLU for recompute_activation."
            )
            mlp_act = "silu"

        rank_zero_logger.info("Initializing UrbanFloodDataset...")
        dataset = UrbanFloodDataset(
            name="urbanflood_dataset",
            data_dir=cfg.data_dir,
            model_name=cfg.model_name,
            split="train",
            n_time_steps=cfg.n_time_steps,
            noise_type=cfg.noise_type,
            noise_std=0.01,
            num_samples=cfg.num_training_samples,
            return_physics=self.use_physics_loss or self.use_local_physics_loss,
            delta_t=self.delta_t,
            use_1d=self.use_1d,
            compute_boundary_mask=self.lc_apply_boundary_mask,
            mask_inlets_in_lc=bool(cfg.get("mask_inlets_in_lc", False)),
            edge_q_prev_as_input=bool(cfg.get("edge_q_prev_as_input", False)),
            lmc_antisymmetric=bool(cfg.get("lmc_antisymmetric", False)),
            train_rollout_length=self.train_rollout_length,
        )
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
        )
        self.dataloader = PyGDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_fn,
        )
        rank_zero_logger.info("Dataset and dataloader initialization complete.")

        model_args = dict(
            input_dim_nodes=cfg.num_input_features,
            input_dim_edges=cfg.num_edge_features,
            output_dim=cfg.num_output_features,
            processor_size=cfg.processor_size,
            mlp_activation_fn=mlp_act,
            num_layers_node_processor=cfg.num_layers_node_processor,
            num_layers_edge_processor=cfg.num_layers_edge_processor,
            hidden_dim_processor=cfg.hidden_dim_processor,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            num_layers_edge_encoder=cfg.num_layers_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
            num_layers_node_decoder=cfg.num_layers_node_decoder,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
            num_harmonics=cfg.get("num_harmonics", 5),
        )
        if self.use_local_physics_loss:
            rank_zero_logger.info(
                "Instantiating HydroGraphKANWithEdgeDecoder model..."
            )
            self.model = HydroGraphKANWithEdgeDecoder(
                **model_args,
                concat_endpoints=bool(
                    cfg.get("edge_decoder_concat_endpoints", False)
                ),
            )
        else:
            rank_zero_logger.info("Instantiating MeshGraphKAN model...")
            self.model = MeshGraphKAN(**model_args)
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphKAN is not yet JIT-compatible.")
            self.model = torch.compile(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        rank_zero_logger.info(
            f"Model instantiated successfully. Total parameters: {total_params:,}"
        )

        if cfg.watch_model and not cfg.jit and self.dist.rank == 0:
            wandb.watch(self.model)

        if self.dist.world_size > 1:
            rank_zero_logger.info("Wrapping model in DistributedDataParallel...")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()
        self.criterion = nn.MSELoss()
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
            else:
                self.optimizer = None
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex is not installed; FusedAdam optimizer will not be used."
            )
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        rank_zero_logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        rank_zero_logger.info("Loading checkpoint if available...")
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        rank_zero_logger.info(
            f"Checkpoint loaded. Starting training from epoch {self.epoch_init}."
        )

    def train(self, batch):
        if self.use_physics_loss or self.use_local_physics_loss:
            graph, physics_data = batch
        else:
            if isinstance(batch, (list, tuple)):
                graph, physics_data = batch
            else:
                graph = batch
                physics_data = None
        graph = graph.to(self.dist.device)
        if physics_data is not None:
            physics_data = {k: v.to(self.dist.device) for k, v in physics_data.items()}
        self.optimizer.zero_grad()
        loss, loss_dict = self.forward(graph, physics_data)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_dict

    def _call_model(self, *args, **kwargs):
        """Call model and unpack tuple return when edge decoder is active."""
        out = self.model(*args, **kwargs)
        if self.use_local_physics_loss:
            return out  # (node_pred, edge_pred)
        return out, None  # node_pred, None

    def _forward_multistep(self, graph, physics_data):
        """Multi-step autoregressive rollout training (DUALFloodGNN regime).

        Unrolls `self.curriculum_O` steps with full backprop through the
        rollout. At each step, the per-step prediction loss, edge loss
        (when LMC is on), and local mass conservation loss are computed
        against the per-step GT targets stacked into the sample. State
        (water_depth/volume window + edge_q_prev) is updated
        autoregressively from the model's own predictions, mirroring the
        inference rollout in inference.py:328-368.
        """
        with autocast(device_type=self.dist.device.type, enabled=self.amp):
            X = graph.x
            n_time = self.dynamic_window
            n_static = X.shape[1] - 2 * n_time
            static_part = X[:, :n_static]
            water_depth_full = X[:, n_static : n_static + n_time]
            volume_full = X[:, n_static + n_time : n_static + 2 * n_time]

            # Initial input window = last n_time_steps cols (matches the
            # one_step branch of the legacy pushforward path). For
            # n_time_steps=2 + pushforward, water_depth_full is 3 cols and
            # this takes cols 1..2 (drop the leading pushforward column).
            water_depth_window = water_depth_full[:, 1:]
            volume_window = volume_full[:, 1:]
            # Initial absolute Q (normalised, per-edge) from GT — only
            # available when LMC / edge supervision is on; the dataset only
            # populates `g.edge_q_prev` under `return_physics=True`.
            edge_q_running = (
                graph.edge_q_prev if self.use_local_physics_loss else None
            )

            O = max(1, int(self.curriculum_O))
            assert O <= self.train_rollout_length, (
                f"curriculum_O={O} exceeds dataset train_rollout_length="
                f"{self.train_rollout_length}"
            )

            # Save originals so we can restore the graph for downstream calls
            # (PyG mutates in-place; this is for safety).
            orig_source_term = getattr(graph, "source_term", None)
            orig_edge_q_prev = getattr(graph, "edge_q_prev", None)

            loss_total = torch.zeros((), device=self.dist.device, dtype=X.dtype)
            comp = {
                "pred_loss": 0.0,
                "edge_loss": 0.0,
                "local_physics_loss": 0.0,
            }

            for o in range(O):
                # Wire per-step state into the graph for LMC residual
                # (only meaningful when LMC is on).
                if self.use_local_physics_loss:
                    graph.edge_q_prev = edge_q_running
                    if hasattr(graph, "source_term_rollout"):
                        graph.source_term = graph.source_term_rollout[:, o]

                X_o = torch.cat(
                    [static_part, water_depth_window, volume_window], dim=1
                )
                node_pred, edge_pred = self._call_model(
                    X_o, graph.edge_attr, graph
                )

                # Per-step targets (slice from rollout stacks).
                y_o = graph.y_rollout[:, o, :]
                pred_loss = self.criterion(node_pred, y_o)
                step_loss = pred_loss
                comp["pred_loss"] += pred_loss.detach().item()

                if self.use_local_physics_loss and edge_pred is not None:
                    edge_y_o = graph.edge_y_rollout[:, o, :]
                    edge_loss_o = compute_edge_flow_loss(
                        edge_pred, graph, edge_y_target=edge_y_o
                    )
                    lmc_loss_o = compute_local_conservation_loss(
                        node_pred, edge_pred, graph, physics_data,
                        delta_t=self.delta_t,
                        smooth_l1_beta=self.local_loss_smooth_l1_beta,
                        apply_boundary_mask=self.lc_apply_boundary_mask,
                        restrict_to_2d=self.lc_restrict_to_2d,
                        node_type_weighting=self.lc_node_type_weighting,
                    )
                    step_loss = (
                        step_loss
                        + self.edge_loss_weight * edge_loss_o
                        + self.lc_eff_weight * lmc_loss_o
                    )
                    comp["edge_loss"] += edge_loss_o.detach().item()
                    comp["local_physics_loss"] += lmc_loss_o.detach().item()

                loss_total = loss_total + step_loss

                # Autoregressive state update (unless this was the last step).
                if o < O - 1:
                    new_wd = water_depth_window[:, -1:] + node_pred[:, 0:1]
                    new_vol = volume_window[:, -1:] + node_pred[:, 1:2]
                    water_depth_window = torch.cat(
                        [water_depth_window[:, 1:], new_wd], dim=1
                    )
                    volume_window = torch.cat(
                        [volume_window[:, 1:], new_vol], dim=1
                    )
                    if edge_pred is not None:
                        edge_q_running = edge_q_running + edge_pred.squeeze(-1)

            # DUALFloodGNN Eq. 25: average loss over the rollout horizon.
            loss_total = loss_total / O

            # Restore graph attrs (defensive — next batch creates a fresh PyG batch).
            if orig_edge_q_prev is not None:
                graph.edge_q_prev = orig_edge_q_prev
            if orig_source_term is not None:
                graph.source_term = orig_source_term

            loss_dict = {
                "total_loss": loss_total,
                "pred_loss": torch.tensor(comp["pred_loss"] / O, device=self.dist.device),
            }
            if self.use_local_physics_loss:
                loss_dict["edge_loss"] = torch.tensor(
                    comp["edge_loss"] / O, device=self.dist.device
                )
                loss_dict["local_physics_loss"] = torch.tensor(
                    comp["local_physics_loss"] / O, device=self.dist.device
                )
        return loss_total, loss_dict

    def forward(self, graph, physics_data):
        # Multi-step rollout training (LMC bundle v5 / DUALFloodGNN regime).
        # When the curriculum schedule has any O > 1, the dataset provides
        # stacked GT targets for O consecutive future steps and the trainer
        # unrolls `self.curriculum_O` autoregressive steps with full backprop
        # through the rollout. The legacy pushforward path is preserved
        # untouched for runs whose schedule is `[[0, 1]]`.
        if self.train_rollout_length > 1:
            return self._forward_multistep(graph, physics_data)
        if self.noise_type == "pushforward":
            with autocast(device_type=self.dist.device.type, enabled=self.amp):
                X = graph.x
                # Dynamic block (water_depth window + volume window) is at the
                # END of every row in both 2D-only and use_1d schemas. Width of
                # each half = n_time_steps (+1 under pushforward). Derive
                # n_static from total - 2 * window so it works for both schemas.
                n_time = self.dynamic_window
                n_static = X.shape[1] - 2 * n_time
                static_part = X[:, :n_static]
                water_depth_full = X[:, n_static : n_static + n_time]
                volume_full = X[:, n_static + n_time : n_static + 2 * n_time]
                # For one-step prediction, use dynamic features from indices 1: (last n_time_steps)
                water_depth_window_one = water_depth_full[:, 1:]
                volume_window_one = volume_full[:, 1:]
                X_one = torch.cat(
                    [static_part, water_depth_window_one, volume_window_one], dim=1
                )
                pred_one, edge_pred_one = self._call_model(
                    X_one, graph.edge_attr, graph
                )
                one_step_loss = self.criterion(pred_one, graph.y)

                # Stability branch (example implementation)
                water_depth_window_stab = water_depth_full[:, : n_time - 1]
                volume_window_stab = volume_full[:, : n_time - 1]
                X_stab = torch.cat(
                    [static_part, water_depth_window_stab, volume_window_stab], dim=1
                )
                pred_stab, _ = self._call_model(X_stab, graph.edge_attr, graph)
                pred_stab_detached = pred_stab.detach()
                water_depth_updated = torch.cat(
                    [
                        water_depth_full[:, 1:2],
                        water_depth_full[:, 1:2] + pred_stab_detached[:, 0:1],
                    ],
                    dim=1,
                )
                volume_updated = torch.cat(
                    [
                        volume_full[:, 1:2],
                        volume_full[:, 1:2] + pred_stab_detached[:, 1:2],
                    ],
                    dim=1,
                )
                X_stab_updated = torch.cat(
                    [static_part, water_depth_updated, volume_updated], dim=1
                )
                pred_stab2, edge_pred_stab2 = self._call_model(
                    X_stab_updated, graph.edge_attr, graph
                )
                stability_loss = self.criterion(pred_stab2, graph.y)

                loss = one_step_loss + stability_loss
                loss_dict = {
                    "total_loss": loss,
                    "loss_one": one_step_loss,
                    "loss_stability": stability_loss,
                }
                if self.use_physics_loss and physics_data is not None:
                    phy_loss = compute_physics_loss(
                        pred_one, physics_data, graph, delta_t=self.delta_t
                    )
                    loss = loss + self.physics_loss_weight * phy_loss
                    loss_dict["physics_loss"] = phy_loss
                if self.use_local_physics_loss and physics_data is not None:
                    # Edge flow supervision (ℒ_edge).
                    edge_loss_one = compute_edge_flow_loss(edge_pred_one, graph)
                    edge_loss_stab = compute_edge_flow_loss(edge_pred_stab2, graph)
                    edge_loss = edge_loss_one + edge_loss_stab
                    loss = loss + self.edge_loss_weight * edge_loss
                    loss_dict["edge_loss"] = edge_loss
                    # Local conservation (ℒ_local) with bundle knobs.
                    local_loss_one = compute_local_conservation_loss(
                        pred_one, edge_pred_one, graph, physics_data,
                        delta_t=self.delta_t,
                        smooth_l1_beta=self.local_loss_smooth_l1_beta,
                        apply_boundary_mask=self.lc_apply_boundary_mask,
                        restrict_to_2d=self.lc_restrict_to_2d,
                        node_type_weighting=self.lc_node_type_weighting,
                    )
                    local_loss_stab = compute_local_conservation_loss(
                        pred_stab2, edge_pred_stab2, graph, physics_data,
                        delta_t=self.delta_t,
                        smooth_l1_beta=self.local_loss_smooth_l1_beta,
                        apply_boundary_mask=self.lc_apply_boundary_mask,
                        restrict_to_2d=self.lc_restrict_to_2d,
                        node_type_weighting=self.lc_node_type_weighting,
                    )
                    local_loss = local_loss_one + local_loss_stab
                    loss = loss + self.lc_eff_weight * local_loss
                    loss_dict["local_physics_loss"] = local_loss
            return loss, loss_dict
        else:
            with autocast(device_type=self.dist.device.type, enabled=self.amp):
                pred, edge_pred = self._call_model(
                    graph.x, graph.edge_attr, graph
                )
                mse_loss = self.criterion(pred, graph.y)
                loss = mse_loss
                loss_dict = {"total_loss": loss, "mse_loss": mse_loss}
                if self.use_physics_loss and physics_data is not None:
                    phy_loss = compute_physics_loss(
                        pred, physics_data, graph, delta_t=self.delta_t
                    )
                    loss = loss + self.physics_loss_weight * phy_loss
                    loss_dict["physics_loss"] = phy_loss
                if self.use_local_physics_loss and physics_data is not None:
                    edge_loss = compute_edge_flow_loss(edge_pred, graph)
                    loss = loss + self.edge_loss_weight * edge_loss
                    loss_dict["edge_loss"] = edge_loss
                    local_loss = compute_local_conservation_loss(
                        pred, edge_pred, graph, physics_data,
                        delta_t=self.delta_t,
                        smooth_l1_beta=self.local_loss_smooth_l1_beta,
                        apply_boundary_mask=self.lc_apply_boundary_mask,
                        restrict_to_2d=self.lc_restrict_to_2d,
                        node_type_weighting=self.lc_node_type_weighting,
                    )
                    loss = loss + self.lc_eff_weight * local_loss
                    loss_dict["local_physics_loss"] = local_loss
            return loss, loss_dict

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    initialize_wandb(
        project="UrbanFlood-HydroGraphNet",
        entity="Modulus",
        name="UrbanFlood-Phase1-Training",
        group="UrbanFlood-DDP-Group",
        mode=cfg.wandb_mode,
    )
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()
    rank_zero_logger.info(f"Starting training process with configuration: {cfg}")
    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Beginning training loop...")
    start_time = time.time()

    # Track loss history for plotting.
    loss_history = {"total_loss": []}
    component_keys = [
        "loss_one", "loss_stability", "mse_loss", "pred_loss",
        "physics_loss", "edge_loss", "local_physics_loss",
    ]
    for key in component_keys:
        loss_history[key] = []

    def _curriculum_O_for_epoch(epoch_idx: int) -> int:
        """Pick the largest O whose epoch_start <= epoch_idx in the schedule."""
        chosen = 1
        for start, O_val in trainer.rollout_curriculum:
            if int(start) <= epoch_idx:
                chosen = int(O_val)
        return chosen

    for epoch in range(trainer.epoch_init, cfg.epochs):
        # Linear warmup for the local-conservation weight. With
        # warmup_epochs=0 the ramp factor stays at 1.0 (no ramp).
        if trainer.local_loss_warmup_epochs > 0:
            ramp = min(epoch / trainer.local_loss_warmup_epochs, 1.0)
        else:
            ramp = 1.0
        trainer.lc_eff_weight = ramp * trainer.local_physics_loss_weight
        # Pick the curriculum rollout horizon for this epoch (LMC bundle v5).
        trainer.curriculum_O = _curriculum_O_for_epoch(epoch)
        rank_zero_logger.info(
            f"Epoch {epoch}: lc_eff_weight={trainer.lc_eff_weight:.4e} "
            f"(ramp={ramp:.2f}, λ={trainer.local_physics_loss_weight}) | "
            f"curriculum_O={trainer.curriculum_O}"
        )

        epoch_loss = 0.0
        epoch_components = {k: 0.0 for k in component_keys}
        num_batches = 0
        for batch in trainer.dataloader:
            loss, loss_dict = trainer.train(batch)
            epoch_loss += loss.detach().item()
            for k in component_keys:
                if k in loss_dict:
                    epoch_components[k] += loss_dict[k].detach().item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        loss_history["total_loss"].append(avg_loss)

        # Build log message with component breakdown.
        log_parts = [f"Epoch {epoch} — Avg Loss: {avg_loss:.4e}"]
        for k in component_keys:
            if epoch_components[k] > 0:
                avg_comp = epoch_components[k] / num_batches
                loss_history[k].append(avg_comp)
                log_parts.append(f"{k}: {avg_comp:.4e}")
            else:
                loss_history[k].append(None)
        rank_zero_logger.info(" | ".join(log_parts))

        # WandB logging (include new loss components).
        wandb_dict = {"epoch": epoch}
        for k in ["total_loss"] + component_keys:
            val = loss_dict.get(k, None)
            if val is not None and isinstance(val, torch.Tensor):
                wandb_dict[k] = val.detach().cpu()
        wandb.log(wandb_dict)

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Checkpoint saved at epoch {epoch}.")

        elapsed = time.time() - start_time
        rank_zero_logger.info(f"Epoch {epoch} duration: {elapsed:.2f} seconds.")
        start_time = time.time()

    # Plot loss curves.
    if dist.rank == 0 and len(loss_history["total_loss"]) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = list(range(trainer.epoch_init, trainer.epoch_init + len(loss_history["total_loss"])))
        ax.plot(epochs_range, loss_history["total_loss"], label="total_loss", linewidth=2)
        for k in component_keys:
            vals = loss_history[k]
            filtered = [(e, v) for e, v in zip(epochs_range, vals) if v is not None]
            if filtered:
                ax.plot(*zip(*filtered), label=k, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title("Training Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(to_absolute_path(cfg.ckpt_path), "loss_curve.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rank_zero_logger.info(f"Loss curve saved to {plot_path}")

    rank_zero_logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
