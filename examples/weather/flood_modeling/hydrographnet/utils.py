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
Utility functions for physics-based loss computation and custom loss definitions.
"""

import torch
import torch.nn.functional as F


def compute_physics_loss(pred, physics_data, graph, delta_t=1200.0):
    """
    Compute a physics-based continuity loss in the denormalized domain.

    For each graph sample, the predicted total volume is computed as:
        predicted_total_volume = past_volume_denorm + volume_std * (sum of predicted volume differences)
    where:
        past_volume_denorm = past_volume_norm * volume_std + (num_nodes * volume_mean)

    Future volume is denormalized similarly:
        future_volume_denorm = future_volume_norm * volume_std + (num_nodes * volume_mean)

    Two continuity terms are computed:
        - term1: Uses average inflow and precipitation (denorm_avg_inflow and denorm_avg_precip)
        - term2: Uses next step's inflow and precipitation (denorm_next_inflow and denorm_next_precip)

    An effective precipitation term is computed as:
        new_precip_term = base_precip * infiltration_area_sum

    Finally, the physics loss is the mean of the sum of term1 and term2 across all graph samples.

    Args:
        pred (torch.Tensor): Model predictions (expected volume difference).
        physics_data (dict): Dictionary containing various denormalized physics parameters.
        graph (PyGData): Batched PyG graph.
        delta_t (float): Time delta over which the continuity is enforced.

    Returns:
        torch.Tensor: Mean physics loss across all graph samples.
    """
    unique_ids = torch.unique(graph.batch)
    predicted_diff = pred[:, 1]  # Predicted volume difference (normalized)
    physics_losses = []

    for uid in unique_ids:
        mask = graph.batch == uid
        pred_diff_sum = predicted_diff[mask].sum()

        idx = (unique_ids == uid).nonzero(as_tuple=False).item()
        past_volume_norm = physics_data["past_volume"][idx]
        future_volume_norm = physics_data["future_volume"][idx]
        # For term1: use average inflow and precipitation
        denorm_avg_inflow = physics_data["avg_inflow"][idx]
        denorm_avg_precip = physics_data["avg_precipitation"][idx]
        # For term2: use next step inflow and precipitation
        denorm_next_inflow = physics_data["next_inflow"][idx]
        denorm_next_precip = physics_data["next_precip"][idx]

        volume_mean = physics_data["volume_mean"][idx]
        volume_std = physics_data["volume_std"][idx]
        num_nodes = physics_data["num_nodes"][idx]
        area_sum = physics_data["area_sum"][idx]
        infiltration_area_sum = physics_data["infiltration_area_sum"][idx]

        # Denormalize past and future volumes.
        past_volume_denorm = past_volume_norm * volume_std + num_nodes * volume_mean
        future_volume_denorm = future_volume_norm * volume_std + num_nodes * volume_mean

        # Compute the predicted total volume.
        pred_total_volume = past_volume_denorm + volume_std * pred_diff_sum

        # Compute effective precipitation terms.
        new_precip_term = denorm_avg_precip * infiltration_area_sum
        new_next_precip_term = denorm_next_precip * infiltration_area_sum

        temp1 = pred_total_volume - (
            past_volume_denorm + delta_t * (denorm_avg_inflow + new_precip_term)
        )

        temp2 = (
            future_volume_denorm
            - pred_total_volume
            - delta_t * (denorm_next_inflow + new_next_precip_term)
        )

        # Compute continuity terms using ReLU to enforce non-negativity.
        term1 = (
            F.relu(
                (
                    pred_total_volume
                    - (
                        past_volume_denorm
                        + delta_t * (denorm_avg_inflow + new_precip_term)
                    )
                )
                / area_sum
            )
            ** 2
        )
        term2 = (
            F.relu(
                (
                    future_volume_denorm
                    - pred_total_volume
                    - delta_t * (denorm_next_inflow + new_next_precip_term)
                )
                / area_sum
            )
            ** 2
        )

        physics_losses.append(term1 + term2)

    if physics_losses:
        return torch.stack(physics_losses).mean()
    else:
        return torch.tensor(0.0, device=pred.device)


def compute_local_conservation_loss(
    node_pred, edge_pred, graph, physics_data, delta_t=1200.0, boundary_mask=None,
    use_huber_residual=False, huber_beta=0.5,
):
    """
    Compute per-node local mass conservation loss (DualFloodGNN Eq. 18).

    All terms are in normalized volume space. The edge decoder outputs Q in
    normalized volume units per timestep (no explicit delta_t/volume_std scaling).

    L_local = mean_i |delta_V_i - (net_flow_i + R_i)|

    Args:
        node_pred (torch.Tensor): Node predictions [N, 2] (delta_depth, delta_volume).
        edge_pred (torch.Tensor): Edge predictions [E, 1] (per-edge flow in normalized volume).
        graph (PyGData): Batched PyG graph with edge_index and node features.
        physics_data (dict): Physics parameters including normalization stats.
        delta_t (float): Time step in seconds.
        boundary_mask (torch.Tensor, optional): Boolean mask [N], True for boundary nodes
            to exclude from the conservation loss.
        use_huber_residual (bool): If True, use smooth L1 (Huber) loss instead of L1
            on the conservation residual. Huber gives L2 treatment to small residuals
            (strong gradient) and L1 to large residuals (bounded gradient).
        huber_beta (float): Beta parameter for smooth_l1_loss (transition point).

    Returns:
        torch.Tensor: Scalar local conservation loss.
    """
    device = node_pred.device
    num_nodes = node_pred.shape[0]

    volume_std = physics_data["volume_std"][0]

    # delta_V in normalized space.
    delta_V = node_pred[:, 1]

    # Rainfall contribution per node in normalized volume units.
    precip_std = physics_data["precip_std"][0]
    precip_mean = physics_data["precip_mean"][0]
    area_std = physics_data["area_std"][0]
    area_mean = physics_data["area_mean"][0]

    precip_norm = graph.x[:, 11]  # precipitation at feature index 11
    precip_real = precip_norm * precip_std + precip_mean
    area_per_node = graph.x[:, 2].squeeze() * area_std + area_mean  # denormalized per-node area

    # Use infiltration-weighted area (matching global physics loss semantics).
    infiltration_std = physics_data["infiltration_std"][0]
    infiltration_mean = physics_data["infiltration_mean"][0]
    infiltration_denorm = graph.x[:, 9].squeeze() * infiltration_std + infiltration_mean
    effective_area = infiltration_denorm * area_per_node / 100.0  # /100 matches global loss

    R = (precip_real * effective_area * delta_t) / volume_std

    # Net flow per node from bidirectional edge predictions.
    edge_index = graph.edge_index  # [2, E]
    Q = edge_pred.squeeze(-1)  # [E], normalized volume units per timestep

    # Q_in[i] = sum of Q on edges pointing TO node i
    Q_in = torch.zeros(num_nodes, device=device)
    Q_in.scatter_add_(0, edge_index[1], Q)
    # Q_out[i] = sum of Q on edges pointing FROM node i
    Q_out = torch.zeros(num_nodes, device=device)
    Q_out.scatter_add_(0, edge_index[0], Q)

    net_flow = Q_in - Q_out  # edge decoder learns in normalized volume units

    # Per-node residual: delta_V should equal net flow + rainfall.
    residual = delta_V - (net_flow + R)
    if boundary_mask is not None:
        # Exclude boundary nodes (where boundary_mask is True).
        interior = ~boundary_mask
        if interior.any():
            residual = residual[interior]

    if use_huber_residual:
        return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=huber_beta)
    return residual.abs().mean()


def compute_edge_regularization_loss(edge_pred, graph, use_smoothness=True):
    """Enforce antisymmetry and spatial smoothness on edge flows.

    On a bidirectional graph, each connection (i,j) has two edges with
    independent Q values. Antisymmetry enforces Q(i->j) ≈ -Q(j->i),
    reducing effective degrees of freedom to 1 per connection (matching
    the paper's single undirected edge). Smoothness encourages spatially
    coherent flow fields.

    Args:
        edge_pred (torch.Tensor): Edge predictions [E, 1].
        graph (PyGData): Batched PyG graph with edge_index.

    Returns:
        torch.Tensor: Scalar regularization loss.
    """
    Q = edge_pred.squeeze(-1)  # [E]
    edge_index = graph.edge_index
    num_nodes = graph.x.shape[0]
    E = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]

    # (a) Antisymmetry: Q(i->j) should equal -Q(j->i).
    # Build reverse edge mapping via hashing.
    edge_hash = src * num_nodes + dst
    rev_hash = dst * num_nodes + src
    sorted_hash, sort_idx = edge_hash.sort()
    rev_positions = torch.searchsorted(sorted_hash, rev_hash)
    rev_positions = rev_positions.clamp(max=E - 1)
    valid = sorted_hash[rev_positions] == rev_hash
    rev_edge_idx = sort_idx[rev_positions]
    antisym_loss = (Q + Q[rev_edge_idx])[valid].pow(2).mean()

    if not use_smoothness:
        return antisym_loss

    # (b) Smoothness: Q on edges from the same source node should be similar.
    Q_mean_per_node = torch.zeros(num_nodes, device=Q.device)
    Q_count = torch.zeros(num_nodes, device=Q.device)
    Q_mean_per_node.scatter_add_(0, src, Q)
    Q_count.scatter_add_(0, src, torch.ones_like(Q))
    Q_count = Q_count.clamp(min=1)
    Q_mean_per_node = Q_mean_per_node / Q_count
    node_means_expanded = Q_mean_per_node[src]
    smooth_loss = ((Q - node_means_expanded) ** 2).mean()

    return 0.5 * antisym_loss + 0.5 * smooth_loss


def custom_loss(pred, targets):
    """
    Compute a custom loss as the sum of MSE losses on water depth and volume predictions.

    Args:
        pred (torch.Tensor): Model predictions with two columns (depth and volume difference).
        targets (torch.Tensor): Ground truth targets.

    Returns:
        dict: Dictionary containing the total loss and individual losses for depth and volume.
    """
    pred_depth = pred[:, 0]
    pred_volume = pred[:, 1]
    target_depth = targets[:, 0]
    target_volume = targets[:, 1]
    loss_depth = F.mse_loss(pred_depth, target_depth, reduction="mean")
    loss_volume = F.mse_loss(pred_volume, target_volume, reduction="mean")
    total_loss = loss_depth + loss_volume
    return {
        "total_loss": total_loss,
        "loss_depth": loss_depth,
        "loss_volume": loss_volume,
    }
