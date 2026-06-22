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


def compute_edge_flow_loss(edge_pred, graph, edge_y_target=None):
    """
    Compute supervised edge flow loss (DualFloodGNN ℒ_edge).

    MSE between predicted delta-Q and ground truth delta-Q. Both are in
    normalized volume-per-step units. Pass `edge_y_target` to override the
    GT tensor (used by the multi-step rollout trainer to feed per-step
    targets sliced from `graph.edge_y_rollout`).

    Args:
        edge_pred (torch.Tensor): Edge predictions [E, 1] (delta-Q in normalized vol/step).
        graph (PyGData): Graph with edge_y attribute containing ground truth delta-Q [E, 1].
        edge_y_target (torch.Tensor, optional): Override GT tensor [E, 1]. If None,
            uses ``graph.edge_y``.

    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    target = graph.edge_y if edge_y_target is None else edge_y_target
    return F.mse_loss(edge_pred, target)


def compute_local_conservation_loss(
    node_pred, edge_pred, graph, physics_data, delta_t=300.0,
    smooth_l1_beta: float = 0.0,
    apply_boundary_mask: bool = False,
    restrict_to_2d: bool = False,
    node_type_weighting: str = "none",
):
    """
    Compute per-node local mass conservation loss (adapted from DualFloodGNN Eq. 18).

    Reconstructs absolute flow Q = Q_prev + delta_Q_pred, denormalises both
    predicted volume changes and edge flows to physical units (m³ per step),
    forms the residual ΔV − (net_flow + S_gt) in physical units, and divides
    by the per-node volume std so each node-type contributes a comparable
    magnitude regardless of its absolute volume scale.

    v2 (1D-coupling): when the dataset stores per-type stats, the graph
    carries `V_std_per_node` and `Q_sigma_per_edge_phys` (both in m³); for
    the legacy 2D-only path these tensors collapse to a single shared std
    so the loss reduces to the v1/v5 formulation bit-identically.

    Bundle knobs (off by default → identical to v3 behaviour):
      * smooth_l1_beta > 0 → use smooth_l1 with that beta instead of plain L1.
      * apply_boundary_mask → if graph.boundary_mask is present, drop those
        nodes from the residual mean (convex hull, optionally OR-ed with
        the 2D-side inlet nodes when the dataset was built with mask_inlets).
      * restrict_to_2d → if graph.node_type is present, drop all 1D nodes
        from the residual mean (H2 ablation).

    L_local = mean_i | (ΔV_phys_i − net_flow_phys_i − S_phys_i) / V_std_per_node_i |

    Args:
        node_pred (torch.Tensor): Node predictions [N, 2] (delta_depth, delta_volume).
        edge_pred (torch.Tensor): Edge predictions [E, 1] (delta-Q normalised
            per edge by `Q_sigma_per_edge_phys`).
        graph (PyGData): Batched PyG graph with edge_index, edge_q_prev,
            source_term, V_std_per_node, and Q_sigma_per_edge_phys attributes.
            Optionally carries `boundary_mask` (bool [N]) and `node_type`
            (long [N], 0=2D, 1=1D).
        physics_data (dict): Physics parameters (unused, kept for API compatibility).
        delta_t (float): Time step in seconds (unused, kept for API compatibility).
        smooth_l1_beta (float): If > 0, apply smooth_l1 with this beta
            instead of the plain L1 absolute value.
        apply_boundary_mask (bool): Drop boundary-masked nodes from the mean.
        restrict_to_2d (bool): Drop 1D nodes from the mean.

    Returns:
        torch.Tensor: Scalar local conservation loss.
    """
    device = node_pred.device
    num_nodes = node_pred.shape[0]

    V_std_per_node     = graph.V_std_per_node.to(device)              # [N], m³
    Q_sigma_per_edge   = graph.Q_sigma_per_edge_phys.to(device)       # [E], m³

    # ΔV in physical m³.
    delta_V_phys = node_pred[:, 1] * V_std_per_node

    # Q in physical m³/step. edge_q_prev / edge_pred are normalised by the
    # per-edge sigma which already includes delta_t.
    #
    # Use model-predicted ΔQ for 2D and 1D edges, but substitute GT ΔQ
    # (graph.edge_y) for connection edges. Connection edges are identified
    # by the 3-col edge-type one-hot in edge_attr cols [7, 8, 9] when
    # use_1d=true (built in hydrographnet_dataset.py:227-233): col 9 is
    # the connection-edge indicator. Treating connection flow as a known
    # boundary condition removes connection-flow prediction error from
    # the LMC residual at 2D inlet nodes — the contamination source
    # identified in urbanflood_lc_bundle_v3.md.
    #
    # The shape[1] >= 10 guard is the use_1d=true signature (10 = 7 base
    # + 3-col one-hot). It also keeps the legacy 2D-only path (3-col
    # edge_attr) bit-identical to v5. Column index 9 is stable whether
    # or not edge_q_prev_as_input has appended an 11th column.
    delta_Q = edge_pred.squeeze(-1).clone()
    if (
        hasattr(graph, "edge_attr")
        and graph.edge_attr is not None
        and graph.edge_attr.shape[1] >= 10
        and hasattr(graph, "edge_y")
        and graph.edge_y is not None
    ):
        conn_mask = graph.edge_attr[:, 9].to(device).bool()
        if conn_mask.any():
            delta_Q[conn_mask] = graph.edge_y.squeeze(-1).to(device)[conn_mask]

    Q_pred_phys = (graph.edge_q_prev.to(device) + delta_Q) * Q_sigma_per_edge

    edge_index = graph.edge_index  # [2, E]
    Q_in_phys  = torch.zeros(num_nodes, device=device, dtype=Q_pred_phys.dtype)
    Q_out_phys = torch.zeros(num_nodes, device=device, dtype=Q_pred_phys.dtype)

    if hasattr(graph, "is_forward_edge") and graph.is_forward_edge is not None:
        # DUALFloodGNN-style architectural antisymmetry (Exp 3/4 of
        # urbanflood_lc_bundle_v3 §7.4). Predict the model's flow only on
        # forward edges and scatter +Q to dst, -Q to src — physically
        # correct without averaging over an independent reverse prediction.
        # Removes the antisymmetry-violation noise floor present in the
        # bidirectional + /2.0 path.
        fwd = graph.is_forward_edge.to(device).bool()
        Q_fwd     = Q_pred_phys[fwd]
        src_fwd   = edge_index[0][fwd]
        dst_fwd   = edge_index[1][fwd]
        Q_in_phys.scatter_add_(0, dst_fwd, Q_fwd)
        Q_out_phys.scatter_add_(0, src_fwd, Q_fwd)
        net_flow_phys = Q_in_phys - Q_out_phys
    else:
        Q_in_phys.scatter_add_(0, edge_index[1], Q_pred_phys)
        Q_out_phys.scatter_add_(0, edge_index[0], Q_pred_phys)
        # Bidirectional edges: forward + (negated) reverse pair → /2 to recover
        # the physical net flow.
        net_flow_phys = (Q_in_phys - Q_out_phys) / 2.0

    # S_gt was stored as S_phys / V_std_per_node (dimensionless).
    S_phys = graph.source_term.to(device) * V_std_per_node

    residual_phys = delta_V_phys - (net_flow_phys + S_phys)            # m³
    residual_norm = residual_phys / V_std_per_node                     # dimensionless

    keep = torch.ones(num_nodes, dtype=torch.bool, device=device)
    if apply_boundary_mask and hasattr(graph, "boundary_mask") and graph.boundary_mask is not None:
        keep &= ~graph.boundary_mask.to(device).bool()
    if restrict_to_2d and hasattr(graph, "node_type") and graph.node_type is not None:
        keep &= (graph.node_type.to(device) == 0)

    residual_kept = residual_norm[keep]
    if residual_kept.numel() == 0:
        return torch.zeros((), device=device, dtype=node_pred.dtype)

    # v7 Exp 7: per-node-type weighted mean. The unweighted mean (default)
    # drowns ~50 1D nodes in ~3733 2D nodes, so the 2D-1D connection-seam
    # residual carries almost no gradient signal. Weighting 1D nodes by
    # N_2d_kept / N_1d_kept makes each type contribute equal total weight.
    weights = None
    if (
        node_type_weighting == "per_type"
        and hasattr(graph, "node_type")
        and graph.node_type is not None
        and not restrict_to_2d  # 1D already dropped → weighting is a no-op
    ):
        nt_kept = graph.node_type.to(device)[keep]
        n_2d = (nt_kept == 0).sum().clamp(min=1)
        n_1d = (nt_kept == 1).sum().clamp(min=1)
        if n_1d > 0:
            w_1d = n_2d.to(node_pred.dtype) / n_1d.to(node_pred.dtype)
            weights = torch.ones_like(residual_kept)
            weights[nt_kept == 1] = w_1d

    if smooth_l1_beta > 0.0:
        target = torch.zeros_like(residual_kept)
        per_node = F.smooth_l1_loss(
            residual_kept, target, beta=smooth_l1_beta, reduction="none"
        )
    else:
        per_node = residual_kept.abs()

    if weights is not None:
        return (weights * per_node).sum() / weights.sum()
    return per_node.mean()


def compute_steady_state_conservation_loss(
    node_pred, edge_pred, graph,
    smooth_l1_beta: float = 0.0,
    apply_boundary_mask: bool = False,
    restrict_to_2d: bool = False,
    node_type_weighting: str = "none",
):
    """Steady-state mass-conservation loss for the v9 time-less regime.

    Sibling of :func:`compute_local_conservation_loss`, but with the time
    derivative dropped. At the storm peak dV/dt≈0, so local continuity reduces
    to ``net_flow + S = 0`` (net edge inflow balances the rainfall source).
    The edge decoder predicts the steady flow Q directly: the dataset sets
    ``edge_q_prev = 0`` so ``Q_pred = edge_pred * Q_sigma_per_edge``. The
    scatter/aggregation, boundary/2D masking, per-type weighting and reduction
    are identical to the autoregressive loss; only the residual changes from
    ``ΔV − (net_flow + S)`` to ``net_flow + S``.

    L_steady = mean_i | (net_flow_phys_i + S_phys_i) / V_std_per_node_i |

    Args:
        node_pred (torch.Tensor): Node predictions [N, 1] (peak depth). Used
            only for device/dtype; it carries no volume channel here.
        edge_pred (torch.Tensor): Edge predictions [E, 1] (steady Q normalised
            per edge by ``Q_sigma_per_edge_phys``).
        graph (PyGData): carries edge_index, edge_q_prev (=0), source_term,
            V_std_per_node, Q_sigma_per_edge_phys; optionally boundary_mask,
            node_type, is_forward_edge.
        smooth_l1_beta, apply_boundary_mask, restrict_to_2d, node_type_weighting:
            same semantics as :func:`compute_local_conservation_loss`.

    Returns:
        torch.Tensor: Scalar steady-state conservation loss.
    """
    device = node_pred.device
    num_nodes = node_pred.shape[0]

    V_std_per_node = graph.V_std_per_node.to(device)              # [N], m³
    Q_sigma_per_edge = graph.Q_sigma_per_edge_phys.to(device)     # [E], m³

    # Steady flow in physical m³/step (edge_q_prev is 0 in static mode).
    Q_pred_phys = (graph.edge_q_prev.to(device) + edge_pred.squeeze(-1)) * Q_sigma_per_edge

    edge_index = graph.edge_index  # [2, E]
    Q_in_phys = torch.zeros(num_nodes, device=device, dtype=Q_pred_phys.dtype)
    Q_out_phys = torch.zeros(num_nodes, device=device, dtype=Q_pred_phys.dtype)

    if hasattr(graph, "is_forward_edge") and graph.is_forward_edge is not None:
        fwd = graph.is_forward_edge.to(device).bool()
        Q_fwd = Q_pred_phys[fwd]
        src_fwd = edge_index[0][fwd]
        dst_fwd = edge_index[1][fwd]
        Q_in_phys.scatter_add_(0, dst_fwd, Q_fwd)
        Q_out_phys.scatter_add_(0, src_fwd, Q_fwd)
        net_flow_phys = Q_in_phys - Q_out_phys
    else:
        Q_in_phys.scatter_add_(0, edge_index[1], Q_pred_phys)
        Q_out_phys.scatter_add_(0, edge_index[0], Q_pred_phys)
        net_flow_phys = (Q_in_phys - Q_out_phys) / 2.0

    # S_gt stored as S_phys / V_std_per_node (dimensionless).
    S_phys = graph.source_term.to(device) * V_std_per_node

    residual_phys = net_flow_phys + S_phys                        # m³/step
    residual_norm = residual_phys / V_std_per_node                # dimensionless

    keep = torch.ones(num_nodes, dtype=torch.bool, device=device)
    if apply_boundary_mask and hasattr(graph, "boundary_mask") and graph.boundary_mask is not None:
        keep &= ~graph.boundary_mask.to(device).bool()
    if restrict_to_2d and hasattr(graph, "node_type") and graph.node_type is not None:
        keep &= (graph.node_type.to(device) == 0)

    residual_kept = residual_norm[keep]
    if residual_kept.numel() == 0:
        return torch.zeros((), device=device, dtype=node_pred.dtype)

    weights = None
    if (
        node_type_weighting == "per_type"
        and hasattr(graph, "node_type")
        and graph.node_type is not None
        and not restrict_to_2d
    ):
        nt_kept = graph.node_type.to(device)[keep]
        n_2d = (nt_kept == 0).sum().clamp(min=1)
        n_1d = (nt_kept == 1).sum().clamp(min=1)
        if n_1d > 0:
            w_1d = n_2d.to(node_pred.dtype) / n_1d.to(node_pred.dtype)
            weights = torch.ones_like(residual_kept)
            weights[nt_kept == 1] = w_1d

    if smooth_l1_beta > 0.0:
        target = torch.zeros_like(residual_kept)
        per_node = F.smooth_l1_loss(
            residual_kept, target, beta=smooth_l1_beta, reduction="none"
        )
    else:
        per_node = residual_kept.abs()

    if weights is not None:
        return (weights * per_node).sum() / weights.sum()
    return per_node.mean()


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
