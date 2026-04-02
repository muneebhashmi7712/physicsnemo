# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation metrics for HydroGraphNet inference, matching the paper
(Taghizadeh et al., DOI: 10.1111/mice.13484, Section 5.1).

All functions expect denormalized tensors in physical units (meters for
water depth, m^3 for volume).
"""

import torch


def compute_rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Root Mean Square Error (Eq. 31).

    RMSE = sqrt(1/N * sum((h_pred_i - h_obs_i)^2))

    Parameters
    ----------
    pred : torch.Tensor
        Predicted water depth [num_nodes].
    gt : torch.Tensor
        Ground truth water depth [num_nodes].

    Returns
    -------
    float
        RMSE in meters. Lower is better.
    """
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


def compute_nse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Nash-Sutcliffe Efficiency coefficient (Eq. 32).

    NSE = 1 - sum((h_pred - h_obs)^2) / sum((h_obs - h_mean)^2)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted water depth [num_nodes].
    gt : torch.Tensor
        Ground truth water depth [num_nodes].

    Returns
    -------
    float
        NSE value. 1.0 = perfect, 0.0 = as good as mean, <0 = worse than mean.
    """
    ss_res = torch.sum((pred - gt) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else float("-inf")
    return (1.0 - ss_res / ss_tot).item()


def compute_csi(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> float:
    """Critical Success Index at a given water depth threshold (Section 5.1).

    CSI = hits / (hits + misses + false_alarms)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted water depth [num_nodes] in meters.
    gt : torch.Tensor
        Ground truth water depth [num_nodes] in meters.
    threshold : float
        Water depth threshold in meters (paper uses 0.05 and 0.3).

    Returns
    -------
    float
        CSI value in [0, 1]. Higher is better. NaN if no events at threshold.
    """
    pred_exceed = pred >= threshold
    gt_exceed = gt >= threshold
    hits = (pred_exceed & gt_exceed).sum().item()
    misses = (~pred_exceed & gt_exceed).sum().item()
    false_alarms = (pred_exceed & ~gt_exceed).sum().item()
    denom = hits + misses + false_alarms
    if denom == 0:
        return float("nan")
    return hits / denom


def compute_mass_balance_error(pred_vol: torch.Tensor, gt_vol: torch.Tensor) -> float:
    """Relative mass balance error.

    MBE = |sum(V_pred) - sum(V_gt)| / sum(V_gt)

    Parameters
    ----------
    pred_vol : torch.Tensor
        Predicted volume per node [num_nodes] in m^3.
    gt_vol : torch.Tensor
        Ground truth volume per node [num_nodes] in m^3.

    Returns
    -------
    float
        Relative mass balance error. 0.0 = perfect conservation.
    """
    total_pred = pred_vol.sum().item()
    total_gt = gt_vol.sum().item()
    if abs(total_gt) < 1e-12:
        return 0.0 if abs(total_pred) < 1e-12 else float("inf")
    return abs(total_pred - total_gt) / abs(total_gt)
