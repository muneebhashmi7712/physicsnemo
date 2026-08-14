# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation metrics for HydroGraphNet inference, matching the paper
(Taghizadeh et al., DOI: 10.1111/mice.13484, Section 5.1).

All functions expect denormalized tensors in physical units (feet for
water depth, ft^3 for volume).

The mass-balance metric is the paper's squared global continuity residual
(Equation 33), not a prediction-versus-ground-truth volume error. Its unit is
the square of the volume unit (ft^6 here); it is not a depth or a relative
error.
"""

import torch


def compute_rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Root Mean Square Error (Eq. 31).

    RMSE = sqrt(1/N * sum((h_pred_i - h_obs_i)^2))
    Returns RMSE in feet. Lower is better.
    """
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


def compute_nse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Nash-Sutcliffe Efficiency coefficient (Eq. 32).

    NSE = 1 - sum((h_pred - h_obs)^2) / sum((h_obs - h_mean)^2)
    1.0 = perfect, 0.0 = as good as mean, <0 = worse than mean.
    """
    ss_res = torch.sum((pred - gt) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else float("-inf")
    return (1.0 - ss_res / ss_tot).item()


def compute_csi(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> float:
    """Critical Success Index at a given water depth threshold (Section 5.1).

    CSI = hits / (hits + misses + false_alarms)
    Higher is better. NaN if no exceedances in either pred or gt.
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


def compute_mass_balance_residual(
    current_volume: torch.Tensor,
    previous_volume: torch.Tensor,
    net_source_rate_previous: torch.Tensor | float,
    net_source_rate_current: torch.Tensor | float,
    delta_t: float,
) -> float:
    r"""Return the signed global continuity residual from paper Equation 33.

    .. math::

        r(t) = \sum_i V_i^{(t)} - \sum_i V_i^{(t-1)}
             - \frac{\Delta t}{2}\left(s^{(t-1)} + s^{(t)}\right)

    net_source_rate_* is the already-combined external source rate s:
    boundary inflow plus effective rainfall minus infiltration and any known
    external outflow. Internal edge flows must not be included because they
    cancel in a whole-domain balance.

    Inputs must be denormalized and mutually consistent. If volume is in ft^3,
    source rate must be in ft^3/s and delta_t in seconds; the returned residual
    is then in ft^3. Summation is deliberately performed in float64 because
    subtracting two large domain totals in float32 can manufacture a substantial
    apparent balance error.
    """
    if not isinstance(delta_t, (int, float)) or not torch.isfinite(
        torch.tensor(float(delta_t), dtype=torch.float64)
    ):
        raise ValueError("delta_t must be a finite positive number")
    if delta_t <= 0:
        raise ValueError("delta_t must be a finite positive number")

    current = torch.as_tensor(current_volume)
    previous = torch.as_tensor(previous_volume, device=current.device)
    source_previous = torch.as_tensor(
        net_source_rate_previous, device=current.device, dtype=torch.float64
    )
    source_current = torch.as_tensor(
        net_source_rate_current, device=current.device, dtype=torch.float64
    )

    values = (current, previous, source_previous, source_current)
    if any(value.numel() == 0 for value in values):
        raise ValueError("mass-balance inputs must be non-empty")
    if any(not torch.isfinite(value).all().item() for value in values):
        raise ValueError("mass-balance inputs must contain only finite values")

    storage_change = current.to(torch.float64).sum() - previous.to(torch.float64).sum()
    integrated_source = 0.5 * float(delta_t) * (
        source_previous.sum() + source_current.sum()
    )
    return (storage_change - integrated_source).item()


def compute_mass_balance_error(
    current_volume: torch.Tensor,
    previous_volume: torch.Tensor,
    net_source_rate_previous: torch.Tensor | float,
    net_source_rate_current: torch.Tensor | float,
    delta_t: float,
) -> float:
    r"""Mass-balance error from HydroGraphNet paper Equation 33.

    The metric is compute_mass_balance_residual(...) ** 2. Perfect global
    conservation gives zero. If volume is expressed in ft^3 the result is in
    ft^6; for m^3 it is in m^6.

    This intentionally does not compare predicted and ground-truth volumes.
    That relative volume-accuracy calculation was the project's previous,
    broken implementation and is not the metric defined by the paper.
    """
    residual = compute_mass_balance_residual(
        current_volume,
        previous_volume,
        net_source_rate_previous,
        net_source_rate_current,
        delta_t,
    )
    return residual * residual
