# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the HydroGraphNet paper evaluation metrics."""

import math
import torch

from metrics import compute_mass_balance_error, compute_mass_balance_residual


def test_mass_balance_is_zero_for_exact_continuity():
    previous = torch.tensor([10.0, 20.0], dtype=torch.float64)
    current = torch.tensor([12.0, 23.0], dtype=torch.float64)

    residual = compute_mass_balance_residual(
        previous_volume=previous,
        current_volume=current,
        net_source_rate_previous=0.5,
        net_source_rate_current=1.5,
        delta_t=5.0,
    )
    error = compute_mass_balance_error(
        previous_volume=previous,
        current_volume=current,
        net_source_rate_previous=0.5,
        net_source_rate_current=1.5,
        delta_t=5.0,
    )

    assert math.isclose(residual, 0.0, abs_tol=1e-12)
    assert math.isclose(error, 0.0, abs_tol=1e-12)


def test_mass_balance_is_squared_signed_continuity_residual():
    previous = torch.tensor([10.0, 20.0], dtype=torch.float64)
    current = torch.tensor([15.0, 25.0], dtype=torch.float64)

    residual = compute_mass_balance_residual(current, previous, 0.5, 1.5, 5.0)
    error = compute_mass_balance_error(current, previous, 0.5, 1.5, 5.0)

    assert math.isclose(residual, 5.0, abs_tol=1e-12)
    assert math.isclose(error, 25.0, abs_tol=1e-12)


def test_mass_balance_sums_source_components_and_uses_float64():
    previous = torch.tensor([1.0e12, 2.0e12], dtype=torch.float64)
    current = torch.tensor([1.0e12 + 3.0, 2.0e12 + 7.0], dtype=torch.float64)
    source_previous = torch.tensor([0.25, 0.75], dtype=torch.float64)
    source_current = torch.tensor([0.5, 0.5], dtype=torch.float64)

    residual = compute_mass_balance_residual(
        current, previous, source_previous, source_current, 10.0
    )
    assert math.isclose(residual, 0.0, abs_tol=1e-12)


def test_mass_balance_rejects_invalid_timestep():
    for delta_t in (0.0, -1.0, float("nan")):
        try:
            compute_mass_balance_error(
                torch.ones(1), torch.ones(1), 0.0, 0.0, delta_t
            )
        except ValueError as exc:
            assert "delta_t" in str(exc)
        else:
            raise AssertionError(f"delta_t={delta_t!r} should be rejected")


if __name__ == "__main__":
    test_mass_balance_is_zero_for_exact_continuity()
    test_mass_balance_is_squared_signed_continuity_residual()
    test_mass_balance_sums_source_components_and_uses_float64()
    test_mass_balance_rejects_invalid_timestep()
    print("mass-balance metric tests passed")
