"""Tests for decay kernel numerical truncation and maximum horizon bounds."""

from quant.services.event_service import compute_decay_kernel


def test_decay_kernel_truncation_beyond_max_horizon() -> None:
    # Lookback horizon beyond 7 days (604,800 seconds)
    decay = compute_decay_kernel(
        delta_t_seconds=700000.0,
        urgency=0.5,
        max_lookback_seconds=604800.0,
    )
    assert decay == 0.0


def test_decay_kernel_truncation_below_tolerance() -> None:
    # Deep time horizon where value falls below 1e-4 tolerance
    decay = compute_decay_kernel(
        delta_t_seconds=500000.0,
        urgency=0.9,
        tau_fast=100.0,
        tau_slow=500.0,
        tolerance=1e-3,
        max_lookback_seconds=None,
    )
    assert decay == 0.0


def test_decay_kernel_within_valid_horizon() -> None:
    decay = compute_decay_kernel(
        delta_t_seconds=1800.0,  # 30 mins
        urgency=0.5,
        tolerance=1e-4,
        max_lookback_seconds=604800.0,
    )
    assert decay > 0.0
