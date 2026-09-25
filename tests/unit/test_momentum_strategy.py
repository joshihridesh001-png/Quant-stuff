"""Unit tests for Memory-Preserving FracDiff Momentum Strategy.

Functional Purpose:
    Verifies fractional differentiation convolution causality, dual EMA trend detection,
    volatility-parity weight allocation, and boundary checks.

Explicit Dependency Tracking:
    - pytest, numpy.
    - quant.analytics.strategies.momentum: FracDiffMomentumStrategy, ERR_STRAT_MOMENTUM_INVALID_PARAMS.
    - quant.analytics.strategies.base: BarHistoryWindow, SignalDirection, StrategyError.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.analytics.strategies.base import BarHistoryWindow, SignalDirection, StrategyError
from quant.analytics.strategies.momentum import (
    ERR_STRAT_MOMENTUM_INVALID_PARAMS,
    FracDiffMomentumStrategy,
)


def test_fracdiff_momentum_param_validation() -> None:
    """Verify invalid parameters trigger StrategyError with ERR_STRAT_MOMENTUM_INVALID_PARAMS."""
    # d_order <= 0 or >= 1.0
    with pytest.raises(StrategyError) as exc_info:
        FracDiffMomentumStrategy(
            strategy_id="mom_invalid",
            monitored_symbols=["SPY"],
            d_order=1.5,
        )
    assert exc_info.value.code == ERR_STRAT_MOMENTUM_INVALID_PARAMS

    # fast_span >= slow_span
    with pytest.raises(StrategyError):
        FracDiffMomentumStrategy(
            strategy_id="mom_invalid_spans",
            monitored_symbols=["SPY"],
            fast_ema_span=20,
            slow_ema_span=10,
        )


def test_fracdiff_momentum_causal_convolution_invariance() -> None:
    """Verify that fractional differencing at time t has zero forward lookahead bias.

    The convoluted value at index t must be strictly identical whether computed on
    series of length t+1 or longer series T.
    """
    strat = FracDiffMomentumStrategy(
        strategy_id="mom_causal",
        monitored_symbols=["SPY"],
        d_order=0.45,
    )

    np.random.seed(123)
    full_series = np.cumsum(np.random.normal(0, 1, 100)) + 100.0

    # Compute on full series
    full_diff = strat._apply_frac_diff(full_series)

    # Compute on truncated slice [0 : 50]
    sub_series = full_series[:50]
    sub_diff = strat._apply_frac_diff(sub_series)

    # The first 50 values must be identical to machine precision
    np.testing.assert_allclose(full_diff[:50], sub_diff, rtol=1e-12, atol=1e-12)


def test_fracdiff_momentum_trend_detection() -> None:
    """Verify that a sustained bull trend generates LONG signal and bear generates SHORT."""
    n_bars = 60
    # Strong upward trend: 100 -> 160
    bull_prices = np.linspace(100.0, 160.0, n_bars) + np.random.normal(0, 0.2, n_bars)
    # Strong downward trend: 160 -> 100
    bear_prices = np.linspace(160.0, 100.0, n_bars) + np.random.normal(0, 0.2, n_bars)

    strat = FracDiffMomentumStrategy(
        strategy_id="mom_test",
        monitored_symbols=["BULL_ASSET", "BEAR_ASSET"],
        d_order=0.40,
        fast_ema_span=5,
        slow_ema_span=15,
        min_warmup_bars=25,
    )

    history = BarHistoryWindow(
        data={
            "BULL_ASSET": {"close": bull_prices},
            "BEAR_ASSET": {"close": bear_prices},
        },
        symbols=("BULL_ASSET", "BEAR_ASSET"),
    )

    signals = strat.compute_signals(history)

    sig_bull = signals["BULL_ASSET"]
    sig_bear = signals["BEAR_ASSET"]

    assert sig_bull.direction == SignalDirection.LONG
    assert sig_bull.target_weight > 0.0
    assert sig_bull.conviction > 0.5

    assert sig_bear.direction == SignalDirection.SHORT
    assert sig_bear.target_weight < 0.0
    assert sig_bear.conviction > 0.5


def test_fracdiff_momentum_volatility_parity_allocation() -> None:
    """Verify that volatility parity allocates larger weight to low-volatility asset."""
    n_bars = 60
    # Both assets trend upward at 1% per day
    # Low-volatility asset has minimal noise (vol ~ 5%)
    low_vol = 100.0 * np.exp(np.linspace(0, 0.3, n_bars))
    # High-volatility asset has large noise swings (vol ~ 40%)
    np.random.seed(42)
    high_vol = 100.0 * np.exp(np.linspace(0, 0.3, n_bars)) + np.random.normal(0, 3.0, n_bars)
    high_vol = np.maximum(high_vol, 50.0)

    strat = FracDiffMomentumStrategy(
        strategy_id="mom_vol_parity",
        monitored_symbols=["LOW_VOL", "HIGH_VOL"],
        d_order=0.35,
        fast_ema_span=5,
        slow_ema_span=15,
        min_warmup_bars=25,
        max_leverage=1.0,
    )

    history = BarHistoryWindow(
        data={
            "LOW_VOL": {"close": low_vol},
            "HIGH_VOL": {"close": high_vol},
        },
        symbols=("LOW_VOL", "HIGH_VOL"),
    )

    signals = strat.compute_signals(history)

    w_low = abs(signals["LOW_VOL"].target_weight)
    w_high = abs(signals["HIGH_VOL"].target_weight)

    # Low vol asset should receive higher weight than high vol asset
    assert w_low > w_high
    # Gross leverage ceiling respected
    assert (w_low + w_high) <= 1.0001
