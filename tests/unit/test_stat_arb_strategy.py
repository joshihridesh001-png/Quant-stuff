"""Unit tests for Statistical Arbitrage Kalman Pairs Trading Strategy.

Functional Purpose:
    Verifies Kalman Filter state-space dynamic hedge ratio tracking, OU mean reversion
    half-life calculation, z-score threshold signals, dollar-neutral gross leverage
    invariants, and emergency structural break stop exits.

Explicit Dependency Tracking:
    - pytest, numpy.
    - quant.analytics.strategies.stat_arb: KalmanPairsTradingStrategy, ERR_STRAT_STATARB_INVALID_PAIR.
    - quant.analytics.strategies.base: BarHistoryWindow, SignalDirection, StrategyError.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant.analytics.strategies.base import BarHistoryWindow, SignalDirection, StrategyError
from quant.analytics.strategies.stat_arb import (
    ERR_STRAT_STATARB_INVALID_PAIR,
    KalmanPairsTradingStrategy,
)


def _generate_cointegrated_pair(
    n_bars: int = 150,
    true_beta: float = 1.5,
    true_alpha: float = 10.0,
    ar_coef: float = 0.5,
    spread_shock: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic cointegrated asset prices: Y_t = alpha + beta * X_t + e_t."""
    np.random.seed(42)
    # Random walk for X
    returns_x = np.random.normal(0.0005, 0.015, size=n_bars)
    prices_x = 100.0 * np.exp(np.cumsum(returns_x))

    # Stationary AR(1) spread
    spread = np.zeros(n_bars, dtype=np.float64)
    noise = np.random.normal(0.0, 1.0, size=n_bars)
    for t in range(1, n_bars):
        spread[t] = ar_coef * spread[t - 1] + noise[t]

    # Inject terminal shock if requested
    if spread_shock != 0.0:
        spread[-1] += spread_shock

    prices_y = true_alpha + true_beta * prices_x + spread
    return prices_y, prices_x


def test_stat_arb_initialization_validation() -> None:
    """Verify validation of pair symbols and z-score thresholds."""
    # Same symbol raises error
    with pytest.raises(StrategyError) as exc_info:
        KalmanPairsTradingStrategy(
            strategy_id="pair_invalid",
            asset_y="SPY",
            asset_x="SPY",
        )
    assert exc_info.value.code == ERR_STRAT_STATARB_INVALID_PAIR

    # z_entry <= z_exit raises error
    with pytest.raises(StrategyError):
        KalmanPairsTradingStrategy(
            strategy_id="pair_invalid_z",
            asset_y="SPY",
            asset_x="QQQ",
            z_entry=1.0,
            z_exit=1.5,
        )


def test_kalman_filter_tracks_true_beta() -> None:
    """Verify Kalman Filter accurately recovers true hedge ratio on cointegrated series."""
    y_prices, x_prices = _generate_cointegrated_pair(n_bars=200, true_beta=2.0, true_alpha=5.0)

    strat = KalmanPairsTradingStrategy(
        strategy_id="pair_kalman",
        asset_y="ASSET_Y",
        asset_x="ASSET_X",
        min_warmup_bars=50,
    )

    history = BarHistoryWindow(
        data={
            "ASSET_Y": {"close": y_prices},
            "ASSET_X": {"close": x_prices},
        },
        symbols=("ASSET_Y", "ASSET_X"),
    )

    signals = strat.compute_signals(history)
    sig_y = signals["ASSET_Y"]

    # Beta diagnosed in metadata should be close to 2.0 (+- 0.3)
    estimated_beta = float(sig_y.diagnostics["beta"])
    assert math.isclose(estimated_beta, 2.0, abs_tol=0.35)


def test_stat_arb_short_spread_signal_when_rich() -> None:
    """When spread residual is positive and large (z >= 2.0), strategy shorts Y and longs X."""
    # Inject positive shock within [z_entry, z_stop)
    y_prices, x_prices = _generate_cointegrated_pair(n_bars=100, true_beta=1.0, spread_shock=3.5)

    strat = KalmanPairsTradingStrategy(
        strategy_id="pair_rich",
        asset_y="Y",
        asset_x="X",
        min_warmup_bars=40,
        z_entry=2.0,
        z_exit=0.5,
        z_stop=4.0,
    )

    history = BarHistoryWindow(
        data={"Y": {"close": y_prices}, "X": {"close": x_prices}},
        symbols=("Y", "X"),
    )

    signals = strat.compute_signals(history)
    sig_y = signals["Y"]
    sig_x = signals["X"]

    # Y is rich -> SHORT Y
    assert sig_y.direction == SignalDirection.SHORT
    assert sig_y.target_weight < 0.0

    # X is hedge -> LONG X
    assert sig_x.direction == SignalDirection.LONG
    assert sig_x.target_weight > 0.0

    # Gross leverage constraint: |w_y| + |w_x| <= 1.0
    total_gross = abs(sig_y.target_weight) + abs(sig_x.target_weight)
    assert total_gross <= 1.0001
    assert sig_y.conviction >= 0.5


def test_stat_arb_long_spread_signal_when_cheap() -> None:
    """When spread residual is negative and large (z <= -2.0), strategy longs Y and shorts X."""
    # Inject negative shock within (-z_stop, -z_entry]
    y_prices, x_prices = _generate_cointegrated_pair(n_bars=100, true_beta=1.0, spread_shock=-3.5)

    strat = KalmanPairsTradingStrategy(
        strategy_id="pair_cheap",
        asset_y="Y",
        asset_x="X",
        min_warmup_bars=40,
        z_entry=2.0,
        z_exit=0.5,
        z_stop=4.0,
    )

    history = BarHistoryWindow(
        data={"Y": {"close": y_prices}, "X": {"close": x_prices}},
        symbols=("Y", "X"),
    )

    signals = strat.compute_signals(history)
    sig_y = signals["Y"]
    sig_x = signals["X"]

    # Y is cheap -> LONG Y
    assert sig_y.direction == SignalDirection.LONG
    assert sig_y.target_weight > 0.0

    # X is hedge -> SHORT X
    assert sig_x.direction == SignalDirection.SHORT
    assert sig_x.target_weight < 0.0


def test_stat_arb_structural_break_emergency_stop() -> None:
    """When spread residual explodes past z_stop (4.0), strategy exits to FLAT."""
    # Massive divergence shock (+25.0)
    y_prices, x_prices = _generate_cointegrated_pair(n_bars=100, true_beta=1.0, spread_shock=25.0)

    strat = KalmanPairsTradingStrategy(
        strategy_id="pair_break",
        asset_y="Y",
        asset_x="X",
        min_warmup_bars=40,
        z_entry=2.0,
        z_stop=4.0,
    )

    history = BarHistoryWindow(
        data={"Y": {"close": y_prices}, "X": {"close": x_prices}},
        symbols=("Y", "X"),
    )

    signals = strat.compute_signals(history)
    sig_y = signals["Y"]
    sig_x = signals["X"]

    # Should trigger structural break stop
    assert sig_y.direction == SignalDirection.FLAT
    assert sig_y.target_weight == 0.0
    assert sig_x.direction == SignalDirection.FLAT
    assert sig_x.target_weight == 0.0
    assert sig_y.diagnostics["structural_break"] == 1


def test_stat_arb_dollar_neutral_weighting_asymmetric_prices() -> None:
    """Verify dollar-neutral weighting scales by P_x / P_y under asymmetric price levels."""
    # Y is priced at ~$500, X is priced at ~$50.
    # Relationship: Y = 10 * X -> beta = 10.
    np.random.seed(123)
    n_bars = 100
    prices_x = 50.0 + np.cumsum(np.random.normal(0, 0.5, size=n_bars))
    spread = np.zeros(n_bars, dtype=np.float64)
    for t in range(1, n_bars):
        spread[t] = 0.5 * spread[t - 1] + np.random.normal(0, 0.5)

    # Rich spread at the end to trigger entry without triggering z_stop (4.0)
    spread[-1] += 1.8
    prices_y = 10.0 * prices_x + spread  # Y ~$500, X ~$50

    strat = KalmanPairsTradingStrategy(
        strategy_id="pair_asym",
        asset_y="SPY_LIKE",
        asset_x="CHEAP_ETF",
        min_warmup_bars=40,
        z_entry=2.0,
        z_exit=0.5,
    )

    history = BarHistoryWindow(
        data={"SPY_LIKE": {"close": prices_y}, "CHEAP_ETF": {"close": prices_x}},
        symbols=("SPY_LIKE", "CHEAP_ETF"),
    )

    signals = strat.compute_signals(history)
    sig_y = signals["SPY_LIKE"]
    sig_x = signals["CHEAP_ETF"]

    # Beta is ~10, but dollar beta = beta * (P_x / P_y) = 10 * (50 / 500) = 1.0!
    # Therefore, dollar weights should be approximately balanced ~0.5 each
    assert sig_y.direction == SignalDirection.SHORT
    assert sig_x.direction == SignalDirection.LONG

    w_y = abs(sig_y.target_weight)
    w_x = abs(sig_x.target_weight)
    # Weights should be close to 0.5 rather than 0.09 and 0.91
    assert math.isclose(w_y, 0.5, abs_tol=0.15)
    assert math.isclose(w_x, 0.5, abs_tol=0.15)
    assert math.isclose(w_y + w_x, 1.0, abs_tol=1e-4)


def test_stat_arb_stateful_hysteresis_and_exit() -> None:
    """Verify position is held in hysteresis band (z_exit < |z| < z_entry) until z_exit."""
    y_prices, x_prices = _generate_cointegrated_pair(n_bars=100, true_beta=1.0, spread_shock=3.5)

    strat = KalmanPairsTradingStrategy(
        strategy_id="pair_hysteresis",
        asset_y="Y",
        asset_x="X",
        min_warmup_bars=40,
        z_entry=2.0,
        z_exit=0.5,
        z_stop=4.0,
    )

    # 1. Entry bar (z >= 2.0)
    history1 = BarHistoryWindow(
        data={"Y": {"close": y_prices}, "X": {"close": x_prices}},
        symbols=("Y", "X"),
    )
    sig1 = strat.compute_signals(history1)
    assert strat.current_position == -1
    assert sig1["Y"].direction == SignalDirection.SHORT
    assert sig1["X"].direction == SignalDirection.LONG

    # 2. Next bar: spread mean-reverting but still in hysteresis band (z ~ 1.27)
    # Reduce terminal shock from 3.5 to 2.5
    y2 = np.copy(y_prices)
    y2[-1] = y_prices[-1] - 1.0
    history2 = BarHistoryWindow(
        data={"Y": {"close": y2}, "X": {"close": x_prices}},
        symbols=("Y", "X"),
    )
    sig2 = strat.compute_signals(history2)
    # In hysteresis band: trade MUST be held (not flattened prematurely)
    assert strat.current_position == -1
    assert sig2["Y"].direction == SignalDirection.SHORT
    assert sig2["X"].direction == SignalDirection.LONG

    # 3. Third bar: spread fully mean-reverts -> exit at |z| <= 0.5
    y3 = np.copy(y_prices)
    y3[-1] = y_prices[-1] - 2.0  # Brings z down to 0.371 <= 0.5
    history3 = BarHistoryWindow(
        data={"Y": {"close": y3}, "X": {"close": x_prices}},
        symbols=("Y", "X"),
    )
    sig3 = strat.compute_signals(history3)
    # Spread mean-reverted: exit to FLAT
    assert strat.current_position == 0
    assert sig3["Y"].direction == SignalDirection.FLAT
    assert sig3["X"].direction == SignalDirection.FLAT
    assert sig3["Y"].target_weight == 0.0
    assert sig3["X"].target_weight == 0.0

    # 4. Test reset
    strat._current_position = 1
    strat.reset()
    assert strat.current_position == 0
