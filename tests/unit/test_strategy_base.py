"""Unit tests for Strategy Protocol, StrategySignal contracts, and BaseAlphaStrategy.

Functional Purpose:
    Verifies signal invariants (INV-STRAT-001 through INV-STRAT-005), strict non-finite & boolean
    rejection, directional consistency, warmup enforcement, and leverage budget scaling.

Explicit Dependency Tracking:
    - pytest, numpy.
    - quant.analytics.strategies.base: StrategySignal, SignalDirection, BarHistoryWindow,
      BaseAlphaStrategy, StrategyContext, InsufficientWarmupError, InvalidSignalError, NonFiniteWeightError.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant.analytics.strategies.base import (
    ERR_STRAT_INSUFFICIENT_WARMUP,
    BarHistoryWindow,
    BaseAlphaStrategy,
    InsufficientWarmupError,
    InvalidSignalError,
    NonFiniteWeightError,
    SignalDirection,
    StrategyContext,
    StrategySignal,
)

# ============================================================================
# StrategySignal Contract & Invariant Tests
# ============================================================================


def test_strategy_signal_valid_long_short_flat() -> None:
    """Verify construction of valid LONG, SHORT, and FLAT signals."""
    sig_long = StrategySignal(
        symbol="SPY",
        direction=SignalDirection.LONG,
        target_weight=0.5,
        conviction=0.85,
        target_horizon_bars=10,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
    )
    assert sig_long.symbol == "SPY"
    assert sig_long.target_weight == 0.5
    assert sig_long.conviction == 0.85
    assert sig_long.direction == SignalDirection.LONG

    sig_short = StrategySignal(
        symbol="QQQ",
        direction=SignalDirection.SHORT,
        target_weight=-0.3,
        conviction=0.7,
        target_horizon_bars=5,
    )
    assert sig_short.target_weight == -0.3
    assert sig_short.direction == SignalDirection.SHORT

    sig_flat = StrategySignal(
        symbol="AAPL",
        direction=SignalDirection.FLAT,
        target_weight=0.0,
        conviction=0.0,
        target_horizon_bars=1,
    )
    assert sig_flat.target_weight == 0.0
    assert sig_flat.direction == SignalDirection.FLAT


def test_strategy_signal_rejects_weight_out_of_bounds() -> None:
    """Verify weights outside [-1.0, 1.0] are rejected (INV-STRAT-001)."""
    with pytest.raises(InvalidSignalError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.LONG,
            target_weight=1.5,
            conviction=0.5,
            target_horizon_bars=5,
        )

    with pytest.raises(InvalidSignalError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.SHORT,
            target_weight=-1.01,
            conviction=0.5,
            target_horizon_bars=5,
        )


def test_strategy_signal_rejects_non_finite_and_booleans() -> None:
    """Verify NaNs, infinities, and booleans are strictly rejected (INV-STRAT-005)."""
    with pytest.raises(NonFiniteWeightError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.LONG,
            target_weight=float("nan"),
            conviction=0.5,
            target_horizon_bars=5,
        )

    with pytest.raises(NonFiniteWeightError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.LONG,
            target_weight=True,  # type: ignore[arg-type]
            conviction=0.5,
            target_horizon_bars=5,
        )

    with pytest.raises(InvalidSignalError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.LONG,
            target_weight=0.5,
            conviction=True,  # type: ignore[arg-type]
            target_horizon_bars=5,
        )


def test_strategy_signal_rejects_directional_inconsistencies() -> None:
    """Verify direction matches target weight sign (INV-STRAT-003)."""
    # LONG with negative weight
    with pytest.raises(InvalidSignalError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.LONG,
            target_weight=-0.5,
            conviction=0.5,
            target_horizon_bars=5,
        )

    # SHORT with positive weight
    with pytest.raises(InvalidSignalError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.SHORT,
            target_weight=0.5,
            conviction=0.5,
            target_horizon_bars=5,
        )

    # FLAT with non-zero weight
    with pytest.raises(InvalidSignalError):
        StrategySignal(
            symbol="SPY",
            direction=SignalDirection.FLAT,
            target_weight=0.1,
            conviction=0.5,
            target_horizon_bars=5,
        )


# ============================================================================
# BarHistoryWindow Tests
# ============================================================================


def test_bar_history_window_accessors() -> None:
    """Verify columnar accessors on BarHistoryWindow."""
    closes = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    opens = np.array([99.0, 100.0, 101.0], dtype=np.float64)
    volumes = np.array([1000.0, 1200.0, 1100.0], dtype=np.float64)

    window = BarHistoryWindow(
        data={"AAPL": {"close": closes, "open": opens, "volume": volumes}},
        symbols=("AAPL",),
    )

    assert window.bar_count("AAPL") == 3
    assert np.array_equal(window.closes("AAPL"), closes)
    assert np.array_equal(window.opens("AAPL"), opens)
    assert np.array_equal(window.volumes("AAPL"), volumes)
    assert window.bar_count("MSFT") == 0
    assert len(window.closes("MSFT")) == 0


# ============================================================================
# BaseAlphaStrategy Subclass & Warmup / Leverage Tests
# ============================================================================


class MockMomentumStrategy(BaseAlphaStrategy):
    """Simple test strategy that emits weight proportional to price difference."""

    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        signals: dict[str, StrategySignal] = {}
        for sym in self.monitored_symbols:
            closes = history.closes(sym)
            diff = closes[-1] - closes[0]
            if diff > 0:
                signals[sym] = StrategySignal(
                    symbol=sym,
                    direction=SignalDirection.LONG,
                    target_weight=0.8,
                    conviction=0.9,
                    target_horizon_bars=5,
                )
            else:
                signals[sym] = StrategySignal(
                    symbol=sym,
                    direction=SignalDirection.SHORT,
                    target_weight=-0.8,
                    conviction=0.9,
                    target_horizon_bars=5,
                )
        return signals


def test_strategy_insufficient_warmup_raises() -> None:
    """Verify InsufficientWarmupError when history contains fewer bars than min_warmup_bars."""
    strat = MockMomentumStrategy(
        strategy_id="test_momentum",
        monitored_symbols=["SPY"],
        min_warmup_bars=10,
    )

    history = BarHistoryWindow(
        data={"SPY": {"close": np.array([100.0, 101.0, 102.0])}},
        symbols=("SPY",),
    )

    with pytest.raises(InsufficientWarmupError) as exc_info:
        strat.compute_signals(history)
    assert exc_info.value.code == ERR_STRAT_INSUFFICIENT_WARMUP


def test_strategy_leverage_proportional_scaling() -> None:
    """Verify that when sum of absolute weights exceeds max_leverage, signals are scaled down."""
    strat = MockMomentumStrategy(
        strategy_id="test_momentum",
        monitored_symbols=["SPY", "QQQ"],
        min_warmup_bars=3,
        max_leverage=1.0,  # Max allowable total leverage is 1.0
    )

    # Both stocks show positive momentum, each emits target_weight=0.8 (total = 1.6 > 1.0)
    history = BarHistoryWindow(
        data={
            "SPY": {"close": np.array([100.0, 101.0, 105.0])},
            "QQQ": {"close": np.array([200.0, 202.0, 210.0])},
        },
        symbols=("SPY", "QQQ"),
    )

    signals = strat.compute_signals(history)
    assert len(signals) == 2

    total_weight = sum(abs(s.target_weight) for s in signals.values())
    assert math.isclose(total_weight, 1.0, rel_tol=1e-5)
    assert math.isclose(signals["SPY"].target_weight, 0.5, rel_tol=1e-5)
    assert math.isclose(signals["QQQ"].target_weight, 0.5, rel_tol=1e-5)
