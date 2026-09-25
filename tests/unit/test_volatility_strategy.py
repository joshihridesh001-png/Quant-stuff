"""Unit tests for Volatility Breakout & Bollinger Range Squeeze Strategy.

Functional Purpose:
    Verifies Parkinson and Garman-Klass range volatility math, Bollinger Bandwidth squeeze
    detection, volume confirmation filters, and directional breakout signal emissions.

Explicit Dependency Tracking:
    - pytest, numpy.
    - quant.analytics.strategies.volatility: VolatilityBreakoutStrategy, compute_parkinson_volatility,
      compute_garman_klass_volatility, ERR_STRAT_VOL_INVALID_PARAMS.
    - quant.analytics.strategies.base: BarHistoryWindow, SignalDirection, StrategyError.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant.analytics.strategies.base import BarHistoryWindow, SignalDirection, StrategyError
from quant.analytics.strategies.volatility import (
    ERR_STRAT_VOL_INVALID_PARAMS,
    VolatilityBreakoutStrategy,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
)

# ============================================================================
# Estimator Tests
# ============================================================================


def test_parkinson_and_garman_klass_volatility_estimators() -> None:
    """Verify Parkinson and Garman-Klass range volatility mathematical correctness."""
    n_bars = 50
    # Synthetic bars with 2% intraday range
    closes = 100.0 * np.exp(np.linspace(0, 0.1, n_bars))
    opens = closes * 0.999
    highs = closes * 1.01
    lows = closes * 0.99

    park_vol = compute_parkinson_volatility(highs, lows, annualized=True)
    gk_vol = compute_garman_klass_volatility(opens, highs, lows, closes, annualized=True)

    assert park_vol > 0.05
    assert gk_vol > 0.05
    assert math.isfinite(park_vol)
    assert math.isfinite(gk_vol)

    # Flat bar edge case (H == L)
    flat_h = np.full(10, 100.0)
    flat_l = np.full(10, 100.0)
    assert compute_parkinson_volatility(flat_h, flat_l) == 0.0


def test_volatility_strategy_parameter_validation() -> None:
    """Verify invalid parameters trigger StrategyError with designated fault code."""
    # Squeeze percentile >= 50
    with pytest.raises(StrategyError) as exc_info:
        VolatilityBreakoutStrategy(
            strategy_id="vol_invalid",
            monitored_symbols=["SPY"],
            squeeze_percentile=60.0,
        )
    assert exc_info.value.code == ERR_STRAT_VOL_INVALID_PARAMS

    # bb_period too small
    with pytest.raises(StrategyError):
        VolatilityBreakoutStrategy(
            strategy_id="vol_invalid_bb",
            monitored_symbols=["SPY"],
            bb_period=2,
        )


# ============================================================================
# Squeeze & Breakout Signal Generation Tests
# ============================================================================


def _generate_squeeze_and_breakout(
    n_squeeze: int = 130,
    breakout_type: str = "bull",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate price history that coils tightly in a squeeze, then breaks out violently."""
    np.random.seed(42)
    # Earlier period has wider fluctuations (105 bars)
    wide_noise = np.random.normal(0, 1.5, 105)
    # Squeeze period coils down tightly (25 bars)
    tight_noise = np.random.normal(0, 0.05, 25)
    consolidation = 100.0 + np.concatenate([wide_noise, tight_noise])

    c_opens = consolidation - 0.05
    c_closes = consolidation + 0.05
    c_highs = np.maximum(c_opens, c_closes) + 0.1
    c_lows = np.minimum(c_opens, c_closes) - 0.1
    c_volumes = np.full(len(consolidation), 1000.0)

    # Add explosive breakout bar with 3x volume and large range
    if breakout_type == "bull":
        b_open = 100.0
        b_close = 103.5  # Strong surge above upper Bollinger Band
        b_high = 104.0
        b_low = 100.0
    elif breakout_type == "bear":
        b_open = 100.0
        b_close = 96.5  # Strong plunge below lower Bollinger Band
        b_high = 100.0
        b_low = 96.0
    else:  # neutral inside squeeze
        b_open = 100.0
        b_close = 100.05
        b_high = 100.1
        b_low = 99.95

    b_volume = 3000.0  # 3x volume surge

    opens = np.append(c_opens, b_open)
    closes = np.append(c_closes, b_close)
    highs = np.append(c_highs, b_high)
    lows = np.append(c_lows, b_low)
    volumes = np.append(c_volumes, b_volume)

    return opens, highs, lows, closes, volumes


def test_volatility_strategy_bullish_breakout() -> None:
    """Verify that an upward surge out of a tight squeeze emits LONG signal with volume confirmation."""
    opens, highs, lows, closes, volumes = _generate_squeeze_and_breakout(
        n_squeeze=130, breakout_type="bull"
    )

    strat = VolatilityBreakoutStrategy(
        strategy_id="vol_bull",
        monitored_symbols=["SPY"],
        bb_period=20,
        squeeze_lookback=100,
        squeeze_percentile=20.0,
        vol_expansion_threshold=1.2,
        min_warmup_bars=120,
    )

    history = BarHistoryWindow(
        data={
            "SPY": {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        },
        symbols=("SPY",),
    )

    signals = strat.compute_signals(history)
    sig = signals["SPY"]

    assert sig.direction == SignalDirection.LONG
    assert sig.target_weight > 0.0
    assert sig.conviction >= 0.6
    assert sig.diagnostics["volume_confirmed"] == 1


def test_volatility_strategy_bearish_breakout() -> None:
    """Verify that a downward plunge out of a tight squeeze emits SHORT signal."""
    opens, highs, lows, closes, volumes = _generate_squeeze_and_breakout(
        n_squeeze=130, breakout_type="bear"
    )

    strat = VolatilityBreakoutStrategy(
        strategy_id="vol_bear",
        monitored_symbols=["QQQ"],
        bb_period=20,
        squeeze_lookback=100,
        squeeze_percentile=20.0,
        vol_expansion_threshold=1.2,
        min_warmup_bars=120,
    )

    history = BarHistoryWindow(
        data={
            "QQQ": {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        },
        symbols=("QQQ",),
    )

    signals = strat.compute_signals(history)
    sig = signals["QQQ"]

    assert sig.direction == SignalDirection.SHORT
    assert sig.target_weight < 0.0
    assert sig.conviction >= 0.6


def test_volatility_strategy_flat_inside_squeeze() -> None:
    """Verify that remaining inside the squeeze bands maintains FLAT signal."""
    opens, highs, lows, closes, volumes = _generate_squeeze_and_breakout(
        n_squeeze=130, breakout_type="neutral"
    )

    strat = VolatilityBreakoutStrategy(
        strategy_id="vol_neutral",
        monitored_symbols=["AAPL"],
        bb_period=20,
        squeeze_lookback=100,
        min_warmup_bars=120,
    )

    history = BarHistoryWindow(
        data={
            "AAPL": {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        },
        symbols=("AAPL",),
    )

    signals = strat.compute_signals(history)
    sig = signals["AAPL"]

    assert sig.direction == SignalDirection.FLAT
    assert sig.target_weight == 0.0
    assert sig.diagnostics["is_squeeze"] == 1
