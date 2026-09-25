"""Unit tests for Composite Swarm Meta-Strategy (RD-DMA).

Functional Purpose:
    Verifies multi-strategy ensemble aggregation, Entropic Mirror Descent reward updates,
    regime prior weighting, and bounded gross leverage invariants.

Explicit Dependency Tracking:
    - pytest, numpy.
    - quant.analytics.strategies.swarm_meta: SwarmMetaStrategy, REGIME_PRIORS, ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES.
    - quant.analytics.strategies.stat_arb: KalmanPairsTradingStrategy.
    - quant.analytics.strategies.momentum: FracDiffMomentumStrategy.
    - quant.analytics.strategies.sentiment: LoughranMcDonaldSentimentStrategy.
    - quant.analytics.strategies.volatility: VolatilityBreakoutStrategy.
    - quant.analytics.strategies.base: BarHistoryWindow, SignalDirection, StrategyContext, StrategyError.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant.analytics.strategies.base import (
    BarHistoryWindow,
    StrategyContext,
    StrategyError,
)
from quant.analytics.strategies.momentum import FracDiffMomentumStrategy
from quant.analytics.strategies.sentiment import (
    LoughranMcDonaldSentimentStrategy,
)
from quant.analytics.strategies.stat_arb import KalmanPairsTradingStrategy
from quant.analytics.strategies.swarm_meta import (
    ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES,
    SwarmMetaStrategy,
)
from quant.analytics.strategies.volatility import VolatilityBreakoutStrategy


def test_swarm_meta_empty_substrategies_raises() -> None:
    """Verify empty sub-strategy list raises StrategyError with ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES."""
    with pytest.raises(StrategyError) as exc_info:
        SwarmMetaStrategy(
            strategy_id="swarm_empty",
            monitored_symbols=["SPY"],
            sub_strategies=[],
        )
    assert exc_info.value.code == ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES


def test_swarm_meta_reward_update_mirror_descent() -> None:
    """Verify Entropic Mirror Descent updates simplex weights dynamically towards winning strategies."""
    strat1 = FracDiffMomentumStrategy(
        "strat1", ["SPY"], fast_ema_span=5, slow_ema_span=15, min_warmup_bars=25
    )
    strat2 = FracDiffMomentumStrategy(
        "strat2", ["SPY"], fast_ema_span=5, slow_ema_span=15, min_warmup_bars=25
    )

    swarm = SwarmMetaStrategy(
        strategy_id="swarm_test",
        monitored_symbols=["SPY"],
        sub_strategies=[strat1, strat2],
        learning_rate=0.5,
        regime_prior_weight=0.0,  # Zero prior weight for clean gradient test
    )

    initial_weights = swarm.strategy_weights
    assert math.isclose(initial_weights[0], 0.5, rel_tol=1e-5)
    assert math.isclose(initial_weights[1], 0.5, rel_tol=1e-5)

    # Strategy 1 generated +2.0 reward, Strategy 2 generated -2.0 reward
    swarm.update_weights_with_rewards(rewards=[2.0, -2.0], regime_label="NORMAL")

    updated_weights = swarm.strategy_weights
    # Strategy 1 should have gained weight, Strategy 2 lost weight
    assert updated_weights[0] > 0.70
    assert updated_weights[1] < 0.30
    assert math.isclose(float(np.sum(updated_weights)), 1.0, rel_tol=1e-5)


def test_swarm_meta_regime_prior_conditioning() -> None:
    """Verify that regime context dynamically conditions strategy simplex priors."""
    s_stat = KalmanPairsTradingStrategy("stat_arb", "SPY", "QQQ", min_warmup_bars=30)
    s_mom = FracDiffMomentumStrategy(
        "momentum", ["SPY", "QQQ"], fast_ema_span=5, slow_ema_span=15, min_warmup_bars=25
    )
    s_sent = LoughranMcDonaldSentimentStrategy("sentiment", ["SPY", "QQQ"], min_warmup_bars=10)
    s_vol = VolatilityBreakoutStrategy(
        "volatility",
        ["SPY", "QQQ"],
        bb_period=10,
        squeeze_lookback=20,
        min_warmup_bars=35,
    )

    swarm = SwarmMetaStrategy(
        strategy_id="swarm_regimes",
        monitored_symbols=["SPY", "QQQ"],
        sub_strategies=[s_stat, s_mom, s_sent, s_vol],
        regime_prior_weight=0.8,  # Heavy weight to regime prior
    )

    # In BULL regime, momentum gets dominant allocation (45%)
    swarm.update_weights_with_rewards(rewards=[0.0, 0.0, 0.0, 0.0], regime_label="BULL")
    bull_weights = swarm.strategy_weights
    # Index 1 is momentum
    assert bull_weights[1] > 0.35

    # In MEAN_REVERTING regime, stat arb gets dominant allocation (50%)
    swarm.update_weights_with_rewards(rewards=[0.0, 0.0, 0.0, 0.0], regime_label="MEAN_REVERTING")
    mr_weights = swarm.strategy_weights
    # Index 0 is stat arb
    assert mr_weights[0] > 0.40


def test_swarm_meta_composite_signal_generation() -> None:
    """Verify that SwarmMetaStrategy computes composite signals across all 4 alphas."""
    s_stat = KalmanPairsTradingStrategy("stat_arb", "SPY", "QQQ", min_warmup_bars=30)
    s_mom = FracDiffMomentumStrategy(
        "momentum", ["SPY", "QQQ"], fast_ema_span=5, slow_ema_span=15, min_warmup_bars=25
    )
    s_sent = LoughranMcDonaldSentimentStrategy("sentiment", ["SPY", "QQQ"], min_warmup_bars=10)
    s_vol = VolatilityBreakoutStrategy(
        "volatility",
        ["SPY", "QQQ"],
        bb_period=10,
        squeeze_lookback=20,
        min_warmup_bars=35,
    )

    swarm = SwarmMetaStrategy(
        strategy_id="swarm_composite",
        monitored_symbols=["SPY", "QQQ"],
        sub_strategies=[s_stat, s_mom, s_sent, s_vol],
        min_warmup_bars=35,
        max_leverage=1.0,
    )

    # Generate multi-asset history
    n_bars = 40
    spy_closes = 400.0 + np.cumsum(np.random.normal(0.2, 1.0, n_bars))
    qqq_closes = 350.0 + np.cumsum(np.random.normal(0.2, 1.0, n_bars))
    timestamps = np.array(
        [1_700_000_000_000_000_000 + i * 86_400_000_000_000 for i in range(n_bars)]
    )

    history = BarHistoryWindow(
        data={
            "SPY": {
                "close": spy_closes,
                "open": spy_closes * 0.999,
                "high": spy_closes * 1.005,
                "low": spy_closes * 0.995,
                "volume": np.full(n_bars, 1000.0),
                "timestamp": timestamps,
            },
            "QQQ": {
                "close": qqq_closes,
                "open": qqq_closes * 0.999,
                "high": qqq_closes * 1.005,
                "low": qqq_closes * 0.995,
                "volume": np.full(n_bars, 1000.0),
                "timestamp": timestamps,
            },
        },
        symbols=("SPY", "QQQ"),
    )

    context = StrategyContext(
        current_timestamp=int(timestamps[-1]),
        current_prices={"SPY": float(spy_closes[-1]), "QQQ": float(qqq_closes[-1])},
        current_positions={"SPY": 0.0, "QQQ": 0.0},
        total_equity=100000.0,
        unencumbered_cash=100000.0,
        regime_label="BULL",
    )

    signals = swarm.compute_signals(history, context)
    assert len(signals) == 2
    assert "SPY" in signals
    assert "QQQ" in signals

    sig_spy = signals["SPY"]
    sig_qqq = signals["QQQ"]

    # Target weights bounded by max_leverage
    assert -1.0 <= sig_spy.target_weight <= 1.0
    assert -1.0 <= sig_qqq.target_weight <= 1.0
    total_leverage = abs(sig_spy.target_weight) + abs(sig_qqq.target_weight)
    assert total_leverage <= 1.0001

    # Diagnostics verification
    assert sig_spy.diagnostics["regime"] == "BULL"
    assert "strategy_entropy" in sig_spy.diagnostics
    assert float(sig_spy.diagnostics["strategy_entropy"]) > 0.0
