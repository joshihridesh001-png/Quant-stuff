"""Composite Swarm Meta-Strategy: Regime-Conditioned Dynamic Model Averaging (RD-DMA).

Functional Purpose:
    Implements a multi-strategy meta-ensemble combining all four quantitative alpha engines:
    (1) Kalman Pairs Trading, (2) FracDiff Momentum, (3) Loughran-McDonald Sentiment, and
    (4) Volatility Breakout. Dynamically allocates strategy weights across the probability
    simplex using Entropic Mirror Descent conditioned on macro regime classifications.

Explicit Dependency Tracking:
    - numpy: Exponential gradient descent, simplex normalization, and entropy metrics.
    - quant.analytics.strategies.base: BaseAlphaStrategy, IAlphaStrategy, StrategySignal,
      SignalDirection, BarHistoryWindow, StrategyContext, StrategyError.
    - quant.analytics.strategies.stat_arb: KalmanPairsTradingStrategy.
    - quant.analytics.strategies.momentum: FracDiffMomentumStrategy.
    - quant.analytics.strategies.sentiment: LoughranMcDonaldSentimentStrategy.
    - quant.analytics.strategies.volatility: VolatilityBreakoutStrategy.

Structural Relationship:
    Flagship composite strategy uniting the entire Phase 14 library. Ingested by BacktestRunner,
    FastAPI backtesting endpoints, and Turnkey LiveSessionOrchestrator.

Defensive Invariants:
    - INV-STRAT-001: Net asset target weights strictly in [-1.0, 1.0].
    - INV-STRAT-002: Conviction scores strictly in [0.0, 1.0].
    - Simplex Invariant: Strategy blending weights sum to 1.0 (sum(w_m) == 1.0, w_m >= 0).
    - Conservative Leverage: Overall gross leverage bounded by max_leverage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from quant.analytics.strategies.base import (
    BarHistoryWindow,
    BaseAlphaStrategy,
    IAlphaStrategy,
    SignalDirection,
    StrategyContext,
    StrategyError,
    StrategySignal,
)

# ============================================================================
# Diagnostic Fault Codes
# ============================================================================

ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES: Final[str] = "ERR-STRAT-050"
ERR_STRAT_SWARM_INVALID_REGIME: Final[str] = "ERR-STRAT-051"


# ============================================================================
# Regime Priors for Alpha Blending
# ============================================================================

# Default prior allocations across [stat_arb, momentum, sentiment, volatility]
REGIME_PRIORS: Final[dict[str, tuple[float, float, float, float]]] = {
    "NORMAL": (0.25, 0.25, 0.25, 0.25),
    "BULL": (0.10, 0.45, 0.20, 0.25),
    "BEAR": (0.15, 0.45, 0.15, 0.25),
    "PANIC": (0.05, 0.15, 0.30, 0.50),
    "MEAN_REVERTING": (0.50, 0.10, 0.20, 0.20),
    "EVENT_DRIVEN": (0.10, 0.15, 0.55, 0.20),
}


# ============================================================================
# Composite Swarm Meta-Strategy
# ============================================================================


class SwarmMetaStrategy(BaseAlphaStrategy):
    """Regime-Conditioned Dynamic Model Averaging (RD-DMA) multi-strategy ensemble.

    Functional Purpose:
        Dynamically weights constituent alpha signals according to prevailing macro regimes
        and past performance updates via Entropic Mirror Descent.
    """

    def __init__(
        self,
        strategy_id: str,
        monitored_symbols: Sequence[str],
        sub_strategies: Sequence[IAlphaStrategy],
        learning_rate: float = 0.05,
        regime_prior_weight: float = 0.20,
        min_warmup_bars: int | None = None,
        max_leverage: float = 1.0,
    ) -> None:
        """Initialize Swarm Meta-Strategy."""
        if not sub_strategies:
            raise StrategyError(
                "sub_strategies must contain at least one strategy",
                code=ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES,
            )

        # Resolve warmup requirement as max of all sub-strategy warmups
        resolved_warmup = (
            min_warmup_bars
            if min_warmup_bars is not None
            else max(s.min_warmup_bars for s in sub_strategies)
        )

        super().__init__(
            strategy_id=strategy_id,
            monitored_symbols=monitored_symbols,
            min_warmup_bars=resolved_warmup,
            max_leverage=max_leverage,
        )

        self._sub_strategies = tuple(sub_strategies)
        self._learning_rate = learning_rate
        self._prior_weight = regime_prior_weight

        # Initialize uniform strategy simplex weights
        m_count = len(self._sub_strategies)
        self._strategy_weights = np.full(m_count, 1.0 / m_count, dtype=np.float64)

    @property
    def sub_strategies(self) -> tuple[IAlphaStrategy, ...]:
        """Tuple of active constituent alpha strategies."""
        return self._sub_strategies

    @property
    def strategy_weights(self) -> np.ndarray:
        """Current simplex weights allocated across constituent strategies."""
        return self._strategy_weights.copy()

    def update_weights_with_rewards(
        self,
        rewards: Sequence[float],
        regime_label: str = "NORMAL",
    ) -> None:
        """Update strategy simplex weights via Entropic Mirror Descent.

        Formulation:
            w_{m, t+1} proportional to w_{m, t} * exp(eta * r_m)
            Regularization: w <- (1 - lambda) * w + lambda * pi(regime)
        """
        if len(rewards) != len(self._sub_strategies):
            raise StrategyError(
                f"Rewards length ({len(rewards)}) does not match sub-strategies count ({len(self._sub_strategies)})"
            )

        # 1. Exponential gradient update
        log_w = np.log(np.maximum(self._strategy_weights, 1e-12))
        step = log_w + self._learning_rate * np.array(rewards, dtype=np.float64)
        # Softmax normalization for numerical stability
        exp_step = np.exp(step - np.max(step))
        updated_w = exp_step / np.sum(exp_step)

        # 2. Regularization towards regime prior
        prior_tuple = REGIME_PRIORS.get(regime_label.upper(), REGIME_PRIORS["NORMAL"])
        if len(prior_tuple) == len(self._sub_strategies):
            prior_arr = np.array(prior_tuple, dtype=np.float64)
            updated_w = (1.0 - self._prior_weight) * updated_w + self._prior_weight * prior_arr
            updated_w /= np.sum(updated_w)  # Renormalize

        self._strategy_weights = updated_w

    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Aggregate constituent strategy signals into composite portfolio allocations."""
        regime = context.regime_label.upper() if context else "NORMAL"

        # If 4 sub-strategies and weights unadjusted, align with regime prior
        if len(self._sub_strategies) == 4 and regime in REGIME_PRIORS:
            prior_arr = np.array(REGIME_PRIORS[regime], dtype=np.float64)
            # Blend current weights with regime prior
            active_weights = (
                1.0 - self._prior_weight
            ) * self._strategy_weights + self._prior_weight * prior_arr
            active_weights /= np.sum(active_weights)
        else:
            active_weights = self._strategy_weights

        # 1. Collect signals from all constituent strategies
        all_signals: list[Mapping[str, StrategySignal]] = []
        for strat in self._sub_strategies:
            strat_signals = strat.compute_signals(history, context)
            all_signals.append(strat_signals)

        # 2. Linear combination across the strategy simplex
        asset_weights: dict[str, float] = defaultdict(float)
        asset_convictions: dict[str, float] = defaultdict(float)
        asset_stops: dict[str, list[float]] = defaultdict(list)
        asset_takes: dict[str, list[float]] = defaultdict(list)

        for m_idx, sig_dict in enumerate(all_signals):
            strat_weight = float(active_weights[m_idx])
            for sym, sig in sig_dict.items():
                asset_weights[sym] += strat_weight * sig.target_weight
                asset_convictions[sym] += strat_weight * sig.conviction
                if sig.stop_loss_pct is not None:
                    asset_stops[sym].append(sig.stop_loss_pct)
                if sig.take_profit_pct is not None:
                    asset_takes[sym].append(sig.take_profit_pct)

        # 3. Shannon Entropy of Strategy Allocation
        entropy_val = -float(np.sum(active_weights * np.log(np.maximum(active_weights, 1e-12))))

        composite_signals: dict[str, StrategySignal] = {}

        for sym in self.monitored_symbols:
            raw_w = asset_weights.get(sym, 0.0)
            raw_c = asset_convictions.get(sym, 0.0)

            # Directional threshold
            if raw_w > 0.01:
                direction = SignalDirection.LONG
                target_w = min(self._max_leverage, raw_w)
            elif raw_w < -0.01:
                direction = SignalDirection.SHORT
                target_w = max(-self._max_leverage, raw_w)
            else:
                direction = SignalDirection.FLAT
                target_w = 0.0

            conviction = float(min(1.0, max(0.0, raw_c)))

            # Average stop and take profit if defined
            stops = asset_stops.get(sym, [])
            takes = asset_takes.get(sym, [])
            avg_stop = round(float(np.mean(stops)), 4) if stops else 0.02
            avg_take = round(float(np.mean(takes)), 4) if takes else 0.04

            diag: dict[str, float | str | int] = {
                "regime": regime,
                "strategy_entropy": round(entropy_val, 4),
                "active_sub_strategies": len(self._sub_strategies),
            }
            # Record individual sub-strategy weight allocations
            for m_idx, strat in enumerate(self._sub_strategies):
                diag[f"weight_{strat.strategy_id}"] = round(float(active_weights[m_idx]), 4)

            signal = StrategySignal(
                symbol=sym,
                direction=direction,
                target_weight=round(target_w, 4),
                conviction=round(conviction, 4),
                target_horizon_bars=10,
                stop_loss_pct=avg_stop,
                take_profit_pct=avg_take,
                diagnostics=diag,
            )
            composite_signals[sym] = signal

        return composite_signals
