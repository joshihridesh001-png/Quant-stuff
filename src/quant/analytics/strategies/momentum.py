"""Trend-Following: Memory-Preserving FracDiff Multi-Asset Momentum Strategy.

Functional Purpose:
    Implements long-memory stationary momentum filtering across multi-asset universes.
    Applies fractional differentiation of order d* to log closing prices, preserving multi-week
    price memory while guaranteeing econometric stationarity. Emits volatility-parity weighted
    signals governed by dual exponential moving average (EMA) crossovers.

Explicit Dependency Tracking:
    - numpy: Vectorized exponential moving average, rolling volatility, and array math.
    - quant.analytics.fractional_diff: compute_fractional_weights.
    - quant.analytics.strategies.base: BaseAlphaStrategy, StrategySignal, SignalDirection,
      BarHistoryWindow, StrategyContext, StrategyError.

Structural Relationship:
    Concrete trend-following alpha strategy in Phase 14 library. Ingested by BacktestRunner
    and combined with other alphas in SwarmMetaStrategy.

Defensive Invariants:
    - INV-STRAT-001: Target weights strictly in [-1.0, 1.0].
    - INV-STRAT-002: Conviction strictly in [0.0, 1.0].
    - Zero Forward Lookahead: Causal convolution filter applied strictly on past observations.
    - Volatility Parity: Normalizes risk budget by inverse asset volatility (1 / sigma_i).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

from quant.analytics.fractional_diff import compute_fractional_weights
from quant.analytics.strategies.base import (
    BarHistoryWindow,
    BaseAlphaStrategy,
    SignalDirection,
    StrategyContext,
    StrategyError,
    StrategySignal,
)

# ============================================================================
# Diagnostic Fault Codes
# ============================================================================

ERR_STRAT_MOMENTUM_INVALID_PARAMS: Final[str] = "ERR-STRAT-020"


# ============================================================================
# Memory-Preserving FracDiff Momentum Strategy
# ============================================================================


class FracDiffMomentumStrategy(BaseAlphaStrategy):
    """Memory-preserving trend following strategy using fractional differentiation.

    Functional Purpose:
        Applies fractional differencing operator (1 - B)^d to log prices to extract
        stationary trend features. Calculates fast and slow EMAs of the differentiated series,
        generating signals proportional to trend divergence normalized by trailing volatility.

    Explicit Dependency Tracking:
        compute_fractional_weights, BaseAlphaStrategy, BarHistoryWindow.

    Defensive Invariants:
        - 0.0 < d_order < 1.0.
        - fast_ema_span < slow_ema_span.
        - Volatility parity allocation: sum(|w_i|) <= max_leverage.
    """

    def __init__(
        self,
        strategy_id: str,
        monitored_symbols: Sequence[str],
        d_order: float = 0.40,
        fast_ema_span: int = 8,
        slow_ema_span: int = 21,
        volatility_lookback: int = 20,
        min_warmup_bars: int = 40,
        max_leverage: float = 1.0,
        weight_threshold: float = 1e-4,
    ) -> None:
        """Initialize FracDiff Momentum Strategy.

        Functional Purpose:
            Validates hyperparameters and precomputes fractional differencing filter weights.
        Explicit Dependency Tracking:
            compute_fractional_weights.
        Defensive Invariants:
            d_order in (0.0, 1.0), fast_span < slow_span.
        """
        if not (0.0 < d_order < 1.0):
            raise StrategyError(
                f"d_order must be in (0.0, 1.0), got {d_order}",
                code=ERR_STRAT_MOMENTUM_INVALID_PARAMS,
            )
        if fast_ema_span >= slow_ema_span or fast_ema_span < 2:
            raise StrategyError(
                f"fast_ema_span ({fast_ema_span}) must be >= 2 and < slow_ema_span ({slow_ema_span})",
                code=ERR_STRAT_MOMENTUM_INVALID_PARAMS,
            )
        if min_warmup_bars < slow_ema_span + 10:
            raise StrategyError(
                f"min_warmup_bars ({min_warmup_bars}) must be >= slow_ema_span + 10 ({slow_ema_span + 10})",
                code=ERR_STRAT_MOMENTUM_INVALID_PARAMS,
            )

        super().__init__(
            strategy_id=strategy_id,
            monitored_symbols=monitored_symbols,
            min_warmup_bars=min_warmup_bars,
            max_leverage=max_leverage,
        )

        self._d_order = d_order
        self._fast_span = fast_ema_span
        self._slow_span = slow_ema_span
        self._vol_lookback = volatility_lookback
        self._weight_threshold = weight_threshold

        # Precompute causal fractional differencing convolution weights
        self._weights = compute_fractional_weights(
            d=self._d_order,
            threshold=self._weight_threshold,
            zero_sum_correction=True,
        )

    @property
    def d_order(self) -> float:
        """Degree of fractional differentiation d*."""
        return self._d_order

    @property
    def fast_span(self) -> int:
        """Fast EMA span in bars."""
        return self._fast_span

    @property
    def slow_span(self) -> int:
        """Slow EMA span in bars."""
        return self._slow_span

    def _apply_frac_diff(self, series: np.ndarray) -> np.ndarray:
        """Apply causal fixed-window linear convolution of fractional weights.

        Model:
            tilde{p}_t = sum_{k=0}^{K-1} omega_k * (series_{t-k} - series_0)
        Invariant:
            Zero forward lookahead: only past observations (t - k) are accessed.
        """
        k_len = len(self._weights)
        n = len(series)
        diff_series = np.zeros(n, dtype=np.float64)
        base_val = series[0] if n > 0 else 0.0

        for t in range(n):
            # Window length available at bar t
            avail_k = min(t + 1, k_len)
            past_slice = series[t - avail_k + 1 : t + 1][::-1] - base_val
            weights_slice = self._weights[:avail_k]
            diff_series[t] = float(np.dot(weights_slice, past_slice))

        return diff_series

    def _compute_ema(self, series: np.ndarray, span: int) -> np.ndarray:
        """Compute exponential moving average recursively without future lookahead."""
        alpha = 2.0 / (span + 1.0)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for t in range(1, len(series)):
            ema[t] = alpha * series[t] + (1.0 - alpha) * ema[t - 1]
        return ema

    def _estimate_annualized_vol(self, closes: np.ndarray) -> float:
        """Estimate trailing annualized realized volatility from simple returns."""
        lookback = min(self._vol_lookback, len(closes) - 1)
        if lookback < 5:
            return 0.20  # Default 20% annualized volatility fallback

        recent_closes = closes[-lookback - 1 :]
        rets = np.diff(recent_closes) / recent_closes[:-1]
        std_ret = float(np.std(rets))
        ann_vol = std_ret * math.sqrt(252.0)
        return max(ann_vol, 0.05)  # Cap floor at 5% vol to avoid division by zero

    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Compute volatility-parity FracDiff momentum signals across monitored assets."""
        raw_scores: dict[str, float] = {}
        vols: dict[str, float] = {}
        fast_emas: dict[str, float] = {}
        slow_emas: dict[str, float] = {}

        for sym in self.monitored_symbols:
            closes = history.closes(sym)
            if len(closes) == 0:
                continue

            # 1. Transform to log prices
            log_prices = np.log(np.maximum(closes, 1e-4))

            # 2. Apply causal fractional differentiation
            frac_series = self._apply_frac_diff(log_prices)

            # 3. Dual EMA crossover on stationary fractionally differentiated series
            fast_ema = self._compute_ema(frac_series, self._fast_span)
            slow_ema = self._compute_ema(frac_series, self._slow_span)

            fast_emas[sym] = float(fast_ema[-1])
            slow_emas[sym] = float(slow_ema[-1])

            # Trend divergence score
            spread = fast_ema - slow_ema
            # Normalization scale using trailing spread standard deviation
            spread_scale = float(np.std(spread[-self._slow_span :]))
            spread_scale = max(spread_scale, 1e-5)

            # Standardized momentum score in [-1.0, 1.0] via hyperbolic tangent
            standardized_score = float(np.tanh(spread[-1] / spread_scale))
            raw_scores[sym] = standardized_score

            # Estimate trailing volatility
            vols[sym] = self._estimate_annualized_vol(closes)

        # 4. Volatility Parity Weight Allocation: w_i proportional to s_i / vol_i
        inv_vol_scores: dict[str, float] = {}
        for sym in self.monitored_symbols:
            s_i = raw_scores.get(sym, 0.0)
            vol_i = vols.get(sym, 0.20)
            inv_vol_scores[sym] = s_i / vol_i

        sum_abs_inv_vol = sum(abs(v) for v in inv_vol_scores.values())

        signals: dict[str, StrategySignal] = {}
        for sym in self.monitored_symbols:
            s_i = raw_scores.get(sym, 0.0)
            vol_i = vols.get(sym, 0.20)
            inv_val = inv_vol_scores.get(sym, 0.0)

            if sum_abs_inv_vol > 1e-8:
                target_w = (inv_val / sum_abs_inv_vol) * self._max_leverage
            else:
                target_w = 0.0

            # Directional assignment
            if target_w > 0.01:
                direction = SignalDirection.LONG
            elif target_w < -0.01:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.FLAT
                target_w = 0.0

            conviction = float(min(1.0, abs(s_i)))
            # Stop loss and take profit proportional to volatility
            daily_vol = vol_i / math.sqrt(252.0)
            stop_loss = round(float(2.0 * daily_vol), 4)
            take_profit = round(float(4.0 * daily_vol), 4)

            diag: dict[str, float | str | int] = {
                "d_order": self._d_order,
                "fast_ema": round(fast_emas.get(sym, 0.0), 5),
                "slow_ema": round(slow_emas.get(sym, 0.0), 5),
                "trend_score": round(s_i, 4),
                "annualized_vol": round(vol_i, 4),
            }

            signal = StrategySignal(
                symbol=sym,
                direction=direction,
                target_weight=round(target_w, 5),
                conviction=round(conviction, 4),
                target_horizon_bars=self._slow_span,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                diagnostics=diag,
            )
            signals[sym] = signal

        return signals
