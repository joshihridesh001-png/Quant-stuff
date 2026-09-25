"""Volatility Breakout & Bollinger Range Squeeze Alpha Strategy.

Functional Purpose:
    Implements regime-adaptive volatility breakout trading combining Parkinson / Garman-Klass
    range volatility estimators with Bollinger Bandwidth squeeze detection. Identifies periods
    of extreme price compression (bandwidth < 10th percentile of lookback) and captures
    momentum expansion breakouts confirmed by abnormal trading volume.

Explicit Dependency Tracking:
    - numpy: Vectorized high-low range estimators, rolling standard deviations, and percentiles.
    - quant.analytics.strategies.base: BaseAlphaStrategy, StrategySignal, SignalDirection,
      BarHistoryWindow, StrategyContext, StrategyError.

Structural Relationship:
    Concrete volatility alpha strategy in Phase 14 library. Ingested by BacktestRunner and
    SwarmMetaStrategy.

Defensive Invariants:
    - INV-STRAT-001: Target weights strictly in [-1.0, 1.0].
    - INV-STRAT-002: Conviction strictly in [0.0, 1.0].
    - Non-negative volatility: Parkinson and Garman-Klass variances guaranteed non-negative.
    - Zero Forward Lookahead: Bandwidth percentile computed strictly on trailing lookback window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

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

ERR_STRAT_VOL_INVALID_PARAMS: Final[str] = "ERR-STRAT-040"


# ============================================================================
# Range Volatility Estimators (Closed-Form)
# ============================================================================


def compute_parkinson_volatility(
    highs: np.ndarray,
    lows: np.ndarray,
    annualized: bool = True,
) -> float:
    """Compute Parkinson (1980) High-Low Range Volatility.

    Formulation:
        sigma_P^2 = (1 / (4 * ln(2) * N)) * sum_{t=1}^N (ln(H_t / L_t))^2
    """
    n = len(highs)
    if n < 2 or len(lows) != n:
        return 0.0

    # Ensure strictly positive prices
    clamped_h = np.maximum(highs, 1e-4)
    clamped_l = np.maximum(lows, 1e-4)
    ratios = np.maximum(clamped_h / clamped_l, 1.0)
    log_hl = np.log(ratios)

    var = float(np.sum(log_hl**2) / (4.0 * math.log(2.0) * n))
    vol = math.sqrt(max(0.0, var))
    return vol * math.sqrt(252.0) if annualized else vol


def compute_garman_klass_volatility(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    annualized: bool = True,
) -> float:
    """Compute Garman-Klass (1980) OHLC Range Volatility.

    Formulation:
        sigma_GK^2 = (1 / N) * sum [ 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 ]
    """
    n = len(closes)
    if n < 2 or len(highs) != n or len(lows) != n or len(opens) != n:
        return 0.0

    clamped_h = np.maximum(highs, 1e-4)
    clamped_l = np.maximum(lows, 1e-4)
    clamped_o = np.maximum(opens, 1e-4)
    clamped_c = np.maximum(closes, 1e-4)

    hl_ratio = np.maximum(clamped_h / clamped_l, 1.0)
    co_ratio = clamped_c / clamped_o

    log_hl = np.log(hl_ratio)
    log_co = np.log(co_ratio)

    factor_hl = 0.5 * (log_hl**2)
    factor_co = (2.0 * math.log(2.0) - 1.0) * (log_co**2)

    var = float(np.mean(factor_hl - factor_co))
    vol = math.sqrt(max(0.0, var))
    return vol * math.sqrt(252.0) if annualized else vol


# ============================================================================
# Volatility Breakout & Squeeze Strategy
# ============================================================================


class VolatilityBreakoutStrategy(BaseAlphaStrategy):
    """Regime-adaptive volatility breakout strategy with Bollinger squeeze detection.

    Functional Purpose:
        Identifies coiling volatility consolidation via Bollinger Bandwidth percentiles.
        Emits directional breakout signals when price breaks through consolidation boundaries
        supported by abnormal trading volume and expanding Garman-Klass volatility.

    Explicit Dependency Tracking:
        compute_garman_klass_volatility, BaseAlphaStrategy, BarHistoryWindow.

    Defensive Invariants:
        - squeeze_percentile in (0.0, 50.0).
        - volume_expansion_mult >= 1.0.
        - min_warmup_bars >= lookback_window + bb_period.
    """

    def __init__(
        self,
        strategy_id: str,
        monitored_symbols: Sequence[str],
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        squeeze_lookback: int = 100,
        squeeze_percentile: float = 15.0,
        vol_expansion_threshold: float = 1.3,
        volume_confirmation_mult: float = 1.15,
        min_warmup_bars: int = 120,
        max_leverage: float = 1.0,
    ) -> None:
        """Initialize Volatility Breakout Strategy."""
        if bb_period < 5:
            raise StrategyError(
                f"bb_period ({bb_period}) must be >= 5", code=ERR_STRAT_VOL_INVALID_PARAMS
            )
        if not (0.0 < squeeze_percentile < 50.0):
            raise StrategyError(
                f"squeeze_percentile ({squeeze_percentile}) must be in (0.0, 50.0)",
                code=ERR_STRAT_VOL_INVALID_PARAMS,
            )
        if min_warmup_bars < squeeze_lookback + bb_period:
            raise StrategyError(
                f"min_warmup_bars ({min_warmup_bars}) must be >= squeeze_lookback + bb_period "
                f"({squeeze_lookback + bb_period})",
                code=ERR_STRAT_VOL_INVALID_PARAMS,
            )

        super().__init__(
            strategy_id=strategy_id,
            monitored_symbols=monitored_symbols,
            min_warmup_bars=min_warmup_bars,
            max_leverage=max_leverage,
        )

        self._bb_period = bb_period
        self._bb_std_dev = bb_std_dev
        self._squeeze_lookback = squeeze_lookback
        self._squeeze_pct = squeeze_percentile
        self._vol_expansion_thresh = vol_expansion_threshold
        self._volume_mult = volume_confirmation_mult

    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Evaluate volatility squeeze and breakout conditions across universe."""
        signals: dict[str, StrategySignal] = {}

        for sym in self.monitored_symbols:
            closes = history.closes(sym)
            highs = history.highs(sym)
            lows = history.lows(sym)
            opens = history.opens(sym)
            volumes = history.volumes(sym)

            n = len(closes)
            if n < self._min_warmup_bars:
                continue

            # 1. Compute rolling Bollinger Bands and Bandwidth history
            bandwidths = np.zeros(self._squeeze_lookback, dtype=np.float64)
            for i in range(self._squeeze_lookback):
                end_idx = n - self._squeeze_lookback + i + 1
                start_idx = end_idx - self._bb_period
                sub_closes = closes[start_idx:end_idx]
                mean_p = float(np.mean(sub_closes))
                std_p = float(np.std(sub_closes))
                bandwidths[i] = (2.0 * self._bb_std_dev * std_p) / max(mean_p, 1e-4)

            current_bandwidth = bandwidths[-1]
            squeeze_threshold = float(np.percentile(bandwidths, self._squeeze_pct))
            is_in_squeeze = current_bandwidth <= squeeze_threshold

            # 2. Compute current Bollinger Bands
            current_window = closes[-self._bb_period :]
            sma_current = float(np.mean(current_window))
            std_current = float(np.std(current_window))
            upper_band = sma_current + self._bb_std_dev * std_current
            lower_band = sma_current - self._bb_std_dev * std_current

            # 3. Garman-Klass Volatility Expansion Check
            recent_gk = compute_garman_klass_volatility(
                opens[-5:], highs[-5:], lows[-5:], closes[-5:]
            )
            baseline_gk = compute_garman_klass_volatility(
                opens[-self._bb_period :],
                highs[-self._bb_period :],
                lows[-self._bb_period :],
                closes[-self._bb_period :],
            )
            vol_expansion_ratio = recent_gk / max(baseline_gk, 1e-4)

            # 4. Volume Confirmation Check
            current_vol = volumes[-1] if len(volumes) > 0 else 1.0
            avg_volume = (
                float(np.mean(volumes[-self._bb_period :]))
                if len(volumes) >= self._bb_period
                else 1.0
            )
            volume_confirmed = current_vol >= (avg_volume * self._volume_mult)

            # 5. Breakout Evaluation
            current_price = closes[-1]
            prev_price = closes[-2]

            direction = SignalDirection.FLAT
            target_w = 0.0
            conviction = 0.0

            # Bullish Breakout: Price crosses above upper band with expanding volatility
            if (
                current_price > upper_band
                and prev_price <= upper_band
                and vol_expansion_ratio >= self._vol_expansion_thresh
            ):
                direction = SignalDirection.LONG
                conviction = 0.6 + (0.3 if volume_confirmed else 0.0)
                conviction = min(1.0, conviction)
                target_w = (self._max_leverage / len(self.monitored_symbols)) * conviction

            # Bearish Breakout: Price crosses below lower band with expanding volatility
            elif (
                current_price < lower_band
                and prev_price >= lower_band
                and vol_expansion_ratio >= self._vol_expansion_thresh
            ):
                direction = SignalDirection.SHORT
                conviction = 0.6 + (0.3 if volume_confirmed else 0.0)
                conviction = min(1.0, conviction)
                target_w = -(self._max_leverage / len(self.monitored_symbols)) * conviction

            # Mean-reversion exit if price returns inside bands after squeeze
            else:
                direction = SignalDirection.FLAT
                target_w = 0.0
                conviction = 0.0

            # Dynamic ATR for stops
            high_low_diff = highs[-self._bb_period :] - lows[-self._bb_period :]
            atr = float(np.mean(high_low_diff)) / current_price

            diag: dict[str, float | str | int] = {
                "bandwidth": round(current_bandwidth, 5),
                "squeeze_thresh": round(squeeze_threshold, 5),
                "is_squeeze": 1 if is_in_squeeze else 0,
                "vol_expansion": round(vol_expansion_ratio, 3),
                "volume_confirmed": 1 if volume_confirmed else 0,
                "gk_vol": round(recent_gk, 4),
            }

            signal = StrategySignal(
                symbol=sym,
                direction=direction,
                target_weight=round(target_w, 4),
                conviction=round(conviction, 4),
                target_horizon_bars=10,
                stop_loss_pct=round(max(0.01, atr * 2.0), 4),
                take_profit_pct=round(max(0.02, atr * 3.5), 4),
                diagnostics=diag,
            )
            signals[sym] = signal

        return signals
