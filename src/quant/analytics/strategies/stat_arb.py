"""Statistical Arbitrage: Cointegration & Kalman Filter Dynamic Hedge-Ratio Pairs Trading.

Functional Purpose:
    Implements institutional statistical arbitrage across two cointegrated assets. Computes
    online dynamic hedge ratios beta_t via a recursive state-space Kalman Filter. Analyzes
    the resulting spread residuals using an Ornstein-Uhlenbeck (OU) mean-reversion process
    to determine the statistical half-life t_{1/2}. Generates dollar-neutral pairs allocations
    governed by z-score threshold bands with structural cointegration breakdown stops.

Explicit Dependency Tracking:
    - numpy: Matrix algebra, rolling statistics, and regression.
    - quant.analytics.strategies.base: BaseAlphaStrategy, StrategySignal, SignalDirection,
      BarHistoryWindow, StrategyContext, InsufficientWarmupError, InvalidSignalError.

Structural Relationship:
    Concrete alpha strategy within Phase 14 library. Ingested by BacktestRunner and
    SwarmMetaStrategy.

Defensive Invariants:
    - INV-STRAT-001: Target weights strictly in [-1.0, 1.0].
    - INV-STRAT-002: Conviction strictly in [0.0, 1.0].
    - INV-STRAT-004: Minimum warmup horizon >= 30 bars.
    - Zero Division: Regularized Kalman variance F_t >= 1e-8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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

ERR_STRAT_STATARB_INVALID_PAIR: Final[str] = "ERR-STRAT-010"
ERR_STRAT_STATARB_SINGULAR_KALMAN: Final[str] = "ERR-STRAT-011"


# ============================================================================
# Kalman Filter State Container
# ============================================================================


@dataclass(slots=True)
class KalmanFilterState:
    """State vector and error covariance for online dynamic pairs regression.

    Model:
        y_t = alpha_t + beta_t * x_t + v_t,   v_t ~ N(0, R)
        theta_t = theta_{t-1} + w_t,         w_t ~ N(0, Q)
        where theta_t = [alpha_t, beta_t]^T.
    """

    alpha: float
    beta: float
    P: np.ndarray  # 2x2 covariance matrix
    R: float  # Measurement noise variance
    Q: np.ndarray  # 2x2 process noise covariance


# ============================================================================
# Cointegration & Kalman Pairs Strategy
# ============================================================================


class KalmanPairsTradingStrategy(BaseAlphaStrategy):
    """Adaptive statistical arbitrage strategy using online recursive Kalman Filtering.

    Functional Purpose:
        Extracts dynamic time-varying hedge ratio beta_t between asset Y and asset X,
        evaluates spread mean-reversion speed theta and half-life t_{1/2}, and emits
        balanced long/short pair signals based on normalized residual z-scores.

    Explicit Dependency Tracking:
        KalmanFilterState, BaseAlphaStrategy, BarHistoryWindow.

    Defensive Invariants:
        - Exactly two assets monitored: (asset_y, asset_x).
        - Dollar-neutral gross leverage constraint: |w_y| + |w_x| <= max_leverage.
    """

    def __init__(
        self,
        strategy_id: str,
        asset_y: str,
        asset_x: str,
        min_warmup_bars: int = 30,
        lookback_window: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        max_leverage: float = 1.0,
        delta_process_noise: float = 1e-4,
        measurement_noise_r: float = 1e-3,
    ) -> None:
        """Initialize Kalman Pairs Trading Strategy.

        Functional Purpose:
            Configures strategy parameters, state priors, and trading thresholds.
        Explicit Dependency Tracking:
            BaseAlphaStrategy constructor.
        Defensive Invariants:
            asset_y != asset_x, z_entry > z_exit, z_stop > z_entry.
        """
        if asset_y == asset_x:
            raise StrategyError(
                f"asset_y and asset_x must be distinct symbols, got '{asset_y}'",
                code=ERR_STRAT_STATARB_INVALID_PAIR,
            )
        if z_entry <= z_exit:
            raise StrategyError(f"z_entry ({z_entry}) must exceed z_exit ({z_exit})")
        if z_stop <= z_entry:
            raise StrategyError(f"z_stop ({z_stop}) must exceed z_entry ({z_entry})")

        super().__init__(
            strategy_id=strategy_id,
            monitored_symbols=(asset_y, asset_x),
            min_warmup_bars=min_warmup_bars,
            max_leverage=max_leverage,
        )

        self._asset_y = asset_y
        self._asset_x = asset_x
        self._lookback_window = lookback_window
        self._z_entry = z_entry
        self._z_exit = z_exit
        self._z_stop = z_stop
        self._delta = delta_process_noise
        self._measurement_noise_r = measurement_noise_r
        self._current_position: int = 0

    def reset(self) -> None:
        """Reset internal strategy state and active position.

        Functional Purpose:
            Flattens tracked state and resets position hysteresis to flat.
        Explicit Dependency Tracking:
            None.
        Structural Relationship:
            Session and backtest lifecycle management.
        Defensive Invariants:
            self._current_position set strictly to 0.
        """
        self._current_position = 0

    @property
    def current_position(self) -> int:
        """Current position state: +1 (Long Y), -1 (Short Y), 0 (Flat)."""
        return self._current_position

    @property
    def asset_y(self) -> str:
        """Dependent target asset symbol (Y)."""
        return self._asset_y

    @property
    def asset_x(self) -> str:
        """Independent hedging asset symbol (X)."""
        return self._asset_x

    def _init_kalman_state(self) -> KalmanFilterState:
        """Initialize prior state and covariance matrices for recursive Kalman Filter."""
        # Process noise covariance Q = (delta / (1 - delta)) * I_2
        q_scale = self._delta / (1.0 - self._delta)
        q_mat = np.eye(2, dtype=np.float64) * q_scale

        return KalmanFilterState(
            alpha=0.0,
            beta=1.0,
            P=np.eye(2, dtype=np.float64) * 1.0,
            R=self._measurement_noise_r,
            Q=q_mat,
        )

    def _run_kalman_filter(
        self, y_prices: np.ndarray, x_prices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Run recursive online Kalman Filter over historical observation vectors.

        Returns:
            (spread_history, beta_history, current_alpha, current_beta)
        """
        n = len(y_prices)
        state = self._init_kalman_state()

        spreads = np.zeros(n, dtype=np.float64)
        betas = np.zeros(n, dtype=np.float64)

        for t in range(n):
            y_t = y_prices[t]
            x_t = x_prices[t]

            # 1. State prediction: theta_{t|t-1} = theta_{t-1|t-1}, P_{t|t-1} = P_{t-1|t-1} + Q
            theta_pred = np.array([state.alpha, state.beta], dtype=np.float64)
            p_pred = state.P + state.Q

            # 2. Measurement prediction: y_hat = H * theta_pred = alpha + beta * x
            h_t = np.array([1.0, x_t], dtype=np.float64)
            y_pred = float(np.dot(h_t, theta_pred))
            err_t = y_t - y_pred  # Spread innovation residual

            # 3. Innovation covariance: F_t = H * P_pred * H^T + R
            f_t = float(np.dot(h_t, np.dot(p_pred, h_t))) + state.R
            f_t = max(f_t, 1e-8)  # Regularization guard

            # 4. Kalman Gain: K_t = P_pred * H^T / F_t
            k_t = np.dot(p_pred, h_t) / f_t

            # 5. State update: theta_{t|t} = theta_pred + K_t * err_t
            theta_upd = theta_pred + k_t * err_t
            # Joseph form or standard covariance update: P_{t|t} = (I - K * H) * P_pred
            i_kh = np.eye(2, dtype=np.float64) - np.outer(k_t, h_t)
            p_upd = np.dot(i_kh, p_pred)

            state.alpha = float(theta_upd[0])
            state.beta = float(theta_upd[1])
            state.P = p_upd

            spreads[t] = err_t
            betas[t] = state.beta

        return spreads, betas, state.alpha, state.beta

    def _estimate_half_life(self, spreads: np.ndarray) -> float:
        """Estimate Ornstein-Uhlenbeck (OU) mean reversion half-life via regression on spread diffs.

        Model:
            Delta e_t = theta * (mu - e_{t-1}) + epsilon
            Regression: Delta e_t = a + b * e_{t-1}
            Half-life = -ln(2) / b
        """
        if len(spreads) < 10:
            return float("nan")

        delta_e = np.diff(spreads)
        lag_e = spreads[:-1]

        # Regress delta_e on lag_e with intercept
        # Ordinary Least Squares: b = Cov(delta_e, lag_e) / Var(lag_e)
        var_lag = np.var(lag_e)
        if var_lag < 1e-12:
            return float("nan")

        cov_val = np.cov(delta_e, lag_e)[0, 1]
        b_coef = cov_val / var_lag

        if b_coef >= -1e-6:
            # Not mean-reverting (drift or explosive)
            return float("inf")

        half_life = -math.log(2.0) / float(b_coef)
        return float(max(0.1, half_life))

    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Compute statistical arbitrage signals for the monitored pair."""
        y_closes = history.closes(self._asset_y)
        x_closes = history.closes(self._asset_x)

        n_bars = min(len(y_closes), len(x_closes))
        y_series = y_closes[-n_bars:]
        x_series = x_closes[-n_bars:]

        # Run Kalman Filter to get spread residuals and latest hedge ratio
        spreads, betas, current_alpha, current_beta = self._run_kalman_filter(y_series, x_series)

        # Focus on trailing lookback window for z-score normalization
        lookback = min(self._lookback_window, len(spreads))
        window_spreads = spreads[-lookback:]

        mean_spread = float(np.mean(window_spreads))
        std_spread = float(np.std(window_spreads))
        std_spread = max(std_spread, 1e-6)

        current_spread = float(spreads[-1])
        z_score = (current_spread - mean_spread) / std_spread

        # Estimate half-life of mean-reversion
        half_life = self._estimate_half_life(window_spreads)

        # Sizing: Target Dollar-Neutral Allocations
        # Gross weight allocated according to hedge ratio |w_y| + |w_x| <= max_leverage
        # Sizing ratio: $1 of Y is hedged with beta of X scaled by price ratio P_x / P_y
        # Weight allocation denominator: (1.0 + dollar_beta)
        p_y = float(y_series[-1])
        p_x = float(x_series[-1])
        price_ratio = (p_x / max(p_y, 1e-6)) if p_y > 0 else 1.0
        dollar_beta = max(0.01, abs(current_beta) * price_ratio)
        norm_factor = 1.0 + dollar_beta

        y_weight_unit = 1.0 / norm_factor
        x_weight_unit = dollar_beta / norm_factor

        # Determine signal direction based on z-score bands
        direction_y = SignalDirection.FLAT
        direction_x = SignalDirection.FLAT
        target_w_y = 0.0
        target_w_x = 0.0
        conviction = 0.0

        is_structural_break = abs(z_score) >= self._z_stop

        if is_structural_break:
            # Cointegration breakdown: emergency flat exit
            self._current_position = 0
            direction_y = SignalDirection.FLAT
            direction_x = SignalDirection.FLAT
            target_w_y = 0.0
            target_w_x = 0.0
            conviction = 0.0
        elif z_score >= self._z_entry:
            # Spread is rich: Short Y, Long X (if beta > 0)
            self._current_position = -1
            direction_y = SignalDirection.SHORT
            target_w_y = -y_weight_unit * self._max_leverage

            if current_beta >= 0:
                direction_x = SignalDirection.LONG
                target_w_x = x_weight_unit * self._max_leverage
            else:
                direction_x = SignalDirection.SHORT
                target_w_x = -x_weight_unit * self._max_leverage

            # Conviction scaled by z-score deviation and half-life viability
            z_excess = min(1.0, (z_score - self._z_entry) / 2.0)
            conviction = 0.5 + 0.5 * z_excess
        elif z_score <= -self._z_entry:
            # Spread is cheap: Long Y, Short X (if beta > 0)
            self._current_position = 1
            direction_y = SignalDirection.LONG
            target_w_y = y_weight_unit * self._max_leverage

            if current_beta >= 0:
                direction_x = SignalDirection.SHORT
                target_w_x = -x_weight_unit * self._max_leverage
            else:
                direction_x = SignalDirection.LONG
                target_w_x = x_weight_unit * self._max_leverage

            z_excess = min(1.0, (abs(z_score) - self._z_entry) / 2.0)
            conviction = 0.5 + 0.5 * z_excess
        elif abs(z_score) <= self._z_exit:
            # Spread has mean-reverted: exit pair trade cleanly
            self._current_position = 0
            direction_y = SignalDirection.FLAT
            direction_x = SignalDirection.FLAT
            target_w_y = 0.0
            target_w_x = 0.0
            conviction = 0.0
        else:
            # In hysteresis band (z_exit < |z| < z_entry): maintain existing position
            if self._current_position == -1:
                direction_y = SignalDirection.SHORT
                target_w_y = -y_weight_unit * self._max_leverage
                if current_beta >= 0:
                    direction_x = SignalDirection.LONG
                    target_w_x = x_weight_unit * self._max_leverage
                else:
                    direction_x = SignalDirection.SHORT
                    target_w_x = -x_weight_unit * self._max_leverage
                conviction = 0.5
            elif self._current_position == 1:
                direction_y = SignalDirection.LONG
                target_w_y = y_weight_unit * self._max_leverage
                if current_beta >= 0:
                    direction_x = SignalDirection.SHORT
                    target_w_x = -x_weight_unit * self._max_leverage
                else:
                    direction_x = SignalDirection.LONG
                    target_w_x = x_weight_unit * self._max_leverage
                conviction = 0.5
            else:
                direction_y = SignalDirection.FLAT
                direction_x = SignalDirection.FLAT
                target_w_y = 0.0
                target_w_x = 0.0
                conviction = 0.0

        # Penalize conviction if half-life is non-reverting (> 45 bars)
        if math.isinf(half_life) or math.isnan(half_life) or half_life > 45.0:
            conviction *= 0.5

        target_horizon = max(5, int(half_life * 2)) if math.isfinite(half_life) else 10

        diag_y: dict[str, float | str | int] = {
            "pair_role": "target_y",
            "z_score": round(z_score, 3),
            "beta": round(current_beta, 4),
            "alpha": round(current_alpha, 4),
            "half_life": round(half_life, 1) if math.isfinite(half_life) else -1.0,
            "structural_break": 1 if is_structural_break else 0,
        }
        diag_x: dict[str, float | str | int] = {
            "pair_role": "hedge_x",
            "z_score": round(z_score, 3),
            "beta": round(current_beta, 4),
            "alpha": round(current_alpha, 4),
            "half_life": round(half_life, 1) if math.isfinite(half_life) else -1.0,
            "structural_break": 1 if is_structural_break else 0,
        }

        signal_y = StrategySignal(
            symbol=self._asset_y,
            direction=direction_y,
            target_weight=target_w_y,
            conviction=conviction,
            target_horizon_bars=target_horizon,
            stop_loss_pct=0.03,
            take_profit_pct=0.04,
            diagnostics=diag_y,
        )

        signal_x = StrategySignal(
            symbol=self._asset_x,
            direction=direction_x,
            target_weight=target_w_x,
            conviction=conviction,
            target_horizon_bars=target_horizon,
            stop_loss_pct=0.03,
            take_profit_pct=0.04,
            diagnostics=diag_x,
        )

        return {
            self._asset_y: signal_y,
            self._asset_x: signal_x,
        }
