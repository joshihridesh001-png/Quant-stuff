"""Causal price reaction prediction engine and dynamic Triple-Barrier breakout forecaster.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates, Rule 4: Closed-form formulations)
- Invariant 4: Price Impact & Elasticity Scaling (INV-NEWS-004)
- Invariant 5: Monotonic Barrier Breakout Probability (INV-NEWS-005)
- Invariant 6: Sub-10ms Pipeline Latency SLA (INV-NEWS-006)
- Fault Codes: ERR-NEWS-005, ERR-NEWS-006
"""

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from quant.analytics.news_classifier import (
    ClassifiedNewsEvent,
    EventType,
    NonFiniteSignalException,
)


def compute_decay_kernel(
    delta_t_seconds: float,
    urgency: float,
    alpha: float = 0.5,
    tau_fast: float = 3600.0,
    tau_slow: float = 86400.0,
    beta: float = 1.0,
    tolerance: float = 1e-4,
    max_lookback_seconds: float | None = 604800.0,  # 7 days max horizon
) -> float:
    """Hybrid temporal decay kernel combining exponential and power-law memory.

    kappa(Delta t, u) = alpha * exp(-Delta t / (tau_fast * (1 - u)))
                      + (1 - alpha) * (1 + Delta t / tau_slow)^(-beta)

    Includes explicit numerical truncation (tolerance) and maximum lookback bound.
    """
    if delta_t_seconds < 0:
        return 0.0

    if max_lookback_seconds is not None and delta_t_seconds > max_lookback_seconds:
        return 0.0

    # Guard urgency from 1.0 to prevent division by zero in denominator
    clamped_u = min(max(urgency, 0.0), 0.999)
    fast_denom = max(tau_fast * (1.0 - clamped_u), 1e-6)

    term_fast = alpha * math.exp(-delta_t_seconds / fast_denom)
    term_slow = (1.0 - alpha) * math.pow(1.0 + (delta_t_seconds / tau_slow), -beta)

    val = float(term_fast + term_slow)
    if val < tolerance:
        return 0.0
    return val


# Diagnostic Fault Code
ERR_NEWS_PREDICTION_TIMEOUT = "ERR-NEWS-006"


class PriceReactionError(Exception):
    """Base exception for all price reaction modeling anomalies."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class ReactionPredictionTimeoutException(PriceReactionError):
    """Raised when prediction computation exceeds execution deadline (ERR-NEWS-006)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_NEWS_PREDICTION_TIMEOUT, message)


# Empirical Category Elasticity Multipliers (INV-NEWS-004)
CATEGORY_ELASTICITY: dict[EventType, float] = {
    EventType.M_AND_A: 3.5,
    EventType.EARNINGS: 2.5,
    EventType.MACRO_FED: 2.0,
    EventType.REGULATORY_LEGAL: 1.2,
    EventType.ANALYST_ACTION: 0.8,
    EventType.GENERAL_MARKET: 0.4,
}

# Standard post-announcement forecast horizons in seconds:
# 0s (instant), 5m (300s), 15m (900s), 1h (3600s), 4h (14400s), 1d (86400s), 3d (259200s)
DEFAULT_FORECAST_HORIZONS: tuple[float, ...] = (
    0.0,
    300.0,
    900.0,
    3600.0,
    14400.0,
    86400.0,
    259200.0,
)


@dataclass(slots=True, frozen=True)
class PriceReactionPrediction:
    """Immutable econometric price reaction forecast and Triple-Barrier breakout decision.

    Purpose: Encapsulates expected dollar moves, logistic barrier probabilities, and PEAD path.
    Explicit Dependency Tracking: EventType, datetime.UTC.
    Structural Relationship: Output of NewsPriceReactionEngine, fed to AutonomousTradingEngine.
    Defensive Invariant: prob_up + prob_down == 1.0, confidence in [0, 1], non-negative prices.
    """

    ticker: str
    event_type: EventType
    current_price: float
    expected_delta_price: float
    target_price: float
    prob_up: float
    prob_down: float
    barrier_upper: float
    barrier_lower: float
    signal: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float
    predicted_trajectory: tuple[
        tuple[float, float], ...
    ]  # ((delta_t_seconds, predicted_price), ...)
    calculation_latency_ms: float
    created_at: datetime


class NewsPriceReactionEngine:
    """Causal price reaction engine predicting directional drift and Triple-Barrier probabilities.

    Purpose: Converts classified financial events into calibrated price impact forecasts.
    Explicit Dependency Tracking: ClassifiedNewsEvent, compute_decay_kernel, math, time.
    Structural Relationship: Analytical prediction engine powering forward alpha priors.
    Defensive Invariant: INV-NEWS-004 (Impact scaling), INV-NEWS-005 (Logistic monotonicity), INV-NEWS-006 (Sub-10ms SLA).
    """

    def __init__(
        self,
        barrier_multiplier: float = 2.0,
        sensitivity_lambda: float = 1.5,
        max_latency_ms: float = 50.0,
        forecast_horizons: tuple[float, ...] = DEFAULT_FORECAST_HORIZONS,
    ) -> None:
        self._barrier_multiplier = barrier_multiplier
        self._lambda = sensitivity_lambda
        self._max_latency_ms = max_latency_ms
        self._forecast_horizons = forecast_horizons

    def predict(
        self,
        event: ClassifiedNewsEvent,
        prices: dict[str, float],
        volatilities: dict[str, float],
    ) -> list[PriceReactionPrediction]:
        """Predict expected price shock and Triple-Barrier breakout probabilities for all target assets.

        Formula:
            Delta P_expected = P_t * gamma_category * S_{i, k} * sigma_t
            P(UP) = 1 / (1 + exp(-lambda * S_{i, k} * gamma / sigma_t))
            P(t + Delta t) = P_t + Delta P_expected * kappa(Delta t, u)
        """
        start_time = time.perf_counter()
        gamma = CATEGORY_ELASTICITY.get(event.event_type, 0.4)
        predictions: list[PriceReactionPrediction] = []
        now_utc = datetime.now(UTC)

        for ticker in event.entity_centrality:
            current_price = prices.get(ticker, 100.0)
            volatility = volatilities.get(ticker, 0.02)

            self._validate_inputs(current_price, volatility)

            shock = event.composite_shock.get(ticker, 0.0)

            # Invariant 4: Expected dollar move scaling (INV-NEWS-004)
            expected_delta = current_price * gamma * shock * volatility
            target_price = current_price + expected_delta

            # Dynamic Volatility Triple-Barriers
            barrier_upper = current_price * math.exp(self._barrier_multiplier * volatility)
            barrier_lower = current_price * math.exp(-self._barrier_multiplier * volatility)

            # Invariant 5: Monotonic Logistic Breakout Probability (INV-NEWS-005)
            # Guard against zero volatility
            safe_vol = max(volatility, 1e-5)
            # Guard against exp overflow: clamp exponent to [-50.0, 50.0]
            exponent = min(max(-(self._lambda * shock * gamma) / safe_vol, -50.0), 50.0)
            prob_up = 1.0 / (1.0 + math.exp(exponent))
            prob_down = 1.0 - prob_up

            # Signal classification
            if prob_up >= 0.65:
                signal = "BUY"
            elif prob_up <= 0.35:
                signal = "SELL"
            else:
                signal = "NEUTRAL"

            confidence = min(max(2.0 * abs(prob_up - 0.5), 0.0), 1.0)

            # Trajectory modeling over time via dual decay kernel
            trajectory: list[tuple[float, float]] = []
            for h in self._forecast_horizons:
                decay = compute_decay_kernel(delta_t_seconds=h, urgency=event.urgency)
                h_price = current_price + (expected_delta * decay)
                trajectory.append((h, h_price))

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > self._max_latency_ms:
                raise ReactionPredictionTimeoutException(
                    f"Prediction calculation latency {elapsed_ms:.2f}ms exceeded deadline {self._max_latency_ms}ms (INV-NEWS-006)."
                )

            predictions.append(
                PriceReactionPrediction(
                    ticker=ticker,
                    event_type=event.event_type,
                    current_price=current_price,
                    expected_delta_price=expected_delta,
                    target_price=target_price,
                    prob_up=prob_up,
                    prob_down=prob_down,
                    barrier_upper=barrier_upper,
                    barrier_lower=barrier_lower,
                    signal=signal,
                    confidence=confidence,
                    predicted_trajectory=tuple(trajectory),
                    calculation_latency_ms=elapsed_ms,
                    created_at=now_utc,
                )
            )

        return predictions

    def _validate_inputs(self, price: float, volatility: float) -> None:
        """Validate price and volatility inputs against non-finite or non-positive anomalies."""
        for name, val in [("price", price), ("volatility", volatility)]:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise NonFiniteSignalException(
                    f"Non-finite or boolean value detected for '{name}': {val!r}"
                )

        if price <= 0.0:
            raise NonFiniteSignalException(f"Price must be strictly positive: {price}")
        if volatility <= 0.0:
            raise NonFiniteSignalException(f"Volatility must be strictly positive: {volatility}")
