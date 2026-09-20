"""Institutional Closed-Form Bayesian Pre-Trade Decision Gate.

Purpose:
    Evaluates algorithmic and discretionary entry orders through an institutional 5-dimensional
    Bayesian log-odds decision filter before orders reach execution gateways. Enforces microstructural
    toxicity guards, macro regime alignment, drawdown budget verification, and zero-latency fail-open
    invariants for risk-reducing exit orders.

Dependencies:
    - dataclasses: High-performance memory-compact slotted structures.
    - enum: Python 3.11+ StrEnum for zero-overhead diagnostic state representations.
    - math: Non-finite scalar verification (math.isfinite) and exp calculations.
    - time: High-resolution nanosecond timestamps (time.time_ns, time.perf_counter_ns).
    - typing: Static typing annotations, Final constants.

Structural Relationship:
    - Upstream: AutonomousTradingEngine, Manual Trading Terminal, and MCP Server.
    - Downstream: SmartOrderRouter (SOR), PreTradeRiskFirewall, and ExecutionGateway.

Invariants Enforced:
    - INV-GATE-001 (Fail-Open on Exits): Position-closing, stop-loss, and emergency liquidation
      orders unconditionally bypass the decision gate without blocking or latency overhead.
    - INV-GATE-002 (Bayesian Log-Odds Toxicity Formulation): Computes posterior adverse selection
      probability using closed-form log-odds aggregation across 5 microstructural and regime dimensions.
    - INV-GATE-003 (Sub-50us Latency SLA): Executes in-memory using pure vectorized math with zero
      network calls, locks, or blocking I/O in the execution path.
    - INV-GATE-004 (Strict Input Sanitization): Rejects NaN, Inf, and non-positive prices/quantities.
    - Rule 1: Four-tier line annotations on every class and method.
    - Rule 2: Diagnostic error codes ERR-GATE-001 through ERR-GATE-006.
    - Rule 4: Zero heuristic shortcuts, strict closed-form estimation without iterative solvers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_GATE_STALE_DATA: Final[str] = "ERR-GATE-001"
ERR_GATE_MACRO_REGIME_CONFLICT: Final[str] = "ERR-GATE-002"
ERR_GATE_DRAWDOWN_BUDGET_EXCEEDED: Final[str] = "ERR-GATE-003"
ERR_GATE_CONSECUTIVE_LOSS_BREACH: Final[str] = "ERR-GATE-004"
ERR_GATE_TOXIC_FLOW_DETECTED: Final[str] = "ERR-GATE-005"
ERR_GATE_NON_FINITE_INPUT: Final[str] = "ERR-GATE-006"


# ============================================================================
# Domain Exception Hierarchy
# ============================================================================


class PreTradeGateError(Exception):
    """Base exception for all pre-trade decision gate failures."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code: Final[str] = code
        self.message: Final[str] = message


class NonFiniteGateInputException(PreTradeGateError):
    """Raised when an order parameter contains NaN, Inf, or boolean scalar."""

    def __init__(self, message: str, code: str = ERR_GATE_NON_FINITE_INPUT) -> None:
        super().__init__(message, code)


# ============================================================================
# Domain Enumerations & Structures
# ============================================================================


class GateDecision(StrEnum):
    """Deterministic verdict emitted by the Pre-Trade Decision Gate."""

    PASS = "PASS"
    REJECT = "REJECT"
    BYPASS_EXIT = "BYPASS_EXIT"


class DimensionName(StrEnum):
    """Individual analytical evaluation dimensions."""

    DATA_FRESHNESS = "data_freshness"
    MACRO_REGIME = "macro_regime"
    MICROSTRUCTURE_TOXICITY = "microstructure_toxicity"
    MOMENTUM_ALIGNMENT = "momentum_alignment"
    CAPITAL_RISK_BUDGET = "capital_risk_budget"


@dataclass(slots=True, frozen=True)
class DimensionCheck:
    """Individual assessment record for a specific risk/regime dimension."""

    name: DimensionName
    passed: bool
    score: float  # Evidence weight contributing to log-odds
    details: str


@dataclass(slots=True, frozen=True)
class PreTradeDecisionRequest:
    """Input parameters representing a proposed order and market context."""

    order_id: str
    symbol: str
    action: str  # "BUY", "SELL", "CLOSE", "LIQUIDATE"
    quantity: float
    reference_price: float
    timestamp_ns: int
    is_position_exit: bool = False
    market_spread_bps: float = 5.0
    order_book_imbalance: float = 0.0  # Range [-1.0, 1.0]
    macro_yield_spread: float = 0.18  # T10Y2Y spread in percentage points
    trailing_volatility_pct: float = 1.2
    current_bar_return_pct: float = 0.05
    consecutive_losses: int = 0
    current_drawdown_pct: float = 0.0
    cvar_95_pct: float = 2.5
    data_age_seconds: float = 1.0

    def __post_init__(self) -> None:
        """Validate input domain boundaries and reject non-finite inputs."""
        # Functional Purpose: Protect the Bayesian engine against NaN propagation.
        # Explicit Dependency Tracking: math.isfinite.
        # Structural Relationship: First barrier before mathematical transformation.
        # Defensive Invariant: Quantities and prices must be finite, strictly positive.
        if (
            isinstance(self.quantity, bool)
            or not math.isfinite(self.quantity)
            or self.quantity <= 0.0
        ):
            raise NonFiniteGateInputException(
                f"quantity must be finite positive float, got {self.quantity}",
                ERR_GATE_NON_FINITE_INPUT,
            )
        if (
            isinstance(self.reference_price, bool)
            or not math.isfinite(self.reference_price)
            or self.reference_price <= 0.0
        ):
            raise NonFiniteGateInputException(
                f"reference_price must be finite positive float, got {self.reference_price}",
                ERR_GATE_NON_FINITE_INPUT,
            )
        for val_name, val in [
            ("market_spread_bps", self.market_spread_bps),
            ("order_book_imbalance", self.order_book_imbalance),
            ("macro_yield_spread", self.macro_yield_spread),
            ("trailing_volatility_pct", self.trailing_volatility_pct),
            ("current_bar_return_pct", self.current_bar_return_pct),
            ("current_drawdown_pct", self.current_drawdown_pct),
            ("cvar_95_pct", self.cvar_95_pct),
            ("data_age_seconds", self.data_age_seconds),
        ]:
            if isinstance(val, bool) or not math.isfinite(val):
                raise NonFiniteGateInputException(
                    f"{val_name} must be finite float, got {val}",
                    ERR_GATE_NON_FINITE_INPUT,
                )


@dataclass(slots=True, frozen=True)
class PreTradeDecisionResult:
    """Immutable audit record detailing the pre-trade decision."""

    decision_id: str
    order_id: str
    symbol: str
    action: str
    allowed: bool
    decision: GateDecision
    toxicity_probability: float
    confidence: float
    primary_code: str
    reason: str
    checks: list[DimensionCheck] = field(default_factory=list)
    latency_us: float = 0.0


# ============================================================================
# Core Pre-Trade Decision Gate Implementation
# ============================================================================


class PreTradeDecisionGate:
    """Institutional Pre-Trade Decision Gate evaluating multi-factor Bayesian log-odds."""

    # Prior baseline probability of toxic flow under normal market conditions: 10%
    PRIOR_TOXIC_PROB: Final[float] = 0.10
    PRIOR_LOG_ODDS: Final[float] = math.log(0.10 / 0.90)  # ~ -2.1972

    # Maximum acceptable posterior toxicity probability threshold (65%)
    MAX_TOXIC_PROBABILITY_THRESHOLD: Final[float] = 0.65

    # Maximum acceptable data staleness in seconds
    MAX_DATA_AGE_SECONDS: Final[float] = 120.0

    # Maximum consecutive losses allowed before entry gating
    MAX_CONSECUTIVE_LOSSES: Final[int] = 4

    # Maximum portfolio drawdown percentage allowed before entry freeze
    MAX_DRAWDOWN_THRESHOLD_PCT: Final[float] = 15.0

    __slots__ = ("_history", "_max_history")

    def __init__(self, max_history: int = 1000) -> None:
        """Initialize the decision gate with an in-memory rolling audit log."""
        # Functional Purpose: Initialize memory ring buffer for live telemetry inspection.
        # Explicit Dependency Tracking: max_history configuration.
        # Structural Relationship: Queried by REST endpoints and trading terminal HUD.
        # Defensive Invariant: max_history must be positive integer.
        self._max_history: int = max(10, max_history)
        self._history: list[PreTradeDecisionResult] = []

    @property
    def history(self) -> list[PreTradeDecisionResult]:
        """Return rolling history of evaluated decisions."""
        return list(self._history)

    def evaluate(self, request: PreTradeDecisionRequest) -> PreTradeDecisionResult:
        """Evaluate a proposed trade through the 5-dimensional Bayesian log-odds filter.

        Args:
            request: Slotted request containing order parameters and market/portfolio context.

        Returns:
            PreTradeDecisionResult: Immutable audit verdict.
        """
        # Functional Purpose: Execute sub-50us pre-trade safety filtering with zero blocking I/O.
        # Explicit Dependency Tracking: request attributes, math.exp.
        # Structural Relationship: Gatekeeper called prior to smart order routing.
        # Defensive Invariant: Exits bypass immediately (INV-GATE-001); non-finite inputs rejected.
        t_start = time.perf_counter_ns()
        decision_id = f"DEC-{request.timestamp_ns}-{request.order_id}"

        # --------------------------------------------------------------------
        # 1. INV-GATE-001: Fail-Open Invariant on Risk-Reducing Exits
        # --------------------------------------------------------------------
        action_clean = request.action.strip().upper()
        if request.is_position_exit or action_clean in {"CLOSE", "LIQUIDATE"}:
            latency_us = (time.perf_counter_ns() - t_start) / 1000.0
            result = PreTradeDecisionResult(
                decision_id=decision_id,
                order_id=request.order_id,
                symbol=request.symbol,
                action=request.action,
                allowed=True,
                decision=GateDecision.BYPASS_EXIT,
                toxicity_probability=0.0,
                confidence=1.0,
                primary_code="OK",
                reason="Risk-reducing exit order bypasses pre-trade filter (INV-GATE-001)",
                checks=[],
                latency_us=latency_us,
            )
            self._record(result)
            return result

        # --------------------------------------------------------------------
        # 2. INV-GATE-002: Evaluate 5 Discrete Dimensions
        # --------------------------------------------------------------------
        checks: list[DimensionCheck] = []
        log_odds_delta = 0.0
        primary_code = "OK"
        rejection_reason = "Order meets Bayesian toxicity and regime thresholds"

        # Dimension A: Data Freshness
        is_fresh = request.data_age_seconds <= self.MAX_DATA_AGE_SECONDS
        freshness_penalty = 0.0 if is_fresh else 2.5
        log_odds_delta += freshness_penalty
        if not is_fresh and primary_code == "OK":
            primary_code = ERR_GATE_STALE_DATA
            rejection_reason = f"Market data stale ({request.data_age_seconds:.1f}s > {self.MAX_DATA_AGE_SECONDS}s)"
        checks.append(
            DimensionCheck(
                name=DimensionName.DATA_FRESHNESS,
                passed=is_fresh,
                score=freshness_penalty,
                details=f"Data age {request.data_age_seconds:.1f}s",
            )
        )

        # Dimension B: Microstructure Toxicity & Order Book Imbalance (OBI)
        # Buying when OBI < -0.35 means ask queue is thick with toxic institutional selling pressure
        # Selling when OBI > 0.35 means bid queue is absorbing toxic supply
        is_buying = action_clean in {"BUY", "OPEN_LONG"}
        obi = request.order_book_imbalance
        micro_passed = True
        micro_score = 0.0

        if is_buying and obi < -0.35:
            micro_passed = False
            micro_score = 2.0 * abs(obi)
            if primary_code == "OK":
                primary_code = ERR_GATE_TOXIC_FLOW_DETECTED
                rejection_reason = (
                    f"Adverse order book imbalance (OBI={obi:.2f} indicates heavy selling pressure)"
                )
        elif not is_buying and obi > 0.35:
            micro_passed = False
            micro_score = 2.0 * abs(obi)
            if primary_code == "OK":
                primary_code = ERR_GATE_TOXIC_FLOW_DETECTED
                rejection_reason = (
                    f"Adverse order book imbalance (OBI={obi:.2f} indicates heavy bid absorption)"
                )
        elif request.market_spread_bps > 35.0:
            micro_passed = False
            micro_score = 1.5
            if primary_code == "OK":
                primary_code = ERR_GATE_TOXIC_FLOW_DETECTED
                rejection_reason = (
                    f"Excessive bid-ask spread width ({request.market_spread_bps:.1f} bps)"
                )

        log_odds_delta += micro_score
        checks.append(
            DimensionCheck(
                name=DimensionName.MICROSTRUCTURE_TOXICITY,
                passed=micro_passed,
                score=micro_score,
                details=f"OBI={obi:.2f}, Spread={request.market_spread_bps:.1f}bps",
            )
        )

        # Dimension C: Macro & Yield Curve Regime
        # Long equity entries in an inverted yield curve (T10Y2Y < -0.50) face heightened recession risk
        macro_passed = True
        macro_score = 0.0
        if is_buying and request.macro_yield_spread < -0.50:
            macro_passed = False
            macro_score = 1.5 * abs(request.macro_yield_spread)
            if primary_code == "OK":
                primary_code = ERR_GATE_MACRO_REGIME_CONFLICT
                rejection_reason = f"Severe yield curve inversion ({request.macro_yield_spread:.2f}% T10Y2Y spread)"

        log_odds_delta += macro_score
        checks.append(
            DimensionCheck(
                name=DimensionName.MACRO_REGIME,
                passed=macro_passed,
                score=macro_score,
                details=f"T10Y2Y yield spread={request.macro_yield_spread:.2f}%",
            )
        )

        # Dimension D: Technical Momentum & Volatility Shock
        momentum_passed = True
        momentum_score = 0.0
        # If current return exceeds 3x trailing volatility, mark as high-variance volatility shock
        if abs(request.current_bar_return_pct) > 3.0 * request.trailing_volatility_pct:
            momentum_passed = False
            momentum_score = 1.2
        # If buying into sharply negative momentum (> 2x vol down)
        if is_buying and request.current_bar_return_pct < -2.0 * request.trailing_volatility_pct:
            momentum_passed = False
            momentum_score += 1.0

        log_odds_delta += momentum_score
        checks.append(
            DimensionCheck(
                name=DimensionName.MOMENTUM_ALIGNMENT,
                passed=momentum_passed,
                score=momentum_score,
                details=f"Bar return={request.current_bar_return_pct:.2f}%, Vol={request.trailing_volatility_pct:.2f}%",
            )
        )

        # Dimension E: Capital Risk Budget & Loss Streak
        risk_passed = True
        risk_score = 0.0
        if request.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            risk_passed = False
            risk_score += 1.8 * (request.consecutive_losses - self.MAX_CONSECUTIVE_LOSSES + 1)
            if primary_code == "OK":
                primary_code = ERR_GATE_CONSECUTIVE_LOSS_BREACH
                rejection_reason = f"Consecutive loss streak ({request.consecutive_losses} losses >= limit {self.MAX_CONSECUTIVE_LOSSES})"

        if request.current_drawdown_pct >= self.MAX_DRAWDOWN_THRESHOLD_PCT:
            risk_passed = False
            risk_score += 2.5
            if primary_code == "OK":
                primary_code = ERR_GATE_DRAWDOWN_BUDGET_EXCEEDED
                rejection_reason = f"Portfolio drawdown budget exceeded ({request.current_drawdown_pct:.1f}% >= limit {self.MAX_DRAWDOWN_THRESHOLD_PCT}%)"

        log_odds_delta += risk_score
        checks.append(
            DimensionCheck(
                name=DimensionName.CAPITAL_RISK_BUDGET,
                passed=risk_passed,
                score=risk_score,
                details=f"Streak={request.consecutive_losses}, Drawdown={request.current_drawdown_pct:.1f}%, CVaR={request.cvar_95_pct:.1f}%",
            )
        )

        # --------------------------------------------------------------------
        # 3. Closed-Form Posterior Calculation & Hard Gate Tripwire
        # --------------------------------------------------------------------
        total_log_odds = self.PRIOR_LOG_ODDS + log_odds_delta
        # Logistic sigmoid: 1 / (1 + exp(-x))
        toxicity_prob = 1.0 / (1.0 + math.exp(-total_log_odds))

        # Rejection occurs if any individual dimension tripped or if cumulative toxicity exceeds threshold
        any_hard_trips = any(not c.passed for c in checks)
        excessive_toxicity = toxicity_prob >= self.MAX_TOXIC_PROBABILITY_THRESHOLD

        allowed = (not any_hard_trips) and (not excessive_toxicity)
        decision = GateDecision.PASS if allowed else GateDecision.REJECT

        confidence = abs(toxicity_prob - self.MAX_TOXIC_PROBABILITY_THRESHOLD) / max(
            self.MAX_TOXIC_PROBABILITY_THRESHOLD, 1.0 - self.MAX_TOXIC_PROBABILITY_THRESHOLD
        )
        confidence = min(1.0, max(0.0, confidence))

        if allowed:
            primary_code = "OK"
            rejection_reason = f"Order passed Bayesian gate (Toxicity={toxicity_prob * 100:.1f}%)"
        elif primary_code == "OK" and excessive_toxicity:
            primary_code = ERR_GATE_TOXIC_FLOW_DETECTED
            rejection_reason = f"Cumulative Bayesian toxicity exceeded threshold ({toxicity_prob * 100:.1f}% >= {self.MAX_TOXIC_PROBABILITY_THRESHOLD * 100:.1f}%)"

        latency_us = (time.perf_counter_ns() - t_start) / 1000.0

        result = PreTradeDecisionResult(
            decision_id=decision_id,
            order_id=request.order_id,
            symbol=request.symbol,
            action=request.action,
            allowed=allowed,
            decision=decision,
            toxicity_probability=round(toxicity_prob, 4),
            confidence=round(confidence, 4),
            primary_code=primary_code,
            reason=rejection_reason,
            checks=checks,
            latency_us=round(latency_us, 2),
        )
        self._record(result)
        return result

    def _record(self, result: PreTradeDecisionResult) -> None:
        """Append to rolling memory buffer maintaining max capacity."""
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)
