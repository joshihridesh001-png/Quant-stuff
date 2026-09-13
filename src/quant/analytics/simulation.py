"""End-to-End Live Replay Simulator & Institutional Benchmarking Subsystem.

Purpose:
    Provides institutional-grade live replay backtesting, modular simulation state machines,
    non-linear market friction modeling, causal portfolio accounting, and statistical
    significance benchmarking for multi-asset quantitative trading strategies.

Dependencies:
    - math: Scalar finiteness checking and mathematical functions.
    - dataclasses: High-performance frozen value objects with slots.
    - typing: Static typing annotations, Final constants, runtime_checkable, and Protocol.

Structural Relationship:
    - Coupler of:
        1. Regime-Conditioned Dynamic Model Averaging (RD-DMA) (ensemble.py)
        2. Epistemic Disagreement Entropy & Circuit Breaker Overlays (circuit_breakers.py)
        3. Semi-Parametric Peaks-Over-Threshold EVT Tail Risk (tail_risk.py)
        4. Unified Convex Execution Sizer (execution_sizing.py)
        5. Kyle-Obizhaeva 3/2-power Market Impact (market_impact.py)
        6. Deflated Sharpe Ratio & Statistical Significance Certification (deflated_sharpe.py)
    - Emits: SimulationConfig, BarExecutionRecord, BenchmarkComparison, BenchmarkAuditReport,
      SimulationListener protocol, ExecutionCostModel, PortfolioLedger, BenchmarkAuditor,
      and ReplayEngine.

Invariants Enforced:
    - INV-SIM-001 (Zero-Lookahead Causality): Strict temporal separation; decision at bar t
      uses information filtration F_{t-1}; execution return realizes over [t-1, t].
    - INV-SIM-002 (Conservation of Capital): Total portfolio equity W_t == cash_t + sum nu_{i, t}
      and W_t - W_{t-1} == PnL_t^{net} everywhere within tolerance 1e-6.
    - INV-SIM-003 (Non-Negative Execution Friction): Total transaction friction C(Delta nu_t) >= 0.0.
    - INV-SIM-004 (Risk Budget & Leverage Adherence): Gross leverage ||nu_t||_1 / W_t <= L_max + eps;
      emergency HALT state enforces nu_t == 0.
    - INV-SIM-005 (Statistical Rigor & Non-Finite Protection): Strict rejection of NaN, Inf,
      and non-finite numeric scalars at all entity instantiation boundaries.
    - INV-SIM-006 (Execution Latency SLA): Deterministic runtime execution; zero iterative solvers
      in the synchronous simulation hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_SIM_LOOKAHEAD_VIOLATION: Final[str] = "ERR-SIM-001"
ERR_SIM_NON_FINITE_INPUT: Final[str] = "ERR-SIM-002"
ERR_SIM_CAPITAL_RUIN: Final[str] = "ERR-SIM-003"
ERR_SIM_NEGATIVE_FRICTION: Final[str] = "ERR-SIM-004"
ERR_SIM_STARVATION: Final[str] = "ERR-SIM-005"
ERR_SIM_DIMENSION_MISMATCH: Final[str] = "ERR-SIM-006"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Exception Protocol)
# ============================================================================


class SimulationError(Exception):
    """Base exception for all live replay simulation and institutional benchmarking errors."""

    def __init__(self, message: str, code: str = "ERR-SIM-000") -> None:
        # Functional Purpose: Initialize base simulation exception with message and diagnostic code.
        # Explicit Dependency Tracking: Exception base class.
        # Structural Relationship: Root of the simulation exception hierarchy.
        # Defensive Invariant: Diagnostic code must be a non-empty string identifier.
        super().__init__(message)
        self.message: str = message
        self.code: str = code


class LookaheadViolationException(SimulationError):
    """Raised when contemporaneous or future information leaks into bar decision logic (INV-SIM-001)."""

    def __init__(self, message: str, code: str = ERR_SIM_LOOKAHEAD_VIOLATION) -> None:
        # Functional Purpose: Signal causal information barrier breach or lookahead temporal leak.
        # Explicit Dependency Tracking: ERR_SIM_LOOKAHEAD_VIOLATION fault vector.
        # Structural Relationship: Thrown by ReplayEngine or data feed validators when t leaks into t-1.
        # Defensive Invariant: code defaults to ERR-SIM-001.
        super().__init__(message=message, code=code)


class DegenerateSimulationException(SimulationError):
    """Raised on non-finite inputs, capital ruin, sample starvation, or structural degradation."""

    def __init__(self, message: str, code: str = ERR_SIM_NON_FINITE_INPUT) -> None:
        # Functional Purpose: Signal mathematical degeneracy, NaN/Inf poisoning, ruin, or starvation.
        # Explicit Dependency Tracking: ERR_SIM_NON_FINITE_INPUT fault vector.
        # Structural Relationship: Thrown during validation, accounting ruin, or insufficient observations.
        # Defensive Invariant: code defaults to ERR-SIM-002.
        super().__init__(message=message, code=code)


class InfeasibleSimulationException(SimulationError):
    """Raised on infeasible execution conditions, such as negative friction (INV-SIM-003)."""

    def __init__(self, message: str, code: str = ERR_SIM_NEGATIVE_FRICTION) -> None:
        # Functional Purpose: Signal physical or mathematical impossibility in trade execution.
        # Explicit Dependency Tracking: ERR_SIM_NEGATIVE_FRICTION fault vector.
        # Structural Relationship: Thrown by ExecutionCostModel or PortfolioLedger when costs < 0.
        # Defensive Invariant: code defaults to ERR-SIM-004.
        super().__init__(message=message, code=code)


# ============================================================================
# Domain Value Objects & Entities (Rule 1 & Rule 4: Best-of-the-Best Contracts)
# ============================================================================


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable simulation configuration parameters and market friction settings.

    Attributes:
        initial_capital: Initial portfolio cash endowment W_0 > 0.0 (default 1,000,000.0).
        risk_free_rate: Annualized risk-free rate r_f >= 0.0 (default 0.02).
        fee_bps: Exchange and clearing transaction fee in basis points >= 0.0 (default 2.0).
        spread_bps: Average bid-ask spread in basis points >= 0.0 (default 1.0).
        impact_coefficient: Kyle-Obizhaeva non-linear market impact coefficient >= 0.0 (default 0.10).
        max_leverage: Maximum allowed gross leverage ceiling L_max > 0.0 (default 1.0).
        mdd_budget: Maximum drawdown tolerance budget fraction in (0.0, 1.0] (default 0.20).
        confidence_level: Downside tail risk confidence level alpha in (0.50, 1.0) (default 0.95).
        annualization_factor: Number of trading bars per calendar year > 0 (default 252).
        num_trials: Number of trials for DSR multiple-testing adjustment >= 1 (default 100).
    """

    initial_capital: float = 1_000_000.0
    risk_free_rate: float = 0.02
    fee_bps: float = 2.0
    spread_bps: float = 1.0
    impact_coefficient: float = 0.10
    max_leverage: float = 1.0
    mdd_budget: float = 0.20
    confidence_level: float = 0.95
    annualization_factor: int = 252
    num_trials: int = 100

    def __post_init__(self) -> None:
        """Validate configuration domain bounds, types, and mathematical invariants upon instantiation."""
        # Functional Purpose: Enforce strict boundary invariants and non-finite protection on simulation configuration.
        # Explicit Dependency Tracking: math.isfinite, DegenerateSimulationException, InfeasibleSimulationException.
        # Structural Relationship: Configures ReplayEngine, ExecutionCostModel, and BenchmarkAuditor.
        # Defensive Invariant: All floats must be strictly finite; non-negative friction; valid probability ranges.

        # 1. Validate float scalar types and finiteness
        float_fields = (
            "initial_capital",
            "risk_free_rate",
            "fee_bps",
            "spread_bps",
            "impact_coefficient",
            "max_leverage",
            "mdd_budget",
            "confidence_level",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise DegenerateSimulationException(
                    f"SimulationConfig {field_name} must be numeric, got {type(val).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not math.isfinite(val):
                raise DegenerateSimulationException(
                    f"SimulationConfig {field_name} must be a finite float, got {val}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 2. Validate integer scalar types
        int_fields = ("annualization_factor", "num_trials")
        for field_name in int_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, int) and not isinstance(val, bool)):
                raise DegenerateSimulationException(
                    f"SimulationConfig {field_name} must be an integer, got {type(val).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 3. Capital domain bounds (W_0 > 0.0)
        if self.initial_capital <= 0.0:
            raise DegenerateSimulationException(
                f"SimulationConfig initial_capital must be strictly positive, got {self.initial_capital}",
                code=ERR_SIM_CAPITAL_RUIN,
            )

        # 4. Risk-free rate bounds (r_f >= 0.0)
        if self.risk_free_rate < 0.0:
            raise DegenerateSimulationException(
                f"SimulationConfig risk_free_rate must be non-negative, got {self.risk_free_rate}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 5. Friction parameters non-negativity (INV-SIM-003)
        for friction_field in ("fee_bps", "spread_bps", "impact_coefficient"):
            friction_val = getattr(self, friction_field)
            if friction_val < 0.0:
                raise InfeasibleSimulationException(
                    f"SimulationConfig {friction_field} must be non-negative (INV-SIM-003), got {friction_val}",
                    code=ERR_SIM_NEGATIVE_FRICTION,
                )

        # 6. Leverage ceiling bounds (L_max > 0.0)
        if self.max_leverage <= 0.0:
            raise DegenerateSimulationException(
                f"SimulationConfig max_leverage must be strictly positive, got {self.max_leverage}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 7. Maximum drawdown budget bounds (mdd_budget in (0.0, 1.0])
        if not (0.0 < self.mdd_budget <= 1.0):
            raise DegenerateSimulationException(
                f"SimulationConfig mdd_budget must be in (0.0, 1.0], got {self.mdd_budget}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 8. Downside tail confidence level bounds (alpha in (0.50, 1.0))
        if not (0.50 < self.confidence_level < 1.0):
            raise DegenerateSimulationException(
                f"SimulationConfig confidence_level must be in (0.50, 1.0), got {self.confidence_level}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 9. Annualization factor bounds (> 0)
        if self.annualization_factor <= 0:
            raise DegenerateSimulationException(
                f"SimulationConfig annualization_factor must be strictly positive, got {self.annualization_factor}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 10. Number of trials bounds (>= 1)
        if self.num_trials < 1:
            raise DegenerateSimulationException(
                f"SimulationConfig num_trials must be at least 1, got {self.num_trials}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )


@dataclass(frozen=True, slots=True)
class BarExecutionRecord:
    """Immutable record of single-bar execution, causal PnL accounting, and portfolio state.

    Attributes:
        step_index: Simulation chronological discrete bar sequence index t >= 0.
        timestamp: Epoch timestamp in nanoseconds >= 0.
        gross_pnl: Gross mark-to-market trading profit/loss before friction.
        net_pnl: Net mark-to-market profit/loss after all execution friction costs.
        friction_cost: Total execution friction incurred on bar t >= 0.0 (INV-SIM-003).
        portfolio_equity: Total marked portfolio net wealth W_t.
        cash_balance: Available cash liquidity balance cash_t.
        effective_leverage: Realized gross portfolio leverage ||nu_t||_1 / W_t >= 0.0.
        drawdown: Peak-to-trough equity drawdown fraction DD_t in [0.0, 1.0].
        circuit_breaker_tier: Active discrete circuit breaker risk tier label.
        circuit_breaker_haircut: Continuous logistic haircut multiplier kappa_t in [0.0, 1.0].
        target_allocations: Continuous target portfolio dollar allocations nu*.
        discretized_allocations: Exchange lot-discretized executed dollar allocations nu~.
    """

    step_index: int
    timestamp: int
    gross_pnl: float
    net_pnl: float
    friction_cost: float
    portfolio_equity: float
    cash_balance: float
    effective_leverage: float
    drawdown: float
    circuit_breaker_tier: str
    circuit_breaker_haircut: float
    target_allocations: tuple[float, ...]
    discretized_allocations: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate single-bar execution record invariants, dimensionalities, and bounds."""
        # Functional Purpose: Guard portfolio accounting records against non-finite values, negative costs, or crossed state.
        # Explicit Dependency Tracking: math.isfinite, InfeasibleSimulationException, DegenerateSimulationException.
        # Structural Relationship: Emitted per bar by PortfolioLedger, ingested by BenchmarkAuditor and SimulationListener.
        # Defensive Invariant: Non-negative friction; drawdown in [0.0, 1.0]; haircut in [0.0, 1.0]; matched allocation dimensions.

        # 1. Validate discrete indices
        if not (isinstance(self.step_index, int) and not isinstance(self.step_index, bool)):
            raise DegenerateSimulationException(
                f"BarExecutionRecord step_index must be an integer, got {type(self.step_index).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if self.step_index < 0:
            raise DegenerateSimulationException(
                f"BarExecutionRecord step_index must be non-negative, got {self.step_index}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        if not (isinstance(self.timestamp, int) and not isinstance(self.timestamp, bool)):
            raise DegenerateSimulationException(
                f"BarExecutionRecord timestamp must be an integer, got {type(self.timestamp).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if self.timestamp < 0:
            raise DegenerateSimulationException(
                f"BarExecutionRecord timestamp must be non-negative, got {self.timestamp}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 2. Validate scalar float fields and finiteness
        float_fields = (
            "gross_pnl",
            "net_pnl",
            "friction_cost",
            "portfolio_equity",
            "cash_balance",
            "effective_leverage",
            "drawdown",
            "circuit_breaker_haircut",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise DegenerateSimulationException(
                    f"BarExecutionRecord {field_name} must be numeric, got {type(val).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not math.isfinite(val):
                raise DegenerateSimulationException(
                    f"BarExecutionRecord {field_name} must be a finite float, got {val}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 3. Friction cost non-negativity (INV-SIM-003)
        if self.friction_cost < 0.0:
            raise InfeasibleSimulationException(
                f"BarExecutionRecord friction_cost must be non-negative (INV-SIM-003), got {self.friction_cost}",
                code=ERR_SIM_NEGATIVE_FRICTION,
            )

        # 4. Effective leverage bounds (>= 0.0)
        if self.effective_leverage < 0.0:
            raise DegenerateSimulationException(
                f"BarExecutionRecord effective_leverage must be non-negative, got {self.effective_leverage}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 5. Drawdown fraction bounds ([0.0, 1.0])
        if not (0.0 <= self.drawdown <= 1.0):
            raise DegenerateSimulationException(
                f"BarExecutionRecord drawdown must be in [0.0, 1.0], got {self.drawdown}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 6. Circuit breaker tier validation
        if not (
            isinstance(self.circuit_breaker_tier, str)
            and len(self.circuit_breaker_tier.strip()) > 0
        ):
            raise DegenerateSimulationException(
                f"BarExecutionRecord circuit_breaker_tier must be a non-empty string, got {self.circuit_breaker_tier!r}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 7. Circuit breaker haircut multiplier bounds ([0.0, 1.0])
        if not (0.0 <= self.circuit_breaker_haircut <= 1.0):
            raise DegenerateSimulationException(
                f"BarExecutionRecord circuit_breaker_haircut must be in [0.0, 1.0], got {self.circuit_breaker_haircut}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 8. Allocation vector finiteness and dimensional consistency
        if not isinstance(self.target_allocations, tuple):
            raise DegenerateSimulationException(
                f"BarExecutionRecord target_allocations must be a tuple, got {type(self.target_allocations).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not isinstance(self.discretized_allocations, tuple):
            raise DegenerateSimulationException(
                f"BarExecutionRecord discretized_allocations must be a tuple, got {type(self.discretized_allocations).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        for x in self.target_allocations:
            if not (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)):
                raise DegenerateSimulationException(
                    f"BarExecutionRecord target_allocations elements must be finite numbers, got {x}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
        for x in self.discretized_allocations:
            if not (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)):
                raise DegenerateSimulationException(
                    f"BarExecutionRecord discretized_allocations elements must be finite numbers, got {x}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        if len(self.target_allocations) != len(self.discretized_allocations):
            raise DegenerateSimulationException(
                f"BarExecutionRecord allocation dimension mismatch: target={len(self.target_allocations)} vs discretized={len(self.discretized_allocations)}",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Performance metrics and tracking statistics for an institutional benchmark baseline.

    Attributes:
        name: Benchmark baseline identifier (e.g. 'SPY_EqualWeight', 'RiskParity', 'Cash').
        total_return: Cumulative cumulative geometric return over simulation horizon.
        annualized_return: Compounded Annual Growth Rate (CAGR) of benchmark.
        annualized_volatility: Annualized sample volatility sigma_ann >= 0.0.
        sharpe_ratio: Annualized Sharpe ratio of benchmark.
        max_drawdown: Maximum peak-to-trough drawdown in [0.0, 1.0].
        alpha: Annualized Jensen's alpha relative to benchmark baseline.
        beta: Systematic market risk sensitivity coefficient beta.
        tracking_error: Annualized tracking error volatility >= 0.0.
        information_ratio: Annualized Information Ratio (alpha / tracking_error).
    """

    name: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float

    def __post_init__(self) -> None:
        """Validate benchmark comparison metrics, bounds, and string identifiers."""
        # Functional Purpose: Guarantee integrity of institutional benchmark comparison statistics.
        # Explicit Dependency Tracking: math.isfinite, DegenerateSimulationException.
        # Structural Relationship: Stored within BenchmarkAuditReport; generated by BenchmarkAuditor.
        # Defensive Invariant: Non-empty name; non-negative volatility and tracking error; max drawdown in [0.0, 1.0].

        # 1. Validate identifier
        if not (isinstance(self.name, str) and len(self.name.strip()) > 0):
            raise DegenerateSimulationException(
                f"BenchmarkComparison name must be a non-empty string, got {self.name!r}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 2. Validate numeric scalars
        float_fields = (
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "alpha",
            "beta",
            "tracking_error",
            "information_ratio",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise DegenerateSimulationException(
                    f"BenchmarkComparison {field_name} must be numeric, got {type(val).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not math.isfinite(val):
                raise DegenerateSimulationException(
                    f"BenchmarkComparison {field_name} must be a finite float, got {val}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 3. Volatility non-negativity
        if self.annualized_volatility < 0.0:
            raise DegenerateSimulationException(
                f"BenchmarkComparison annualized_volatility must be non-negative, got {self.annualized_volatility}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 4. Max drawdown bounds ([0.0, 1.0])
        if not (0.0 <= self.max_drawdown <= 1.0):
            raise DegenerateSimulationException(
                f"BenchmarkComparison max_drawdown must be in [0.0, 1.0], got {self.max_drawdown}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 5. Tracking error non-negativity
        if self.tracking_error < 0.0:
            raise DegenerateSimulationException(
                f"BenchmarkComparison tracking_error must be non-negative, got {self.tracking_error}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )


@dataclass(frozen=True, slots=True)
class BenchmarkAuditReport:
    """Comprehensive institutional tear sheet and statistical significance audit.

    Attributes:
        initial_capital: Initial portfolio wealth W_0 > 0.0.
        final_equity: Final mark-to-market portfolio equity W_T.
        total_return: Net cumulative strategy return (W_T - W_0) / W_0.
        cagr: Compounded Annual Growth Rate.
        annualized_volatility: Annualized return volatility sigma_ann >= 0.0.
        sharpe_ratio: Annualized strategy Sharpe ratio.
        sortino_ratio: Downside-deviation calibrated Sortino ratio.
        calmar_ratio: Return-to-maximum-drawdown Calmar ratio.
        max_drawdown: Maximum historical peak-to-trough drawdown in [0.0, 1.0].
        realized_var_95: Realized empirical 95% Value-at-Risk loss quantile.
        realized_cvar_95: Realized empirical 95% Expected Shortfall (CVaR).
        realized_var_99: Realized empirical 99% Value-at-Risk loss quantile.
        realized_cvar_99: Realized empirical 99% Expected Shortfall (CVaR).
        tail_ratio: Realized 95th percentile gain to 95th percentile loss ratio >= 0.0.
        peak_leverage: Maximum gross portfolio leverage observed >= 0.0.
        deflated_sharpe_ratio: Multiple-testing deflated Sharpe probability in [0.0, 1.0].
        min_backtest_length: Minimum backtest track record length required in days >= 0.0.
        is_statistically_significant: Flag indicating DSR >= 0.95 and T >= MinBTL.
        total_friction_cost: Cumulative transaction fee, slippage, and impact costs >= 0.0.
        circuit_breaker_counts: Mapping of bars spent in each discrete circuit breaker tier.
        benchmark_comparisons: Dictionary of comparative metrics against institutional baselines.
    """

    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    realized_var_95: float
    realized_cvar_95: float
    realized_var_99: float
    realized_cvar_99: float
    tail_ratio: float
    peak_leverage: float
    deflated_sharpe_ratio: float
    min_backtest_length: float
    is_statistically_significant: bool
    total_friction_cost: float
    circuit_breaker_counts: dict[str, int]
    benchmark_comparisons: dict[str, BenchmarkComparison]

    def __post_init__(self) -> None:
        """Validate tear sheet fields, statistical metrics, bounds, and nested entities."""
        # Functional Purpose: Validate institutional tear sheet report data integrity and boundary invariants.
        # Explicit Dependency Tracking: math.isfinite, DegenerateSimulationException, InfeasibleSimulationException.
        # Structural Relationship: Final deliverable emitted by BenchmarkAuditor; consumed by reporting pipelines.
        # Defensive Invariant: Positive initial capital; valid DSR in [0.0, 1.0]; non-negative friction; finite metrics.

        # 1. Capital ruin check on initial capital
        if not (
            isinstance(self.initial_capital, (int, float))
            and not isinstance(self.initial_capital, bool)
        ):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport initial_capital must be numeric, got {type(self.initial_capital).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not math.isfinite(self.initial_capital):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport initial_capital must be finite, got {self.initial_capital}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if self.initial_capital <= 0.0:
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport initial_capital must be strictly positive, got {self.initial_capital}",
                code=ERR_SIM_CAPITAL_RUIN,
            )

        # 2. Validate scalar float metrics
        float_fields = (
            "final_equity",
            "total_return",
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "realized_var_95",
            "realized_cvar_95",
            "realized_var_99",
            "realized_cvar_99",
            "tail_ratio",
            "peak_leverage",
            "deflated_sharpe_ratio",
            "total_friction_cost",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise DegenerateSimulationException(
                    f"BenchmarkAuditReport {field_name} must be numeric, got {type(val).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not math.isfinite(val):
                raise DegenerateSimulationException(
                    f"BenchmarkAuditReport {field_name} must be a finite float, got {val}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 3. Volatility bounds
        if self.annualized_volatility < 0.0:
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport annualized_volatility must be non-negative, got {self.annualized_volatility}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 4. Max drawdown bounds ([0.0, 1.0])
        if not (0.0 <= self.max_drawdown <= 1.0):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport max_drawdown must be in [0.0, 1.0], got {self.max_drawdown}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 5. Tail ratio bounds (>= 0.0)
        if self.tail_ratio < 0.0:
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport tail_ratio must be non-negative, got {self.tail_ratio}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 6. Peak leverage bounds (>= 0.0)
        if self.peak_leverage < 0.0:
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport peak_leverage must be non-negative, got {self.peak_leverage}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 7. Deflated Sharpe Ratio bounds ([0.0, 1.0])
        if not (0.0 <= self.deflated_sharpe_ratio <= 1.0):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport deflated_sharpe_ratio must be in [0.0, 1.0], got {self.deflated_sharpe_ratio}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 8. Minimum backtest length bounds (>= 0.0 or positive infinity for sub-benchmark SR)
        if not (
            isinstance(self.min_backtest_length, (int, float))
            and not isinstance(self.min_backtest_length, bool)
        ):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport min_backtest_length must be numeric, got {type(self.min_backtest_length).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if math.isnan(self.min_backtest_length) or self.min_backtest_length < 0.0:
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport min_backtest_length must be non-negative and non-NaN, got {self.min_backtest_length}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 9. Total friction cost non-negativity (INV-SIM-003)
        if self.total_friction_cost < 0.0:
            raise InfeasibleSimulationException(
                f"BenchmarkAuditReport total_friction_cost must be non-negative (INV-SIM-003), got {self.total_friction_cost}",
                code=ERR_SIM_NEGATIVE_FRICTION,
            )

        # 10. Statistical significance type
        if not isinstance(self.is_statistically_significant, bool):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport is_statistically_significant must be a boolean, got {type(self.is_statistically_significant).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 11. Circuit breaker counts validation
        if not isinstance(self.circuit_breaker_counts, dict):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport circuit_breaker_counts must be a dict, got {type(self.circuit_breaker_counts).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        for tier_name, count in self.circuit_breaker_counts.items():
            if not isinstance(tier_name, str) or len(tier_name.strip()) == 0:
                raise DegenerateSimulationException(
                    f"BenchmarkAuditReport circuit_breaker_counts key must be a non-empty str, got {tier_name!r}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not (isinstance(count, int) and not isinstance(count, bool)) or count < 0:
                raise DegenerateSimulationException(
                    f"BenchmarkAuditReport circuit_breaker_counts[{tier_name}] must be a non-negative int, got {count}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 12. Benchmark comparisons validation
        if not isinstance(self.benchmark_comparisons, dict):
            raise DegenerateSimulationException(
                f"BenchmarkAuditReport benchmark_comparisons must be a dict, got {type(self.benchmark_comparisons).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        for b_name, b_comp in self.benchmark_comparisons.items():
            if not isinstance(b_name, str) or len(b_name.strip()) == 0:
                raise DegenerateSimulationException(
                    f"BenchmarkAuditReport benchmark_comparisons key must be a non-empty str, got {b_name!r}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not isinstance(b_comp, BenchmarkComparison):
                raise DegenerateSimulationException(
                    f"BenchmarkAuditReport benchmark_comparisons[{b_name}] must be BenchmarkComparison, got {type(b_comp).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 13. Defensive copies of mapping containers to prevent caller mutation (ISSUE-SIM-002)
        # Functional Purpose: Prevent post-instantiation caller mutation of report dictionary fields.
        # Explicit Dependency Tracking: object.__setattr__ on frozen slots dataclass.
        # Structural Relationship: Protects BenchmarkAuditReport data integrity for consumers.
        # Defensive Invariant: Deep separation between external caller mutable dicts and internal state.
        object.__setattr__(self, "circuit_breaker_counts", dict(self.circuit_breaker_counts))
        object.__setattr__(self, "benchmark_comparisons", dict(self.benchmark_comparisons))


# ============================================================================
# Observer Protocol (Rule 1 & Rule 4: Decoupled Event Architecture)
# ============================================================================


@runtime_checkable
class SimulationListener(Protocol):
    """Protocol for observer event listener hooks across simulation execution stages.

    Enables decoupled auditing, visualization, latency tracking, and metric collection
    without contaminating or mutating the core execution loop state.
    """

    def on_bar_start(self, step: int, timestamp: int) -> None:
        """Invoked at the beginning of discrete bar t before signals or decisions are computed.

        Args:
            step: Simulation chronological bar index t >= 0.
            timestamp: Bar open or evaluation epoch timestamp in nanoseconds.
        """
        ...

    def on_decision(self, step: int, decision: Any) -> None:
        """Invoked when the execution sizer formulates target allocations nu*.

        Args:
            step: Simulation chronological bar index t >= 0.
            decision: SizingDecision object containing target allocations, haircuts, and KKT multipliers.
        """
        ...

    def on_fill(self, step: int, record: BarExecutionRecord) -> None:
        """Invoked immediately after simulated order execution and market friction deduction.

        Args:
            step: Simulation chronological bar index t >= 0.
            record: BarExecutionRecord capturing trade friction, fill prices, and allocations.
        """
        ...

    def on_bar_end(self, step: int, record: BarExecutionRecord) -> None:
        """Invoked at the conclusion of bar t following mark-to-market accounting.

        Args:
            step: Simulation chronological bar index t >= 0.
            record: BarExecutionRecord capturing updated portfolio equity, cash, and drawdown.
        """
        ...


# ============================================================================
# Execution Cost Model (Rule 1 & Rule 4: Microstructure Friction Architecture)
# ============================================================================


class ExecutionCostModel:
    """Microstructure execution friction and non-linear market impact model.

    Mathematical Formulation:
        For a trade position dollar adjustment vector Delta nu_t = nu_t - nu_{t-1} in R^N:
            C(Delta nu_t) = C_{fee}(Delta nu_t) + C_{spread}(Delta nu_t) + C_{impact}(Delta nu_t)

        1. Exchange Taker Fee:
            C_{fee}(Delta nu_t) = fee_{bps} * 10^{-4} * ||Delta nu_t||_1

        2. Bid-Ask Spread Half-Crossing Slippage:
            C_{spread}(Delta nu_t) = (spread_{bps} * 10^{-4} / 2) * ||Delta nu_t||_1

        3. 3/2-Power Kyle-Obizhaeva Non-Linear Market Impact:
            C_{impact}(Delta nu_t) = sum_{i=1}^N lambda_i * sigma_{i, t} * |Delta nu_{i, t}|^{3/2}
            where lambda_i = impact_coefficient / sqrt(ADV_i) if ADV_i > 0 is provided,
            else lambda_i = impact_coefficient.

    Defensive Invariants:
        - INV-SIM-003: Non-negative execution friction (C >= 0.0 everywhere).
        - INV-SIM-006: Hot-path execution latency SLA (sub-microsecond evaluation for N=10 assets).
        - Zero trade (Delta nu = 0) returns exactly 0.0 friction and (0.0, 0.0, 0.0) breakdown.
        - Dimension consistency: len(Delta nu) == len(sigma) (== len(ADV) if provided).
        - Strictly finite numeric inputs: rejection of NaN, Inf, negative volatility, and non-positive ADV.
    """

    def __init__(
        self,
        fee_bps: float = 2.0,
        spread_bps: float = 1.0,
        impact_coefficient: float = 0.10,
    ) -> None:
        """Initialize microstructure friction parameters and precompute constant fee schedules.

        Args:
            fee_bps: Exchange and clearing fee rate in basis points >= 0.0 (default 2.0).
            spread_bps: Average bid-ask spread in basis points >= 0.0 (default 1.0).
            impact_coefficient: Kyle-Obizhaeva non-linear market impact parameter >= 0.0 (default 0.10).

        Raises:
            DegenerateSimulationException: If any parameter is non-numeric, boolean, or non-finite (ERR-SIM-002).
            InfeasibleSimulationException: If any parameter is strictly negative (INV-SIM-003 / ERR-SIM-004).
        """
        # Functional Purpose: Initialize transaction friction parameters for exchange fees, half-spread slippage, and Kyle impact.
        # Explicit Dependency Tracking: math.isfinite, InfeasibleSimulationException, DegenerateSimulationException, ERR_SIM_NEGATIVE_FRICTION, ERR_SIM_NON_FINITE_INPUT.
        # Structural Relationship: Instantiated by ReplayEngine or standalone simulation backtesters; computes friction applied to PortfolioLedger.
        # Defensive Invariant: Parameters must be strictly finite, non-boolean numeric floats/ints, and non-negative (>= 0.0). Precomputes multipliers for hot-path speed.

        # 1. Validate numeric scalar types and finiteness
        for param_name, param_val in (
            ("fee_bps", fee_bps),
            ("spread_bps", spread_bps),
            ("impact_coefficient", impact_coefficient),
        ):
            if not (isinstance(param_val, (int, float)) and not isinstance(param_val, bool)):
                raise DegenerateSimulationException(
                    f"ExecutionCostModel {param_name} must be numeric, got {type(param_val).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not math.isfinite(param_val):
                raise DegenerateSimulationException(
                    f"ExecutionCostModel {param_name} must be a finite float, got {param_val}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if param_val < 0.0:
                raise InfeasibleSimulationException(
                    f"ExecutionCostModel {param_name} must be non-negative (INV-SIM-003), got {param_val}",
                    code=ERR_SIM_NEGATIVE_FRICTION,
                )

        self.fee_bps: float = float(fee_bps)
        self.spread_bps: float = float(spread_bps)
        self.impact_coefficient: float = float(impact_coefficient)
        self._fee_multiplier: float = self.fee_bps * 1e-4
        self._spread_multiplier: float = 0.5 * self.spread_bps * 1e-4
        self._linear_multiplier: float = self._fee_multiplier + self._spread_multiplier

    def compute_cost_breakdown(
        self,
        delta_positions: np.ndarray,
        asset_volatilities: np.ndarray,
        advs: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Compute the granular three-component friction breakdown (fee, spread, impact).

        Args:
            delta_positions: Target position dollar adjustments Delta nu_t in R^N.
            asset_volatilities: Instantaneous per-asset return volatilities sigma_t in R^N (sigma_i >= 0.0).
            advs: Optional Average Daily Volume dollar participation baselines ADV in R^N (ADV_i > 0.0).

        Returns:
            Tuple of (fee_cost, spread_cost, impact_cost) in dollars.

        Raises:
            DegenerateSimulationException: On dimension mismatch (ERR-SIM-006), non-finite inputs,
                negative volatilities, or non-positive ADVs (ERR-SIM-002).
            InfeasibleSimulationException: On negative friction cost invariant violation (ERR-SIM-004).
        """
        # Functional Purpose: Evaluate the orthogonal transaction friction components: exchange fees, half-spread slippage, and 3/2-power Kyle-Obizhaeva impact.
        # Explicit Dependency Tracking: numpy array operations, math.isfinite, DegenerateSimulationException, InfeasibleSimulationException, ERR_SIM_DIMENSION_MISMATCH, ERR_SIM_NON_FINITE_INPUT, ERR_SIM_NEGATIVE_FRICTION.
        # Structural Relationship: Ingested by PortfolioLedger to deduct cash liquidity and record bar execution costs; also invoked by compute_cost.
        # Defensive Invariant: Strict 1D vector dimensionality; len(delta) == len(vols) == len(advs); non-negative volatilities; strictly positive ADVs; non-negative cost components (INV-SIM-003).

        # 1. Type validation: must be numpy ndarrays with numeric non-boolean dtypes
        if not isinstance(delta_positions, np.ndarray):
            raise DegenerateSimulationException(
                f"delta_positions must be a numpy ndarray, got {type(delta_positions).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not (
            np.issubdtype(delta_positions.dtype, np.number)
            and not np.issubdtype(delta_positions.dtype, np.bool_)
        ):
            raise DegenerateSimulationException(
                f"delta_positions must have numeric dtype, got {delta_positions.dtype}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        if not isinstance(asset_volatilities, np.ndarray):
            raise DegenerateSimulationException(
                f"asset_volatilities must be a numpy ndarray, got {type(asset_volatilities).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not (
            np.issubdtype(asset_volatilities.dtype, np.number)
            and not np.issubdtype(asset_volatilities.dtype, np.bool_)
        ):
            raise DegenerateSimulationException(
                f"asset_volatilities must have numeric dtype, got {asset_volatilities.dtype}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        if advs is not None:
            if not isinstance(advs, np.ndarray):
                raise DegenerateSimulationException(
                    f"advs must be a numpy ndarray if provided, got {type(advs).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not (
                np.issubdtype(advs.dtype, np.number) and not np.issubdtype(advs.dtype, np.bool_)
            ):
                raise DegenerateSimulationException(
                    f"advs must have numeric dtype, got {advs.dtype}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 2. Dimensionality validation: must strictly be 1D arrays
        if delta_positions.ndim != 1:
            raise DegenerateSimulationException(
                f"delta_positions must be 1-dimensional, got ndim={delta_positions.ndim}",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )
        if asset_volatilities.ndim != 1:
            raise DegenerateSimulationException(
                f"asset_volatilities must be 1-dimensional, got ndim={asset_volatilities.ndim}",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )
        if advs is not None and advs.ndim != 1:
            raise DegenerateSimulationException(
                f"advs must be 1-dimensional, got ndim={advs.ndim}",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )

        # 3. Dimension match validation
        n = delta_positions.shape[0]
        if asset_volatilities.shape[0] != n:
            raise DegenerateSimulationException(
                f"Dimension mismatch between delta_positions ({n}) and asset_volatilities ({asset_volatilities.shape[0]})",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )
        if advs is not None and advs.shape[0] != n:
            raise DegenerateSimulationException(
                f"Dimension mismatch between delta_positions ({n}) and advs ({advs.shape[0]})",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )

        # 4. Empty trade array boundary check (N=0)
        if n == 0:
            return (0.0, 0.0, 0.0)

        # 5. Non-finite value validation (NaN or Inf)
        if not np.isfinite(delta_positions).all():
            raise DegenerateSimulationException(
                "delta_positions contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not np.isfinite(asset_volatilities).all():
            raise DegenerateSimulationException(
                "asset_volatilities contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if advs is not None and not np.isfinite(advs).all():
            raise DegenerateSimulationException(
                "advs contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 6. Domain bounds validation: volatilities >= 0.0, ADVs > 0.0
        if (asset_volatilities < 0.0).any():
            raise DegenerateSimulationException(
                "asset_volatilities must be non-negative (sigma_i >= 0.0)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if advs is not None and (advs <= 0.0).any():
            raise DegenerateSimulationException(
                "advs must be strictly positive (ADV_i > 0.0)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 7. Absolute position changes and L1 turnover norm
        abs_delta = np.abs(delta_positions)
        l1_turnover = float(abs_delta.sum())

        # Exact zero trade shortcut: if all deltas are 0.0, return bit-exact (0.0, 0.0, 0.0)
        if l1_turnover == 0.0:
            return (0.0, 0.0, 0.0)

        # 8. Compute fee and spread crossing costs
        fee_cost = float(self._fee_multiplier * l1_turnover)
        spread_cost = float(self._spread_multiplier * l1_turnover)

        # 9. Compute 3/2-power non-linear Kyle-Obizhaeva market impact cost:
        # C_impact = sum lambda_i * sigma_i * |Delta nu_i|^(3/2)
        # Using x * sqrt(x) for hardware-accelerated square-root law without transcendental pow()
        abs_delta_3_2 = abs_delta * np.sqrt(abs_delta)
        if advs is None:
            impact_cost = float(
                self.impact_coefficient * float(np.dot(asset_volatilities, abs_delta_3_2))
            )
        else:
            lambda_vec = self.impact_coefficient / np.sqrt(advs)
            impact_cost = float(np.dot(lambda_vec * asset_volatilities, abs_delta_3_2))

        # 10. Invariant enforcement: INV-SIM-003 non-negative execution friction
        if fee_cost < 0.0 or spread_cost < 0.0 or impact_cost < 0.0:
            raise InfeasibleSimulationException(
                f"Negative friction component detected (INV-SIM-003): fee={fee_cost}, spread={spread_cost}, impact={impact_cost}",
                code=ERR_SIM_NEGATIVE_FRICTION,
            )

        return (fee_cost, spread_cost, impact_cost)

    def compute_cost(
        self,
        delta_positions: np.ndarray,
        asset_volatilities: np.ndarray,
        advs: np.ndarray | None = None,
    ) -> float:
        """Compute the total execution friction cost C(Delta nu_t) = C_fee + C_spread + C_impact.

        Args:
            delta_positions: Target position dollar adjustments Delta nu_t in R^N.
            asset_volatilities: Instantaneous per-asset return volatilities sigma_t in R^N.
            advs: Optional Average Daily Volume dollar participation baselines ADV in R^N.

        Returns:
            Total transaction friction cost in dollars >= 0.0.

        Raises:
            DegenerateSimulationException: On dimension mismatch (ERR-SIM-006), non-finite inputs,
                negative volatilities, or non-positive ADVs (ERR-SIM-002).
            InfeasibleSimulationException: On negative friction cost invariant violation (ERR-SIM-004).
        """
        # Functional Purpose: Aggregate total execution friction across fees, half-spread slippage, and Kyle impact.
        # Explicit Dependency Tracking: compute_cost_breakdown, InfeasibleSimulationException, ERR_SIM_NEGATIVE_FRICTION.
        # Structural Relationship: Primary cost interface consumed by ReplayEngine and PortfolioLedger.
        # Defensive Invariant: Returned total friction must be strictly >= 0.0 (INV-SIM-003).
        fee_cost, spread_cost, impact_cost = self.compute_cost_breakdown(
            delta_positions=delta_positions,
            asset_volatilities=asset_volatilities,
            advs=advs,
        )
        total_cost = fee_cost + spread_cost + impact_cost
        if total_cost < 0.0:
            raise InfeasibleSimulationException(
                f"Total friction cost must be non-negative (INV-SIM-003), got {total_cost}",
                code=ERR_SIM_NEGATIVE_FRICTION,
            )
        return total_cost
