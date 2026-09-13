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

from quant.analytics.deflated_sharpe import DeflatedSharpeEngine, DSRConfig

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


# ============================================================================
# Portfolio Accounting Ledger (Rule 1 & Rule 4: Causal Accounting Engine)
# ============================================================================


class PortfolioLedger:
    """Causal portfolio accounting ledger and capital conservation state machine.

    Mathematical Formulation:
        Let T be the number of discrete simulation bars, N be the number of tradeable assets.
        At each bar t in [0, T-1]:
        1. Causal Mark-to-Market PnL:
           - Step t = 0:
             Initial positions prior to t=0 are nu_{-1} = 0.
             Gross mark-to-market trading profit/loss is:
                 PnL_0^{gross} = 0.0
             Net profit/loss deducting initial execution friction C_0 >= 0:
                 PnL_0^{net} = -C_0
             Portfolio equity immediately following execution:
                 W_0 = initial_capital - C_0
             Cash balance:
                 cash_0 = W_0 - sum_{i=1}^N nu_{i, 0}
           - Step t >= 1:
             Held positions nu_{t-1} earn return r_t over [t-1, t]:
                 PnL_t^{gross} = sum_{i=1}^N nu_{i, t-1} * r_{i, t}
             Net profit/loss deducting friction C_t >= 0:
                 PnL_t^{net} = PnL_t^{gross} - C_t
             Portfolio marked equity:
                 W_t = W_{t-1} + PnL_t^{net}
             Cash balance:
                 cash_t = W_t - sum_{i=1}^N nu_{i, t}
        2. High-Water Mark (HWM) & Peak-to-Trough Drawdown:
           - HWM_t = max(HWM_{t-1}, W_t), with HWM_{-1} = initial_capital
           - DD_t = max(0.0, min(1.0, (HWM_t - W_t) / HWM_t))
        3. Gross Leverage:
           - L_t = (sum_{i=1}^N |nu_{i, t}|) / W_t if W_t > 0 else 0.0
        4. Net Fractional Returns:
           - r_0^{net} = (W_0 - initial_capital) / initial_capital
           - r_t^{net} = (W_t - W_{t-1}) / W_{t-1} = PnL_t^{net} / W_{t-1} for t >= 1
             Satisfies exact telescopic compounding: prod_{s=0}^t (1 + r_s^{net}) = W_t / initial_capital.

    Defensive Invariants:
        - INV-SIM-001 (Zero-Lookahead Causality): Strict sequential progression (step_index == last_step + 1).
        - INV-SIM-002 (Conservation of Capital): |W_t - (cash_t + sum nu_{i, t})| < 1e-5 everywhere.
        - INV-SIM-003 (Non-Negative Execution Friction): C_t >= 0.0.
        - INV-SIM-005 (Statistical Rigor & Non-Finite Protection): Rejection of NaN, Inf, non-numeric.
        - INV-SIM-006 (Execution Latency SLA): Sub-microsecond ledger update (<= 0.05ms for N=10).
        - Fail-fast ruin check: If W_t <= 0.0, raises DegenerateSimulationException(ERR-SIM-003).
    """

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        """Initialize the portfolio accounting ledger with initial capital.

        Args:
            initial_capital: Initial cash endowment W_0 > 0.0 (default 1,000,000.0).

        Raises:
            DegenerateSimulationException: If initial_capital is non-numeric, boolean, non-finite (ERR-SIM-002),
                or <= 0.0 (ERR-SIM-003).
        """
        # Functional Purpose: Initialize accounting state with validated positive starting capital endowment.
        # Explicit Dependency Tracking: math.isfinite, DegenerateSimulationException, ERR_SIM_NON_FINITE_INPUT, ERR_SIM_CAPITAL_RUIN.
        # Structural Relationship: Instantiated by ReplayEngine or standalone backtesters; tracks portfolio state.
        # Defensive Invariant: initial_capital must be finite numeric scalar > 0.0; rejects bool.

        # 1. Type and finiteness validation
        if not (
            isinstance(initial_capital, (int, float)) and not isinstance(initial_capital, bool)
        ):
            raise DegenerateSimulationException(
                f"PortfolioLedger initial_capital must be numeric, got {type(initial_capital).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not math.isfinite(initial_capital):
            raise DegenerateSimulationException(
                f"PortfolioLedger initial_capital must be a finite float, got {initial_capital}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 2. Strict positive capital boundary (W_0 > 0)
        if initial_capital <= 0.0:
            raise DegenerateSimulationException(
                f"PortfolioLedger initial_capital must be strictly positive, got {initial_capital}",
                code=ERR_SIM_CAPITAL_RUIN,
            )

        self.initial_capital: float = float(initial_capital)
        self._equity: float = self.initial_capital
        self._cash: float = self.initial_capital
        self._positions: np.ndarray | None = None
        self._hwm: float = self.initial_capital
        self._drawdown: float = 0.0
        self._last_step_index: int = -1
        self._history: list[BarExecutionRecord] = []
        self._net_returns: list[float] = []
        self._equity_curve: list[float] = []

    def update(
        self,
        step_index: int,
        timestamp: int,
        return_vector: np.ndarray,
        new_positions: np.ndarray,
        friction_cost: float,
        circuit_breaker_tier: str,
        circuit_breaker_haircut: float,
        target_allocations: np.ndarray,
    ) -> BarExecutionRecord:
        """Advance portfolio accounting state by one discrete simulation bar.

        Args:
            step_index: Chronological discrete bar sequence index t >= 0 (must equal last_step + 1).
            timestamp: Bar evaluation epoch timestamp in nanoseconds >= 0.
            return_vector: Contemporaneous realized asset return vector r_t in R^N over [t-1, t].
            new_positions: Updated executed asset dollar positions nu_t in R^N.
            friction_cost: Total execution friction C_t >= 0.0 incurred on bar t trades.
            circuit_breaker_tier: Active discrete circuit breaker risk tier label.
            circuit_breaker_haircut: Continuous logistic haircut multiplier kappa_t in [0.0, 1.0].
            target_allocations: Continuous target portfolio dollar allocations nu_t* in R^N.

        Returns:
            Immutable BarExecutionRecord capturing bar mark-to-market PnL, wealth, cash, and leverage.

        Raises:
            LookaheadViolationException: On non-sequential step index jump or out-of-order execution (ERR-SIM-001).
            DegenerateSimulationException: On dimension mismatch (ERR-SIM-006), non-finite inputs (ERR-SIM-002),
                or portfolio capital ruin / bankruptcy (ERR-SIM-003).
            InfeasibleSimulationException: On negative friction cost (INV-SIM-003 / ERR-SIM-004).
        """
        # Functional Purpose: Execute strictly causal portfolio accounting, settle mark-to-market PnL, deduct trade friction, enforce conservation of capital, and guard against capital ruin.
        # Explicit Dependency Tracking: numpy dot product/sum, math.isfinite, BarExecutionRecord, DegenerateSimulationException, LookaheadViolationException, InfeasibleSimulationException, ERR_SIM_LOOKAHEAD_VIOLATION, ERR_SIM_NON_FINITE_INPUT, ERR_SIM_CAPITAL_RUIN, ERR_SIM_NEGATIVE_FRICTION, ERR_SIM_DIMENSION_MISMATCH.
        # Structural Relationship: Called on every simulation bar by ReplayEngine; emits BarExecutionRecord consumed by listeners and BenchmarkAuditor.
        # Defensive Invariant: Sequential step progression (t == t_last + 1); identical vector dimensions N; non-negative friction; W_t > 0; exact capital conservation |W_t - (cash_t + sum nu_t)| < 1e-5.

        # 1. Validate discrete step index and strict causal sequencing (INV-SIM-001)
        if not (isinstance(step_index, int) and not isinstance(step_index, bool)):
            raise DegenerateSimulationException(
                f"step_index must be an integer, got {type(step_index).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if step_index < 0:
            raise DegenerateSimulationException(
                f"step_index must be non-negative, got {step_index}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if step_index != self._last_step_index + 1:
            raise LookaheadViolationException(
                f"Out-of-order execution step index: expected {self._last_step_index + 1}, got {step_index} (INV-SIM-001)",
                code=ERR_SIM_LOOKAHEAD_VIOLATION,
            )

        # 2. Validate timestamp
        if not (isinstance(timestamp, int) and not isinstance(timestamp, bool)):
            raise DegenerateSimulationException(
                f"timestamp must be an integer, got {type(timestamp).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if timestamp < 0:
            raise DegenerateSimulationException(
                f"timestamp must be non-negative, got {timestamp}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 3. Validate friction cost finiteness and non-negativity (INV-SIM-003)
        if not (isinstance(friction_cost, (int, float)) and not isinstance(friction_cost, bool)):
            raise DegenerateSimulationException(
                f"friction_cost must be numeric, got {type(friction_cost).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not math.isfinite(friction_cost):
            raise DegenerateSimulationException(
                f"friction_cost must be a finite float, got {friction_cost}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if friction_cost < 0.0:
            raise InfeasibleSimulationException(
                f"friction_cost must be non-negative (INV-SIM-003), got {friction_cost}",
                code=ERR_SIM_NEGATIVE_FRICTION,
            )

        # 4. Validate circuit breaker tier and haircut
        if not (isinstance(circuit_breaker_tier, str) and len(circuit_breaker_tier.strip()) > 0):
            raise DegenerateSimulationException(
                f"circuit_breaker_tier must be a non-empty string, got {circuit_breaker_tier!r}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not (
            isinstance(circuit_breaker_haircut, (int, float))
            and not isinstance(circuit_breaker_haircut, bool)
        ):
            raise DegenerateSimulationException(
                f"circuit_breaker_haircut must be numeric, got {type(circuit_breaker_haircut).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not math.isfinite(circuit_breaker_haircut) or not (
            0.0 <= circuit_breaker_haircut <= 1.0
        ):
            raise DegenerateSimulationException(
                f"circuit_breaker_haircut must be in [0.0, 1.0], got {circuit_breaker_haircut}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 5. Validate vector array types, dimensions, and finiteness
        for arr_name, arr in (
            ("return_vector", return_vector),
            ("new_positions", new_positions),
            ("target_allocations", target_allocations),
        ):
            if not isinstance(arr, np.ndarray):
                raise DegenerateSimulationException(
                    f"{arr_name} must be a numpy ndarray, got {type(arr).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if not (np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_)):
                raise DegenerateSimulationException(
                    f"{arr_name} must have numeric dtype, got {arr.dtype}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            if arr.ndim != 1:
                raise DegenerateSimulationException(
                    f"{arr_name} must be 1-dimensional, got ndim={arr.ndim}",
                    code=ERR_SIM_DIMENSION_MISMATCH,
                )

        # 6. Dimension consistency check across all vectors
        n = return_vector.shape[0]
        if new_positions.shape[0] != n:
            raise DegenerateSimulationException(
                f"Dimension mismatch between return_vector ({n}) and new_positions ({new_positions.shape[0]})",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )
        if target_allocations.shape[0] != n:
            raise DegenerateSimulationException(
                f"Dimension mismatch between new_positions ({n}) and target_allocations ({target_allocations.shape[0]})",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )
        if self._positions is not None and self._positions.shape[0] != n:
            raise DegenerateSimulationException(
                f"Asset universe dimension changed from {self._positions.shape[0]} to {n}",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )

        # 7. Non-finite array element check
        if not np.isfinite(return_vector).all():
            raise DegenerateSimulationException(
                "return_vector contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not np.isfinite(new_positions).all():
            raise DegenerateSimulationException(
                "new_positions contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not np.isfinite(target_allocations).all():
            raise DegenerateSimulationException(
                "target_allocations contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 8. Causal Mark-to-Market PnL Settlement (INV-SIM-001)
        friction_float = float(friction_cost)
        if step_index == 0:
            # At bar 0, prior positions were nu_{-1} = 0; gross PnL is bit-exact 0.0.
            gross_pnl = 0.0
            net_pnl = -friction_float
            equity = self.initial_capital + net_pnl
            prev_equity = self.initial_capital
        else:
            # At bar t >= 1, return is earned on previously committed positions nu_{t-1}.
            assert self._positions is not None
            gross_pnl = float(np.dot(self._positions, return_vector)) if n > 0 else 0.0
            net_pnl = gross_pnl - friction_float
            prev_equity = self._equity
            equity = prev_equity + net_pnl

        # 9. Capital Ruin Tripwire (W_t <= 0.0 -> ERR-SIM-003)
        if equity <= 0.0:
            raise DegenerateSimulationException(
                f"Total capital ruin detected at step {step_index}: equity={equity:.4f} <= 0.0 (INV-SIM-002)",
                code=ERR_SIM_CAPITAL_RUIN,
            )

        # 10. Cash balance and capital conservation verification (INV-SIM-002)
        pos_sum = float(np.sum(new_positions)) if n > 0 else 0.0
        cash = equity - pos_sum
        capital_discrepancy = abs(equity - (cash + pos_sum))
        if capital_discrepancy >= 1e-5:
            raise DegenerateSimulationException(
                f"Capital conservation identity violated (INV-SIM-002): |{equity} - ({cash} + {pos_sum})| = {capital_discrepancy} >= 1e-5",
                code=ERR_SIM_CAPITAL_RUIN,
            )

        # 11. High-Water Mark and Peak-to-Trough Drawdown calculation
        hwm = max(self._hwm, equity)
        self._hwm = hwm
        drawdown = max(0.0, min(1.0, (hwm - equity) / hwm))
        self._drawdown = drawdown

        # 12. Gross leverage calculation
        abs_pos_sum = float(np.sum(np.abs(new_positions))) if n > 0 else 0.0
        effective_leverage = abs_pos_sum / equity if equity > 0.0 else 0.0

        # 13. Track fractional net return and equity curve
        net_return = (equity - prev_equity) / prev_equity
        self._net_returns.append(net_return)
        self._equity_curve.append(equity)

        # 14. Advance internal state
        self._equity = equity
        self._cash = cash
        self._positions = new_positions.copy()
        self._last_step_index = step_index

        # 15. Construct and record immutable BarExecutionRecord
        record = BarExecutionRecord(
            step_index=step_index,
            timestamp=timestamp,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            friction_cost=friction_float,
            portfolio_equity=equity,
            cash_balance=cash,
            effective_leverage=effective_leverage,
            drawdown=drawdown,
            circuit_breaker_tier=circuit_breaker_tier,
            circuit_breaker_haircut=float(circuit_breaker_haircut),
            target_allocations=tuple(float(x) for x in target_allocations),
            discretized_allocations=tuple(float(x) for x in new_positions),
        )
        self._history.append(record)
        return record

    @property
    def current_equity(self) -> float:
        """Return the current marked portfolio net wealth W_t."""
        # Functional Purpose: Provide instant read access to active portfolio mark-to-market wealth.
        # Explicit Dependency Tracking: self._equity internal state.
        # Structural Relationship: Queried by ReplayEngine, risk limit monitors, and test fixtures.
        # Defensive Invariant: Returns strictly positive float W_t > 0.0.
        return self._equity

    @property
    def current_cash(self) -> float:
        """Return the current available cash liquidity balance cash_t."""
        # Functional Purpose: Expose unallocated liquid cash reserves available for collateral or new trades.
        # Explicit Dependency Tracking: self._cash internal state.
        # Structural Relationship: Ingested by margin models and portfolio rebalancers.
        # Defensive Invariant: Satisfies W_t == cash_t + sum nu_{i, t} (INV-SIM-002).
        return self._cash

    @property
    def current_positions(self) -> np.ndarray:
        """Return a defensive copy of currently held asset positions nu_t."""
        # Functional Purpose: Expose active position holdings while preventing caller mutation of internal state.
        # Explicit Dependency Tracking: self._positions numpy ndarray.
        # Structural Relationship: Consumed by ExecutionCostModel to compute Delta nu_t and ExecutionSizer.
        # Defensive Invariant: Returns isolated copy; returns empty array if no bars have executed.
        if self._positions is None:
            return np.zeros(0, dtype=np.float64)
        return self._positions.copy()

    @property
    def high_water_mark(self) -> float:
        """Return the historical peak portfolio equity HWM_t."""
        # Functional Purpose: Provide reference peak equity level for institutional drawdown auditing.
        # Explicit Dependency Tracking: self._hwm internal state.
        # Structural Relationship: Ingested by BenchmarkAuditor and circuit breaker risk monitors.
        # Defensive Invariant: Monotonically non-decreasing; HWM_t >= initial_capital.
        return self._hwm

    @property
    def current_drawdown(self) -> float:
        """Return the current peak-to-trough equity drawdown fraction DD_t in [0.0, 1.0]."""
        # Functional Purpose: Provide instantaneous drawdown metric for stop-loss and circuit breaker activation.
        # Explicit Dependency Tracking: self._drawdown internal state.
        # Structural Relationship: Queried by CircuitBreakerOverlayEngine to trigger haircuts and halts.
        # Defensive Invariant: Strictly bounded in [0.0, 1.0].
        return self._drawdown

    @property
    def history(self) -> list[BarExecutionRecord]:
        """Return a defensive shallow copy of historical bar execution records."""
        # Functional Purpose: Provide read-only chronological record audit trail of all executed simulation bars.
        # Explicit Dependency Tracking: self._history internal list container.
        # Structural Relationship: Ingested by BenchmarkAuditor and tear sheet report generators.
        # Defensive Invariant: Returns shallow copy containing immutable frozen BarExecutionRecord value objects.
        return list(self._history)

    def get_equity_curve(self) -> np.ndarray:
        """Return 1D array of marked portfolio equity across all executed bars."""
        # Functional Purpose: Export continuous equity trajectory for time-series visualization and CAGR auditing.
        # Explicit Dependency Tracking: self._equity_curve internal list.
        # Structural Relationship: Consumed by BenchmarkAuditor to compute equity metrics and tear sheets.
        # Defensive Invariant: Returns newly constructed 1D numpy array with float64 dtype.
        return np.array(self._equity_curve, dtype=np.float64)

    def get_net_returns(self) -> np.ndarray:
        """Return 1D array of fractional net returns across all executed bars."""
        # Functional Purpose: Export discrete per-bar net return series r_t^{net} for statistical benchmarking.
        # Explicit Dependency Tracking: self._net_returns internal list.
        # Structural Relationship: Ingested by DeflatedSharpeEngine, VaR/CVaR estimators, and BenchmarkAuditor.
        # Defensive Invariant: Returns newly constructed 1D numpy array satisfying telescopic compounding.
        return np.array(self._net_returns, dtype=np.float64)


# ============================================================================
# Institutional Benchmark & Statistical Auditor (Rule 1 & Rule 4: Auditor Architecture)
# ============================================================================


class BenchmarkAuditor:
    """Institutional performance auditor and statistical significance certification engine.

    Evaluates simulated trading trajectories against institutional benchmarks and rigorous
    statistical tests:
        1. Compounded Annual Growth Rate (CAGR), Annualized Volatility, Sharpe, Sortino, Calmar.
        2. Realized Empirical Value-at-Risk (VaR 95%, 99%) and Expected Shortfall (CVaR 95%, 99%).
        3. Tail Ratio, Peak Gross Leverage, and Total Transaction Friction.
        4. Deflated Sharpe Ratio (DSR) and Minimum Backtest Length (MinBTL) via DeflatedSharpeEngine.
        5. Relative performance attribution (Jensen's Alpha, Beta, Tracking Error, Information Ratio)
           against institutional baselines: Equal Weight, Risk Parity, Inverse Volatility, and Cash.

    Invariants Enforced:
        - INV-SIM-005 (Statistical Rigor & Sample Sufficiency): Requires T >= 30 bars; rejection
          of NaN, Inf, and non-finite values in records, asset returns, and benchmark series.
        - Non-negative volatility, tracking error, tail ratio, peak leverage, and friction cost.
        - Drawdown bounded strictly in [0.0, 1.0].
        - CVaR >= VaR everywhere in loss space.
        - Deflated Sharpe Ratio bounded strictly in [0.0, 1.0].
        - MinBTL >= 0.0 (or float('inf') when strategy underperforms hurdle).
        - INV-SIM-006 (Execution Latency SLA): Vectorized closed-form evaluation; 1000 bars in < 5ms.
    """

    def __init__(
        self,
        config: SimulationConfig,
        dsr_engine: Any | None = None,
    ) -> None:
        """Initialize the institutional benchmark and statistical significance auditor.

        Args:
            config: SimulationConfig containing capital, rates, confidence levels, and trial count.
            dsr_engine: Optional DeflatedSharpeEngine instance or mock; instantiates default if None.

        Raises:
            DegenerateSimulationException: If config is not a SimulationConfig instance (ERR-SIM-002)
                or if dsr_engine does not provide a callable evaluate_strategy method.
        """
        # Functional Purpose: Initialize institutional auditor with validated configuration and DSR certification engine.
        # Explicit Dependency Tracking: SimulationConfig, DeflatedSharpeEngine, DSRConfig, DegenerateSimulationException, ERR_SIM_NON_FINITE_INPUT.
        # Structural Relationship: Instantiated by ReplayEngine or standalone backtesting audit pipelines; emits BenchmarkAuditReport.
        # Defensive Invariant: config must strictly be an instance of SimulationConfig; dsr_engine must provide evaluate_strategy.

        # 1. Validate configuration entity type
        if not isinstance(config, SimulationConfig):
            raise DegenerateSimulationException(
                f"BenchmarkAuditor config must be a SimulationConfig instance, got {type(config).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        self.config: SimulationConfig = config

        # 2. Configure Deflated Sharpe Ratio engine (Phase 2 Step 6 integration)
        if dsr_engine is not None:
            if not hasattr(dsr_engine, "evaluate_strategy") or not callable(
                dsr_engine.evaluate_strategy
            ):
                raise DegenerateSimulationException(
                    f"dsr_engine must provide a callable evaluate_strategy method, got {type(dsr_engine).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            self._dsr_engine: Any = dsr_engine
        else:
            dsr_cfg = DSRConfig(
                significance_level=config.confidence_level,
                benchmark_sharpe=0.0,
                annualization_factor=float(config.annualization_factor),
                min_sample_length=10,
            )
            self._dsr_engine = DeflatedSharpeEngine(config=dsr_cfg)

    def audit(
        self,
        records: list[BarExecutionRecord],
        asset_returns: np.ndarray,
        benchmark_returns: dict[str, np.ndarray] | None = None,
    ) -> BenchmarkAuditReport:
        """Perform comprehensive institutional performance, risk, DSR, and benchmark audit.

        Args:
            records: Chronological list of BarExecutionRecord objects from simulation replay (T >= 30).
            asset_returns: 2D numpy array of shape (T, N) containing contemporaneous asset returns.
            benchmark_returns: Optional mapping of custom benchmark names to 1D return series of shape (T,).

        Returns:
            Frozen, immutable BenchmarkAuditReport containing institutional metrics and attributions.

        Raises:
            DegenerateSimulationException: On starvation (T < 30, ERR-SIM-005), dimension mismatch
                (shape[0] != T or ndim != 2, ERR-SIM-006), capital ruin (ERR-SIM-003), or non-finite
                inputs / NaN poisoning (ERR-SIM-002).
        """
        # Functional Purpose: Execute end-to-end post-simulation institutional tear sheet auditing, DSR certification, and multi-benchmark relative attribution.
        # Explicit Dependency Tracking: DeflatedSharpeEngine, BenchmarkAuditReport, BenchmarkComparison, numpy statistics, math.isfinite.
        # Structural Relationship: Primary reporting interface of the simulation module; consumed by executive tear sheets and strategy fitness selectors.
        # Defensive Invariant: INV-SIM-005 sample sufficiency T >= 30; non-finite rejection; CVaR >= VaR; valid DSR in [0.0, 1.0]; non-negative friction.

        # 1. Validate records container type and sample sufficiency (INV-SIM-005)
        if not isinstance(records, list):
            raise DegenerateSimulationException(
                f"records must be a list of BarExecutionRecord, got {type(records).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        t_len = len(records)
        if t_len < 30:
            raise DegenerateSimulationException(
                f"Simulation record starvation: {t_len} bars < 30 required (INV-SIM-005)",
                code=ERR_SIM_STARVATION,
            )

        for idx, rec in enumerate(records):
            if not isinstance(rec, BarExecutionRecord):
                raise DegenerateSimulationException(
                    f"records[{idx}] must be a BarExecutionRecord instance, got {type(rec).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )

        # 2. Validate asset_returns array type, numeric dtype, and dimensions
        if not isinstance(asset_returns, np.ndarray):
            raise DegenerateSimulationException(
                f"asset_returns must be a numpy ndarray, got {type(asset_returns).__name__}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if not (
            np.issubdtype(asset_returns.dtype, np.number)
            and not np.issubdtype(asset_returns.dtype, np.bool_)
        ):
            raise DegenerateSimulationException(
                f"asset_returns must have numeric dtype, got {asset_returns.dtype}",
                code=ERR_SIM_NON_FINITE_INPUT,
            )
        if asset_returns.ndim != 2:
            raise DegenerateSimulationException(
                f"asset_returns must be 2-dimensional (T, N), got ndim={asset_returns.ndim}",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )
        if asset_returns.shape[0] != t_len:
            raise DegenerateSimulationException(
                f"Dimension mismatch: asset_returns length ({asset_returns.shape[0]}) != records length ({t_len})",
                code=ERR_SIM_DIMENSION_MISMATCH,
            )
        if not np.isfinite(asset_returns).all():
            raise DegenerateSimulationException(
                "asset_returns contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 3. Validate custom benchmark returns mapping if provided
        if benchmark_returns is not None:
            if not isinstance(benchmark_returns, dict):
                raise DegenerateSimulationException(
                    f"benchmark_returns must be a dict if provided, got {type(benchmark_returns).__name__}",
                    code=ERR_SIM_NON_FINITE_INPUT,
                )
            for b_name, b_ret in benchmark_returns.items():
                if not isinstance(b_name, str) or len(b_name.strip()) == 0:
                    raise DegenerateSimulationException(
                        f"benchmark_returns key must be a non-empty string, got {b_name!r}",
                        code=ERR_SIM_NON_FINITE_INPUT,
                    )
                if not isinstance(b_ret, np.ndarray):
                    raise DegenerateSimulationException(
                        f"benchmark_returns[{b_name}] must be a numpy ndarray, got {type(b_ret).__name__}",
                        code=ERR_SIM_NON_FINITE_INPUT,
                    )
                if not (
                    np.issubdtype(b_ret.dtype, np.number)
                    and not np.issubdtype(b_ret.dtype, np.bool_)
                ):
                    raise DegenerateSimulationException(
                        f"benchmark_returns[{b_name}] must have numeric dtype, got {b_ret.dtype}",
                        code=ERR_SIM_NON_FINITE_INPUT,
                    )
                if b_ret.ndim != 1:
                    raise DegenerateSimulationException(
                        f"benchmark_returns[{b_name}] must be 1-dimensional, got ndim={b_ret.ndim}",
                        code=ERR_SIM_DIMENSION_MISMATCH,
                    )
                if b_ret.shape[0] != t_len:
                    raise DegenerateSimulationException(
                        f"Dimension mismatch: benchmark_returns[{b_name}] length ({b_ret.shape[0]}) != records length ({t_len})",
                        code=ERR_SIM_DIMENSION_MISMATCH,
                    )
                if not np.isfinite(b_ret).all():
                    raise DegenerateSimulationException(
                        f"benchmark_returns[{b_name}] contains non-finite values (NaN or Inf)",
                        code=ERR_SIM_NON_FINITE_INPUT,
                    )

        # 4. Extract portfolio equity curve and compute discrete net return series
        w_0 = float(self.config.initial_capital)
        equities = np.empty(t_len + 1, dtype=np.float64)
        equities[0] = w_0
        for i, rec in enumerate(records):
            equities[i + 1] = rec.portfolio_equity

        if not np.isfinite(equities).all():
            raise DegenerateSimulationException(
                "Simulation record equity curve contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # Capital ruin tripwire
        if (equities <= 0.0).any():
            raise DegenerateSimulationException(
                "Simulation records contain non-positive portfolio equity (capital ruin)",
                code=ERR_SIM_CAPITAL_RUIN,
            )

        # Causal fractional net returns r_t^{net} = (W_t - W_{t-1}) / W_{t-1}
        r_net = (equities[1:] - equities[:-1]) / equities[:-1]
        if not np.isfinite(r_net).all():
            raise DegenerateSimulationException(
                "Computed net return series contains non-finite values (NaN or Inf)",
                code=ERR_SIM_NON_FINITE_INPUT,
            )

        # 5. Compute institutional growth, volatility, and risk-adjusted return ratios
        w_t = float(equities[-1])
        total_return = float((w_t - w_0) / w_0)
        ann_factor = float(self.config.annualization_factor)
        t_float = float(t_len)
        rf_rate = float(self.config.risk_free_rate)
        rf_bar = rf_rate / ann_factor

        # CAGR calculation (guarded against negative or zero final wealth)
        cagr = -1.0 if w_t <= 0.0 else float((w_t / w_0) ** (ann_factor / t_float) - 1.0)

        # Sample moments with ddof=1
        mean_r = float(np.mean(r_net))
        std_r = float(np.std(r_net, ddof=1))
        annualized_vol = max(0.0, float(np.sqrt(ann_factor) * std_r)) if std_r >= 1e-12 else 0.0

        # Sharpe Ratio
        if std_r < 1e-12:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float(np.sqrt(ann_factor) * (mean_r - rf_bar) / std_r)

        # Sortino Ratio (downside semi-deviation below risk-free benchmark)
        downside_diff = np.minimum(0.0, r_net - rf_bar)
        downside_variance = float(np.mean(downside_diff**2))
        downside_dev = float(np.sqrt(downside_variance))
        if downside_dev < 1e-12:
            sortino_ratio = 0.0
        else:
            sortino_ratio = float(np.sqrt(ann_factor) * (mean_r - rf_bar) / downside_dev)

        # Maximum Drawdown and Calmar Ratio
        max_drawdown = float(max(rec.drawdown for rec in records))
        max_drawdown = min(1.0, max(0.0, max_drawdown))
        calmar_ratio = 0.0 if max_drawdown < 1e-12 else float(cagr / max_drawdown)

        # 6. Realized Empirical Tail Risk (VaR & CVaR in loss space, Tail Ratio)
        q01, q05, q95 = (float(v) for v in np.quantile(r_net, [0.01, 0.05, 0.95]))
        realized_var_95 = float(-q05)
        tail_losses_95 = r_net[r_net <= q05]
        mean_loss_95 = (
            float(-np.mean(tail_losses_95)) if len(tail_losses_95) > 0 else realized_var_95
        )
        realized_cvar_95 = float(max(realized_var_95, mean_loss_95))

        realized_var_99 = float(-q01)
        tail_losses_99 = r_net[r_net <= q01]
        mean_loss_99 = (
            float(-np.mean(tail_losses_99)) if len(tail_losses_99) > 0 else realized_var_99
        )
        realized_cvar_99 = float(max(realized_var_99, mean_loss_99))

        denom_tail = abs(q05)
        tail_ratio = float(max(0.0, q95 / denom_tail)) if denom_tail >= 1e-12 else 0.0

        # Peak leverage, friction cost, and circuit breaker operational counts
        peak_leverage = float(max(rec.effective_leverage for rec in records))
        total_friction_cost = float(sum(rec.friction_cost for rec in records))

        circuit_breaker_counts: dict[str, int] = {}
        for rec in records:
            circuit_breaker_counts[rec.circuit_breaker_tier] = (
                circuit_breaker_counts.get(rec.circuit_breaker_tier, 0) + 1
            )

        # 7. Statistical Significance Certification via Deflated Sharpe Ratio
        dsr_result = self._dsr_engine.evaluate_strategy(
            returns=r_net,
            n_trials=self.config.num_trials,
        )
        deflated_sharpe = float(dsr_result.deflated_sharpe_ratio)
        min_btl = float(dsr_result.min_backtest_length)
        is_statistically_significant = bool(
            deflated_sharpe >= self.config.confidence_level and t_len >= min_btl
        )

        # 8. Institutional Baseline Benchmarks Construction
        n_assets = asset_returns.shape[1]

        # 8.1 Equal Weight (EW): r_t^{EW} = (1/N) sum r_{i, t}
        if n_assets > 0:
            r_ew = np.asarray(np.mean(asset_returns, axis=1), dtype=np.float64)
        else:
            r_ew = np.zeros(t_len, dtype=np.float64)

        # 8.2 Risk Parity (RP): weights inversely proportional to trailing volatility (vectorized)
        if n_assets > 0:
            r_rp = np.empty(t_len, dtype=np.float64)
            r_rp[0] = float(np.mean(asset_returns[0]))
            if t_len > 1:
                r_rp[1] = float(np.mean(asset_returns[1]))
            if t_len > 2:
                window = 20
                u = asset_returns - np.mean(asset_returns, axis=0, keepdims=True)
                cumsum_u = np.vstack(
                    [np.zeros((1, n_assets), dtype=np.float64), np.cumsum(u, axis=0)]
                )
                cumsum_u2 = np.vstack(
                    [np.zeros((1, n_assets), dtype=np.float64), np.cumsum(u**2, axis=0)]
                )
                t_idx = np.arange(2, t_len, dtype=np.int64)
                s_idx = np.maximum(0, t_idx - window)
                k_vec = (t_idx - s_idx)[:, np.newaxis]
                sum_u = cumsum_u[t_idx] - cumsum_u[s_idx]
                sum_u2 = cumsum_u2[t_idx] - cumsum_u2[s_idx]
                var_t = np.maximum(0.0, (sum_u2 - (sum_u**2) / k_vec) / (k_vec - 1.0))
                vols_t = np.sqrt(np.maximum(var_t, 1e-24))
                vols_t = np.where(vols_t < 1e-12, 1e-12, vols_t)
                inv_vols_t = 1.0 / vols_t
                tot_inv = np.sum(inv_vols_t, axis=1, keepdims=True)
                w_rp = np.where(tot_inv > 0.0, inv_vols_t / tot_inv, 1.0 / float(n_assets))
                r_rp[2:] = np.sum(w_rp * asset_returns[2:], axis=1)
        else:
            r_rp = np.zeros(t_len, dtype=np.float64)

        # 8.3 Inverse Volatility: weights inversely proportional to sigma_i^2
        if n_assets > 0:
            sample_vars = np.var(asset_returns, axis=0, ddof=1)
            sample_vars = np.where(sample_vars < 1e-12, 1e-12, sample_vars)
            inv_vars = 1.0 / sample_vars
            tot_inv_var = float(np.sum(inv_vars))
            w_inv_vol = (
                inv_vars / tot_inv_var
                if tot_inv_var > 0.0
                else np.full(n_assets, 1.0 / float(n_assets), dtype=np.float64)
            )
            r_inv_vol = np.asarray(asset_returns @ w_inv_vol, dtype=np.float64)
        else:
            r_inv_vol = np.zeros(t_len, dtype=np.float64)

        # 8.4 Cash: r_t^{Cash} = r_f / A
        r_cash = np.full(t_len, rf_bar, dtype=np.float64)

        # Merge standard benchmarks and optional caller benchmarks
        all_benchmarks: dict[str, np.ndarray] = {
            "EqualWeight": r_ew,
            "RiskParity": r_rp,
            "InverseVolatility": r_inv_vol,
            "Cash": r_cash,
        }
        if benchmark_returns is not None:
            for b_name, b_series in benchmark_returns.items():
                all_benchmarks[b_name] = b_series

        # 9. Compute comparative attribution metrics against each benchmark
        comparisons: dict[str, BenchmarkComparison] = {}
        for b_name, b_ret in all_benchmarks.items():
            b_total_return = float(np.prod(1.0 + b_ret) - 1.0)
            b_cagr = (
                -1.0
                if 1.0 + b_total_return <= 0.0
                else float((1.0 + b_total_return) ** (ann_factor / t_float) - 1.0)
            )

            b_std = float(np.std(b_ret, ddof=1))
            b_vol = max(0.0, float(np.sqrt(ann_factor) * b_std)) if b_std >= 1e-12 else 0.0
            b_mean = float(np.mean(b_ret))
            if b_std < 1e-12:
                b_sharpe = 0.0
            else:
                b_sharpe = float(np.sqrt(ann_factor) * (b_mean - rf_bar) / b_std)

            # Benchmark peak-to-trough drawdown
            b_wealth = np.cumprod(np.insert(1.0 + b_ret, 0, 1.0))
            b_hwm = np.maximum.accumulate(b_wealth)
            b_dd = (b_hwm - b_wealth) / np.maximum(b_hwm, 1e-12)
            b_mdd = min(1.0, max(0.0, float(np.max(b_dd))))

            # OLS Alpha and Beta regression: r_net - rf_bar = alpha_daily + beta * (b_ret - rf_bar)
            y = r_net - rf_bar
            x = b_ret - rf_bar
            x_bar = float(np.mean(x))
            y_bar = float(np.mean(y))
            dx = x - x_bar
            dy = y - y_bar
            var_x = float(np.dot(dx, dx) / (t_float - 1.0))
            if var_x < 1e-12:
                beta = 0.0
                alpha_daily = y_bar
            else:
                cov_xy = float(np.dot(dx, dy) / (t_float - 1.0))
                beta = float(cov_xy / var_x)
                alpha_daily = float(y_bar - beta * x_bar)
            alpha = float(alpha_daily * ann_factor)

            # Tracking Error and Information Ratio
            diff = r_net - b_ret
            std_diff = float(np.std(diff, ddof=1))
            te = max(0.0, float(np.sqrt(ann_factor) * std_diff)) if std_diff >= 1e-12 else 0.0
            if te < 1e-12 or std_diff < 1e-12:
                ir = 0.0
            else:
                ir = float(np.sqrt(ann_factor) * float(np.mean(diff)) / std_diff)

            comparisons[b_name] = BenchmarkComparison(
                name=b_name,
                total_return=b_total_return,
                annualized_return=b_cagr,
                annualized_volatility=b_vol,
                sharpe_ratio=b_sharpe,
                max_drawdown=b_mdd,
                alpha=alpha,
                beta=beta,
                tracking_error=te,
                information_ratio=ir,
            )

        # 10. Construct and return frozen BenchmarkAuditReport
        return BenchmarkAuditReport(
            initial_capital=w_0,
            final_equity=w_t,
            total_return=total_return,
            cagr=cagr,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            realized_var_95=realized_var_95,
            realized_cvar_95=realized_cvar_95,
            realized_var_99=realized_var_99,
            realized_cvar_99=realized_cvar_99,
            tail_ratio=tail_ratio,
            peak_leverage=peak_leverage,
            deflated_sharpe_ratio=deflated_sharpe,
            min_backtest_length=min_btl,
            is_statistically_significant=is_statistically_significant,
            total_friction_cost=total_friction_cost,
            circuit_breaker_counts=circuit_breaker_counts,
            benchmark_comparisons=comparisons,
        )
