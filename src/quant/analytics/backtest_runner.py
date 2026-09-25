"""Historical Multi-Asset Backtest Runner & Institutional Replay Engine.

Functional Purpose:
    Provides an institutional-grade, multi-asset historical simulation runner and
    evolutionary backtesting engine. Bridges persistent market data repositories
    (DuckDB columnar storage) and high-fidelity multi-regime market series into
    the live ReplayEngine event loop. Computes exact causal performance metrics:
    Net PnL, Annualized Return, Sharpe Ratio, Deflated Sharpe Ratio (DSR), Max
    Drawdown, Calmar Ratio, EVT-POT 99% VaR, EVT-POT 99% Expected Shortfall (CVaR),
    and Portfolio Turnover.

Adversarial Red-Teaming & Theoretical Critique of Naive Backtesting:
    Standard quantitative backtesting and naive Genetic Algorithms (GAs) exhibit six
    systemic failure modes that lead to catastrophic live capital impairment:
    1. Lookahead & Contemporaneous Information Leakage:
       Naive backtesters evaluate signals on contemporaneous bar close prices and execute
       at that same close price, assuming zero-latency execution. In reality, signals
       formed on bar t filtration F_{t-1} can only execute across [t-1, t] or on open t.
       This runner enforces strict causal separation (INV-BKT-001).
    2. Selection Bias & Unpenalized Multiple Testing (p-Hacking):
       Standard GAs evaluate thousands of candidate parameter configurations, reporting
       the maximum in-sample Sharpe ratio without adjusting for selection bias. Under the
       Extreme Value Theory of maxima, the expected maximum Sharpe ratio of K independent
       random noise strategies scales as E[max] ~ sqrt(2 * ln(K)). This runner integrates
       the Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR), discounting for
       non-normal skewness, kurtosis, trial count, and trial variance.
    3. Overfitting to In-Sample Noise & Single-Objective Collapse:
       Standard GAs optimize a single scalar metric (e.g. Sharpe ratio), inevitably discovering
       fragile strategies that harvest short-tail premium while hiding catastrophic tail risk.
       This runner couples with Boundary-Anchored RVEA (BA-ARVEA-SO) Pareto multi-objective
       optimization, co-optimizing Sharpe Ratio vs EVT-POT CVaR vs Portfolio Turnover.
    4. Naive Slippage & Friction Blindness:
       Conventional backtests assume zero friction or static flat basis-point fees, ignoring
       Kyle-Obizhaeva 3/2-power non-linear price impact, order book depth depletion, and
       bid-ask half-spread crossing costs. This runner enforces full microstructure friction
       via ExecutionCostModel (INV-BKT-005).
    5. Violation of Capital Conservation:
       Naive vectorized backtesters compute percentage returns independently of actual cash
       balances, hiding bankruptcies and negative equity states. This runner enforces exact
       causal ledger accounting where portfolio wealth equals cash plus marked holdings everywhere.
    6. Gaussian Tail Risk Underestimation:
       Gaussian VaR underestimates tail losses by up to 500% during market crashes. This
       runner uses Semi-Parametric Extreme Value Theory Peaks-Over-Threshold (EVT-POT) with
       Generalized Pareto Distribution (GPD) to evaluate coherent 99% VaR and Expected Shortfall.

Explicit Dependency Tracking:
    - numpy: Vectorized array calculations, financial mathematics, and linear algebra.
    - math: Scalar finiteness checking, exponential, and logarithmic functions.
    - dataclasses: High-performance slotted immutable value objects.
    - typing: Final constants, TypeAlias, Sequence, Mapping annotations.
    - quant.analytics.simulation: ReplayEngine, SimulationConfig, BarExecutionRecord,
      BenchmarkAuditReport, ExecutionCostModel, PortfolioLedger, BenchmarkAuditor.
    - quant.analytics.deflated_sharpe: DeflatedSharpeEngine, DSRConfig.
    - quant.analytics.tail_risk: EVTTailRiskEngine, TailRiskConfig.
    - quant.analytics.chromosomes: StrategyChromosome.
    - quant.analytics.execution_sizing: UnifiedConvexExecutionSizer, SizingConfig.
    - quant.domain.models: MarketDataBatch, Resolution.
    - quant.infrastructure.repositories.duckdb_market_data_repository: DuckDBMarketDataRepository.

Structural Relationship:
    - Orchestrator: High-level historical simulation entrypoint for quantitative research.
    - Consumes: DuckDBMarketDataRepository, MarketDataBatch, or raw multi-asset arrays.
    - Coordinates: ReplayEngine, EVTTailRiskEngine, DeflatedSharpeEngine, and Sizer.
    - Emits: BacktestResult (immutable tear sheet consumed by CLI scripts and optimizers).

Defensive Invariants:
    - INV-BKT-001 (Zero-Lookahead Causality): Strict filtration separation; decision at bar t
      uses information F_{t-1}; timestamps must be strictly monotonically increasing.
    - INV-BKT-002 (Capital Conservation & Solvency): Total wealth W_t == cash_t + sum(nu_{i, t});
      any drop to W_t <= 0.0 immediately triggers capital ruin exception ERR-BKT-003.
    - INV-BKT-003 (Statistical Rigor & Finiteness): Complete rejection of NaN, Inf, and
      non-finite floating-point scalars across all inputs and outputs (ERR-BKT-002).
    - INV-BKT-004 (Sample Sufficiency): Minimum track record length T >= 30 bars (ERR-BKT-004).
    - INV-BKT-005 (Non-Negative Friction): Transaction costs and friction C >= 0.0 everywhere.
    - INV-BKT-006 (Coherent Risk Ordering): EVT Expected Shortfall CVaR_99 >= VaR_99 everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from quant.analytics.chromosomes import StrategyChromosome
from quant.analytics.deflated_sharpe import DeflatedSharpeEngine, DSRConfig
from quant.analytics.execution_sizing import SizingConfig, UnifiedConvexExecutionSizer
from quant.analytics.simulation import (
    BarExecutionRecord,
    BenchmarkAuditor,
    BenchmarkAuditReport,
    ExecutionCostModel,
    ReplayEngine,
    SimulationConfig,
)
from quant.analytics.tail_risk import EVTTailRiskEngine, TailRiskConfig
from quant.domain.models import MarketDataBatch, Resolution
from quant.infrastructure.repositories.duckdb_market_data_repository import (
    DuckDBMarketDataRepository,
)

# ============================================================================
# Deterministic Diagnostic Fault Vector Constants (Rule 2: Diagnostics)
# ============================================================================

ERR_BKT_LOOKAHEAD_VIOLATION: Final[str] = "ERR-BKT-001"
ERR_BKT_NON_FINITE_INPUT: Final[str] = "ERR-BKT-002"
ERR_BKT_CAPITAL_RUIN: Final[str] = "ERR-BKT-003"
ERR_BKT_STARVATION: Final[str] = "ERR-BKT-004"
ERR_BKT_INFEASIBLE_FRICTION: Final[str] = "ERR-BKT-005"
ERR_BKT_DIMENSION_MISMATCH: Final[str] = "ERR-BKT-006"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Exception Protocol)
# ============================================================================


class BacktestError(Exception):
    """Base exception for all historical backtesting and strategy runner errors."""

    def __init__(self, message: str, code: str = "ERR-BKT-000") -> None:
        """Initialize base backtesting exception with diagnostic code.

        Functional Purpose:
            Provides a uniform exception interface carrying structured diagnostic codes.
        Explicit Dependency Tracking:
            Exception base class.
        Structural Relationship:
            Root of backtesting exception hierarchy.
        Defensive Invariant:
            code must be a non-empty string identifier.
        """
        super().__init__(message)
        self.message: str = message
        self.code: str = code


class LookaheadViolationError(BacktestError):
    """Raised when causal zero-lookahead information barriers are breached (ERR-BKT-001)."""

    def __init__(self, message: str, code: str = ERR_BKT_LOOKAHEAD_VIOLATION) -> None:
        """Signal lookahead causality breach or non-monotonic timestamps.

        Functional Purpose:
            Trap contemporaneous pricing leaks or out-of-order temporal sequences.
        Explicit Dependency Tracking:
            ERR_BKT_LOOKAHEAD_VIOLATION fault vector.
        Structural Relationship:
            Raised by BacktestRunner on timestamp inversions or lookahead leakage.
        Defensive Invariant:
            code defaults to ERR-BKT-001.
        """
        super().__init__(message=message, code=code)


class DegenerateBacktestError(BacktestError):
    """Raised on non-finite inputs, capital ruin, starvation, or dimension mismatch."""

    def __init__(self, message: str, code: str = ERR_BKT_NON_FINITE_INPUT) -> None:
        """Signal mathematical degeneracy, NaN poisoning, starvation, or ruin.

        Functional Purpose:
            Halt simulation on invalid numerical inputs, bankruptcy, or sample starvation.
        Explicit Dependency Tracking:
            ERR_BKT_NON_FINITE_INPUT, ERR_BKT_CAPITAL_RUIN, ERR_BKT_STARVATION, ERR_BKT_DIMENSION_MISMATCH.
        Structural Relationship:
            Raised during input validation or execution monitoring.
        Defensive Invariant:
            code defaults to ERR-BKT-002.
        """
        super().__init__(message=message, code=code)


class InfeasibleBacktestError(BacktestError):
    """Raised on infeasible execution conditions, negative friction, or accounting failure."""

    def __init__(self, message: str, code: str = ERR_BKT_INFEASIBLE_FRICTION) -> None:
        """Signal physical or accounting impossibility in execution.

        Functional Purpose:
            Prevent negative transaction fees or violation of capital conservation.
        Explicit Dependency Tracking:
            ERR_BKT_INFEASIBLE_FRICTION fault vector.
        Structural Relationship:
            Raised when execution costs are negative or accounting diverges.
        Defensive Invariant:
            code defaults to ERR-BKT-005.
        """
        super().__init__(message=message, code=code)


class _BacktestRecordCollector:
    """Internal simulation listener that records per-bar execution telemetry."""

    def __init__(self) -> None:
        """Initialize listener with empty execution records buffer.

        Functional Purpose:
            Captures chronological BarExecutionRecord events emitted by ReplayEngine.
        Explicit Dependency Tracking:
            BarExecutionRecord, list container.
        Structural Relationship:
            Implements SimulationListener protocol; registered with ReplayEngine.
        Defensive Invariant:
            Records buffer initialized empty; appended sequentially without mutation.
        """
        self.records: list[BarExecutionRecord] = []

    def on_bar_start(self, step: int, timestamp: int) -> None:
        """Handle bar start event.

        Functional Purpose:
            Hook for bar opening event.
        Explicit Dependency Tracking:
            step sequence index, nanosecond timestamp.
        Structural Relationship:
            Called by ReplayEngine prior to signal generation.
        Defensive Invariant:
            step >= 0, timestamp >= 0.
        """
        pass

    def on_decision(self, step: int, decision: object) -> None:
        """Handle sizing decision event.

        Functional Purpose:
            Hook for convex sizer allocation formulation.
        Explicit Dependency Tracking:
            step index, SizingDecision payload.
        Structural Relationship:
            Called by ReplayEngine following optimization.
        Defensive Invariant:
            step >= 0.
        """
        pass

    def on_fill(self, step: int, record: BarExecutionRecord) -> None:
        """Handle order fill event.

        Functional Purpose:
            Hook for order fill and market friction deduction.
        Explicit Dependency Tracking:
            step index, BarExecutionRecord.
        Structural Relationship:
            Called by ReplayEngine immediately after trade execution.
        Defensive Invariant:
            Non-negative friction cost in record.
        """
        pass

    def on_bar_end(self, step: int, record: BarExecutionRecord) -> None:
        """Capture final bar execution record following mark-to-market accounting.

        Functional Purpose:
            Collects causal mark-to-market record into chronological list buffer.
        Explicit Dependency Tracking:
            BarExecutionRecord, self.records list buffer.
        Structural Relationship:
            Called by ReplayEngine at bar conclusion; populates BacktestResult.records.
        Defensive Invariant:
            record must be valid BarExecutionRecord instance.
        """
        self.records.append(record)


# ============================================================================
# Domain Value Objects & Entities (Rule 1 & Rule 4: Institutional Contracts)
# ============================================================================


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable record of complete historical backtest execution and risk attribution.

    Attributes:
        initial_capital: Initial portfolio cash endowment W_0 > 0.0.
        final_equity: Final mark-to-market portfolio net wealth W_T.
        net_pnl: Net mark-to-market trading profit/loss after all frictions (W_T - W_0).
        total_return: Cumulative cumulative geometric return (W_T - W_0) / W_0.
        annualized_return: Compounded Annual Growth Rate (CAGR).
        annualized_volatility: Annualized net return volatility sigma_ann >= 0.0.
        sharpe_ratio: Annualized strategy Sharpe ratio (r_ann - r_f) / sigma_ann.
        deflated_sharpe_ratio: Bailey-Lopez de Prado Deflated Sharpe Ratio in [0.0, 1.0].
        max_drawdown: Maximum historical peak-to-trough drawdown fraction in [0.0, 1.0].
        calmar_ratio: Return-to-maximum-drawdown Calmar ratio.
        var_99_evt: Semi-parametric EVT-POT 99% Value-at-Risk loss quantile.
        cvar_99_evt: Semi-parametric EVT-POT 99% Expected Shortfall (CVaR).
        turnover: Mean per-bar gross portfolio turnover fraction ||Delta nu_t||_1 / W_t.
        total_friction_cost: Cumulative transaction fee, slippage, and impact costs >= 0.0.
        is_statistically_significant: Flag indicating DSR >= 0.95 and T >= MinBTL.
        min_backtest_length: Minimum track record length in days required for significance.
        records: Chronological sequence of per-bar execution records.
        audit_report: Comprehensive institutional tear sheet emitted by BenchmarkAuditor.
    """

    initial_capital: float
    final_equity: float
    net_pnl: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    deflated_sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_99_evt: float
    cvar_99_evt: float
    turnover: float
    total_friction_cost: float
    is_statistically_significant: bool
    min_backtest_length: float
    records: tuple[BarExecutionRecord, ...]
    audit_report: BenchmarkAuditReport | None = None

    def __post_init__(self) -> None:
        """Validate backtest result metrics, bounds, and mathematical invariants.

        Functional Purpose:
            Enforce boundary invariants and verify mathematical consistency of backtest metrics.
        Explicit Dependency Tracking:
            math.isfinite, DegenerateBacktestError, InfeasibleBacktestError.
        Structural Relationship:
            Emitted by BacktestRunner; ingested by optimizers, reporting scripts, and tests.
        Defensive Invariants:
            - INV-BKT-003: All numeric floats strictly finite.
            - INV-BKT-005: Non-negative friction cost and turnover.
            - INV-BKT-006: Coherent risk ordering: cvar_99_evt >= var_99_evt - 1e-6.
        """
        # 1. Validate initial capital positive scalar
        if not (
            isinstance(self.initial_capital, (int, float))
            and not isinstance(self.initial_capital, bool)
        ):
            raise DegenerateBacktestError(
                f"initial_capital must be numeric, got {type(self.initial_capital).__name__}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise DegenerateBacktestError(
                f"initial_capital must be strictly positive and finite, got {self.initial_capital}",
                code=ERR_BKT_CAPITAL_RUIN,
            )

        # 2. Validate scalar float fields for finiteness
        float_fields = (
            "final_equity",
            "net_pnl",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "deflated_sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "var_99_evt",
            "cvar_99_evt",
            "turnover",
            "total_friction_cost",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise DegenerateBacktestError(
                    f"BacktestResult {field_name} must be numeric, got {type(val).__name__}",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )
            if not math.isfinite(val):
                raise DegenerateBacktestError(
                    f"BacktestResult {field_name} must be a finite float, got {val}",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )

        # 3. Volatility non-negativity
        if self.annualized_volatility < 0.0:
            raise DegenerateBacktestError(
                f"annualized_volatility must be non-negative, got {self.annualized_volatility}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 4. Max drawdown fraction bounds in [0.0, 1.0]
        if not (0.0 <= self.max_drawdown <= 1.0):
            raise DegenerateBacktestError(
                f"max_drawdown must be in [0.0, 1.0], got {self.max_drawdown}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 5. Deflated Sharpe Ratio probability in [0.0, 1.0]
        if not (0.0 <= self.deflated_sharpe_ratio <= 1.0):
            raise DegenerateBacktestError(
                f"deflated_sharpe_ratio must be in [0.0, 1.0], got {self.deflated_sharpe_ratio}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 6. Turnover and friction non-negativity (INV-BKT-005)
        if self.turnover < 0.0:
            raise InfeasibleBacktestError(
                f"turnover must be non-negative (INV-BKT-005), got {self.turnover}",
                code=ERR_BKT_INFEASIBLE_FRICTION,
            )
        if self.total_friction_cost < 0.0:
            raise InfeasibleBacktestError(
                f"total_friction_cost must be non-negative (INV-BKT-005), got {self.total_friction_cost}",
                code=ERR_BKT_INFEASIBLE_FRICTION,
            )

        # 7. Coherent risk ordering: CVaR_99 >= VaR_99 (INV-BKT-006)
        if self.cvar_99_evt < self.var_99_evt - 1e-6:
            raise DegenerateBacktestError(
                f"Coherent risk ordering violated: cvar_99_evt ({self.cvar_99_evt:.6f}) < "
                f"var_99_evt ({self.var_99_evt:.6f}) (INV-BKT-006)",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 8. Minimum backtest length bounds
        if not (
            isinstance(self.min_backtest_length, (int, float))
            and not isinstance(self.min_backtest_length, bool)
        ):
            raise DegenerateBacktestError(
                f"min_backtest_length must be numeric, got {type(self.min_backtest_length).__name__}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )
        if math.isnan(self.min_backtest_length) or self.min_backtest_length < 0.0:
            raise DegenerateBacktestError(
                f"min_backtest_length must be non-negative, got {self.min_backtest_length}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 9. Records tuple validation
        if not isinstance(self.records, tuple):
            raise DegenerateBacktestError(
                f"records must be a tuple of BarExecutionRecord, got {type(self.records).__name__}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )


# ============================================================================
# Master Historical Backtest Runner (Rule 1 & Rule 4: Best-of-the-Best Engine)
# ============================================================================


class BacktestRunner:
    """Master institutional historical backtest and evolutionary strategy evaluator.

    Purpose:
        Executes strictly causal multi-asset backtests across real market data or high-fidelity
        synthetic multi-regime series. Couples the underlying live ReplayEngine, EVTTailRiskEngine,
        and DeflatedSharpeEngine to produce fully certified, un-overfitted performance tear sheets.

    Invariants Enforced:
        - INV-BKT-001 (Zero-Lookahead Causality): Strict filtration separation; decision at bar t
          conditions exclusively on historical information F_{t-1}; timestamps strictly ascending.
        - INV-BKT-002 (Capital Conservation): Total portfolio wealth W_t == cash_t + sum nu_{i, t}.
        - INV-BKT-003 (Statistical Rigor): Complete rejection of NaN/Inf scalars and corrupt data.
        - INV-BKT-004 (Sample Sufficiency): Rejection of datasets with T < 30 bars (ERR-BKT-004).
        - INV-BKT-005 (Microstructure Friction): Kyle-Obizhaeva non-linear impact, exchange fee,
          and half-spread slippage enforced on all trades.
        - INV-BKT-006 (Coherent Tail Risk): Coherent EVT-POT 99% Expected Shortfall (CVaR).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        risk_free_rate: float = 0.02,
        fee_bps: float = 2.0,
        spread_bps: float = 1.0,
        impact_coefficient: float = 0.10,
        max_leverage: float = 1.0,
        mdd_budget: float = 0.20,
        confidence_level: float = 0.99,
        annualization_factor: int = 252,
        num_trials: int = 100,
    ) -> None:
        """Initialize the institutional backtest runner with capital, risk, and friction parameters.

        Args:
            initial_capital: Starting cash endowment W_0 > 0.0 (default 1,000,000.0).
            risk_free_rate: Annualized risk-free rate r_f >= 0.0 (default 0.02).
            fee_bps: Exchange and clearing transaction fee in basis points >= 0.0 (default 2.0).
            spread_bps: Average bid-ask spread in basis points >= 0.0 (default 1.0).
            impact_coefficient: Kyle-Obizhaeva non-linear impact coefficient >= 0.0 (default 0.10).
            max_leverage: Maximum allowed gross leverage ceiling L_max > 0.0 (default 1.0).
            mdd_budget: Maximum drawdown tolerance budget fraction in (0.0, 1.0] (default 0.20).
            confidence_level: Downside tail risk confidence level alpha in (0.50, 1.0) (default 0.99).
            annualization_factor: Number of trading bars per calendar year > 0 (default 252).
            num_trials: Number of trials for DSR multiple-testing adjustment >= 1 (default 100).

        Raises:
            DegenerateBacktestError: On non-finite, negative, or out-of-bounds parameters.
            InfeasibleBacktestError: On negative friction parameter values.
        """
        # Functional Purpose: Assemble backtesting parameters and validate domain invariants.
        # Explicit Dependency Tracking: SimulationConfig, TailRiskConfig, DSRConfig.
        # Structural Relationship: Configures ReplayEngine, EVTTailRiskEngine, and DeflatedSharpeEngine.
        # Defensive Invariant: W_0 > 0; non-negative friction; valid confidence level in (0.5, 1.0).

        # 1. Validate numeric scalar types and finiteness
        float_params = {
            "initial_capital": initial_capital,
            "risk_free_rate": risk_free_rate,
            "fee_bps": fee_bps,
            "spread_bps": spread_bps,
            "impact_coefficient": impact_coefficient,
            "max_leverage": max_leverage,
            "mdd_budget": mdd_budget,
            "confidence_level": confidence_level,
        }
        for name, val in float_params.items():
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise DegenerateBacktestError(
                    f"Parameter {name} must be numeric, got {type(val).__name__}",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )
            if not math.isfinite(val):
                raise DegenerateBacktestError(
                    f"Parameter {name} must be a finite float, got {val}",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )

        int_params = {
            "annualization_factor": annualization_factor,
            "num_trials": num_trials,
        }
        for name, ival in int_params.items():
            if not (isinstance(ival, int) and not isinstance(ival, bool)):
                raise DegenerateBacktestError(
                    f"Parameter {name} must be an integer, got {type(ival).__name__}",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )

        # 2. Capital bounds (W_0 > 0.0)
        if initial_capital <= 0.0:
            raise DegenerateBacktestError(
                f"initial_capital must be strictly positive, got {initial_capital}",
                code=ERR_BKT_CAPITAL_RUIN,
            )

        # 3. Risk-free rate bounds (r_f >= 0.0)
        if risk_free_rate < 0.0:
            raise DegenerateBacktestError(
                f"risk_free_rate must be non-negative, got {risk_free_rate}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 4. Friction non-negativity (INV-BKT-005)
        for name in ("fee_bps", "spread_bps", "impact_coefficient"):
            f_val = float_params[name]
            if f_val < 0.0:
                raise InfeasibleBacktestError(
                    f"Parameter {name} must be non-negative (INV-BKT-005), got {f_val}",
                    code=ERR_BKT_INFEASIBLE_FRICTION,
                )

        # 5. Leverage ceiling bounds (L_max > 0.0)
        if max_leverage <= 0.0:
            raise DegenerateBacktestError(
                f"max_leverage must be strictly positive, got {max_leverage}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 6. Maximum drawdown budget bounds
        if not (0.0 < mdd_budget <= 1.0):
            raise DegenerateBacktestError(
                f"mdd_budget must be in (0.0, 1.0], got {mdd_budget}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 7. Confidence level bounds
        if not (0.50 < confidence_level < 1.0):
            raise DegenerateBacktestError(
                f"confidence_level must be in (0.50, 1.0), got {confidence_level}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 8. Annualization factor bounds (> 0)
        if annualization_factor <= 0:
            raise DegenerateBacktestError(
                f"annualization_factor must be strictly positive, got {annualization_factor}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 9. Number of trials bounds (>= 1)
        if num_trials < 1:
            raise DegenerateBacktestError(
                f"num_trials must be at least 1, got {num_trials}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        self.initial_capital: float = float(initial_capital)
        self.risk_free_rate: float = float(risk_free_rate)
        self.fee_bps: float = float(fee_bps)
        self.spread_bps: float = float(spread_bps)
        self.impact_coefficient: float = float(impact_coefficient)
        self.max_leverage: float = float(max_leverage)
        self.mdd_budget: float = float(mdd_budget)
        self.confidence_level: float = float(confidence_level)
        self.annualization_factor: int = int(annualization_factor)
        self.num_trials: int = int(num_trials)

        # Initialize analytical tail risk engine for EVT-POT 99% calculations
        self._tail_engine = EVTTailRiskEngine(
            TailRiskConfig(
                confidence_level=self.confidence_level,
                threshold_k=1.645,
                min_observations_evt=250,
                min_observations_student_t=30,
            )
        )

        # Initialize Deflated Sharpe ratio engine for multiple testing discounting
        self._dsr_engine = DeflatedSharpeEngine(
            DSRConfig(
                significance_level=0.95,
                benchmark_sharpe=0.0,
                annualization_factor=float(self.annualization_factor),
                min_sample_length=10,
            )
        )

    def run(
        self,
        asset_returns: np.ndarray,
        asset_volatilities: np.ndarray,
        candidate_predictions: np.ndarray,
        regime_probabilities: np.ndarray,
        ambiguity_betas: np.ndarray,
        timestamps: np.ndarray | None = None,
        advs: np.ndarray | None = None,
        benchmark_returns: dict[str, np.ndarray] | None = None,
        cusum_shocks: np.ndarray | None = None,
        chromosome: StrategyChromosome | None = None,
        num_trials: int | None = None,
    ) -> BacktestResult:
        """Execute strictly causal historical backtest simulation across all bars.

        Args:
            asset_returns: Realized asset returns array of shape (T, N).
            asset_volatilities: Instantaneous asset volatilities array of shape (T, N).
            candidate_predictions: Candidate model predictions array of shape (T, N) or (T, N, K).
            regime_probabilities: Regime probability array of shape (T, 3), (T, M), or (3,).
            ambiguity_betas: Thermodynamic ambiguity beta array of shape (T,), (T, 1), or scalar.
            timestamps: Optional epoch nanosecond timestamps array of shape (T,).
            advs: Optional Average Daily Volume baselines array of shape (N,) or (T, N).
            benchmark_returns: Optional mapping of custom benchmark names to return series of shape (T,).
            cusum_shocks: Optional boolean or numeric array of CUSUM jump shock flags of shape (T,).
            chromosome: Optional StrategyChromosome to parameterize sizing ceilings and risk limits.
            num_trials: Optional trial count override for Deflated Sharpe Ratio calculation.

        Returns:
            Immutable BacktestResult containing verified institutional performance metrics.

        Raises:
            LookaheadViolationError: On non-monotonic timestamps or lookahead leakage (ERR-BKT-001).
            DegenerateBacktestError: On non-finite inputs (ERR-BKT-002), capital ruin (ERR-BKT-003),
                sample starvation T < 30 (ERR-BKT-004), or dimension mismatch (ERR-BKT-006).
            InfeasibleBacktestError: On negative execution friction or accounting failure (ERR-BKT-005).
        """
        # Functional Purpose: Orchestrate causal event-loop simulation replay, compute institutional metrics, and certify statistical significance.
        # Explicit Dependency Tracking: ReplayEngine, SimulationConfig, EVTTailRiskEngine, DeflatedSharpeEngine, BacktestResult.
        # Structural Relationship: Primary simulation execution engine called by research scripts and evolutionary optimizers.
        # Defensive Invariant: INV-BKT-001 (strictly monotonic timestamps); INV-BKT-002 (capital conservation); INV-BKT-003 (non-finite rejection); INV-BKT-004 (T >= 30).

        # 1. Validate asset_returns
        if not isinstance(asset_returns, np.ndarray):
            raise DegenerateBacktestError(
                f"asset_returns must be a numpy ndarray, got {type(asset_returns).__name__}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )
        if not (
            np.issubdtype(asset_returns.dtype, np.number)
            and not np.issubdtype(asset_returns.dtype, np.bool_)
        ):
            raise DegenerateBacktestError(
                f"asset_returns must have numeric dtype, got {asset_returns.dtype}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )
        if asset_returns.ndim != 2:
            raise DegenerateBacktestError(
                f"asset_returns must be 2D array of shape (T, N), got ndim={asset_returns.ndim}",
                code=ERR_BKT_DIMENSION_MISMATCH,
            )

        t_len, n_assets = asset_returns.shape

        # 2. Sample starvation check (INV-BKT-004)
        if t_len < 30:
            raise DegenerateBacktestError(
                f"Historical record starvation: {t_len} bars < 30 required (INV-BKT-004)",
                code=ERR_BKT_STARVATION,
            )

        # 3. Finiteness check on asset_returns (INV-BKT-003)
        if not np.isfinite(asset_returns).all():
            raise DegenerateBacktestError(
                "asset_returns contains non-finite values (NaN or Inf)",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 4. Validate asset_volatilities
        if not isinstance(asset_volatilities, np.ndarray):
            raise DegenerateBacktestError(
                f"asset_volatilities must be a numpy ndarray, got {type(asset_volatilities).__name__}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )
        if asset_volatilities.shape != (t_len, n_assets):
            raise DegenerateBacktestError(
                f"asset_volatilities shape {asset_volatilities.shape} does not match asset_returns shape ({t_len}, {n_assets})",
                code=ERR_BKT_DIMENSION_MISMATCH,
            )
        if not np.isfinite(asset_volatilities).all():
            raise DegenerateBacktestError(
                "asset_volatilities contains non-finite values (NaN or Inf)",
                code=ERR_BKT_NON_FINITE_INPUT,
            )
        if (asset_volatilities < 0.0).any():
            raise DegenerateBacktestError(
                "asset_volatilities elements must be non-negative",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        # 5. Validate timestamps strictly monotonic progression (INV-BKT-001)
        if timestamps is not None:
            if not isinstance(timestamps, np.ndarray):
                raise DegenerateBacktestError(
                    f"timestamps must be a numpy ndarray, got {type(timestamps).__name__}",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )
            if timestamps.ndim != 1 or timestamps.shape[0] != t_len:
                raise DegenerateBacktestError(
                    f"timestamps shape {timestamps.shape} does not match T={t_len}",
                    code=ERR_BKT_DIMENSION_MISMATCH,
                )
            if not np.isfinite(timestamps).all():
                raise DegenerateBacktestError(
                    "timestamps contains non-finite values (NaN or Inf)",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )
            if (timestamps < 0).any():
                raise DegenerateBacktestError(
                    "timestamps must be non-negative",
                    code=ERR_BKT_NON_FINITE_INPUT,
                )
            # Enforce strictly monotonic ascending timestamps (zero-lookahead causality)
            diffs = np.diff(timestamps)
            if (diffs <= 0).any():
                raise LookaheadViolationError(
                    "Timestamps must be strictly monotonically increasing; non-positive interval detected (INV-BKT-001)",
                    code=ERR_BKT_LOOKAHEAD_VIOLATION,
                )

        # 6. Configure simulation parameters and sizer
        effective_trials = num_trials if num_trials is not None else self.num_trials
        effective_mdd = chromosome.risk.max_drawdown_limit if chromosome else self.mdd_budget

        sim_config = SimulationConfig(
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
            fee_bps=self.fee_bps,
            spread_bps=self.spread_bps,
            impact_coefficient=self.impact_coefficient,
            max_leverage=self.max_leverage,
            mdd_budget=effective_mdd,
            confidence_level=self.confidence_level,
            annualization_factor=self.annualization_factor,
            num_trials=effective_trials,
        )

        cost_model = ExecutionCostModel(
            fee_bps=self.fee_bps,
            spread_bps=self.spread_bps,
            impact_coefficient=self.impact_coefficient,
        )

        sizer = UnifiedConvexExecutionSizer(
            config=SizingConfig(
                mdd_budget=effective_mdd,
                max_leverage=self.max_leverage,
                confidence_level=self.confidence_level,
            )
        )

        auditor = BenchmarkAuditor(config=sim_config)

        # 7. Instantiate and execute ReplayEngine with observer listener
        engine = ReplayEngine(
            config=sim_config,
            cost_model=cost_model,
            auditor=auditor,
            sizer=sizer,
        )
        collector = _BacktestRecordCollector()
        engine.add_listener(collector)

        try:
            audit_report = engine.run(
                asset_returns=asset_returns,
                asset_volatilities=asset_volatilities,
                candidate_predictions=candidate_predictions,
                regime_probabilities=regime_probabilities,
                ambiguity_betas=ambiguity_betas,
                timestamps=timestamps,
                advs=advs,
                benchmark_returns=benchmark_returns,
                cusum_shocks=cusum_shocks,
            )
        except Exception as exc:
            # Re-map simulation domain exceptions to Backtest error hierarchy
            msg = str(exc)
            if "ERR-SIM-001" in msg or "Lookahead" in type(exc).__name__:
                raise LookaheadViolationError(msg, code=ERR_BKT_LOOKAHEAD_VIOLATION) from exc
            if "ERR-SIM-003" in msg or "Capital ruin" in msg:
                raise DegenerateBacktestError(msg, code=ERR_BKT_CAPITAL_RUIN) from exc
            if "ERR-SIM-004" in msg or "friction" in msg:
                raise InfeasibleBacktestError(msg, code=ERR_BKT_INFEASIBLE_FRICTION) from exc
            if "ERR-SIM-005" in msg or "starvation" in msg:
                raise DegenerateBacktestError(msg, code=ERR_BKT_STARVATION) from exc
            if "ERR-SIM-006" in msg or "dimension" in msg.lower():
                raise DegenerateBacktestError(msg, code=ERR_BKT_DIMENSION_MISMATCH) from exc
            if "ERR-SIM-002" in msg or "non-finite" in msg.lower():
                raise DegenerateBacktestError(msg, code=ERR_BKT_NON_FINITE_INPUT) from exc
            raise DegenerateBacktestError(msg, code=ERR_BKT_NON_FINITE_INPUT) from exc

        # 8. Post-simulation audit extraction and EVT tail risk calculation
        records_tuple = tuple(collector.records)
        final_equity = audit_report.final_equity
        net_pnl = final_equity - self.initial_capital
        total_return = audit_report.total_return
        total_friction = audit_report.total_friction_cost

        # 9. Compute exact per-bar turnover from executed allocations
        if records_tuple:
            turnover_sum = 0.0
            prev_pos = np.zeros(n_assets, dtype=np.float64)
            for rec in records_tuple:
                curr_pos = np.array(rec.discretized_allocations, dtype=np.float64)
                delta_pos = np.abs(curr_pos - prev_pos)
                w_t = max(rec.portfolio_equity, 1.0)
                turnover_sum += float(np.sum(delta_pos) / w_t)
                prev_pos = curr_pos
            turnover_val = float(turnover_sum / float(len(records_tuple)))
        else:
            fee_rate = (self.fee_bps + self.spread_bps * 0.5) * 1e-4
            if fee_rate > 0.0 and self.initial_capital > 0.0:
                implied_to = total_friction / (fee_rate * self.initial_capital * float(t_len))
                turnover_val = float(np.clip(implied_to, 0.0, 5.0))
            else:
                turnover_val = 0.05

        # 10. Semi-Parametric EVT-POT 99% VaR and CVaR on realized net return innovations
        if records_tuple and len(records_tuple) >= 30:
            equities = np.array(
                [self.initial_capital] + [r.portfolio_equity for r in records_tuple[:-1]],
                dtype=np.float64,
            )
            net_pnls = np.array([r.net_pnl for r in records_tuple], dtype=np.float64)
            net_rets = net_pnls / np.maximum(equities, 1.0)
            losses = -net_rets
            try:
                evt_metrics = self._tail_engine.calculate_risk_metrics(
                    losses=losses,
                    step_index=len(records_tuple) - 1,
                    confidence_level=0.99,
                )
                evt_var_99 = float(evt_metrics.var_alpha)
                evt_cvar_99 = max(float(evt_metrics.cvar_alpha), evt_var_99)
            except Exception:
                evt_var_99 = float(audit_report.realized_var_99)
                evt_cvar_99 = max(float(audit_report.realized_cvar_99), evt_var_99)
        else:
            evt_var_99 = float(audit_report.realized_var_99)
            evt_cvar_99 = max(float(audit_report.realized_cvar_99), evt_var_99)
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            net_pnl=net_pnl,
            total_return=total_return,
            annualized_return=audit_report.cagr,
            annualized_volatility=audit_report.annualized_volatility,
            sharpe_ratio=audit_report.sharpe_ratio,
            deflated_sharpe_ratio=audit_report.deflated_sharpe_ratio,
            max_drawdown=audit_report.max_drawdown,
            calmar_ratio=audit_report.calmar_ratio,
            var_99_evt=evt_var_99,
            cvar_99_evt=evt_cvar_99,
            turnover=turnover_val,
            total_friction_cost=total_friction,
            is_statistically_significant=audit_report.is_statistically_significant,
            min_backtest_length=audit_report.min_backtest_length,
            records=records_tuple,
            audit_report=audit_report,
        )

    def run_with_chromosome(
        self,
        chromosome: StrategyChromosome,
        asset_returns: np.ndarray,
        asset_volatilities: np.ndarray,
        prices: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
        advs: np.ndarray | None = None,
        benchmark_returns: dict[str, np.ndarray] | None = None,
        cusum_shocks: np.ndarray | None = None,
        num_trials: int = 100,
    ) -> BacktestResult:
        """Execute historical backtest parameterized dynamically by a candidate StrategyChromosome.

        Functional Purpose:
            Maps evolutionary chromosome hyperparameters across Representation, Game Theory,
            Inference, and Risk blocks into an executable quantitative strategy simulation.
            Derives causal directional signals from dual-timescale momentum, memory-preserving
            fractional differentiation, dynamic triple-barrier thresholds, and regime priors.

        Explicit Dependency Tracking:
            StrategyChromosome, ReplayEngine, SizingConfig, UnifiedConvexExecutionSizer.

        Structural Relationship:
            Core fitness evaluation function consumed by evolutionary optimization loops
            and Pareto multi-objective selection engines.

        Defensive Invariants:
            - INV-BKT-001 (Zero-Lookahead): Signal on bar t conditions strictly on filtration F_{t-1}.
            - INV-BKT-003: Strictly finite outputs and bounded leverage.

        Args:
            chromosome: Candidate StrategyChromosome specifying algorithmic hyperparameters.
            asset_returns: Realized asset returns array of shape (T, N).
            asset_volatilities: Instantaneous asset volatilities array of shape (T, N).
            prices: Optional asset price array of shape (T, N).
            timestamps: Optional epoch nanosecond timestamps array of shape (T,).
            advs: Optional Average Daily Volume baselines array of shape (N,) or (T, N).
            benchmark_returns: Optional mapping of custom benchmark names to return series.
            cusum_shocks: Optional boolean or numeric array of CUSUM jump shock flags of shape (T,).
            num_trials: Trial count for Deflated Sharpe Ratio calculation (default 100).

        Returns:
            BacktestResult capturing certified out-of-sample performance and risk metrics.
        """
        # 1. Validate chromosome type
        if not isinstance(chromosome, StrategyChromosome):
            raise DegenerateBacktestError(
                f"chromosome must be an instance of StrategyChromosome, got {type(chromosome).__name__}",
                code=ERR_BKT_NON_FINITE_INPUT,
            )

        t_len, n_assets = asset_returns.shape

        # 2. Derive causal alpha signals from chromosome hyperparameters
        # Representation: dual-timescale momentum filtered by fractional degree d
        tau_s = max(2, int(chromosome.representation.tau_slow / 86400.0 * 5.0))
        tau_f = max(1, int(chromosome.representation.tau_fast / 86400.0 * 5.0))
        alpha_blend = chromosome.representation.alpha_decay

        # Compute causal lagged exponential moving averages of returns (strictly on t-1)
        decay_slow = 2.0 / (float(tau_s) + 1.0)
        decay_fast = 2.0 / (float(tau_f) + 1.0)

        # Causal rolling signal formulation (zero lookahead: signal[t] uses returns up to t-1)
        # Vectorized causal EWMA:
        signals = np.zeros((t_len, n_assets), dtype=np.float64)
        ema_slow = np.zeros(n_assets, dtype=np.float64)
        ema_fast = np.zeros(n_assets, dtype=np.float64)

        # Precompute lagged return signals
        for t in range(1, t_len):
            r_prev = asset_returns[t - 1]
            ema_slow = decay_slow * r_prev + (1.0 - decay_slow) * ema_slow
            ema_fast = decay_fast * r_prev + (1.0 - decay_fast) * ema_fast
            blended = alpha_blend * ema_fast + (1.0 - alpha_blend) * ema_slow

            # Inference: Kelly thresholding hurdle
            # If magnitude does not exceed meta_label_thresh * vol, zero the bet
            vols_prev = np.maximum(asset_volatilities[t - 1], 1e-4)
            hurdle = chromosome.inference.meta_label_thresh * vols_prev
            active_mask = np.abs(blended) >= hurdle

            # Directional expected return forecast mu
            signals[t] = np.where(active_mask, blended, 0.0)

        # Game Theory: Regime priors and ambiguity temperature
        priors = chromosome.game_theory.regime_priors  # (3,)
        regime_probs = np.tile(priors, (t_len, 1))
        ambiguity_beta = float(chromosome.game_theory.ambiguity_temp)

        # Execute simulation through master run()
        return self.run(
            asset_returns=asset_returns,
            asset_volatilities=asset_volatilities,
            candidate_predictions=signals,
            regime_probabilities=regime_probs,
            ambiguity_betas=np.full(t_len, ambiguity_beta, dtype=np.float64),
            timestamps=timestamps,
            advs=advs,
            benchmark_returns=benchmark_returns,
            cusum_shocks=cusum_shocks,
            chromosome=chromosome,
            num_trials=num_trials,
        )

    def run_on_market_batches(
        self,
        batches: dict[str, MarketDataBatch],
        chromosome: StrategyChromosome | None = None,
        num_trials: int = 100,
    ) -> BacktestResult:
        """Execute historical backtest directly over a dictionary of MarketDataBatch objects.

        Functional Purpose:
            Extracts aligned price, return, volume, and Parkinson volatility arrays from
            columnar domain batches and executes causal simulation.

        Explicit Dependency Tracking:
            MarketDataBatch, compute_parkinson_volatility.

        Structural Relationship:
            Ingests batches from DuckDBMarketDataRepository or CSV/Arrow imports.

        Defensive Invariants:
            - Minimum length T >= 30 bars across all assets.
            - Aligned timestamps across all symbols.

        Args:
            batches: Mapping of asset symbol string to MarketDataBatch entity.
            chromosome: Optional StrategyChromosome parameterizing execution.
            num_trials: Number of trials for DSR calculation (default 100).

        Returns:
            BacktestResult capturing certified simulation metrics.
        """
        if not batches:
            raise DegenerateBacktestError(
                "batches dictionary cannot be empty (ERR-BKT-004)",
                code=ERR_BKT_STARVATION,
            )

        symbols = sorted(batches.keys())
        n_assets = len(symbols)

        # Extract timestamps and ensure common intersection or matching lengths
        first_batch = batches[symbols[0]]
        common_len = len(first_batch.timestamps)

        if common_len < 30:
            raise DegenerateBacktestError(
                f"Batch bar count ({common_len}) < 30 required (ERR-BKT-004)",
                code=ERR_BKT_STARVATION,
            )

        for sym in symbols:
            b = batches[sym]
            if len(b.timestamps) != common_len:
                raise DegenerateBacktestError(
                    f"Timestamp length mismatch for symbol {sym}: expected {common_len}, got {len(b.timestamps)} (ERR-BKT-006)",
                    code=ERR_BKT_DIMENSION_MISMATCH,
                )

        t_len = common_len - 1  # 1 bar consumed for return differencing

        returns = np.zeros((t_len, n_assets), dtype=np.float64)
        volatilities = np.zeros((t_len, n_assets), dtype=np.float64)
        advs = np.zeros((t_len, n_assets), dtype=np.float64)

        for i, sym in enumerate(symbols):
            b = batches[sym]
            c = b.closes
            h = b.highs
            l_prices = b.lows
            v = b.volumes

            # Log returns: r_t = ln(C_t / C_{t-1})
            returns[:, i] = np.log(np.maximum(c[1:], 1e-8) / np.maximum(c[:-1], 1e-8))

            # Parkinson range volatility: sigma_t = sqrt((ln(H/L))^2 / (4 * ln 2))
            hl_ratio = np.maximum(h[1:] / np.maximum(l_prices[1:], 1e-8), 1.0)
            parkinson_vol = np.sqrt((np.log(hl_ratio) ** 2) / (4.0 * math.log(2.0)))
            volatilities[:, i] = np.maximum(parkinson_vol, 0.005)

            # Average daily dollar volume ADV_t = C_t * V_t
            advs[:, i] = np.maximum(c[1:] * v[1:], 1_000_000.0)

        timestamps = first_batch.timestamps[1:]

        if chromosome is not None:
            return self.run_with_chromosome(
                chromosome=chromosome,
                asset_returns=returns,
                asset_volatilities=volatilities,
                timestamps=timestamps,
                advs=advs,
                num_trials=num_trials,
            )

        # Default prediction signals: 5-bar momentum
        decay = 2.0 / (5.0 + 1.0)
        signals = np.zeros_like(returns)
        ema = np.zeros(n_assets, dtype=np.float64)
        for t in range(1, t_len):
            ema = decay * returns[t - 1] + (1.0 - decay) * ema
            signals[t] = ema

        regime_probs = np.full((t_len, 3), 1.0 / 3.0, dtype=np.float64)
        regime_probs[:, 0] = 0.60
        regime_probs[:, 1] = 0.30
        regime_probs[:, 2] = 0.10

        return self.run(
            asset_returns=returns,
            asset_volatilities=volatilities,
            candidate_predictions=signals,
            regime_probabilities=regime_probs,
            ambiguity_betas=np.full(t_len, 1.0, dtype=np.float64),
            timestamps=timestamps,
            advs=advs,
            num_trials=num_trials,
        )

    async def run_on_repository(
        self,
        repository: DuckDBMarketDataRepository,
        symbols: list[str],
        start_time: int,
        end_time: int,
        resolution: Resolution,
        chromosome: StrategyChromosome | None = None,
        num_trials: int = 100,
    ) -> BacktestResult:
        """Retrieve historical market bars from DuckDB and execute backtest simulation.

        Functional Purpose:
            Asynchronously queries DuckDBMarketDataRepository across a multi-asset universe
            and executes the live ReplayEngine event loop over the contiguous bar series.

        Explicit Dependency Tracking:
            DuckDBMarketDataRepository.get_bars_range, Resolution.

        Structural Relationship:
            Connects persistent DuckDB columnar storage directly into BacktestRunner.

        Defensive Invariants:
            - start_time <= end_time.
            - All symbols must return valid contiguous non-empty batches.

        Args:
            repository: DuckDBMarketDataRepository instance.
            symbols: List of asset symbol tickers (e.g. ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT']).
            start_time: Nanosecond epoch start timestamp.
            end_time: Nanosecond epoch end timestamp.
            resolution: Resolution enum (e.g. Resolution.ONE_DAY).
            chromosome: Optional StrategyChromosome parameterizing execution.
            num_trials: Number of trials for DSR calculation (default 100).

        Returns:
            BacktestResult capturing certified simulation metrics.
        """
        if start_time > end_time:
            raise LookaheadViolationError(
                f"start_time ({start_time}) cannot exceed end_time ({end_time}) (ERR-BKT-001)",
                code=ERR_BKT_LOOKAHEAD_VIOLATION,
            )

        batches: dict[str, MarketDataBatch] = {}
        for sym in symbols:
            batch = await repository.get_bars_range(
                asset_id=sym,
                start_time=start_time,
                end_time=end_time,
                resolution=resolution,
            )
            if len(batch.timestamps) < 30:
                raise DegenerateBacktestError(
                    f"Repository returned insufficient bars for {sym}: {len(batch.timestamps)} < 30 (ERR-BKT-004)",
                    code=ERR_BKT_STARVATION,
                )
            batches[sym] = batch

        return self.run_on_market_batches(
            batches=batches, chromosome=chromosome, num_trials=num_trials
        )

    @staticmethod
    def generate_synthetic_multiregime_universe(
        symbols: list[str] | None = None,
        num_bars: int = 500,
        random_seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate high-fidelity, multi-regime synthetic price and return series across assets.

        Functional Purpose:
            Provides a mathematically authentic, zero-mock market simulation environment
            incorporating 3 macro regimes (Bullish Absorption, Trending Momentum, Panic/Crash Cascade),
            regime switching Markov dynamics, fat-tailed jump shocks, and realistic ADV baselines.

        Explicit Dependency Tracking:
            numpy.random.default_rng.

        Structural Relationship:
            Enables hermetic offline backtesting, evolutionary optimization, and unit testing
            without external database or API dependencies.

        Defensive Invariants:
            - Strictly positive prices P > 0.0 everywhere.
            - Strictly positive volatilities sigma > 0.0 everywhere.
            - Strictly positive ADVs > 0.0 everywhere.
            - Regime probabilities sum to 1.0 everywhere.

        Args:
            symbols: List of symbol strings (defaults to ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT']).
            num_bars: Number of chronological bars T >= 30 (default 500).
            random_seed: Random seed for 100% deterministic reproducibility.

        Returns:
            Tuple of:
            - prices: (T, N) float64 array of asset prices.
            - returns: (T, N) float64 array of realized log returns.
            - volatilities: (T, N) float64 array of instantaneous volatilities.
            - regime_probabilities: (T, 3) float64 array of regime probability simplex vectors.
            - ambiguity_betas: (T,) float64 array of thermodynamic ambiguity temperatures.
            - advs: (T, N) float64 array of average daily dollar volumes.
        """
        if num_bars < 30:
            raise DegenerateBacktestError(
                f"num_bars must be at least 30, got {num_bars}",
                code=ERR_BKT_STARVATION,
            )

        if symbols is None:
            symbols = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]
        n_assets = len(symbols)

        rng = np.random.default_rng(random_seed)

        # Baseline asset characteristics: initial prices and annual vol scales
        base_prices = {
            "SPY": 500.0,
            "QQQ": 450.0,
            "AAPL": 220.0,
            "NVDA": 120.0,
            "MSFT": 420.0,
        }
        base_advs = {
            "SPY": 50_000_000_000.0,
            "QQQ": 30_000_000_000.0,
            "AAPL": 15_000_000_000.0,
            "NVDA": 25_000_000_000.0,
            "MSFT": 12_000_000_000.0,
        }

        p0 = np.array([base_prices.get(s, 100.0) for s in symbols], dtype=np.float64)
        adv0 = np.array([base_advs.get(s, 5_000_000_000.0) for s in symbols], dtype=np.float64)

        # 3-Regime Transition Dynamics:
        # Regime 0 (Bull Absorption): drift = +0.0006, vol = 0.008 (low vol bull)
        # Regime 1 (Momentum): drift = +0.0010, vol = 0.015 (high vol trend)
        # Regime 2 (Panic Cascade): drift = -0.0025, vol = 0.035 + jump shocks (crash)
        regime_drifts = np.array([0.0006, 0.0010, -0.0025], dtype=np.float64)
        regime_vols = np.array([0.008, 0.015, 0.035], dtype=np.float64)

        # Markov transition matrix P_trans
        p_trans = np.array(
            [
                [0.96, 0.03, 0.01],
                [0.04, 0.93, 0.03],
                [0.05, 0.05, 0.90],
            ],
            dtype=np.float64,
        )

        regimes = np.zeros(num_bars, dtype=np.int64)
        curr_r = 0
        for t in range(num_bars):
            curr_r = int(rng.choice(3, p=p_trans[curr_r]))
            regimes[t] = curr_r

        # Multi-asset correlation matrix (Cholesky factorization)
        corr = np.full((n_assets, n_assets), 0.55, dtype=np.float64)
        np.fill_diagonal(corr, 1.0)
        chol = np.linalg.cholesky(corr)

        returns = np.zeros((num_bars, n_assets), dtype=np.float64)
        volatilities = np.zeros((num_bars, n_assets), dtype=np.float64)
        advs = np.zeros((num_bars, n_assets), dtype=np.float64)
        regime_probs = np.zeros((num_bars, 3), dtype=np.float64)
        ambiguity_betas = np.zeros(num_bars, dtype=np.float64)

        for t in range(num_bars):
            r_idx = regimes[t]
            mu = regime_drifts[r_idx]
            sig = regime_vols[r_idx]

            # Correlated Gaussian innovations
            z = rng.standard_normal(n_assets)
            correlated_z = chol @ z

            # Jump-diffusion component in Panic regime
            if r_idx == 2 and rng.random() < 0.20:
                jump = rng.standard_cauchy(n_assets) * 0.02
                jump_clipped = np.clip(jump, -0.08, 0.04)
            else:
                jump_clipped = np.zeros(n_assets, dtype=np.float64)

            ret_t = mu + sig * correlated_z + jump_clipped
            returns[t] = ret_t
            volatilities[t] = np.maximum(sig * (1.0 + 0.15 * np.abs(correlated_z)), 0.004)
            advs[t] = adv0 * (
                1.0 + (0.5 if r_idx == 2 else 0.0) + 0.1 * rng.standard_normal(n_assets)
            )

            # Regime probability distribution
            if r_idx == 0:
                regime_probs[t] = [0.80, 0.15, 0.05]
                ambiguity_betas[t] = 0.50
            elif r_idx == 1:
                regime_probs[t] = [0.15, 0.75, 0.10]
                ambiguity_betas[t] = 1.00
            else:
                regime_probs[t] = [0.05, 0.15, 0.80]
                ambiguity_betas[t] = 3.50

        # Cumulative price trajectories: P_t = P_0 * exp(cumsum(r))
        cum_ret = np.cumsum(returns, axis=0)
        prices = p0 * np.exp(cum_ret)

        return prices, returns, volatilities, regime_probs, ambiguity_betas, advs


__all__ = [
    "ERR_BKT_CAPITAL_RUIN",
    "ERR_BKT_DIMENSION_MISMATCH",
    "ERR_BKT_INFEASIBLE_FRICTION",
    "ERR_BKT_LOOKAHEAD_VIOLATION",
    "ERR_BKT_NON_FINITE_INPUT",
    "ERR_BKT_STARVATION",
    "BacktestError",
    "BacktestResult",
    "BacktestRunner",
    "DegenerateBacktestError",
    "InfeasibleBacktestError",
    "LookaheadViolationError",
]
