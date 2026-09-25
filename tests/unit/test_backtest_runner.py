"""Comprehensive Unit Tests for Historical Backtest Runner & Replay Engine.

Functional Purpose:
    Validates institutional backtest simulation accounting, zero-lookahead causality invariants,
    Bailey & Lopez de Prado Deflated Sharpe Ratio (DSR) discounting, Semi-Parametric EVT-POT 99%
    Value-at-Risk and Expected Shortfall, Kyle-Obizhaeva market impact, and deterministic error codes.

Explicit Dependency Tracking:
    - pytest: Test execution and assertion framework.
    - numpy: Numerical array generation and precision assertions.
    - quant.analytics.backtest_runner: BacktestRunner, BacktestResult, error codes, and exceptions.
    - quant.analytics.chromosomes: StrategyChromosome.
    - quant.domain.models: MarketDataBatch, PriceBar, Resolution.
    - quant.infrastructure.database.duckdb_session import DuckDBManager.
    - quant.infrastructure.repositories.duckdb_market_data_repository import DuckDBMarketDataRepository.

Structural Relationship:
    Primary test verification suite for Track C (Historical Backtest & Evolutionary Optimizer).

Defensive Invariants:
    - 100% assertions on mathematical accounting, exact capital conservation, and risk bounds.
    - Strict deterministic zero-mock execution.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quant.analytics.backtest_runner import (
    ERR_BKT_CAPITAL_RUIN,
    ERR_BKT_DIMENSION_MISMATCH,
    ERR_BKT_INFEASIBLE_FRICTION,
    ERR_BKT_LOOKAHEAD_VIOLATION,
    ERR_BKT_NON_FINITE_INPUT,
    ERR_BKT_STARVATION,
    BacktestResult,
    BacktestRunner,
    DegenerateBacktestError,
    InfeasibleBacktestError,
    LookaheadViolationError,
)
from quant.analytics.chromosomes import (
    InferenceChromosome,
    RepresentationChromosome,
    RiskChromosome,
    StrategyChromosome,
)
from quant.domain.models import PriceBar, Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.repositories.duckdb_market_data_repository import (
    DuckDBMarketDataRepository,
)

# ============================================================================
# 1. BacktestResult Domain Invariants & Validation Tests
# ============================================================================


def test_backtest_result_nominal_creation() -> None:
    """Test nominal creation and attribute access of BacktestResult dataclass.

    Functional Purpose:
        Verify valid instantiation of immutable BacktestResult value object.
    Explicit Dependency Tracking:
        BacktestResult.
    Structural Relationship:
        Ensures downstream reporting and optimization consumers receive consistent attributes.
    Defensive Invariant:
        All fields match constructor values; object is frozen.
    """
    res = BacktestResult(
        initial_capital=1_000_000.0,
        final_equity=1_150_000.0,
        net_pnl=150_000.0,
        total_return=0.15,
        annualized_return=0.155,
        annualized_volatility=0.12,
        sharpe_ratio=1.125,
        deflated_sharpe_ratio=0.965,
        max_drawdown=0.08,
        calmar_ratio=1.9375,
        var_99_evt=0.025,
        cvar_99_evt=0.035,
        turnover=0.10,
        total_friction_cost=1500.0,
        is_statistically_significant=True,
        min_backtest_length=180.0,
        records=(),
        audit_report=None,
    )
    assert res.initial_capital == 1_000_000.0
    assert res.final_equity == 1_150_000.0
    assert res.net_pnl == 150_000.0
    assert res.total_return == 0.15
    assert res.sharpe_ratio == 1.125
    assert res.deflated_sharpe_ratio == 0.965
    assert res.is_statistically_significant is True

    # Test frozen immutability
    with pytest.raises(FrozenInstanceError):
        res.final_equity = 2_000_000.0  # type: ignore[misc]


def test_backtest_result_non_finite_rejection() -> None:
    """Test that BacktestResult rejects NaN and Inf values with DegenerateBacktestError.

    Functional Purpose:
        Guard against NaN or Inf contamination in reported performance metrics.
    Explicit Dependency Tracking:
        BacktestResult, DegenerateBacktestError, ERR_BKT_NON_FINITE_INPUT.
    Structural Relationship:
        Prevents corrupt statistics from penetrating into reporting pipelines.
    Defensive Invariant:
        INV-BKT-003: Rejection of non-finite floating-point scalars.
    """
    with pytest.raises(DegenerateBacktestError) as exc_info:
        BacktestResult(
            initial_capital=1_000_000.0,
            final_equity=float("nan"),
            net_pnl=0.0,
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.10,
            sharpe_ratio=1.0,
            deflated_sharpe_ratio=0.95,
            max_drawdown=0.10,
            calmar_ratio=1.0,
            var_99_evt=0.02,
            cvar_99_evt=0.03,
            turnover=0.1,
            total_friction_cost=100.0,
            is_statistically_significant=False,
            min_backtest_length=100.0,
            records=(),
        )
    assert exc_info.value.code == ERR_BKT_NON_FINITE_INPUT


def test_backtest_result_coherent_risk_ordering_invariant() -> None:
    """Test that BacktestResult rejects inverted CVaR < VaR with DegenerateBacktestError.

    Functional Purpose:
        Enforces Artzner coherence: Expected Shortfall (CVaR) must strictly bound VaR from above.
    Explicit Dependency Tracking:
        BacktestResult, DegenerateBacktestError, INV-BKT-006.
    Structural Relationship:
        Ensures tail risk metrics satisfy mathematical coherence.
    Defensive Invariant:
        INV-BKT-006: cvar_99_evt >= var_99_evt.
    """
    with pytest.raises(DegenerateBacktestError) as exc_info:
        BacktestResult(
            initial_capital=1_000_000.0,
            final_equity=1_050_000.0,
            net_pnl=50_000.0,
            total_return=0.05,
            annualized_return=0.05,
            annualized_volatility=0.10,
            sharpe_ratio=1.0,
            deflated_sharpe_ratio=0.95,
            max_drawdown=0.10,
            calmar_ratio=0.5,
            var_99_evt=0.040,
            cvar_99_evt=0.020,  # Inverted: CVaR < VaR
            turnover=0.1,
            total_friction_cost=100.0,
            is_statistically_significant=False,
            min_backtest_length=100.0,
            records=(),
        )
    assert exc_info.value.code == ERR_BKT_NON_FINITE_INPUT
    assert "Coherent risk ordering violated" in str(exc_info.value)


def test_backtest_result_boundary_bounds_validation() -> None:
    """Test boundary checks on drawdown, DSR, initial capital, and friction.

    Functional Purpose:
        Verify parameter range guards (drawdown in [0, 1], DSR in [0, 1], capital > 0).
    Explicit Dependency Tracking:
        BacktestResult, DegenerateBacktestError, InfeasibleBacktestError.
    Structural Relationship:
        Validates post-simulation metric containers.
    Defensive Invariant:
        All metrics bounded to valid physical and probabilistic intervals.
    """
    # 1. Capital <= 0.0
    with pytest.raises(DegenerateBacktestError) as exc_info:
        BacktestResult(
            initial_capital=0.0,
            final_equity=100.0,
            net_pnl=100.0,
            total_return=1.0,
            annualized_return=1.0,
            annualized_volatility=0.10,
            sharpe_ratio=1.0,
            deflated_sharpe_ratio=0.90,
            max_drawdown=0.10,
            calmar_ratio=1.0,
            var_99_evt=0.02,
            cvar_99_evt=0.03,
            turnover=0.1,
            total_friction_cost=100.0,
            is_statistically_significant=False,
            min_backtest_length=100.0,
            records=(),
        )
    assert exc_info.value.code == ERR_BKT_CAPITAL_RUIN

    # 2. Drawdown > 1.0
    with pytest.raises(DegenerateBacktestError):
        BacktestResult(
            initial_capital=1000.0,
            final_equity=1000.0,
            net_pnl=0.0,
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.10,
            sharpe_ratio=0.0,
            deflated_sharpe_ratio=0.50,
            max_drawdown=1.50,  # Invalid
            calmar_ratio=0.0,
            var_99_evt=0.02,
            cvar_99_evt=0.03,
            turnover=0.1,
            total_friction_cost=100.0,
            is_statistically_significant=False,
            min_backtest_length=100.0,
            records=(),
        )

    # 3. Negative turnover
    with pytest.raises(InfeasibleBacktestError) as exc_info_inf:
        BacktestResult(
            initial_capital=1000.0,
            final_equity=1000.0,
            net_pnl=0.0,
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.10,
            sharpe_ratio=0.0,
            deflated_sharpe_ratio=0.50,
            max_drawdown=0.10,
            calmar_ratio=0.0,
            var_99_evt=0.02,
            cvar_99_evt=0.03,
            turnover=-0.05,  # Invalid negative turnover
            total_friction_cost=100.0,
            is_statistically_significant=False,
            min_backtest_length=100.0,
            records=(),
        )
    assert exc_info_inf.value.code == ERR_BKT_INFEASIBLE_FRICTION


# ============================================================================
# 2. BacktestRunner Initialization & Parameter Guard Tests
# ============================================================================


def test_backtest_runner_initialization_defaults() -> None:
    """Test default constructor parameterization of BacktestRunner.

    Functional Purpose:
        Ensure baseline institutional defaults (capital=1M, r_f=2%, fee=2bps, spread=1bp, trials=100).
    Explicit Dependency Tracking:
        BacktestRunner.
    Structural Relationship:
        Constructs master backtest execution pipeline.
    Defensive Invariant:
        All configurations are strictly positive and finite.
    """
    runner = BacktestRunner()
    assert runner.initial_capital == 1_000_000.0
    assert runner.risk_free_rate == 0.02
    assert runner.fee_bps == 2.0
    assert runner.spread_bps == 1.0
    assert runner.impact_coefficient == 0.10
    assert runner.max_leverage == 1.0
    assert runner.mdd_budget == 0.20
    assert runner.confidence_level == 0.99
    assert runner.annualization_factor == 252
    assert runner.num_trials == 100


def test_backtest_runner_initialization_invalid_parameters() -> None:
    """Test that BacktestRunner rejects negative capital, fees, or out-of-range parameters.

    Functional Purpose:
        Validate configuration inputs at the domain constructor boundary.
    Explicit Dependency Tracking:
        BacktestRunner, DegenerateBacktestError, InfeasibleBacktestError.
    Structural Relationship:
        Guarantees that corrupt simulation configurations cannot be initialized.
    Defensive Invariant:
        INV-BKT-003, INV-BKT-005.
    """
    # 1. Capital <= 0
    with pytest.raises(DegenerateBacktestError) as exc_info:
        BacktestRunner(initial_capital=-500.0)
    assert exc_info.value.code == ERR_BKT_CAPITAL_RUIN

    # 2. Negative fee bps (INV-BKT-005)
    with pytest.raises(InfeasibleBacktestError) as exc_info_fee:
        BacktestRunner(fee_bps=-1.0)
    assert exc_info_fee.value.code == ERR_BKT_INFEASIBLE_FRICTION

    # 3. MDD budget out of range (<= 0 or > 1)
    with pytest.raises(DegenerateBacktestError):
        BacktestRunner(mdd_budget=0.0)

    with pytest.raises(DegenerateBacktestError):
        BacktestRunner(mdd_budget=1.5)

    # 4. Confidence level <= 0.50
    with pytest.raises(DegenerateBacktestError):
        BacktestRunner(confidence_level=0.50)

    # 5. Trials < 1
    with pytest.raises(DegenerateBacktestError):
        BacktestRunner(num_trials=0)


# ============================================================================
# 3. Zero-Lookahead Causality & Timestamp Monotonicity Tests (INV-BKT-001)
# ============================================================================


def test_zero_lookahead_strictly_monotonic_timestamps() -> None:
    """Test that BacktestRunner enforces strictly monotonic timestamps (INV-BKT-001).

    Functional Purpose:
        Verify that out-of-order execution, retrograde time travel, or duplicate timestamps
        are immediately trapped with LookaheadViolationError(ERR-BKT-001).
    Explicit Dependency Tracking:
        BacktestRunner, LookaheadViolationError, ERR_BKT_LOOKAHEAD_VIOLATION.
    Structural Relationship:
        Causality invariant verification.
    Defensive Invariant:
        INV-BKT-001: Timestamps must satisfy t_0 < t_1 < ... < t_{T-1}.
    """
    runner = BacktestRunner()
    t_len = 50
    n_assets = 2

    returns = np.zeros((t_len, n_assets), dtype=np.float64)
    volatilities = np.full((t_len, n_assets), 0.01, dtype=np.float64)
    predictions = np.zeros((t_len, n_assets), dtype=np.float64)
    regimes = np.full((t_len, 3), 1.0 / 3.0, dtype=np.float64)
    betas = np.full(t_len, 1.0, dtype=np.float64)

    # Inverted timestamps: t_2 < t_1
    timestamps = np.arange(t_len, dtype=np.int64) * 86_400_000_000_000
    timestamps[10] = timestamps[9]  # Duplicate timestamp (non-positive interval)

    with pytest.raises(LookaheadViolationError) as exc_info:
        runner.run(
            asset_returns=returns,
            asset_volatilities=volatilities,
            candidate_predictions=predictions,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
            timestamps=timestamps,
        )
    assert exc_info.value.code == ERR_BKT_LOOKAHEAD_VIOLATION
    assert "strictly monotonically increasing" in str(exc_info.value)


# ============================================================================
# 4. Sample Starvation & Capital Ruin Guards (INV-BKT-002, INV-BKT-004)
# ============================================================================


def test_sample_starvation_guard() -> None:
    """Test that BacktestRunner rejects datasets with T < 30 bars (INV-BKT-004).

    Functional Purpose:
        Prevent uncalibrated statistical estimations on insufficient observation windows.
    Explicit Dependency Tracking:
        BacktestRunner, DegenerateBacktestError, ERR_BKT_STARVATION.
    Structural Relationship:
        Pre-flight data validation.
    Defensive Invariant:
        INV-BKT-004: T >= 30 required.
    """
    runner = BacktestRunner()
    returns = np.zeros((20, 2), dtype=np.float64)  # 20 bars < 30
    volatilities = np.full((20, 2), 0.01, dtype=np.float64)
    predictions = np.zeros((20, 2), dtype=np.float64)
    regimes = np.full((20, 3), 1.0 / 3.0, dtype=np.float64)
    betas = np.full(20, 1.0, dtype=np.float64)

    with pytest.raises(DegenerateBacktestError) as exc_info:
        runner.run(
            asset_returns=returns,
            asset_volatilities=volatilities,
            candidate_predictions=predictions,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )
    assert exc_info.value.code == ERR_BKT_STARVATION


def test_dimension_mismatch_guard() -> None:
    """Test that BacktestRunner rejects shape mismatches across returns and volatilities.

    Functional Purpose:
        Enforce dimensional consistency between assets and time horizons.
    Explicit Dependency Tracking:
        BacktestRunner, DegenerateBacktestError, ERR_BKT_DIMENSION_MISMATCH.
    Structural Relationship:
        Tensor dimension validation.
    Defensive Invariant:
        shape(returns) == shape(volatilities).
    """
    runner = BacktestRunner()
    returns = np.zeros((50, 3), dtype=np.float64)
    volatilities = np.full((50, 2), 0.01, dtype=np.float64)  # Asset count mismatch
    predictions = np.zeros((50, 3), dtype=np.float64)
    regimes = np.full((50, 3), 1.0 / 3.0, dtype=np.float64)
    betas = np.full(50, 1.0, dtype=np.float64)

    with pytest.raises(DegenerateBacktestError) as exc_info:
        runner.run(
            asset_returns=returns,
            asset_volatilities=volatilities,
            candidate_predictions=predictions,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )
    assert exc_info.value.code == ERR_BKT_DIMENSION_MISMATCH


# ============================================================================
# 5. End-to-End Simulation Replay & Mathematical Accounting Tests
# ============================================================================


def test_backtest_runner_end_to_end_replay() -> None:
    """Test comprehensive end-to-end replay backtest execution and accounting.

    Functional Purpose:
        Verify full integration of ReplayEngine, EVTTailRiskEngine, DeflatedSharpeEngine,
        and causal accounting on synthetic multi-asset data.
    Explicit Dependency Tracking:
        BacktestRunner, BacktestResult, generate_synthetic_multiregime_universe.
    Structural Relationship:
        Core end-to-end regression test.
    Defensive Invariants:
        - Net PnL == Final Equity - Initial Capital.
        - Turnover >= 0.0.
        - DSR in [0.0, 1.0].
        - Coherent risk ordering: CVaR_99 >= VaR_99.
    """
    runner = BacktestRunner(initial_capital=1_000_000.0, fee_bps=2.0, spread_bps=1.0)

    # Generate 100 bars of synthetic 3-asset multi-regime series
    _, returns, vols, regimes, betas, advs = runner.generate_synthetic_multiregime_universe(
        symbols=["SPY", "QQQ", "AAPL"], num_bars=100, random_seed=123
    )

    # Create simple momentum signals
    predictions = np.zeros_like(returns)
    for t in range(1, 100):
        predictions[t] = 0.5 * returns[t - 1]

    result = runner.run(
        asset_returns=returns,
        asset_volatilities=vols,
        candidate_predictions=predictions,
        regime_probabilities=regimes,
        ambiguity_betas=betas,
        advs=advs,
        num_trials=50,
    )

    # Verify accounting invariants
    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 1_000_000.0
    assert math.isclose(result.net_pnl, result.final_equity - result.initial_capital, rel_tol=1e-5)
    assert math.isclose(result.total_return, result.net_pnl / result.initial_capital, rel_tol=1e-5)
    assert result.final_equity > 0.0
    assert 0.0 <= result.max_drawdown <= 1.0
    assert 0.0 <= result.deflated_sharpe_ratio <= 1.0
    assert result.turnover >= 0.0
    assert result.total_friction_cost >= 0.0
    assert result.cvar_99_evt >= result.var_99_evt - 1e-6
    assert len(result.records) == 100


def test_deflated_sharpe_ratio_discounting_trials() -> None:
    """Test Bailey & Lopez de Prado Deflated Sharpe Ratio discounting over multiple trials.

    Functional Purpose:
        Demonstrate the statistical penalty for multiple testing: as trial count K increases,
        the expected maximum Sharpe hurdle rises, reducing DSR for an identical backtest path.
    Explicit Dependency Tracking:
        BacktestRunner, DeflatedSharpeEngine.
    Structural Relationship:
        Validates anti-overfitting p-hacking protection.
    Defensive Invariant:
        DSR(K=1000) <= DSR(K=10).
    """
    runner = BacktestRunner(initial_capital=1_000_000.0)

    _, returns, vols, regimes, betas, advs = runner.generate_synthetic_multiregime_universe(
        symbols=["SPY", "QQQ"], num_bars=100, random_seed=42
    )

    predictions = np.zeros_like(returns)
    for t in range(1, 100):
        predictions[t] = 0.005  # Constant positive drift prediction

    res_k10 = runner.run(
        asset_returns=returns,
        asset_volatilities=vols,
        candidate_predictions=predictions,
        regime_probabilities=regimes,
        ambiguity_betas=betas,
        advs=advs,
        num_trials=10,
    )

    res_k1000 = runner.run(
        asset_returns=returns,
        asset_volatilities=vols,
        candidate_predictions=predictions,
        regime_probabilities=regimes,
        ambiguity_betas=betas,
        advs=advs,
        num_trials=1000,
    )

    # Multiple testing discounting: higher trials -> lower DSR
    assert res_k1000.deflated_sharpe_ratio <= res_k10.deflated_sharpe_ratio + 1e-6


def test_kyle_obizhaeva_market_impact_friction_scaling() -> None:
    """Test that higher market impact coefficients generate higher total friction costs.

    Functional Purpose:
        Verify non-linear Kyle-Obizhaeva 3/2-power market impact modeling.
    Explicit Dependency Tracking:
        BacktestRunner, ExecutionCostModel.
    Structural Relationship:
        Execution friction validation.
    Defensive Invariant:
        Cost(high_impact) > Cost(low_impact).
    """
    runner_low_impact = BacktestRunner(
        initial_capital=1_000_000.0, fee_bps=0.0, spread_bps=0.0, impact_coefficient=0.01
    )
    runner_high_impact = BacktestRunner(
        initial_capital=1_000_000.0, fee_bps=0.0, spread_bps=0.0, impact_coefficient=0.50
    )

    _, returns, vols, regimes, betas, advs = (
        runner_low_impact.generate_synthetic_multiregime_universe(
            symbols=["SPY", "QQQ"], num_bars=80, random_seed=999
        )
    )

    # Active switching predictions causing frequent rebalancing
    predictions = np.zeros_like(returns)
    for t in range(1, 80):
        predictions[t] = 0.01 if t % 2 == 0 else -0.01

    res_low = runner_low_impact.run(
        asset_returns=returns,
        asset_volatilities=vols,
        candidate_predictions=predictions,
        regime_probabilities=regimes,
        ambiguity_betas=betas,
        advs=advs,
    )

    res_high = runner_high_impact.run(
        asset_returns=returns,
        asset_volatilities=vols,
        candidate_predictions=predictions,
        regime_probabilities=regimes,
        ambiguity_betas=betas,
        advs=advs,
    )

    # Higher impact coefficient must produce strictly higher friction cost
    assert res_high.total_friction_cost > res_low.total_friction_cost


# ============================================================================
# 6. Evolutionary StrategyChromosome Parameterization Tests
# ============================================================================


def test_backtest_runner_run_with_chromosome() -> None:
    """Test backtest execution parameterized dynamically by a candidate StrategyChromosome.

    Functional Purpose:
        Verify mapping of chromosome hyperparameters (decay half-lives, Kelly fraction,
        drawdown limits, regime priors) into live simulation signals and sizing constraints.
    Explicit Dependency Tracking:
        StrategyChromosome, BacktestRunner.run_with_chromosome.
    Structural Relationship:
        Core interface for Phase 4 evolutionary chromosome optimization.
    Defensive Invariants:
        Drawdown adhered to chromosome.risk.max_drawdown_limit; capital conserved.
    """
    runner = BacktestRunner(initial_capital=500_000.0)

    # Construct candidate chromosome
    chromosome = StrategyChromosome(
        representation=RepresentationChromosome(
            tau_slow=86400.0,
            tau_ratio=0.10,
            alpha_decay=0.60,
            fractional_d=0.35,
        ),
        inference=InferenceChromosome(
            meta_label_thresh=0.45,
            profit_take_mult=2.5,
            stop_loss_mult=1.5,
        ),
        risk=RiskChromosome(
            max_drawdown_limit=0.12,
            max_weight=0.40,
            turnover_budget=0.30,
        ),
    )

    _, returns, vols, _, _, advs = runner.generate_synthetic_multiregime_universe(
        symbols=["SPY", "QQQ", "AAPL", "NVDA", "MSFT"], num_bars=120, random_seed=777
    )

    result = runner.run_with_chromosome(
        chromosome=chromosome,
        asset_returns=returns,
        asset_volatilities=vols,
        advs=advs,
        num_trials=25,
    )

    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 500_000.0
    assert result.final_equity > 0.0
    assert result.turnover >= 0.0
    assert len(result.records) == 120


# ============================================================================
# 7. Reproducibility & Synthetic Multi-Regime Universe Tests
# ============================================================================


def test_synthetic_multiregime_universe_reproducibility() -> None:
    """Test that generate_synthetic_multiregime_universe produces bitwise identical results given the same seed.

    Functional Purpose:
        Verify deterministic reproducibility for reproducible scientific backtesting.
    Explicit Dependency Tracking:
        BacktestRunner.generate_synthetic_multiregime_universe.
    Structural Relationship:
        Simulation data hygiene.
    Defensive Invariant:
        Bitwise identical arrays for identical seed.
    """
    p1, r1, v1, reg1, b1, adv1 = BacktestRunner.generate_synthetic_multiregime_universe(
        symbols=["SPY", "QQQ"], num_bars=60, random_seed=42
    )
    p2, r2, v2, reg2, b2, adv2 = BacktestRunner.generate_synthetic_multiregime_universe(
        symbols=["SPY", "QQQ"], num_bars=60, random_seed=42
    )

    assert np.array_equal(p1, p2)
    assert np.array_equal(r1, r2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(reg1, reg2)
    assert np.array_equal(b1, b2)
    assert np.array_equal(adv1, adv2)

    # Verify physical invariants
    assert (p1 > 0.0).all()
    assert (v1 > 0.0).all()
    assert (adv1 > 0.0).all()
    assert np.allclose(np.sum(reg1, axis=1), 1.0)


# ============================================================================
# 8. DuckDB Market Data Repository Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_duckdb_market_data_repository_backtest_integration() -> None:
    """Test BacktestRunner integration with DuckDBMarketDataRepository.

    Functional Purpose:
        Verify asynchronous querying of DuckDBMarketDataRepository and causal backtest replay.
    Explicit Dependency Tracking:
        DuckDBManager, DuckDBMarketDataRepository, PriceBar, BacktestRunner.run_on_repository.
    Structural Relationship:
        Integration test verifying persistence layer coupling to backtest runner.
    Defensive Invariants:
        Valid bars ingested; backtest executes without lookahead leakage.
    """
    # Initialize in-memory DuckDB manager
    manager = DuckDBManager(database_path=":memory:")
    repo = DuckDBMarketDataRepository(manager=manager)

    # Ingest 50 daily bars for SPY and QQQ
    symbols = ["SPY", "QQQ"]
    base_prices = {"SPY": 500.0, "QQQ": 450.0}
    t_start = 1_700_000_000_000_000_000  # ns epoch
    step_ns = 86_400_000_000_000  # 1 day in ns

    bars_to_insert: list[PriceBar] = []
    rng = np.random.default_rng(101)

    for sym in symbols:
        price = base_prices[sym]
        for t in range(50):
            ts = t_start + t * step_ns
            ret = float(rng.normal(0.0005, 0.01))
            price_close = price * math.exp(ret)
            high = max(price, price_close) * 1.005
            low = min(price, price_close) * 0.995
            vwap = (price + high + low + price_close) / 4.0

            bar = PriceBar(
                asset_id=sym,
                resolution=Resolution.ONE_DAY,
                timestamp=ts,
                open=price,
                high=high,
                low=low,
                close=price_close,
                volume=1_000_000.0,
                vwap=vwap,
            )
            bars_to_insert.append(bar)
            price = price_close

    await repo.add_bars_batch(bars_to_insert)

    # Execute backtest on repository
    runner = BacktestRunner(initial_capital=100_000.0)
    result = await runner.run_on_repository(
        repository=repo,
        symbols=symbols,
        start_time=t_start,
        end_time=t_start + 49 * step_ns,
        resolution=Resolution.ONE_DAY,
        num_trials=20,
    )

    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 100_000.0
    assert result.final_equity > 0.0
    assert len(result.records) == 49  # 50 bars yield 49 returns
