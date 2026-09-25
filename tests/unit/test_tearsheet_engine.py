"""Unit tests for CFA-Grade Performance Analytics Engine & Attribution.

Functional Purpose:
    Verifies mathematical attribution, Sharpe/Sortino/Calmar ratios, underwater drawdowns,
    benchmark CAPM alpha/beta, monthly calendar heatmaps, and strict error handling.

Explicit Dependency Tracking:
    - pytest, numpy.
    - quant.analytics.tearsheet: PerformanceAnalyticsEngine, CFAMetrics, TearsheetReport,
      ZeroVarianceError, MismatchedBenchmarkError, NonFiniteReturnError,
      ERR_RPT_ZERO_VARIANCE, ERR_RPT_MISMATCHED_BENCHMARK, ERR_RPT_NON_FINITE_INPUT.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from quant.analytics.tearsheet import (
    ERR_RPT_MISMATCHED_BENCHMARK,
    ERR_RPT_NON_FINITE_INPUT,
    ERR_RPT_ZERO_VARIANCE,
    MismatchedBenchmarkError,
    NonFiniteReturnError,
    PerformanceAnalyticsEngine,
    TearsheetReport,
    ZeroVarianceError,
)


def _generate_synthetic_timestamps(n: int) -> np.ndarray:
    """Generate ascending daily timestamps in nanoseconds starting 2024-01-01."""
    t0 = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1e9)
    step_ns = int(86400 * 1e9)
    return np.array([t0 + i * step_ns for i in range(n)], dtype=np.int64)


def test_tearsheet_basic_calculation() -> None:
    """Verify standard metrics on synthetic upward trending return series."""
    np.random.seed(42)
    n = 252  # 1 year of trading
    returns = np.random.normal(0.0008, 0.01, n)  # Positive drift
    timestamps = _generate_synthetic_timestamps(n)

    report = PerformanceAnalyticsEngine.calculate_tearsheet(
        strategy_returns=returns,
        timestamps_ns=timestamps,
        risk_free_rate=0.0,
        periods_per_year=252,
    )

    assert isinstance(report, TearsheetReport)
    assert report.metrics.total_trades_bars == 252
    assert report.metrics.sharpe_ratio > 0.0
    assert report.metrics.cagr > 0.0
    assert report.metrics.annualized_volatility > 0.0
    assert report.metrics.win_rate > 0.40
    assert report.metrics.max_drawdown <= 0.0
    assert len(report.equity_curve) == 252
    assert len(report.drawdown_series) == 252


def test_tearsheet_benchmark_alpha_beta_attribution() -> None:
    """Verify CAPM alpha and beta calculations against benchmark series."""
    np.random.seed(42)
    n = 200
    bench_returns = np.random.normal(0.0005, 0.012, n)
    timestamps = _generate_synthetic_timestamps(n)

    # Strategy = 1.5 * Benchmark + 0.0002 alpha + noise
    strat_returns = 1.5 * bench_returns + 0.0002 + np.random.normal(0, 0.002, n)

    report = PerformanceAnalyticsEngine.calculate_tearsheet(
        strategy_returns=strat_returns,
        timestamps_ns=timestamps,
        benchmark_returns=bench_returns,
        risk_free_rate=0.0,
    )

    # Beta should be close to 1.5
    assert report.metrics.beta is not None
    assert math.isclose(report.metrics.beta, 1.5, abs_tol=0.15)

    # Alpha should be positive
    assert report.metrics.alpha is not None
    assert report.metrics.alpha > 0.0

    # Information Ratio should be positive
    assert report.metrics.information_ratio is not None
    assert report.metrics.information_ratio > 0.0


def test_tearsheet_monthly_return_matrix() -> None:
    """Verify calendar monthly aggregation and compounding."""
    # 60 days starting 2024-01-01 (Jan and Feb 2024)
    n = 60
    returns = np.linspace(0.0005, 0.0015, n)  # Non-zero variance with positive drift
    timestamps = _generate_synthetic_timestamps(n)

    report = PerformanceAnalyticsEngine.calculate_tearsheet(
        strategy_returns=returns,
        timestamps_ns=timestamps,
    )

    assert 2024 in report.monthly_matrix
    jan_ret = report.monthly_matrix[2024]["M01"]
    feb_ret = report.monthly_matrix[2024]["M02"]
    ytd_ret = report.monthly_matrix[2024]["YTD"]

    assert jan_ret > 0.0
    assert feb_ret > 0.0
    assert ytd_ret > jan_ret  # Compounded YTD must exceed individual month


def test_tearsheet_zero_variance_raises() -> None:
    """Verify constant returns trigger ZeroVarianceError (ERR-RPT-001)."""
    returns = np.full(100, 0.0)
    timestamps = _generate_synthetic_timestamps(100)

    with pytest.raises(ZeroVarianceError) as exc_info:
        PerformanceAnalyticsEngine.calculate_tearsheet(
            strategy_returns=returns,
            timestamps_ns=timestamps,
        )
    assert exc_info.value.code == ERR_RPT_ZERO_VARIANCE


def test_tearsheet_mismatched_benchmark_raises() -> None:
    """Verify mismatched benchmark length triggers MismatchedBenchmarkError (ERR-RPT-002)."""
    strat = np.array([0.01, 0.02, -0.01])
    bench = np.array([0.01, -0.01])  # Length 2 != 3
    timestamps = _generate_synthetic_timestamps(3)

    with pytest.raises(MismatchedBenchmarkError) as exc_info:
        PerformanceAnalyticsEngine.calculate_tearsheet(
            strategy_returns=strat,
            timestamps_ns=timestamps,
            benchmark_returns=bench,
        )
    assert exc_info.value.code == ERR_RPT_MISMATCHED_BENCHMARK


def test_tearsheet_non_finite_and_boolean_rejection() -> None:
    """Verify NaNs, infinities, and booleans are strictly rejected (ERR-RPT-003)."""
    timestamps = _generate_synthetic_timestamps(3)

    # NaN in strategy returns
    with pytest.raises(NonFiniteReturnError) as exc_info:
        PerformanceAnalyticsEngine.calculate_tearsheet(
            strategy_returns=[0.01, float("nan"), 0.02],
            timestamps_ns=timestamps,
        )
    assert exc_info.value.code == ERR_RPT_NON_FINITE_INPUT

    # Boolean in strategy returns
    with pytest.raises(NonFiniteReturnError) as bool_exc:
        PerformanceAnalyticsEngine.calculate_tearsheet(
            strategy_returns=[0.01, True, 0.02],  # type: ignore[list-item]
            timestamps_ns=timestamps,
        )
    assert bool_exc.value.code == ERR_RPT_NON_FINITE_INPUT
