"""CFA-Grade Performance Analytics Engine & Mathematical Attribution Calculator.

Functional Purpose:
    Computes institutional CFA/CAIA performance metrics, underwater drawdowns, monthly
    return heatmaps, and CAPM alpha/beta attribution against benchmark equity curves.
    Powers standalone HTML tear sheets, backtest reporting, and live HUD analytics.

Explicit Dependency Tracking:
    - numpy: Vectorized moment calculations, covariance, and quantiles.
    - datetime: Calendar monthly partitioning.

Structural Relationship:
    Analytical foundation of Phase 15. Consumed by HtmlReportGenerator, FastAPI backtest
    endpoints, and the Web BacktestStudioView.

Defensive Invariants:
    - Zero Variance Guard: std(r) > 1e-8 required for Sharpe/Sortino ratios (ERR-RPT-001).
    - Benchmark Dimension Alignment: len(benchmark) == len(strategy) (ERR-RPT-002).
    - Strict Non-Finite Rejection: All input return elements must be finite floats (ERR-RPT-003).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_RPT_ZERO_VARIANCE: Final[str] = "ERR-RPT-001"
ERR_RPT_MISMATCHED_BENCHMARK: Final[str] = "ERR-RPT-002"
ERR_RPT_NON_FINITE_INPUT: Final[str] = "ERR-RPT-003"


# ============================================================================
# Exception Taxonomy
# ============================================================================


class TearsheetError(Exception):
    """Base exception for all reporting and performance calculation failures."""

    def __init__(self, message: str, code: str = "ERR-RPT-000") -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class ZeroVarianceError(TearsheetError):
    """Raised when return series has zero variance, preventing ratio calculations."""

    def __init__(self, message: str, code: str = ERR_RPT_ZERO_VARIANCE) -> None:
        super().__init__(message, code=code)


class MismatchedBenchmarkError(TearsheetError):
    """Raised when benchmark series length differs from strategy series length."""

    def __init__(self, message: str, code: str = ERR_RPT_MISMATCHED_BENCHMARK) -> None:
        super().__init__(message, code=code)


class NonFiniteReturnError(TearsheetError):
    """Raised when input return vector contains NaNs, infinities, or booleans."""

    def __init__(self, message: str, code: str = ERR_RPT_NON_FINITE_INPUT) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Performance Analytics DTOs
# ============================================================================


@dataclass(frozen=True, slots=True)
class CFAMetrics:
    """Comprehensive institutional performance statistics."""

    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    max_drawdown_duration_bars: int
    win_rate: float
    profit_factor: float
    gain_to_pain_ratio: float
    tail_ratio: float
    var_95: float
    cvar_95: float
    total_trades_bars: int
    alpha: float | None = None
    beta: float | None = None
    information_ratio: float | None = None
    tracking_error: float | None = None


@dataclass(frozen=True, slots=True)
class MonthlyReturnRecord:
    """Monthly calendar return entry."""

    year: int
    month: int
    monthly_return: float


@dataclass(frozen=True, slots=True)
class TearsheetReport:
    """Immutable comprehensive performance tear sheet container."""

    metrics: CFAMetrics
    timestamps: np.ndarray
    equity_curve: np.ndarray
    drawdown_series: np.ndarray
    benchmark_equity_curve: np.ndarray | None
    monthly_matrix: dict[int, dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Performance Analytics Calculator
# ============================================================================


class PerformanceAnalyticsEngine:
    """Pure analytical engine producing CFA-grade quant statistics and attribution."""

    @staticmethod
    def calculate_tearsheet(
        strategy_returns: Sequence[float] | np.ndarray,
        timestamps_ns: Sequence[int] | np.ndarray,
        benchmark_returns: Sequence[float] | np.ndarray | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        metadata: dict[str, Any] | None = None,
    ) -> TearsheetReport:
        """Compute full tearsheet analytics over strategy and benchmark return series.

        Defensive Invariants:
            - len(strategy_returns) >= 2.
            - All returns must be finite floats (no NaNs, no booleans).
            - len(benchmark_returns) == len(strategy_returns) if benchmark provided.
        """
        strat_arr = np.asarray(strategy_returns, dtype=np.float64)
        ts_arr = np.asarray(timestamps_ns, dtype=np.int64)

        if len(strat_arr) < 2:
            raise TearsheetError(
                f"strategy_returns must contain at least 2 observations, got {len(strat_arr)}"
            )
        if len(ts_arr) != len(strat_arr):
            raise TearsheetError(
                f"timestamps length ({len(ts_arr)}) does not match returns length ({len(strat_arr)})"
            )

        # Non-finite and boolean rejection
        for idx, val in enumerate(strategy_returns):
            if isinstance(val, bool) or not math.isfinite(float(val)):
                raise NonFiniteReturnError(
                    f"Non-finite or boolean return detected at index {idx}: {val!r}"
                )

        bench_arr: np.ndarray | None = None
        if benchmark_returns is not None:
            bench_arr = np.asarray(benchmark_returns, dtype=np.float64)
            if len(bench_arr) != len(strat_arr):
                raise MismatchedBenchmarkError(
                    f"Benchmark length ({len(bench_arr)}) does not match strategy length ({len(strat_arr)})"
                )
            for idx, val in enumerate(benchmark_returns):
                if isinstance(val, bool) or not math.isfinite(float(val)):
                    raise NonFiniteReturnError(
                        f"Non-finite return in benchmark at index {idx}: {val!r}"
                    )

        # 1. Equity Curves and Drawdowns
        equity = np.cumprod(1.0 + strat_arr)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / np.maximum(peaks, 1e-8)
        mdd = float(np.min(drawdowns))

        # Max drawdown duration in bars
        mdd_duration = 0
        curr_duration = 0
        for dd in drawdowns:
            if dd < -1e-6:
                curr_duration += 1
                mdd_duration = max(mdd_duration, curr_duration)
            else:
                curr_duration = 0

        # Benchmark equity curve
        bench_equity: np.ndarray | None = None
        if bench_arr is not None:
            bench_equity = np.cumprod(1.0 + bench_arr)

        # 2. Return & Volatility Metrics
        total_ret = float(equity[-1] - 1.0)
        t_years = len(strat_arr) / float(periods_per_year)
        if t_years > 0 and equity[-1] > 0:
            cagr = float(math.pow(equity[-1], 1.0 / t_years) - 1.0)
        else:
            cagr = 0.0

        daily_std = float(np.std(strat_arr))
        if daily_std < 1e-8:
            raise ZeroVarianceError(
                f"Strategy returns have zero variance (std={daily_std:.2e}), cannot compute Sharpe ratio"
            )

        ann_vol = daily_std * math.sqrt(periods_per_year)

        # 3. Sharpe & Sortino
        daily_rf = risk_free_rate / float(periods_per_year)
        excess_returns = strat_arr - daily_rf
        sharpe = float(np.mean(excess_returns) / daily_std * math.sqrt(periods_per_year))

        downside_diff = np.minimum(excess_returns, 0.0)
        downside_std = float(np.sqrt(np.mean(downside_diff**2))) * math.sqrt(periods_per_year)
        sortino = float((cagr - risk_free_rate) / downside_std) if downside_std > 1e-8 else 0.0

        # Calmar Ratio
        calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-6 else 0.0

        # 4. Omega Ratio (relative to daily risk free rate)
        pos_excess = np.sum(np.maximum(strat_arr - daily_rf, 0.0))
        neg_excess = np.sum(np.maximum(daily_rf - strat_arr, 0.0))
        omega = float(pos_excess / neg_excess) if neg_excess > 1e-8 else 0.0

        # 5. Win Rate & Profit Factor
        pos_rets = strat_arr[strat_arr > 0.0]
        neg_rets = strat_arr[strat_arr < 0.0]
        win_rate = float(len(pos_rets) / len(strat_arr))
        profit_factor = (
            float(np.sum(pos_rets) / abs(np.sum(neg_rets)))
            if len(neg_rets) > 0 and abs(np.sum(neg_rets)) > 1e-8
            else 0.0
        )

        # 6. Gain to Pain & Tail Ratio
        total_pnl = float(np.sum(strat_arr))
        total_pain = float(np.sum(np.maximum(-strat_arr, 0.0)))
        gain_to_pain = float(total_pnl / total_pain) if total_pain > 1e-8 else 0.0

        p95 = float(np.percentile(strat_arr, 95))
        p5 = float(np.percentile(strat_arr, 5))
        tail_ratio = float(abs(p95 / p5)) if abs(p5) > 1e-8 else 0.0

        var_95 = float(abs(p5))
        tail_losses = strat_arr[strat_arr <= p5]
        cvar_95 = float(abs(np.mean(tail_losses))) if len(tail_losses) > 0 else var_95

        # 7. Benchmark Alpha & Beta Attribution
        alpha_val: float | None = None
        beta_val: float | None = None
        ir_val: float | None = None
        te_val: float | None = None

        if bench_arr is not None:
            bench_var = float(np.var(bench_arr))
            if bench_var > 1e-8:
                cov_sb = float(np.cov(strat_arr, bench_arr)[0, 1])
                beta_val = float(cov_sb / bench_var)
                alpha_daily = float(np.mean(strat_arr) - beta_val * np.mean(bench_arr))
                alpha_val = float(alpha_daily * periods_per_year)

                active_diff = strat_arr - bench_arr
                te_val = float(np.std(active_diff) * math.sqrt(periods_per_year))
                ir_val = float(
                    np.mean(active_diff) * math.sqrt(periods_per_year) / max(te_val, 1e-8)
                )

        # 8. Monthly Return Matrix
        monthly_matrix = PerformanceAnalyticsEngine._build_monthly_matrix(strat_arr, ts_arr)

        metrics = CFAMetrics(
            total_return=round(total_ret, 4),
            cagr=round(cagr, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            omega_ratio=round(omega, 3),
            max_drawdown=round(mdd, 4),
            max_drawdown_duration_bars=mdd_duration,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 3),
            gain_to_pain_ratio=round(gain_to_pain, 3),
            tail_ratio=round(tail_ratio, 3),
            var_95=round(var_95, 4),
            cvar_95=round(cvar_95, 4),
            total_trades_bars=len(strat_arr),
            alpha=round(alpha_val, 4) if alpha_val is not None else None,
            beta=round(beta_val, 4) if beta_val is not None else None,
            information_ratio=round(ir_val, 3) if ir_val is not None else None,
            tracking_error=round(te_val, 4) if te_val is not None else None,
        )

        return TearsheetReport(
            metrics=metrics,
            timestamps=ts_arr,
            equity_curve=equity,
            drawdown_series=drawdowns,
            benchmark_equity_curve=bench_equity,
            monthly_matrix=monthly_matrix,
            metadata=metadata or {},
        )

    @staticmethod
    def _build_monthly_matrix(
        returns: np.ndarray, timestamps_ns: np.ndarray
    ) -> dict[int, dict[str, float]]:
        """Group returns into Year x Month grid with compounding and YTD totals."""
        by_year_month: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

        for ret, ts in zip(returns, timestamps_ns, strict=True):
            dt = datetime.fromtimestamp(ts / 1e9, tz=UTC)
            by_year_month[dt.year][dt.month].append(float(ret))

        matrix: dict[int, dict[str, float]] = {}
        for year in sorted(by_year_month.keys()):
            matrix[year] = {}
            ytd_equity = 1.0
            for month in range(1, 13):
                m_rets = by_year_month[year].get(month, [])
                if m_rets:
                    m_compound = float(np.prod([1.0 + r for r in m_rets]) - 1.0)
                    matrix[year][f"M{month:02d}"] = round(m_compound, 4)
                    ytd_equity *= 1.0 + m_compound
                else:
                    matrix[year][f"M{month:02d}"] = 0.0

            matrix[year]["YTD"] = round(ytd_equity - 1.0, 4)

        return matrix
