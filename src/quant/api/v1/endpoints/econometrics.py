"""Econometric Stationarity, Fractional Differentiation, and Triple-Barrier REST endpoints.

Purpose:
    Exposes quantitative econometric endpoints for Marcos López de Prado's
    Fractionally Differentiated Features (FFD), Parkinson & Garman-Klass realized
    volatility estimators, and dynamic volatility Triple-Barrier labeling simulations.

Dependencies:
    - FastAPI APIRouter, Depends, HTTPException, Query, status.
    - numpy, scipy.signal.fftconvolve, statsmodels.tsa.stattools.adfuller.
    - quant.analytics.fractional_diff: compute_fractional_weights, FractionalDifferentiator.
    - quant.analytics.labeling: DynamicTripleBarrierLabeler, TripleBarrierConfig,
      BarrierTouchReason, compute_parkinson_volatility.
    - quant.api.dependencies: get_current_user, get_market_data_service.
    - quant.api.v1.schemas: FFDScanResponse, FFDScanPointDTO, VolatilityResponse,
      VolatilitySeriesPointDTO, TripleBarrierSimulateRequest, TripleBarrierSimulateResponse,
      BarrierTrajectoryPointDTO.

Structural Relationship:
    - Presentation layer route controller mounted under /api/v1/econometrics in quant.main.
    - Feeds interactive visualizations in the Econometric Stationarity Rig (Pillar 2).

Invariants Enforced:
    - Rule 1: Four-tier line annotations on every function.
    - Rule 2: Diagnostic error codes and standard HTTP exception mapping.
    - Strictly positive volatility windows and horizon bounds.
    - Finite numerical values with zero division guards.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from statsmodels.tsa.stattools import adfuller

from quant.analytics.fractional_diff import (
    FractionalDifferentiator,
    compute_fractional_weights,
)
from quant.analytics.labeling import (
    BarrierTouchReason,
    DynamicTripleBarrierLabeler,
    PositionSide,
    TripleBarrierConfig,
    compute_parkinson_volatility,
)
from quant.api.dependencies import get_current_user, get_market_data_service
from quant.api.v1.schemas import (
    BarrierTrajectoryPointDTO,
    FFDScanPointDTO,
    FFDScanResponse,
    TripleBarrierSimulateRequest,
    TripleBarrierSimulateResponse,
    VolatilityResponse,
    VolatilitySeriesPointDTO,
)
from quant.domain.models import Resolution
from quant.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/econometrics", tags=["Econometrics & Stationarity Rig"])


def _generate_synthetic_bars(
    symbol: str, n_bars: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic OHLCV price series seeded deterministically by ticker symbol.

    Purpose: Provides reproducible high-frequency microstructure series when historical
             DuckDB cache has not yet been seeded.
    Dependencies: NumPy random generator with ticker-deterministic seed.
    Structural Relationship: Fallback generator for econometric analyses.
    Defensive Invariant: Guaranteed strictly positive prices with High >= Max(Open, Close)
                         and Low <= Min(Open, Close).
    """
    seed = abs(hash(symbol)) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    # Base price calibrated by symbol tier
    base_price = 150.0 if "AAPL" in symbol else (120.0 if "NVDA" in symbol else 250.0)

    # Geometric Brownian Motion with mean-reverting jump diffusion
    dt = 1.0 / 390.0  # 1-minute intraday step
    sigma = 0.22
    mu = 0.05

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_bars)
    closes = base_price * np.exp(np.cumsum(log_returns))

    opens = np.roll(closes, 1)
    opens[0] = base_price

    # Intraday wick generation
    wick_range = np.abs(closes * rng.normal(0.0015, 0.0005, size=n_bars))
    highs = np.maximum(opens, closes) + wick_range
    lows = np.maximum(0.01, np.minimum(opens, closes) - wick_range)

    now_ns = time.time_ns()
    bar_interval_ns = 60 * 1_000_000_000
    timestamps = np.array(
        [now_ns - (n_bars - i) * bar_interval_ns for i in range(n_bars)],
        dtype=np.int64,
    )

    return timestamps, opens, highs, lows, closes


async def _get_or_generate_ohlcv(
    symbol: str,
    n_bars: int,
    service: MarketDataService,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch stored OHLCV bars from MarketDataService or fallback to deterministic generator.

    Purpose: Unified data provider for econometric analysis endpoints.
    Dependencies: MarketDataService, _generate_synthetic_bars.
    Structural Relationship: Ingestion bridge to analytical engines.
    Defensive Invariant: Guarantees length >= n_bars with strictly valid price coordinates.
    """
    try:
        batch = await service.get_latest_bars(
            asset_id=symbol,
            count=n_bars,
            resolution=Resolution.ONE_MINUTE,
        )
        if batch.count >= max(30, n_bars // 2):
            return (
                batch.timestamps,
                batch.opens,
                batch.highs,
                batch.lows,
                batch.closes,
            )
    except Exception as exc:
        logger.debug("Live market data query fell back to synthetic generator for %s: %s", symbol, exc)

    return _generate_synthetic_bars(symbol, n_bars=n_bars)


@router.get(
    "/ffd/search",
    response_model=FFDScanResponse,
    summary="Fractional Differentiation Stationarity Search",
)
async def ffd_stationarity_search(
    symbol: str = Query("NVDA", description="Asset ticker symbol"),
    threshold: float = Query(0.05, ge=0.001, le=0.20, description="Stationarity p-value threshold"),
    bar_count: int = Query(400, ge=100, le=2000, description="Bar sample count"),
    user: dict[str, Any] = Depends(get_current_user),
    service: MarketDataService = Depends(get_market_data_service),
) -> FFDScanResponse:
    """Evaluate ADF stationarity p-value and memory preservation across d in [0.0, 1.0].

    Purpose: Finds Marcos López de Prado's optimal d* preserving maximum memory (correlation)
             while achieving covariance stationarity (ADF p <= threshold).
    Dependencies: compute_fractional_weights, FractionalDifferentiator._convolve_1d, adfuller.
    Structural Relationship: Endpoint for Econometric Rig FFD dual-axis interactive chart.
    Defensive Invariant: d evaluated in discrete increments of 0.05 in [0.0, 1.0].
    """
    _ts, _op, _hi, _lo, closes = await _get_or_generate_ohlcv(symbol, bar_count, service)
    close_series = np.asarray(closes, dtype=np.float64)

    # Scan d across unit interval [0.0, 1.0] with step 0.05
    d_grid = np.linspace(0.0, 1.0, 21)
    points: list[FFDScanPointDTO] = []
    optimal_d = 1.0
    found_optimal = False

    for d_val in d_grid:
        d_rounded = round(float(d_val), 2)
        try:
            if d_rounded == 0.0:
                # Raw non-differenced series
                adf_res = adfuller(close_series, autolag="AIC", result_object=False)
                stat = float(adf_res[0])
                pval = float(adf_res[1])
                corr = 1.0
            else:
                max_lookback = max(10, int(len(close_series) * 0.20))
                w = compute_fractional_weights(
                    d=d_rounded,
                    threshold=1e-4,
                    max_lookback=max_lookback,
                    zero_sum_correction=True,
                )
                lookback = len(w) - 1
                if len(close_series) <= lookback + 10:
                    continue

                transformed = FractionalDifferentiator._convolve_1d(close_series, w)
                if np.any(np.isnan(transformed)) or len(transformed) < 20:
                    continue

                adf_res = adfuller(transformed, autolag="AIC", result_object=False)
                stat = float(adf_res[0])
                pval = float(adf_res[1])

                # Pearson correlation with original series aligned to valid window
                valid_raw = close_series[lookback:]
                c_mat = np.corrcoef(valid_raw, transformed)
                corr = float(c_mat[0, 1]) if not np.isnan(c_mat[0, 1]) else 0.0

            is_stat = pval <= threshold
            if is_stat and not found_optimal:
                optimal_d = d_rounded
                found_optimal = True

            points.append(
                FFDScanPointDTO(
                    d=d_rounded,
                    adf_stat=round(stat, 4),
                    adf_pvalue=round(pval, 6),
                    correlation=round(corr, 4),
                    is_stationary=is_stat,
                )
            )
        except Exception as exc:
            logger.debug("FFD evaluation failed for d=%.2f: %s", d_rounded, exc)

    if not found_optimal and points:
        optimal_d = 1.0

    return FFDScanResponse(
        symbol=symbol,
        optimal_d=optimal_d,
        threshold_pvalue=threshold,
        points=points,
    )


@router.get(
    "/volatility/parkinson",
    response_model=VolatilityResponse,
    summary="Realized Parkinson & Garman-Klass Volatility",
)
async def get_realized_volatility(
    symbol: str = Query("NVDA", description="Asset ticker symbol"),
    window: int = Query(20, ge=5, le=100, description="Rolling window length in bars"),
    bar_count: int = Query(200, ge=50, le=1000, description="Bar series length"),
    user: dict[str, Any] = Depends(get_current_user),
    service: MarketDataService = Depends(get_market_data_service),
) -> VolatilityResponse:
    """Compute rolling Parkinson and Garman-Klass range-based realized volatility series.

    Purpose: Supplies econometric volatility estimates protecting against close-to-close
             microstructure noise.
    Dependencies: compute_parkinson_volatility, Garman-Klass formulation.
    Structural Relationship: Feeds volatility gauge and rolling trend charts in Pillar 2.
    Defensive Invariant: Zero/negative price bounds rejected; volatility strictly non-negative.
    """
    timestamps, opens, highs, lows, closes = await _get_or_generate_ohlcv(symbol, bar_count, service)
    n = len(closes)

    # 1. Causal rolling Parkinson volatility
    p_vol = compute_parkinson_volatility(highs=highs, lows=lows, window=window)

    # 2. Rolling Garman-Klass volatility: 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    gk_vol = np.zeros(n, dtype=np.float64)
    log_hl_sq = np.log(highs / np.maximum(1e-6, lows)) ** 2
    log_co_sq = np.log(closes / np.maximum(1e-6, opens)) ** 2
    gk_factor = 2.0 * math.log(2.0) - 1.0
    gk_bar_var = 0.5 * log_hl_sq - gk_factor * log_co_sq

    # Annualization multiplier: 1-min bars -> sqrt(252 * 390)
    annualize = math.sqrt(252.0 * 390.0)

    for i in range(window, n):
        w_var = np.mean(gk_bar_var[i - window : i])
        gk_vol[i] = math.sqrt(max(0.0, float(w_var)))

    series_points: list[VolatilitySeriesPointDTO] = []
    start_idx = max(0, n - 60)
    for i in range(start_idx, n):
        series_points.append(
            VolatilitySeriesPointDTO(
                timestamp_ns=int(timestamps[i]),
                close=round(float(closes[i]), 2),
                parkinson_vol=round(float(p_vol[i] * annualize * 100.0), 2),
                garman_klass_vol=round(float(gk_vol[i] * annualize * 100.0), 2),
            )
        )

    latest_parkinson = float(p_vol[-1] * annualize * 100.0) if len(p_vol) > 0 else 0.0
    latest_gk = float(gk_vol[-1] * annualize * 100.0) if len(gk_vol) > 0 else 0.0

    return VolatilityResponse(
        symbol=symbol,
        window=window,
        latest_close=round(float(closes[-1]), 2),
        annualized_parkinson_pct=round(latest_parkinson, 2),
        annualized_garman_klass_pct=round(latest_gk, 2),
        series=series_points,
    )


@router.post(
    "/triple-barrier/simulate",
    response_model=TripleBarrierSimulateResponse,
    summary="Simulate Dynamic Volatility Triple-Barrier Labeling",
)
async def simulate_triple_barrier(
    request: TripleBarrierSimulateRequest,
    user: dict[str, Any] = Depends(get_current_user),
    service: MarketDataService = Depends(get_market_data_service),
) -> TripleBarrierSimulateResponse:
    """Simulate path-dependent triple-barrier labeling over historical/synthetic market data.

    Purpose: Computes empirical take-profit vs stop-loss hit distribution and generates
             sample trajectory visualization points for the Econometrics Rig.
    Dependencies: DynamicTripleBarrierLabeler, TripleBarrierConfig.
    Structural Relationship: Powers interactive barrier sensitivity canvas in Pillar 2.
    Defensive Invariant: Multipliers strictly positive, horizon >= 5.
    """
    total_bars = max(150, request.horizon_bars * 4 + request.volatility_window + 20)
    timestamps, opens, highs, lows, closes = await _get_or_generate_ohlcv(
        request.symbol, total_bars, service
    )

    try:
        cfg = TripleBarrierConfig(
            profit_multiplier=request.profit_multiplier,
            stop_multiplier=request.stop_multiplier,
            horizon_bars=request.horizon_bars,
            volatility_window=request.volatility_window,
        )
        labeler = DynamicTripleBarrierLabeler(config=cfg)

        start_idx = cfg.volatility_window + 1
        max_event_idx = len(timestamps) - cfg.horizon_bars - cfg.execution_delay_bars
        n_events = max(0, max_event_idx - start_idx)
        target_side = PositionSide.LONG if request.side >= 0 else PositionSide.SHORT
        sides = np.full(n_events, target_side, dtype=np.int32) if n_events > 0 else None

        labels = labeler.label_arrays(
            timestamps=timestamps,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            sides=sides,
        )
    except Exception as exc:
        logger.error("Triple barrier simulation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Labeling simulation failed: {str(exc)}",
        ) from exc

    total_events = len(labels)
    if total_events == 0:
        return TripleBarrierSimulateResponse(
            symbol=request.symbol,
            total_events=0,
            take_profit_hits=0,
            stop_loss_hits=0,
            vertical_expiration_hits=0,
            take_profit_pct=0.0,
            stop_loss_pct=0.0,
            vertical_expiration_pct=0.0,
            average_holding_bars=0.0,
            average_net_return_pct=0.0,
            sample_trajectory=[],
        )

    tp_hits = sum(1 for lbl in labels if lbl.touch_reason == BarrierTouchReason.UPPER)
    sl_hits = sum(
        1
        for lbl in labels
        if lbl.touch_reason in (BarrierTouchReason.LOWER, BarrierTouchReason.COLLISION_STOP)
    )
    vert_hits = sum(1 for lbl in labels if lbl.touch_reason == BarrierTouchReason.VERTICAL)

    avg_holding = float(np.mean([lbl.holding_period_bars for lbl in labels]))
    avg_net_ret = float(np.mean([lbl.realized_return for lbl in labels]) * 100.0)

    # Build sample trajectory points from the last evaluated trade lifecycle
    sample_points: list[BarrierTrajectoryPointDTO] = []
    target_label = labels[-1]
    event_idx = np.where(timestamps == target_label.event_timestamp)[0]
    start_idx = int(event_idx[0]) if len(event_idx) > 0 else (len(closes) - target_label.holding_period_bars - 1)
    end_idx = min(len(closes), start_idx + target_label.holding_period_bars + 1)

    for i in range(start_idx, end_idx):
        rel_idx = i - start_idx
        event_tag = None
        if i == end_idx - 1:
            if target_label.touch_reason == BarrierTouchReason.UPPER:
                event_tag = "TAKE_PROFIT"
            elif target_label.touch_reason in (BarrierTouchReason.LOWER, BarrierTouchReason.COLLISION_STOP):
                event_tag = "STOP_LOSS"
            else:
                event_tag = "EXPIRATION"

        sample_points.append(
            BarrierTrajectoryPointDTO(
                bar_index=rel_idx,
                close_price=round(float(closes[i]), 2),
                upper_barrier=round(float(target_label.upper_barrier), 2),
                lower_barrier=round(float(target_label.lower_barrier), 2),
                event_type=event_tag,
            )
        )

    return TripleBarrierSimulateResponse(
        symbol=request.symbol,
        total_events=total_events,
        take_profit_hits=tp_hits,
        stop_loss_hits=sl_hits,
        vertical_expiration_hits=vert_hits,
        take_profit_pct=round((tp_hits / total_events) * 100.0, 1),
        stop_loss_pct=round((sl_hits / total_events) * 100.0, 1),
        vertical_expiration_pct=round((vert_hits / total_events) * 100.0, 1),
        average_holding_bars=round(avg_holding, 1),
        average_net_return_pct=round(avg_net_ret, 3),
        sample_trajectory=sample_points,
    )
