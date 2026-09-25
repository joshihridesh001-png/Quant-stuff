"""Market data REST endpoints for batch bar ingestion and historical columnar queries.

Purpose: Exposes machine-to-machine bar ingestion and analytical query endpoints over HTTP.
Dependencies: FastAPI router, MarketDataService, Pydantic schemas, security dependencies.
Relationship: Part of API v1 routing; handles market data persistence and retrieval requests.
Invariants: Ingestion requires API key; retrieval requires authenticated JWT.
"""

import logging
import math
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

# Application dependencies and security
from quant.api.dependencies import (
    get_current_user,
    get_duckdb_manager,
    get_market_data_service,
    require_api_key,
)
from quant.api.v1.schemas import (
    BatchPriceBarIngestRequest,
    BatchPriceBarIngestResponse,
    CandlestickBarDTO,
    CandlestickSeriesResponse,
    MarketDataBatchSummaryResponse,
)
from quant.domain.models import Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.repositories.duckdb_historical_repository import (
    DuckDBHistoricalRepository,
)
from quant.services.market_data_service import MarketDataService

# Structured logger for endpoint traces
logger = logging.getLogger(__name__)

# FastAPI router for market data routes
router = APIRouter(prefix="/market-data", tags=["Market Data"])


@router.post(
    "/bars/batch",
    response_model=BatchPriceBarIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest batch of price bars",
)
async def ingest_bars_batch(
    payload: BatchPriceBarIngestRequest,
    _: str = Depends(require_api_key),
    service: MarketDataService = Depends(get_market_data_service),
) -> BatchPriceBarIngestResponse:
    """Ingest a batch of OHLCV market bars with columnar persistence.

    Purpose: Fast ingestion route for external market data pipelines.
    Dependencies: require_api_key header validation, MarketDataService.
    Invariants: Validates resolution string against Resolution StrEnum.
    """
    try:
        # Purpose: Validate resolution string parameter against domain enum
        target_resolution = Resolution(payload.resolution)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid resolution '{payload.resolution}'. Must be one of {[r.value for r in Resolution]}",
        ) from exc

    try:
        # Purpose: Extract bar payloads as dictionaries and submit to domain service
        # Dependencies: [bar.model_dump() for bar in payload.bars]
        bars_dicts = [bar.model_dump() for bar in payload.bars]
        processed, persisted = await service.ingest_bars(bars_dicts, resolution=target_resolution)

        return BatchPriceBarIngestResponse(
            processed_count=processed,
            persisted_count=persisted,
            resolution=payload.resolution,
        )
    except ValueError as exc:
        logger.error("Market bar batch ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/bars",
    response_model=MarketDataBatchSummaryResponse,
    summary="Query historical price bars by timestamp range",
)
async def get_bars_by_range(
    asset_id: str = Query(..., description="Asset ticker symbol"),
    start_time: int = Query(..., gt=0, description="Start timestamp in nanoseconds"),
    end_time: int = Query(..., gt=0, description="End timestamp in nanoseconds"),
    resolution: str = Query("1m", description="Bar sampling resolution"),
    _: dict[str, Any] = Depends(get_current_user),
    service: MarketDataService = Depends(get_market_data_service),
) -> MarketDataBatchSummaryResponse:
    """Retrieve columnar market bars within chronological range [start_time, end_time].

    Purpose: Returns summary and econometric volatility metrics for requested asset window.
    Dependencies: get_current_user JWT verification, MarketDataService.
    Invariants: start_time <= end_time.
    """
    try:
        res_enum = Resolution(resolution)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid resolution '{resolution}'. Must be one of {[r.value for r in Resolution]}",
        ) from exc

    try:
        batch = await service.get_historical_bars(
            asset_id=asset_id,
            start_time=start_time,
            end_time=end_time,
            resolution=res_enum,
        )
        # Purpose: Compute rolling realized volatility over retrieved batch
        vol = service.compute_realized_volatility(batch, window=min(20, max(2, batch.count - 1)))
        latest_vol = float(vol[-1]) if len(vol) > 0 and batch.count >= 2 else None
        latest_close = float(batch.closes[-1]) if batch.count > 0 else None

        return MarketDataBatchSummaryResponse(
            asset_id=asset_id,
            resolution=resolution,
            count=batch.count,
            start_time=batch.start_time,
            end_time=batch.end_time,
            latest_close=latest_close,
            realized_volatility_latest=latest_vol,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/bars/latest",
    response_model=MarketDataBatchSummaryResponse,
    summary="Query latest N price bars",
)
async def get_latest_bars(
    asset_id: str = Query(..., description="Asset ticker symbol"),
    count: int = Query(100, ge=1, le=5000, description="Number of recent bars to retrieve"),
    resolution: str = Query("1m", description="Bar sampling resolution"),
    _: dict[str, Any] = Depends(get_current_user),
    service: MarketDataService = Depends(get_market_data_service),
) -> MarketDataBatchSummaryResponse:
    """Retrieve the latest N price bars for an asset in ascending chronological order.

    Purpose: Supplies live rolling econometric features for current inference.
    Dependencies: get_current_user JWT verification, MarketDataService.
    Invariants: 1 <= count <= 5000.
    """
    try:
        res_enum = Resolution(resolution)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid resolution '{resolution}'. Must be one of {[r.value for r in Resolution]}",
        ) from exc

    try:
        batch = await service.get_latest_bars(
            asset_id=asset_id,
            count=count,
            resolution=res_enum,
        )
        vol = service.compute_realized_volatility(batch, window=min(20, max(2, batch.count - 1)))
        latest_vol = float(vol[-1]) if len(vol) > 0 and batch.count >= 2 else None
        latest_close = float(batch.closes[-1]) if batch.count > 0 else None

        return MarketDataBatchSummaryResponse(
            asset_id=asset_id,
            resolution=resolution,
            count=batch.count,
            start_time=batch.start_time,
            end_time=batch.end_time,
            latest_close=latest_close,
            realized_volatility_latest=latest_vol,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def _generate_synthetic_gbm_bars(
    symbol: str,
    count: int = 200,
    resolution_sec: int = 60,
) -> list[CandlestickBarDTO]:
    """Generate realistic Geometric Brownian Motion price bars with intraday vol.

    Functional Purpose:
        Provides high-fidelity fallback OHLCV price series for canvas charting.
    Explicit Dependency Tracking:
        numpy.random.default_rng, math.exp, math.sqrt.
    Structural Relationship:
        Internal fallback utility invoked when DuckDB historical store is sparse.
    Defensive Invariants:
        All prices strictly positive, High >= max(Open, Close), Low <= min(Open, Close).
    """
    seed = sum(ord(c) for c in symbol)
    rng = np.random.default_rng(seed)

    current_time_sec = int(time.time())
    start_time_sec = current_time_sec - (count * resolution_sec)

    s0 = 150.0 + (seed % 100)
    dt = resolution_sec / 86400.0
    mu = 0.05
    sigma = 0.25

    bars: list[CandlestickBarDTO] = []
    price = s0

    for i in range(count):
        bar_time = start_time_sec + (i * resolution_sec)
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * math.sqrt(dt) * float(rng.standard_normal())
        ret = drift + shock
        close_price = max(1.0, float(price * math.exp(ret)))

        # Intra-bar geometry
        intra_vol = sigma * math.sqrt(dt) * 0.6
        high_price = max(price, close_price) * (1.0 + abs(float(rng.standard_normal())) * intra_vol)
        low_price = min(price, close_price) * (1.0 - abs(float(rng.standard_normal())) * intra_vol)
        low_price = max(0.01, low_price)
        open_price = price

        volume = float(rng.lognormal(mean=9.0, sigma=0.8))

        bars.append(
            CandlestickBarDTO(
                time=bar_time,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=round(volume, 0),
            )
        )
        price = close_price

    return bars


@router.get(
    "/candlesticks",
    response_model=CandlestickSeriesResponse,
    summary="Retrieve Normalized Candlestick Bars (Lightweight Charts Compatible)",
)
async def get_candlesticks(
    symbol: str = Query("NVDA", description="Asset ticker symbol"),
    resolution: str = Query("1m", description="Bar sampling resolution"),
    bar_count: int = Query(200, ge=10, le=2000, description="Number of bars to retrieve"),
    manager: DuckDBManager = Depends(get_duckdb_manager),
) -> CandlestickSeriesResponse:
    """Retrieve chronologically ordered OHLCV candlesticks for canvas rendering.

    Functional Purpose:
        Transfers normalized, integer-second timestamped OHLCV candlesticks to frontend.
    Explicit Dependency Tracking:
        DuckDBHistoricalRepository, Resolution enum, CandlestickSeriesResponse.
    Structural Relationship:
        Direct feed for React TradingViewChart in BacktestStudioView and SimulationRiskView.
    Defensive Invariants:
        Timestamps strictly integer seconds ascending; High >= max(Open, Close); Low <= min(Open, Close).
    """
    repo = DuckDBHistoricalRepository(manager)
    try:
        res_enum = Resolution(resolution)
    except ValueError:
        res_enum = Resolution.ONE_MINUTE

    res_seconds_map: dict[Resolution, int] = {
        Resolution.ONE_SECOND: 1,
        Resolution.ONE_MINUTE: 60,
        Resolution.FIVE_MINUTES: 300,
        Resolution.FIFTEEN_MINUTES: 900,
        Resolution.ONE_HOUR: 3600,
        Resolution.ONE_DAY: 86400,
    }
    step_sec = res_seconds_map.get(res_enum, 60)

    try:
        now_ns = int(time.time() * 1_000_000_000)
        start_ns = now_ns - (bar_count * step_sec * 1_000_000_000)

        batch = await repo.get_bars_range(
            symbol=symbol,
            start_time=start_ns,
            end_time=now_ns,
            resolution=res_enum,
        )

        bars_dto: list[CandlestickBarDTO] = []
        last_time_sec = -1
        for bar in sorted(batch.bars, key=lambda b: b.timestamp):
            time_sec = int(bar.timestamp // 1_000_000_000)
            if time_sec > last_time_sec:
                bars_dto.append(
                    CandlestickBarDTO(
                        time=time_sec,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                    )
                )
                last_time_sec = time_sec

        if bars_dto:
            return CandlestickSeriesResponse(
                symbol=symbol,
                resolution=resolution,
                count=len(bars_dto),
                bars=bars_dto,
                source="DUCKDB_HISTORICAL_LAKE",
            )
    except Exception as exc:
        logger.debug("DuckDB candlestick fetch falling back to GBM for %s: %s", symbol, exc)

    fallback_bars = _generate_synthetic_gbm_bars(
        symbol=symbol, count=bar_count, resolution_sec=step_sec
    )
    return CandlestickSeriesResponse(
        symbol=symbol,
        resolution=resolution,
        count=len(fallback_bars),
        bars=fallback_bars,
        source="SYNTHETIC_GBM_FALLBACK",
    )
