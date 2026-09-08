"""Market data REST endpoints for batch bar ingestion and historical columnar queries.

Purpose: Exposes machine-to-machine bar ingestion and analytical query endpoints over HTTP.
Dependencies: FastAPI router, MarketDataService, Pydantic schemas, security dependencies.
Relationship: Part of API v1 routing; handles market data persistence and retrieval requests.
Invariants: Ingestion requires API key; retrieval requires authenticated JWT.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

# Application dependencies and security
from quant.api.dependencies import get_current_user, get_market_data_service, require_api_key
from quant.api.v1.schemas import (
    BatchPriceBarIngestRequest,
    BatchPriceBarIngestResponse,
    MarketDataBatchSummaryResponse,
)
from quant.domain.models import Resolution
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
