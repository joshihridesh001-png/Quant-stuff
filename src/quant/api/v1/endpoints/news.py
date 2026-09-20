"""REST API endpoints for real-world financial news harvesting and causal price reaction predictions.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Endpoints: GET /latest, POST /harvest, POST /predict
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from quant.api.dependencies import (
    get_current_user,
    get_event_service,
    get_news_prediction_service,
)
from quant.services.event_service import EventService
from quant.services.news_prediction_service import NewsPredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["Financial News & Causal Price Reaction"])


class NewsPredictionRequest(BaseModel):
    """Payload schema for ad-hoc headline price reaction prediction."""

    headline: str = Field(..., min_length=3, description="Financial news headline.")
    summary: str = Field(default="", description="Detailed article summary or commentary.")
    ticker: str | None = Field(default=None, description="Optional target stock ticker symbol.")
    current_price: float = Field(default=100.0, gt=0.0, description="Current asset price.")
    volatility: float = Field(
        default=0.02, gt=0.0, description="Parkinson or historical volatility."
    )


class PriceReactionPredictionDTO(BaseModel):
    """Serialized representation of an econometric price reaction forecast."""

    ticker: str
    event_type: str
    current_price: float
    expected_delta_price: float
    target_price: float
    prob_up: float
    prob_down: float
    barrier_upper: float
    barrier_lower: float
    signal: str
    confidence: float
    predicted_trajectory: list[list[float]]
    calculation_latency_ms: float
    created_at: str


def _serialize_prediction(p: Any) -> dict[str, Any]:
    return {
        "ticker": p.ticker,
        "event_type": str(p.event_type),
        "current_price": round(p.current_price, 4),
        "expected_delta_price": round(p.expected_delta_price, 4),
        "target_price": round(p.target_price, 4),
        "prob_up": round(p.prob_up, 4),
        "prob_down": round(p.prob_down, 4),
        "barrier_upper": round(p.barrier_upper, 4),
        "barrier_lower": round(p.barrier_lower, 4),
        "signal": p.signal,
        "confidence": round(p.confidence, 4),
        "predicted_trajectory": [[round(t, 1), round(val, 4)] for t, val in p.predicted_trajectory],
        "calculation_latency_ms": round(p.calculation_latency_ms, 3),
        "created_at": p.created_at.isoformat(),
    }


@router.get(
    "/latest",
    response_model=list[PriceReactionPredictionDTO],
    summary="Get latest breaking news price reaction predictions",
)
async def get_latest_news_predictions(
    limit: int = Query(default=50, ge=1, le=200),
    service: NewsPredictionService = Depends(get_news_prediction_service),
    current_user: Any = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Retrieve the most recent financial news headlines and predicted price breakout targets."""
    predictions = service.get_latest_predictions(limit=limit)
    return [_serialize_prediction(p) for p in predictions]


@router.post(
    "/harvest",
    response_model=list[PriceReactionPredictionDTO],
    summary="Trigger immediate multi-source RSS/Atom wire harvest",
)
async def harvest_news_feed(
    service: NewsPredictionService = Depends(get_news_prediction_service),
    event_service: EventService = Depends(get_event_service),
    current_user: Any = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Poll all configured financial RSS feeds, classify breaking events, and compute price shocks."""
    try:
        if service._event_service is None:
            service._event_service = event_service
        predictions = await service.harvest_and_predict(fallback_to_synthetic=True)
        if hasattr(event_service.event_repo, "session"):
            await event_service.event_repo.session.commit()
        return [_serialize_prediction(p) for p in predictions]
    except Exception as e:
        logger.error("Error during news harvest: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"News harvest failed: {e}",
        ) from e


@router.post(
    "/predict",
    response_model=list[PriceReactionPredictionDTO],
    summary="Predict causal price reaction for a given headline",
)
async def predict_custom_headline(
    request: NewsPredictionRequest,
    service: NewsPredictionService = Depends(get_news_prediction_service),
    event_service: EventService = Depends(get_event_service),
    current_user: Any = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Classify headline sentiment, extract entities, and forecast Triple-Barrier price movement."""
    try:
        if service._event_service is None:
            service._event_service = event_service
        predictions = await service.predict_headline(
            headline=request.headline,
            summary=request.summary,
            ticker=request.ticker,
            current_price=request.current_price,
            volatility=request.volatility,
        )
        return [_serialize_prediction(p) for p in predictions]
    except Exception as e:
        logger.error("Error predicting headline: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process headline: {e}",
        ) from e
