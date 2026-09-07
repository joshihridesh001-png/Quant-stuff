"""News event ingestion and temporal decay query endpoints."""

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from quant.api.dependencies import get_event_service, require_api_key
from quant.api.v1.schemas import ActiveStateResponse, EventIngestRequest, EventResponse
from quant.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["News Ingestion & State"])


@router.post(
    "/ingest",
    response_model=EventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest unstructured news event",
)
async def ingest_event(
    payload: EventIngestRequest,
    event_service: EventService = Depends(get_event_service),
    _api_key: str = Depends(require_api_key),
) -> EventResponse:
    """Ingest a news document with asset centrality mappings, triggering vector decomposition."""
    event = await event_service.ingest_event(
        headline=payload.headline,
        raw_text=payload.raw_text,
        timestamp=payload.timestamp,
        ticker_weights=payload.ticker_weights,
        dense_embedding=payload.dense_embedding,
        sentiment_polarity=payload.sentiment_polarity,
        sentiment_subjectivity=payload.sentiment_subjectivity,
        sentiment_novelty=payload.sentiment_novelty,
        urgency=payload.urgency,
        source=payload.source,
    )
    return EventResponse(
        id=event.id,
        headline=event.headline,
        timestamp=event.timestamp,
        sentiment_polarity=event.sentiment_polarity,
        sentiment_subjectivity=event.sentiment_subjectivity,
        sentiment_novelty=event.sentiment_novelty,
        urgency=event.urgency,
        source=event.source,
        created_at=event.created_at,
    )


@router.get(
    "/{event_id}",
    response_model=EventResponse,
    summary="Retrieve news event by ID",
)
async def get_event(
    event_id: UUID,
    event_service: EventService = Depends(get_event_service),
) -> EventResponse:
    """Fetch an ingested event by its unique UUID."""
    event = await event_service.event_repo.get_by_id(event_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    return EventResponse(
        id=event.id,
        headline=event.headline,
        timestamp=event.timestamp,
        sentiment_polarity=event.sentiment_polarity,
        sentiment_subjectivity=event.sentiment_subjectivity,
        sentiment_novelty=event.sentiment_novelty,
        urgency=event.urgency,
        source=event.source,
        created_at=event.created_at,
    )


@router.get(
    "/state/{ticker}",
    response_model=ActiveStateResponse,
    summary="Calculate active decayed news state vector",
)
async def get_active_news_state(
    ticker: str,
    as_of_time: datetime | None = Query(None, description="Point-in-time calculation horizon"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Fast decay weighting factor"),
    tau_fast: float = Query(3600.0, gt=0.0, description="Fast decay timescale in seconds"),
    tau_slow: float = Query(86400.0, gt=0.0, description="Slow decay timescale in seconds"),
    event_service: EventService = Depends(get_event_service),
) -> ActiveStateResponse:
    """Compute active time-decayed state vector S_news^{(k)}(t) for asset k."""
    calc_time = as_of_time or datetime.now(UTC)
    vector, count = await event_service.get_active_news_state(
        ticker=ticker,
        as_of_time=calc_time,
        alpha=alpha,
        tau_fast=tau_fast,
        tau_slow=tau_slow,
    )
    return ActiveStateResponse(
        ticker=ticker.upper(),
        as_of_time=calc_time,
        state_vector=vector,
        event_count=count,
    )
