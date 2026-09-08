"""Pydantic request and response schemas (DTOs) for API v1."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# Event Schemas
class EventIngestRequest(BaseModel):
    headline: str = Field(..., min_length=3, max_length=500)
    raw_text: str = Field(..., min_length=10)
    timestamp: datetime
    ticker_weights: dict[str, float] = Field(
        ..., description="Mapping of ticker to centrality weight c_{i,k}"
    )
    dense_embedding: list[float] | None = Field(
        default=None, description="Optional dense Transformer vector"
    )
    sentiment_polarity: float = Field(0.0, ge=-1.0, le=1.0)
    sentiment_subjectivity: float = Field(0.0, ge=0.0, le=1.0)
    sentiment_novelty: float = Field(0.0, ge=0.0, le=1.0)
    urgency: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field("GENERIC", max_length=50)


class EventResponse(BaseModel):
    id: UUID
    headline: str
    timestamp: datetime
    sentiment_polarity: float
    sentiment_subjectivity: float
    sentiment_novelty: float
    urgency: float
    source: str
    created_at: datetime


class BatchEventIngestRequest(BaseModel):
    events: list[EventIngestRequest] = Field(
        ..., min_length=1, max_length=500, description="Batch of news events to ingest atomically"
    )


class BatchEventIngestResponse(BaseModel):
    ingested_count: int
    events: list[EventResponse]


class ActiveStateResponse(BaseModel):
    ticker: str
    as_of_time: datetime
    state_vector: list[float]
    event_count: int
    use_projected_subspace: bool = False


# Genotype Schemas
class GenotypeCreateRequest(BaseModel):
    generation: int = Field(0, ge=0)
    cohort: str = Field("ASPIRANT", pattern="^(ALPHA|ASPIRANT)$")
    chromosome_repr: dict[str, Any]
    chromosome_game: dict[str, Any]
    chromosome_infer: dict[str, Any]
    chromosome_risk: dict[str, Any]


class GenotypeEvaluateRequest(BaseModel):
    deflated_sharpe: float = Field(..., description="Deflated Sharpe Ratio (DSR)")
    max_drawdown: float = Field(..., ge=0.0, description="Historical Max Drawdown")
    regret_score: float = Field(0.0, description="Minimax Regret metric")
    novelty_score: float = Field(0.0, description="Phenotypic novelty distance")


class GenotypeResponse(BaseModel):
    id: UUID
    generation: int
    cohort: str
    chromosome_repr: dict[str, Any]
    chromosome_game: dict[str, Any]
    chromosome_infer: dict[str, Any]
    chromosome_risk: dict[str, Any]
    fitness_score: float | None = None
    deflated_sharpe: float | None = None
    max_drawdown: float | None = None
    regret_score: float | None = None
    novelty_score: float | None = None
    created_at: datetime


class ParetoRankedGenotypeResponse(BaseModel):
    genotype: GenotypeResponse
    pareto_rank: int
    crowding_distance: float


class PopulationSeedRequest(BaseModel):
    population_size: int = Field(20, ge=4, le=500)


# Auth Schemas
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Market Data Schemas
class PriceBarDTO(BaseModel):
    """Data transfer object for a single discrete price bar."""

    asset_id: str = Field(..., min_length=1, max_length=30, description="Asset ticker symbol")
    timestamp: int = Field(..., gt=0, description="Unix epoch nanoseconds timestamp")
    open: float = Field(..., gt=0.0, description="Opening price")
    high: float = Field(..., gt=0.0, description="Highest price in interval")
    low: float = Field(..., gt=0.0, description="Lowest price in interval")
    close: float = Field(..., gt=0.0, description="Closing price")
    volume: float = Field(..., ge=0.0, description="Traded volume")
    vwap: float = Field(..., ge=0.0, description="Volume-Weighted Average Price")
    resolution: str = Field("1m", description="Bar sampling resolution (e.g. 1m, 5m, 1h, 1d)")


class BatchPriceBarIngestRequest(BaseModel):
    """Batch ingestion request for high-throughput price bars."""

    resolution: str = Field("1m", description="Resolution applied to incoming batch")
    bars: list[PriceBarDTO] = Field(
        ..., min_length=1, max_length=10000, description="Chronological list of price bars"
    )


class BatchPriceBarIngestResponse(BaseModel):
    """Batch ingestion response reporting processed and persisted bar counts."""

    processed_count: int
    persisted_count: int
    resolution: str


class MarketDataBatchSummaryResponse(BaseModel):
    """Summary response for historical bar range queries."""

    asset_id: str
    resolution: str
    count: int
    start_time: int | None
    end_time: int | None
    latest_close: float | None
    realized_volatility_latest: float | None
