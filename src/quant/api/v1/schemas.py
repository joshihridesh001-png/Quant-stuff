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


class ActiveStateResponse(BaseModel):
    ticker: str
    as_of_time: datetime
    state_vector: list[float]
    event_count: int


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


class PopulationSeedRequest(BaseModel):
    population_size: int = Field(20, ge=4, le=500)


# Auth Schemas
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
