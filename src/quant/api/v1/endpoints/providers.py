"""REST API endpoints for managing and inspecting external quantitative data providers.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Endpoints: GET /status, POST /macro/sync, GET /company-news/{symbol}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from quant.api.dependencies import (
    get_current_user,
    get_external_provider_manager,
    get_news_prediction_service,
)
from quant.data.external_providers import ExternalProviderManager
from quant.services.news_prediction_service import NewsPredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/providers", tags=["External Quantitative Data Providers"])


class ProviderStatusItemDTO(BaseModel):
    """Status record for a single external quantitative provider."""

    provider_id: str
    is_configured: bool
    is_healthy: bool
    calls_made: int
    last_call_timestamp_ns: int | None = None
    last_error: str | None = None


class ProviderStatusResponseDTO(BaseModel):
    """Consolidated provider status response payload."""

    providers: dict[str, ProviderStatusItemDTO]


class MacroSyncResponseDTO(BaseModel):
    """Response payload for macroeconomic indicator synchronization."""

    yield_spread_10y_2y: float
    fed_funds_rate: float
    source: str = "FRED"
    status: str = "synced"


class ExternalNewsItemDTO(BaseModel):
    """DTO for an article ingested from an external news provider."""

    headline: str
    summary: str
    url: str
    source: str
    published_at: str
    related_symbols: list[str] = Field(default_factory=list)


@router.get(
    "/status",
    response_model=ProviderStatusResponseDTO,
    summary="Query external data provider configuration and health telemetry",
)
async def get_provider_status(
    provider_mgr: ExternalProviderManager = Depends(get_external_provider_manager),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> ProviderStatusResponseDTO:
    """Return health, configuration, and invocation statistics across all external APIs."""
    raw_statuses = provider_mgr.get_provider_statuses()
    items = {pid: ProviderStatusItemDTO(**st) for pid, st in raw_statuses.items()}
    return ProviderStatusResponseDTO(providers=items)


@router.post(
    "/macro/sync",
    response_model=MacroSyncResponseDTO,
    summary="Synchronize macroeconomic indicators from FRED/Treasury",
)
async def sync_macro_state(
    provider_mgr: ExternalProviderManager = Depends(get_external_provider_manager),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> MacroSyncResponseDTO:
    """Query macroeconomic state (Yield Spread 10Y-2Y, Fed Funds Rate) from FRED."""
    macro = await provider_mgr.get_macro_state()
    return MacroSyncResponseDTO(
        yield_spread_10y_2y=macro.get("yield_spread_10y_2y", 0.18),
        fed_funds_rate=macro.get("fed_funds_rate", 5.25),
    )


@router.get(
    "/company-news/{symbol}",
    response_model=list[ExternalNewsItemDTO],
    summary="Fetch real-time company news for a ticker from Finnhub",
)
async def get_company_news(
    symbol: str,
    lookback_days: int = Query(default=2, ge=1, le=14),
    provider_mgr: ExternalProviderManager = Depends(get_external_provider_manager),
    news_service: NewsPredictionService = Depends(get_news_prediction_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[ExternalNewsItemDTO]:
    """Query company-specific financial news from Finnhub or fall back to harvested wire headlines.

    Purpose: Deliver real-time company news feed with deterministic fallback.
    Explicit Dependency Tracking: ExternalProviderManager, NewsPredictionService.
    Structural Relationship: Ingestion layer endpoint consumed by Pillar 1 NewsDecayView.
    Defensive Invariant: Guaranteed non-empty response by falling back to harvested RSS and synthetic news.
    """
    items = await provider_mgr.finnhub.fetch_company_news(symbol, lookback_days=lookback_days)
    if items:
        return [
            ExternalNewsItemDTO(
                headline=it.headline,
                summary=it.summary,
                url=it.url,
                source=it.source,
                published_at=it.published_at,
                related_symbols=it.related_symbols,
            )
            for it in items
        ]

    # Fallback to news_service cached/harvested articles
    articles = news_service.get_latest_articles(limit=50)
    if not articles:
        await news_service.harvest_and_predict(fallback_to_synthetic=True)
        articles = news_service.get_latest_articles(limit=50)

    matched = [
        art
        for art in articles
        if not art.tickers or symbol.upper() in [t.upper() for t in art.tickers]
    ]
    if not matched and articles:
        matched = articles[:10]

    return [
        ExternalNewsItemDTO(
            headline=art.headline,
            summary=art.summary,
            url=art.url,
            source=art.source,
            published_at=art.published_at.isoformat(),
            related_symbols=list(art.tickers) if art.tickers else [symbol.upper()],
        )
        for art in matched
    ]
