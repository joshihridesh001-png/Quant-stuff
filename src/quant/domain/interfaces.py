"""Abstract repository and service interfaces enforcing Dependency Inversion."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from quant.domain.models import (
    Asset,
    EventCentrality,
    Genotype,
    MarketDataBatch,
    NewsEvent,
    PriceBar,
    Resolution,
)


class IAssetRepository(ABC):
    """Abstract persistence interface for assets."""

    @abstractmethod
    async def add(self, asset: Asset) -> Asset:
        """Persist a new asset."""

    @abstractmethod
    async def get_by_ticker(self, ticker: str) -> Asset | None:
        """Fetch asset by unique ticker symbol."""

    @abstractmethod
    async def list_active(self) -> list[Asset]:
        """Return all active universe assets."""


class IEventRepository(ABC):
    """Abstract persistence interface for news events and causal graph state."""

    @abstractmethod
    async def add(self, event: NewsEvent, centralities: list[EventCentrality]) -> NewsEvent:
        """Persist event and its multi-asset centrality linkages."""

    @abstractmethod
    async def add_batch(
        self, batch: list[tuple[NewsEvent, list[EventCentrality]]]
    ) -> list[NewsEvent]:
        """Persist a batch of events and their centrality linkages atomically."""

    @abstractmethod
    async def get_by_id(self, event_id: UUID) -> NewsEvent | None:
        """Retrieve single event by ID."""

    @abstractmethod
    async def get_events_for_asset(
        self, asset_id: UUID, start_time: datetime, end_time: datetime
    ) -> list[tuple[NewsEvent, float]]:
        """Retrieve time-windowed events for an asset paired with centrality weights."""


class IGenotypeRepository(ABC):
    """Abstract persistence interface for evolutionary strategy chromosomes."""

    @abstractmethod
    async def add(self, genotype: Genotype) -> Genotype:
        """Persist a newly generated strategy chromosome."""

    @abstractmethod
    async def get_by_id(self, genotype_id: UUID) -> Genotype | None:
        """Fetch chromosome by unique identifier."""

    @abstractmethod
    async def get_generation(self, generation: int) -> list[Genotype]:
        """Retrieve all genotypes belonging to a given epoch/generation."""

    @abstractmethod
    async def get_alpha_cohort(self, limit: int = 50) -> list[Genotype]:
        """Retrieve top elite performers sorted by multi-objective fitness."""

    @abstractmethod
    async def update_evaluation(
        self,
        genotype_id: UUID,
        fitness_score: float,
        deflated_sharpe: float,
        max_drawdown: float,
        regret_score: float,
    ) -> None:
        """Update fitness evaluation metrics after backtest/scenario execution."""


class INewsIngestionEngine(ABC):
    """Abstract interface for asynchronous high-throughput event ingestion and state query."""

    @abstractmethod
    async def ingest_batch(
        self,
        events_data: list[dict[str, Any]],
    ) -> list[NewsEvent]:
        """Asynchronously ingest and persist a batch of events with ticker weights."""

    @abstractmethod
    async def get_state_vectors(
        self,
        tickers: list[str],
        as_of_time: datetime | None = None,
    ) -> dict[str, tuple[list[float], int]]:
        """Asynchronously compute active decayed news state vectors across multiple assets."""


class IMarketDataRepository(ABC):
    """Abstract persistence interface for high-throughput columnar market price bars.

    Purpose: Isolates downstream econometric feature generators from underlying storage mechanics.
    Dependencies: PriceBar, MarketDataBatch, Resolution.
    Relationship: Implemented by DuckDBMarketDataRepository; consumed by MarketDataService and feature generators.
    """

    @abstractmethod
    async def add_bars_batch(self, bars: Sequence[PriceBar]) -> int:
        """Persist a batch of price bars with upsert semantics.

        Purpose: Bulk loads price bars into high-performance columnar storage.
        Dependencies: Sequence[PriceBar] containing validated observations.
        Post-conditions: Records written to storage; returns total count of inserted/replaced rows.
        """

    @abstractmethod
    async def get_bars_range(
        self, asset_id: str, start_time: int, end_time: int, resolution: Resolution
    ) -> MarketDataBatch:
        """Retrieve contiguous columnar market bars within timestamp range [start_time, end_time].

        Purpose: Supplies chronological price/volume arrays for backtesting and feature calculation.
        Dependencies: Nanosecond epoch start and end bounds.
        Post-conditions: Returns MarketDataBatch sorted strictly in ascending chronological order.
        """

    @abstractmethod
    async def get_latest_bars(
        self, asset_id: str, count: int, resolution: Resolution
    ) -> MarketDataBatch:
        """Retrieve the most recent N contiguous columnar market bars in ascending order.

        Purpose: Feeds rolling econometric windows (e.g., rolling realized volatility) for live inference.
        Dependencies: count > 0.
        Post-conditions: Returns MarketDataBatch containing up to N bars ordered chronologically ascending.
        """

    @abstractmethod
    async def get_available_range(
        self, asset_id: str, resolution: Resolution
    ) -> tuple[int, int] | None:
        """Retrieve the earliest and latest available timestamps for an asset.

        Purpose: Enables discovery of available historical depth before executing large query sweeps.
        Post-conditions: Returns (min_timestamp, max_timestamp) in nanoseconds, or None if no bars exist.
        """
