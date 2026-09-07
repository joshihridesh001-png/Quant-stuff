"""Abstract repository and service interfaces enforcing Dependency Inversion."""

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from quant.domain.models import Asset, EventCentrality, Genotype, NewsEvent


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
