"""Pure domain entities and value objects for the quantitative platform."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class GenotypeCohort(StrEnum):
    """Evolutionary cohort stratification."""

    ALPHA = "ALPHA"
    ASPIRANT = "ASPIRANT"


class ScenarioType(StrEnum):
    """Adversarial market scenario classes."""

    IMMEDIATE_REVERSAL = "IMMEDIATE_REVERSAL"
    MOMENTUM_CASCADE = "MOMENTUM_CASCADE"
    LIQUIDITY_SQUEEZE = "LIQUIDITY_SQUEEZE"


@dataclass(frozen=True)
class Asset:
    """Target universe traded asset."""

    ticker: str
    name: str
    sector: str
    id: UUID = field(default_factory=uuid4)
    is_active: bool = True


@dataclass(frozen=True)
class EventCentrality:
    """Continuous association weight between a news event and an asset."""

    event_id: UUID
    asset_id: UUID
    centrality: float  # c_{i,k} in [0, 1]


@dataclass
class NewsEvent:
    """Unstructured news event with semantic decomposition and sentiment axes."""

    headline: str
    raw_text: str
    timestamp: datetime
    id: UUID = field(default_factory=uuid4)
    dense_embedding: list[float] = field(default_factory=list)  # v_dense
    sentiment_polarity: float = 0.0  # [-1.0, 1.0]
    sentiment_subjectivity: float = 0.0  # [0.0, 1.0]
    sentiment_novelty: float = 0.0  # [0.0, 1.0]
    urgency: float = 0.5  # u in [0.0, 1.0]
    source: str = "GENERIC"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def sentiment_vector(self) -> list[float]:
        """Tri-axial sentiment representation s_sentiment."""
        return [self.sentiment_polarity, self.sentiment_subjectivity, self.sentiment_novelty]


@dataclass
class Genotype:
    """Algorithmic chromosome representing a candidate predictive strategy."""

    generation: int
    cohort: GenotypeCohort
    chromosome_repr: dict[str, Any]  # tau_fast, tau_slow, alpha
    chromosome_game: dict[str, Any]  # belief priors, lambda, gamma
    chromosome_infer: dict[str, Any]  # model depths, sensitivity thresholds
    chromosome_risk: dict[str, Any]  # vol target, max drawdown threshold
    id: UUID = field(default_factory=uuid4)
    fitness_score: float | None = None
    deflated_sharpe: float | None = None
    max_drawdown: float | None = None
    regret_score: float | None = None
    novelty_score: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class ScenarioProfile:
    """Adversarial scenario specification with payoff distribution parameters."""

    scenario_type: ScenarioType
    expected_returns: dict[str, float]
    covariance_matrix: list[list[float]]
    crowding_penalty_factor: float
    description: str = ""
