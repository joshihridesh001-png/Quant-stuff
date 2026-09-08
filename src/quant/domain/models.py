"""Pure domain entities and value objects for the quantitative platform."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

# Numerical and columnar processing libraries for zero-copy econometric arrays
import numpy as np
import pyarrow as pa


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


class Resolution(StrEnum):
    """Market price bar sampling resolution.

    Purpose: Standardizes discrete time aggregation windows for econometric series.
    Dependencies: Python standard library enum.StrEnum.
    Relationship: Parameterizes PriceBar storage partitions and IMarketDataRepository lookups.
    """

    ONE_SECOND = "1s"
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"


@dataclass(frozen=True)
class PriceBar:
    """Immutable market price bar (OHLCV) with defensive econometric invariants.

    Purpose: Encapsulates atomic discrete price and volume observations for a single timeframe.
    Dependencies: Resolution enum for timeframe stratification.
    Relationship: Ingested from market feeds and stored via DuckDBMarketDataRepository.
    Invariants: High >= max(Open, Close); Low <= min(Open, Close); Price > 0; Volume >= 0; Timestamp > 0.
    """

    # Purpose: Unique asset ticker symbol (e.g., 'AAPL', 'BTC-USDT')
    # Dependencies: Matches Asset.ticker in the domain model
    asset_id: str

    # Purpose: Nanosecond-precision Unix epoch timestamp representing bar close time
    # Dependencies: Nanosecond resolution prevents ordering collisions in high-frequency feeds
    # Invariant: Strictly positive integer > 0
    timestamp: int

    # Purpose: Opening traded price during the interval
    # Invariant: Strictly positive float > 0.0
    open: float

    # Purpose: Highest traded price during the interval
    # Invariant: Must satisfy high >= max(open, close)
    high: float

    # Purpose: Lowest traded price during the interval
    # Invariant: Must satisfy low <= min(open, close)
    low: float

    # Purpose: Closing traded price during the interval
    # Invariant: Strictly positive float > 0.0
    close: float

    # Purpose: Aggregate traded volume in base units during the interval
    # Invariant: Non-negative float >= 0.0
    volume: float

    # Purpose: Volume-Weighted Average Price across all executions within the bar
    # Invariant: Must be > 0.0 if volume > 0.0; bounded within [low * 0.95, high * 1.05]
    vwap: float

    # Purpose: Bar aggregation granularity
    # Dependencies: Resolution StrEnum
    resolution: Resolution = Resolution.ONE_MINUTE

    def __post_init__(self) -> None:
        """Enforce strict financial and mathematical invariants upon creation.

        Purpose: Prevents corrupted or physically impossible market data from polluting downstream models.
        Dependencies: None.
        Relationship: Executed automatically on frozen dataclass instantiation.
        Invariants: Validates timestamp > 0, prices > 0, high/low boundaries, volume >= 0, vwap > 0.
        """
        # Invariant check: Timestamp must be positive nanoseconds
        if self.timestamp <= 0:
            raise ValueError(
                f"PriceBar timestamp must be positive nanoseconds, got {self.timestamp}"
            )

        # Invariant check: Pricing components must be non-zero positive real numbers
        if self.open <= 0.0 or self.high <= 0.0 or self.low <= 0.0 or self.close <= 0.0:
            raise ValueError(
                f"All OHLC price components must be strictly positive: O={self.open}, H={self.high}, L={self.low}, C={self.close}"
            )

        # Invariant check: High must form the absolute upper envelope of Open and Close
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"High ({self.high}) cannot be strictly less than max(Open, Close) ({max(self.open, self.close)})"
            )

        # Invariant check: Low must form the absolute lower envelope of Open and Close
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"Low ({self.low}) cannot be strictly greater than min(Open, Close) ({min(self.open, self.close)})"
            )

        # Invariant check: Traded volume cannot be negative
        if self.volume < 0.0:
            raise ValueError(f"Traded volume cannot be negative, got {self.volume}")

        # Invariant check: VWAP must be strictly positive if any volume was transacted
        if self.volume > 0.0 and self.vwap <= 0.0:
            raise ValueError(f"VWAP must be strictly positive when volume > 0, got {self.vwap}")


@dataclass
class MarketDataBatch:
    """Contiguous columnar market data container for zero-copy econometric ingestion.

    Purpose: Provides high-performance, contiguous NumPy and PyArrow views of historical bar sequences.
    Dependencies: numpy and pyarrow for contiguous memory buffer allocation.
    Relationship: Produced by DuckDBMarketDataRepository; feeds Fractional Differentiation and Triple-Barrier estimators.
    Invariants: All 1D NumPy array buffers must share the exact identical length equal to count.
    """

    # Purpose: Target asset ticker identifier
    asset_id: str

    # Purpose: Aggregation timeframe resolution for all observations in the batch
    resolution: Resolution

    # Purpose: 1D contiguous Int64 array of bar timestamps (nanoseconds)
    timestamps: np.ndarray

    # Purpose: 1D contiguous Float64 array of opening prices
    opens: np.ndarray

    # Purpose: 1D contiguous Float64 array of highest prices
    highs: np.ndarray

    # Purpose: 1D contiguous Float64 array of lowest prices
    lows: np.ndarray

    # Purpose: 1D contiguous Float64 array of closing prices
    closes: np.ndarray

    # Purpose: 1D contiguous Float64 array of traded volumes
    volumes: np.ndarray

    # Purpose: 1D contiguous Float64 array of VWAP values
    vwaps: np.ndarray

    def __post_init__(self) -> None:
        """Validate dimension alignment across all columnar vector buffers.

        Purpose: Enforces zero ragged-array conditions across contiguous memory buffers.
        Dependencies: len() checks on NumPy arrays.
        Invariants: All arrays must have length equal to len(self.timestamps).
        """
        # Obtain baseline dimension from timestamp vector
        n = len(self.timestamps)

        # Invariant verification across all price and volume buffers
        for name, arr in [
            ("opens", self.opens),
            ("highs", self.highs),
            ("lows", self.lows),
            ("closes", self.closes),
            ("volumes", self.volumes),
            ("vwaps", self.vwaps),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"Dimension mismatch in MarketDataBatch: timestamps has length {n}, but {name} has length {len(arr)}"
                )

    @property
    def count(self) -> int:
        """Return total observation count in batch.

        Purpose: Informs downstream consumers of sample size for degrees-of-freedom calculations.
        """
        return len(self.timestamps)

    @property
    def start_time(self) -> int | None:
        """Return earliest timestamp in nanoseconds, or None if batch is empty."""
        return int(self.timestamps[0]) if len(self.timestamps) > 0 else None

    @property
    def end_time(self) -> int | None:
        """Return latest timestamp in nanoseconds, or None if batch is empty."""
        return int(self.timestamps[-1]) if len(self.timestamps) > 0 else None

    def to_arrow(self) -> pa.Table:
        """Export contiguous buffers directly to a zero-copy PyArrow Table.

        Purpose: Facilitates zero-copy IPC and parquet serialization without data duplication.
        Dependencies: pyarrow Table construction from typed arrays.
        Relationship: Used by persistence layers and parquet file writers.
        """
        # Purpose: Wrap contiguous memory into zero-copy PyArrow columnar table
        # Dependencies: pa.Table.from_arrays with explicit primitive schema
        return pa.Table.from_arrays(
            [
                pa.array(self.timestamps, type=pa.int64()),
                pa.array(self.opens, type=pa.float64()),
                pa.array(self.highs, type=pa.float64()),
                pa.array(self.lows, type=pa.float64()),
                pa.array(self.closes, type=pa.float64()),
                pa.array(self.volumes, type=pa.float64()),
                pa.array(self.vwaps, type=pa.float64()),
            ],
            names=["timestamp", "open", "high", "low", "close", "volume", "vwap"],
        )
