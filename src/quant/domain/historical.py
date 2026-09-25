"""Pure domain entities, value objects, exceptions, and diagnostic error codes for historical market data.

Functional Purpose:
    Defines immutable domain models for historical price bars, corporate action records (stock splits,
    cash dividends), price adjustment mechanisms (unadjusted, split-adjusted, total return adjusted),
    and historical batch collections. Enforces mathematical invariants ensuring strict temporal monotonicity,
    price geometry feasibility, and finite positive boundaries across all ingested historical data.

Explicit Dependency Tracking:
    - dataclasses: High-performance immutable value objects with slots.
    - enum: Python 3.11+ StrEnum for zero-overhead string-compatible enumerations.
    - math: Non-finite scalar verification (math.isfinite).
    - typing: Static typing annotations and Final constants.
    - quant.domain.models: Resolution enum and PriceBar value object.

Structural Relationship:
    - Sits at the domain foundation of Phase 13 (Real Historical Data Lake).
    - Consumed by:
        1. HistoricalDataProvider and concrete adapters (Yahoo, Polygon, Alpaca) in Step 13.2 & 13.3.
        2. DuckDBHistoricalRepository in Step 13.4.
        3. scripts/ingest_historical_data.py in Step 13.5.
        4. Modular alpha strategies in Phase 14 and BacktestRunner in Phase 15.

Defensive Invariants:
    - INV-DATA-004: Monotonic temporal ordering: t_k > t_{k-1} with zero duplicate timestamps per symbol.
    - INV-DATA-005: Price geometry feasibility: min(Open, High, Low, Close) > 0, High >= max(Open, Close), Low <= min(Open, Close).
    - INV-DATA-006: Volume non-negativity: Volume >= 0.0; VWAP strictly positive if Volume > 0.0.
    - INV-DATA-007: Corporate action validity: Split ratio s > 0, cash dividend d >= 0, finite float, no booleans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from quant.domain.models import PriceBar, Resolution

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_DATA_NON_MONOTONIC_TIMESTAMP: Final[str] = "ERR-DATA-005"
ERR_DATA_INVALID_BAR_GEOMETRY: Final[str] = "ERR-DATA-006"
ERR_DATA_NON_FINITE_INPUT: Final[str] = "ERR-DATA-007"
ERR_DATA_INVALID_SPLIT_FACTOR: Final[str] = "ERR-DATA-008"
ERR_DATA_EMPTY_BATCH: Final[str] = "ERR-DATA-009"
ERR_DATA_INVALID_QUERY_RANGE: Final[str] = "ERR-DATA-010"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Exception Protocol)
# ============================================================================


class HistoricalDataError(Exception):
    """Base exception for all historical data, corporate action, and ingestion errors."""

    def __init__(self, message: str, code: str = "ERR-DATA-000") -> None:
        # Functional Purpose: Initialize base historical data exception with diagnostic code.
        # Explicit Dependency Tracking: Exception base class.
        # Structural Relationship: Root of the historical data error taxonomy.
        # Defensive Invariant: Message must be non-empty string.
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class NonMonotonicTimestampError(HistoricalDataError):
    """Raised when historical bars violate strict temporal monotonicity (INV-DATA-004)."""

    def __init__(self, message: str, code: str = ERR_DATA_NON_MONOTONIC_TIMESTAMP) -> None:
        super().__init__(message, code=code)


class InvalidBarGeometryError(HistoricalDataError):
    """Raised when OHLC price envelopes violate physical consistency (INV-DATA-005)."""

    def __init__(self, message: str, code: str = ERR_DATA_INVALID_BAR_GEOMETRY) -> None:
        super().__init__(message, code=code)


class NonFiniteHistoricalInputError(HistoricalDataError):
    """Raised when pricing, volume, or split adjustments contain non-finite numbers or booleans (INV-DATA-007)."""

    def __init__(self, message: str, code: str = ERR_DATA_NON_FINITE_INPUT) -> None:
        super().__init__(message, code=code)


class InvalidCorporateActionError(HistoricalDataError):
    """Raised when stock split factors or dividends violate positivity or boundary bounds (INV-DATA-007)."""

    def __init__(self, message: str, code: str = ERR_DATA_INVALID_SPLIT_FACTOR) -> None:
        super().__init__(message, code=code)


class EmptyHistoricalBatchError(HistoricalDataError):
    """Raised when attempting to construct or process an empty historical bar batch."""

    def __init__(self, message: str, code: str = ERR_DATA_EMPTY_BATCH) -> None:
        super().__init__(message, code=code)


class InvalidHistoricalQueryError(HistoricalDataError):
    """Raised when query time ranges or symbol universes violate validity bounds."""

    def __init__(self, message: str, code: str = ERR_DATA_INVALID_QUERY_RANGE) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Enumerations
# ============================================================================


class PriceAdjustmentType(StrEnum):
    """Historical bar pricing adjustment methodology.

    Purpose:
        Distinguishes between raw unadjusted auction prints, split-adjusted prices
        (preserving share lot continuity), and dividend-split adjusted (total return) series.
    """

    UNADJUSTED = "UNADJUSTED"
    SPLIT_ADJUSTED = "SPLIT_ADJUSTED"
    DIVIDEND_SPLIT_ADJUSTED = "DIVIDEND_SPLIT_ADJUSTED"


class CorporateActionType(StrEnum):
    """Corporate action classification for price retro-adjustment.

    Purpose:
        Categorizes discrete capitalization events affecting price continuity.
    """

    SPLIT = "SPLIT"
    CASH_DIVIDEND = "CASH_DIVIDEND"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"


# ============================================================================
# Value Objects & Domain Models
# ============================================================================


@dataclass(frozen=True, slots=True)
class CorporateActionRecord:
    """Immutable record of an ex-dividend or stock split capitalization event.

    Functional Purpose:
        Captures discrete corporate events needed to adjust historical price series without
        inducing artificial price step discontinuities.

    Explicit Dependency Tracking:
        CorporateActionType for event classification.

    Structural Relationship:
        Aggregated in HistoricalBarBatch and processed by adjustment calculators.

    Defensive Invariants:
        INV-DATA-007: Split ratio must be strictly positive finite float. Cash dividend must be non-negative finite float.
    """

    asset_id: str
    timestamp: int  # Nanoseconds UTC epoch
    action_type: CorporateActionType
    split_ratio: float = 1.0  # E.g., 4.0 for a 4-for-1 forward split, 0.5 for 1-for-2 reverse split
    cash_dividend: float = 0.0  # Gross dividend distributed per share

    def __post_init__(self) -> None:
        # Functional Purpose: Validate corporate action numerical boundaries.
        # Explicit Dependency Tracking: math.isfinite.
        # Structural Relationship: Automatic frozen post-init validator.
        # Defensive Invariant: INV-DATA-007 strict non-finite and boolean rejection.
        if not self.asset_id or not isinstance(self.asset_id, str):
            raise NonFiniteHistoricalInputError("asset_id must be non-empty string")

        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, int)
            or self.timestamp <= 0
        ):
            raise NonFiniteHistoricalInputError(
                f"timestamp must be positive integer nanoseconds, got {self.timestamp}"
            )

        if isinstance(self.split_ratio, bool) or not isinstance(self.split_ratio, (int, float)):
            raise NonFiniteHistoricalInputError(
                f"split_ratio cannot be boolean, got {self.split_ratio}"
            )

        if not math.isfinite(self.split_ratio):
            raise NonFiniteHistoricalInputError(
                f"split_ratio must be finite float, got {self.split_ratio}"
            )

        if self.split_ratio <= 0.0:
            raise InvalidCorporateActionError(
                f"split_ratio must be strictly positive finite float, got {self.split_ratio}"
            )

        if isinstance(self.cash_dividend, bool) or not isinstance(self.cash_dividend, (int, float)):
            raise NonFiniteHistoricalInputError(
                f"cash_dividend cannot be boolean, got {self.cash_dividend}"
            )

        if not math.isfinite(self.cash_dividend):
            raise NonFiniteHistoricalInputError(
                f"cash_dividend must be finite float, got {self.cash_dividend}"
            )

        if self.cash_dividend < 0.0:
            raise InvalidCorporateActionError(
                f"cash_dividend must be non-negative finite float, got {self.cash_dividend}"
            )


@dataclass(frozen=True, slots=True)
class HistoricalPriceBar:
    """Immutable market price bar enhanced with corporate adjustment metadata.

    Functional Purpose:
        Encapsulates atomic discrete OHLCV observations along with adjusted close, cumulative split factor,
        and dividend attribution for econometric time-series modeling.

    Explicit Dependency Tracking:
        Resolution from quant.domain.models.

    Structural Relationship:
        Primary value object emitted by historical providers and stored in columnar DuckDB lake.
        Converts seamlessly to base PriceBar for simulation engine execution.

    Defensive Invariants:
        INV-DATA-005 (OHLC physical consistency), INV-DATA-006 (Volume non-negativity & VWAP bounds).
    """

    asset_id: str
    timestamp: int  # Nanosecond Unix epoch
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    resolution: Resolution = Resolution.ONE_DAY
    adj_close: float | None = None
    split_factor: float = 1.0
    dividend_amount: float = 0.0

    def __post_init__(self) -> None:
        # Functional Purpose: Enforce physical price geometry and non-finite boundary defenses.
        # Explicit Dependency Tracking: math.isfinite.
        # Structural Relationship: Auto-executed on dataclass instantiation.
        # Defensive Invariant: INV-DATA-005, INV-DATA-006, INV-DATA-007.
        if not self.asset_id or not isinstance(self.asset_id, str):
            raise NonFiniteHistoricalInputError("asset_id must be non-empty string")

        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, int)
            or self.timestamp <= 0
        ):
            raise NonFiniteHistoricalInputError(
                f"timestamp must be positive integer nanoseconds, got {self.timestamp}"
            )

        # Validate pricing floats
        for name, val in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
            ("vwap", self.vwap),
            ("split_factor", self.split_factor),
            ("dividend_amount", self.dividend_amount),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise NonFiniteHistoricalInputError(f"{name} cannot be boolean, got {val}")
            if not math.isfinite(val):
                raise NonFiniteHistoricalInputError(f"{name} must be finite real number, got {val}")

        if self.open <= 0.0 or self.high <= 0.0 or self.low <= 0.0 or self.close <= 0.0:
            raise InvalidBarGeometryError(
                f"All OHLC price components must be strictly positive: O={self.open}, H={self.high}, L={self.low}, C={self.close}"
            )

        # Physical Envelope: High >= max(Open, Close); Low <= min(Open, Close)
        if self.high < max(self.open, self.close):
            raise InvalidBarGeometryError(
                f"High ({self.high}) cannot be strictly less than max(Open, Close) ({max(self.open, self.close)})"
            )

        if self.low > min(self.open, self.close):
            raise InvalidBarGeometryError(
                f"Low ({self.low}) cannot be strictly greater than min(Open, Close) ({min(self.open, self.close)})"
            )

        if self.volume < 0.0:
            raise InvalidBarGeometryError(f"Traded volume cannot be negative, got {self.volume}")

        if self.volume > 0.0:
            if self.vwap <= 0.0:
                raise InvalidBarGeometryError(
                    f"VWAP must be positive when volume > 0, got {self.vwap}"
                )
            if self.vwap < (self.low - 1e-4) or self.vwap > (self.high + 1e-4):
                raise InvalidBarGeometryError(
                    f"VWAP ({self.vwap}) must lie within bar price envelope [{self.low}, {self.high}] when volume > 0"
                )

        # Validate adj_close if provided
        if self.adj_close is not None:
            if isinstance(self.adj_close, bool) or not isinstance(self.adj_close, (int, float)):
                raise NonFiniteHistoricalInputError(
                    f"adj_close cannot be boolean, got {self.adj_close}"
                )
            if not math.isfinite(self.adj_close) or self.adj_close <= 0.0:
                raise InvalidBarGeometryError(
                    f"adj_close must be strictly positive finite float, got {self.adj_close}"
                )

        if self.split_factor <= 0.0:
            raise InvalidCorporateActionError(
                f"split_factor must be strictly positive, got {self.split_factor}"
            )

        if self.dividend_amount < 0.0:
            raise InvalidCorporateActionError(
                f"dividend_amount cannot be negative, got {self.dividend_amount}"
            )

    def to_price_bar(self) -> PriceBar:
        """Convert into a standard quant engine PriceBar domain value object.

        Functional Purpose:
            Provides backward and forward compatibility with the Phase 2/Phase 5 ReplayEngine.
        Explicit Dependency Tracking:
            quant.domain.models.PriceBar.
        Structural Relationship:
            Bridge adapter between historical data lake and execution engines.
        Defensive Invariant:
            Emits valid PriceBar meeting all domain invariants.
        """
        return PriceBar(
            asset_id=self.asset_id,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=self.vwap,
            resolution=self.resolution,
        )

    def to_adjusted_bar(self) -> HistoricalPriceBar:
        """Return a HistoricalPriceBar with OHLC and VWAP scaled by the split adjustment factor.

        Functional Purpose:
            Normalizes historical price levels for backtesting across stock split horizons.
        Explicit Dependency Tracking:
            HistoricalPriceBar constructor.
        Structural Relationship:
            Transformation operator used by backtest data loaders.
        Defensive Invariant:
            Preserves physical envelope relationships post-scaling.
        """
        factor = self.split_factor
        if math.isclose(factor, 1.0, rel_tol=1e-9):
            return self

        return HistoricalPriceBar(
            asset_id=self.asset_id,
            timestamp=self.timestamp,
            open=self.open * factor,
            high=self.high * factor,
            low=self.low * factor,
            close=self.close * factor,
            volume=self.volume / factor if factor > 0 else self.volume,
            vwap=self.vwap * factor,
            resolution=self.resolution,
            adj_close=self.adj_close,
            split_factor=1.0,
            dividend_amount=self.dividend_amount * factor,
        )


@dataclass(frozen=True, slots=True)
class HistoricalBarBatch:
    """Chronologically verified sequence of historical bars for an asset.

    Functional Purpose:
        Encapsulates an ordered collection of historical bars and corporate actions,
        strictly verifying temporal monotonicity (INV-DATA-004) upon construction.

    Explicit Dependency Tracking:
        HistoricalPriceBar and CorporateActionRecord.

    Structural Relationship:
        Unit of columnar storage transfer between data harvesters, DuckDB, and backtesters.

    Defensive Invariants:
        INV-DATA-004: Strictly monotonic timestamps (t_k > t_{k-1}) with zero duplicates.
    """

    symbol: str
    resolution: Resolution
    bars: tuple[HistoricalPriceBar, ...]
    corporate_actions: tuple[CorporateActionRecord, ...] = ()

    def __post_init__(self) -> None:
        # Functional Purpose: Verify batch non-emptiness and strict temporal monotonicity.
        # Explicit Dependency Tracking: NonMonotonicTimestampError, EmptyHistoricalBatchError.
        # Structural Relationship: Boundary integrity guard for bar sequences.
        # Defensive Invariant: INV-DATA-004 (Strict Monotonicity).
        if not self.symbol or not isinstance(self.symbol, str):
            raise NonFiniteHistoricalInputError("symbol must be non-empty string")

        if not self.bars:
            raise EmptyHistoricalBatchError(
                f"HistoricalBarBatch for symbol '{self.symbol}' cannot be empty"
            )

        # Verify temporal monotonicity (INV-DATA-004)
        for i in range(1, len(self.bars)):
            prev_ts = self.bars[i - 1].timestamp
            curr_ts = self.bars[i].timestamp
            if curr_ts <= prev_ts:
                raise NonMonotonicTimestampError(
                    f"Temporal monotonicity violation in '{self.symbol}' at index {i}: "
                    f"bar[{i - 1}] ts={prev_ts} >= bar[{i}] ts={curr_ts}"
                )

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def start_timestamp(self) -> int:
        """First bar timestamp in nanoseconds."""
        return self.bars[0].timestamp

    @property
    def end_timestamp(self) -> int:
        """Last bar timestamp in nanoseconds."""
        return self.bars[-1].timestamp

    def timestamps(self) -> tuple[int, ...]:
        """Extract sequence of bar timestamps."""
        return tuple(b.timestamp for b in self.bars)

    def closes(self) -> tuple[float, ...]:
        """Extract sequence of unadjusted closing prices."""
        return tuple(b.close for b in self.bars)

    def adj_closes(self) -> tuple[float, ...]:
        """Extract sequence of adjusted closing prices (falls back to close if adj_close is None)."""
        return tuple(b.adj_close if b.adj_close is not None else b.close for b in self.bars)

    def volumes(self) -> tuple[float, ...]:
        """Extract sequence of traded volumes."""
        return tuple(b.volume for b in self.bars)

    def to_returns(self, use_adj_close: bool = True) -> tuple[float, ...]:
        """Compute discrete simple returns r_t = P_t / P_{t-1} - 1.

        Functional Purpose:
            Generates econometric return series with zero lookahead bias.
        Explicit Dependency Tracking:
            None.
        Structural Relationship:
            Feeds directly into FractionalDiff, Regimes, and BacktestRunner.
        Defensive Invariant:
            Returns length is len(bars) - 1.
        """
        prices = self.adj_closes() if use_adj_close else self.closes()
        if len(prices) < 2:
            return ()
        return tuple(prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices)))


@dataclass(frozen=True, slots=True)
class HistoricalDataQuery:
    """Specification criteria for querying historical market data across universe.

    Functional Purpose:
        Encapsulates parameterized time-range and symbol universe retrieval requests.

    Explicit Dependency Tracking:
        Resolution and PriceAdjustmentType.

    Structural Relationship:
        Input payload for HistoricalDataProvider and DuckDBMarketDataRepository.

    Defensive Invariants:
        Strictly positive time range (end_timestamp > start_timestamp > 0), non-empty symbols.
    """

    symbols: tuple[str, ...]
    start_timestamp: int  # Nanoseconds UTC
    end_timestamp: int  # Nanoseconds UTC
    resolution: Resolution = Resolution.ONE_DAY
    adjustment: PriceAdjustmentType = PriceAdjustmentType.SPLIT_ADJUSTED

    def __post_init__(self) -> None:
        # Functional Purpose: Validate query specification parameters.
        # Explicit Dependency Tracking: InvalidHistoricalQueryError.
        # Structural Relationship: Query boundary defense.
        # Defensive Invariant: start_timestamp > 0 and end_timestamp > start_timestamp.
        if not self.symbols:
            raise InvalidHistoricalQueryError("Query symbols tuple cannot be empty")

        for sym in self.symbols:
            if not sym or not isinstance(sym, str):
                raise InvalidHistoricalQueryError("Each symbol in query must be non-empty string")

        if (
            isinstance(self.start_timestamp, bool)
            or not isinstance(self.start_timestamp, int)
            or self.start_timestamp <= 0
        ):
            raise InvalidHistoricalQueryError(
                f"start_timestamp must be positive integer nanoseconds, got {self.start_timestamp}"
            )

        if (
            isinstance(self.end_timestamp, bool)
            or not isinstance(self.end_timestamp, int)
            or self.end_timestamp <= self.start_timestamp
        ):
            raise InvalidHistoricalQueryError(
                f"end_timestamp ({self.end_timestamp}) must be strictly greater than start_timestamp ({self.start_timestamp})"
            )
