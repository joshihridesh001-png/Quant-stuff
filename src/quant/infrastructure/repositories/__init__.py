"""Concrete repository implementations for analytical and relational persistence."""

from quant.infrastructure.repositories.duckdb_historical_repository import (
    ERR_DATA_EMPTY_UNIVERSE,
    ERR_DATA_INSUFFICIENT_TIMESTAMPS,
    ERR_DATA_STORAGE_IO,
    ERR_DATA_SYMBOL_NOT_FOUND,
    AlignedReturnsResult,
    DuckDBHistoricalRepository,
    EmptyUniverseError,
    HistoricalStorageError,
    HistoricalSymbolNotFoundError,
    InsufficientTimestampsError,
)
from quant.infrastructure.repositories.duckdb_market_data_repository import (
    DuckDBMarketDataRepository,
)

__all__ = [
    "ERR_DATA_EMPTY_UNIVERSE",
    "ERR_DATA_INSUFFICIENT_TIMESTAMPS",
    "ERR_DATA_STORAGE_IO",
    "ERR_DATA_SYMBOL_NOT_FOUND",
    "AlignedReturnsResult",
    "DuckDBHistoricalRepository",
    "DuckDBMarketDataRepository",
    "EmptyUniverseError",
    "HistoricalStorageError",
    "HistoricalSymbolNotFoundError",
    "InsufficientTimestampsError",
]
