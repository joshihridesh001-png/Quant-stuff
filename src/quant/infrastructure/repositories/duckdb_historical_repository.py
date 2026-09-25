"""Partitioned DuckDB columnar storage repository for institutional historical market data.

Functional Purpose:
    Implements high-throughput persistence, indexing, and vectorized retrieval of historical
    price bars, dividend events, and stock splits. Backed by embedded DuckDB and Apache PyArrow.

Explicit Dependency Tracking:
    - duckdb: Analytical embedded SQL engine.
    - pyarrow: Zero-copy columnar in-memory data exchange.
    - numpy: Vectorized multidimensional mathematical calculations.
    - quant.domain.historical: Domain models, invariants, and validation rules.
    - quant.infrastructure.database.duckdb_session: Thread-safe DuckDB connection manager.

Structural Relationship:
    Primary storage layer connecting raw data harvesters (Yahoo Finance, Polygon, Alpaca)
    with analytical engines, backtesters, and multi-asset econometric models.

Defensive Invariants:
    - INV-DATA-004: Monotonic timestamps per symbol.
    - INV-DATA-005: Physical price envelope consistency.
    - INV-DATA-006: Non-negative volume and finite VWAP.
    - Zero forward lookahead: Causal forward-filling only during universe alignment.
"""

from __future__ import annotations

import contextlib
import logging
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pyarrow as pa

from quant.domain.historical import (
    ERR_DATA_NON_FINITE_INPUT,
    EmptyHistoricalBatchError,
    HistoricalBarBatch,
    HistoricalDataError,
    HistoricalPriceBar,
    NonFiniteHistoricalInputError,
    NonMonotonicTimestampError,
)
from quant.domain.models import Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager

logger = logging.getLogger(__name__)

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Diagnostic Error Taxonomy)
# ============================================================================

ERR_DATA_STORAGE_IO: Final[str] = "ERR-DATA-015"
ERR_DATA_EMPTY_UNIVERSE: Final[str] = "ERR-DATA-016"
ERR_DATA_INSUFFICIENT_TIMESTAMPS: Final[str] = "ERR-DATA-017"
ERR_DATA_SYMBOL_NOT_FOUND: Final[str] = "ERR-DATA-018"


# ============================================================================
# Exception Taxonomy
# ============================================================================


class HistoricalStorageError(HistoricalDataError):
    """Base exception for all historical repository storage and retrieval failures."""

    def __init__(self, message: str, code: str = ERR_DATA_STORAGE_IO) -> None:
        super().__init__(message, code=code)


class EmptyUniverseError(HistoricalStorageError):
    """Raised when an operation is requested on an empty asset universe."""

    def __init__(self, message: str, code: str = ERR_DATA_EMPTY_UNIVERSE) -> None:
        super().__init__(message, code=code)


class InsufficientTimestampsError(HistoricalStorageError):
    """Raised when historical observations are insufficient to compute differentials."""

    def __init__(self, message: str, code: str = ERR_DATA_INSUFFICIENT_TIMESTAMPS) -> None:
        super().__init__(message, code=code)


class HistoricalSymbolNotFoundError(HistoricalStorageError):
    """Raised when querying a symbol that possesses zero historical bar records."""

    def __init__(self, message: str, code: str = ERR_DATA_SYMBOL_NOT_FOUND) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Aligned Returns Result DTO
# ============================================================================


@dataclass(frozen=True, slots=True)
class AlignedReturnsResult:
    """Aligned returns matrix and temporal coordinates across multiple assets.

    Functional Purpose:
        Encapsulates an aligned (T, N) returns matrix across an asset universe with
        guaranteed causal alignment, zero forward lookahead, and explicit timestamps.

    Explicit Dependency Tracking:
        numpy.ndarray for matrix data.

    Structural Relationship:
        Supplied to covariance estimators, multi-asset alpha models, and portfolio optimizers.

    Defensive Invariants:
        - matrix.ndim == 2 and matrix.shape == (len(timestamps), len(symbols)).
        - timestamps are strictly monotonic.
    """

    matrix: np.ndarray
    timestamps: np.ndarray
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        # Functional Purpose: Verify matrix dimensionality and shape consistency.
        # Explicit Dependency Tracking: numpy.ndarray shapes.
        # Structural Relationship: Boundary validation for mathematical operations.
        # Defensive Invariant: Matrix rows must strictly match timestamp count.
        if self.matrix.ndim != 2:
            raise NonFiniteHistoricalInputError(
                f"Aligned returns matrix must be 2D, got shape {self.matrix.shape}"
            )
        if len(self.timestamps) != self.matrix.shape[0]:
            raise NonFiniteHistoricalInputError(
                f"Timestamps length ({len(self.timestamps)}) does not match matrix rows ({self.matrix.shape[0]})"
            )
        if len(self.symbols) != self.matrix.shape[1]:
            raise NonFiniteHistoricalInputError(
                f"Symbols length ({len(self.symbols)}) does not match matrix columns ({self.matrix.shape[1]})"
            )


# ============================================================================
# DuckDB Historical Repository
# ============================================================================


class DuckDBHistoricalRepository:
    """High-performance DuckDB columnar repository for historical price bars.

    Functional Purpose:
        Provides ACID-compliant idempotent upserts and vectorized chronological range queries
        for historical market bars, incorporating split-adjusted prices and corporate actions.

    Explicit Dependency Tracking:
        - DuckDBManager for thread-safe embedded DuckDB connection management.
        - Apache PyArrow for zero-copy bulk ingestion into DuckDB.
        - NumPy for contiguous memory matrix returns extraction.

    Structural Relationship:
        Serves historical data queries to backtesting engines, strategy signal generators,
        and machine learning feature pipelines.

    Defensive Invariants:
        - Idempotent upserts on composite key (symbol, resolution, timestamp).
        - Ascending chronological ordering on all retrieved batches.
        - Zero lookahead: causal forward-fill alignment only.
    """

    def __init__(self, manager: DuckDBManager) -> None:
        """Initialize repository with injected DuckDB session manager.

        Functional Purpose:
            Establishes connection to DuckDB and ensures historical_bars table exists.
        Explicit Dependency Tracking:
            DuckDBManager.
        Structural Relationship:
            Injected via constructor dependency injection.
        Defensive Invariant:
            historical_bars table and composite index must be initialized.
        """
        self._manager = manager
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure historical_bars table and composite lookup index exist."""
        conn = self._manager.get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_bars (
                symbol VARCHAR NOT NULL,
                resolution VARCHAR NOT NULL,
                timestamp BIGINT NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                vwap DOUBLE NOT NULL,
                adj_close DOUBLE NOT NULL,
                split_factor DOUBLE NOT NULL,
                dividend_amount DOUBLE NOT NULL,
                PRIMARY KEY (symbol, resolution, timestamp)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_historical_bars_lookup
            ON historical_bars (symbol, resolution, timestamp);
            """
        )

    # ------------------------------------------------------------------------
    # Bulk Bar Ingestion (Upsert)
    # ------------------------------------------------------------------------

    def save_batch_sync(self, batch: HistoricalBarBatch) -> int:
        """Persist a HistoricalBarBatch synchronously with idempotent upsert semantics.

        Functional Purpose:
            Ingests price bars via Apache PyArrow zero-copy tables in a single transaction.
        Explicit Dependency Tracking:
            pyarrow.Table, duckdb.INSERT OR REPLACE.
        Defensive Invariants:
            INV-DATA-004: Validated by HistoricalBarBatch constructor.
        """
        if not batch.bars:
            return 0

        # Construct zero-copy columnar PyArrow table
        arrow_table = pa.Table.from_arrays(
            [
                pa.array([b.asset_id for b in batch.bars], type=pa.string()),
                pa.array([str(batch.resolution) for _ in batch.bars], type=pa.string()),
                pa.array([b.timestamp for b in batch.bars], type=pa.int64()),
                pa.array([b.open for b in batch.bars], type=pa.float64()),
                pa.array([b.high for b in batch.bars], type=pa.float64()),
                pa.array([b.low for b in batch.bars], type=pa.float64()),
                pa.array([b.close for b in batch.bars], type=pa.float64()),
                pa.array([b.volume for b in batch.bars], type=pa.float64()),
                pa.array([b.vwap for b in batch.bars], type=pa.float64()),
                pa.array([b.adj_close for b in batch.bars], type=pa.float64()),
                pa.array([b.split_factor for b in batch.bars], type=pa.float64()),
                pa.array([b.dividend_amount for b in batch.bars], type=pa.float64()),
            ],
            names=[
                "symbol",
                "resolution",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "adj_close",
                "split_factor",
                "dividend_amount",
            ],
        )

        view_name = f"incoming_historical_view_{uuid.uuid4().hex}"
        conn = self._manager.get_connection()

        conn.register(view_name, arrow_table)
        try:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO historical_bars
                SELECT * FROM {view_name};
                """
            )
        finally:
            with contextlib.suppress(Exception):
                conn.unregister(view_name)

        return len(batch.bars)

    async def save_batch(self, batch: HistoricalBarBatch) -> int:
        """Persist a HistoricalBarBatch asynchronously via worker threadpool."""
        return await self._manager.run_sync(lambda _: self.save_batch_sync(batch))

    # ------------------------------------------------------------------------
    # Range Query
    # ------------------------------------------------------------------------

    def get_bars_range_sync(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Retrieve chronological historical bars for a symbol within [start_time, end_time].

        Functional Purpose:
            Supplies contiguous, split-adjusted price bars for backtesting and strategy evaluation.
        Explicit Dependency Tracking:
            duckdb.DuckDBPyConnection.fetchnumpy().
        Defensive Invariants:
            INV-DATA-004: Returned batch is guaranteed strictly monotonic.
            Raises EmptyHistoricalBatchError if zero bars match criteria.
        """
        if not symbol or not isinstance(symbol, str):
            raise NonFiniteHistoricalInputError(
                "symbol must be non-empty string", code=ERR_DATA_NON_FINITE_INPUT
            )
        if isinstance(start_time, bool) or isinstance(end_time, bool):
            raise NonFiniteHistoricalInputError(
                "Timestamps cannot be boolean values", code=ERR_DATA_NON_FINITE_INPUT
            )
        if start_time > end_time:
            raise NonMonotonicTimestampError(
                f"start_time ({start_time}) cannot exceed end_time ({end_time})"
            )

        query = """
            SELECT symbol, resolution, timestamp, open, high, low, close, volume, vwap,
                   adj_close, split_factor, dividend_amount
            FROM historical_bars
            WHERE symbol = ? AND resolution = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC;
        """
        conn = self._manager.get_connection()
        res = conn.execute(query, [symbol, str(resolution), start_time, end_time]).fetchall()

        if not res:
            raise EmptyHistoricalBatchError(
                f"Zero historical bars found for symbol '{symbol}' at resolution '{resolution}' "
                f"in range [{start_time}, {end_time}]",
                code=ERR_DATA_SYMBOL_NOT_FOUND,
            )

        bars: list[HistoricalPriceBar] = []
        for row in res:
            close_price = float(row[6])
            adj_close_raw = row[9]
            adj_close_val = (
                float(adj_close_raw)
                if adj_close_raw is not None
                and not math.isnan(float(adj_close_raw))
                and float(adj_close_raw) > 0.0
                else close_price
            )
            bar = HistoricalPriceBar(
                asset_id=str(row[0]),
                timestamp=int(row[2]),
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=close_price,
                volume=float(row[7]),
                vwap=float(row[8]),
                resolution=resolution,
                adj_close=adj_close_val,
                split_factor=float(row[10])
                if row[10] is not None and not math.isnan(float(row[10])) and float(row[10]) > 0.0
                else 1.0,
                dividend_amount=float(row[11])
                if row[11] is not None and not math.isnan(float(row[11])) and float(row[11]) >= 0.0
                else 0.0,
            )
            bars.append(bar)

        return HistoricalBarBatch(
            symbol=symbol,
            resolution=resolution,
            bars=tuple(bars),
        )

    async def get_bars_range(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Retrieve chronological historical bars asynchronously."""
        return await self._manager.run_sync(
            lambda _: self.get_bars_range_sync(symbol, start_time, end_time, resolution)
        )

    # ------------------------------------------------------------------------
    # Metadata & Boundary Queries
    # ------------------------------------------------------------------------

    def get_available_range_sync(
        self,
        symbol: str,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> tuple[int, int] | None:
        """Retrieve earliest and latest nanosecond timestamps for a symbol."""
        if not symbol or not isinstance(symbol, str):
            raise NonFiniteHistoricalInputError(
                "symbol must be non-empty string", code=ERR_DATA_NON_FINITE_INPUT
            )

        query = """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM historical_bars
            WHERE symbol = ? AND resolution = ?;
        """
        conn = self._manager.get_connection()
        row = conn.execute(query, [symbol, str(resolution)]).fetchone()
        if not row or row[0] is None or row[1] is None:
            return None
        return int(row[0]), int(row[1])

    async def get_available_range(
        self,
        symbol: str,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> tuple[int, int] | None:
        """Retrieve earliest and latest nanosecond timestamps asynchronously."""
        return await self._manager.run_sync(
            lambda _: self.get_available_range_sync(symbol, resolution)
        )

    def get_all_symbols_sync(self, resolution: Resolution | None = None) -> list[str]:
        """Retrieve distinct symbol tickers stored in the historical lake."""
        conn = self._manager.get_connection()
        if resolution is not None:
            query = "SELECT DISTINCT symbol FROM historical_bars WHERE resolution = ? ORDER BY symbol ASC;"
            rows = conn.execute(query, [str(resolution)]).fetchall()
        else:
            query = "SELECT DISTINCT symbol FROM historical_bars ORDER BY symbol ASC;"
            rows = conn.execute(query).fetchall()
        return [str(r[0]) for r in rows]

    async def get_all_symbols(self, resolution: Resolution | None = None) -> list[str]:
        """Retrieve distinct symbol tickers asynchronously."""
        return await self._manager.run_sync(lambda _: self.get_all_symbols_sync(resolution))

    def get_bar_count_sync(
        self, symbol: str | None = None, resolution: Resolution | None = None
    ) -> int:
        """Return total row count in the historical lake."""
        conn = self._manager.get_connection()
        clauses: list[str] = []
        params: list[str] = []

        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if resolution is not None:
            clauses.append("resolution = ?")
            params.append(str(resolution))

        where_str = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT COUNT(*) FROM historical_bars{where_str};"
        row = conn.execute(query, params).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    async def get_bar_count(
        self, symbol: str | None = None, resolution: Resolution | None = None
    ) -> int:
        """Return total row count asynchronously."""
        return await self._manager.run_sync(
            lambda _: self.get_bar_count_sync(symbol=symbol, resolution=resolution)
        )

    # ------------------------------------------------------------------------
    # Vectorized Aligned Returns Matrix (Causal Forward-Fill Alignment)
    # ------------------------------------------------------------------------

    def get_aligned_returns_matrix_sync(
        self,
        symbols: Sequence[str],
        start_time: int,
        end_time: int,
        resolution: Resolution = Resolution.ONE_DAY,
        price_field: str = "adj_close",
        return_type: str = "simple",
    ) -> AlignedReturnsResult:
        """Extract a causally aligned returns matrix across multiple assets.

        Functional Purpose:
            Constructs a zero-copy NumPy returns matrix $R \\in \\mathbb{R}^{(T-1) \\times N}$
            for multi-asset risk estimation, statistical arbitrage, and factor modeling.

        Explicit Dependency Tracking:
            - DuckDB parameterized IN-clause query.
            - NumPy 2D array operations.

        Causal Alignment Invariant:
            - Collects the union of all discrete observation timestamps $t_0 < t_1 < \\dots < t_{K-1}$.
            - Forward-fills missing observations strictly from the past ($P_{k, j} = P_{k-1, j}$).
            - NEVER back-fills from future prices (Zero Forward Lookahead).
            - Computes returns differentials $r_{k, j}$ strictly across valid prior observations.

        Defensive Invariants:
            - symbols must be non-empty and non-boolean.
            - start_time <= end_time.
            - price_field must be one of: adj_close, close, open, high, low, vwap.
            - return_type must be: 'simple' or 'log'.
        """
        # Validate universe
        if not symbols:
            raise EmptyUniverseError(
                "symbols sequence cannot be empty", code=ERR_DATA_EMPTY_UNIVERSE
            )

        deduped_symbols: list[str] = []
        for s in symbols:
            if isinstance(s, bool) or not isinstance(s, str) or not s:
                raise NonFiniteHistoricalInputError(
                    f"Symbol must be non-empty string, got {s!r}", code=ERR_DATA_NON_FINITE_INPUT
                )
            if s not in deduped_symbols:
                deduped_symbols.append(s)

        if isinstance(start_time, bool) or isinstance(end_time, bool):
            raise NonFiniteHistoricalInputError(
                "Timestamps cannot be boolean values", code=ERR_DATA_NON_FINITE_INPUT
            )
        if start_time > end_time:
            raise NonMonotonicTimestampError(
                f"start_time ({start_time}) cannot exceed end_time ({end_time})"
            )

        valid_fields = ("adj_close", "close", "open", "high", "low", "vwap")
        if price_field not in valid_fields:
            raise NonFiniteHistoricalInputError(
                f"Invalid price_field '{price_field}'. Must be one of {valid_fields}",
                code=ERR_DATA_NON_FINITE_INPUT,
            )

        if return_type not in ("simple", "log"):
            raise NonFiniteHistoricalInputError(
                f"Invalid return_type '{return_type}'. Must be 'simple' or 'log'",
                code=ERR_DATA_NON_FINITE_INPUT,
            )

        # Build parameterized query
        placeholders = ", ".join(["?"] * len(deduped_symbols))
        query = f"""
            SELECT symbol, timestamp, COALESCE(NULLIF({price_field}, 0.0), close) AS price
            FROM historical_bars
            WHERE symbol IN ({placeholders}) AND resolution = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC, symbol ASC;
        """
        params = [*deduped_symbols, str(resolution), start_time, end_time]

        conn = self._manager.get_connection()
        rows = conn.execute(query, params).fetchall()

        if not rows:
            raise InsufficientTimestampsError(
                f"Zero observations found for requested universe in range [{start_time}, {end_time}]",
                code=ERR_DATA_INSUFFICIENT_TIMESTAMPS,
            )

        # Extract discrete sorted unique timestamps
        ts_set = sorted({int(r[1]) for r in rows})
        if len(ts_set) < 2:
            raise InsufficientTimestampsError(
                f"Insufficient discrete observation timestamps ({len(ts_set)} < 2) to compute return differentials",
                code=ERR_DATA_INSUFFICIENT_TIMESTAMPS,
            )

        k_periods = len(ts_set)
        n_assets = len(deduped_symbols)

        ts_to_idx = {ts: idx for idx, ts in enumerate(ts_set)}
        sym_to_idx = {sym: idx for idx, sym in enumerate(deduped_symbols)}

        # Populate price matrix with NaNs
        prices = np.full((k_periods, n_assets), np.nan, dtype=np.float64)
        for r in rows:
            sym_str = str(r[0])
            t_int = int(r[1])
            p_val = float(r[2])
            if math.isfinite(p_val) and p_val > 0.0:
                prices[ts_to_idx[t_int], sym_to_idx[sym_str]] = p_val

        # Causal forward-fill: strictly from past to present (Zero Forward Lookahead)
        for j in range(n_assets):
            last_price = np.nan
            for k in range(k_periods):
                curr_p = prices[k, j]
                if np.isfinite(curr_p):
                    last_price = curr_p
                else:
                    prices[k, j] = last_price

        # Compute return differentials (length: k_periods - 1)
        returns_matrix = np.zeros((k_periods - 1, n_assets), dtype=np.float64)
        for k in range(1, k_periods):
            for j in range(n_assets):
                prev_p = prices[k - 1, j]
                curr_p = prices[k, j]
                if np.isfinite(prev_p) and np.isfinite(curr_p) and prev_p > 0.0 and curr_p > 0.0:
                    if return_type == "log":
                        returns_matrix[k - 1, j] = math.log(curr_p / prev_p)
                    else:
                        returns_matrix[k - 1, j] = (curr_p - prev_p) / prev_p
                else:
                    # Unlisted or zero/non-finite prior history receives 0.0 return
                    returns_matrix[k - 1, j] = 0.0

        timestamps_out = np.ascontiguousarray(ts_set[1:], dtype=np.int64)
        matrix_out = np.ascontiguousarray(returns_matrix, dtype=np.float64)

        return AlignedReturnsResult(
            matrix=matrix_out,
            timestamps=timestamps_out,
            symbols=tuple(deduped_symbols),
        )

    async def get_aligned_returns_matrix(
        self,
        symbols: Sequence[str],
        start_time: int,
        end_time: int,
        resolution: Resolution = Resolution.ONE_DAY,
        price_field: str = "adj_close",
        return_type: str = "simple",
    ) -> AlignedReturnsResult:
        """Extract a causally aligned returns matrix asynchronously."""
        return await self._manager.run_sync(
            lambda _: self.get_aligned_returns_matrix_sync(
                symbols=symbols,
                start_time=start_time,
                end_time=end_time,
                resolution=resolution,
                price_field=price_field,
                return_type=return_type,
            )
        )

    def close(self) -> None:
        """Close repository and release embedded DuckDB connection locks."""
        self._manager.close()
