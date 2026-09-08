"""High-performance columnar market data repository backed by embedded DuckDB and PyArrow.

Purpose: Implements IMarketDataRepository providing sub-millisecond vectorized bar ingestion and queries.
Dependencies: DuckDBManager, PyArrow, NumPy, domain interfaces and models.
Relationship: Injected into MarketDataService and econometric feature pipelines.
Invariants: Always returns contiguous, chronologically sorted MarketDataBatch instances.
"""

from collections.abc import Sequence

# High-performance columnar database handle
import duckdb

# Numerical array and columnar data exchange libraries
import numpy as np
import pyarrow as pa

# Domain abstractions and value objects
from quant.domain.interfaces import IMarketDataRepository
from quant.domain.models import MarketDataBatch, PriceBar, Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager


class DuckDBMarketDataRepository(IMarketDataRepository):
    """Concrete repository persisting market bars inside DuckDB columnar storage.

    Purpose: Executes zero-copy batch upserts and vectorized chronological range queries.
    Dependencies: DuckDBManager for thread-safe session access; pyarrow for zero-copy bulk ingestion.
    Relationship: Implements IMarketDataRepository interface.
    Invariants: Guarantees ascending chronological ordering on all retrieved batches.
    """

    def __init__(self, manager: DuckDBManager) -> None:
        """Initialize repository with injected DuckDB session manager.

        Purpose: Establishes dependency injection of the embedded database handle.
        Dependencies: DuckDBManager instance.
        """
        # Purpose: Store database session manager
        # Invariant: manager must be an initialized DuckDBManager instance
        self._manager = manager

    async def add_bars_batch(self, bars: Sequence[PriceBar]) -> int:
        """Persist a batch of price bars with upsert semantics using PyArrow zero-copy tables.

        Purpose: Ingests thousands of OHLCV bars into DuckDB in a single vectorized operation.
        Dependencies: pyarrow Table construction, DuckDB INSERT OR REPLACE.
        Post-conditions: Inserts new bars and replaces existing duplicates with identical primary keys.
        """
        # Purpose: Handle empty batch edge case cleanly
        if not bars:
            return 0

        # Purpose: Construct zero-copy columnar PyArrow table from validated PriceBar instances
        # Dependencies: pa.Table.from_arrays with explicit 64-bit primitive types
        arrow_table = pa.Table.from_arrays(
            [
                pa.array([b.asset_id for b in bars], type=pa.string()),
                pa.array([str(b.resolution) for b in bars], type=pa.string()),
                pa.array([b.timestamp for b in bars], type=pa.int64()),
                pa.array([b.open for b in bars], type=pa.float64()),
                pa.array([b.high for b in bars], type=pa.float64()),
                pa.array([b.low for b in bars], type=pa.float64()),
                pa.array([b.close for b in bars], type=pa.float64()),
                pa.array([b.volume for b in bars], type=pa.float64()),
                pa.array([b.vwap for b in bars], type=pa.float64()),
            ],
            names=[
                "asset_id",
                "resolution",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
            ],
        )

        def _insert(conn: duckdb.DuckDBPyConnection) -> int:
            # Purpose: Temporarily register PyArrow table in DuckDB memory space
            # Dependencies: conn.register
            conn.register("incoming_batch_view", arrow_table)

            # Purpose: Execute bulk upsert replacing conflicting composite primary keys
            # Dependencies: INSERT OR REPLACE INTO market_bars
            conn.execute(
                """
                INSERT OR REPLACE INTO market_bars
                SELECT * FROM incoming_batch_view;
                """
            )

            # Purpose: Cleanly release arrow view registration
            conn.unregister("incoming_batch_view")
            return len(bars)

        # Purpose: Delegate database write to background worker thread
        return await self._manager.run_sync(_insert)

    async def get_bars_range(
        self, asset_id: str, start_time: int, end_time: int, resolution: Resolution
    ) -> MarketDataBatch:
        """Retrieve contiguous columnar market bars within timestamp range [start_time, end_time].

        Purpose: Provides chronological price/volume vectors for econometric modeling and backtesting.
        Dependencies: Nanosecond start and end epoch timestamps, Resolution enum.
        Post-conditions: Returns MarketDataBatch with contiguous 1D NumPy arrays sorted by timestamp ASC.
        """
        query = """
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM market_bars
            WHERE asset_id = ? AND resolution = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC;
        """

        def _query(conn: duckdb.DuckDBPyConnection) -> MarketDataBatch:
            # Purpose: Execute indexed range query and fetch results directly as NumPy dictionary of arrays
            # Dependencies: conn.execute(...).fetchnumpy()
            res = conn.execute(
                query, [asset_id, str(resolution), start_time, end_time]
            ).fetchnumpy()

            # Purpose: Construct MarketDataBatch wrapping contiguous 1D NumPy arrays
            # Invariant: Ensure zero-copy contiguous memory buffers with explicit dtypes
            return MarketDataBatch(
                asset_id=asset_id,
                resolution=resolution,
                timestamps=np.ascontiguousarray(res["timestamp"], dtype=np.int64),
                opens=np.ascontiguousarray(res["open"], dtype=np.float64),
                highs=np.ascontiguousarray(res["high"], dtype=np.float64),
                lows=np.ascontiguousarray(res["low"], dtype=np.float64),
                closes=np.ascontiguousarray(res["close"], dtype=np.float64),
                volumes=np.ascontiguousarray(res["volume"], dtype=np.float64),
                vwaps=np.ascontiguousarray(res["vwap"], dtype=np.float64),
            )

        # Purpose: Execute query in threadpool
        return await self._manager.run_sync(_query)

    async def get_latest_bars(
        self, asset_id: str, count: int, resolution: Resolution
    ) -> MarketDataBatch:
        """Retrieve the most recent N contiguous columnar market bars in ascending order.

        Purpose: Supplies sliding observation windows for live realized volatility and signal inference.
        Dependencies: count integer > 0.
        Post-conditions: Returns MarketDataBatch containing up to N bars ordered chronologically ascending.
        """
        # Invariant check: Requested bar count must be strictly positive
        if count <= 0:
            raise ValueError(f"Requested bar count must be strictly positive, got {count}")

        query = """
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM (
                SELECT timestamp, open, high, low, close, volume, vwap
                FROM market_bars
                WHERE asset_id = ? AND resolution = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC;
        """

        def _query(conn: duckdb.DuckDBPyConnection) -> MarketDataBatch:
            # Purpose: Execute inner descending limit query and outer ascending sort
            # Dependencies: conn.execute(...).fetchnumpy()
            res = conn.execute(query, [asset_id, str(resolution), count]).fetchnumpy()

            # Purpose: Construct contiguous columnar batch
            return MarketDataBatch(
                asset_id=asset_id,
                resolution=resolution,
                timestamps=np.ascontiguousarray(res["timestamp"], dtype=np.int64),
                opens=np.ascontiguousarray(res["open"], dtype=np.float64),
                highs=np.ascontiguousarray(res["high"], dtype=np.float64),
                lows=np.ascontiguousarray(res["low"], dtype=np.float64),
                closes=np.ascontiguousarray(res["close"], dtype=np.float64),
                volumes=np.ascontiguousarray(res["volume"], dtype=np.float64),
                vwaps=np.ascontiguousarray(res["vwap"], dtype=np.float64),
            )

        # Purpose: Execute query in threadpool
        return await self._manager.run_sync(_query)

    async def get_available_range(
        self, asset_id: str, resolution: Resolution
    ) -> tuple[int, int] | None:
        """Retrieve earliest and latest available timestamps for an asset.

        Purpose: Identifies data boundaries without scanning the full table.
        Post-conditions: Returns (min_timestamp, max_timestamp) or None if no bars exist.
        """
        query = """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM market_bars
            WHERE asset_id = ? AND resolution = ?;
        """

        def _query(conn: duckdb.DuckDBPyConnection) -> tuple[int, int] | None:
            # Purpose: Execute aggregation query over index
            cursor = conn.execute(query, [asset_id, str(resolution)])
            row = cursor.fetchone()

            # If no rows or NULL timestamp values, return None
            if not row or row[0] is None or row[1] is None:
                return None

            return int(row[0]), int(row[1])

        # Purpose: Execute query in threadpool
        return await self._manager.run_sync(_query)
