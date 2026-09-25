"""Embedded DuckDB session and connection manager for high-throughput time series.

Purpose: Manages embedded DuckDB connection lifecycles, schema initialization, and thread safety.
Dependencies: duckdb, asyncio for event-loop offloading, threading for concurrency protection.
Relationship: Consumed by DuckDBMarketDataRepository to execute columnar queries without blocking FastAPI.
Invariants: Automatically ensures market_bars table and primary keys exist before query execution.
"""

import asyncio
import os
import threading
import time
from collections.abc import Callable
from typing import Final, TypeVar

# High-performance analytical columnar database engine
import duckdb

# Application settings providing default DUCKDB_PATH configuration
from quant.core.config import get_settings

# Generic return type for threadpool offloading
T = TypeVar("T")

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_DATA_DB_LOCKED: Final[str] = "ERR-DATA-008"


class DatabaseLockedError(Exception):
    """Raised when DuckDB database file is locked by another process and cannot be accessed.

    Functional Purpose:
        Signals database lock contention preventing read/write transactions without silent fallback.
    Explicit Dependency Tracking:
        Exception, ERR_DATA_DB_LOCKED.
    Structural Relationship:
        Raised by DuckDBManager during connection acquisition failure.
    Defensive Invariant:
        Contains deterministic fault code ERR-DATA-008.
    """

    def __init__(self, message: str, code: str = ERR_DATA_DB_LOCKED) -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class DuckDBManager:
    """Manages embedded DuckDB connection lifecycles and schema migrations.

    Purpose: Provides thread-safe, non-blocking asynchronous access to embedded columnar database.
    Dependencies: duckdb library, asyncio for thread offloading, threading.Lock.
    Relationship: Injected into DuckDBMarketDataRepository.
    Invariants: Ensures schema tables and indexes exist prior to executing any read/write queries.
    """

    def __init__(
        self,
        database_path: str | None = None,
        read_only: bool = False,
        max_retries: int = 5,
    ) -> None:
        """Initialize DuckDB connection and ensure database schema is prepared.

        Purpose: Establishes active embedded database handle and provisions tables.
        Dependencies: Target database file path or ':memory:' sentinel.
        """
        # Purpose: Resolve target database path from settings if not explicitly injected
        # Dependencies: get_settings().DUCKDB_PATH
        # Invariant: Must be a non-empty string path or ':memory:'
        self._db_path = database_path or get_settings().DUCKDB_PATH

        # Purpose: Mutex lock to serialize write transactions and connection state access across threads
        # Dependencies: threading.Lock
        self._lock = threading.Lock()

        # Purpose: Guard flag ensuring schema initialization runs exactly once
        self._initialized = False

        # Purpose: Ensure target filesystem parent directory exists for file-backed databases
        # Dependencies: os.makedirs, os.path.dirname, os.path.abspath
        # Invariant: Directory is created with mode 777 (subject to umask)
        if self._db_path != ":memory:":
            parent_dir = os.path.dirname(os.path.abspath(self._db_path))
            os.makedirs(parent_dir, exist_ok=True)

        # Purpose: Establish connection handle to embedded DuckDB database with retries
        # Dependencies: duckdb.connect, time.sleep
        # Invariant: File-backed database never silently falls back to in-memory storage
        if self._db_path == ":memory:":
            self._conn = duckdb.connect(database=":memory:")
            self._ensure_schema()
        else:
            last_err: Exception | None = None
            for attempt in range(max_retries):
                try:
                    self._conn = duckdb.connect(database=self._db_path, read_only=read_only)
                    if not read_only:
                        self._ensure_schema()
                    else:
                        self._initialized = True
                    break
                except duckdb.IOException as io_err:
                    last_err = io_err
                    if attempt < max_retries - 1:
                        time.sleep(0.05 * (2**attempt))
            else:
                raise DatabaseLockedError(
                    f"Embedded DuckDB at '{self._db_path}' could not be locked after {max_retries} attempts: {last_err}",
                    code=ERR_DATA_DB_LOCKED,
                ) from last_err

    def _ensure_schema(self) -> None:
        """Initialize market_bars table and composite lookup index.

        Purpose: Establishes primary relational and index structures for OHLCV bars.
        Dependencies: self._conn.execute
        Invariants: Table market_bars has composite primary key (asset_id, resolution, timestamp).
        """
        with self._lock:
            if not self._initialized:
                # Purpose: Create table for columnar storage with strict primitive datatypes
                # Invariant: Primary key enforces uniqueness across asset, timeframe, and timestamp
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_bars (
                        asset_id VARCHAR NOT NULL,
                        resolution VARCHAR NOT NULL,
                        timestamp BIGINT NOT NULL,
                        open DOUBLE NOT NULL,
                        high DOUBLE NOT NULL,
                        low DOUBLE NOT NULL,
                        close DOUBLE NOT NULL,
                        volume DOUBLE NOT NULL,
                        vwap DOUBLE NOT NULL,
                        PRIMARY KEY (asset_id, resolution, timestamp)
                    );
                    """
                )

                # Purpose: Create secondary composite index for accelerated time-range slicing
                # Invariant: Covers asset_id, resolution, and timestamp for index-only range scans
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_market_bars_lookup
                    ON market_bars (asset_id, resolution, timestamp);
                    """
                )

                # Purpose: Create historical_bars table for institutional split/dividend adjusted bars
                # Invariant: Primary key enforces uniqueness across symbol, resolution, and timestamp
                self._conn.execute(
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

                # Purpose: Create secondary composite index for accelerated historical lookup
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_historical_bars_lookup
                    ON historical_bars (symbol, resolution, timestamp);
                    """
                )

                # Set initialized state flag
                self._initialized = True

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Return the active DuckDB connection handle.

        Purpose: Exposes underlying connection for direct execution within synchronized contexts.
        Dependencies: self._conn
        """
        return self._conn

    async def run_sync(self, func: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """Execute synchronous DuckDB operation inside a threadpool to prevent blocking the ASGI event loop.

        Purpose: Offloads CPU/disk bound queries to Python worker threads while maintaining thread safety.
        Dependencies: asyncio.to_thread, self._lock, callable query function.
        Invariants: Serializes access via self._lock to ensure thread safety across concurrent requests.
        """

        def _wrapper() -> T:
            with self._lock:
                return func(self._conn)

        # Purpose: Delegate execution to default asyncio executor threadpool
        return await asyncio.to_thread(_wrapper)

    def close(self) -> None:
        """Close embedded database connection cleanly.

        Purpose: Flushes write-ahead log (WAL) and releases file locks.
        Dependencies: self._conn.close()
        """
        with self._lock:
            self._conn.close()
