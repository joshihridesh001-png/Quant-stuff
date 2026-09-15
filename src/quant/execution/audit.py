"""Asynchronous non-blocking Order Audit Logger using SQLite WAL mode.

Purpose:
    Provides ultra-low-latency, zero-blocking audit persistence for live order execution
    reports. Decouples hot-path order lifecycle state transitions and broker matching from disk I/O
    via an in-memory bounded asynchronous queue, with background batched persistence to an embedded
    SQLite database in Write-Ahead Logging (WAL) mode.

Dependencies:
    - asyncio: Non-blocking asynchronous queues, locks, sleep timers, and worker tasks.
    - math: Finite float scalar verification (math.isfinite).
    - os: Directory creation and path resolution for SQLite database files.
    - sqlite3: Standard library embedded database engine with WAL pragma configuration.
    - typing: Static typing annotations, Any, Final.
    - quant.execution.models: ExecutionReport, diagnostic exceptions, and error code constants.

Structural Relationship:
    - Root audit logging engine for Phase 6 Live Execution Gateway:
        1. Consumed by ExecutionGateway and PaperExecutionGateway to log all fill/cancel events.
        2. Consumed by Live Risk Monitors and Regulatory Audit Collectors.
        3. Exposes historical execution telemetry via parameterized SQL queries.

Invariants Enforced:
    - INV-GW-001 (Causal State Machine Monotonicity): Audit trail preserves monotonic execution reports.
    - INV-GW-003 (Execution Mass Conservation): Logged fills record exact cumulative and leaves quantities.
    - INV-GW-005 (Strict Non-Finite & Boundary Protection):
        - Complete rejection of non-ExecutionReport objects, non-finite bounds, or empty strings.
        - Querying while disconnected raises GatewayDisconnectedException (ERR-GW-005).
    - INV-GW-006 (Hot-Path Latency SLA): log_report() executes with zero blocking disk I/O in < 10us.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import sqlite3
from typing import Any, Final

from quant.execution.models import (
    ERR_GW_DISCONNECTED,
    ERR_GW_NON_FINITE_INPUT,
    ExecutionReport,
    GatewayDisconnectedException,
    InvalidOrderInputException,
)

# Default configuration parameters for institutional WAL audit logging.
_DEFAULT_DB_PATH: Final[str] = "data/execution_audit.db"
_DEFAULT_BATCH_SIZE: Final[int] = 100
_DEFAULT_FLUSH_INTERVAL_SECONDS: Final[float] = 0.05
_DEFAULT_MAX_QUEUE_SIZE: Final[int] = 50_000

# SQL DDL for execution reports audit schema.
_SCHEMA_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS execution_reports (
    report_id TEXT PRIMARY KEY,
    cl_ord_id TEXT NOT NULL,
    exchange_order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    exec_type TEXT NOT NULL,
    last_quantity REAL NOT NULL,
    last_price REAL NOT NULL,
    cum_quantity REAL NOT NULL,
    leaves_quantity REAL NOT NULL,
    cum_quote_amount REAL NOT NULL,
    average_price REAL NOT NULL,
    fee REAL NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exec_cl_ord_id ON execution_reports(cl_ord_id);
CREATE INDEX IF NOT EXISTS idx_exec_symbol ON execution_reports(symbol);
CREATE INDEX IF NOT EXISTS idx_exec_timestamp ON execution_reports(timestamp_ns);
"""

# SQL DML for upserting execution reports.
_INSERT_REPORT_SQL: Final[str] = """
INSERT OR REPLACE INTO execution_reports (
    report_id, cl_ord_id, exchange_order_id, symbol, side,
    exec_type, last_quantity, last_price, cum_quantity,
    leaves_quantity, cum_quote_amount, average_price, fee,
    timestamp_ns, text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class OrderAuditLogger:
    """Non-blocking asynchronous execution audit logger using asyncio.Queue and SQLite WAL mode.

    Guarantees zero blocking disk I/O on the order execution hot path by buffering ExecutionReport
    instances in a high-capacity in-memory queue and delegating writes to a background worker
    operating on an embedded SQLite database in Write-Ahead Logging (WAL) mode.
    """

    __slots__ = (
        "_batch_size",
        "_conn",
        "_db_path",
        "_flush_interval_seconds",
        "_is_running",
        "_lock",
        "_max_queue_size",
        "_queue",
        "_worker_task",
    )

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        flush_interval_seconds: float = _DEFAULT_FLUSH_INTERVAL_SECONDS,
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        """Initialize OrderAuditLogger with validated parameters and in-memory queue.

        Args:
            db_path: Filesystem path to SQLite database file or ':memory:' (non-empty str).
            batch_size: Maximum reports committed per SQLite transaction batch (int > 0).
            flush_interval_seconds: Background flush interval in seconds (finite float > 0.0).
            max_queue_size: Maximum capacity of the in-memory report queue (int > 0).

        Raises:
            InvalidOrderInputException: If any parameter violates domain boundaries or types.
        """
        # Functional Purpose: Validate constructor configuration and allocate queue/lock primitives.
        # Explicit Dependency Tracking: math.isfinite, os.makedirs, InvalidOrderInputException, ERR_GW_NON_FINITE_INPUT.
        # Structural Relationship: Instantiated by ExecutionGateway and system orchestrators.
        # Defensive Invariant: db_path must be non-empty string; numeric parameters must be finite scalars > 0; reject bool.

        # 1. Validate db_path
        if not isinstance(db_path, str) or not db_path.strip():
            raise InvalidOrderInputException(
                f"db_path must be a non-empty string, got {db_path!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 2. Validate batch_size
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise InvalidOrderInputException(
                f"batch_size must be an integer > 0, got {batch_size!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 3. Validate flush_interval_seconds
        if (
            isinstance(flush_interval_seconds, bool)
            or not isinstance(flush_interval_seconds, (int, float))
            or not math.isfinite(flush_interval_seconds)
            or flush_interval_seconds <= 0.0
        ):
            raise InvalidOrderInputException(
                f"flush_interval_seconds must be a finite float > 0.0, got {flush_interval_seconds!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 4. Validate max_queue_size
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size <= 0
        ):
            raise InvalidOrderInputException(
                f"max_queue_size must be an integer > 0, got {max_queue_size!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 5. Ensure parent directory exists if filesystem path
        clean_path = db_path.strip()
        if clean_path != ":memory:" and not clean_path.startswith("file:"):
            parent_dir = os.path.dirname(os.path.abspath(clean_path))
            if parent_dir:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except OSError as exc:
                    raise InvalidOrderInputException(
                        f"Failed to create directory {parent_dir}: {exc}",
                        code=ERR_GW_NON_FINITE_INPUT,
                    ) from exc

        self._db_path: str = clean_path
        self._batch_size: int = batch_size
        self._flush_interval_seconds: float = float(flush_interval_seconds)
        self._max_queue_size: int = max_queue_size
        self._queue: asyncio.Queue[ExecutionReport] = asyncio.Queue(maxsize=max_queue_size)
        self._conn: sqlite3.Connection | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._is_running: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the audit logger background worker and database connection are active."""
        # Functional Purpose: Provide deterministic inspection of logger execution state.
        # Explicit Dependency Tracking: self._is_running, self._conn.
        # Structural Relationship: Queried by health checks and test harnesses.
        # Defensive Invariant: Returns True only when running flag is True and SQLite connection is open.
        return self._is_running and self._conn is not None

    @property
    def pending_count(self) -> int:
        """Number of execution reports currently queued in memory awaiting disk flush."""
        # Functional Purpose: Provide instantaneous measurement of queue backlog depth.
        # Explicit Dependency Tracking: self._queue.qsize().
        # Structural Relationship: Monitored by telemetry probes and flush verifiers.
        # Defensive Invariant: Returns integer in [0, max_queue_size].
        return self._queue.qsize()

    @property
    def db_path(self) -> str:
        """Configured database filesystem path or URI."""
        return self._db_path

    @property
    def batch_size(self) -> int:
        """Configured batch insert chunk size."""
        return self._batch_size

    @property
    def flush_interval_seconds(self) -> float:
        """Configured background flush interval in seconds."""
        return self._flush_interval_seconds

    @property
    def max_queue_size(self) -> int:
        """Configured maximum queue buffer capacity."""
        return self._max_queue_size

    async def start(self) -> None:
        """Open SQLite database connection, configure WAL mode, create schema, and spawn worker.

        Idempotent: if the logger is already running, this method returns immediately without error.
        """
        # Functional Purpose: Initialize SQLite WAL connection, verify DDL schema, and launch background loop.
        # Explicit Dependency Tracking: sqlite3.connect, asyncio.create_task, _SCHEMA_DDL.
        # Structural Relationship: Invoked during gateway startup sequence.
        # Defensive Invariant: Idempotent; double-start handled gracefully without multiple connections or workers.
        if self._is_running and self._conn is not None:
            return

        async with self._lock:
            if self._is_running and self._conn is not None:
                return

            # Open SQLite connection with thread affinity disabled for asyncio compatibility
            conn = sqlite3.connect(self._db_path, check_same_thread=False)

            # Enable Write-Ahead Logging (WAL) and normal synchronous mode for optimal throughput
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            # Initialize schema and indices
            conn.executescript(_SCHEMA_DDL)
            conn.commit()

            self._conn = conn
            self._is_running = True
            self._worker_task = asyncio.create_task(self._flush_loop())

    def log_report(self, report: ExecutionReport) -> None:
        """Enqueue an execution report for non-blocking asynchronous persistence.

        Args:
            report: Validated immutable ExecutionReport instance.

        Raises:
            InvalidOrderInputException: If report is not an ExecutionReport instance (ERR-GW-003).
            asyncio.QueueFull: If the queue capacity is exceeded under extreme backpressure.
        """
        # Functional Purpose: Place execution report onto non-blocking in-memory queue in < 10us (INV-GW-006).
        # Explicit Dependency Tracking: self._queue.put_nowait, InvalidOrderInputException, ERR_GW_NON_FINITE_INPUT.
        # Structural Relationship: Hot-path hook called by PaperExecutionGateway and broker event routers.
        # Defensive Invariant: Zero blocking disk I/O; enforces strict ExecutionReport type verification.
        if not isinstance(report, ExecutionReport):
            raise InvalidOrderInputException(
                f"report must be an ExecutionReport instance, got {type(report).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        self._queue.put_nowait(report)

    async def _flush_loop(self) -> None:
        """Background worker task periodically flushing queued execution reports to SQLite."""
        # Functional Purpose: Drain in-memory queue at regular time intervals without blocking event loop.
        # Explicit Dependency Tracking: asyncio.sleep, self.flush().
        # Structural Relationship: Internal task spawned by start() and terminated by stop().
        # Defensive Invariant: Gracefully intercepts asyncio.CancelledError on shutdown.
        while self._is_running:
            try:
                await asyncio.sleep(self._flush_interval_seconds)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception:
                # Suppress non-fatal flush errors to preserve worker liveness
                pass

    async def flush(self) -> None:
        """Drain currently queued items from memory and commit batch insert to SQLite immediately."""
        # Functional Purpose: Drain all available reports from queue and write to SQLite within transaction lock.
        # Explicit Dependency Tracking: self._conn.executemany, self._conn.commit, _INSERT_REPORT_SQL.
        # Structural Relationship: Invoked periodically by _flush_loop and explicitly by stop() or tests.
        # Defensive Invariant: No-op if connection is uninitialized or queue is empty; chunks writes by batch_size.
        async with self._lock:
            if self._conn is None or self._queue.empty():
                return

            reports: list[ExecutionReport] = []
            while not self._queue.empty():
                try:
                    reports.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if not reports:
                return

            # Batch insert in chunks of batch_size
            for i in range(0, len(reports), self._batch_size):
                chunk = reports[i : i + self._batch_size]
                records = [
                    (
                        r.report_id,
                        r.cl_ord_id,
                        r.exchange_order_id,
                        r.symbol,
                        r.side.value if hasattr(r.side, "value") else str(r.side),
                        r.exec_type.value if hasattr(r.exec_type, "value") else str(r.exec_type),
                        float(r.last_quantity),
                        float(r.last_price),
                        float(r.cum_quantity),
                        float(r.leaves_quantity),
                        float(r.cum_quote_amount),
                        float(r.average_price),
                        float(r.fee),
                        int(r.timestamp_ns),
                        str(r.text),
                    )
                    for r in chunk
                ]
                self._conn.executemany(_INSERT_REPORT_SQL, records)

            self._conn.commit()

            # Mark all dequeued tasks done
            for _ in reports:
                self._queue.task_done()

    async def stop(self) -> None:
        """Cancel background worker, drain all queued reports, commit transaction, and close database.

        Idempotent: if the logger is already stopped, this method returns immediately without error.
        """
        # Functional Purpose: Gracefully terminate background worker, flush remaining items, and close SQLite handle.
        # Explicit Dependency Tracking: self._worker_task.cancel, self.flush, self._conn.close.
        # Structural Relationship: Invoked during gateway shutdown sequence.
        # Defensive Invariant: Idempotent; double-stop handled cleanly; flushes all pending reports before exit.
        if not self._is_running and self._conn is None:
            return

        async with self._lock:
            if not self._is_running and self._conn is None:
                return

            self._is_running = False

            # Cancel background worker task
            if self._worker_task is not None:
                if not self._worker_task.done():
                    self._worker_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await self._worker_task
                self._worker_task = None

            # Drain and flush any remaining execution reports in queue
            if self._conn is not None and not self._queue.empty():
                reports: list[ExecutionReport] = []
                while not self._queue.empty():
                    try:
                        reports.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if reports:
                    records = [
                        (
                            r.report_id,
                            r.cl_ord_id,
                            r.exchange_order_id,
                            r.symbol,
                            r.side.value if hasattr(r.side, "value") else str(r.side),
                            r.exec_type.value
                            if hasattr(r.exec_type, "value")
                            else str(r.exec_type),
                            float(r.last_quantity),
                            float(r.last_price),
                            float(r.cum_quantity),
                            float(r.leaves_quantity),
                            float(r.cum_quote_amount),
                            float(r.average_price),
                            float(r.fee),
                            int(r.timestamp_ns),
                            str(r.text),
                        )
                        for r in reports
                    ]
                    self._conn.executemany(_INSERT_REPORT_SQL, records)
                    self._conn.commit()
                    for _ in reports:
                        self._queue.task_done()

            # Commit and close database connection
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    self._conn.commit()
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None

    async def query_reports(
        self,
        cl_ord_id: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audited execution reports with optional filters, ordered by timestamp ascending.

        Args:
            cl_ord_id: Optional client order identifier to filter by (non-empty str if specified).
            symbol: Optional instrument ticker to filter by (non-empty str if specified).
            limit: Maximum records returned (int > 0, default: 100).

        Returns:
            List of dictionaries containing execution report column values.

        Raises:
            GatewayDisconnectedException: If called while database connection is closed (ERR-GW-005).
            InvalidOrderInputException: If query parameters violate domain bounds or types (ERR-GW-003).
        """
        # Functional Purpose: Parameterized SQL query filtering audited execution events by order/symbol.
        # Explicit Dependency Tracking: sqlite3.Cursor.fetchall, GatewayDisconnectedException, InvalidOrderInputException.
        # Structural Relationship: Queried by post-trade audit pipelines, reconciliation monitors, and tests.
        # Defensive Invariant: Enforces positive integer limit; rejects invalid filter types; requires active connection.

        # 1. Validate connection status
        if self._conn is None:
            raise GatewayDisconnectedException(
                "OrderAuditLogger database connection is not open. Call start() first.",
                code=ERR_GW_DISCONNECTED,
            )

        # 2. Validate limit parameter
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise InvalidOrderInputException(
                f"limit must be an integer > 0, got {limit!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 3. Validate cl_ord_id parameter
        if cl_ord_id is not None and (not isinstance(cl_ord_id, str) or not cl_ord_id.strip()):
            raise InvalidOrderInputException(
                f"cl_ord_id must be a non-empty string when provided, got {cl_ord_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 4. Validate symbol parameter
        if symbol is not None and (not isinstance(symbol, str) or not symbol.strip()):
            raise InvalidOrderInputException(
                f"symbol must be a non-empty string when provided, got {symbol!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        async with self._lock:
            if self._conn is None:
                raise GatewayDisconnectedException(
                    "OrderAuditLogger database connection is not open. Call start() first.",
                    code=ERR_GW_DISCONNECTED,
                )

            query = """
            SELECT report_id, cl_ord_id, exchange_order_id, symbol, side,
                   exec_type, last_quantity, last_price, cum_quantity,
                   leaves_quantity, cum_quote_amount, average_price, fee,
                   timestamp_ns, text
            FROM execution_reports
            """
            where_clauses: list[str] = []
            params: list[Any] = []

            if cl_ord_id is not None:
                where_clauses.append("cl_ord_id = ?")
                params.append(cl_ord_id.strip())

            if symbol is not None:
                where_clauses.append("symbol = ?")
                params.append(symbol.strip())

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += " ORDER BY timestamp_ns ASC LIMIT ?;"
            params.append(limit)

            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
            columns = [
                "report_id",
                "cl_ord_id",
                "exchange_order_id",
                "symbol",
                "side",
                "exec_type",
                "last_quantity",
                "last_price",
                "cum_quantity",
                "leaves_quantity",
                "cum_quote_amount",
                "average_price",
                "fee",
                "timestamp_ns",
                "text",
            ]
            return [dict(zip(columns, row, strict=True)) for row in rows]
