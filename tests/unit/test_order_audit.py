"""Unit tests for OrderAuditLogger non-blocking WAL execution audit subsystem.

Governing Standards:
- Rules.md (Rule 1: Line annotations; Rule 2: Diagnostic error codes; Rule 3: Quality gates; Rule 4: Mandatory adversarial red-teaming)
- INV-GW-001: Causal State Machine Monotonicity
- INV-GW-003: Execution Mass Conservation
- INV-GW-005: Strict Boundary & Input Protection
- INV-GW-006: Hot-Path Latency SLA (< 10us per log_report call; 1000 calls < 10ms)
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

import pytest

from quant.execution.audit import OrderAuditLogger
from quant.execution.models import (
    ERR_GW_DISCONNECTED,
    ERR_GW_NON_FINITE_INPUT,
    ExecutionReport,
    GatewayDisconnectedException,
    InvalidOrderInputException,
    OrderSide,
    OrderState,
)


def _make_report(
    report_id: str,
    cl_ord_id: str = "ord-001",
    exchange_order_id: str = "ex-001",
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    exec_type: OrderState = OrderState.FILLED,
    last_quantity: float = 100.0,
    last_price: float = 150.0,
    cum_quantity: float = 100.0,
    leaves_quantity: float = 0.0,
    cum_quote_amount: float = 15000.0,
    average_price: float = 150.0,
    fee: float = 3.0,
    timestamp_ns: int = 1_000_000_000,
    text: str = "Fill confirmed",
) -> ExecutionReport:
    """Helper factory creating valid ExecutionReport value objects for test execution."""
    return ExecutionReport(
        report_id=report_id,
        cl_ord_id=cl_ord_id,
        exchange_order_id=exchange_order_id,
        symbol=symbol,
        side=side,
        exec_type=exec_type,
        last_quantity=last_quantity,
        last_price=last_price,
        cum_quantity=cum_quantity,
        leaves_quantity=leaves_quantity,
        cum_quote_amount=cum_quote_amount,
        average_price=average_price,
        fee=fee,
        timestamp_ns=timestamp_ns,
        text=text,
    )


# ============================================================================
# 1. Constructor Parameter Validation & Defensive Boundary Tests
# ============================================================================


@pytest.mark.parametrize(
    "invalid_path",
    ["", "   ", None, 123, True, False, []],
)
def test_init_invalid_db_path_rejected(invalid_path: object) -> None:
    """Verify OrderAuditLogger rejects empty, whitespace, or non-string db_path."""
    with pytest.raises(InvalidOrderInputException) as exc_info:
        OrderAuditLogger(db_path=invalid_path)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "invalid_batch_size",
    [0, -1, -100, True, False, 1.5, math.nan, math.inf, None, "100"],
)
def test_init_invalid_batch_size_rejected(invalid_batch_size: object) -> None:
    """Verify OrderAuditLogger rejects invalid batch_size parameters."""
    with pytest.raises(InvalidOrderInputException) as exc_info:
        OrderAuditLogger(batch_size=invalid_batch_size)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "invalid_flush_interval",
    [0.0, -0.05, -1.0, True, False, math.nan, math.inf, -math.inf, None, "0.05"],
)
def test_init_invalid_flush_interval_rejected(invalid_flush_interval: object) -> None:
    """Verify OrderAuditLogger rejects non-positive or non-finite flush_interval_seconds."""
    with pytest.raises(InvalidOrderInputException) as exc_info:
        OrderAuditLogger(flush_interval_seconds=invalid_flush_interval)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "invalid_queue_size",
    [0, -1, -10_000, True, False, 50.5, math.nan, math.inf, None, "50000"],
)
def test_init_invalid_max_queue_size_rejected(invalid_queue_size: object) -> None:
    """Verify OrderAuditLogger rejects non-positive or boolean max_queue_size."""
    with pytest.raises(InvalidOrderInputException) as exc_info:
        OrderAuditLogger(max_queue_size=invalid_queue_size)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_GW_NON_FINITE_INPUT


def test_init_creates_nested_directory(tmp_path: Path) -> None:
    """Verify OrderAuditLogger automatically creates parent directory for SQLite file."""
    nested_db = tmp_path / "deep" / "nested" / "dir" / "audit.db"
    assert not nested_db.parent.exists()
    logger = OrderAuditLogger(db_path=str(nested_db))
    assert nested_db.parent.exists()
    assert logger.db_path == str(nested_db)
    assert logger.batch_size == 100
    assert logger.flush_interval_seconds == 0.05
    assert logger.max_queue_size == 50_000
    assert not logger.is_running
    assert logger.pending_count == 0


# ============================================================================
# 2. Lifecycle & Idempotent Start/Stop Verification
# ============================================================================


@pytest.mark.asyncio
async def test_start_and_stop_lifecycle(tmp_path: Path) -> None:
    """Verify standard start and stop lifecycle transitions."""
    db_file = tmp_path / "lifecycle.db"
    logger = OrderAuditLogger(db_path=str(db_file))

    assert not logger.is_running
    await logger.start()
    assert logger.is_running

    await logger.stop()
    assert not logger.is_running


@pytest.mark.asyncio
async def test_double_start_is_graceful_and_idempotent(tmp_path: Path) -> None:
    """Verify calling start() multiple times is completely idempotent."""
    db_file = tmp_path / "double_start.db"
    logger = OrderAuditLogger(db_path=str(db_file))

    await logger.start()
    assert logger.is_running

    # Second start should return without error or creating a duplicate worker
    await logger.start()
    assert logger.is_running

    await logger.stop()
    assert not logger.is_running


@pytest.mark.asyncio
async def test_double_stop_is_graceful_and_idempotent(tmp_path: Path) -> None:
    """Verify calling stop() multiple times or before start() is completely safe."""
    db_file = tmp_path / "double_stop.db"
    logger = OrderAuditLogger(db_path=str(db_file))

    # Stop before start should be a clean no-op
    await logger.stop()
    assert not logger.is_running

    await logger.start()
    assert logger.is_running

    await logger.stop()
    assert not logger.is_running

    # Second stop should be a clean no-op
    await logger.stop()
    assert not logger.is_running


# ============================================================================
# 3. Non-Blocking Ingestion & Background Flush Mechanics
# ============================================================================


@pytest.mark.asyncio
async def test_log_single_report_and_background_flush(tmp_path: Path) -> None:
    """Verify single report is placed in queue and flushed by the background worker."""
    db_file = tmp_path / "single_flush.db"
    logger = OrderAuditLogger(db_path=str(db_file), flush_interval_seconds=0.03)
    await logger.start()

    report = _make_report("rep-001", cl_ord_id="ord-alpha", symbol="AAPL", last_price=155.0)
    logger.log_report(report)

    # Immediately after logging, pending count is 1 (or 0 if worker already flushed)
    assert logger.pending_count in (0, 1)

    # Wait for background flush loop interval (0.03s + margin)
    await asyncio.sleep(0.08)
    assert logger.pending_count == 0

    # Query reports and verify exact fields
    results = await logger.query_reports(cl_ord_id="ord-alpha")
    assert len(results) == 1
    row = results[0]
    assert row["report_id"] == "rep-001"
    assert row["cl_ord_id"] == "ord-alpha"
    assert row["exchange_order_id"] == "ex-001"
    assert row["symbol"] == "AAPL"
    assert row["side"] == "BUY"
    assert row["exec_type"] == "FILLED"
    assert math.isclose(row["last_quantity"], 100.0)
    assert math.isclose(row["last_price"], 155.0)
    assert math.isclose(row["cum_quantity"], 100.0)
    assert math.isclose(row["leaves_quantity"], 0.0)
    assert math.isclose(row["fee"], 3.0)
    assert row["timestamp_ns"] == 1_000_000_000
    assert row["text"] == "Fill confirmed"

    await logger.stop()


@pytest.mark.asyncio
async def test_log_batch_reports_and_manual_flush(tmp_path: Path) -> None:
    """Verify logging multiple reports across chunks and manually flushing."""
    db_file = tmp_path / "batch_flush.db"
    logger = OrderAuditLogger(db_path=str(db_file), batch_size=25, flush_interval_seconds=10.0)
    await logger.start()

    # Log 60 reports (spanning multiple chunks of batch_size 25)
    for i in range(60):
        rep = _make_report(
            report_id=f"rep-batch-{i:03d}",
            cl_ord_id=f"ord-{i // 10}",
            symbol="MSFT",
            last_price=300.0 + i,
            timestamp_ns=1_000_000_000 + i * 1_000_000,
        )
        logger.log_report(rep)

    assert logger.pending_count == 60

    # Trigger explicit flush
    await logger.flush()
    assert logger.pending_count == 0

    results = await logger.query_reports(symbol="MSFT", limit=100)
    assert len(results) == 60
    assert results[0]["report_id"] == "rep-batch-000"
    assert results[-1]["report_id"] == "rep-batch-059"

    await logger.stop()


# ============================================================================
# 4. Latency SLA Benchmark: Sub-10us Non-Blocking Dispatch (INV-GW-006)
# ============================================================================


@pytest.mark.asyncio
async def test_non_blocking_hot_path_speed(tmp_path: Path) -> None:
    """Verify 1,000 log_report calls complete in < 10ms (< 10us per call) under INV-GW-006."""
    db_file = tmp_path / "latency.db"
    logger = OrderAuditLogger(
        db_path=str(db_file),
        batch_size=200,
        flush_interval_seconds=5.0,  # Prevent background flush from interfering with pure enqueue benchmark
        max_queue_size=10_000,
    )
    await logger.start()

    reports = [
        _make_report(
            report_id=f"rep-sla-{i:04d}",
            cl_ord_id=f"ord-{i:04d}",
            symbol="NVDA",
            timestamp_ns=2_000_000_000 + i,
        )
        for i in range(1_000)
    ]

    # Benchmark pure hot-path enqueue latency
    t0 = time.perf_counter()
    for rep in reports:
        logger.log_report(rep)
    elapsed_seconds = time.perf_counter() - t0

    elapsed_ms = elapsed_seconds * 1000.0
    avg_us_per_call = (elapsed_seconds / 1000.0) * 1_000_000.0

    # Assert SLA: 1,000 calls in < 10ms (< 10us per call)
    assert elapsed_ms < 10.0, f"Hot path took {elapsed_ms:.3f}ms for 1000 calls (SLA: < 10ms)"
    assert avg_us_per_call < 10.0, f"Average call took {avg_us_per_call:.2f}us (SLA: < 10us)"

    # Now flush and verify all 1000 reports were persisted
    await logger.flush()
    assert logger.pending_count == 0

    results = await logger.query_reports(symbol="NVDA", limit=2000)
    assert len(results) == 1_000

    await logger.stop()


# ============================================================================
# 5. Persistence Across Logger Instances
# ============================================================================


@pytest.mark.asyncio
async def test_persistence_across_separate_instances(tmp_path: Path) -> None:
    """Verify reports logged and flushed survive process restart and logger re-instantiation."""
    db_file = tmp_path / "persistence.db"

    # Instance 1: Write and flush 10 reports, then cleanly stop
    logger1 = OrderAuditLogger(db_path=str(db_file))
    await logger1.start()

    for i in range(10):
        rep = _make_report(
            report_id=f"persist-{i}",
            cl_ord_id=f"ord-persist-{i}",
            symbol="TSLA",
            last_price=200.0 + i,
            timestamp_ns=3_000_000_000 + i * 100,
        )
        logger1.log_report(rep)

    await logger1.stop()
    assert not logger1.is_running

    # Instance 2: Open same DB, start, and query reports
    logger2 = OrderAuditLogger(db_path=str(db_file))
    await logger2.start()

    reports = await logger2.query_reports(symbol="TSLA", limit=50)
    assert len(reports) == 10
    for i, r in enumerate(reports):
        assert r["report_id"] == f"persist-{i}"
        assert r["cl_ord_id"] == f"ord-persist-{i}"
        assert r["symbol"] == "TSLA"
        assert math.isclose(r["last_price"], 200.0 + i)

    await logger2.stop()


# ============================================================================
# 6. Query Filtering & Sorting Tests
# ============================================================================


@pytest.mark.asyncio
async def test_query_filtering_by_cl_ord_id_and_symbol(tmp_path: Path) -> None:
    """Verify query_reports correctly filters by cl_ord_id, symbol, and respects limit."""
    db_file = tmp_path / "filtering.db"
    logger = OrderAuditLogger(db_path=str(db_file))
    await logger.start()

    # Seed data:
    # 2 AAPL reports for ord-1
    # 1 AAPL report for ord-2
    # 2 GOOG reports for ord-3
    logger.log_report(_make_report("r1", cl_ord_id="ord-1", symbol="AAPL", timestamp_ns=100))
    logger.log_report(_make_report("r2", cl_ord_id="ord-1", symbol="AAPL", timestamp_ns=200))
    logger.log_report(_make_report("r3", cl_ord_id="ord-2", symbol="AAPL", timestamp_ns=300))
    logger.log_report(_make_report("r4", cl_ord_id="ord-3", symbol="GOOG", timestamp_ns=400))
    logger.log_report(_make_report("r5", cl_ord_id="ord-3", symbol="GOOG", timestamp_ns=500))

    await logger.flush()

    # 1. Filter by cl_ord_id="ord-1" -> expect r1, r2
    res_ord1 = await logger.query_reports(cl_ord_id="ord-1")
    assert [r["report_id"] for r in res_ord1] == ["r1", "r2"]

    # 2. Filter by symbol="AAPL" -> expect r1, r2, r3
    res_aapl = await logger.query_reports(symbol="AAPL")
    assert [r["report_id"] for r in res_aapl] == ["r1", "r2", "r3"]

    # 3. Filter by symbol="GOOG" -> expect r4, r5
    res_goog = await logger.query_reports(symbol="GOOG")
    assert [r["report_id"] for r in res_goog] == ["r4", "r5"]

    # 4. Filter by both cl_ord_id and symbol
    res_both = await logger.query_reports(cl_ord_id="ord-1", symbol="AAPL")
    assert [r["report_id"] for r in res_both] == ["r1", "r2"]

    # 5. Filter with non-matching combo
    res_none = await logger.query_reports(cl_ord_id="ord-1", symbol="GOOG")
    assert len(res_none) == 0

    # 6. Limit parameter truncation
    res_limit = await logger.query_reports(limit=2)
    assert len(res_limit) == 2
    assert [r["report_id"] for r in res_limit] == ["r1", "r2"]

    await logger.stop()


# ============================================================================
# 7. Defensive Rejection & Exception Checks
# ============================================================================


def test_log_report_rejects_non_execution_report(tmp_path: Path) -> None:
    """Verify log_report raises InvalidOrderInputException on non-ExecutionReport objects."""
    logger = OrderAuditLogger(db_path=str(tmp_path / "reject.db"))

    with pytest.raises(InvalidOrderInputException) as exc_info1:
        logger.log_report({"report_id": "bad"})  # type: ignore[arg-type]
    assert exc_info1.value.code == ERR_GW_NON_FINITE_INPUT

    with pytest.raises(InvalidOrderInputException) as exc_info2:
        logger.log_report(None)  # type: ignore[arg-type]
    assert exc_info2.value.code == ERR_GW_NON_FINITE_INPUT

    with pytest.raises(InvalidOrderInputException) as exc_info3:
        logger.log_report("invalid string")  # type: ignore[arg-type]
    assert exc_info3.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.asyncio
async def test_query_reports_before_start_raises_disconnected(tmp_path: Path) -> None:
    """Verify query_reports raises GatewayDisconnectedException when not started."""
    logger = OrderAuditLogger(db_path=str(tmp_path / "disconnected.db"))

    with pytest.raises(GatewayDisconnectedException) as exc_info:
        await logger.query_reports()
    assert exc_info.value.code == ERR_GW_DISCONNECTED


@pytest.mark.asyncio
async def test_query_reports_after_stop_raises_disconnected(tmp_path: Path) -> None:
    """Verify query_reports raises GatewayDisconnectedException when connection closed."""
    logger = OrderAuditLogger(db_path=str(tmp_path / "stop_disconnected.db"))
    await logger.start()
    await logger.stop()

    with pytest.raises(GatewayDisconnectedException) as exc_info:
        await logger.query_reports()
    assert exc_info.value.code == ERR_GW_DISCONNECTED


@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, -50, True, False, 10.5, math.nan, math.inf, None, "100"],
)
@pytest.mark.asyncio
async def test_query_reports_invalid_limit_rejected(tmp_path: Path, invalid_limit: object) -> None:
    """Verify query_reports rejects non-positive or boolean limits."""
    logger = OrderAuditLogger(db_path=str(tmp_path / "limit.db"))
    await logger.start()

    with pytest.raises(InvalidOrderInputException) as exc_info:
        await logger.query_reports(limit=invalid_limit)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_GW_NON_FINITE_INPUT

    await logger.stop()


@pytest.mark.parametrize(
    "invalid_filter",
    ["", "   ", 123, True, False, []],
)
@pytest.mark.asyncio
async def test_query_reports_invalid_filters_rejected(
    tmp_path: Path, invalid_filter: object
) -> None:
    """Verify query_reports rejects empty strings or non-strings for cl_ord_id and symbol."""
    logger = OrderAuditLogger(db_path=str(tmp_path / "filter_validation.db"))
    await logger.start()

    with pytest.raises(InvalidOrderInputException) as exc_info1:
        await logger.query_reports(cl_ord_id=invalid_filter)  # type: ignore[arg-type]
    assert exc_info1.value.code == ERR_GW_NON_FINITE_INPUT

    with pytest.raises(InvalidOrderInputException) as exc_info2:
        await logger.query_reports(symbol=invalid_filter)  # type: ignore[arg-type]
    assert exc_info2.value.code == ERR_GW_NON_FINITE_INPUT

    await logger.stop()


def test_queue_full_raises_queue_full(tmp_path: Path) -> None:
    """Verify log_report raises asyncio.QueueFull when max_queue_size capacity is reached."""
    logger = OrderAuditLogger(db_path=str(tmp_path / "queue_full.db"), max_queue_size=2)
    rep1 = _make_report("rep-1")
    rep2 = _make_report("rep-2")
    rep3 = _make_report("rep-3")

    logger.log_report(rep1)
    logger.log_report(rep2)
    assert logger.pending_count == 2

    # Third report should breach queue capacity
    with pytest.raises(asyncio.QueueFull):
        logger.log_report(rep3)


@pytest.mark.asyncio
async def test_in_memory_db_mode() -> None:
    """Verify OrderAuditLogger operates cleanly with SQLite in-memory database :memory:."""
    logger = OrderAuditLogger(db_path=":memory:")
    await logger.start()
    assert logger.is_running

    rep = _make_report("mem-01", cl_ord_id="ord-mem", symbol="ETH")
    logger.log_report(rep)
    await logger.flush()

    res = await logger.query_reports(cl_ord_id="ord-mem")
    assert len(res) == 1
    assert res[0]["symbol"] == "ETH"

    await logger.stop()
    assert not logger.is_running


@pytest.mark.asyncio
async def test_stop_flushes_pending_reports_before_closing(tmp_path: Path) -> None:
    """Verify stop() flushes any remaining items in queue before closing database."""
    db_file = tmp_path / "stop_flush.db"
    logger = OrderAuditLogger(
        db_path=str(db_file),
        flush_interval_seconds=100.0,  # Long interval so background worker doesn't flush
    )
    await logger.start()

    logger.log_report(_make_report("rep-stop-1", cl_ord_id="ord-stop"))
    logger.log_report(_make_report("rep-stop-2", cl_ord_id="ord-stop"))
    assert logger.pending_count == 2

    # stop() should drain and flush the 2 items
    await logger.stop()
    assert logger.pending_count == 0

    # Re-open and verify both items were written to SQLite
    logger2 = OrderAuditLogger(db_path=str(db_file))
    await logger2.start()
    reports = await logger2.query_reports(cl_ord_id="ord-stop")
    assert len(reports) == 2
    assert [r["report_id"] for r in reports] == ["rep-stop-1", "rep-stop-2"]
    await logger2.stop()
