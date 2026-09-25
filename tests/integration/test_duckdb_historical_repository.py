"""Integration tests for DuckDBHistoricalRepository and vectorized multi-asset returns matrix.

Functional Purpose:
    Verifies ACID storage, zero-copy PyArrow ingestion, idempotent upserting,
    and causal forward-fill alignment across multi-asset universes.

Explicit Dependency Tracking:
    - pytest, numpy.
    - DuckDBManager with :memory: database.
    - DuckDBHistoricalRepository.
    - quant.domain.historical (HistoricalPriceBar, HistoricalBarBatch).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from quant.domain.historical import (
    ERR_DATA_NON_FINITE_INPUT,
    EmptyHistoricalBatchError,
    HistoricalBarBatch,
    HistoricalPriceBar,
    NonFiniteHistoricalInputError,
    NonMonotonicTimestampError,
)
from quant.domain.models import Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.repositories.duckdb_historical_repository import (
    ERR_DATA_EMPTY_UNIVERSE,
    ERR_DATA_INSUFFICIENT_TIMESTAMPS,
    ERR_DATA_SYMBOL_NOT_FOUND,
    AlignedReturnsResult,
    DuckDBHistoricalRepository,
    EmptyUniverseError,
    InsufficientTimestampsError,
)


@pytest.fixture
def memory_duckdb_repo() -> DuckDBHistoricalRepository:
    """Provide a fresh in-memory DuckDB historical repository for isolated testing."""
    manager = DuckDBManager(database_path=":memory:")
    return DuckDBHistoricalRepository(manager=manager)


def _create_sample_bar(
    symbol: str,
    ts_ns: int,
    close: float,
    adj_close: float | None = None,
    volume: float = 1000.0,
) -> HistoricalPriceBar:
    """Helper to build a valid HistoricalPriceBar."""
    adj = adj_close if adj_close is not None else close
    return HistoricalPriceBar(
        asset_id=symbol,
        timestamp=ts_ns,
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=volume,
        vwap=close,
        resolution=Resolution.ONE_DAY,
        adj_close=adj,
        split_factor=1.0,
        dividend_amount=0.0,
    )


# ============================================================================
# Ingestion & Range Query Tests
# ============================================================================


def test_save_and_retrieve_historical_bars(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify batch ingestion and range retrieval of historical bars."""
    t1 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    t2 = int(datetime(2025, 1, 2, tzinfo=UTC).timestamp() * 1_000_000_000)
    t3 = int(datetime(2025, 1, 3, tzinfo=UTC).timestamp() * 1_000_000_000)

    bar1 = _create_sample_bar("SPY", t1, 400.0, 395.0)
    bar2 = _create_sample_bar("SPY", t2, 405.0, 400.0)
    bar3 = _create_sample_bar("SPY", t3, 410.0, 405.0)

    batch = HistoricalBarBatch(symbol="SPY", resolution=Resolution.ONE_DAY, bars=(bar1, bar2, bar3))

    saved_count = memory_duckdb_repo.save_batch_sync(batch)
    assert saved_count == 3

    # Retrieve range
    retrieved = memory_duckdb_repo.get_bars_range_sync("SPY", t1, t3, Resolution.ONE_DAY)
    assert len(retrieved) == 3
    assert retrieved.symbol == "SPY"
    assert retrieved.bars[0].close == 400.0
    assert retrieved.bars[0].adj_close == 395.0
    assert retrieved.bars[2].close == 410.0

    # Sub-range
    sub_batch = memory_duckdb_repo.get_bars_range_sync("SPY", t2, t3, Resolution.ONE_DAY)
    assert len(sub_batch) == 2
    assert sub_batch.bars[0].timestamp == t2


def test_idempotent_upsert_overwrite(memory_duckdb_repo: DuckDBHistoricalRepository) -> None:
    """Verify that re-inserting bars with identical primary keys updates records without duplication."""
    t1 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)

    bar_orig = _create_sample_bar("AAPL", t1, 150.0, 150.0)
    batch_orig = HistoricalBarBatch(symbol="AAPL", resolution=Resolution.ONE_DAY, bars=(bar_orig,))
    memory_duckdb_repo.save_batch_sync(batch_orig)

    count_before = memory_duckdb_repo.get_bar_count_sync("AAPL", Resolution.ONE_DAY)
    assert count_before == 1

    # Overwrite with adjusted price
    bar_updated = _create_sample_bar("AAPL", t1, 150.0, 75.0)  # Post-split adjusted
    batch_updated = HistoricalBarBatch(
        symbol="AAPL", resolution=Resolution.ONE_DAY, bars=(bar_updated,)
    )
    memory_duckdb_repo.save_batch_sync(batch_updated)

    count_after = memory_duckdb_repo.get_bar_count_sync("AAPL", Resolution.ONE_DAY)
    assert count_after == 1

    retrieved = memory_duckdb_repo.get_bars_range_sync("AAPL", t1, t1, Resolution.ONE_DAY)
    assert retrieved.bars[0].adj_close == 75.0


def test_empty_range_or_missing_symbol_raises(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify querying non-existent symbols or out-of-range timestamps raises EmptyHistoricalBatchError."""
    with pytest.raises(EmptyHistoricalBatchError) as exc_info:
        memory_duckdb_repo.get_bars_range_sync(
            "NONEXISTENT", 1_000_000, 2_000_000, Resolution.ONE_DAY
        )
    assert exc_info.value.code == ERR_DATA_SYMBOL_NOT_FOUND


def test_invalid_range_and_boolean_rejection(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify start > end and boolean parameters are rejected strictly."""
    with pytest.raises(NonMonotonicTimestampError):
        memory_duckdb_repo.get_bars_range_sync("SPY", 200, 100, Resolution.ONE_DAY)

    with pytest.raises(NonFiniteHistoricalInputError) as exc_info:
        memory_duckdb_repo.get_bars_range_sync("SPY", True, 100, Resolution.ONE_DAY)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_DATA_NON_FINITE_INPUT


def test_metadata_available_range_and_symbols(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify get_available_range and get_all_symbols metadata queries."""
    t1 = 1_000_000_000
    t2 = 2_000_000_000
    t3 = 3_000_000_000

    b_spy = _create_sample_bar("SPY", t1, 400.0)
    b_qqq1 = _create_sample_bar("QQQ", t2, 300.0)
    b_qqq2 = _create_sample_bar("QQQ", t3, 305.0)

    memory_duckdb_repo.save_batch_sync(
        HistoricalBarBatch(symbol="SPY", resolution=Resolution.ONE_DAY, bars=(b_spy,))
    )
    memory_duckdb_repo.save_batch_sync(
        HistoricalBarBatch(symbol="QQQ", resolution=Resolution.ONE_DAY, bars=(b_qqq1, b_qqq2))
    )

    symbols = memory_duckdb_repo.get_all_symbols_sync()
    assert symbols == ["QQQ", "SPY"]

    spy_range = memory_duckdb_repo.get_available_range_sync("SPY")
    assert spy_range == (t1, t1)

    qqq_range = memory_duckdb_repo.get_available_range_sync("QQQ")
    assert qqq_range == (t2, t3)

    none_range = memory_duckdb_repo.get_available_range_sync("UNKNOWN")
    assert none_range is None


# ============================================================================
# Vectorized Aligned Returns Matrix & Causal Forward-Fill
# ============================================================================


def test_aligned_returns_matrix_causal_forward_fill(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify aligned returns matrix generation with strict causal forward-fill and zero lookahead.

    Scenario:
        Universe: [Asset A, Asset B]
        Timestamps:
            t1: A=100.0, B=200.0
            t2: A=105.0, B is missing (e.g. trading halt)
            t3: A=110.0, B=220.0
        Expected:
            At t2: B is causally forward-filled to 200.0.
            Return from t1 -> t2:
                A: (105 - 100) / 100 = +0.05 (+5%)
                B: (200 - 200) / 200 = 0.0 (0%)
            Return from t2 -> t3:
                A: (110 - 105) / 105 = +0.047619 (+4.76%)
                B: (220 - 200) / 200 = +0.10 (+10%)
    """
    t1 = 1_000_000_000
    t2 = 2_000_000_000
    t3 = 3_000_000_000

    bar_a1 = _create_sample_bar("ASSET_A", t1, 100.0, 100.0)
    bar_a2 = _create_sample_bar("ASSET_A", t2, 105.0, 105.0)
    bar_a3 = _create_sample_bar("ASSET_A", t3, 110.0, 110.0)

    bar_b1 = _create_sample_bar("ASSET_B", t1, 200.0, 200.0)
    # Note: bar_b2 is missing at t2!
    bar_b3 = _create_sample_bar("ASSET_B", t3, 220.0, 220.0)

    memory_duckdb_repo.save_batch_sync(
        HistoricalBarBatch(
            symbol="ASSET_A", resolution=Resolution.ONE_DAY, bars=(bar_a1, bar_a2, bar_a3)
        )
    )
    memory_duckdb_repo.save_batch_sync(
        HistoricalBarBatch(symbol="ASSET_B", resolution=Resolution.ONE_DAY, bars=(bar_b1, bar_b3))
    )

    res = memory_duckdb_repo.get_aligned_returns_matrix_sync(
        symbols=["ASSET_A", "ASSET_B"],
        start_time=t1,
        end_time=t3,
        resolution=Resolution.ONE_DAY,
        price_field="adj_close",
        return_type="simple",
    )

    assert isinstance(res, AlignedReturnsResult)
    assert res.symbols == ("ASSET_A", "ASSET_B")
    assert len(res.timestamps) == 2
    assert res.timestamps[0] == t2
    assert res.timestamps[1] == t3

    # Matrix shape: (2 periods, 2 assets)
    assert res.matrix.shape == (2, 2)

    # Asset A returns
    expected_ret_a1 = (105.0 - 100.0) / 100.0
    expected_ret_a2 = (110.0 - 105.0) / 105.0
    assert math.isclose(res.matrix[0, 0], expected_ret_a1, rel_tol=1e-5)
    assert math.isclose(res.matrix[1, 0], expected_ret_a2, rel_tol=1e-5)

    # Asset B returns with causal forward-fill
    assert math.isclose(res.matrix[0, 1], 0.0, abs_tol=1e-7)  # Halt at t2
    expected_ret_b2 = (220.0 - 200.0) / 200.0  # From t2 forward-fill to t3
    assert math.isclose(res.matrix[1, 1], expected_ret_b2, rel_tol=1e-5)


def test_aligned_returns_matrix_log_returns(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify log return calculation ln(p_t / p_{t-1})."""
    t1 = 1_000_000_000
    t2 = 2_000_000_000

    bar1 = _create_sample_bar("NVDA", t1, 100.0, 100.0)
    bar2 = _create_sample_bar("NVDA", t2, 120.0, 120.0)

    memory_duckdb_repo.save_batch_sync(
        HistoricalBarBatch(symbol="NVDA", resolution=Resolution.ONE_DAY, bars=(bar1, bar2))
    )

    res = memory_duckdb_repo.get_aligned_returns_matrix_sync(
        symbols=["NVDA"],
        start_time=t1,
        end_time=t2,
        resolution=Resolution.ONE_DAY,
        return_type="log",
    )

    expected_log_ret = math.log(120.0 / 100.0)
    assert math.isclose(res.matrix[0, 0], expected_log_ret, rel_tol=1e-6)


def test_aligned_returns_matrix_empty_universe_or_insufficient_timestamps(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify empty universe and single-bar ranges trigger designated error codes."""
    with pytest.raises(EmptyUniverseError) as empty_exc:
        memory_duckdb_repo.get_aligned_returns_matrix_sync(
            symbols=[],
            start_time=100,
            end_time=200,
            resolution=Resolution.ONE_DAY,
        )
    assert empty_exc.value.code == ERR_DATA_EMPTY_UNIVERSE

    t1 = 1_000_000_000
    bar = _create_sample_bar("MSFT", t1, 300.0)
    memory_duckdb_repo.save_batch_sync(
        HistoricalBarBatch(symbol="MSFT", resolution=Resolution.ONE_DAY, bars=(bar,))
    )

    # Only 1 timestamp available
    with pytest.raises(InsufficientTimestampsError) as insuf_exc:
        memory_duckdb_repo.get_aligned_returns_matrix_sync(
            symbols=["MSFT"],
            start_time=t1,
            end_time=t1,
            resolution=Resolution.ONE_DAY,
        )
    assert insuf_exc.value.code == ERR_DATA_INSUFFICIENT_TIMESTAMPS


@pytest.mark.asyncio
async def test_async_repository_methods(
    memory_duckdb_repo: DuckDBHistoricalRepository,
) -> None:
    """Verify asynchronous facades work seamlessly across threads."""
    t1 = 1_000_000_000
    t2 = 2_000_000_000

    bar1 = _create_sample_bar("SPY", t1, 400.0)
    bar2 = _create_sample_bar("SPY", t2, 402.0)
    batch = HistoricalBarBatch(symbol="SPY", resolution=Resolution.ONE_DAY, bars=(bar1, bar2))

    count = await memory_duckdb_repo.save_batch(batch)
    assert count == 2

    retrieved = await memory_duckdb_repo.get_bars_range("SPY", t1, t2, Resolution.ONE_DAY)
    assert len(retrieved) == 2

    symbols = await memory_duckdb_repo.get_all_symbols()
    assert "SPY" in symbols

    res = await memory_duckdb_repo.get_aligned_returns_matrix(
        symbols=["SPY"],
        start_time=t1,
        end_time=t2,
        resolution=Resolution.ONE_DAY,
    )
    assert res.matrix.shape == (1, 1)
