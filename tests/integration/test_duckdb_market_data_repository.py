"""Integration tests for DuckDBMarketDataRepository using in-memory DuckDB instance.

Purpose: Verifies high-throughput columnar storage, range queries, and upsert idempotency.
Dependencies: pytest, DuckDBManager, DuckDBMarketDataRepository, PriceBar, Resolution.
Relationship: Validates the physical persistence adapter for market data.
Invariants: All queries must return ascending chronological ordering and maintain atomic consistency.
"""

import pytest

from quant.domain.models import PriceBar, Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.repositories.duckdb_market_data_repository import (
    DuckDBMarketDataRepository,
)


@pytest.fixture
def memory_duckdb_repo() -> DuckDBMarketDataRepository:
    """Provide an isolated in-memory DuckDB repository fixture."""
    manager = DuckDBManager(":memory:")
    return DuckDBMarketDataRepository(manager)


@pytest.mark.asyncio
async def test_duckdb_repository_empty_batch(
    memory_duckdb_repo: DuckDBMarketDataRepository,
) -> None:
    """Verify empty sequence insertion returns 0 without database errors."""
    persisted = await memory_duckdb_repo.add_bars_batch([])
    assert persisted == 0


@pytest.mark.asyncio
async def test_duckdb_repository_batch_insert_and_range_query(
    memory_duckdb_repo: DuckDBMarketDataRepository,
) -> None:
    """Verify multi-bar insertion and precise chronological range retrieval."""
    bars = [
        PriceBar(
            asset_id="BTC-USDT",
            timestamp=1000 * (i + 1),
            open=50000.0 + i * 10,
            high=50100.0 + i * 10,
            low=49900.0 + i * 10,
            close=50050.0 + i * 10,
            volume=1.5,
            vwap=50020.0 + i * 10,
            resolution=Resolution.ONE_MINUTE,
        )
        for i in range(50)
    ]

    persisted = await memory_duckdb_repo.add_bars_batch(bars)
    assert persisted == 50

    # Query middle sub-range
    batch = await memory_duckdb_repo.get_bars_range(
        asset_id="BTC-USDT",
        start_time=10000,
        end_time=30000,
        resolution=Resolution.ONE_MINUTE,
    )

    # Bars with timestamp in [10000, 30000] are indices 9 to 29 (21 bars)
    assert batch.count == 21
    assert batch.start_time == 10000
    assert batch.end_time == 30000
    # Verify strict ascending timestamp order
    assert all(
        batch.timestamps[j] < batch.timestamps[j + 1] for j in range(len(batch.timestamps) - 1)
    )
    assert batch.closes[0] == 50050.0 + 9 * 10


@pytest.mark.asyncio
async def test_duckdb_repository_upsert_idempotency(
    memory_duckdb_repo: DuckDBMarketDataRepository,
) -> None:
    """Verify duplicate bar timestamp updates the record rather than creating duplicate rows."""
    bar1 = PriceBar(
        asset_id="ETH-USDT",
        timestamp=5000,
        open=3000.0,
        high=3050.0,
        low=2990.0,
        close=3020.0,
        volume=10.0,
        vwap=3015.0,
        resolution=Resolution.FIVE_MINUTES,
    )
    await memory_duckdb_repo.add_bars_batch([bar1])

    # Re-insert identical asset, resolution, and timestamp with updated closing price
    bar2 = PriceBar(
        asset_id="ETH-USDT",
        timestamp=5000,
        open=3000.0,
        high=3080.0,
        low=2990.0,
        close=3075.0,  # Updated close
        volume=25.0,
        vwap=3040.0,
        resolution=Resolution.FIVE_MINUTES,
    )
    await memory_duckdb_repo.add_bars_batch([bar2])

    # Query back
    batch = await memory_duckdb_repo.get_bars_range(
        asset_id="ETH-USDT",
        start_time=1000,
        end_time=10000,
        resolution=Resolution.FIVE_MINUTES,
    )
    assert batch.count == 1
    assert batch.closes[0] == 3075.0
    assert batch.highs[0] == 3080.0
    assert batch.volumes[0] == 25.0


@pytest.mark.asyncio
async def test_duckdb_repository_get_latest_bars(
    memory_duckdb_repo: DuckDBMarketDataRepository,
) -> None:
    """Verify get_latest_bars returns most recent N bars sorted ascending."""
    bars = [
        PriceBar(
            asset_id="SOL-USDT",
            timestamp=i * 100,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=50.0,
            vwap=101.0,
            resolution=Resolution.ONE_MINUTE,
        )
        for i in range(1, 21)
    ]
    await memory_duckdb_repo.add_bars_batch(bars)

    # Request latest 5 bars
    latest = await memory_duckdb_repo.get_latest_bars(
        asset_id="SOL-USDT", count=5, resolution=Resolution.ONE_MINUTE
    )
    assert latest.count == 5
    assert latest.timestamps[0] == 1600
    assert latest.timestamps[-1] == 2000
    assert latest.timestamps[0] < latest.timestamps[-1]


@pytest.mark.asyncio
async def test_duckdb_repository_available_range(
    memory_duckdb_repo: DuckDBMarketDataRepository,
) -> None:
    """Verify available timestamp range discovery returns min and max bounds."""
    # When empty, returns None
    empty_range = await memory_duckdb_repo.get_available_range(
        asset_id="NONEXISTENT", resolution=Resolution.ONE_MINUTE
    )
    assert empty_range is None

    # Insert bars
    bars = [
        PriceBar(
            asset_id="NVDA",
            timestamp=500,
            open=400.0,
            high=410.0,
            low=390.0,
            close=405.0,
            volume=1000.0,
            vwap=402.0,
        ),
        PriceBar(
            asset_id="NVDA",
            timestamp=1500,
            open=405.0,
            high=420.0,
            low=400.0,
            close=418.0,
            volume=1500.0,
            vwap=412.0,
        ),
    ]
    await memory_duckdb_repo.add_bars_batch(bars)

    date_range = await memory_duckdb_repo.get_available_range(
        asset_id="NVDA", resolution=Resolution.ONE_MINUTE
    )
    assert date_range == (500, 1500)
