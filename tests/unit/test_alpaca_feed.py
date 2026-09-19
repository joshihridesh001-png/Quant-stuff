"""Unit tests for AlpacaMarketDataFeed and live DuckDB/FracDiff streaming."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import numpy as np
import pytest

from quant.analytics.fractional_diff import StreamingFracDiffBuffer
from quant.data.alpaca_feed import AlpacaMarketDataFeed
from quant.domain.models import PriceBar, Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.repositories.duckdb_market_data_repository import (
    DuckDBMarketDataRepository,
)


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture
def in_memory_duckdb() -> DuckDBManager:
    return DuckDBManager(database_path=":memory:")


@pytest.fixture
def market_repo(in_memory_duckdb: DuckDBManager) -> DuckDBMarketDataRepository:
    return DuckDBMarketDataRepository(in_memory_duckdb)


@pytest.fixture
def frac_buffers() -> dict[str, StreamingFracDiffBuffer]:
    weights = np.array([1.0, -0.4, -0.12], dtype=np.float64)
    buf_aapl = StreamingFracDiffBuffer(weights=weights)
    buf_aapl.hydrate([180.0, 180.2])
    buf_nvda = StreamingFracDiffBuffer(weights=weights)
    buf_nvda.hydrate([125.0, 125.2])
    buf_msft = StreamingFracDiffBuffer(weights=weights)
    buf_msft.hydrate([420.0, 420.2])
    return {"AAPL": buf_aapl, "NVDA": buf_nvda, "MSFT": buf_msft}


class TestAlpacaMarketDataFeed:
    """Verify live bar fetching, parsing, and repository/buffer updates."""

    @pytest.mark.asyncio
    async def test_fetch_latest_bars_success(
        self,
        market_repo: DuckDBMarketDataRepository,
        frac_buffers: dict[str, StreamingFracDiffBuffer],
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["APCA-API-KEY-ID"] == "test-key"
            assert request.headers["APCA-API-SECRET-KEY"] == "test-secret"
            if "/v2/stocks/bars/latest" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "bars": {
                            "AAPL": {
                                "t": "2026-09-19T15:30:00Z",
                                "o": 180.50,
                                "h": 181.20,
                                "l": 180.10,
                                "c": 180.90,
                                "v": 25000,
                                "vw": 180.70,
                            },
                            "NVDA": {
                                "t": "2026-09-19T15:30:00Z",
                                "o": 125.00,
                                "h": 126.50,
                                "l": 124.80,
                                "c": 126.10,
                                "v": 50000,
                                "vw": 125.80,
                            },
                        }
                    },
                )
            return httpx.Response(404)

        transport = _mock_transport(handler)
        feed = AlpacaMarketDataFeed(
            api_key="test-key",
            secret_key="test-secret",
            data_url="https://data.alpaca.markets",
            repository=market_repo,
            frac_diff_buffers=frac_buffers,
            transport=transport,
        )

        bars = await feed.fetch_latest_bars(["AAPL", "NVDA"])
        assert len(bars) == 2
        assert "AAPL" in bars
        assert "NVDA" in bars

        aapl_bar = bars["AAPL"]
        assert isinstance(aapl_bar, PriceBar)
        assert aapl_bar.open == 180.50
        assert aapl_bar.high == 181.20
        assert aapl_bar.low == 180.10
        assert aapl_bar.close == 180.90
        assert aapl_bar.volume == 25000.0

        # Verify repository was updated
        history = await market_repo.get_latest_bars(
            "AAPL", count=10, resolution=Resolution.ONE_MINUTE
        )
        assert history.count == 1
        assert history.closes[0] == 180.90

        # Verify FracDiff buffer ingested latest close
        assert frac_buffers["AAPL"].is_hydrated
        assert frac_buffers["AAPL"]._buffer[-1] == 180.90

    @pytest.mark.asyncio
    async def test_offline_synthetic_fallback(
        self,
        market_repo: DuckDBMarketDataRepository,
        frac_buffers: dict[str, StreamingFracDiffBuffer],
    ) -> None:
        feed = AlpacaMarketDataFeed(
            api_key="",
            secret_key="",
            repository=market_repo,
            frac_diff_buffers=frac_buffers,
            offline_mode=True,
        )

        bars = await feed.fetch_latest_bars(["AAPL", "MSFT"])
        assert len(bars) == 2
        assert "AAPL" in bars
        assert "MSFT" in bars
        assert bars["AAPL"].close > 0.0
        assert bars["MSFT"].close > 0.0

        # Verify repository and buffer updated even in offline fallback
        history = await market_repo.get_latest_bars(
            "AAPL", count=5, resolution=Resolution.ONE_MINUTE
        )
        assert history.count == 1
        assert frac_buffers["AAPL"].is_hydrated
        assert frac_buffers["AAPL"]._buffer[-1] == bars["AAPL"].close

        await feed.aclose()
