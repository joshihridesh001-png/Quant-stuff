"""Unit tests for institutional historical providers (Polygon.io, Alpaca v2) and HistoricalDataHub.

Functional Purpose:
    Tests cursor pagination, RFC 3339 parsing, authentication error trapping,
    and zero-failure fallback cascades (INV-DATA-008) to Yahoo Finance.

Explicit Dependency Tracking:
    - pytest, pytest_asyncio.
    - httpx: Mock transports for deterministic hermetic testing.
    - quant.data.institutional_providers: PolygonHistoricalProvider, AlpacaHistoricalProvider,
      HistoricalDataHub, and custom exception taxonomy.
    - quant.domain.historical: HistoricalBarBatch, Resolution.
"""

from typing import Any

import httpx
import pytest

from quant.data.institutional_providers import (
    AlpacaHistoricalError,
    AlpacaHistoricalProvider,
    HistoricalDataHub,
    PolygonHistoricalProvider,
    PolygonProviderError,
)
from quant.data.yahoo_provider import YahooFinanceHistoricalProvider
from quant.domain.models import Resolution

# ============================================================================
# Polygon.io Provider Tests
# ============================================================================


def test_polygon_is_configured() -> None:
    """Ensure is_configured accurately reflects API key state."""
    p_unconf = PolygonHistoricalProvider(api_key="")
    assert not p_unconf.is_configured

    p_conf = PolygonHistoricalProvider(api_key="valid_polygon_key")
    assert p_conf.is_configured


def test_polygon_parse_results_valid() -> None:
    """Validate parsing of Polygon aggregate results."""
    provider = PolygonHistoricalProvider(api_key="test_key")
    raw_results: list[dict[str, Any]] = [
        {
            "t": 1600000000000,
            "o": 100.0,
            "h": 105.0,
            "l": 99.0,
            "c": 103.0,
            "v": 10000,
            "vw": 102.5,
        },
        {
            "t": 1600086400000,
            "o": 103.0,
            "h": 107.0,
            "l": 102.0,
            "c": 106.0,
            "v": 12000,
            "vw": 105.0,
        },
    ]

    batch = provider.parse_polygon_results("AAPL", Resolution.ONE_DAY, raw_results)
    assert batch.symbol == "AAPL"
    assert len(batch) == 2
    assert batch.closes() == (103.0, 106.0)
    assert batch.volumes() == (10000.0, 12000.0)


@pytest.mark.asyncio
async def test_polygon_fetch_async_unconfigured_raises() -> None:
    """Ensure fetching from unconfigured Polygon provider raises PolygonProviderError."""
    provider = PolygonHistoricalProvider(api_key="")
    with pytest.raises(PolygonProviderError):
        await provider.fetch_historical_bars_async("AAPL", 1600000000, 1600100000)


@pytest.mark.asyncio
async def test_polygon_fetch_async_pagination() -> None:
    """Validate cursor pagination following next_url across 2 pages."""
    page_1 = {
        "ticker": "AAPL",
        "queryCount": 1,
        "resultsCount": 1,
        "results": [{"t": 1600000000000, "o": 100.0, "h": 105.0, "l": 99.0, "c": 103.0, "v": 1000}],
        "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/next_cursor_page",
    }
    page_2 = {
        "ticker": "AAPL",
        "queryCount": 1,
        "resultsCount": 1,
        "results": [
            {"t": 1600086400000, "o": 103.0, "h": 107.0, "l": 102.0, "c": 106.0, "v": 1200}
        ],
        "next_url": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "next_cursor_page" in str(request.url):
            return httpx.Response(200, json=page_2)
        return httpx.Response(200, json=page_1)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = PolygonHistoricalProvider(api_key="mock_key", client=client)
        batch = await provider.fetch_historical_bars_async("AAPL", 1600000000, 1600100000)
        assert len(batch) == 2
        assert batch.closes() == (103.0, 106.0)


# ============================================================================
# Alpaca Market Data v2 Provider Tests
# ============================================================================


def test_alpaca_is_configured() -> None:
    """Ensure is_configured checks both key and secret."""
    a_none = AlpacaHistoricalProvider(api_key="", secret_key="")
    assert not a_none.is_configured

    a_partial = AlpacaHistoricalProvider(api_key="key", secret_key="")
    assert not a_partial.is_configured

    a_full = AlpacaHistoricalProvider(api_key="key", secret_key="secret")
    assert a_full.is_configured


def test_alpaca_parse_bars_valid() -> None:
    """Validate RFC 3339 timestamp parsing for Alpaca bars."""
    provider = AlpacaHistoricalProvider(api_key="key", secret_key="secret")
    raw_bars = [
        {
            "t": "2024-01-02T05:00:00Z",
            "o": 180.0,
            "h": 182.0,
            "l": 179.0,
            "c": 181.5,
            "v": 50000,
            "vw": 181.0,
        },
        {
            "t": "2024-01-03T05:00:00Z",
            "o": 181.5,
            "h": 184.0,
            "l": 181.0,
            "c": 183.0,
            "v": 60000,
            "vw": 182.8,
        },
    ]
    batch = provider.parse_alpaca_bars("MSFT", Resolution.ONE_DAY, raw_bars)
    assert batch.symbol == "MSFT"
    assert len(batch) == 2
    assert batch.closes() == (181.5, 183.0)


@pytest.mark.asyncio
async def test_alpaca_fetch_async_unconfigured_raises() -> None:
    """Ensure fetching from unconfigured Alpaca provider raises AlpacaHistoricalError."""
    provider = AlpacaHistoricalProvider(api_key="", secret_key="")
    with pytest.raises(AlpacaHistoricalError):
        await provider.fetch_historical_bars_async("MSFT", 1600000000, 1600100000)


@pytest.mark.asyncio
async def test_alpaca_fetch_async_pagination() -> None:
    """Validate page_token pagination across multiple requests."""
    page_1 = {
        "bars": {
            "NVDA": [
                {
                    "t": "2024-01-02T05:00:00Z",
                    "o": 480.0,
                    "h": 490.0,
                    "l": 475.0,
                    "c": 485.0,
                    "v": 1000,
                }
            ]
        },
        "next_page_token": "token_abc123",
    }
    page_2 = {
        "bars": {
            "NVDA": [
                {
                    "t": "2024-01-03T05:00:00Z",
                    "o": 485.0,
                    "h": 495.0,
                    "l": 480.0,
                    "c": 492.0,
                    "v": 1200,
                }
            ]
        },
        "next_page_token": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "page_token=token_abc123" in str(request.url):
            return httpx.Response(200, json=page_2)
        return httpx.Response(200, json=page_1)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = AlpacaHistoricalProvider(api_key="k", secret_key="s", client=client)
        batch = await provider.fetch_historical_bars_async("NVDA", 1600000000, 1600100000)
        assert len(batch) == 2
        assert batch.closes() == (485.0, 492.0)


# ============================================================================
# HistoricalDataHub Fallback Tests (INV-DATA-008)
# ============================================================================


@pytest.mark.asyncio
async def test_historical_data_hub_fallback_cascade_to_yahoo() -> None:
    """Validate that when Polygon and Alpaca fail or are unconfigured, hub cascades to Yahoo."""
    # Polygon throws error
    poly = PolygonHistoricalProvider(api_key="invalid_key")
    # Alpaca is unconfigured
    alpaca = AlpacaHistoricalProvider(api_key="", secret_key="")

    # Mock Yahoo with valid chart JSON
    yahoo_payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1600000000, 1600086400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [200.0, 205.0],
                                "high": [206.0, 210.0],
                                "low": [198.0, 203.0],
                                "close": [204.0, 208.0],
                                "volume": [5000, 6000],
                            }
                        ],
                        "adjclose": [{"adjclose": [204.0, 208.0]}],
                    },
                }
            ],
            "error": None,
        }
    }

    def yahoo_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=yahoo_payload)

    yahoo_client = httpx.AsyncClient(transport=httpx.MockTransport(yahoo_handler))
    yahoo = YahooFinanceHistoricalProvider(client=yahoo_client)

    hub = HistoricalDataHub(
        polygon_provider=poly,
        alpaca_provider=alpaca,
        yahoo_provider=yahoo,
        preferred_provider="auto",
    )

    batch = await hub.fetch_bars_async("SPY", 1600000000, 1600100000)
    assert batch.symbol == "SPY"
    assert len(batch) == 2
    assert batch.closes() == (204.0, 208.0)

    await yahoo_client.aclose()


def test_historical_data_hub_sync_fetch_delegation() -> None:
    """Validate synchronous fetch_bars wrapper delegates cleanly."""
    yahoo_payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1600000000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0],
                                "high": [105.0],
                                "low": [98.0],
                                "close": [102.0],
                                "volume": [1000],
                            }
                        ],
                    },
                }
            ],
            "error": None,
        }
    }

    def yahoo_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=yahoo_payload)

    sync_client = httpx.Client(transport=httpx.MockTransport(yahoo_handler))
    yahoo = YahooFinanceHistoricalProvider(sync_client=sync_client)
    hub = HistoricalDataHub(yahoo_provider=yahoo, preferred_provider="yahoo")

    batch = hub.fetch_bars("QQQ", 1600000000, 1600100000)
    assert batch.symbol == "QQQ"
    assert len(batch) == 1
    assert batch.closes() == (102.0,)
