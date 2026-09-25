"""Unit tests for Yahoo Finance v8 historical market data provider.

Functional Purpose:
    Tests YahooFinanceHistoricalProvider chart JSON parsing, corporate actions extraction (splits, dividends),
    monotonic timestamp ordering (INV-DATA-004), physical price geometry (INV-DATA-005),
    error classification (ERR-DATA-011 through ERR-DATA-014), and exponential backoff retry handling.

Explicit Dependency Tracking:
    - pytest, pytest_asyncio.
    - httpx: Mock transports for deterministic hermetic HTTP testing.
    - quant.data.yahoo_provider: YahooFinanceHistoricalProvider, fault codes, exceptions.
    - quant.domain.historical: CorporateActionType.
    - quant.domain.models: Resolution.
"""

from typing import Any

import httpx
import pytest

from quant.data.yahoo_provider import (
    ERR_DATA_PROVIDER_HTTP_ERROR,
    ERR_DATA_PROVIDER_PARSE_ERROR,
    ERR_DATA_PROVIDER_SYMBOL_NOT_FOUND,
    ERR_DATA_PROVIDER_UNREACHABLE,
    YahooFinanceHistoricalProvider,
    YahooParseError,
    YahooProviderError,
    YahooSymbolNotFoundError,
    YahooUnreachableError,
)
from quant.domain.historical import CorporateActionType, EmptyHistoricalBatchError
from quant.domain.models import Resolution

# ============================================================================
# Synthetic Sample Chart JSON Fixtures
# ============================================================================


def make_valid_chart_json(symbol: str = "SPY") -> dict[str, Any]:
    """Generate a realistic Yahoo Finance chart JSON response with splits and dividends."""
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "symbol": symbol,
                        "exchangeName": "PCX",
                    },
                    "timestamp": [1600000000, 1600086400, 1600172800],
                    "events": {
                        "splits": {
                            "1600086400": {
                                "date": 1600086400,
                                "numerator": 4,
                                "denominator": 1,
                                "splitRatio": "4:1",
                            }
                        },
                        "dividends": {
                            "1600172800": {
                                "date": 1600172800,
                                "amount": 1.50,
                            }
                        },
                    },
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 102.0, 105.0],
                                "high": [104.0, 106.0, 108.0],
                                "low": [99.0, 101.0, 104.0],
                                "close": [103.0, 105.0, 107.0],
                                "volume": [1000000, 1500000, 2000000],
                            }
                        ],
                        "adjclose": [
                            {
                                "adjclose": [98.5, 100.5, 107.0],
                            }
                        ],
                    },
                }
            ],
            "error": None,
        }
    }


# ============================================================================
# Parser Tests
# ============================================================================


def test_parse_chart_response_valid() -> None:
    """Validate successful parsing of complete Yahoo chart payload."""
    provider = YahooFinanceHistoricalProvider()
    raw_data = make_valid_chart_json("SPY")

    batch = provider.parse_chart_response("SPY", Resolution.ONE_DAY, raw_data)

    assert batch.symbol == "SPY"
    assert len(batch) == 3
    assert batch.start_timestamp == 1600000000 * 1_000_000_000
    assert batch.end_timestamp == 1600172800 * 1_000_000_000
    assert batch.closes() == (103.0, 105.0, 107.0)
    assert batch.adj_closes() == (98.5, 100.5, 107.0)
    assert batch.volumes() == (1000000.0, 1500000.0, 2000000.0)

    # Corporate Actions
    assert len(batch.corporate_actions) == 2
    split_act = [a for a in batch.corporate_actions if a.action_type == CorporateActionType.SPLIT][
        0
    ]
    assert split_act.split_ratio == 4.0
    div_act = [
        a for a in batch.corporate_actions if a.action_type == CorporateActionType.CASH_DIVIDEND
    ][0]
    assert div_act.cash_dividend == 1.50


def test_parse_chart_response_filters_null_bars() -> None:
    """Ensure null/None sessions (holidays) are safely omitted without breaking ordering."""
    provider = YahooFinanceHistoricalProvider()
    data = {
        "chart": {
            "result": [
                {
                    "timestamp": [1000, 2000, 3000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, None, 105.0],
                                "high": [104.0, None, 108.0],
                                "low": [99.0, None, 104.0],
                                "close": [103.0, None, 107.0],
                                "volume": [1000, None, 2000],
                            }
                        ],
                        "adjclose": [{"adjclose": [100.0, None, 105.0]}],
                    },
                }
            ],
            "error": None,
        }
    }
    batch = provider.parse_chart_response("AAPL", Resolution.ONE_DAY, data)
    assert len(batch) == 2
    assert batch.timestamps() == (1000 * 1_000_000_000, 3000 * 1_000_000_000)


def test_parse_chart_response_error_scenarios() -> None:
    """Validate specific Yahoo exceptions and error codes."""
    provider = YahooFinanceHistoricalProvider()

    # Symbol Not Found
    err_not_found = {
        "chart": {
            "result": None,
            "error": {
                "code": "Not Found",
                "description": "No data found for this symbol",
            },
        }
    }
    with pytest.raises(YahooSymbolNotFoundError) as exc_info:
        provider.parse_chart_response("BAD_TICKER", Resolution.ONE_DAY, err_not_found)
    assert exc_info.value.code == ERR_DATA_PROVIDER_SYMBOL_NOT_FOUND

    # Generic Yahoo API Error
    err_generic = {
        "chart": {
            "result": None,
            "error": {
                "code": "Service Unavailable",
                "description": "Internal rate limit error",
            },
        }
    }
    with pytest.raises(YahooProviderError) as exc_info:
        provider.parse_chart_response("SPY", Resolution.ONE_DAY, err_generic)
    assert exc_info.value.code == ERR_DATA_PROVIDER_HTTP_ERROR

    # Empty result array
    with pytest.raises(YahooParseError) as exc_info:
        provider.parse_chart_response("SPY", Resolution.ONE_DAY, {"chart": {"result": []}})
    assert exc_info.value.code == ERR_DATA_PROVIDER_PARSE_ERROR

    # All null bars
    all_null = {
        "chart": {
            "result": [
                {
                    "timestamp": [1000],
                    "indicators": {
                        "quote": [{"open": [None], "high": [None], "low": [None], "close": [None]}],
                    },
                }
            ]
        }
    }
    with pytest.raises(EmptyHistoricalBatchError):
        provider.parse_chart_response("SPY", Resolution.ONE_DAY, all_null)


# ============================================================================
# HTTP Transport & Retry Tests (Async & Sync)
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_historical_bars_async_success() -> None:
    """Validate async fetch using mock transport."""
    mock_payload = make_valid_chart_json("QQQ")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = YahooFinanceHistoricalProvider(client=client)
        batch = await provider.fetch_historical_bars_async(
            symbol="QQQ",
            start=1600000000,
            end=1600200000,
            resolution=Resolution.ONE_DAY,
        )
        assert batch.symbol == "QQQ"
        assert len(batch) == 3


@pytest.mark.asyncio
async def test_fetch_historical_bars_async_404_raises_symbol_not_found() -> None:
    """Validate HTTP 404 raises YahooSymbolNotFoundError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = YahooFinanceHistoricalProvider(client=client, max_retries=1)
        with pytest.raises(YahooSymbolNotFoundError) as exc_info:
            await provider.fetch_historical_bars_async(
                symbol="INVALID_TICKER_XYZ",
                start=1600000000,
                end=1600200000,
            )
        assert exc_info.value.code == ERR_DATA_PROVIDER_SYMBOL_NOT_FOUND


@pytest.mark.asyncio
async def test_fetch_historical_bars_async_retry_on_429() -> None:
    """Validate that HTTP 429 triggers backoff retry and succeeds on subsequent 200."""
    calls = 0
    mock_payload = make_valid_chart_json("AAPL")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(200, json=mock_payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = YahooFinanceHistoricalProvider(
            client=client,
            max_retries=2,
            base_backoff_sec=0.01,  # Fast backoff for testing
        )
        batch = await provider.fetch_historical_bars_async(
            symbol="AAPL",
            start=1600000000,
            end=1600200000,
        )
        assert calls == 2
        assert batch.symbol == "AAPL"


def test_fetch_historical_bars_sync_success() -> None:
    """Validate synchronous fetch using sync mock transport."""
    mock_payload = make_valid_chart_json("NVDA")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_payload)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        provider = YahooFinanceHistoricalProvider(sync_client=client)
        batch = provider.fetch_historical_bars(
            symbol="NVDA",
            start=1600000000,
            end=1600200000,
            resolution=Resolution.ONE_DAY,
        )
        assert batch.symbol == "NVDA"
        assert len(batch) == 3


def test_fetch_historical_bars_sync_invalid_time_range() -> None:
    """Ensure end <= start raises ValueError."""
    provider = YahooFinanceHistoricalProvider()
    with pytest.raises(ValueError):
        provider.fetch_historical_bars(
            symbol="SPY",
            start=2000,
            end=1000,
        )


@pytest.mark.asyncio
async def test_fetch_historical_bars_async_network_failure_raises_unreachable() -> None:
    """Ensure network connection failure after all retries raises YahooUnreachableError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network is down", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = YahooFinanceHistoricalProvider(
            client=client,
            max_retries=1,
            base_backoff_sec=0.01,
        )
        with pytest.raises(YahooUnreachableError) as exc_info:
            await provider.fetch_historical_bars_async(
                symbol="MSFT",
                start=1600000000,
                end=1600200000,
            )
        assert exc_info.value.code == ERR_DATA_PROVIDER_UNREACHABLE
