"""Unit tests for external quantitative data providers and provider manager.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Fault Codes: ERR-EXT-001 through ERR-EXT-006
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from quant.data.external_providers import (
    ERR_PROVIDER_INVALID_KEY,
    ERR_PROVIDER_NON_FINITE_VALUE,
    ERR_PROVIDER_RATE_LIMIT,
    ExternalNewsItem,
    ExternalProviderManager,
    FinnhubClient,
    FredClient,
    MacroIndicatorRecord,
    NewsApiClient,
    PolygonClient,
    ProviderInvalidKeyException,
    ProviderNonFiniteValueException,
    ProviderRateLimitException,
)


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestExternalProviderModels:
    """Verify domain invariant validation and numeric integrity on provider models."""

    def test_macro_indicator_record_success(self) -> None:
        rec = MacroIndicatorRecord(series_id="T10Y2Y", value=0.22, date="2026-09-18", source="FRED")
        assert rec.series_id == "T10Y2Y"
        assert rec.value == 0.22
        assert rec.source == "FRED"

    def test_macro_indicator_record_rejects_nan_and_inf(self) -> None:
        with pytest.raises(ProviderNonFiniteValueException) as exc_info:
            MacroIndicatorRecord(
                series_id="T10Y2Y", value=float("nan"), date="2026-09-18", source="FRED"
            )
        assert exc_info.value.code == ERR_PROVIDER_NON_FINITE_VALUE

        with pytest.raises(ProviderNonFiniteValueException) as exc_info:
            MacroIndicatorRecord(
                series_id="DFF", value=float("inf"), date="2026-09-18", source="FRED"
            )
        assert exc_info.value.code == ERR_PROVIDER_NON_FINITE_VALUE

    def test_macro_indicator_record_rejects_boolean(self) -> None:
        with pytest.raises(ProviderNonFiniteValueException):
            MacroIndicatorRecord(series_id="DFF", value=True, date="2026-09-18", source="FRED")  # type: ignore[arg-type]

    def test_external_news_item_stripping(self) -> None:
        item = ExternalNewsItem(
            headline="  Strong GDP Growth Reported  ",
            summary="  US GDP grew at an annualized rate of 2.8%. ",
            url=" https://example.com/news/1 ",
            source=" Bloomberg ",
            published_at="2026-09-18T12:00:00Z",
            related_symbols=["aapl", " msft "],
        )
        assert item.headline == "Strong GDP Growth Reported"
        assert item.summary == "US GDP grew at an annualized rate of 2.8%."
        assert item.url == "https://example.com/news/1"
        assert item.source == "Bloomberg"
        assert item.related_symbols == ["AAPL", "MSFT"]


class TestFredClient:
    """Verify FRED macroeconomic observation retrieval, synthetic fallback, and error handling."""

    @pytest.mark.asyncio
    async def test_unconfigured_fred_fallback(self) -> None:
        client = FredClient(api_key=None)
        assert not client.is_configured

        obs = await client.fetch_series_observations("T10Y2Y")
        assert len(obs) == 1
        assert obs[0].series_id == "T10Y2Y"
        assert obs[0].value == 0.18
        assert obs[0].source == "FRED_FALLBACK"

        snapshot = await client.get_macro_snapshot()
        assert snapshot["yield_spread_10y_2y"] == 0.18
        assert snapshot["fed_funds_rate"] == 5.25

    @pytest.mark.asyncio
    async def test_configured_fred_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "series/observations" in request.url.path
            assert request.url.params["api_key"] == "valid_fred_key"
            payload = {
                "observations": [
                    {"date": "2026-09-18", "value": "0.25"},
                    {"date": "2026-09-17", "value": "0.21"},
                ]
            }
            return httpx.Response(200, json=payload)

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        fred = FredClient(api_key="valid_fred_key", client=http_client)
        assert fred.is_configured

        obs = await fred.fetch_series_observations("T10Y2Y", limit=2)
        assert len(obs) == 2
        assert obs[0].value == 0.25
        assert obs[1].value == 0.21
        assert obs[0].source == "FRED"
        await http_client.aclose()

    @pytest.mark.asyncio
    async def test_configured_fred_rate_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Rate limit exceeded")

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        fred = FredClient(api_key="valid_fred_key", client=http_client)

        with pytest.raises(ProviderRateLimitException) as exc_info:
            await fred.fetch_series_observations("T10Y2Y")
        assert exc_info.value.code == ERR_PROVIDER_RATE_LIMIT
        await http_client.aclose()

    @pytest.mark.asyncio
    async def test_configured_fred_invalid_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Unauthorized API key")

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        fred = FredClient(api_key="bad_key", client=http_client)

        with pytest.raises(ProviderInvalidKeyException) as exc_info:
            await fred.fetch_series_observations("T10Y2Y")
        assert exc_info.value.code == ERR_PROVIDER_INVALID_KEY
        await http_client.aclose()

    @pytest.mark.asyncio
    async def test_configured_fred_server_error_triggers_fallback(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        fred = FredClient(api_key="valid_key", client=http_client)

        obs = await fred.fetch_series_observations("T10Y2Y")
        assert len(obs) == 1
        assert obs[0].source == "FRED_FALLBACK"
        await http_client.aclose()


class TestFinnhubClient:
    """Verify Finnhub real-time news and quote client."""

    @pytest.mark.asyncio
    async def test_unconfigured_finnhub(self) -> None:
        client = FinnhubClient(api_key=None)
        assert not client.is_configured
        news = await client.fetch_company_news("AAPL")
        assert news == []

        quote = await client.fetch_quote("AAPL")
        assert quote["current_price"] == 100.0

    @pytest.mark.asyncio
    async def test_configured_finnhub_company_news_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["token"] == "fh_token"
            assert request.url.params["symbol"] == "NVDA"
            payload = [
                {
                    "headline": "NVIDIA unveils next-gen Blackwell GPU architecture",
                    "summary": "Record demand reported across cloud service providers.",
                    "url": "https://finnhub.io/news/123",
                    "source": "Finnhub",
                    "datetime": 1726700000,
                    "related": "NVDA",
                }
            ]
            return httpx.Response(200, json=payload)

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        client = FinnhubClient(api_key="fh_token", client=http_client)

        news = await client.fetch_company_news("NVDA")
        assert len(news) == 1
        assert news[0].headline == "NVIDIA unveils next-gen Blackwell GPU architecture"
        assert news[0].related_symbols == ["NVDA"]
        assert "Finnhub" in news[0].source
        await http_client.aclose()

    @pytest.mark.asyncio
    async def test_configured_finnhub_quote_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = {
                "c": 225.50,
                "h": 228.00,
                "l": 224.00,
                "o": 224.50,
                "pc": 223.00,
                "t": 1726700000,
            }
            return httpx.Response(200, json=payload)

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        client = FinnhubClient(api_key="fh_token", client=http_client)

        quote = await client.fetch_quote("AAPL")
        assert quote["current_price"] == 225.50
        assert quote["prev_close"] == 223.00
        await http_client.aclose()


class TestNewsApiClient:
    """Verify NewsAPI.org client for business breaking news."""

    @pytest.mark.asyncio
    async def test_unconfigured_news_api(self) -> None:
        client = NewsApiClient(api_key=None)
        assert not client.is_configured
        items = await client.fetch_top_business_headlines()
        assert items == []

    @pytest.mark.asyncio
    async def test_configured_news_api_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["apiKey"] == "news_key"
            payload = {
                "status": "ok",
                "articles": [
                    {
                        "title": "Fed signals pause on interest rates",
                        "description": "FOMC members vote unanimously to maintain target range.",
                        "url": "https://reuters.com/news/123",
                        "source": {"name": "Reuters"},
                        "publishedAt": "2026-09-18T14:30:00Z",
                    }
                ],
            }
            return httpx.Response(200, json=payload)

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        client = NewsApiClient(api_key="news_key", client=http_client)

        items = await client.fetch_top_business_headlines()
        assert len(items) == 1
        assert items[0].headline == "Fed signals pause on interest rates"
        assert "Reuters" in items[0].source
        await http_client.aclose()


class TestPolygonClient:
    """Verify Polygon.io aggregate market data client."""

    @pytest.mark.asyncio
    async def test_unconfigured_polygon(self) -> None:
        client = PolygonClient(api_key=None)
        assert not client.is_configured
        bar = await client.fetch_previous_close("SPY")
        assert bar is None

    @pytest.mark.asyncio
    async def test_configured_polygon_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["apiKey"] == "poly_key"
            payload = {
                "status": "OK",
                "results": [
                    {"T": "SPY", "c": 560.25, "h": 562.00, "l": 558.10, "o": 559.00, "v": 45000000}
                ],
            }
            return httpx.Response(200, json=payload)

        http_client = httpx.AsyncClient(transport=_mock_transport(handler))
        client = PolygonClient(api_key="poly_key", client=http_client)

        bar = await client.fetch_previous_close("SPY")
        assert bar is not None
        assert bar["close"] == 560.25
        assert bar["volume"] == 45000000
        await http_client.aclose()


class TestExternalProviderManager:
    """Verify master facade coordinating external providers, status telemetry, and news aggregation."""

    @pytest.mark.asyncio
    async def test_provider_manager_unconfigured_initialization(self) -> None:
        manager = ExternalProviderManager()
        statuses = manager.get_provider_statuses()

        assert "FRED" in statuses
        assert "FINNHUB" in statuses
        assert "NEWSAPI" in statuses
        assert "POLYGON" in statuses

        assert not statuses["FRED"]["is_configured"]
        assert statuses["FRED"]["is_healthy"]

    @pytest.mark.asyncio
    async def test_provider_manager_macro_state_sync(self) -> None:
        manager = ExternalProviderManager()
        macro = await manager.get_macro_state()

        assert "yield_spread_10y_2y" in macro
        assert "fed_funds_rate" in macro
        assert macro["yield_spread_10y_2y"] == 0.18
        assert macro["fed_funds_rate"] == 5.25

    @pytest.mark.asyncio
    async def test_provider_manager_aggregated_news(self) -> None:
        manager = ExternalProviderManager()
        news = await manager.fetch_aggregated_news(ticker="AAPL")
        assert isinstance(news, list)
