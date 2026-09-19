"""External quantitative financial, macroeconomic, and news API provider clients.

Integrates high-signal public APIs selected from the public-apis directory:
- FRED (Federal Reserve Economic Data: Yield spreads, Fed Funds Rate, CPI)
- Finnhub (Real-time stock quotes, institutional company news, earnings calendar)
- NewsAPI.org (Global breaking business and macroeconomic news)
- Polygon.io (Institutional US equities aggregate market data)

Governing Standards:
- Rules.md (Rule 1: Line annotations; Rule 2: Diagnostic error codes; Rule 3: Quality gates)
- Fault Codes: ERR-EXT-001 through ERR-EXT-006
- Safe Fallback: Hermetic offline mock fallback when API keys are unconfigured.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from quant.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Diagnostic Fault Codes (Rules.md Rule 2)
ERR_PROVIDER_UNREACHABLE = "ERR-EXT-001"
ERR_PROVIDER_RATE_LIMIT = "ERR-EXT-002"
ERR_PROVIDER_INVALID_KEY = "ERR-EXT-003"
ERR_PROVIDER_MALFORMED_DATA = "ERR-EXT-004"
ERR_PROVIDER_NON_FINITE_VALUE = "ERR-EXT-005"
ERR_PROVIDER_CIRCUIT_TRIP = "ERR-EXT-006"


class ExternalProviderError(Exception):
    """Base exception for all external provider integration anomalies."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class ProviderUnreachableException(ExternalProviderError):
    """Raised when external provider network request fails or times out (ERR-EXT-001)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_PROVIDER_UNREACHABLE, message)


class ProviderRateLimitException(ExternalProviderError):
    """Raised when external provider returns HTTP 429 Too Many Requests (ERR-EXT-002)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_PROVIDER_RATE_LIMIT, message)


class ProviderInvalidKeyException(ExternalProviderError):
    """Raised when external provider rejects credentials with HTTP 401/403 (ERR-EXT-003)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_PROVIDER_INVALID_KEY, message)


class ProviderMalformedDataException(ExternalProviderError):
    """Raised when external provider returns unexpected schema or missing fields (ERR-EXT-004)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_PROVIDER_MALFORMED_DATA, message)


class ProviderNonFiniteValueException(ExternalProviderError):
    """Raised when external provider emits NaN, Inf, or non-finite numeric scalars (ERR-EXT-005)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_PROVIDER_NON_FINITE_VALUE, message)


class ProviderCircuitTrippedException(ExternalProviderError):
    """Raised when external provider is quarantined due to consecutive failures (ERR-EXT-006)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_PROVIDER_CIRCUIT_TRIP, message)


@dataclass(slots=True, frozen=True)
class MacroIndicatorRecord:
    """Immutable macroeconomic observation record from official statistical sources."""

    series_id: str
    value: float
    date: str
    source: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        # Functional Purpose: Validate numeric integrity of macroeconomic observations.
        # Explicit Dependency Tracking: Python math module.
        # Structural Relationship: Core domain observation emitted by FredClient.
        # Defensive Invariant: Value must be finite float; rejects bool and NaN.
        if isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ProviderNonFiniteValueException(
                f"Macro indicator value for {self.series_id} must be a finite float, got {self.value}"
            )


@dataclass(slots=True, frozen=True)
class ExternalNewsItem:
    """Standardized news article ingested from external financial news APIs."""

    headline: str
    summary: str
    url: str
    source: str
    published_at: str
    related_symbols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headline", self.headline.strip())
        object.__setattr__(self, "summary", self.summary.strip())
        object.__setattr__(self, "url", self.url.strip())
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(
            self, "related_symbols", [s.strip().upper() for s in self.related_symbols if s.strip()]
        )


@dataclass(slots=True)
class ProviderStatus:
    """Runtime telemetry and health state for an external quantitative provider."""

    provider_id: str
    is_configured: bool
    is_healthy: bool
    calls_made: int = 0
    last_call_timestamp_ns: int | None = None
    last_error: str | None = None


# ============================================================================
# 1. FRED Client (Federal Reserve Economic Data)
# ============================================================================
class FredClient:
    """Asynchronous client for St. Louis Federal Reserve Economic Data (FRED)."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        # Functional Purpose: Initialize FRED macro client with credentials and connection pool.
        # Explicit Dependency Tracking: httpx.AsyncClient, Settings.
        # Structural Relationship: Supplies macro regime signals to CausalBayesianRegimeFilter.
        # Defensive Invariant: Operates in offline synthetic fallback mode if api_key is empty.
        self._api_key = (api_key or "").strip()
        self._client = client
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Return True if a non-empty API key is provisioned."""
        return bool(self._api_key)

    async def fetch_series_observations(
        self, series_id: str, limit: int = 5
    ) -> list[MacroIndicatorRecord]:
        """Fetch historical observations for a FRED macroeconomic series (e.g. T10Y2Y, DFF).

        Args:
            series_id: FRED economic data series symbol (e.g., 'T10Y2Y', 'DFF', 'CPIAUCSL').
            limit: Maximum count of recent observations to retrieve.

        Returns:
            List of MacroIndicatorRecord objects ordered chronologically.
        """
        if not self.is_configured:
            # Safe hermetic fallback: return synthetic observation calibrated to nominal historical regimes
            return self._synthetic_fallback(series_id)

        url = f"{self.BASE_URL}/series/observations"
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }

        try:
            if self._client is not None:
                resp = await self._client.get(url, params=params, timeout=self._timeout)
            else:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(url, params=params, timeout=self._timeout)

            if resp.status_code == 429:
                raise ProviderRateLimitException(f"FRED rate limit exceeded for {series_id}")
            if resp.status_code in (401, 403):
                raise ProviderInvalidKeyException("FRED authentication rejected for key")
            if resp.status_code != 200:
                raise ProviderUnreachableException(
                    f"FRED returned HTTP {resp.status_code}: {resp.text[:100]}"
                )

            data = resp.json()
            raw_obs = data.get("observations", [])
            records: list[MacroIndicatorRecord] = []

            for item in raw_obs:
                raw_val = item.get("value", "")
                if raw_val in (".", "", None):
                    continue
                try:
                    val = float(raw_val)
                    if math.isfinite(val):
                        records.append(
                            MacroIndicatorRecord(
                                series_id=series_id,
                                value=val,
                                date=item.get("date", ""),
                                source="FRED",
                            )
                        )
                except (ValueError, TypeError):
                    continue

            return records or self._synthetic_fallback(series_id)

        except (ProviderRateLimitException, ProviderInvalidKeyException):
            raise
        except Exception as exc:
            logger.warning(
                "FRED network request failed for %s (%s). Using fallback.", series_id, exc
            )
            return self._synthetic_fallback(series_id)

    async def get_macro_snapshot(self) -> dict[str, float]:
        """Retrieve key macroeconomic inputs for Bayesian temperature and regime priors.

        Returns:
            Dictionary containing 'yield_spread_10y_2y' and 'fed_funds_rate'.
        """
        spread_records = await self.fetch_series_observations("T10Y2Y", limit=1)
        dff_records = await self.fetch_series_observations("DFF", limit=1)

        spread = spread_records[0].value if spread_records else 0.15
        dff = dff_records[0].value if dff_records else 5.25

        return {
            "yield_spread_10y_2y": spread,
            "fed_funds_rate": dff,
        }

    def _synthetic_fallback(self, series_id: str) -> list[MacroIndicatorRecord]:
        """Generate deterministic fallback records for offline testing."""
        defaults = {
            "T10Y2Y": 0.18,  # Slightly upward sloping yield curve (normal regime)
            "DFF": 5.25,  # Current target federal funds benchmark rate
            "CPIAUCSL": 314.5,
        }
        val = defaults.get(series_id, 1.0)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return [
            MacroIndicatorRecord(
                series_id=series_id,
                value=val,
                date=today,
                source="FRED_FALLBACK",
            )
        ]


# ============================================================================
# 2. Finnhub Client (Company News & Earnings Events)
# ============================================================================
class FinnhubClient:
    """Asynchronous client for Finnhub institutional market data and company news."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        # Functional Purpose: Initialize Finnhub client for company news and stock quotes.
        # Explicit Dependency Tracking: httpx.AsyncClient, Settings.
        # Structural Relationship: Feeds real-time ticker events into NewsHarvester.
        # Defensive Invariant: Operates in offline fallback mode if api_key is empty.
        self._api_key = (api_key or "").strip()
        self._client = client
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is provisioned."""
        return bool(self._api_key)

    async def fetch_company_news(
        self, symbol: str, lookback_days: int = 2
    ) -> list[ExternalNewsItem]:
        """Fetch institutional news articles for a specific stock ticker symbol.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL', 'NVDA', 'SPY').
            lookback_days: Days back to query.

        Returns:
            List of standardized ExternalNewsItem records.
        """
        if not self.is_configured:
            return []

        now = datetime.now(UTC)
        from_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/company-news"
        params = {
            "symbol": symbol.upper(),
            "from": from_date,
            "to": to_date,
            "token": self._api_key,
        }

        try:
            if self._client is not None:
                resp = await self._client.get(url, params=params, timeout=self._timeout)
            else:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(url, params=params, timeout=self._timeout)

            if resp.status_code == 429:
                raise ProviderRateLimitException(f"Finnhub rate limit exceeded for {symbol}")
            if resp.status_code in (401, 403):
                raise ProviderInvalidKeyException("Finnhub authentication rejected")
            if resp.status_code != 200:
                raise ProviderUnreachableException(f"Finnhub returned HTTP {resp.status_code}")

            articles = resp.json()
            if not isinstance(articles, list):
                return []

            items: list[ExternalNewsItem] = []
            for art in articles[:10]:
                headline = art.get("headline", "").strip()
                if not headline:
                    continue
                pub_ts = art.get("datetime", 0)
                pub_dt = (
                    datetime.fromtimestamp(pub_ts, tz=UTC).isoformat()
                    if pub_ts
                    else datetime.now(UTC).isoformat()
                )
                items.append(
                    ExternalNewsItem(
                        headline=headline,
                        summary=art.get("summary", ""),
                        url=art.get("url", ""),
                        source=f"Finnhub:{art.get('source', 'Market')}",
                        published_at=pub_dt,
                        related_symbols=[symbol.upper()],
                    )
                )
            return items

        except (ProviderRateLimitException, ProviderInvalidKeyException):
            raise
        except Exception as exc:
            logger.warning("Finnhub news request failed for %s: %s", symbol, exc)
            return []

    async def fetch_quote(self, symbol: str) -> dict[str, float]:
        """Fetch current market quote for a symbol.

        Returns:
            Dictionary containing 'current_price', 'high', 'low', 'open', 'prev_close'.
        """
        if not self.is_configured:
            return {"current_price": 100.0, "prev_close": 100.0}

        url = f"{self.BASE_URL}/quote"
        params = {"symbol": symbol.upper(), "token": self._api_key}

        try:
            if self._client is not None:
                resp = await self._client.get(url, params=params, timeout=self._timeout)
            else:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(url, params=params, timeout=self._timeout)

            if resp.status_code == 200:
                data = resp.json()
                price = float(data.get("c", 0.0))
                prev = float(data.get("pc", price))
                if price > 0.0 and math.isfinite(price):
                    return {"current_price": price, "prev_close": prev}

            return {"current_price": 100.0, "prev_close": 100.0}
        except Exception as exc:
            logger.warning("Finnhub quote fetch failed for %s: %s", symbol, exc)
            return {"current_price": 100.0, "prev_close": 100.0}


# ============================================================================
# 3. NewsAPI.org Client (Breaking Financial & Business News)
# ============================================================================
class NewsApiClient:
    """Asynchronous client for NewsAPI.org global business news aggregation."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        # Functional Purpose: Ingest global financial headlines from NewsAPI.
        # Explicit Dependency Tracking: httpx.AsyncClient, Settings.
        # Structural Relationship: Ingests macro and corporate news into NewsHarvester.
        # Defensive Invariant: Operates in offline mode if api_key is empty.
        self._api_key = (api_key or "").strip()
        self._client = client
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is provisioned."""
        return bool(self._api_key)

    async def fetch_top_business_headlines(
        self, query: str = "stocks OR Fed OR inflation", page_size: int = 10
    ) -> list[ExternalNewsItem]:
        """Fetch top breaking business and financial headlines.

        Args:
            query: Keyword query filter.
            page_size: Maximum number of articles to retrieve.

        Returns:
            List of standardized ExternalNewsItem records.
        """
        if not self.is_configured:
            return []

        url = f"{self.BASE_URL}/top-headlines"
        params: dict[str, str | int] = {
            "category": "business",
            "language": "en",
            "q": query,
            "pageSize": min(page_size, 50),
            "apiKey": self._api_key,
        }

        try:
            if self._client is not None:
                resp = await self._client.get(url, params=params, timeout=self._timeout)
            else:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(url, params=params, timeout=self._timeout)

            if resp.status_code == 429:
                raise ProviderRateLimitException("NewsAPI rate limit exceeded")
            if resp.status_code in (401, 403):
                raise ProviderInvalidKeyException("NewsAPI authentication rejected")
            if resp.status_code != 200:
                raise ProviderUnreachableException(f"NewsAPI returned HTTP {resp.status_code}")

            data = resp.json()
            articles = data.get("articles", [])
            items: list[ExternalNewsItem] = []

            for art in articles:
                title = art.get("title", "").strip()
                if not title or title == "[Removed]":
                    continue
                source_name = art.get("source", {}).get("name", "NewsAPI")
                items.append(
                    ExternalNewsItem(
                        headline=title,
                        summary=art.get("description", "") or "",
                        url=art.get("url", ""),
                        source=f"NewsAPI:{source_name}",
                        published_at=art.get("publishedAt", datetime.now(UTC).isoformat()),
                        related_symbols=[],
                    )
                )
            return items

        except (ProviderRateLimitException, ProviderInvalidKeyException):
            raise
        except Exception as exc:
            logger.warning("NewsAPI request failed: %s", exc)
            return []


# ============================================================================
# 4. Polygon.io Client (Institutional Market Data & Aggregates)
# ============================================================================
class PolygonClient:
    """Asynchronous client for Polygon.io market data aggregates."""

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        # Functional Purpose: Provide reference pricing and previous day bar aggregates.
        # Explicit Dependency Tracking: httpx.AsyncClient, Settings.
        # Structural Relationship: Auxiliary pricing feed for DuckDB and SOR.
        # Defensive Invariant: Returns None in offline mode if api_key is empty.
        self._api_key = (api_key or "").strip()
        self._client = client
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is provisioned."""
        return bool(self._api_key)

    async def fetch_previous_close(self, ticker: str) -> dict[str, float] | None:
        """Fetch the previous day's aggregate bar for a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL').

        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume', or None if unavailable.
        """
        if not self.is_configured:
            return None

        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker.upper()}/prev"
        params = {"apiKey": self._api_key}

        try:
            if self._client is not None:
                resp = await self._client.get(url, params=params, timeout=self._timeout)
            else:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(url, params=params, timeout=self._timeout)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    bar = results[0]
                    return {
                        "open": float(bar.get("o", 0.0)),
                        "high": float(bar.get("h", 0.0)),
                        "low": float(bar.get("l", 0.0)),
                        "close": float(bar.get("c", 0.0)),
                        "volume": float(bar.get("v", 0.0)),
                    }
            return None
        except Exception as exc:
            logger.warning("Polygon.io previous close fetch failed for %s: %s", ticker, exc)
            return None


# ============================================================================
# 5. Master External Provider Manager Facade
# ============================================================================
class ExternalProviderManager:
    """Unified master coordinator managing all external quantitative data providers."""

    __slots__ = (
        "_finnhub",
        "_fred",
        "_newsapi",
        "_polygon",
        "_settings",
        "_statuses",
    )

    def __init__(self, settings: Settings | None = None) -> None:
        # Functional Purpose: Orchestrate external providers, track metrics, and manage fallbacks.
        # Explicit Dependency Tracking: Settings, FredClient, FinnhubClient, NewsApiClient, PolygonClient.
        # Structural Relationship: Injected into dependencies.py and services.
        # Defensive Invariant: Initializes all clients with safe zero-secret defaults.
        self._settings = settings or get_settings()

        self._fred = FredClient(api_key=self._settings.FRED_API_KEY)
        self._finnhub = FinnhubClient(api_key=self._settings.FINNHUB_API_KEY)
        self._newsapi = NewsApiClient(api_key=self._settings.NEWS_API_KEY)
        self._polygon = PolygonClient(api_key=self._settings.POLYGON_API_KEY)

        self._statuses: dict[str, ProviderStatus] = {
            "FRED": ProviderStatus(
                provider_id="FRED",
                is_configured=self._fred.is_configured,
                is_healthy=True,
            ),
            "FINNHUB": ProviderStatus(
                provider_id="FINNHUB",
                is_configured=self._finnhub.is_configured,
                is_healthy=True,
            ),
            "NEWSAPI": ProviderStatus(
                provider_id="NEWSAPI",
                is_configured=self._newsapi.is_configured,
                is_healthy=True,
            ),
            "POLYGON": ProviderStatus(
                provider_id="POLYGON",
                is_configured=self._polygon.is_configured,
                is_healthy=True,
            ),
        }

    @property
    def fred(self) -> FredClient:
        """Access the FRED economic data client."""
        return self._fred

    @property
    def finnhub(self) -> FinnhubClient:
        """Access the Finnhub financial client."""
        return self._finnhub

    @property
    def newsapi(self) -> NewsApiClient:
        """Access the NewsAPI client."""
        return self._newsapi

    @property
    def polygon(self) -> PolygonClient:
        """Access the Polygon.io market data client."""
        return self._polygon

    def get_provider_statuses(self) -> dict[str, dict[str, Any]]:
        """Return runtime health telemetry across all external data providers."""
        return {
            pid: {
                "provider_id": s.provider_id,
                "is_configured": s.is_configured,
                "is_healthy": s.is_healthy,
                "calls_made": s.calls_made,
                "last_call_timestamp_ns": s.last_call_timestamp_ns,
                "last_error": s.last_error,
            }
            for pid, s in self._statuses.items()
        }

    async def harvest_external_news(self, universe: list[str]) -> list[ExternalNewsItem]:
        """Aggregate real-world financial news across active authenticated providers.

        Args:
            universe: List of stock symbols to query for company-specific news.

        Returns:
            Consolidated list of ExternalNewsItem records.
        """
        all_news: list[ExternalNewsItem] = []

        # 1. Ingest Finnhub company news for monitored universe
        if self._finnhub.is_configured:
            for sym in universe[:5]:  # Bound rate usage to top 5 tickers
                try:
                    items = await self._finnhub.fetch_company_news(sym)
                    all_news.extend(items)
                    self._record_success("FINNHUB")
                except Exception as exc:
                    self._record_error("FINNHUB", str(exc))

        # 2. Ingest top business headlines from NewsAPI
        if self._newsapi.is_configured:
            try:
                macro_news = await self._newsapi.fetch_top_business_headlines()
                all_news.extend(macro_news)
                self._record_success("NEWSAPI")
            except Exception as exc:
                self._record_error("NEWSAPI", str(exc))

        return all_news

    async def fetch_aggregated_news(self, ticker: str | None = None) -> list[ExternalNewsItem]:
        """Fetch aggregated external news optionally focused on a specific ticker."""
        return await self.harvest_external_news([ticker] if ticker else [])

    async def get_macro_state(self) -> dict[str, float]:
        """Query macroeconomic regime indicators from FRED.

        Returns:
            Dictionary with 'yield_spread_10y_2y' and 'fed_funds_rate'.
        """
        try:
            snapshot = await self._fred.get_macro_snapshot()
            self._record_success("FRED")
            return snapshot
        except Exception as exc:
            self._record_error("FRED", str(exc))
            return {"yield_spread_10y_2y": 0.18, "fed_funds_rate": 5.25}

    def _record_success(self, provider_id: str) -> None:
        """Update provider telemetry on successful call."""
        if provider_id in self._statuses:
            status = self._statuses[provider_id]
            status.calls_made += 1
            status.is_healthy = True
            status.last_call_timestamp_ns = time.time_ns()
            status.last_error = None

    def _record_error(self, provider_id: str, error_msg: str) -> None:
        """Update provider telemetry on failed call."""
        if provider_id in self._statuses:
            status = self._statuses[provider_id]
            status.calls_made += 1
            status.is_healthy = False
            status.last_call_timestamp_ns = time.time_ns()
            status.last_error = error_msg
