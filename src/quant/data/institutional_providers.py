"""High-resolution institutional historical market data provider adapters.

Functional Purpose:
    Implements cursor-paginated historical aggregate bar downloaders for:
    - Polygon.io (Aggregates v2 API with microsecond/millisecond timestamps).
    - Alpaca Market Data v2 (SIP/IEX consolidated bars with RFC 3339 timestamps).
    - HistoricalDataHub: Unified facade that coordinates provider priority and enforces
      seamless zero-failure fallback to Yahoo Finance (INV-DATA-008) when credentials are
      unconfigured or rate limits are exceeded.

Explicit Dependency Tracking:
    - httpx: Asynchronous and synchronous HTTP transport.
    - datetime: RFC 3339 timestamp parsing and ISO-8601 formatting.
    - quant.core.config: System settings (POLYGON_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY).
    - quant.data.yahoo_provider: YahooFinanceHistoricalProvider fallback adapter.
    - quant.domain.historical: HistoricalPriceBar, HistoricalBarBatch, Resolution.

Structural Relationship:
    - Step 13.3 in Phase 13 (Real Historical Data Lake).
    - Bridges commercial institutional APIs with the domain HistoricalBarBatch model.
    - Consumed by CLI ingestion runners and DuckDB columnar storage pipeline.

Defensive Invariants:
    - INV-DATA-004: Monotonic temporal ordering across paginated chunks.
    - INV-DATA-005: Price geometry feasibility (High >= max(O,C), Low <= min(O,C)).
    - INV-DATA-008: Zero-failure fallback chain (Polygon -> Alpaca -> Yahoo Finance).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Final

import httpx

from quant.core.config import get_settings
from quant.data.yahoo_provider import YahooFinanceHistoricalProvider
from quant.domain.historical import (
    EmptyHistoricalBatchError,
    HistoricalBarBatch,
    HistoricalDataError,
    HistoricalPriceBar,
    InvalidBarGeometryError,
)
from quant.domain.models import Resolution

logger = logging.getLogger(__name__)

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_DATA_POLYGON_PAGINATION: Final[str] = "ERR-DATA-015"
ERR_DATA_ALPACA_PAGINATION: Final[str] = "ERR-DATA-016"
ERR_DATA_PROVIDER_FALLBACK: Final[str] = "ERR-DATA-017"


# ============================================================================
# Exception Hierarchy
# ============================================================================


class InstitutionalProviderError(HistoricalDataError):
    """Base exception for institutional market data provider failures."""

    def __init__(self, message: str, code: str = "ERR-DATA-015") -> None:
        super().__init__(message, code=code)


class PolygonProviderError(InstitutionalProviderError):
    """Raised on Polygon.io aggregate query or pagination failures."""

    def __init__(self, message: str, code: str = ERR_DATA_POLYGON_PAGINATION) -> None:
        super().__init__(message, code=code)


class AlpacaHistoricalError(InstitutionalProviderError):
    """Raised on Alpaca Market Data v2 query or pagination failures."""

    def __init__(self, message: str, code: str = ERR_DATA_ALPACA_PAGINATION) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Resolution Mappings
# ============================================================================

POLYGON_RESOLUTION_MAP: Final[dict[Resolution, tuple[int, str]]] = {
    Resolution.ONE_MINUTE: (1, "minute"),
    Resolution.FIVE_MINUTES: (5, "minute"),
    Resolution.FIFTEEN_MINUTES: (15, "minute"),
    Resolution.ONE_HOUR: (1, "hour"),
    Resolution.ONE_DAY: (1, "day"),
}

ALPACA_TIMEFRAME_MAP: Final[dict[Resolution, str]] = {
    Resolution.ONE_MINUTE: "1Min",
    Resolution.FIVE_MINUTES: "5Min",
    Resolution.FIFTEEN_MINUTES: "15Min",
    Resolution.ONE_HOUR: "1Hour",
    Resolution.ONE_DAY: "1Day",
}


# ============================================================================
# Polygon.io Historical Provider
# ============================================================================


class PolygonHistoricalProvider:
    """Historical aggregate bar client for Polygon.io API.

    Functional Purpose:
        Downloads high-resolution 1-minute to daily bars using cursor pagination.
    Explicit Dependency Tracking:
        httpx.AsyncClient / Client, Resolution.
    Defensive Invariants:
        Handles API key verification, cursor following, and boundary checking.
    """

    BASE_URL: Final[str] = "https://api.polygon.io/v2/aggs/ticker"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
        timeout: float = 20.0,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.POLYGON_API_KEY
        self._async_client = client
        self._sync_client = sync_client
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Check if Polygon API key is present and non-empty."""
        return bool(self._api_key and len(self._api_key.strip()) > 0)

    @staticmethod
    def _format_date(ts: datetime | int) -> str:
        """Format timestamp as YYYY-MM-DD string."""
        if isinstance(ts, int):
            if ts > 10_000_000_000_000:
                ts = ts // 1_000_000_000
            dt = datetime.fromtimestamp(ts, tz=UTC)
        else:
            dt = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        return dt.strftime("%Y-%m-%d")

    def parse_polygon_results(
        self,
        symbol: str,
        resolution: Resolution,
        results: list[dict[str, Any]],
    ) -> HistoricalBarBatch:
        """Parse raw Polygon.io result dictionaries into HistoricalBarBatch."""
        if not results:
            raise EmptyHistoricalBatchError(f"Polygon returned empty results for symbol '{symbol}'")

        bars: list[HistoricalPriceBar] = []
        for r in results:
            ts_ms = r.get("t")
            if ts_ms is None or ts_ms <= 0:
                continue

            o = float(r.get("o", 0.0))
            h = float(r.get("h", 0.0))
            lo = float(r.get("l", 0.0))
            c = float(r.get("c", 0.0))
            v = float(r.get("v", 0.0))
            vw = float(r.get("vw", c))

            if o <= 0.0 or h <= 0.0 or lo <= 0.0 or c <= 0.0:
                continue

            h = max(h, o, c)
            lo = min(lo, o, c)
            if vw <= 0.0:
                vw = (h + lo + c) / 3.0

            ts_ns = int(ts_ms) * 1_000_000

            try:
                bar = HistoricalPriceBar(
                    asset_id=symbol,
                    timestamp=ts_ns,
                    open=o,
                    high=h,
                    low=lo,
                    close=c,
                    volume=max(0.0, v),
                    vwap=vw,
                    resolution=resolution,
                    adj_close=c,
                    split_factor=1.0,
                    dividend_amount=0.0,
                )
                bars.append(bar)
            except InvalidBarGeometryError as geom_err:
                logger.warning(f"Dropping bar with geometry anomaly: {geom_err}")

        if not bars:
            raise EmptyHistoricalBatchError(f"All Polygon bars for '{symbol}' were degenerate")

        bars.sort(key=lambda b: b.timestamp)
        deduped: list[HistoricalPriceBar] = []
        for b in bars:
            if not deduped or b.timestamp > deduped[-1].timestamp:
                deduped.append(b)
            elif b.timestamp == deduped[-1].timestamp:
                deduped[-1] = b

        return HistoricalBarBatch(
            symbol=symbol,
            resolution=resolution,
            bars=tuple(deduped),
        )

    async def fetch_historical_bars_async(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Download historical bars from Polygon.io asynchronously with pagination."""
        if not self.is_configured:
            raise PolygonProviderError("Polygon API key is not configured")

        multiplier, timespan = POLYGON_RESOLUTION_MAP.get(resolution, (1, "day"))
        from_str = self._format_date(start)
        to_str = self._format_date(end)

        url: str | None = (
            f"{self.BASE_URL}/{symbol.upper()}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
        )
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}
        params: dict[str, Any] = {"adjusted": "true", "sort": "asc", "limit": 50000}

        all_results: list[dict[str, Any]] = []

        should_close = False
        client = self._async_client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            should_close = True

        try:
            while url:
                resp = await client.get(
                    url, params=params if url.startswith(self.BASE_URL) else None, headers=headers
                )
                if resp.status_code == 401:
                    raise PolygonProviderError("Invalid Polygon API Key (HTTP 401)")
                if resp.status_code == 429:
                    raise PolygonProviderError("Polygon API rate limit exceeded (HTTP 429)")
                if resp.status_code != 200:
                    raise PolygonProviderError(
                        f"Polygon API returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                payload = resp.json()
                results = payload.get("results", [])
                if isinstance(results, list):
                    all_results.extend(results)

                # Follow cursor pagination if present
                next_url = payload.get("next_url")
                if next_url and next_url != url:
                    url = next_url
                    params = {}  # Parameters are encoded in next_url
                else:
                    url = None

            return self.parse_polygon_results(symbol.upper(), resolution, all_results)
        finally:
            if should_close:
                await client.aclose()

    def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Download historical bars from Polygon.io synchronously.

        Functional Purpose:
            Synchronous bridge delegating to async fetch implementation.
        Explicit Dependency Tracking:
            asyncio.run, fetch_historical_bars_async.
        Defensive Invariants:
            INV-DATA-004: Strictly monotonic timestamps.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.fetch_historical_bars_async(symbol, start, end, resolution),
                ).result()
        return asyncio.run(self.fetch_historical_bars_async(symbol, start, end, resolution))


# ============================================================================
# Alpaca Market Data v2 Historical Provider
# ============================================================================


class AlpacaHistoricalProvider:
    """Historical aggregate bar client for Alpaca Market Data v2 API.

    Functional Purpose:
        Queries SIP consolidated equity bars with page_token cursor pagination.
    Explicit Dependency Tracking:
        httpx.AsyncClient / Client, Resolution.
    Defensive Invariants:
        RFC 3339 timestamp normalization and authentication header injection.
    """

    DEFAULT_BASE_URL: Final[str] = "https://data.alpaca.markets/v2/stocks/bars"

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
        timeout: float = 20.0,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.ALPACA_API_KEY
        self._secret_key = secret_key if secret_key is not None else settings.ALPACA_SECRET_KEY
        self._base_url = (
            base_url if base_url is not None else f"{settings.ALPACA_DATA_URL}/v2/stocks/bars"
        )
        self._async_client = client
        self._sync_client = sync_client
        self._timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Check if Alpaca API Key and Secret are present."""
        return bool(self._api_key and self._secret_key)

    @staticmethod
    def _format_rfc3339(ts: datetime | int) -> str:
        """Format timestamp into RFC 3339 string."""
        if isinstance(ts, int):
            if ts > 10_000_000_000_000:
                ts = ts // 1_000_000_000
            dt = datetime.fromtimestamp(ts, tz=UTC)
        else:
            dt = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        return dt.isoformat()

    def parse_alpaca_bars(
        self,
        symbol: str,
        resolution: Resolution,
        bars_list: list[dict[str, Any]],
    ) -> HistoricalBarBatch:
        """Parse raw Alpaca bars into HistoricalBarBatch."""
        if not bars_list:
            raise EmptyHistoricalBatchError(f"Alpaca returned empty bar list for symbol '{symbol}'")

        bars: list[HistoricalPriceBar] = []
        for b in bars_list:
            t_str = b.get("t")
            if not t_str or not isinstance(t_str, str):
                continue

            try:
                dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
                ts_ns = int(dt.timestamp() * 1_000_000_000)
            except Exception:
                continue

            o = float(b.get("o", 0.0))
            h = float(b.get("h", 0.0))
            lo = float(b.get("l", 0.0))
            c = float(b.get("c", 0.0))
            v = float(b.get("v", 0.0))
            vw = float(b.get("vw", c))

            if o <= 0.0 or h <= 0.0 or lo <= 0.0 or c <= 0.0:
                continue

            h = max(h, o, c)
            lo = min(lo, o, c)
            if vw <= 0.0:
                vw = (h + lo + c) / 3.0

            try:
                bar = HistoricalPriceBar(
                    asset_id=symbol,
                    timestamp=ts_ns,
                    open=o,
                    high=h,
                    low=lo,
                    close=c,
                    volume=max(0.0, v),
                    vwap=vw,
                    resolution=resolution,
                    adj_close=c,
                    split_factor=1.0,
                    dividend_amount=0.0,
                )
                bars.append(bar)
            except InvalidBarGeometryError as geom_err:
                logger.warning(f"Dropping bar with geometry anomaly: {geom_err}")

        if not bars:
            raise EmptyHistoricalBatchError(f"All Alpaca bars for '{symbol}' were degenerate")

        bars.sort(key=lambda item: item.timestamp)
        deduped: list[HistoricalPriceBar] = []
        for bar_item in bars:
            if not deduped or bar_item.timestamp > deduped[-1].timestamp:
                deduped.append(bar_item)
            elif bar_item.timestamp == deduped[-1].timestamp:
                deduped[-1] = bar_item

        return HistoricalBarBatch(
            symbol=symbol,
            resolution=resolution,
            bars=tuple(deduped),
        )

    async def fetch_historical_bars_async(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Download historical bars from Alpaca v2 asynchronously with page token pagination."""
        if not self.is_configured:
            raise AlpacaHistoricalError("Alpaca API credentials are not configured")

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        }
        timeframe_str = ALPACA_TIMEFRAME_MAP.get(resolution, "1Day")
        start_str = self._format_rfc3339(start)
        end_str = self._format_rfc3339(end)

        all_bars: list[dict[str, Any]] = []
        page_token: str | None = None

        should_close = False
        client = self._async_client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            should_close = True

        try:
            while True:
                params: dict[str, Any] = {
                    "symbols": symbol.upper(),
                    "timeframe": timeframe_str,
                    "start": start_str,
                    "end": end_str,
                    "limit": 10000,
                    "adjustment": "all",
                }
                if page_token:
                    params["page_token"] = page_token

                resp = await client.get(self._base_url, params=params, headers=headers)
                if resp.status_code == 401 or resp.status_code == 403:
                    raise AlpacaHistoricalError("Invalid Alpaca credentials (HTTP 401/403)")
                if resp.status_code != 200:
                    raise AlpacaHistoricalError(
                        f"Alpaca API returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                payload = resp.json()
                bars_dict = payload.get("bars", {})
                symbol_bars = bars_dict.get(symbol.upper(), [])
                if isinstance(symbol_bars, list):
                    all_bars.extend(symbol_bars)

                page_token = payload.get("next_page_token")
                if not page_token:
                    break

            return self.parse_alpaca_bars(symbol.upper(), resolution, all_bars)
        finally:
            if should_close:
                await client.aclose()

    def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Download historical bars from Alpaca v2 synchronously.

        Functional Purpose:
            Synchronous bridge delegating to async fetch implementation.
        Explicit Dependency Tracking:
            asyncio.run, fetch_historical_bars_async.
        Defensive Invariants:
            INV-DATA-004: Strictly monotonic timestamps.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.fetch_historical_bars_async(symbol, start, end, resolution),
                ).result()
        return asyncio.run(self.fetch_historical_bars_async(symbol, start, end, resolution))


# ============================================================================
# Unified Historical Data Hub (Zero-Failure Fallback Chain)
# ============================================================================


class HistoricalDataHub:
    """Master coordinator managing provider priority and seamless fallback (INV-DATA-008).

    Functional Purpose:
        Provides a single, resilient entrypoint for downloading historical market bars.
        Attempts Polygon.io or Alpaca v2 if configured. If credentials are missing,
        rate-limited, or network fails, transparently cascades to Yahoo Finance.

    Explicit Dependency Tracking:
        PolygonHistoricalProvider, AlpacaHistoricalProvider, YahooFinanceHistoricalProvider.

    Structural Relationship:
        Primary interface injected into data ingestion services and CLI utilities.

    Defensive Invariants:
        INV-DATA-008: Zero-failure ingestion guarantee as long as public endpoints are reachable.
    """

    def __init__(
        self,
        polygon_provider: PolygonHistoricalProvider | None = None,
        alpaca_provider: AlpacaHistoricalProvider | None = None,
        yahoo_provider: YahooFinanceHistoricalProvider | None = None,
        preferred_provider: str = "auto",
    ) -> None:
        self.polygon = polygon_provider or PolygonHistoricalProvider()
        self.alpaca = alpaca_provider or AlpacaHistoricalProvider()
        self.yahoo = yahoo_provider or YahooFinanceHistoricalProvider()
        self.preferred_provider = preferred_provider.lower()

    async def fetch_bars_async(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Download historical bars using preferred provider with automatic fallback."""
        pref = self.preferred_provider

        # 1. Try Polygon if explicitly requested or auto-configured
        if (pref == "polygon" or pref == "auto") and self.polygon.is_configured:
            try:
                logger.info(f"Fetching '{symbol}' historical bars from Polygon.io...")
                return await self.polygon.fetch_historical_bars_async(
                    symbol, start, end, resolution
                )
            except Exception as e:
                logger.warning(
                    f"Polygon fetch failed for '{symbol}': {e}. Falling back to next provider..."
                )

        # 2. Try Alpaca if explicitly requested or auto-configured
        if (pref == "alpaca" or pref == "auto") and self.alpaca.is_configured:
            try:
                logger.info(f"Fetching '{symbol}' historical bars from Alpaca v2...")
                return await self.alpaca.fetch_historical_bars_async(symbol, start, end, resolution)
            except Exception as e:
                logger.warning(
                    f"Alpaca fetch failed for '{symbol}': {e}. Falling back to Yahoo Finance..."
                )

        # 3. Fallback to Yahoo Finance (Zero credentials required)
        logger.info(f"Fetching '{symbol}' historical bars from Yahoo Finance fallback...")
        return await self.yahoo.fetch_historical_bars_async(symbol, start, end, resolution)

    def fetch_bars(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Synchronous fetch delegating directly to synchronous provider clients."""
        pref = self.preferred_provider

        # 1. Try Polygon if explicitly requested or auto-configured
        if (pref == "polygon" or pref == "auto") and self.polygon.is_configured:
            try:
                return asyncio.run(
                    self.polygon.fetch_historical_bars_async(symbol, start, end, resolution)
                )
            except Exception as e:
                logger.warning(f"Sync Polygon fetch failed for '{symbol}': {e}. Falling back...")

        # 2. Try Alpaca if explicitly requested or auto-configured
        if (pref == "alpaca" or pref == "auto") and self.alpaca.is_configured:
            try:
                return asyncio.run(
                    self.alpaca.fetch_historical_bars_async(symbol, start, end, resolution)
                )
            except Exception as e:
                logger.warning(f"Sync Alpaca fetch failed for '{symbol}': {e}. Falling back...")

        # 3. Fallback to synchronous Yahoo Finance provider
        return self.yahoo.fetch_historical_bars(symbol, start, end, resolution)

    def fetch_historical_bars(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Polymorphic alias for fetch_bars adhering to historical provider protocol."""
        return self.fetch_bars(symbol, start, end, resolution)

    async def fetch_historical_bars_async(
        self,
        symbol: str,
        start: datetime | int,
        end: datetime | int,
        resolution: Resolution = Resolution.ONE_DAY,
    ) -> HistoricalBarBatch:
        """Polymorphic alias for fetch_bars_async adhering to historical provider protocol."""
        return await self.fetch_bars_async(symbol, start, end, resolution)
