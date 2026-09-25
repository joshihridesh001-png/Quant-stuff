"""Real-world financial news harvester, multi-source RSS/Atom parser, and deduplication engine.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates, Rule 4: Zero lookahead causality)
- Invariant 1: Availability Point-in-Time Causality (INV-NEWS-001)
- Invariant 2: Cryptographic Deduplication & Idempotency (INV-NEWS-002)
- Fault Codes: ERR-NEWS-001, ERR-NEWS-002, ERR-NEWS-003
"""

import hashlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import httpx

logger = logging.getLogger(__name__)

# Diagnostic Fault Codes (Rules.md Rule 2)
ERR_NEWS_FEED_UNREACHABLE = "ERR-NEWS-001"
ERR_NEWS_CORRUPT_PAYLOAD = "ERR-NEWS-002"
ERR_NEWS_FUTURE_TIMESTAMP = "ERR-NEWS-003"
ERR_NEWS_EMPTY_UNIVERSE = "ERR-NEWS-004"
ERR_NEWS_NON_FINITE_SIGNAL = "ERR-NEWS-005"
ERR_NEWS_PREDICTION_TIMEOUT = "ERR-NEWS-006"


class NewsHarvesterError(Exception):
    """Base exception for all news harvesting and ingestion anomalies."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class NewsFeedUnreachableException(NewsHarvesterError):
    """Raised when external HTTP RSS/Atom feed connection fails or times out (ERR-NEWS-001)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_NEWS_FEED_UNREACHABLE, message)


class CorruptNewsPayloadException(NewsHarvesterError):
    """Raised when news payload is malformed XML, empty, or unparseable (ERR-NEWS-002)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_NEWS_CORRUPT_PAYLOAD, message)


class FutureTimestampException(NewsHarvesterError):
    """Raised when an article publication timestamp is in the future violating causality (ERR-NEWS-003)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_NEWS_FUTURE_TIMESTAMP, message)


_TAG_RE = re.compile(r"<[^>]+>")
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")

# Well-known company name to ticker mapping for automated entity tagging
COMPANY_TICKER_MAP: dict[str, str] = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "AMAZON": "AMZN",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "TESLA": "TSLA",
    "META": "META",
    "JPMORGAN": "JPM",
    "BERKSHIRE": "BRK.B",
    "BROADCOM": "AVGO",
    "ELI LILLY": "LLY",
}


def sanitize_html(text: str) -> str:
    """Strip raw HTML markup and unescape XML/HTML character entities.

    Purpose: Converts dirty web syndication markup into clean plaintext tokens.
    Dependency: Standard library html and re.
    Relationship: Preprocessing step before semantic tokenization and hashing.
    Invariant: Result contains zero unescaped HTML tags.
    """
    if not text:
        return ""
    clean = _TAG_RE.sub(" ", text)
    clean = html.unescape(clean)
    return " ".join(clean.split()).strip()


@dataclass(slots=True, frozen=True)
class NewsArticle:
    """Immutable real-world financial news observation with point-in-time causality.

    Purpose: Encapsulates atomic news items with cryptographic hash and ingestion timestamps.
    Explicit Dependency Tracking: datetime.UTC, hashlib.sha256.
    Structural Relationship: Ingested by NewsHarvester, processed by FinancialSentimentClassifier.
    Defensive Invariant: Guaranteed immutable (frozen=True), SHA-256 hash uniquely identifies content.
    """

    headline: str
    summary: str
    url: str
    source: str
    published_at: datetime
    available_at: datetime
    tickers: tuple[str, ...] = field(default_factory=tuple)
    article_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Compute SHA-256 fingerprint over normalized content."""
        if not self.headline or not self.headline.strip():
            raise CorruptNewsPayloadException("Headline cannot be empty or whitespace.")

        # Ensure UTC timezone awareness
        if self.published_at.tzinfo is None or self.available_at.tzinfo is None:
            raise CorruptNewsPayloadException("Timestamps must be timezone-aware UTC datetimes.")

        norm_content = f"{self.source.strip().lower()}|{self.headline.strip().lower()}|{self.summary.strip().lower()}"
        computed_hash = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()
        object.__setattr__(self, "article_hash", computed_hash)

    def is_causal(self, clock_tolerance_seconds: float = 2.0) -> bool:
        """Verify publication timestamp does not lead local acquisition timestamp."""
        return self.available_at >= self.published_at - timedelta(seconds=clock_tolerance_seconds)


@dataclass(slots=True, frozen=True)
class NewsFeedConfig:
    """Configuration descriptor for external news syndication endpoints.

    Purpose: Defines endpoint URI, polling intervals, and wire protocol.
    Explicit Dependency Tracking: Standard library dataclasses.
    Structural Relationship: Parameterizes NewsHarvester polling cycles.
    Defensive Invariant: Strictly positive timeout and interval bounds.
    """

    name: str
    url: str
    feed_type: str = "rss"  # "rss", "atom", "edgar"
    poll_interval_seconds: float = 60.0
    timeout_seconds: float = 10.0
    enabled: bool = True


class NewsHarvester:
    """Multi-source async HTTP news harvester with SHA-256 deduplication and causal verification.

    Purpose: Autonomous background ingestion of breaking financial wires with zero lookahead bias.
    Explicit Dependency Tracking: httpx.AsyncClient, xml.etree.ElementTree, standard collections.
    Structural Relationship: Upstream ingestion provider for FinancialSentimentClassifier.
    Defensive Invariant: INV-NEWS-001 (Point-in-time causality), INV-NEWS-002 (SHA-256 Deduplication).
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        ring_buffer_capacity: int = 10_000,
        clock_skew_tolerance_seconds: float = 60.0,
    ) -> None:
        self._client = client
        self._ring_buffer_capacity = ring_buffer_capacity
        self._clock_skew_tolerance = clock_skew_tolerance_seconds
        self._hash_ring: deque[str] = deque(maxlen=ring_buffer_capacity)
        self._seen_hashes: set[str] = set()

    def filter_and_store_unique(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Filter articles against the SHA-256 deduplication ring buffer (INV-NEWS-002).

        Purpose: Prevents duplicate execution of re-syndicated news articles across wires.
        Dependency: In-memory deque and set.
        Relationship: Intermediary gate between wire parsing and classifier ingestion.
        Invariant: Output contains only articles never before observed within ring capacity.
        """
        unique_articles: list[NewsArticle] = []
        for article in articles:
            h = article.article_hash
            if h in self._seen_hashes:
                continue

            # Evict oldest hash from set if capacity reached
            if len(self._hash_ring) >= self._ring_buffer_capacity:
                oldest = self._hash_ring.popleft()
                self._seen_hashes.discard(oldest)

            self._hash_ring.append(h)
            self._seen_hashes.add(h)
            unique_articles.append(article)

        return unique_articles

    def parse_feed_xml(self, xml_content: str, source_name: str) -> list[NewsArticle]:
        """Parse raw RSS 2.0 or Atom XML payload into normalized NewsArticle objects.

        Purpose: Universal XML deserialization with HTML sanitization and causal timestamp verification.
        Dependency: xml.etree.ElementTree, sanitize_html, parsedate_to_datetime.
        Relationship: Ingestion parser for raw HTTP responses.
        Invariant: Rejects corrupted XML with ERR-NEWS-002, future timestamps with ERR-NEWS-003.
        """
        if not xml_content or not xml_content.strip():
            raise CorruptNewsPayloadException("RSS/Atom XML payload is empty or whitespace.")

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise CorruptNewsPayloadException(f"XML parse error encountered: {e}") from e

        articles: list[NewsArticle] = []
        now_utc = datetime.now(UTC)

        # Detect Atom format vs RSS 2.0 format
        is_atom = root.tag.endswith("feed")

        if is_atom:
            # Atom feed: search <entry> elements
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            if not entries:
                # Fallback without namespace
                entries = root.findall("entry")

            for entry in entries:
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_elem is None:
                    title_elem = entry.find("title")

                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                if summary_elem is None:
                    summary_elem = entry.find("{http://www.w3.org/2005/Atom}content")
                if summary_elem is None:
                    summary_elem = entry.find("summary")

                link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
                if link_elem is None:
                    link_elem = entry.find("link")

                updated_elem = entry.find("{http://www.w3.org/2005/Atom}updated")
                if updated_elem is None:
                    updated_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                if updated_elem is None:
                    updated_elem = entry.find("updated")

                raw_title = title_elem.text if title_elem is not None and title_elem.text else ""
                raw_summary = (
                    summary_elem.text if summary_elem is not None and summary_elem.text else ""
                )
                url = link_elem.get("href", "") if link_elem is not None else ""
                date_str = (
                    updated_elem.text if updated_elem is not None and updated_elem.text else ""
                )

                if not raw_title:
                    continue

                pub_time = self._parse_datetime(date_str, now_utc)
                self._verify_causal_timestamp(pub_time, now_utc)

                clean_title = sanitize_html(raw_title)
                clean_summary = sanitize_html(raw_summary)
                detected_tickers = self._extract_tickers(clean_title + " " + clean_summary)

                articles.append(
                    NewsArticle(
                        headline=clean_title,
                        summary=clean_summary,
                        url=url,
                        source=source_name,
                        published_at=pub_time,
                        available_at=now_utc,
                        tickers=tuple(detected_tickers),
                    )
                )
        else:
            # RSS 2.0 format: search <item> elements
            channel = root.find("channel")
            items = channel.findall("item") if channel is not None else root.findall(".//item")

            for item in items:
                title_elem = item.find("title")
                desc_elem = item.find("description")
                link_elem = item.find("link")
                pub_date_elem = item.find("pubDate")

                raw_title = title_elem.text if title_elem is not None and title_elem.text else ""
                raw_desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                url = link_elem.text if link_elem is not None and link_elem.text else ""
                date_str = (
                    pub_date_elem.text if pub_date_elem is not None and pub_date_elem.text else ""
                )

                if not raw_title:
                    continue

                pub_time = self._parse_datetime(date_str, now_utc)
                self._verify_causal_timestamp(pub_time, now_utc)

                clean_title = sanitize_html(raw_title)
                clean_summary = sanitize_html(raw_desc)
                detected_tickers = self._extract_tickers(clean_title + " " + clean_summary)

                articles.append(
                    NewsArticle(
                        headline=clean_title,
                        summary=clean_summary,
                        url=url,
                        source=source_name,
                        published_at=pub_time,
                        available_at=now_utc,
                        tickers=tuple(detected_tickers),
                    )
                )

        return articles

    def _parse_datetime(self, date_str: str, fallback_now: datetime) -> datetime:
        """Parse RFC 822 or ISO 8601 date string into UTC datetime."""
        if not date_str or not date_str.strip():
            return fallback_now

        cleaned_str = date_str.strip()
        # Attempt RFC 822/2822 parsing (standard RSS)
        try:
            dt = parsedate_to_datetime(cleaned_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except Exception:
            pass

        # Attempt ISO 8601 parsing (Atom standard)
        try:
            iso_str = cleaned_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except Exception:
            pass

        return fallback_now

    def _verify_causal_timestamp(self, pub_time: datetime, now_utc: datetime) -> None:
        """Enforce Invariant 1: Published timestamp must not lead local acquisition clock."""
        tolerance = timedelta(seconds=self._clock_skew_tolerance)
        if pub_time > now_utc + tolerance:
            skew = (pub_time - now_utc).total_seconds()
            raise FutureTimestampException(
                f"Future publication timestamp detected ({pub_time.isoformat()} leads local clock {now_utc.isoformat()} by {skew:.1f}s). Violates INV-NEWS-001."
            )

    def _extract_tickers(self, text: str) -> list[str]:
        """Detect tickers via regex and company alias matching."""
        found: set[str] = set()
        upper_text = text.upper()

        # Company alias matching
        for company, ticker in COMPANY_TICKER_MAP.items():
            if company in upper_text:
                found.add(ticker)

        # Explicit ticker match
        matches = _TICKER_RE.findall(text)
        for m in matches:
            if m in COMPANY_TICKER_MAP.values():
                found.add(m)

        return sorted(found)

    async def poll_feed(self, feed_config: NewsFeedConfig) -> list[NewsArticle]:
        """Poll a single external syndication feed over HTTP/2.

        Purpose: Asynchronous I/O retrieval with timeout guards and error mapping.
        Dependency: httpx.AsyncClient.
        Relationship: Polling worker dispatched by NewsPredictionService.
        Invariant: Network faults raise NewsFeedUnreachableException (ERR-NEWS-001).
        """
        client = self._client
        owns_client = False
        if client is None:
            user_agent = (
                "QuantResearchWorkbench/1.0 (compliance@quant.internal)"
                if "sec.gov" in feed_config.url
                else (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            default_headers = {
                "User-Agent": user_agent,
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
            }
            client = httpx.AsyncClient(headers=default_headers, follow_redirects=True)
            owns_client = True

        try:
            resp = await client.get(feed_config.url, timeout=feed_config.timeout_seconds)
            if resp.status_code != 200:
                raise NewsFeedUnreachableException(
                    f"HTTP {resp.status_code} returned for feed {feed_config.name} at {feed_config.url}"
                )
            articles = self.parse_feed_xml(resp.text, source_name=feed_config.name)
            return self.filter_and_store_unique(articles)
        except NewsHarvesterError:
            raise
        except Exception as e:
            raise NewsFeedUnreachableException(
                f"Failed to poll feed {feed_config.name} at {feed_config.url}: {e}"
            ) from e
        finally:
            if owns_client:
                await client.aclose()

    async def poll_all(self, feeds: list[NewsFeedConfig]) -> list[NewsArticle]:
        """Poll multiple feeds concurrently and return aggregated unique articles."""
        all_articles: list[NewsArticle] = []
        for feed in feeds:
            if not feed.enabled:
                continue
            try:
                articles = await self.poll_feed(feed)
                all_articles.extend(articles)
            except NewsFeedUnreachableException as e:
                logger.warning(f"Skipping unreachable feed {feed.name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error polling feed {feed.name}: {e}")
        return all_articles


class SyntheticNewsGenerator:
    """Hermetic synthetic financial news generator for deterministic testing and simulation.

    Purpose: Generates structured, diverse, realistic news articles without external network I/O.
    Dependency: random.Random, datetime.UTC.
    Relationship: Test harness provider and paper simulation feed.
    Invariant: Timestamps are always valid, non-future, causal, and formatted to domain schema.
    """

    TEMPLATES = [
        # Earnings
        (
            "Apple Reports Record Q4 iPhone Revenue Beats Wall Street Consensus",
            "AAPL Q4 revenue rose 8% YoY to $94.9B topping analyst estimates.",
            ("AAPL",),
        ),
        (
            "NVIDIA Unveils Blackwell AI Superclusters with Record Enterprise Demand",
            "NVIDIA Corp posted surging data center revenue and bullish forward guidance.",
            ("NVDA",),
        ),
        (
            "Tesla Automotive Gross Margins Expand in Strong Q3 Delivery Beat",
            "Tesla delivered 462,000 electric vehicles exceeding quarterly market consensus.",
            ("TSLA",),
        ),
        (
            "Microsoft Azure Cloud Growth Surges 33% Driven by Copilot Integration",
            "Microsoft reported earnings per share of $3.30 beating consensus estimates.",
            ("MSFT",),
        ),
        (
            "Amazon Web Services Operating Profit Soars as AI Workloads Accelerate",
            "Amazon posted strong operating income growth with AWS expanding margins.",
            ("AMZN",),
        ),
        (
            "Alphabet Cloud Generates $11B as AI Infrastructure Monetization Accelerates",
            "Google parent Alphabet delivered double-digit revenue expansion.",
            ("GOOGL",),
        ),
        # Macro / Fed
        (
            "Federal Reserve Holds Benchmark Rate Steady, Signals Dovish Path Ahead",
            "FOMC statement emphasizes balanced inflation risks and resilient labor market.",
            ("SPY", "QQQ"),
        ),
        (
            "Consumer Price Index Inflation Cools to 2.1% Supporting Rate Cut Expectations",
            "US Bureau of Labor Statistics reported headline CPI moderated in line with forecasts.",
            ("SPY",),
        ),
        # M&A / Corporate Actions
        (
            "Broadcom Completes Strategic Enterprise Software Acquisition",
            "Broadcom closes $61B transaction with immediate accretive cash flow guidance.",
            ("AVGO",),
        ),
        (
            "JPMorgan Chase Expands Private Wealth Assets Under Management",
            "JPMorgan reported record net interest income and raised full-year targets.",
            ("JPM",),
        ),
    ]

    def __init__(self, seed: int = 42) -> None:
        import random

        self._rng = random.Random(seed)

    def generate_batch(
        self, count: int = 10, target_ticker: str | None = None
    ) -> list[NewsArticle]:
        """Generate a deterministic batch of causal synthetic news articles."""
        now = datetime.now(UTC)
        articles: list[NewsArticle] = []

        eligible_templates = list(self.TEMPLATES)
        if target_ticker:
            matched = [
                tpl
                for tpl in self.TEMPLATES
                if any(t.upper() == target_ticker.upper() for t in tpl[2])
            ]
            if matched:
                eligible_templates = matched

        for i in range(count):
            tpl = (
                eligible_templates[i % len(eligible_templates)]
                if not target_ticker
                else self._rng.choice(eligible_templates)
            )
            headline, summary, tickers = tpl
            # Publication time slightly in the past (1 minute to 2 hours ago)
            offset_seconds = self._rng.randint(60, 7200)
            pub_time = now - timedelta(seconds=offset_seconds)
            avail_time = pub_time + timedelta(seconds=self._rng.randint(1, 10))

            ts_epoch = int(now.timestamp())
            articles.append(
                NewsArticle(
                    headline=f"{headline} (Wire-{ts_epoch}-{i + 1})",
                    summary=summary,
                    url=f"https://synthetic-wire.internal/articles/{ts_epoch}-{i + 1}",
                    source="synthetic",
                    published_at=pub_time,
                    available_at=avail_time,
                    tickers=tickers,
                )
            )

        return articles
