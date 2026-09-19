"""Unit tests for Phase 9 real-world financial news harvester and deduplication engine.

Governing Standards:
- Rules.md (Rule 1, Rule 2, Rule 3, Rule 4)
- INV-NEWS-001: Availability Point-in-Time Causality
- INV-NEWS-002: Cryptographic Deduplication & Idempotency
- Fault codes: ERR-NEWS-001, ERR-NEWS-002, ERR-NEWS-003
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant.data.news_harvester import (
    CorruptNewsPayloadException,
    FutureTimestampException,
    NewsArticle,
    NewsFeedConfig,
    NewsFeedUnreachableException,
    NewsHarvester,
    SyntheticNewsGenerator,
)

SAMPLE_YAHOO_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Yahoo Finance: Latest News</title>
    <link>https://finance.yahoo.com</link>
    <description>Financial market updates and breaking corporate news.</description>
    <item>
      <title>Apple Reports Record Q4 Services Revenue Beats Estimates</title>
      <link>https://finance.yahoo.com/news/apple-record-q4-12345.html</link>
      <description>&lt;p&gt;Apple Inc. announced quarterly revenue of $94.9 billion, driven by iPhone 16 demand.&lt;/p&gt;</description>
      <pubDate>Fri, 18 Sep 2026 14:30:00 +0000</pubDate>
      <guid isPermaLink="false">yahoo-apple-20260918-01</guid>
    </item>
    <item>
      <title>Fed Signals Possible Interest Rate Cut as Inflation Cools</title>
      <link>https://finance.yahoo.com/news/fed-signals-cut-67890.html</link>
      <description>Federal Reserve officials indicated openness to lowering policy rates at upcoming FOMC meeting.</description>
      <pubDate>Fri, 18 Sep 2026 15:00:00 +0000</pubDate>
      <guid isPermaLink="false">yahoo-fed-20260918-02</guid>
    </item>
  </channel>
</rss>
"""

SAMPLE_SEC_EDGAR_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>SEC EDGAR 8-K Current Reports</title>
  <link href="https://www.sec.gov/edgar" rel="self"/>
  <updated>2026-09-18T16:00:00Z</updated>
  <entry>
    <title>8-K: NVIDIA CORP (0001045810) - Material Definitive Agreement</title>
    <link href="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000012/nvda-8k.htm"/>
    <summary type="html">&lt;b&gt;Item 1.01&lt;/b&gt;: Entry into Material Definitive Agreement for AI Datacenter Infrastructure.</summary>
    <updated>2026-09-18T15:45:00Z</updated>
    <id>urn:tag:sec.gov,2026:edgar:0001045810-26-000012</id>
  </entry>
</feed>
"""


class TestNewsArticle:
    """Unit tests verifying NewsArticle value object immutability and hashing."""

    def test_news_article_instantiation_and_hash_stability(self) -> None:
        pub_time = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
        avail_time = datetime(2026, 9, 18, 12, 0, 1, tzinfo=UTC)

        article1 = NewsArticle(
            headline="Tesla Unveils Full Autonomous Robotaxi Fleet",
            summary="Tesla revealed its next generation dedicated robotaxi with unboxed manufacturing.",
            url="https://finance.example.com/tsla-robotaxi",
            source="reuters",
            published_at=pub_time,
            available_at=avail_time,
            tickers=("TSLA",),
        )

        article2 = NewsArticle(
            headline="Tesla Unveils Full Autonomous Robotaxi Fleet",
            summary="Tesla revealed its next generation dedicated robotaxi with unboxed manufacturing.",
            url="https://finance.example.com/tsla-robotaxi-duplicate",
            source="reuters",
            published_at=pub_time,
            available_at=avail_time,
            tickers=("TSLA",),
        )

        assert article1.article_hash == article2.article_hash
        assert len(article1.article_hash) == 64  # SHA-256 hex digest
        assert article1.is_causal()

    def test_news_article_immutability(self) -> None:
        now = datetime.now(UTC)
        article = NewsArticle(
            headline="Microsoft Announces Cloud Dividend",
            summary="MSFT board authorizes dividend increase.",
            url="https://msft.example.com",
            source="bloomberg",
            published_at=now,
            available_at=now,
        )
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            article.headline = "Mutated headline"  # type: ignore[misc]


class TestNewsHarvester:
    """Unit tests verifying RSS XML ingestion, deduplication, and point-in-time invariants."""

    def test_parse_rss_yahoo_feed(self) -> None:
        harvester = NewsHarvester()
        articles = harvester.parse_feed_xml(SAMPLE_YAHOO_RSS, source_name="yahoo")

        assert len(articles) == 2

        first = articles[0]
        assert "Apple Reports Record" in first.headline
        assert "Apple Inc. announced quarterly revenue" in first.summary
        assert "<p>" not in first.summary
        assert first.source == "yahoo"
        assert first.published_at == datetime(2026, 9, 18, 14, 30, 0, tzinfo=UTC)
        assert first.available_at.tzinfo == UTC
        assert first.is_causal()
        assert "AAPL" in first.tickers

        second = articles[1]
        assert "Fed Signals Possible Interest Rate Cut" in second.headline
        assert second.source == "yahoo"

    def test_parse_atom_edgar_feed(self) -> None:
        harvester = NewsHarvester()
        articles = harvester.parse_feed_xml(SAMPLE_SEC_EDGAR_ATOM, source_name="sec_edgar")

        assert len(articles) == 1
        entry = articles[0]
        assert "NVIDIA CORP" in entry.headline
        assert "Item 1.01" in entry.summary
        assert "<b>" not in entry.summary
        assert entry.source == "sec_edgar"
        assert entry.published_at == datetime(2026, 9, 18, 15, 45, 0, tzinfo=UTC)
        assert "NVDA" in entry.tickers

    def test_corrupt_payload_raises_exception(self) -> None:
        harvester = NewsHarvester()
        malformed_xml = "<rss><channel><item><title>Broken XML"

        with pytest.raises(CorruptNewsPayloadException) as exc_info:
            harvester.parse_feed_xml(malformed_xml, source_name="yahoo")

        assert exc_info.value.code == "ERR-NEWS-002"

    def test_empty_xml_payload_raises_exception(self) -> None:
        harvester = NewsHarvester()
        with pytest.raises(CorruptNewsPayloadException) as exc_info:
            harvester.parse_feed_xml("", source_name="test")
        assert exc_info.value.code == "ERR-NEWS-002"

    def test_deduplication_ring_buffer_rejects_re_syndication(self) -> None:
        harvester = NewsHarvester(ring_buffer_capacity=100)
        articles = harvester.parse_feed_xml(SAMPLE_YAHOO_RSS, source_name="yahoo")
        assert len(articles) == 2

        # Second parse of identical content: deduplicator must drop both articles
        deduped = harvester.filter_and_store_unique(articles)
        assert len(deduped) == 2

        second_attempt = harvester.filter_and_store_unique(articles)
        assert len(second_attempt) == 0

    def test_future_timestamp_rejection_invariant_inv_news_001(self) -> None:
        harvester = NewsHarvester(clock_skew_tolerance_seconds=30.0)
        future_time = datetime.now(UTC) + timedelta(hours=3)
        future_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Time Traveler News</title>
            <item>
              <title>Future Quantum Computing Breakthrough</title>
              <link>https://future.example.com</link>
              <description>News from tomorrow.</description>
              <pubDate>{future_time.strftime("%a, %d %b %Y %H:%M:%S +0000")}</pubDate>
            </item>
          </channel>
        </rss>
        """

        with pytest.raises(FutureTimestampException) as exc_info:
            harvester.parse_feed_xml(future_xml, source_name="test_future")

        assert exc_info.value.code == "ERR-NEWS-003"
        assert "ERR-NEWS-003" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_poll_feed_success(self) -> None:
        config = NewsFeedConfig(
            name="mock_yahoo",
            url="https://finance.yahoo.com/rss/mock",
            feed_type="rss",
            timeout_seconds=5.0,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_YAHOO_RSS
        mock_client.get.return_value = mock_response

        harvester = NewsHarvester(client=mock_client)
        articles = await harvester.poll_feed(config)

        assert len(articles) == 2
        mock_client.get.assert_called_once_with("https://finance.yahoo.com/rss/mock", timeout=5.0)

    @pytest.mark.asyncio
    async def test_async_poll_feed_unreachable_exception(self) -> None:
        config = NewsFeedConfig(
            name="broken_feed",
            url="https://unreachable.example.com/rss",
            feed_type="rss",
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused / DNS lookup timeout")

        harvester = NewsHarvester(client=mock_client)
        with pytest.raises(NewsFeedUnreachableException) as exc_info:
            await harvester.poll_feed(config)

        assert exc_info.value.code == "ERR-NEWS-001"

    @pytest.mark.asyncio
    async def test_async_poll_all_aggregates_and_handles_failures(self) -> None:
        feed_good = NewsFeedConfig(name="good", url="https://good.rss", enabled=True)
        feed_disabled = NewsFeedConfig(name="disabled", url="https://disabled.rss", enabled=False)
        feed_bad = NewsFeedConfig(name="bad", url="https://bad.rss", enabled=True)

        mock_client = AsyncMock()

        async def mock_get(url: str, **kwargs: object) -> MagicMock:
            if "good" in url:
                m = MagicMock()
                m.status_code = 200
                m.text = SAMPLE_YAHOO_RSS
                return m
            raise Exception("Timeout")

        mock_client.get.side_effect = mock_get

        harvester = NewsHarvester(client=mock_client)
        articles = await harvester.poll_all([feed_good, feed_disabled, feed_bad])

        assert len(articles) == 2
        assert articles[0].source == "good"


class TestSyntheticNewsGenerator:
    """Unit tests verifying hermetic synthetic news replay stream generation."""

    def test_synthetic_stream_generation(self) -> None:
        generator = SyntheticNewsGenerator(seed=42)
        stream = generator.generate_batch(count=15)

        assert len(stream) == 15
        now = datetime.now(UTC)
        for article in stream:
            assert isinstance(article, NewsArticle)
            assert article.source == "synthetic"
            assert article.published_at <= now + timedelta(seconds=1)
            assert article.available_at >= article.published_at
            assert len(article.tickers) >= 1
            assert article.is_causal()
            assert len(article.article_hash) == 64

    def test_synthetic_event_distribution(self) -> None:
        generator = SyntheticNewsGenerator(seed=123)
        stream = generator.generate_batch(count=50)

        headlines = [a.headline.lower() for a in stream]
        has_earnings = any("earnings" in h or "revenue" in h or "profit" in h for h in headlines)
        has_fed = any("fed" in h or "rate" in h or "inflation" in h for h in headlines)
        has_acquisition = any(
            "acquisition" in h or "acquire" in h or "merger" in h or "deal" in h for h in headlines
        )

        assert has_earnings
        assert has_fed
        assert has_acquisition
