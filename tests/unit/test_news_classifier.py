"""Unit tests for Phase 9 financial sentiment classifier and event taxonomy engine.

Governing Standards:
- Rules.md (Rule 1, Rule 2, Rule 3, Rule 4)
- INV-NEWS-003: Bounded Tri-Axial Sentiment Normalization
- Fault Codes: ERR-NEWS-004, ERR-NEWS-005
"""

from datetime import UTC, datetime

import pytest

from quant.analytics.news_classifier import (
    EmptyUniverseException,
    EventType,
    FinancialSentimentClassifier,
    NonFiniteSignalException,
)
from quant.data.news_harvester import NewsArticle


def create_mock_article(
    headline: str,
    summary: str = "",
    source: str = "wire",
    tickers: tuple[str, ...] = (),
) -> NewsArticle:
    now = datetime.now(UTC)
    return NewsArticle(
        headline=headline,
        summary=summary,
        url="https://mock.news/test",
        source=source,
        published_at=now,
        available_at=now,
        tickers=tickers,
    )


class TestFinancialSentimentClassifier:
    """Unit tests for Loughran-McDonald sentiment analysis, negation, and entity centrality."""

    def test_loughran_mcdonald_positive_sentiment(self) -> None:
        classifier = FinancialSentimentClassifier()
        article = create_mock_article(
            headline="Apple Q4 earnings beat estimates with record revenue and surging gross margins",
            summary="Strong demand drove exceptional net profit and dividend boost.",
            tickers=("AAPL",),
        )

        classified = classifier.classify(article)
        assert classified.event_type == EventType.EARNINGS
        assert classified.polarity > 0.4
        assert classified.polarity <= 1.0
        assert classified.urgency >= 0.85
        assert classified.novelty >= 0.1
        assert "AAPL" in classified.entity_centrality
        assert classified.entity_centrality["AAPL"] == 1.0  # Ticker in headline
        assert classified.composite_shock["AAPL"] > 0.0

    def test_loughran_mcdonald_negative_sentiment(self) -> None:
        classifier = FinancialSentimentClassifier()
        article = create_mock_article(
            headline="Tesla slashes vehicle forecast as deliveries miss expectations and revenue plummets",
            summary="Weak operating income and mounting inventory losses hit full-year outlook.",
            tickers=("TSLA",),
        )

        classified = classifier.classify(article)
        assert classified.event_type == EventType.EARNINGS
        assert classified.polarity < -0.4
        assert classified.polarity >= -1.0
        assert classified.composite_shock["TSLA"] < 0.0

    def test_financial_negation_loss_narrowed_is_positive(self) -> None:
        classifier = FinancialSentimentClassifier()
        # "loss narrowed" is financially positive (narrowing of negative)
        article = create_mock_article(
            headline="Rivian net loss narrowed significantly as manufacturing efficiency improved",
            summary="Electric vehicle maker reported smaller operating deficit and steady growth.",
            tickers=("RIVN",),
        )

        classified = classifier.classify(article)
        assert classified.polarity > 0.0  # Positive due to negation of loss

    def test_financial_negation_failed_to_deliver_is_negative(self) -> None:
        classifier = FinancialSentimentClassifier()
        article = create_mock_article(
            headline="Biotech firm failed to achieve primary endpoint in Phase 3 clinical trial",
            summary="Company did not meet statistical efficacy criteria.",
        )

        classified = classifier.classify(article)
        assert classified.polarity < 0.0

    def test_event_taxonomy_classification(self) -> None:
        classifier = FinancialSentimentClassifier()

        # Macro / Fed
        macro_art = create_mock_article(
            headline="Federal Reserve signals interest rate cuts as CPI inflation cools to 2.3%",
            summary="FOMC statement shows monetary easing on track.",
        )
        macro_evt = classifier.classify(macro_art)
        assert macro_evt.event_type == EventType.MACRO_FED
        assert "SPY" in macro_evt.entity_centrality  # Proxy mapping for macro

        # M&A
        ma_art = create_mock_article(
            headline="NVIDIA to acquire AI chip startup for $5 billion in cash and stock deal",
            summary="Strategic acquisition strengthens high-bandwidth interconnect architecture.",
            tickers=("NVDA",),
        )
        ma_evt = classifier.classify(ma_art)
        assert ma_evt.event_type == EventType.M_AND_A
        assert ma_evt.urgency >= 0.85

        # Regulatory / Legal
        reg_art = create_mock_article(
            headline="SEC launches formal investigation into accounting irregularities and disclosure fraud",
            summary="Department of Justice files antitrust lawsuit.",
        )
        reg_evt = classifier.classify(reg_art)
        assert reg_evt.event_type == EventType.REGULATORY_LEGAL

        # Analyst Action
        analyst_art = create_mock_article(
            headline="Goldman Sachs upgrades Microsoft to Conviction Buy with raised price target",
            summary="Analyst notes accelerating enterprise AI copilot adoption.",
            tickers=("MSFT",),
        )
        analyst_evt = classifier.classify(analyst_art)
        assert analyst_evt.event_type == EventType.ANALYST_ACTION

    def test_entity_centrality_weighting(self) -> None:
        classifier = FinancialSentimentClassifier()
        # AAPL in headline, MSFT only in summary
        article = create_mock_article(
            headline="Apple announces breakthrough silicon chips",
            summary="The new processors outpace competing systems from Microsoft in server benchmarks.",
            tickers=("AAPL", "MSFT"),
        )

        classified = classifier.classify(article)
        assert classified.entity_centrality["AAPL"] == 1.0  # Headline centrality
        assert classified.entity_centrality["MSFT"] == 0.5  # Summary context centrality

    def test_tri_axial_bounds_invariant_inv_news_003(self) -> None:
        classifier = FinancialSentimentClassifier()
        article = create_mock_article(
            headline="General corporate announcement regarding upcoming shareholder meeting",
            summary="No financial results discussed.",
        )

        classified = classifier.classify(article)
        assert -1.0 <= classified.polarity <= 1.0
        assert 0.0 <= classified.urgency <= 1.0
        assert 0.0 <= classified.novelty <= 1.0
        for c in classified.entity_centrality.values():
            assert 0.0 <= c <= 1.0
        for s in classified.composite_shock.values():
            assert -1.0 <= s <= 1.0

    def test_non_finite_input_rejection(self) -> None:
        classifier = FinancialSentimentClassifier()
        # Verify validation rejects non-finite or boolean numbers
        with pytest.raises(NonFiniteSignalException) as exc_info:
            classifier.validate_signal_bounds(polarity=float("nan"), urgency=0.5, novelty=0.8)
        assert exc_info.value.code == "ERR-NEWS-005"

        with pytest.raises(NonFiniteSignalException) as exc_info2:
            classifier.validate_signal_bounds(polarity=0.5, urgency=float("inf"), novelty=0.8)
        assert exc_info2.value.code == "ERR-NEWS-005"

        with pytest.raises(NonFiniteSignalException) as exc_info3:
            classifier.validate_signal_bounds(polarity=True, urgency=0.5, novelty=0.8)  # type: ignore[arg-type]
        assert exc_info3.value.code == "ERR-NEWS-005"

    def test_empty_universe_resolution_fallback(self) -> None:
        classifier = FinancialSentimentClassifier()
        # Headline with unknown obscure entity
        article = create_mock_article(
            headline="Local bakery co-op expands neighborhood delivery operations",
            summary="Family business opens a second branch in town.",
        )
        # When no ticker can be resolved and not macro, empty universe exception or fallback to SPY
        classified = classifier.classify(article, fallback_to_macro_proxy=True)
        assert "SPY" in classified.entity_centrality

        with pytest.raises(EmptyUniverseException) as exc_info:
            classifier.classify(article, fallback_to_macro_proxy=False)
        assert exc_info.value.code == "ERR-NEWS-004"
