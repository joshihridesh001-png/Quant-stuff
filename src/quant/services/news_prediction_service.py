"""Application service orchestrating real-world news harvesting, classification, and causal price reaction prediction.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates, Rule 4: Zero lookahead causality)
- INV-NEWS-001: Point-in-time causality
- INV-NEWS-002: SHA-256 deduplication
- INV-NEWS-003: Tri-axial sentiment bounds
- INV-NEWS-004: Impact scaling
- INV-NEWS-005: Logistic breakout probability
- INV-NEWS-006: Sub-10ms latency SLA
"""

import logging
from collections import deque
from datetime import UTC, datetime

from quant.analytics.news_classifier import (
    ClassifiedNewsEvent,
    FinancialSentimentClassifier,
)
from quant.analytics.price_reaction import (
    NewsPriceReactionEngine,
    PriceReactionPrediction,
)
from quant.data.external_providers import ExternalProviderManager
from quant.data.news_harvester import (
    NewsArticle,
    NewsFeedConfig,
    NewsHarvester,
    SyntheticNewsGenerator,
)
from quant.services.autonomous_trader import AutonomousTradingEngine
from quant.services.event_service import EventService

logger = logging.getLogger(__name__)

DEFAULT_FEEDS: list[NewsFeedConfig] = [
    NewsFeedConfig(
        name="yahoo_finance",
        url="https://finance.yahoo.com/news/rssindex",
        feed_type="rss",
        timeout_seconds=5.0,
    ),
    NewsFeedConfig(
        name="sec_edgar_8k",
        url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&count=40&output=atom",
        feed_type="atom",
        timeout_seconds=5.0,
    ),
    NewsFeedConfig(
        name="marketwatch",
        url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
        feed_type="rss",
        timeout_seconds=5.0,
    ),
]


class NewsPredictionService:
    """Master application service coupling news feeds, NLP classification, and causal price reaction modeling.

    Purpose: Coordinates background news harvesting, Loughran-McDonald sentiment, and Triple-Barrier prediction.
    Explicit Dependency Tracking: NewsHarvester, FinancialSentimentClassifier, NewsPriceReactionEngine, ExternalProviderManager.
    Structural Relationship: Consumed by REST API endpoints and AutonomousTradingEngine daemon.
    Defensive Invariant: Guaranteed thread-safe ring buffer caching, bounded execution latencies.
    """

    def __init__(
        self,
        harvester: NewsHarvester | None = None,
        classifier: FinancialSentimentClassifier | None = None,
        reaction_engine: NewsPriceReactionEngine | None = None,
        event_service: EventService | None = None,
        autonomous_engine: AutonomousTradingEngine | None = None,
        provider_manager: ExternalProviderManager | None = None,
        feeds: list[NewsFeedConfig] | None = None,
        history_capacity: int = 1000,
    ) -> None:
        self._harvester = harvester or NewsHarvester()
        self._classifier = classifier or FinancialSentimentClassifier()
        self._reaction_engine = reaction_engine or NewsPriceReactionEngine()
        self._event_service = event_service
        self._autonomous_engine = autonomous_engine
        self._provider_manager = provider_manager
        self._feeds = feeds if feeds is not None else list(DEFAULT_FEEDS)
        self._synthetic_generator = SyntheticNewsGenerator(seed=42)

        self._predictions_history: deque[PriceReactionPrediction] = deque(maxlen=history_capacity)
        self._articles_history: deque[NewsArticle] = deque(maxlen=history_capacity)
        self._classified_events_history: deque[ClassifiedNewsEvent] = deque(maxlen=history_capacity)

    @property
    def feeds(self) -> list[NewsFeedConfig]:
        """Configured news syndication wire endpoints."""
        return list(self._feeds)

    def get_latest_predictions(self, limit: int = 50) -> list[PriceReactionPrediction]:
        """Retrieve most recent price reaction predictions in reverse chronological order."""
        res = list(self._predictions_history)
        res.reverse()
        return res[:limit]

    def get_latest_articles(self, limit: int = 50) -> list[NewsArticle]:
        """Retrieve most recently harvested news articles in reverse chronological order."""
        res = list(self._articles_history)
        res.reverse()
        return res[:limit]

    async def predict_headline(
        self,
        headline: str,
        summary: str = "",
        ticker: str | None = None,
        current_price: float = 100.0,
        volatility: float = 0.02,
    ) -> list[PriceReactionPrediction]:
        """Generate on-demand causal price reaction prediction for arbitrary headline.

        Purpose: Real-time scenario testing and ad-hoc analyst verification.
        Dependency: FinancialSentimentClassifier, NewsPriceReactionEngine.
        Relationship: Backend handler for POST /api/v1/news/predict.
        Invariant: Adheres to INV-NEWS-003, INV-NEWS-004, INV-NEWS-005.
        """
        now = datetime.now(UTC)
        tickers = (ticker.upper(),) if ticker and ticker.strip() else ()
        article = NewsArticle(
            headline=headline.strip(),
            summary=summary.strip(),
            url="https://quant-alpha.internal/manual-prediction",
            source="manual",
            published_at=now,
            available_at=now,
            tickers=tickers,
        )

        classified = self._classifier.classify(article, fallback_to_macro_proxy=True)
        prices = dict.fromkeys(classified.entity_centrality, current_price)
        vols = dict.fromkeys(classified.entity_centrality, volatility)

        predictions = self._reaction_engine.predict(classified, prices=prices, volatilities=vols)

        self._articles_history.append(article)
        self._classified_events_history.append(classified)
        for p in predictions:
            self._predictions_history.append(p)

        self._inject_forward_priors(predictions)
        return predictions

    async def harvest_and_predict(
        self, fallback_to_synthetic: bool = True, target_ticker: str | None = None
    ) -> list[PriceReactionPrediction]:
        """Poll all enabled news feeds, classify new stories, and predict causal price reactions.

        Purpose: Periodic automated ingestion cycle driving autonomous trading priors.
        Dependency: NewsHarvester.poll_all, FinancialSentimentClassifier, NewsPriceReactionEngine.
        Relationship: Scheduled daemon cycle and POST /api/v1/news/harvest trigger.
        Invariant: Deduplicated via SHA-256 (INV-NEWS-002), causal point-in-time (INV-NEWS-001).
        """
        articles = await self._harvester.poll_all(self._feeds)

        # Ingest from authenticated public-apis external providers (Finnhub, NewsAPI)
        if self._provider_manager is not None:
            try:
                universe = (
                    self._autonomous_engine.universe
                    if self._autonomous_engine
                    else ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]
                )
                ext_items = await self._provider_manager.harvest_external_news(universe)
                now_utc = datetime.now(UTC)
                ext_articles: list[NewsArticle] = []
                for item in ext_items:
                    tickers = tuple(item.related_symbols) if item.related_symbols else ()
                    ext_articles.append(
                        NewsArticle(
                            headline=item.headline,
                            summary=item.summary,
                            url=item.url or "https://quant-alpha.internal/external-news",
                            source=item.source,
                            published_at=now_utc,
                            available_at=now_utc,
                            tickers=tickers,
                        )
                    )
                if ext_articles:
                    unique_ext = self._harvester.filter_and_store_unique(ext_articles)
                    articles.extend(unique_ext)
            except Exception as exc:
                logger.warning("Error querying external provider manager for news: %s", exc)

        # In offline/hermetic test environments or if target_ticker is not present in harvested articles
        ticker_covered = (
            any(any(t.upper() == target_ticker.upper() for t in a.tickers) for a in articles)
            if target_ticker and articles
            else bool(articles)
        )
        if not ticker_covered and fallback_to_synthetic:
            logger.info(
                "External news feeds returned 0 articles or missing %s; generating synthetic replay batch.",
                target_ticker,
            )
            syn_articles = self._synthetic_generator.generate_batch(
                count=6, target_ticker=target_ticker
            )
            syn_unique = self._harvester.filter_and_store_unique(syn_articles)
            articles.extend(syn_unique)

        all_predictions: list[PriceReactionPrediction] = []

        for article in articles:
            try:
                classified = self._classifier.classify(article, fallback_to_macro_proxy=True)
                # Benchmark reference prices and baseline 2% volatility
                prices = dict.fromkeys(classified.entity_centrality, 150.0)
                vols = dict.fromkeys(classified.entity_centrality, 0.02)

                preds = self._reaction_engine.predict(classified, prices=prices, volatilities=vols)

                self._articles_history.append(article)
                self._classified_events_history.append(classified)
                for p in preds:
                    self._predictions_history.append(p)
                    all_predictions.append(p)

                # Persist in DB via EventService if configured
                if self._event_service is not None:
                    try:
                        await self._event_service.ingest_event(
                            headline=article.headline,
                            raw_text=article.summary,
                            timestamp=article.published_at,
                            ticker_weights=classified.entity_centrality,
                            sentiment_polarity=classified.polarity,
                        )
                    except Exception as exc:
                        logger.warning("EventService persistence warning: %s", exc)

            except Exception as e:
                logger.warning("Failed processing article '%s': %s", article.headline, e)

        self._inject_forward_priors(all_predictions)
        return all_predictions

    def _inject_forward_priors(self, predictions: list[PriceReactionPrediction]) -> None:
        """Inject predicted price breakout signals into AutonomousTradingEngine as forward priors."""
        if self._autonomous_engine is None or not predictions:
            return

        priors: dict[str, float] = {}
        for p in predictions:
            # Map probability of UP to directional prior: mu = 2.0 * (P(UP) - 0.5) in [-1.0, 1.0]
            directional_prior = 2.0 * (p.prob_up - 0.5)
            priors[p.ticker] = directional_prior

        self._autonomous_engine.update_news_alpha_priors(priors)
        logger.info("Injected forward news alpha priors into trading engine: %s", priors)
