"""Financial sentiment classification and event taxonomy engine with Loughran-McDonald lexicon.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates, Rule 4: Closed-form formulations)
- Invariant 3: Bounded Tri-Axial Sentiment Normalization (INV-NEWS-003)
- Fault Codes: ERR-NEWS-004, ERR-NEWS-005
"""

import math
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from quant.data.news_harvester import COMPANY_TICKER_MAP, NewsArticle

# Diagnostic Fault Codes
ERR_NEWS_EMPTY_UNIVERSE = "ERR-NEWS-004"
ERR_NEWS_NON_FINITE_SIGNAL = "ERR-NEWS-005"


class NewsClassifierError(Exception):
    """Base exception for all news classification anomalies."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class EmptyUniverseException(NewsClassifierError):
    """Raised when extracted entities cannot be resolved to any traded assets (ERR-NEWS-004)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_NEWS_EMPTY_UNIVERSE, message)


class NonFiniteSignalException(NewsClassifierError):
    """Raised when non-finite, NaN, or boolean values are detected in sentiment signals (ERR-NEWS-005)."""

    def __init__(self, message: str) -> None:
        super().__init__(ERR_NEWS_NON_FINITE_SIGNAL, message)


class EventType(StrEnum):
    """Discrete financial event taxonomy classes."""

    EARNINGS = "EARNINGS"
    MACRO_FED = "MACRO_FED"
    M_AND_A = "M_AND_A"
    REGULATORY_LEGAL = "REGULATORY_LEGAL"
    ANALYST_ACTION = "ANALYST_ACTION"
    GENERAL_MARKET = "GENERAL_MARKET"


@dataclass(slots=True, frozen=True)
class ClassifiedNewsEvent:
    """Immutable classified news event with Loughran-McDonald sentiment and entity shocks.

    Purpose: Encapsulates domain taxonomy, tri-axial sentiment bounds, and asset-level shock tensors.
    Explicit Dependency Tracking: NewsArticle from data.news_harvester, EventType.
    Structural Relationship: Ingested by NewsPriceReactionEngine and EventService.
    Defensive Invariant: Polarity in [-1, 1], Urgency in [0, 1], Novelty in [0, 1] (INV-NEWS-003).
    """

    article: NewsArticle
    event_type: EventType
    polarity: float
    urgency: float
    novelty: float
    entity_centrality: dict[str, float]
    composite_shock: dict[str, float]


# Loughran-McDonald Curated Lexicon for Financial NLP
LM_POSITIVE_TERMS: frozenset[str] = frozenset(
    {
        "beat",
        "beats",
        "beating",
        "surged",
        "surging",
        "surge",
        "record",
        "profit",
        "profitable",
        "profitability",
        "boom",
        "booming",
        "growth",
        "growing",
        "gain",
        "gained",
        "gains",
        "exceed",
        "exceeded",
        "exceeds",
        "exceeding",
        "outperform",
        "outperformed",
        "outperforming",
        "rally",
        "rallied",
        "rallying",
        "top",
        "topped",
        "topping",
        "boost",
        "boosted",
        "boosting",
        "dividend",
        "upgrade",
        "upgraded",
        "upgrades",
        "upgrading",
        "expansion",
        "expanding",
        "accretive",
        "efficiency",
        "optimism",
        "bullish",
        "soared",
        "soaring",
        "soar",
        "acceleration",
        "accelerated",
        "accelerating",
        "achieve",
        "achieved",
        "achieving",
        "meet",
        "met",
        "meeting",
        "deliver",
        "delivered",
        "delivering",
        "approval",
        "approved",
        "approving",
        "success",
        "successful",
        "positive",
        "strong",
        "higher",
    }
)

LM_NEGATIVE_TERMS: frozenset[str] = frozenset(
    {
        "miss",
        "missed",
        "misses",
        "missing",
        "slump",
        "slumped",
        "slumping",
        "loss",
        "losses",
        "drop",
        "dropped",
        "dropping",
        "plunge",
        "plunged",
        "plunging",
        "fall",
        "falling",
        "fell",
        "decline",
        "declined",
        "declining",
        "declines",
        "plummet",
        "plummeted",
        "plummeting",
        "deficit",
        "debt",
        "default",
        "lawsuit",
        "sued",
        "fraud",
        "probe",
        "investigate",
        "investigation",
        "fine",
        "penalty",
        "penalties",
        "subpoena",
        "warning",
        "cut",
        "cuts",
        "cutting",
        "down",
        "lowered",
        "lowering",
        "slash",
        "slashed",
        "slashing",
        "downgrade",
        "downgraded",
        "downgrades",
        "weakness",
        "bearish",
        "recession",
        "inflation",
        "irregularities",
        "delinquency",
        "contraction",
        "shortfall",
        "fail",
        "failed",
        "failing",
        "failure",
        "failures",
        "weak",
        "lower",
        "negative",
    }
)

# General Negation Tokens
NEGATION_TOKENS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "none",
        "unable",
        "failed",
        "fail",
        "barely",
        "hardly",
        "without",
        "little",
    }
)

# Inversion Tokens where following negative is a positive improvement
INVERSION_TOKENS: frozenset[str] = frozenset(
    {
        "narrowed",
        "narrowing",
        "slowed",
        "slowing",
        "decreased",
        "decreasing",
        "diminished",
        "curbed",
        "eased",
        "easing",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")


class FinancialSentimentClassifier:
    """Loughran-McDonald financial sentiment classifier and entity-event taxonomy engine.

    Purpose: Converts unstructured news articles into bounded econometric shock tensors.
    Explicit Dependency Tracking: NewsArticle, ClassifiedNewsEvent, EventType.
    Structural Relationship: Intermediate layer feeding NewsPriceReactionEngine.
    Defensive Invariant: INV-NEWS-003: All outputs strictly bounded in normalized spaces.
    """

    def __init__(self, novelty_decay_horizon_seconds: float = 7200.0) -> None:
        self._novelty_decay_horizon = novelty_decay_horizon_seconds
        # In-memory history for novelty tracking: (article_hash, timestamp, headline_words)
        self._history: deque[tuple[str, datetime, set[str]]] = deque(maxlen=2000)

    def validate_signal_bounds(self, polarity: float, urgency: float, novelty: float) -> None:
        """Enforce Invariant 3: Reject non-finite, NaN, and boolean scalar values (INV-NEWS-003).

        Purpose: Defensive validation ensuring downstream numerical pipelines never encounter NaNs.
        Dependency: math.isfinite.
        Relationship: Pre-condition check for ClassifiedNewsEvent construction.
        Invariant: Raises NonFiniteSignalException on boolean or non-finite inputs.
        """
        for name, val in [("polarity", polarity), ("urgency", urgency), ("novelty", novelty)]:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise NonFiniteSignalException(
                    f"Non-finite or boolean signal detected for '{name}': {val!r}"
                )

        if not (-1.0001 <= polarity <= 1.0001):
            raise NonFiniteSignalException(f"Polarity out of range [-1, 1]: {polarity}")
        if not (-0.0001 <= urgency <= 1.0001):
            raise NonFiniteSignalException(f"Urgency out of range [0, 1]: {urgency}")
        if not (-0.0001 <= novelty <= 1.0001):
            raise NonFiniteSignalException(f"Novelty out of range [0, 1]: {novelty}")

    def score_sentiment(self, text: str) -> float:
        """Compute Loughran-McDonald sentiment polarity with 3-token negation and inversion window.

        Formula:
            s = (N_pos - N_neg) / (N_pos + N_neg + 1e-6) in [-1.0, 1.0]
        """
        tokens = _WORD_RE.findall(text.lower())
        if not tokens:
            return 0.0

        n_pos = 0.0
        n_neg = 0.0
        n = len(tokens)

        for i, word in enumerate(tokens):
            # Check 2-token preceding window for negators
            preceding = tokens[max(0, i - 2) : i]
            is_negated = any(p in NEGATION_TOKENS for p in preceding)
            is_inverted = any(p in INVERSION_TOKENS for p in preceding)

            # Check immediate subsequent word for inversion (e.g. "loss narrowed")
            following = tokens[i + 1 : min(n, i + 3)]
            is_followed_by_inversion = any(f in INVERSION_TOKENS for f in following)

            if word in LM_POSITIVE_TERMS:
                if is_negated:
                    n_neg += 1.0
                else:
                    n_pos += 1.0
            elif word in LM_NEGATIVE_TERMS:
                if is_negated or is_inverted or is_followed_by_inversion:
                    # e.g. "loss narrowed" or "not dropping" counts positively
                    n_pos += 0.8
                else:
                    n_neg += 1.0

        if n_pos + n_neg == 0:
            return 0.0

        raw_polarity = (n_pos - n_neg) / (n_pos + n_neg + 1e-6)
        return float(min(max(raw_polarity, -1.0), 1.0))

    def detect_event_type(
        self, headline: str, summary: str, source: str
    ) -> tuple[EventType, float]:
        """Classify event taxonomy and assign category base urgency."""
        combined = f"{headline} {summary}".lower()

        if "8-k" in headline.lower() or source == "sec_edgar":
            return EventType.REGULATORY_LEGAL, 0.95

        # Macro / Fed keywords
        macro_patterns = [
            "fed",
            "federal reserve",
            "fomc",
            "rate cut",
            "rate hike",
            "cpi",
            "inflation",
            "payroll",
            "gdp",
        ]
        if any(p in combined for p in macro_patterns):
            return EventType.MACRO_FED, 0.85

        # M&A keywords
        ma_patterns = ["acquire", "acquires", "acquisition", "merger", "buyout", "takeover", "deal"]
        if any(p in combined for p in ma_patterns):
            return EventType.M_AND_A, 0.85

        # Regulatory / Legal
        reg_patterns = [
            "sec",
            "doj",
            "investigation",
            "probe",
            "lawsuit",
            "antitrust",
            "fraud",
            "subpoena",
        ]
        if any(p in combined for p in reg_patterns):
            return EventType.REGULATORY_LEGAL, 0.70

        # Analyst Action
        analyst_patterns = [
            "upgrade",
            "upgrades",
            "upgraded",
            "downgrade",
            "downgrades",
            "downgraded",
            "price target",
        ]
        if any(p in combined for p in analyst_patterns):
            return EventType.ANALYST_ACTION, 0.60

        # Earnings keywords
        earnings_patterns = [
            "earnings",
            "revenue",
            "profit",
            "net income",
            "gross margin",
            "guidance",
            "eps",
            "deliveries",
            "quarterly",
        ]
        if any(p in combined for p in earnings_patterns):
            return EventType.EARNINGS, 0.90

        return EventType.GENERAL_MARKET, 0.30

    def compute_novelty(self, article: NewsArticle) -> float:
        """Compute information novelty score H in [0.1, 1.0] based on story repetition."""
        words = set(_WORD_RE.findall(article.headline.lower()))
        if not words:
            return 1.0

        max_similarity = 0.0
        now = article.available_at

        for prev_hash, prev_time, prev_words in self._history:
            if prev_hash == article.article_hash:
                return 0.1
            dt = (now - prev_time).total_seconds()
            if dt < self._novelty_decay_horizon:
                intersection = len(words & prev_words)
                union = len(words | prev_words)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard > max_similarity:
                    max_similarity = jaccard

        self._history.append((article.article_hash, article.available_at, words))
        # Decay novelty if highly similar story appeared recently
        novelty = math.exp(-2.0 * max_similarity)
        return float(min(max(novelty, 0.1), 1.0))

    def resolve_centrality(
        self,
        article: NewsArticle,
        event_type: EventType,
        fallback_to_macro_proxy: bool = True,
    ) -> dict[str, float]:
        """Compute continuous entity centrality weights c_{i, k} in [0.0, 1.0]."""
        centrality_map: dict[str, float] = {}
        headline_upper = article.headline.upper()
        summary_upper = article.summary.upper()

        # Known tickers from article or extracted
        all_candidates: set[str] = set(article.tickers)

        # Check company aliases
        ticker_aliases: dict[str, list[str]] = {}
        for company, ticker in COMPANY_TICKER_MAP.items():
            ticker_aliases.setdefault(ticker, []).append(company)
            if company in headline_upper or company in summary_upper:
                all_candidates.add(ticker)

        for ticker in all_candidates:
            aliases = ticker_aliases.get(ticker, [])
            in_headline = (ticker in headline_upper) or any(a in headline_upper for a in aliases)
            in_summary = (ticker in summary_upper) or any(a in summary_upper for a in aliases)

            if in_headline:
                centrality_map[ticker] = 1.0
            elif in_summary:
                centrality_map[ticker] = 0.5
            else:
                centrality_map[ticker] = 0.3

        if not centrality_map:
            if event_type == EventType.MACRO_FED:
                centrality_map["SPY"] = 0.8
                centrality_map["QQQ"] = 0.8
            elif fallback_to_macro_proxy:
                centrality_map["SPY"] = 0.5
            else:
                raise EmptyUniverseException(
                    f"No resolvable entity tickers found in article '{article.headline}'."
                )

        return centrality_map

    def classify(
        self,
        article: NewsArticle,
        fallback_to_macro_proxy: bool = True,
    ) -> ClassifiedNewsEvent:
        """Classify article into domain taxonomy, score LM sentiment, and calculate shock tensor."""
        combined_text = f"{article.headline}. {article.summary}"
        polarity = self.score_sentiment(combined_text)
        event_type, urgency = self.detect_event_type(
            article.headline, article.summary, article.source
        )
        novelty = self.compute_novelty(article)

        self.validate_signal_bounds(polarity=polarity, urgency=urgency, novelty=novelty)

        centrality = self.resolve_centrality(
            article, event_type, fallback_to_macro_proxy=fallback_to_macro_proxy
        )

        # Composite shock tensor for each asset: S_{i, k} = Polarity * Urgency * Novelty * Centrality
        composite_shock: dict[str, float] = {}
        for ticker, c_val in centrality.items():
            shock = polarity * urgency * novelty * c_val
            composite_shock[ticker] = float(min(max(shock, -1.0), 1.0))

        return ClassifiedNewsEvent(
            article=article,
            event_type=event_type,
            polarity=polarity,
            urgency=urgency,
            novelty=novelty,
            entity_centrality=centrality,
            composite_shock=composite_shock,
        )
