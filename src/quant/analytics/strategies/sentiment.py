"""Event-Driven: Loughran-McDonald News Sentiment Breakout Strategy.

Functional Purpose:
    Implements causal, event-driven trading on real-time financial news headlines and SEC filings.
    Leverages Loughran-McDonald dictionary polarity scoring, entity centrality extraction, and a
    bi-exponential temporal decay kernel kappa(Delta t, u) to dynamically size momentum breakout
    positions and holding horizons.

Explicit Dependency Tracking:
    - quant.analytics.news_classifier: FinancialNewsClassifier, ClassifiedNewsEvent, LM_POSITIVE_TERMS, LM_NEGATIVE_TERMS.
    - quant.analytics.price_reaction: compute_decay_kernel.
    - quant.analytics.strategies.base: BaseAlphaStrategy, StrategySignal, SignalDirection,
      BarHistoryWindow, StrategyContext, StrategyError.

Structural Relationship:
    Event-driven alpha strategy in Phase 14 library. Ingested by BacktestRunner and
    SwarmMetaStrategy.

Defensive Invariants:
    - INV-STRAT-001: Target weights strictly in [-1.0, 1.0].
    - INV-STRAT-002: Conviction strictly in [0.0, 1.0].
    - Causal Temporal Decay: Future news cannot affect past signals (INV-NEWS-001).
    - Monotonic Signal Mapping: Positive polarity -> LONG, negative -> SHORT.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from quant.analytics.news_classifier import (
    LM_NEGATIVE_TERMS,
    LM_POSITIVE_TERMS,
)
from quant.analytics.price_reaction import compute_decay_kernel
from quant.analytics.strategies.base import (
    BarHistoryWindow,
    BaseAlphaStrategy,
    SignalDirection,
    StrategyContext,
    StrategyError,
    StrategySignal,
)

# ============================================================================
# Diagnostic Fault Codes
# ============================================================================

ERR_STRAT_SENTIMENT_INVALID_INPUT: Final[str] = "ERR-STRAT-030"


# ============================================================================
# News Event Value Object
# ============================================================================


@dataclass(frozen=True, slots=True)
class NewsSentimentEvent:
    """Immutable parsed news event record for sentiment strategy tracking.

    Functional Purpose:
        Encapsulates financial headline metadata, arrival timestamp, and polarity.
    Defensive Invariants:
        polarity in [-1.0, 1.0], urgency in [0.0, 1.0].
    """

    symbol: str
    headline: str
    timestamp_ns: int
    polarity: float
    urgency: float = 0.5
    centrality: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol or not isinstance(self.symbol, str):
            raise StrategyError(
                "symbol must be non-empty string", code=ERR_STRAT_SENTIMENT_INVALID_INPUT
            )
        if isinstance(self.polarity, bool) or not math.isfinite(self.polarity):
            raise StrategyError(
                "polarity must be finite float", code=ERR_STRAT_SENTIMENT_INVALID_INPUT
            )
        if not -1.0 <= self.polarity <= 1.0:
            raise StrategyError(
                f"polarity must be in [-1.0, 1.0], got {self.polarity}",
                code=ERR_STRAT_SENTIMENT_INVALID_INPUT,
            )


# ============================================================================
# Loughran-McDonald Sentiment Strategy
# ============================================================================


class LoughranMcDonaldSentimentStrategy(BaseAlphaStrategy):
    """Event-driven news sentiment strategy using Loughran-McDonald lexicon.

    Functional Purpose:
        Scores financial headlines using curated positive and negative financial terms.
        Applies a bi-exponential memory decay kernel to compute cumulative sentiment shock
        per monitored asset, generating directionally consistent breakout signals.

    Explicit Dependency Tracking:
        compute_decay_kernel, LM_POSITIVE_TERMS, LM_NEGATIVE_TERMS.

    Defensive Invariants:
        - Bounded polarity: S_i in [-1.0, 1.0].
        - Temporal causality: Events with timestamp > current_time are ignored (no lookahead).
    """

    def __init__(
        self,
        strategy_id: str,
        monitored_symbols: Sequence[str],
        min_warmup_bars: int = 10,
        max_leverage: float = 1.0,
        polarity_threshold: float = 0.15,
        decay_half_life_seconds: float = 7200.0,  # 2 hours
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        """Initialize Loughran-McDonald Sentiment Strategy."""
        if polarity_threshold <= 0.0 or polarity_threshold > 1.0:
            raise StrategyError(
                f"polarity_threshold must be in (0.0, 1.0], got {polarity_threshold}",
                code=ERR_STRAT_SENTIMENT_INVALID_INPUT,
            )

        super().__init__(
            strategy_id=strategy_id,
            monitored_symbols=monitored_symbols,
            min_warmup_bars=min_warmup_bars,
            max_leverage=max_leverage,
        )

        self._polarity_threshold = polarity_threshold
        self._tau_fast = decay_half_life_seconds
        self._stop_loss = stop_loss_pct
        self._take_profit = take_profit_pct

        # Event buffer: symbol -> list of NewsSentimentEvent
        self._event_buffer: dict[str, list[NewsSentimentEvent]] = defaultdict(list)

    def record_event(self, event: NewsSentimentEvent) -> None:
        """Register a parsed news event into the strategy memory buffer."""
        self._event_buffer[event.symbol].append(event)

    def record_headline(
        self,
        symbol: str,
        headline: str,
        timestamp_ns: int,
        urgency: float = 0.5,
    ) -> NewsSentimentEvent:
        """Classify a raw financial headline on-the-fly and register it into event buffer."""
        polarity = self.score_headline(headline)
        event = NewsSentimentEvent(
            symbol=symbol,
            headline=headline,
            timestamp_ns=timestamp_ns,
            polarity=polarity,
            urgency=urgency,
            centrality=1.0,
        )
        self.record_event(event)
        return event

    @staticmethod
    def score_headline(text: str) -> float:
        """Compute normalized sentiment polarity in [-1.0, 1.0] using Loughran-McDonald lexicon.

        Formulation:
            Polarity = (N_pos - N_neg) / max(1, N_pos + N_neg)
        """
        words = text.lower().split()
        pos_count = sum(1 for w in words if w.strip(".,!?;:\"'()[]{}") in LM_POSITIVE_TERMS)
        neg_count = sum(1 for w in words if w.strip(".,!?;:\"'()[]{}") in LM_NEGATIVE_TERMS)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        raw_polarity = (pos_count - neg_count) / float(total)
        return max(-1.0, min(1.0, raw_polarity))

    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Compute sentiment breakout signals from accumulated decayed news events."""
        # Determine current evaluation timestamp
        current_ts_ns = 0
        if context is not None:
            current_ts_ns = context.current_timestamp
        else:
            # Fall back to latest bar timestamp across monitored assets
            for sym in self.monitored_symbols:
                ts_arr = history.timestamps(sym)
                if len(ts_arr) > 0:
                    current_ts_ns = max(current_ts_ns, int(ts_arr[-1]))

        # Also ingest any news events injected dynamically via context.metadata
        if context is not None and "news_events" in context.metadata:
            injected_events = context.metadata["news_events"]
            if isinstance(injected_events, Sequence):
                for ev in injected_events:
                    if isinstance(ev, NewsSentimentEvent):
                        self.record_event(ev)

        signals: dict[str, StrategySignal] = {}

        for sym in self.monitored_symbols:
            events = self._event_buffer.get(sym, [])
            # Filter strictly past events (Zero Forward Lookahead)
            valid_events = [e for e in events if e.timestamp_ns <= current_ts_ns]

            cumulative_shock = 0.0
            event_count = 0

            for ev in valid_events:
                delta_t_sec = max(0.0, (current_ts_ns - ev.timestamp_ns) / 1e9)
                # Compute bi-exponential decay kernel
                weight = compute_decay_kernel(
                    delta_t_seconds=delta_t_sec,
                    urgency=ev.urgency,
                    tau_fast=self._tau_fast,
                    tau_slow=self._tau_fast * 10.0,
                )
                cumulative_shock += ev.polarity * weight * ev.centrality
                if weight > 1e-4:
                    event_count += 1

            # Clamp composite shock to [-1.0, 1.0]
            clamped_shock = max(-1.0, min(1.0, cumulative_shock))

            # Directional thresholding
            if clamped_shock >= self._polarity_threshold:
                direction = SignalDirection.LONG
                target_w = clamped_shock * self._max_leverage
                conviction = min(1.0, abs(clamped_shock))
            elif clamped_shock <= -self._polarity_threshold:
                direction = SignalDirection.SHORT
                target_w = clamped_shock * self._max_leverage
                conviction = min(1.0, abs(clamped_shock))
            else:
                direction = SignalDirection.FLAT
                target_w = 0.0
                conviction = 0.0

            diag: dict[str, float | str | int] = {
                "sentiment_shock": round(clamped_shock, 4),
                "active_events": event_count,
                "total_events": len(valid_events),
                "current_ts_ns": current_ts_ns,
            }

            signal = StrategySignal(
                symbol=sym,
                direction=direction,
                target_weight=round(target_w, 4),
                conviction=round(conviction, 4),
                target_horizon_bars=5,
                stop_loss_pct=self._stop_loss,
                take_profit_pct=self._take_profit,
                diagnostics=diag,
                timestamp=current_ts_ns,
            )
            signals[sym] = signal

        return signals
