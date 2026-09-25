"""Unit tests for Event-Driven Loughran-McDonald Sentiment Strategy.

Functional Purpose:
    Verifies financial lexicon scoring, temporal decay kernel integration,
    zero forward lookahead event causality, and directionally consistent signal emissions.

Explicit Dependency Tracking:
    - pytest, numpy.
    - quant.analytics.strategies.sentiment: LoughranMcDonaldSentimentStrategy, NewsSentimentEvent.
    - quant.analytics.strategies.base: BarHistoryWindow, SignalDirection, StrategyContext.
"""

from __future__ import annotations

import math

import numpy as np

from quant.analytics.strategies.base import (
    BarHistoryWindow,
    SignalDirection,
    StrategyContext,
)
from quant.analytics.strategies.sentiment import (
    LoughranMcDonaldSentimentStrategy,
)


def _make_dummy_history(symbol: str, n_bars: int = 20) -> BarHistoryWindow:
    """Helper creating basic bar history window for warmup satisfaction."""
    ts_base = 1_700_000_000_000_000_000
    timestamps = np.array([ts_base + i * 86_400_000_000_000 for i in range(n_bars)])
    closes = np.full(n_bars, 100.0, dtype=np.float64)
    return BarHistoryWindow(
        data={symbol: {"close": closes, "timestamp": timestamps}},
        symbols=(symbol,),
    )


def test_loughran_mcdonald_headline_polarity_scoring() -> None:
    """Verify dictionary polarity scoring on bullish, bearish, and neutral headlines."""
    # Bullish terms: beat, surges, profit
    pos_score = LoughranMcDonaldSentimentStrategy.score_headline(
        "Nvidia beats revenue expectations and surges to record profit growth"
    )
    assert pos_score > 0.5

    # Bearish terms: loss, decline, default
    neg_score = LoughranMcDonaldSentimentStrategy.score_headline(
        "Company reports massive loss, default warning, and declining sales"
    )
    assert neg_score < -0.5

    # Neutral / non-financial terms
    neutral_score = LoughranMcDonaldSentimentStrategy.score_headline(
        "Market opens at nine thirty am local time"
    )
    assert neutral_score == 0.0


def test_sentiment_strategy_long_and_short_signals() -> None:
    """Verify positive headlines generate LONG signals and negative generate SHORT."""
    strat = LoughranMcDonaldSentimentStrategy(
        strategy_id="sent_test",
        monitored_symbols=["AAPL", "TSLA"],
        min_warmup_bars=10,
        polarity_threshold=0.2,
    )

    t_now = 1_700_000_000_000_000_000  # Evaluation timestamp in ns
    # AAPL: Bullish headline 5 minutes ago
    strat.record_headline(
        symbol="AAPL",
        headline="Apple beats earnings and initiates massive stock dividend growth",
        timestamp_ns=t_now - int(300 * 1e9),
        urgency=0.8,
    )

    # TSLA: Bearish headline 10 minutes ago
    strat.record_headline(
        symbol="TSLA",
        headline="Tesla faces lawsuit, regulatory breach, and heavy operating loss",
        timestamp_ns=t_now - int(600 * 1e9),
        urgency=0.8,
    )

    history = _make_dummy_history("AAPL", 15)
    # Combine history for both symbols
    history = BarHistoryWindow(
        data={
            "AAPL": history.data["AAPL"],
            "TSLA": history.data["AAPL"],
        },
        symbols=("AAPL", "TSLA"),
    )

    context = StrategyContext(
        current_timestamp=t_now,
        current_prices={"AAPL": 150.0, "TSLA": 200.0},
        current_positions={"AAPL": 0.0, "TSLA": 0.0},
        total_equity=100000.0,
        unencumbered_cash=100000.0,
    )

    signals = strat.compute_signals(history, context)

    sig_aapl = signals["AAPL"]
    sig_tsla = signals["TSLA"]

    # AAPL -> LONG
    assert sig_aapl.direction == SignalDirection.LONG
    assert sig_aapl.target_weight > 0.0
    assert sig_aapl.conviction > 0.3

    # TSLA -> SHORT
    assert sig_tsla.direction == SignalDirection.SHORT
    assert sig_tsla.target_weight < 0.0
    assert sig_tsla.conviction > 0.3


def test_sentiment_strategy_temporal_decay_to_flat() -> None:
    """Verify that an ancient news event decays below threshold and results in FLAT."""
    strat = LoughranMcDonaldSentimentStrategy(
        strategy_id="sent_decay",
        monitored_symbols=["MSFT"],
        min_warmup_bars=10,
        polarity_threshold=0.15,
        decay_half_life_seconds=3600.0,  # 1 hour decay
    )

    t_now = 1_700_000_000_000_000_000
    # Old news event from 5 days ago (432,000 seconds ago)
    strat.record_headline(
        symbol="MSFT",
        headline="Microsoft beats profit and surges",
        timestamp_ns=t_now - int(432000 * 1e9),
    )

    history = _make_dummy_history("MSFT", 15)
    context = StrategyContext(
        current_timestamp=t_now,
        current_prices={"MSFT": 350.0},
        current_positions={"MSFT": 0.0},
        total_equity=100000.0,
        unencumbered_cash=100000.0,
    )

    signals = strat.compute_signals(history, context)
    sig_msft = signals["MSFT"]

    # Should have decayed to FLAT
    assert sig_msft.direction == SignalDirection.FLAT
    assert math.isclose(sig_msft.target_weight, 0.0)


def test_sentiment_strategy_zero_forward_lookahead() -> None:
    """Verify future news events (timestamp > current_ts) are strictly ignored."""
    strat = LoughranMcDonaldSentimentStrategy(
        strategy_id="sent_future",
        monitored_symbols=["NVDA"],
        min_warmup_bars=10,
    )

    t_now = 1_700_000_000_000_000_000
    # Event in the future (t_now + 1 hour)
    strat.record_headline(
        symbol="NVDA",
        headline="Nvidia surges on record breaking guidance",
        timestamp_ns=t_now + int(3600 * 1e9),
    )

    history = _make_dummy_history("NVDA", 15)
    context = StrategyContext(
        current_timestamp=t_now,
        current_prices={"NVDA": 500.0},
        current_positions={"NVDA": 0.0},
        total_equity=100000.0,
        unencumbered_cash=100000.0,
    )

    signals = strat.compute_signals(history, context)
    sig_nvda = signals["NVDA"]

    # Future event must NOT trigger signal
    assert sig_nvda.direction == SignalDirection.FLAT
    assert sig_nvda.target_weight == 0.0
    assert sig_nvda.diagnostics["total_events"] == 0
