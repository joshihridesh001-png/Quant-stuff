"""Unit tests for Phase 9 causal price reaction prediction engine.

Governing Standards:
- Rules.md (Rule 1, Rule 2, Rule 3, Rule 4)
- Invariant 4: Price Impact & Elasticity Scaling (INV-NEWS-004)
- Invariant 5: Monotonic Barrier Breakout Probability (INV-NEWS-005)
- Invariant 6: Sub-10ms Pipeline Latency SLA (INV-NEWS-006)
- Fault Codes: ERR-NEWS-005, ERR-NEWS-006
"""

import math
import time
from datetime import UTC, datetime

import pytest

from quant.analytics.news_classifier import (
    ClassifiedNewsEvent,
    EventType,
    NonFiniteSignalException,
)
from quant.analytics.price_reaction import (
    NewsPriceReactionEngine,
)
from quant.data.news_harvester import NewsArticle


def create_sample_event(
    event_type: EventType,
    polarity: float,
    urgency: float,
    ticker: str = "AAPL",
    centrality: float = 1.0,
) -> ClassifiedNewsEvent:
    now = datetime.now(UTC)
    article = NewsArticle(
        headline=f"Sample News headline for {ticker}",
        summary="Detailed financial commentary and operational overview.",
        url="https://sample.news",
        source="test_wire",
        published_at=now,
        available_at=now,
        tickers=(ticker,),
    )
    novelty = 1.0
    shock = polarity * urgency * novelty * centrality
    return ClassifiedNewsEvent(
        article=article,
        event_type=event_type,
        polarity=polarity,
        urgency=urgency,
        novelty=novelty,
        entity_centrality={ticker: centrality},
        composite_shock={ticker: shock},
    )


class TestNewsPriceReactionEngine:
    """Unit tests for price reaction, logistic breakout probabilities, and decay trajectories."""

    def test_expected_dollar_move_scaling_invariant_inv_news_004(self) -> None:
        engine = NewsPriceReactionEngine()
        event = create_sample_event(
            event_type=EventType.EARNINGS,
            polarity=0.8,
            urgency=0.9,
            ticker="AAPL",
            centrality=1.0,
        )

        current_price = 200.0
        volatility = 0.02  # 2% volatility

        predictions = engine.predict(
            event, prices={"AAPL": current_price}, volatilities={"AAPL": volatility}
        )
        assert len(predictions) == 1
        pred = predictions[0]

        assert pred.ticker == "AAPL"
        assert pred.current_price == 200.0
        # Earnings elasticity = 2.5
        # shock = 0.8 * 0.9 * 1.0 * 1.0 = 0.72
        # expected_delta = 200 * 2.5 * 0.72 * 0.02 = 7.20
        assert pred.expected_delta_price == pytest.approx(7.20, abs=1e-3)
        assert pred.target_price == pytest.approx(207.20, abs=1e-3)
        assert pred.target_price > pred.current_price

    def test_logistic_breakout_monotonicity_invariant_inv_news_005(self) -> None:
        engine = NewsPriceReactionEngine()
        current_price = 100.0
        volatility = 0.02

        # Test increasing shocks monotonically increase prob_up
        shocks = [-0.8, -0.4, 0.0, 0.4, 0.8]
        probabilities: list[float] = []

        for s in shocks:
            evt = create_sample_event(
                event_type=EventType.GENERAL_MARKET,
                polarity=s,
                urgency=1.0,
                ticker="SPY",
            )
            preds = engine.predict(
                evt, prices={"SPY": current_price}, volatilities={"SPY": volatility}
            )
            probabilities.append(preds[0].prob_up)

        # Monotonicity check
        for i in range(len(probabilities) - 1):
            assert probabilities[i] < probabilities[i + 1]

        # Neutral shock should yield exactly 50% probability
        assert probabilities[2] == pytest.approx(0.50, abs=1e-4)

        # High positive shock should give BUY signal with high confidence
        high_pos = engine.predict(
            create_sample_event(EventType.EARNINGS, polarity=0.9, urgency=0.9, ticker="NVDA"),
            prices={"NVDA": 120.0},
            volatilities={"NVDA": 0.03},
        )[0]
        assert high_pos.prob_up > 0.65
        assert high_pos.signal == "BUY"
        assert high_pos.confidence > 0.3

        # High negative shock should give SELL signal
        high_neg = engine.predict(
            create_sample_event(
                EventType.REGULATORY_LEGAL, polarity=-0.9, urgency=0.9, ticker="NVDA"
            ),
            prices={"NVDA": 120.0},
            volatilities={"NVDA": 0.03},
        )[0]
        assert high_neg.prob_up < 0.35
        assert high_neg.prob_down > 0.65
        assert high_neg.signal == "SELL"

    def test_triple_barrier_alignment(self) -> None:
        engine = NewsPriceReactionEngine(barrier_multiplier=2.0)
        event = create_sample_event(EventType.M_AND_A, polarity=0.5, urgency=0.8, ticker="MSFT")
        current_price = 400.0
        volatility = 0.015

        pred = engine.predict(
            event, prices={"MSFT": current_price}, volatilities={"MSFT": volatility}
        )[0]

        expected_upper = 400.0 * math.exp(2.0 * 0.015)
        expected_lower = 400.0 * math.exp(-2.0 * 0.015)

        assert pred.barrier_upper == pytest.approx(expected_upper, abs=1e-4)
        assert pred.barrier_lower == pytest.approx(expected_lower, abs=1e-4)
        assert pred.barrier_upper > pred.current_price > pred.barrier_lower

    def test_post_announcement_drift_pead_trajectory(self) -> None:
        engine = NewsPriceReactionEngine()
        event = create_sample_event(EventType.EARNINGS, polarity=0.8, urgency=0.9, ticker="AMZN")
        pred = engine.predict(event, prices={"AMZN": 180.0}, volatilities={"AMZN": 0.02})[0]

        traj = pred.predicted_trajectory
        assert len(traj) >= 5

        # At t=0 (or small t), price is close to peak target
        # Over time (days ahead), memory decays towards baseline
        t0_sec, p0 = traj[0]
        t_last_sec, p_last = traj[-1]

        assert t0_sec < t_last_sec
        # Positive surprise decays back towards baseline over horizon
        assert p0 > p_last >= 180.0

    def test_non_finite_input_rejection(self) -> None:
        engine = NewsPriceReactionEngine()
        event = create_sample_event(EventType.EARNINGS, polarity=0.5, urgency=0.5, ticker="AAPL")

        # Zero or negative price
        with pytest.raises(NonFiniteSignalException) as exc_info:
            engine.predict(event, prices={"AAPL": -10.0}, volatilities={"AAPL": 0.02})
        assert exc_info.value.code == "ERR-NEWS-005"

        # NaN volatility
        with pytest.raises(NonFiniteSignalException):
            engine.predict(event, prices={"AAPL": 100.0}, volatilities={"AAPL": float("nan")})

    def test_latency_sla_under_10ms_invariant_inv_news_006(self) -> None:
        engine = NewsPriceReactionEngine()
        event = create_sample_event(EventType.EARNINGS, polarity=0.7, urgency=0.9, ticker="GOOGL")

        start = time.perf_counter()
        predictions = engine.predict(event, prices={"GOOGL": 170.0}, volatilities={"GOOGL": 0.02})
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert len(predictions) == 1
        # Invariant 6: SLA < 10ms
        assert elapsed_ms < 10.0
        assert predictions[0].calculation_latency_ms < 10.0
