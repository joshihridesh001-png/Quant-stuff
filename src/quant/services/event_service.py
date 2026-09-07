"""Application service orchestrating news event ingestion, causal graph updates, and decay state calculation."""

import math
from datetime import UTC, datetime

import numpy as np

from quant.domain.interfaces import IAssetRepository, IEventRepository
from quant.domain.models import Asset, EventCentrality, NewsEvent


def compute_decay_kernel(
    delta_t_seconds: float,
    urgency: float,
    alpha: float = 0.5,
    tau_fast: float = 3600.0,
    tau_slow: float = 86400.0,
    beta: float = 1.0,
) -> float:
    """Hybrid temporal decay kernel combining exponential and power-law memory.

    kappa(Delta t, u) = alpha * exp(-Delta t / (tau_fast * (1 - u)))
                      + (1 - alpha) * (1 + Delta t / tau_slow)^(-beta)
    """
    if delta_t_seconds < 0:
        return 0.0

    # Guard urgency from 1.0 to prevent division by zero in denominator
    clamped_u = min(max(urgency, 0.0), 0.999)
    fast_denom = max(tau_fast * (1.0 - clamped_u), 1e-6)

    term_fast = alpha * math.exp(-delta_t_seconds / fast_denom)
    term_slow = (1.0 - alpha) * math.pow(1.0 + (delta_t_seconds / tau_slow), -beta)

    return float(term_fast + term_slow)


class EventService:
    """Service handling news ingestion and decayed state vector generation."""

    def __init__(self, event_repo: IEventRepository, asset_repo: IAssetRepository) -> None:
        self.event_repo = event_repo
        self.asset_repo = asset_repo

    async def ingest_event(
        self,
        headline: str,
        raw_text: str,
        timestamp: datetime,
        ticker_weights: dict[str, float],
        dense_embedding: list[float] | None = None,
        sentiment_polarity: float = 0.0,
        sentiment_subjectivity: float = 0.0,
        sentiment_novelty: float = 0.0,
        urgency: float = 0.5,
        source: str = "GENERIC",
    ) -> NewsEvent:
        """Process and persist an unstructured news event with asset centrality mappings."""
        event = NewsEvent(
            headline=headline,
            raw_text=raw_text,
            timestamp=timestamp,
            dense_embedding=dense_embedding or [],
            sentiment_polarity=sentiment_polarity,
            sentiment_subjectivity=sentiment_subjectivity,
            sentiment_novelty=sentiment_novelty,
            urgency=urgency,
            source=source,
        )

        centralities: list[EventCentrality] = []
        for ticker, weight in ticker_weights.items():
            asset = await self.asset_repo.get_by_ticker(ticker)
            if not asset:
                asset = await self.asset_repo.add(
                    Asset(ticker=ticker.upper(), name=ticker.upper(), sector="UNKNOWN")
                )
            centralities.append(
                EventCentrality(event_id=event.id, asset_id=asset.id, centrality=float(weight))
            )

        return await self.event_repo.add(event, centralities)

    async def get_active_news_state(
        self,
        ticker: str,
        as_of_time: datetime | None = None,
        lookback_seconds: float = 604800.0,  # 7 days
        alpha: float = 0.5,
        tau_fast: float = 3600.0,
        tau_slow: float = 86400.0,
        beta: float = 1.0,
    ) -> tuple[list[float], int]:
        """Compute the active time-decayed state vector S_news^{(k)}(t) for asset k.

        Returns (state_vector, event_count).
        """
        now = as_of_time or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        asset = await self.asset_repo.get_by_ticker(ticker)
        if not asset:
            return ([], 0)

        start_time = datetime.fromtimestamp(now.timestamp() - lookback_seconds, tz=UTC)
        events_with_weights = await self.event_repo.get_events_for_asset(
            asset_id=asset.id, start_time=start_time, end_time=now
        )

        if not events_with_weights:
            return ([], 0)

        # Base composite vector dimension: [sentiment(3), urgency(1), centrality(1)]
        # + optional dense embedding dimensions
        first_event, _ = events_with_weights[0]
        dense_dim = len(first_event.dense_embedding)
        total_dim = 5 + dense_dim

        accumulated_state = np.zeros(total_dim, dtype=np.float64)

        for event, centrality in events_with_weights:
            ev_ts = event.timestamp
            if ev_ts.tzinfo is None:
                ev_ts = ev_ts.replace(tzinfo=UTC)
            delta_t = max((now - ev_ts).total_seconds(), 0.0)
            decay_factor = compute_decay_kernel(
                delta_t_seconds=delta_t,
                urgency=event.urgency,
                alpha=alpha,
                tau_fast=tau_fast,
                tau_slow=tau_slow,
                beta=beta,
            )

            # Construct e_i = [dense || sentiment || urgency || centrality]
            dense_part = event.dense_embedding if dense_dim > 0 else []
            feat_vector = np.array(
                dense_part + event.sentiment_vector + [event.urgency, centrality],
                dtype=np.float64,
            )

            # Accumulate c_{i,k} * e_i * kappa(Delta t, u_i)
            effective_weight = centrality * decay_factor
            accumulated_state += effective_weight * feat_vector

        return (accumulated_state.tolist(), len(events_with_weights))
