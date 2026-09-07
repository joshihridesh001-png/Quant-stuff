"""Application service orchestrating news event ingestion, causal graph updates, and decay state calculation."""

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from quant.domain.interfaces import IAssetRepository, IEventRepository, INewsIngestionEngine
from quant.domain.models import Asset, EventCentrality, NewsEvent


def compute_decay_kernel(
    delta_t_seconds: float,
    urgency: float,
    alpha: float = 0.5,
    tau_fast: float = 3600.0,
    tau_slow: float = 86400.0,
    beta: float = 1.0,
    tolerance: float = 1e-4,
    max_lookback_seconds: float | None = 604800.0,  # 7 days max horizon
) -> float:
    """Hybrid temporal decay kernel combining exponential and power-law memory.

    kappa(Delta t, u) = alpha * exp(-Delta t / (tau_fast * (1 - u)))
                      + (1 - alpha) * (1 + Delta t / tau_slow)^(-beta)

    Includes explicit numerical truncation (tolerance) and maximum lookback bound.
    """
    if delta_t_seconds < 0:
        return 0.0

    if max_lookback_seconds is not None and delta_t_seconds > max_lookback_seconds:
        return 0.0

    # Guard urgency from 1.0 to prevent division by zero in denominator
    clamped_u = min(max(urgency, 0.0), 0.999)
    fast_denom = max(tau_fast * (1.0 - clamped_u), 1e-6)

    term_fast = alpha * math.exp(-delta_t_seconds / fast_denom)
    term_slow = (1.0 - alpha) * math.pow(1.0 + (delta_t_seconds / tau_slow), -beta)

    val = float(term_fast + term_slow)
    if val < tolerance:
        return 0.0
    return val


def compute_projected_feature_vector(
    dense_embedding: list[float],
    sentiment_vector: list[float],
    urgency: float,
    centrality: float,
    target_dense_dim: int = 16,
) -> np.ndarray:
    """Balance high-dimensional embeddings with low-dimensional scalar signals.

    Solves the dimensional asymmetry problem where a 768-d semantic vector mathematically
    overwhelms 3-d sentiment, 1-d urgency, and 1-d centrality in downstream distance metrics.
    Projects dense embedding into a normalized subspace of dimension `target_dense_dim`
    using deterministic random projection, ensuring comparable variance.
    """
    if not dense_embedding:
        projected_dense = np.zeros(target_dense_dim, dtype=np.float64)
    else:
        src = np.array(dense_embedding, dtype=np.float64)
        n_src = len(src)
        if n_src <= target_dense_dim:
            projected_dense = np.zeros(target_dense_dim, dtype=np.float64)
            projected_dense[:n_src] = src
        else:
            rng = np.random.default_rng(42)
            proj_matrix = rng.normal(
                0, 1.0 / np.sqrt(target_dense_dim), size=(n_src, target_dense_dim)
            )
            projected_dense = src @ proj_matrix

        norm = float(np.linalg.norm(projected_dense))
        if norm > 1e-8:
            projected_dense = projected_dense / norm

    scalar_features = np.array(sentiment_vector + [urgency, centrality], dtype=np.float64)
    return np.concatenate([projected_dense, scalar_features])


class EventService(INewsIngestionEngine):
    """Service handling news ingestion, batch processing, and decayed state vector generation."""

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

    async def ingest_batch(
        self,
        events_data: list[dict[str, Any]],
    ) -> list[NewsEvent]:
        """Process and persist a batch of unstructured news events atomically."""
        if not events_data:
            return []

        batch_to_persist: list[tuple[NewsEvent, list[EventCentrality]]] = []
        for data in events_data:
            event = NewsEvent(
                headline=data["headline"],
                raw_text=data["raw_text"],
                timestamp=data["timestamp"],
                dense_embedding=data.get("dense_embedding") or [],
                sentiment_polarity=data.get("sentiment_polarity", 0.0),
                sentiment_subjectivity=data.get("sentiment_subjectivity", 0.0),
                sentiment_novelty=data.get("sentiment_novelty", 0.0),
                urgency=data.get("urgency", 0.5),
                source=data.get("source", "GENERIC"),
            )
            centralities: list[EventCentrality] = []
            ticker_weights = data.get("ticker_weights", {})
            for ticker, weight in ticker_weights.items():
                asset = await self.asset_repo.get_by_ticker(ticker)
                if not asset:
                    asset = await self.asset_repo.add(
                        Asset(ticker=ticker.upper(), name=ticker.upper(), sector="UNKNOWN")
                    )
                centralities.append(
                    EventCentrality(event_id=event.id, asset_id=asset.id, centrality=float(weight))
                )
            batch_to_persist.append((event, centralities))

        return await self.event_repo.add_batch(batch_to_persist)

    async def get_state_vectors(
        self,
        tickers: list[str],
        as_of_time: datetime | None = None,
    ) -> dict[str, tuple[list[float], int]]:
        """Asynchronously compute active decayed news state vectors across multiple assets."""
        results: dict[str, tuple[list[float], int]] = {}
        for ticker in tickers:
            results[ticker.upper()] = await self.get_active_news_state(
                ticker=ticker, as_of_time=as_of_time
            )
        return results

    async def get_active_news_state(
        self,
        ticker: str,
        as_of_time: datetime | None = None,
        lookback_seconds: float = 604800.0,  # 7 days
        alpha: float = 0.5,
        tau_fast: float = 3600.0,
        tau_slow: float = 86400.0,
        beta: float = 1.0,
        use_projected_subspace: bool = False,
        target_dense_dim: int = 16,
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

        first_event, _ = events_with_weights[0]
        dense_dim = target_dense_dim if use_projected_subspace else len(first_event.dense_embedding)
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

            if use_projected_subspace:
                feat_vector = compute_projected_feature_vector(
                    dense_embedding=event.dense_embedding,
                    sentiment_vector=event.sentiment_vector,
                    urgency=event.urgency,
                    centrality=centrality,
                    target_dense_dim=target_dense_dim,
                )
            else:
                dense_part = event.dense_embedding if dense_dim > 0 else []
                feat_vector = np.array(
                    dense_part + event.sentiment_vector + [event.urgency, centrality],
                    dtype=np.float64,
                )

            effective_weight = centrality * decay_factor
            accumulated_state += effective_weight * feat_vector

        return (accumulated_state.tolist(), len(events_with_weights))

        return (accumulated_state.tolist(), len(events_with_weights))
