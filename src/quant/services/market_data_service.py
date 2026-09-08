"""Market data application service managing ingestion, validation, and econometric feature extraction.

Purpose: Orchestrates price bar ingestion, data quality validation, and realized volatility calculation.
Dependencies: IMarketDataRepository, IAssetRepository, NumPy, PriceBar, MarketDataBatch, Resolution.
Relationship: Consumed by API endpoints and downstream econometric pipelines (Fractional Diff, Triple Barrier).
Invariants: Enforces strict data quality invariants before committing bars to columnar storage.
"""

import logging
from typing import Any

# Numerical analysis library for vectorized array operations
import numpy as np

# Domain abstractions and value objects
from quant.domain.interfaces import IAssetRepository, IMarketDataRepository
from quant.domain.models import MarketDataBatch, PriceBar, Resolution

# Structured application logger
logger = logging.getLogger(__name__)


class MarketDataService:
    """Application service for market data ingestion and numerical preparation.

    Purpose: Coordinates bar validation, persistence to DuckDB, and calculation of realized volatility.
    Dependencies: Injected IMarketDataRepository and optional IAssetRepository.
    Relationship: Primary facade for market time series consumed by econometric engines.
    """

    def __init__(
        self,
        market_repo: IMarketDataRepository,
        asset_repo: IAssetRepository | None = None,
    ) -> None:
        """Initialize market data service with injected repository adapters.

        Purpose: Establishes dependency inversion for data access.
        Dependencies: IMarketDataRepository implementation, optional IAssetRepository.
        """
        # Purpose: Store market data persistence adapter
        # Invariant: Must implement IMarketDataRepository
        self._market_repo = market_repo

        # Purpose: Store asset registry adapter for universe validation
        self._asset_repo = asset_repo

    async def ingest_bars(
        self,
        bars_data: list[dict[str, Any]],
        resolution: Resolution = Resolution.ONE_MINUTE,
    ) -> tuple[int, int]:
        """Validate, construct, and bulk-persist a list of raw price bar dictionaries.

        Purpose: Ingests external market data feeds with defensive invariant validation.
        Dependencies: PriceBar constructor, IMarketDataRepository.add_bars_batch.
        Post-conditions: Valid bars persisted to storage; returns (processed_count, persisted_count).
        """
        # Purpose: Return immediately if input collection is empty
        if not bars_data:
            return 0, 0

        # Purpose: Parse and validate each input dictionary into an immutable PriceBar value object
        # Dependencies: PriceBar.__post_init__ enforces mathematical invariants
        parsed_bars: list[PriceBar] = []
        for idx, item in enumerate(bars_data):
            try:
                bar = PriceBar(
                    asset_id=str(item["asset_id"]),
                    timestamp=int(item["timestamp"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item["volume"]),
                    vwap=float(item["vwap"]),
                    resolution=resolution,
                )
                parsed_bars.append(bar)
            except (KeyError, ValueError, TypeError) as exc:
                # Log diagnostic warning detailing offending record coordinates
                logger.warning(
                    "Validation failed for market bar at index %d: %s. Record discarded.",
                    idx,
                    exc,
                )
                raise ValueError(f"Invalid market bar at index {idx}: {exc}") from exc

        # Purpose: Verify asset registration if asset repository is injected
        if self._asset_repo is not None:
            unique_assets = {b.asset_id for b in parsed_bars}
            for ticker in unique_assets:
                asset = await self._asset_repo.get_by_ticker(ticker)
                if asset is None:
                    logger.warning("Ingesting bars for unregistered asset ticker: %s", ticker)

        # Purpose: Commit validated bars to high-performance columnar storage
        # Dependencies: self._market_repo.add_bars_batch
        persisted_count = await self._market_repo.add_bars_batch(parsed_bars)

        return len(parsed_bars), persisted_count

    async def get_historical_bars(
        self,
        asset_id: str,
        start_time: int,
        end_time: int,
        resolution: Resolution = Resolution.ONE_MINUTE,
    ) -> MarketDataBatch:
        """Retrieve contiguous columnar market bars within timestamp range [start_time, end_time].

        Purpose: Supplies chronological price/volume arrays for backtesting and feature calculation.
        Dependencies: self._market_repo.get_bars_range.
        Invariants: start_time <= end_time.
        """
        # Invariant check: Temporal boundary ordering
        if start_time > end_time:
            raise ValueError(
                f"start_time ({start_time}) cannot be strictly greater than end_time ({end_time})"
            )

        return await self._market_repo.get_bars_range(asset_id, start_time, end_time, resolution)

    async def get_latest_bars(
        self,
        asset_id: str,
        count: int,
        resolution: Resolution = Resolution.ONE_MINUTE,
    ) -> MarketDataBatch:
        """Retrieve the most recent N contiguous columnar market bars in ascending order.

        Purpose: Feeds rolling econometric windows for live inference.
        Dependencies: self._market_repo.get_latest_bars.
        Invariants: count > 0.
        """
        if count <= 0:
            raise ValueError(f"Requested bar count must be strictly positive, got {count}")

        return await self._market_repo.get_latest_bars(asset_id, count, resolution)

    @staticmethod
    def compute_realized_volatility(batch: MarketDataBatch, window: int = 20) -> np.ndarray:
        """Calculate rolling realized standard deviation of log returns on closing prices.

        Purpose: Estimates instantaneous volatility sigma_t for dynamic Triple-Barrier scaling.
        Dependencies: NumPy vectorized log and diff operations.
        Mathematical Formulation:
            r_t = ln(P_t / P_{t-1})
            sigma_t = sqrt( (1 / (W - 1)) * sum_{k=0}^{W-1} (r_{t-k} - mean(r))^2 )
        Invariants: Output array matches batch length; uncomputed warm-up prefix populated with zeros.
        """
        # Purpose: Determine total observation length
        n = batch.count

        # Purpose: If observations are fewer than 2, volatility cannot be defined
        if n < 2 or window < 2:
            return np.zeros(n, dtype=np.float64)

        # Purpose: Compute continuously compounded log returns
        # Dependencies: np.log, np.diff
        prices = batch.closes
        log_returns = np.diff(np.log(prices))  # length n - 1

        # Purpose: Pre-allocate output volatility array initialized with zeros
        volatility = np.zeros(n, dtype=np.float64)

        # Purpose: Compute rolling standard deviation using sliding window
        # Invariant: Indices i < window remain 0.0 (warm-up window)
        for i in range(window, n):
            # Window slice over log returns
            window_slice = log_returns[i - window : i]
            # Sample standard deviation with ddof=1 (Bessel's correction)
            volatility[i] = float(np.std(window_slice, ddof=1))

        return volatility
