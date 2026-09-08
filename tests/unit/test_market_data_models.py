"""Unit tests for market data domain models, invariants, and econometric volatility estimation.

Purpose: Verifies defensive invariant enforcement in PriceBar, MarketDataBatch, and realized volatility.
Dependencies: pytest, numpy, pyarrow, PriceBar, MarketDataBatch, Resolution, MarketDataService.
Relationship: Validates domain model contracts before persistence or econometric transformation.
Invariants: Asserts that invalid prices, volumes, timestamps, and dimensions are strictly rejected.
"""

import numpy as np
import pytest

from quant.domain.models import MarketDataBatch, PriceBar, Resolution
from quant.services.market_data_service import MarketDataService


def test_valid_price_bar_instantiation() -> None:
    """Verify standard valid price bar instantiation succeeds with correct attributes."""
    bar = PriceBar(
        asset_id="AAPL",
        timestamp=1700000000000000000,
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        volume=1000.0,
        vwap=152.5,
        resolution=Resolution.ONE_MINUTE,
    )
    assert bar.asset_id == "AAPL"
    assert bar.close == 154.0
    assert bar.resolution == Resolution.ONE_MINUTE


def test_price_bar_invalid_timestamp_rejected() -> None:
    """Verify non-positive timestamps raise ValueError."""
    with pytest.raises(ValueError, match="positive nanoseconds"):
        PriceBar(
            asset_id="AAPL",
            timestamp=0,
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=100.0,
            vwap=152.0,
        )


def test_price_bar_non_positive_prices_rejected() -> None:
    """Verify negative or zero prices raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        PriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=-150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=100.0,
            vwap=152.0,
        )


def test_price_bar_high_less_than_open_close_rejected() -> None:
    """Verify high below open or close raises ValueError."""
    with pytest.raises(ValueError, match="cannot be strictly less"):
        PriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=150.0,
            high=148.0,  # Invalid: high is less than open
            low=145.0,
            close=147.0,
            volume=100.0,
            vwap=147.5,
        )


def test_price_bar_low_greater_than_open_close_rejected() -> None:
    """Verify low above open or close raises ValueError."""
    with pytest.raises(ValueError, match="cannot be strictly greater"):
        PriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=150.0,
            high=155.0,
            low=152.0,  # Invalid: low is greater than open
            close=153.0,
            volume=100.0,
            vwap=153.0,
        )


def test_price_bar_negative_volume_rejected() -> None:
    """Verify negative volume raises ValueError."""
    with pytest.raises(ValueError, match="cannot be negative"):
        PriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=-10.0,
            vwap=152.0,
        )


def test_price_bar_zero_vwap_with_positive_volume_rejected() -> None:
    """Verify zero VWAP with non-zero traded volume raises ValueError."""
    with pytest.raises(ValueError, match="VWAP must be strictly positive"):
        PriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=500.0,
            vwap=0.0,
        )


def test_market_data_batch_dimension_mismatch_rejected() -> None:
    """Verify MarketDataBatch enforces identical 1D array lengths."""
    with pytest.raises(ValueError, match="Dimension mismatch"):
        MarketDataBatch(
            asset_id="AAPL",
            resolution=Resolution.ONE_MINUTE,
            timestamps=np.array([1, 2, 3], dtype=np.int64),
            opens=np.array([100.0, 101.0], dtype=np.float64),  # Length 2 instead of 3
            highs=np.array([105.0, 106.0, 107.0], dtype=np.float64),
            lows=np.array([99.0, 100.0, 101.0], dtype=np.float64),
            closes=np.array([104.0, 105.0, 106.0], dtype=np.float64),
            volumes=np.array([50.0, 60.0, 70.0], dtype=np.float64),
            vwaps=np.array([102.0, 103.0, 104.0], dtype=np.float64),
        )


def test_market_data_batch_to_arrow_and_properties() -> None:
    """Verify MarketDataBatch metadata properties and PyArrow conversion."""
    batch = MarketDataBatch(
        asset_id="AAPL",
        resolution=Resolution.ONE_MINUTE,
        timestamps=np.array([100, 200, 300], dtype=np.int64),
        opens=np.array([100.0, 101.0, 102.0], dtype=np.float64),
        highs=np.array([105.0, 106.0, 107.0], dtype=np.float64),
        lows=np.array([99.0, 100.0, 101.0], dtype=np.float64),
        closes=np.array([104.0, 105.0, 106.0], dtype=np.float64),
        volumes=np.array([50.0, 60.0, 70.0], dtype=np.float64),
        vwaps=np.array([102.0, 103.0, 104.0], dtype=np.float64),
    )
    assert batch.count == 3
    assert batch.start_time == 100
    assert batch.end_time == 300

    arrow_table = batch.to_arrow()
    assert arrow_table.num_rows == 3
    assert arrow_table.num_columns == 7
    assert "close" in arrow_table.column_names


def test_compute_realized_volatility_zero_on_constant_prices() -> None:
    """Verify that a series of constant prices yields exactly zero realized volatility."""
    n = 30
    batch = MarketDataBatch(
        asset_id="AAPL",
        resolution=Resolution.ONE_MINUTE,
        timestamps=np.arange(1, n + 1, dtype=np.int64),
        opens=np.full(n, 100.0, dtype=np.float64),
        highs=np.full(n, 100.0, dtype=np.float64),
        lows=np.full(n, 100.0, dtype=np.float64),
        closes=np.full(n, 100.0, dtype=np.float64),
        volumes=np.full(n, 100.0, dtype=np.float64),
        vwaps=np.full(n, 100.0, dtype=np.float64),
    )
    vol = MarketDataService.compute_realized_volatility(batch, window=10)
    assert len(vol) == n
    # For constant prices, log return is 0 everywhere, so sample std is 0.0
    assert np.all(vol[10:] == 0.0)


def test_compute_realized_volatility_short_series_graceful() -> None:
    """Verify that batches with fewer observations than window return zero-filled array."""
    batch = MarketDataBatch(
        asset_id="AAPL",
        resolution=Resolution.ONE_MINUTE,
        timestamps=np.array([1], dtype=np.int64),
        opens=np.array([100.0], dtype=np.float64),
        highs=np.array([100.0], dtype=np.float64),
        lows=np.array([100.0], dtype=np.float64),
        closes=np.array([100.0], dtype=np.float64),
        volumes=np.array([100.0], dtype=np.float64),
        vwaps=np.array([100.0], dtype=np.float64),
    )
    vol = MarketDataService.compute_realized_volatility(batch, window=5)
    assert len(vol) == 1
    assert vol[0] == 0.0
