"""Comprehensive unit tests for the Fractional Differentiation Engine.

Purpose: Verifies mathematical invariants, transformer behavior, and streaming ring buffer.
Dependencies: pytest, numpy, quant.analytics.fractional_diff, domain interfaces.
"""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from quant.analytics.fractional_diff import (
    FractionalDifferentiator,
    StreamingFracDiffBuffer,
    compute_fractional_weights,
)
from quant.domain.interfaces import IMarketDataRepository
from quant.domain.models import MarketDataBatch, Resolution

# =====================================================================
# 1. Tests for compute_fractional_weights
# =====================================================================


def test_fractional_weights_boundary_d0() -> None:
    """d=0.0 must yield identity filter [1.0]."""
    weights = compute_fractional_weights(d=0.0)
    np.testing.assert_array_equal(weights, np.array([1.0]))


def test_fractional_weights_boundary_d1() -> None:
    """d=1.0 must yield standard first-difference filter [1.0, -1.0]."""
    weights = compute_fractional_weights(d=1.0)
    np.testing.assert_array_equal(weights, np.array([1.0, -1.0]))


def test_fractional_weights_invalid_bounds() -> None:
    """d outside [0.0, 1.0] must raise ValueError."""
    with pytest.raises(ValueError, match=r"Differencing degree d must be in \[0.0, 1.0\]"):
        compute_fractional_weights(d=-0.1)

    with pytest.raises(ValueError, match=r"Differencing degree d must be in \[0.0, 1.0\]"):
        compute_fractional_weights(d=1.05)


def test_fractional_weights_invalid_threshold() -> None:
    """Non-positive threshold must raise ValueError."""
    with pytest.raises(ValueError, match="Tolerance threshold must be strictly positive"):
        compute_fractional_weights(d=0.5, threshold=0.0)

    with pytest.raises(ValueError, match="Tolerance threshold must be strictly positive"):
        compute_fractional_weights(d=0.5, threshold=-1e-4)


def test_fractional_weights_zero_sum_correction() -> None:
    """With zero_sum_correction=True, sum(weights) must be numerically 0.0."""
    for d in [0.2, 0.4, 0.6, 0.8]:
        weights = compute_fractional_weights(d=d, threshold=1e-4, zero_sum_correction=True)
        assert len(weights) > 1
        assert abs(float(np.sum(weights))) < 1e-12


def test_fractional_weights_without_zero_sum_correction() -> None:
    """Without zero-sum correction, omega_0 must be strictly 1.0 and sum > 0."""
    weights = compute_fractional_weights(d=0.4, threshold=1e-4, zero_sum_correction=False)
    assert weights[0] == 1.0
    assert float(np.sum(weights)) > 0.0


def test_fractional_weights_max_lookback_cap() -> None:
    """Weight length must not exceed max_lookback + 1."""
    max_lb = 15
    weights = compute_fractional_weights(d=0.4, threshold=1e-7, max_lookback=max_lb)
    assert len(weights) <= max_lb + 1


# =====================================================================
# 2. Tests for FractionalDifferentiator Transformer
# =====================================================================


def test_transformer_unfitted_transform_raises() -> None:
    """Calling transform or inverse_transform before fit must raise RuntimeError."""
    diff = FractionalDifferentiator()
    dummy = np.ones(50)
    with pytest.raises(RuntimeError, match="must be fitted before calling transform"):
        diff.transform(dummy)

    with pytest.raises(RuntimeError, match="must be fitted before calling inverse_transform"):
        diff.inverse_transform(dummy, dummy)


def test_transformer_invalid_input_shapes_and_values() -> None:
    """fit() must reject 2D inputs, series with length < 30, and NaNs/Infs."""
    diff = FractionalDifferentiator(d=0.5)

    # 2D array
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        diff.fit(np.ones((10, 10)))

    # Short array (< 30 observations)
    with pytest.raises(ValueError, match="must contain at least 30 observations"):
        diff.fit(np.ones(20))

    # Array containing NaNs
    bad_arr = np.ones(50)
    bad_arr[10] = np.nan
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        diff.fit(bad_arr)

    # Array containing Infs
    bad_arr[10] = np.inf
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        diff.fit(bad_arr)


def test_transformer_short_transform_series_rejected() -> None:
    """transform() series length <= lookback must be rejected."""
    diff = FractionalDifferentiator(d=0.5, threshold=1e-3)
    np.random.seed(42)
    series = 100.0 + np.cumsum(np.random.normal(0.0, 1.0, 100))
    diff.fit(series)

    short_series = np.ones(diff.lookback_)
    with pytest.raises(ValueError, match="strictly greater than lookback horizon"):
        diff.transform(short_series)


def test_transformer_fixed_d_fit_and_transform() -> None:
    """Transformer with explicitly given d must compute weights and transform properly."""
    # Synthetic random walk
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 200)
    prices = 100.0 * np.exp(np.cumsum(returns))

    diff = FractionalDifferentiator(d=0.5, threshold=1e-3, zero_sum_weights=True)
    diff.fit(prices)

    assert diff.d_ == 0.5
    assert diff.weights_ is not None
    assert diff.lookback_ == len(diff.weights_) - 1
    assert diff.adf_stat_ is not None
    assert diff.adf_pvalue_ is not None
    assert diff.correlation_ is not None

    # Unpadded transform
    transformed = diff.transform(prices, pad_nans=False)
    assert len(transformed) == len(prices) - diff.lookback_
    assert not np.any(np.isnan(transformed))

    # Padded transform
    padded = diff.transform(prices, pad_nans=True)
    assert len(padded) == len(prices)
    assert np.all(np.isnan(padded[: diff.lookback_]))
    np.testing.assert_array_equal(padded[diff.lookback_ :], transformed)

    # fit_transform equivalence
    fit_transformed = FractionalDifferentiator(
        d=0.5, threshold=1e-3, zero_sum_weights=True
    ).fit_transform(prices, pad_nans=False)
    np.testing.assert_array_almost_equal(transformed, fit_transformed)


def test_transformer_bisection_search_converges() -> None:
    """Bisection search without pre-specified d must find a stationary d* in (0, 1]."""
    np.random.seed(123)
    # Generate non-stationary integrated random walk series
    innovations = np.random.normal(0.0, 1.0, 300)
    non_stationary_walk = 100.0 + np.cumsum(innovations)

    diff = FractionalDifferentiator(threshold=1e-3, significance_level=0.05)
    diff.fit(non_stationary_walk)

    assert diff.d_ is not None
    assert 0.0 < diff.d_ <= 1.0
    assert diff.adf_pvalue_ is not None
    assert diff.adf_pvalue_ <= 0.05
    assert diff.correlation_ is not None
    assert diff.correlation_ > 0.0


def test_transformer_bisection_stationary_input_returns_d0() -> None:
    """Input series that is already stationary should return d*=0."""
    np.random.seed(42)
    # Stationary i.i.d white noise series
    stationary_series = np.random.normal(0.0, 1.0, 500)

    diff = FractionalDifferentiator(threshold=1e-3, significance_level=0.05)
    diff.fit(stationary_series)

    assert diff.d_ == 0.0
    assert diff.lookback_ == 0
    np.testing.assert_array_equal(diff.weights_, np.array([1.0]))


def test_transformer_inverse_transform_exact_reconstruction() -> None:
    """inverse_transform must exactly reconstruct original series within floating-point epsilon."""
    np.random.seed(99)
    prices = 100.0 + np.cumsum(np.random.normal(0.0, 1.0, 150))

    diff = FractionalDifferentiator(d=0.4, threshold=1e-3, zero_sum_weights=True)
    diff.fit(prices)

    l_star = diff.lookback_
    transformed = diff.transform(prices, pad_nans=False)

    # Provide initial history anchor of length l_star
    history_anchor = prices[:l_star]

    # Reconstruct
    reconstructed = diff.inverse_transform(y_diff=transformed, history_anchor=history_anchor)

    assert len(reconstructed) == len(prices) - l_star
    # Verify numerical identity against the actual price history
    expected_prices = prices[l_star:]
    np.testing.assert_allclose(reconstructed, expected_prices, rtol=1e-9, atol=1e-9)


def test_transformer_inverse_transform_short_anchor_rejected() -> None:
    """history_anchor shorter than lookback must raise ValueError."""
    diff = FractionalDifferentiator(d=0.4, threshold=1e-3)
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.normal(0.0, 1.0, 100))
    diff.fit(prices)

    short_anchor = prices[: diff.lookback_ - 1]
    with pytest.raises(ValueError, match="must be >= lookback horizon"):
        diff.inverse_transform(y_diff=np.array([1.0, 2.0]), history_anchor=short_anchor)


# =====================================================================
# 3. Tests for StreamingFracDiffBuffer
# =====================================================================


def test_streaming_buffer_unhydrated_update_raises() -> None:
    """Invoking update() before hydrate() must raise RuntimeError."""
    weights = np.array([1.0, -0.5, -0.2, -0.1])
    buffer = StreamingFracDiffBuffer(weights)

    assert not buffer.is_hydrated
    assert buffer.lookback == 3

    with pytest.raises(RuntimeError, match="buffer is unhydrated"):
        buffer.update(100.0)


def test_streaming_buffer_insufficient_hydration_history_raises() -> None:
    """Hydrating with history shorter than lookback must raise ValueError."""
    weights = np.array([1.0, -0.5, -0.2, -0.1])  # lookback = 3
    buffer = StreamingFracDiffBuffer(weights)

    with pytest.raises(ValueError, match="must be >= lookback horizon"):
        buffer.hydrate([100.0, 101.0])


def test_streaming_buffer_exact_equivalence_to_batch_transform() -> None:
    """Consecutive update() calls must yield identical outputs to batch transform()."""
    np.random.seed(77)
    prices = 100.0 + np.cumsum(np.random.normal(0.0, 1.0, 100))

    diff = FractionalDifferentiator(d=0.35, threshold=1e-3, zero_sum_weights=True)
    diff.fit(prices)

    batch_output = diff.transform(prices, pad_nans=False)
    l_star = diff.lookback_

    # Initialize streaming buffer with the same weights
    assert diff.weights_ is not None
    stream_buffer = StreamingFracDiffBuffer(diff.weights_)

    # Hydrate with prices up to l_star
    stream_buffer.hydrate(prices[:l_star])
    assert stream_buffer.is_hydrated

    # Ingest remaining prices one by one and record output
    streaming_outputs = []
    for price in prices[l_star:]:
        val = stream_buffer.update(price)
        streaming_outputs.append(val)

    streaming_arr = np.array(streaming_outputs, dtype=np.float64)
    np.testing.assert_allclose(streaming_arr, batch_output, rtol=1e-10, atol=1e-10)


@pytest.mark.asyncio
async def test_streaming_buffer_hydrate_from_repository() -> None:
    """hydrate_from_repository fetches latest bars and successfully primes buffer."""
    weights = np.array([1.0, -0.4, -0.3])  # lookback = 2
    stream_buffer = StreamingFracDiffBuffer(weights)

    # Mock repository returning sufficient bars
    mock_repo = AsyncMock(spec=IMarketDataRepository)
    closes = np.array([101.0, 102.0, 103.0], dtype=np.float64)
    mock_batch = MarketDataBatch(
        asset_id="BTC-USDT",
        resolution=Resolution.ONE_MINUTE,
        timestamps=np.array([1, 2, 3], dtype=np.int64),
        opens=closes,
        highs=closes,
        lows=closes,
        closes=closes,
        volumes=np.array([10.0, 20.0, 30.0], dtype=np.float64),
        vwaps=closes,
    )
    mock_repo.get_latest_bars.return_value = mock_batch

    await stream_buffer.hydrate_from_repository(
        asset_id="BTC-USDT",
        repository=mock_repo,
        resolution=Resolution.ONE_MINUTE,
    )

    assert stream_buffer.is_hydrated

    # Test update works after repository hydration
    next_val = stream_buffer.update(104.0)
    assert isinstance(next_val, float)


@pytest.mark.asyncio
async def test_streaming_buffer_hydrate_from_repository_insufficient_bars() -> None:
    """Repository returning fewer bars than lookback must raise ValueError."""
    weights = np.array([1.0, -0.4, -0.3, -0.2, -0.1])  # lookback = 4
    stream_buffer = StreamingFracDiffBuffer(weights)

    mock_repo = AsyncMock(spec=IMarketDataRepository)
    closes = np.array([101.0, 102.0], dtype=np.float64)  # only 2 bars
    mock_batch = MarketDataBatch(
        asset_id="BTC-USDT",
        resolution=Resolution.ONE_MINUTE,
        timestamps=np.array([1, 2], dtype=np.int64),
        opens=closes,
        highs=closes,
        lows=closes,
        closes=closes,
        volumes=np.array([10.0, 20.0], dtype=np.float64),
        vwaps=closes,
    )
    mock_repo.get_latest_bars.return_value = mock_batch

    with pytest.raises(ValueError, match="repository contains only 2 bars"):
        await stream_buffer.hydrate_from_repository(
            asset_id="BTC-USDT",
            repository=mock_repo,
        )
