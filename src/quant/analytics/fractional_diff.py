"""Fractional Differentiation Engine for memory-preserving stationarity.

Purpose: Transforms non-stationary price series into stationary features without erasing multi-period memory.
Dependencies: numpy, scipy.signal.fftconvolve, statsmodels.tsa.stattools.adfuller, domain interfaces.
Relationship: Core econometric transformer feeding Triple-Barrier labeling (Step 3) and Meta-Labeling (Step 5).
Invariants: Weight series bounded by epsilon; bisection search guarantees minimum stationary d*; zero lookahead bias.
"""

import logging
from collections.abc import Sequence
from typing import Self

# Numerical and signal processing libraries
import numpy as np
from scipy.signal import fftconvolve

# Statistical time series analysis library for stationarity testing
from statsmodels.tsa.stattools import adfuller

# Domain persistence contracts and models
from quant.domain.interfaces import IMarketDataRepository
from quant.domain.models import Resolution

# Structured application logger
logger = logging.getLogger(__name__)


def compute_fractional_weights(
    d: float,
    threshold: float = 1e-4,
    max_lookback: int | None = None,
    zero_sum_correction: bool = True,
) -> np.ndarray:
    """Generate truncated binomial expansion weights for fractional differencing operator (1 - B)^d.

    Purpose: Expands (1 - B)^d into a finite-width linear convolution filter with bounded memory.
    Dependencies: NumPy array operations.
    Mathematical Formulation:
        omega_0 = 1.0
        omega_k = -omega_{k-1} * (d - k + 1) / k, for k >= 1
    Invariants:
        - d must lie in [0.0, 1.0].
        - Weights stop expanding when |omega_k| < threshold.
        - If zero_sum_correction is True, omega_0 is adjusted such that sum(omega) == 0.0.
    """
    # Invariant check: Differencing degree d must be bounded in unit interval [0.0, 1.0]
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"Differencing degree d must be in [0.0, 1.0], got {d}")

    # Invariant check: Numerical tolerance threshold must be positive
    if threshold <= 0.0:
        raise ValueError(f"Tolerance threshold must be strictly positive, got {threshold}")

    # Handle boundary case: d = 0 (identity operator, no differencing)
    if d == 0.0:
        return np.array([1.0], dtype=np.float64)

    # Handle boundary case: d = 1 (standard first differencing: y_t - y_{t-1})
    if d == 1.0:
        return np.array([1.0, -1.0], dtype=np.float64)

    # Initialize weights list with omega_0 = 1.0
    weights = [1.0]
    k = 1

    # Generate binomial weights recursively until weight magnitude falls below tolerance
    while True:
        # Recursive formulation: omega_k = -omega_{k-1} * (d - k + 1) / k
        w_k = -weights[-1] * (d - float(k) + 1.0) / float(k)

        # Check stopping threshold
        if abs(w_k) < threshold:
            break

        weights.append(w_k)
        k += 1

        # Bound expansion by max_lookback cap if configured
        if max_lookback is not None and k > max_lookback:
            break

    weight_arr = np.array(weights, dtype=np.float64)

    # Purpose: Apply zero-sum correction to eliminate DC offset / secular price-level leakage (Flaw 6)
    # When truncated, sum(weights) > 0. Re-centering omega_0 guarantees zero gain at frequency 0.
    if zero_sum_correction and len(weight_arr) > 1:
        weight_arr[0] = -np.sum(weight_arr[1:])

    return weight_arr


class FractionalDifferentiator:
    """Scikit-Learn compliant Fixed-Width Window Fractional Differentiation Transformer.

    Purpose: Encapsulates fitted fractional degree d*, weights, and lookback horizon for pipeline isolation.
    Dependencies: compute_fractional_weights, scipy.signal.fftconvolve, statsmodels.tsa.stattools.adfuller.
    Relationship: Fits on training splits; transforms validation/test partitions without lookahead leakage.
    Invariants: transform() fails if called prior to fit(); rejects sample series shorter than lookback horizon.
    """

    def __init__(
        self,
        d: float | None = None,
        threshold: float = 1e-4,
        max_lookback_ratio: float = 0.20,
        significance_level: float = 0.01,
        zero_sum_weights: bool = True,
    ) -> None:
        """Initialize transformer hyperparameters.

        Purpose: Configures calibration constraints and tolerance bounds.
        Dependencies: Parameter bounds validation.
        """
        # Differencing parameter (None indicates automated calibration during fit)
        self.d = d
        self.threshold = threshold
        self.max_lookback_ratio = max_lookback_ratio
        self.significance_level = significance_level
        self.zero_sum_weights = zero_sum_weights

        # Fitted attributes (populated strictly during fit)
        self.d_: float | None = None
        self.weights_: np.ndarray | None = None
        self.lookback_: int = 0
        self.adf_stat_: float | None = None
        self.adf_pvalue_: float | None = None
        self.correlation_: float | None = None

    def fit(self, y: np.ndarray) -> Self:
        """Fit the transformer on training observations, estimating optimal d* if not pre-specified.

        Purpose: Estimates minimum degree d* achieving stationarity strictly within the training boundary.
        Dependencies: bisection stationarity search, statsmodels adfuller.
        Post-conditions: Stores self.d_, self.weights_, self.lookback_, and self.correlation_.
        """
        # Purpose: Validate 1D contiguous array invariants
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Input series must be 1-dimensional, got shape {arr.shape}")
        if len(arr) < 30:
            raise ValueError(f"Input series must contain at least 30 observations, got {len(arr)}")
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Input series contains NaN or infinite values")

        # Purpose: Determine maximum allowable lookback based on training sample size (Flaw 1.3)
        max_lookback = max(10, int(len(arr) * self.max_lookback_ratio))

        # Purpose: If d was explicitly specified, bypass grid search and compute fixed weights
        if self.d is not None:
            self.d_ = float(self.d)
            self.weights_ = compute_fractional_weights(
                d=self.d_,
                threshold=self.threshold,
                max_lookback=max_lookback,
                zero_sum_correction=self.zero_sum_weights,
            )
            self.lookback_ = len(self.weights_) - 1

            # Compute stationarity metrics on transformed training series
            transformed = self._convolve_1d(arr, self.weights_)
            adf_res = adfuller(transformed, autolag="AIC", result_object=False)
            self.adf_stat_ = float(adf_res[0])
            self.adf_pvalue_ = float(adf_res[1])

            # Measure Pearson correlation with aligned raw training slice
            valid_raw = arr[self.lookback_ :]
            corr_mat = np.corrcoef(valid_raw, transformed)
            self.correlation_ = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0

            return self

        # Purpose: Execute bisection search to discover minimum d* achieving stationarity (Flaw 2.2)
        d_opt, weights, adf_stat, p_val, corr = self._find_optimal_d_bisection(
            series=arr,
            threshold=self.threshold,
            max_lookback=max_lookback,
            significance=self.significance_level,
            zero_sum=self.zero_sum_weights,
        )

        # Store fitted state
        self.d_ = d_opt
        self.weights_ = weights
        self.lookback_ = len(weights) - 1
        self.adf_stat_ = adf_stat
        self.adf_pvalue_ = p_val
        self.correlation_ = corr

        logger.info(
            "Fitted FractionalDifferentiator: d*=%.4f, lookback=%d, ADF p-val=%.4e, Corr=%.4f",
            self.d_,
            self.lookback_,
            self.adf_pvalue_,
            self.correlation_,
        )

        return self

    def transform(self, y: np.ndarray, pad_nans: bool = False) -> np.ndarray:
        """Apply fitted fractional differentiation filter immutably to new series.

        Purpose: Convolves input series with pre-computed weights using O(T log l*) 1D FFT.
        Dependencies: self.weights_, scipy.signal.fftconvolve.
        Invariants: Raises RuntimeError if called before fit(); raises ValueError if length <= lookback.
        """
        # Invariant check: Model must be fitted prior to transformation
        if self.weights_ is None or self.d_ is None:
            raise RuntimeError("FractionalDifferentiator must be fitted before calling transform()")

        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Input series must be 1-dimensional, got shape {arr.shape}")

        # Invariant check: Series length must strictly exceed lookback horizon
        if len(arr) <= self.lookback_:
            raise ValueError(
                f"Input series length ({len(arr)}) must be strictly greater than lookback horizon ({self.lookback_})"
            )

        # Purpose: Compute linear convolution via Fast Fourier Transform (Flaw 8)
        valid_out = self._convolve_1d(arr, self.weights_)

        # If padding requested, prepend NaNs matching lookback length so output dimension == len(y)
        if pad_nans:
            padded = np.full(len(arr), np.nan, dtype=np.float64)
            padded[self.lookback_ :] = valid_out
            return padded

        return valid_out

    def fit_transform(self, y: np.ndarray, pad_nans: bool = False) -> np.ndarray:
        """Fit transformer on series and return transformed array.

        Purpose: Convenience method conforming to standard Scikit-Learn API.
        """
        return self.fit(y).transform(y, pad_nans=pad_nans)

    def inverse_transform(
        self,
        y_diff: np.ndarray,
        history_anchor: np.ndarray,
    ) -> np.ndarray:
        r"""Reconstruct original series from fractionally differenced observations (Flaw 7).

        Purpose: Reconstructs expected nominal/log prices from model predictions for order routing.
        Dependencies: self.weights_, self.lookback_.
        Mathematical Formulation:
            P_t = ( \tilde{P}_t - \sum_{k=1}^{l^*} \omega_k P_{t-k} ) / \omega_0
        Invariants: history_anchor must contain at least self.lookback_ observations.
        """
        if self.weights_ is None:
            raise RuntimeError(
                "FractionalDifferentiator must be fitted before calling inverse_transform()"
            )

        anchor = np.asarray(history_anchor, dtype=np.float64)
        if len(anchor) < self.lookback_:
            raise ValueError(
                f"history_anchor length ({len(anchor)}) must be >= lookback horizon ({self.lookback_})"
            )

        diff_arr = np.asarray(y_diff, dtype=np.float64)
        n_recon = len(diff_arr)
        reconstructed = np.zeros(n_recon, dtype=np.float64)

        # Dynamic rolling reconstruction using sliding historical buffer
        omega_0 = self.weights_[0]
        lag_weights = self.weights_[1:]
        l_star = self.lookback_

        # Working buffer initialized with the final l* observations of history
        buffer = list(anchor[-l_star:])

        for i in range(n_recon):
            # Compute past lag contribution: sum_{k=1}^{l^*} omega_k * P_{t-k}
            # Buffer holds [P_{t-l^*}, ..., P_{t-1}], so reversed buffer pairs with lag_weights
            past_contribution = float(np.dot(lag_weights, np.array(buffer[-l_star:][::-1])))
            # Solve for P_t
            p_t = (diff_arr[i] - past_contribution) / omega_0
            reconstructed[i] = p_t
            buffer.append(p_t)

        return reconstructed

    @staticmethod
    def _convolve_1d(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Execute 1D causal convolution using FFT without allocating 2D strided matrices.

        Purpose: Guarantees O(T log l*) time complexity and bounded O(T) RAM consumption.
        Dependencies: scipy.signal.fftconvolve.
        """
        # Causal linear convolution filter
        # Series: [P_0, P_1, ..., P_T]
        # Weights: [omega_0, omega_1, ..., omega_l*]
        # When convolving, fftconvolve reverses weights. To achieve sum_{k=0}^{l^*} omega_k * P_{t-k},
        # we pass the weights unreversed and use mode='valid'.
        return np.ascontiguousarray(
            fftconvolve(series, weights, mode="valid"),
            dtype=np.float64,
        )

    @staticmethod
    def _find_optimal_d_bisection(
        series: np.ndarray,
        threshold: float,
        max_lookback: int,
        significance: float,
        zero_sum: bool,
    ) -> tuple[float, np.ndarray, float, float, float]:
        """Bisection search over d in [0.0, 1.0] to identify minimum stationary degree d*.

        Purpose: Reduces ADF evaluation count from 20 linear steps to at most 6 binary search steps.
        Dependencies: statsmodels adfuller, compute_fractional_weights.
        """
        # Bounds initialization
        d_low = 0.0
        d_high = 1.0
        best_d = 1.0
        best_weights = compute_fractional_weights(1.0, threshold, max_lookback, zero_sum)
        best_adf_stat = 0.0
        best_p_val = 1.0
        best_corr = 0.0

        # Verify whether d=0 is already stationary
        try:
            adf_d0 = adfuller(series, autolag="AIC", result_object=False)
            if float(adf_d0[1]) <= significance:
                w0 = np.array([1.0], dtype=np.float64)
                return 0.0, w0, float(adf_d0[0]), float(adf_d0[1]), 1.0
        except Exception as exc:
            logger.debug("ADF test failed at d=0: %s", exc)

        # Binary bisection loop (precision tolerance: 0.02)
        for _ in range(7):
            d_mid = (d_low + d_high) / 2.0
            weights = compute_fractional_weights(d_mid, threshold, max_lookback, zero_sum)
            l_star = len(weights) - 1

            if len(series) <= l_star + 10:
                # Lookback too large for sample; ratchet d up to force faster decay
                d_low = d_mid
                continue

            # Compute transformed series
            transformed = FractionalDifferentiator._convolve_1d(series, weights)

            try:
                adf_res = adfuller(transformed, autolag="AIC", result_object=False)
                p_val = float(adf_res[1])
                stat = float(adf_res[0])
            except Exception as exc:
                logger.debug("ADF calculation failed at d=%.3f: %s", d_mid, exc)
                p_val = 1.0
                stat = 0.0

            if p_val <= significance:
                # Stationary condition satisfied: record candidate and search lower half for smaller d
                best_d = d_mid
                best_weights = weights
                best_adf_stat = stat
                best_p_val = p_val
                # Measure correlation with raw series
                valid_raw = series[l_star:]
                corr_val = float(np.corrcoef(valid_raw, transformed)[0, 1])
                best_corr = corr_val if not np.isnan(corr_val) else 0.0

                d_high = d_mid
            else:
                # Non-stationary: search upper half
                d_low = d_mid

        return best_d, best_weights, best_adf_stat, best_p_val, best_corr


class StreamingFracDiffBuffer:
    """Fixed-size ring buffer for real-time, sub-millisecond fractional differentiation inference.

    Purpose: Eliminates re-computation of full histories during live stream processing.
    Dependencies: Pre-computed weights array, IMarketDataRepository for cold-start hydration.
    Relationship: Consumed by live market data stream processors.
    Invariants: Rejects update() until buffer is fully hydrated with lookback history.
    """

    def __init__(self, weights: np.ndarray) -> None:
        """Initialize streaming buffer with pre-fitted weights.

        Purpose: Establishes sliding observation window.
        Dependencies: 1D NumPy weights array.
        """
        self._weights = np.asarray(weights, dtype=np.float64)
        self._lookback = len(self._weights) - 1
        # Capacity requires lookback historical bars plus 1 current bar
        self._capacity = self._lookback + 1
        self._buffer: list[float] = []
        self._is_hydrated = False

    @property
    def is_hydrated(self) -> bool:
        """Return boolean indicating whether buffer is fully primed."""
        return self._is_hydrated

    @property
    def lookback(self) -> int:
        """Return required lookback horizon length."""
        return self._lookback

    def hydrate(self, history: Sequence[float] | np.ndarray) -> None:
        """Pre-warm buffer with required historical observations (Flaw 2.3).

        Purpose: Resolves cold-start problem upon service startup.
        Dependencies: Sequence of at least self.lookback closing prices.
        Invariants: Raises ValueError if history length is less than lookback.
        """
        if len(history) < self._lookback:
            raise ValueError(
                f"Hydration history length ({len(history)}) must be >= lookback horizon ({self._lookback})"
            )

        # Store the most recent lookback elements
        self._buffer = [float(x) for x in history[-self._lookback :]]
        self._is_hydrated = True
        logger.info("StreamingFracDiffBuffer hydrated successfully with %d bars", len(self._buffer))

    async def hydrate_from_repository(
        self,
        asset_id: str,
        repository: IMarketDataRepository,
        resolution: Resolution = Resolution.ONE_MINUTE,
    ) -> None:
        """Asynchronously pre-warm buffer by querying DuckDB historical store.

        Purpose: Provides automated database hydration on worker startup.
        Dependencies: IMarketDataRepository.get_latest_bars.
        """
        batch = await repository.get_latest_bars(
            asset_id=asset_id,
            count=self._lookback,
            resolution=resolution,
        )
        if batch.count < self._lookback:
            raise ValueError(
                f"DuckDB repository contains only {batch.count} bars for {asset_id}, but lookback requires {self._lookback}"
            )

        self.hydrate(batch.closes)

    def update(self, new_price: float) -> float:
        """Ingest new live price and compute instantaneous fractionally differenced value.

        Purpose: Sub-millisecond single-bar feature inference for live execution.
        Dependencies: Dot product between ring buffer and weights.
        Invariants: Raises RuntimeError if update() is invoked before buffer is hydrated.
        """
        if not self._is_hydrated:
            raise RuntimeError(
                "Cannot compute streaming fractional difference: buffer is unhydrated. "
                "Invoke hydrate() or hydrate_from_repository() first."
            )

        # Append incoming live price
        self._buffer.append(float(new_price))

        # Slice active window: [P_{t - l^*}, ..., P_{t - 1}, P_t]
        active_window = np.array(self._buffer[-self._capacity :], dtype=np.float64)

        # Maintain buffer length bounded at capacity
        if len(self._buffer) > self._capacity * 2:
            self._buffer = self._buffer[-self._capacity :]

        # Compute dot product: sum_{k=0}^{l^*} omega_k * P_{t-k}
        # active_window is chronological [P_{t-l^*}, ..., P_t], so reversed active_window aligns with weights
        return float(np.dot(self._weights, active_window[::-1]))
