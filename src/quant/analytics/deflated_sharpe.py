"""Deflated Sharpe Ratio (DSR) & Statistical Significance Subsystem.

Purpose: Distinguishes genuine predictive trading skill from selection bias, backtest overfitting,
         and random noise by adjusting empirical Sharpe ratios for non-normality (skewness, kurtosis),
         effective independent trial count (K_eff), and sample length (MinBTL).
Dependencies: numpy, scipy.special, scipy.stats, dataclasses, logging.
Structural Relationship: Consumes out-of-sample path Sharpe ratios from Step 4 CPCV and fractional
                         Kelly trade returns from Step 5 Meta-Labeling; acts as the primary fitness
                         filter for Phase 4 Evolutionary Search.
Invariants:
    - Kurtosis is strictly bounded by the Pearson inequality: gamma_4 >= 1.0 + gamma_3^2.
    - Expected maximum Sharpe E[max] collapses to mean_sharpe when K <= 1 or var_trials <= 0.
    - MinBTL strictly evaluates to +inf when observed Sharpe does not exceed the benchmark hurdle.
    - Dual institutional certification requires: (DSR >= significance_level) AND (T >= MinBTL).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.special as sc_special
import scipy.stats as sc_stats

# Structured application logger
logger = logging.getLogger(__name__)

# Euler-Mascheroni constant for Gumbel Extreme Value Theory approximation
EULER_MASCHERONI: float = 0.5772156649015328606065120900824


@dataclass(frozen=True)
class DSRConfig:
    """Hyperparameter configuration for Deflated Sharpe Ratio evaluation.

    Purpose: Configures significance thresholds, benchmark hurdles, moment winsorization,
             and False Discovery Rate controls.
    Invariants:
        - 0.50 < significance_level < 1.0 (default 0.95 for 95% statistical confidence).
        - annualization_factor > 0.0 (default 252.0 for daily, 252*390 for 1-minute).
        - min_sample_length >= 3 (minimum observations for non-degenerate variance and moments).
        - 0.0 <= winsorize_quantile < 0.50 (tail winsorization bound for outlier protection).
        - 0.0 < fdr_q <= 1.0 (target False Discovery Rate threshold).
    """

    # Statistical significance confidence threshold (1 - alpha, default 0.95)
    significance_level: float = 0.95

    # Benchmark null hypothesis Sharpe ratio (SR*, default 0.0 for positive alpha)
    benchmark_sharpe: float = 0.0

    # Annualization multiplier (periods per year; default 252.0 daily bars)
    annualization_factor: float = 252.0

    # Minimum sample observations required before computing significance
    min_sample_length: int = 10

    # Two-sided winsorization quantile for outlier protection in moment estimation
    winsorize_quantile: float = 0.005

    # False Discovery Rate control method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)
    fdr_method: Literal["bh", "by"] = "bh"

    # Target False Discovery Rate threshold for population cohort screening
    fdr_q: float = 0.05

    def __post_init__(self) -> None:
        """Enforce strict configuration domain invariants."""
        if not (0.50 < self.significance_level < 1.0):
            raise ValueError(
                f"significance_level must be in (0.50, 1.0), got {self.significance_level}"
            )
        if self.annualization_factor <= 0.0:
            raise ValueError(
                f"annualization_factor must be strictly positive, got {self.annualization_factor}"
            )
        if self.min_sample_length < 3:
            raise ValueError(
                f"min_sample_length must be at least 3, got {self.min_sample_length}"
            )
        if not (0.0 <= self.winsorize_quantile < 0.50):
            raise ValueError(
                f"winsorize_quantile must be in [0.0, 0.50), got {self.winsorize_quantile}"
            )
        if not (0.0 < self.fdr_q <= 1.0):
            raise ValueError(f"fdr_q must be in (0.0, 1.0], got {self.fdr_q}")
        if self.fdr_method not in ("bh", "by"):
            raise ValueError(f"fdr_method must be 'bh' or 'by', got '{self.fdr_method}'")


@dataclass(frozen=True)
class DSRResult:
    """Comprehensive diagnostic container for Deflated Sharpe Ratio evaluation.

    Purpose: Encapsulates observed performance, higher statistical moments, multiple-testing
             hurdles, and institutional dual-gate certification status.
    Invariants:
        - 0.0 <= probabilistic_sharpe_ratio <= 1.0
        - 0.0 <= deflated_sharpe_ratio <= 1.0
        - effective_trials >= 1.0
        - min_backtest_length >= 1.0 (or float('inf'))
    """

    # Observed strategy Sharpe ratio (annualized if evaluated with annualization)
    sharpe_ratio: float

    # Non-normality adjusted Probabilistic Sharpe Ratio against benchmark_sharpe
    probabilistic_sharpe_ratio: float

    # Multiple-testing adjusted Deflated Sharpe Ratio against expected_max_sharpe
    deflated_sharpe_ratio: float

    # Expected maximum Sharpe ratio under the null hypothesis of selection bias
    expected_max_sharpe: float

    # Effective number of independent trials (K_eff) after correlation adjustment
    effective_trials: float

    # Sample skewness (gamma_3)
    skewness: float

    # Sample Pearson kurtosis (gamma_4, normal distribution = 3.0)
    kurtosis: float

    # Number of return observations evaluated (T)
    sample_length: int

    # Minimum backtest length required for 95% confidence (MinBTL in observations)
    min_backtest_length: float

    # Dual institutional gate: (DSR >= significance_level) AND (T >= MinBTL)
    is_statistically_significant: bool

    # Human-readable diagnostic classification
    status: str


def compute_moments(
    returns: Sequence[float] | np.ndarray,
    winsorize_quantile: float = 0.005,
) -> tuple[float, float, float, float]:
    """Compute robust sample mean, standard deviation, skewness, and Pearson kurtosis.

    Purpose: Provides stable higher-order moment estimation for financial return distributions
             by applying two-sided winsorization to suppress outlier wicks and enforcing the
             theoretical Pearson lower bound (kurtosis >= 1 + skewness^2).
    Invariants:
        - len(returns) >= 3.
        - kurtosis >= 1.0 + skewness^2.
    """
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 3:
        raise ValueError(f"Input returns length must be at least 3, got {len(arr)}")

    # Two-sided winsorization for robust outlier dampening
    if winsorize_quantile > 0.0:
        q_low = float(np.quantile(arr, winsorize_quantile))
        q_high = float(np.quantile(arr, 1.0 - winsorize_quantile))
        clipped = np.clip(arr, q_low, q_high)
    else:
        clipped = arr

    mean_val = float(np.mean(clipped))
    std_val = float(np.std(clipped, ddof=1))

    if std_val < 1e-12:
        # Zero variance: return default Gaussian baseline moments
        return mean_val, 0.0, 0.0, 3.0

    # Unbiased Fisher-Pearson skewness
    skew_val = float(sc_stats.skew(clipped, bias=False))

    # Pearson kurtosis (normal distribution = 3.0)
    kurt_val = float(sc_stats.kurtosis(clipped, fisher=False, bias=False))

    # Enforce theoretical lower bound on Pearson kurtosis: gamma_4 >= 1 + gamma_3^2
    min_kurtosis = 1.0 + (skew_val**2) + 1e-6
    if kurt_val < min_kurtosis:
        kurt_val = min_kurtosis

    return mean_val, std_val, skew_val, kurt_val


def compute_probabilistic_sharpe_ratio(
    sharpe_ratio: float,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sample_length: int = 10,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR) under non-normal returns.

    Purpose: Evaluates the probability that an observed Sharpe ratio exceeds a benchmark hurdle,
             adjusting the estimation standard error for skewness and excess kurtosis.
    Mathematical Formulation:
        sigma_SR = sqrt((1 - gamma_3 * SR + (gamma_4 - 1)/4 * SR^2) / (T - 1))
        PSR = Phi((SR - SR*) / sigma_SR)
    Invariants:
        - sample_length >= 3.
        - Returned PSR is bounded in [0.0, 1.0].
    """
    if sample_length < 3:
        raise ValueError(f"sample_length must be at least 3, got {sample_length}")

    # Enforce theoretical lower bound on kurtosis to guarantee strictly positive denominator
    min_kurt = 1.0 + (skewness**2) + 1e-6
    kurt = max(kurtosis, min_kurt)

    # Compute variance term under non-normality
    var_term = 1.0 - (skewness * sharpe_ratio) + (((kurt - 1.0) / 4.0) * (sharpe_ratio**2))
    # Defensive floor against numerical floating-point inaccuracies
    var_term = max(1e-8, var_term)

    std_error = np.sqrt(var_term / float(sample_length - 1))
    if std_error < 1e-12:
        return 1.0 if sharpe_ratio >= benchmark_sharpe else 0.0

    z_score = (sharpe_ratio - benchmark_sharpe) / std_error
    psr = float(sc_special.ndtr(z_score))
    return float(np.clip(psr, 0.0, 1.0))


def compute_expected_max_sharpe(
    n_trials: float | int,
    var_sharpe: float,
    mean_sharpe: float = 0.0,
) -> float:
    """Compute the Expected Maximum Sharpe Ratio across K trials under the null hypothesis.

    Purpose: Evaluates the expected value of the maximum Sharpe ratio achieved purely by chance
             from testing K strategy variations with empirical cross-trial variance V[{SR}].
    Mathematical Formulation:
        E[max_K {SR}] = mean_SR + sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/K) + gamma * Phi^-1(1 - 1/(K*e)))
    Invariants:
        - If n_trials <= 1.0 or var_sharpe <= 0.0, returns mean_sharpe.
        - Argument to probit function is clamped to [1e-15, 1.0 - 1e-15] to prevent infinities.
    """
    k_val = float(n_trials)
    if k_val <= 1.0 or var_sharpe <= 0.0:
        return float(mean_sharpe)

    std_sharpe = float(np.sqrt(var_sharpe))

    # Numerical probit argument clamping
    q1 = np.clip(1.0 - (1.0 / k_val), 1e-15, 1.0 - 1e-15)
    q2 = np.clip(1.0 - (1.0 / (k_val * np.e)), 1e-15, 1.0 - 1e-15)

    z1 = float(sc_special.ndtri(q1))
    z2 = float(sc_special.ndtri(q2))

    max_z = ((1.0 - EULER_MASCHERONI) * z1) + (EULER_MASCHERONI * z2)
    expected_max = mean_sharpe + (std_sharpe * max_z)
    return float(expected_max)


def compute_min_backtest_length(
    sharpe_ratio: float,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    significance_level: float = 0.95,
) -> float:
    """Compute the Minimum Backtest Length (MinBTL) required to certify statistical significance.

    Purpose: Evaluates the exact minimum number of return observations (T) required for an observed
             Sharpe ratio to reject the null hypothesis at the specified confidence level.
    Mathematical Formulation:
        MinBTL = 1 + (1 - gamma_3 * SR + (gamma_4 - 1)/4 * SR^2) * (Phi^-1(significance_level) / (SR - SR*))^2
    Invariants:
        - If sharpe_ratio <= benchmark_sharpe, MinBTL strictly evaluates to +infinity.
        - Returned value is strictly >= 1.0.
    """
    if not (0.50 < significance_level < 1.0):
        raise ValueError(
            f"significance_level must be in (0.50, 1.0), got {significance_level}"
        )

    # If the strategy fails to beat the benchmark, no finite sample length can make it significant
    if sharpe_ratio <= benchmark_sharpe:
        return float("inf")

    min_kurt = 1.0 + (skewness**2) + 1e-6
    kurt = max(kurtosis, min_kurt)

    var_term = 1.0 - (skewness * sharpe_ratio) + (((kurt - 1.0) / 4.0) * (sharpe_ratio**2))
    var_term = max(1e-8, var_term)

    z_alpha = float(sc_special.ndtri(significance_level))
    diff = sharpe_ratio - benchmark_sharpe

    min_btl = 1.0 + (var_term * ((z_alpha / diff) ** 2))
    return float(max(1.0, min_btl))


def compute_effective_trials(
    correlation_matrix: np.ndarray,
) -> float:
    """Compute the effective number of independent trials (K_eff) via the Frobenius Participation Ratio.

    Purpose: Solves the 'Independence Fallacy' by calculating the effective rank of the correlation
             matrix of trial returns without requiring an expensive O(K^3) eigenvalue decomposition.
    Mathematical Formulation:
        K_eff = (tr(C))^2 / tr(C^2) = K^2 / sum_{i,j} C_{ij}^2
    Invariants:
        - 1.0 <= K_eff <= K.
        - Identical trials (all C_ij = 1.0) yield K_eff = 1.0.
        - Orthogonal trials (C = Identity) yield K_eff = K.
    """
    c_mat = np.asarray(correlation_matrix, dtype=float)
    if c_mat.ndim != 2 or c_mat.shape[0] != c_mat.shape[1]:
        raise ValueError(f"correlation_matrix must be square 2D, got shape {c_mat.shape}")

    k_total = c_mat.shape[0]
    if k_total <= 1:
        return 1.0

    # Frobenius norm squared: sum_{i,j} C_ij^2 = tr(C^2)
    sum_c_sq = float(np.sum(c_mat**2))
    if sum_c_sq <= 0.0:
        return float(k_total)

    k_eff = (float(k_total) ** 2) / sum_c_sq
    return float(np.clip(k_eff, 1.0, float(k_total)))


def adjust_p_values_fdr(
    p_values: Sequence[float] | np.ndarray,
    method: Literal["bh", "by"] = "bh",
    q_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust p-values using the Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY) FDR procedure.

    Purpose: Controls the False Discovery Rate across an entire cohort of candidate strategies
             in population screening, preventing multiple-testing false discoveries.
    Mathematical Formulation:
        BH: p_(i)^adj = min_{j >= i} (M / j * p_(j))
        BY: p_(i)^adj = min_{j >= i} (M * c(M) / j * p_(j)), where c(M) = sum_{k=1}^M (1 / k)
    Invariants:
        - Adjusted p-values are monotonic with respect to rank and bounded in [0.0, 1.0].
        - Returns boolean discovery mask where adjusted_p <= q_threshold.
    """
    p_arr = np.asarray(p_values, dtype=float)
    m_count = len(p_arr)
    if m_count == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    if m_count == 1:
        adj = np.clip(p_arr, 0.0, 1.0)
        return adj, adj <= q_threshold

    # Sort p-values in ascending order
    sort_idx = np.argsort(p_arr)
    sorted_p = p_arr[sort_idx]

    ranks = np.arange(1, m_count + 1, dtype=float)

    if method == "by":
        # Harmonic series sum c(M) = sum_{k=1}^M (1/k)
        c_m = float(np.sum(1.0 / ranks))
        adjusted_sorted = sorted_p * (float(m_count) * c_m / ranks)
    else:  # 'bh'
        adjusted_sorted = sorted_p * (float(m_count) / ranks)

    # Monotonic stepdown accumulation: min_{j >= i} p_j
    accumulated = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    accumulated = np.clip(accumulated, 0.0, 1.0)

    # Invert sorting to match original input order
    original_idx = np.empty_like(sort_idx)
    original_idx[sort_idx] = np.arange(m_count)
    adjusted_p_values = accumulated[original_idx]

    is_significant = adjusted_p_values <= q_threshold
    return adjusted_p_values, is_significant


class DeflatedSharpeEngine:
    """Master institutional validation engine for Deflated Sharpe Ratio and multi-testing corrections.

    Purpose: Coordinates robust moment extraction, non-normality adjustment, Extreme Value Theory
             selection bias hurdles, and dual institutional gate certification.
    Structural Relationship: Consumes backtest paths from Step 4 CPCV and trade returns from Step 5
                             Meta-Labeling; outputs certified fitness metrics for Phase 4.
    """

    def __init__(self, config: DSRConfig | None = None) -> None:
        """Initialize the DeflatedSharpeEngine with configuration."""
        self._config = config or DSRConfig()

    @property
    def config(self) -> DSRConfig:
        """Return the immutable engine configuration."""
        return self._config

    def evaluate_strategy(
        self,
        returns: Sequence[float] | np.ndarray,
        n_trials: float | int = 1,
        var_trials: float = 0.0,
        mean_trials: float = 0.0,
        benchmark_sharpe: float | None = None,
        annualize: bool = True,
    ) -> DSRResult:
        """Evaluate a strategy return series under the Deflated Sharpe Ratio framework.

        Purpose: Computes observed Sharpe ratio, higher-order moments, non-normality PSR,
                 Extreme Value Theory selection bias hurdle, DSR, and MinBTL.
        Invariants:
            - len(returns) >= min_sample_length.
            - Strategy is certified significant only if DSR >= significance_level AND T >= MinBTL.
        """
        arr = np.asarray(returns, dtype=float)
        sample_len = len(arr)
        if sample_len < self._config.min_sample_length:
            raise ValueError(
                f"Sample length ({sample_len}) is less than min_sample_length ({self._config.min_sample_length})"
            )

        mean_val, std_val, skew_val, kurt_val = compute_moments(
            arr, winsorize_quantile=self._config.winsorize_quantile
        )

        # Baseline hurdle: custom override or config default
        sr_hurdle = (
            benchmark_sharpe
            if benchmark_sharpe is not None
            else self._config.benchmark_sharpe
        )

        # Flatline / zero variance guard
        if std_val < 1e-12:
            return DSRResult(
                sharpe_ratio=0.0,
                probabilistic_sharpe_ratio=0.0,
                deflated_sharpe_ratio=0.0,
                expected_max_sharpe=sr_hurdle,
                effective_trials=float(n_trials),
                skewness=0.0,
                kurtosis=3.0,
                sample_length=sample_len,
                min_backtest_length=float("inf"),
                is_statistically_significant=False,
                status="ZERO_VARIANCE_FLATLINE",
            )

        # Calculate per-bar Sharpe ratio
        sr_bar = mean_val / std_val

        if annualize:
            ann_factor = self._config.annualization_factor
            ann_multiplier = np.sqrt(ann_factor)
            observed_sr = sr_bar * ann_multiplier
        else:
            observed_sr = sr_bar

        # Compute Probabilistic Sharpe Ratio against benchmark hurdle
        psr = compute_probabilistic_sharpe_ratio(
            sharpe_ratio=observed_sr,
            benchmark_sharpe=sr_hurdle,
            skewness=skew_val,
            kurtosis=kurt_val,
            sample_length=sample_len,
        )

        # Compute Expected Maximum Sharpe Ratio under null hypothesis of selection bias
        e_max = compute_expected_max_sharpe(
            n_trials=n_trials,
            var_sharpe=var_trials,
            mean_sharpe=mean_trials if mean_trials != 0.0 else sr_hurdle,
        )

        # Deflated Sharpe Ratio evaluates PSR using e_max as the hurdle
        dsr = compute_probabilistic_sharpe_ratio(
            sharpe_ratio=observed_sr,
            benchmark_sharpe=e_max,
            skewness=skew_val,
            kurtosis=kurt_val,
            sample_length=sample_len,
        )

        # Compute Minimum Backtest Length (in return observations)
        min_btl = compute_min_backtest_length(
            sharpe_ratio=observed_sr,
            benchmark_sharpe=e_max,
            skewness=skew_val,
            kurtosis=kurt_val,
            significance_level=self._config.significance_level,
        )

        # Dual Institutional Gate
        is_significant = (dsr >= self._config.significance_level) and (
            sample_len >= min_btl
        )

        if is_significant:
            status = "CERTIFIED_SIGNIFICANT"
        elif dsr >= self._config.significance_level and sample_len < min_btl:
            status = "INSUFFICIENT_SAMPLE_LENGTH"
        elif dsr < self._config.significance_level and observed_sr > e_max:
            status = "INSUFFICIENT_CONFIDENCE"
        else:
            status = "SELECTION_BIAS_REJECTED"

        return DSRResult(
            sharpe_ratio=float(observed_sr),
            probabilistic_sharpe_ratio=float(psr),
            deflated_sharpe_ratio=float(dsr),
            expected_max_sharpe=float(e_max),
            effective_trials=float(n_trials),
            skewness=float(skew_val),
            kurtosis=float(kurt_val),
            sample_length=sample_len,
            min_backtest_length=float(min_btl),
            is_statistically_significant=is_significant,
            status=status,
        )

    def evaluate_cpcv_results(
        self,
        cpcv_sharpes: Sequence[float] | np.ndarray,
        strategy_returns: Sequence[float] | np.ndarray,
        n_trials: float | int | None = None,
        de_correlate_variance: bool = True,
        cpcv_k_split: int = 2,
        cpcv_n_splits: int = 6,
    ) -> DSRResult:
        """Evaluate strategy returns directly against Step 4 CPCV path Sharpe ratio outputs.

        Purpose: Provides seamless end-to-end integration between Step 4 Combinatorial Purged CV
                 and Step 6 Deflated Sharpe Ratio, automatically extracting the empirical variance
                 of reconstructed out-of-sample paths and applying the CPCV de-correlation factor.
        Invariants:
            - len(cpcv_sharpes) >= 2.
        """
        sharpes_arr = np.asarray(cpcv_sharpes, dtype=float)
        if len(sharpes_arr) < 2:
            raise ValueError(
                f"cpcv_sharpes must have at least 2 path observations, got {len(sharpes_arr)}"
            )

        mean_path_sr = float(np.mean(sharpes_arr))
        var_path_sr = float(np.var(sharpes_arr, ddof=1))

        if de_correlate_variance and cpcv_n_splits > 1:
            # Theoretical average pairwise overlap between CPCV combinatorial paths: rho ~= (k - 1) / (N - 1)
            rho_cpcv = min(
                0.90, max(0.0, float(cpcv_k_split - 1) / float(cpcv_n_splits - 1))
            )
            # De-correlate variance: V* = V / (1 - rho)
            var_path_sr = var_path_sr / max(0.10, 1.0 - rho_cpcv)

        trial_count = float(n_trials) if n_trials is not None else float(len(sharpes_arr))

        return self.evaluate_strategy(
            returns=strategy_returns,
            n_trials=trial_count,
            var_trials=var_path_sr,
            mean_trials=mean_path_sr,
            annualize=True,
        )

    def evaluate_cohort(
        self,
        candidate_returns: Sequence[Sequence[float] | np.ndarray],
        benchmark_sharpe: float | None = None,
    ) -> list[DSRResult]:
        """Evaluate an entire population cohort of candidate strategies with FDR control.

        Purpose: Evaluates multiple strategies tested concurrently, constructing the empirical
                 return correlation matrix, computing K_eff, evaluating individual DSR results,
                 and applying Benjamini-Hochberg / Benjamini-Yekutieli FDR filtering.
        Invariants:
            - len(candidate_returns) >= 1.
            - All candidate return series must have identical length T >= min_sample_length.
        """
        k_count = len(candidate_returns)
        if k_count == 0:
            return []

        # Convert to aligned matrix: shape (T, K)
        mat = np.column_stack([np.asarray(r, dtype=float) for r in candidate_returns])
        t_len, k_len = mat.shape

        if t_len < self._config.min_sample_length:
            raise ValueError(
                f"Candidate returns length ({t_len}) is less than min_sample_length ({self._config.min_sample_length})"
            )

        if k_len == 1:
            single_res = self.evaluate_strategy(
                returns=mat[:, 0],
                n_trials=1.0,
                var_trials=0.0,
                benchmark_sharpe=benchmark_sharpe,
            )
            return [single_res]

        # Compute return correlation matrix across candidate strategies
        corr_matrix = np.corrcoef(mat, rowvar=False)
        # Replace any NaNs from constant columns with zero correlation
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # Compute effective number of independent trials
        k_eff = compute_effective_trials(corr_matrix)

        # Calculate empirical Sharpe ratios across all candidates to obtain cross-sectional variance
        candidate_sharpes: list[float] = []
        for i in range(k_len):
            m_i, s_i, _, _ = compute_moments(
                mat[:, i], winsorize_quantile=self._config.winsorize_quantile
            )
            sr_i = (
                (m_i / s_i) * np.sqrt(self._config.annualization_factor)
                if s_i > 1e-12
                else 0.0
            )
            candidate_sharpes.append(sr_i)

        var_sharpes = float(np.var(candidate_sharpes, ddof=1)) if k_len > 1 else 0.0
        mean_sharpes = float(np.mean(candidate_sharpes))

        # Evaluate each strategy with K_eff and cross-sectional variance
        initial_results: list[DSRResult] = []
        p_values: list[float] = []

        null_mean = benchmark_sharpe if benchmark_sharpe is not None else self._config.benchmark_sharpe

        for i in range(k_len):
            res = self.evaluate_strategy(
                returns=mat[:, i],
                n_trials=k_eff,
                var_trials=var_sharpes,
                mean_trials=null_mean,
                benchmark_sharpe=benchmark_sharpe,
                annualize=True,
            )
            initial_results.append(res)
            # Two-sided empirical p-value from DSR: p = 1 - DSR
            p_values.append(1.0 - res.deflated_sharpe_ratio)

        # Apply False Discovery Rate stepdown adjustment
        _, fdr_mask = adjust_p_values_fdr(
            p_values=p_values,
            method=self._config.fdr_method,
            q_threshold=self._config.fdr_q,
        )

        final_results: list[DSRResult] = []
        for i, res in enumerate(initial_results):
            # Candidate passes cohort certification if it clears individual DSR dual gate AND FDR mask
            cohort_significant = res.is_statistically_significant and bool(fdr_mask[i])
            cohort_status = (
                "COHORT_FDR_CERTIFIED"
                if cohort_significant
                else (
                    "FDR_REJECTED"
                    if res.is_statistically_significant
                    else res.status
                )
            )
            updated_res = DSRResult(
                sharpe_ratio=res.sharpe_ratio,
                probabilistic_sharpe_ratio=res.probabilistic_sharpe_ratio,
                deflated_sharpe_ratio=res.deflated_sharpe_ratio,
                expected_max_sharpe=res.expected_max_sharpe,
                effective_trials=res.effective_trials,
                skewness=res.skewness,
                kurtosis=res.kurtosis,
                sample_length=res.sample_length,
                min_backtest_length=res.min_backtest_length,
                is_statistically_significant=cohort_significant,
                status=cohort_status,
            )
            final_results.append(updated_res)

        return final_results
