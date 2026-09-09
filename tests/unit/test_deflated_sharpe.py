"""Unit tests for Deflated Sharpe Ratio (DSR) and Multiple-Testing Correction Subsystem.

Purpose: Validates non-normality moment calculations, Probabilistic Sharpe Ratio (PSR),
         Extreme Value Theory selection bias hurdle (E[max]), Minimum Backtest Length (MinBTL),
         effective independent trials (K_eff), False Discovery Rate (FDR) controls, and the
         DeflatedSharpeEngine master orchestrator.
Dependencies: pytest, numpy, quant.analytics.deflated_sharpe.
Invariants Tested:
    - Pearson kurtosis lower bound: gamma_4 >= 1.0 + gamma_3^2.
    - K=1 probit singularity guard: E[max] collapses to mean_sharpe.
    - MinBTL piecewise behavior: returns +inf when SR <= benchmark.
    - K_eff bounds: 1.0 <= K_eff <= K (1.0 for collinear, K for orthogonal).
    - Dual gate certification: (DSR >= significance_level) AND (T >= MinBTL).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.analytics.deflated_sharpe import (
    DSRConfig,
    DSRResult,
    DeflatedSharpeEngine,
    adjust_p_values_fdr,
    compute_effective_trials,
    compute_expected_max_sharpe,
    compute_min_backtest_length,
    compute_moments,
    compute_probabilistic_sharpe_ratio,
)


def test_dsr_config_validation() -> None:
    """Verify DSRConfig enforces hyperparameter domain bounds."""
    cfg = DSRConfig()
    assert cfg.significance_level == 0.95
    assert cfg.benchmark_sharpe == 0.0
    assert cfg.annualization_factor == 252.0
    assert cfg.min_sample_length == 10
    assert cfg.winsorize_quantile == 0.005
    assert cfg.fdr_method == "bh"
    assert cfg.fdr_q == 0.05

    # Invalid significance level
    with pytest.raises(ValueError, match="significance_level must be in"):
        DSRConfig(significance_level=0.50)
    with pytest.raises(ValueError, match="significance_level must be in"):
        DSRConfig(significance_level=1.0)

    # Invalid annualization factor
    with pytest.raises(ValueError, match="annualization_factor must be strictly positive"):
        DSRConfig(annualization_factor=0.0)

    # Invalid min_sample_length
    with pytest.raises(ValueError, match="min_sample_length must be at least 3"):
        DSRConfig(min_sample_length=2)

    # Invalid winsorize_quantile
    with pytest.raises(ValueError, match="winsorize_quantile must be in"):
        DSRConfig(winsorize_quantile=0.55)

    # Invalid fdr_q
    with pytest.raises(ValueError, match="fdr_q must be in"):
        DSRConfig(fdr_q=0.0)

    # Invalid fdr_method
    with pytest.raises(ValueError, match="fdr_method must be 'bh' or 'by'"):
        DSRConfig(fdr_method="invalid")  # type: ignore[arg-type]


def test_compute_moments_and_pearson_bound() -> None:
    """Verify compute_moments accurately estimates moments and enforces Pearson bound."""
    # Length < 3 rejected
    with pytest.raises(ValueError, match="must be at least 3"):
        compute_moments([0.01, 0.02])

    # Standard normal returns
    rng = np.random.default_rng(42)
    norm_returns = rng.normal(loc=0.001, scale=0.02, size=5000)

    mean_v, std_v, skew_v, kurt_v = compute_moments(norm_returns, winsorize_quantile=0.0)
    assert np.isclose(mean_v, 0.001, atol=0.002)
    assert np.isclose(std_v, 0.02, atol=0.002)
    assert abs(skew_v) < 0.15
    assert np.isclose(kurt_v, 3.0, atol=0.30)
    assert kurt_v >= 1.0 + (skew_v**2)

    # Winsorization dampens extreme artificial outlier
    returns_with_spike = norm_returns.copy()
    returns_with_spike[0] = 50.0  # Massive outlier

    _, _, _, kurt_raw = compute_moments(returns_with_spike, winsorize_quantile=0.0)
    _, _, _, kurt_winsor = compute_moments(returns_with_spike, winsorize_quantile=0.01)
    assert kurt_raw > kurt_winsor

    # Flat returns (zero variance)
    flat_returns = np.full(50, 0.05)
    f_mean, f_std, f_skew, f_kurt = compute_moments(flat_returns)
    assert np.isclose(f_mean, 0.05)
    assert f_std == 0.0
    assert f_skew == 0.0
    assert f_kurt == 3.0


def test_probabilistic_sharpe_ratio_invariants() -> None:
    """Verify Probabilistic Sharpe Ratio under symmetric and skewed/fat-tailed distributions."""
    # Length < 3 rejected
    with pytest.raises(ValueError, match="sample_length must be at least 3"):
        compute_probabilistic_sharpe_ratio(sharpe_ratio=1.0, sample_length=2)

    # At benchmark, PSR is exactly 0.50
    psr_at_benchmark = compute_probabilistic_sharpe_ratio(
        sharpe_ratio=0.0, benchmark_sharpe=0.0, skewness=0.0, kurtosis=3.0, sample_length=100
    )
    assert np.isclose(psr_at_benchmark, 0.50, atol=1e-6)

    # Above benchmark, PSR > 0.50
    psr_above = compute_probabilistic_sharpe_ratio(
        sharpe_ratio=1.5, benchmark_sharpe=0.0, skewness=0.0, kurtosis=3.0, sample_length=100
    )
    assert psr_above > 0.90

    # Below benchmark, PSR < 0.50
    psr_below = compute_probabilistic_sharpe_ratio(
        sharpe_ratio=-0.5, benchmark_sharpe=0.0, skewness=0.0, kurtosis=3.0, sample_length=100
    )
    assert psr_below < 0.50

    # Negative skewness penalizes PSR (increases estimation uncertainty)
    psr_normal = compute_probabilistic_sharpe_ratio(
        sharpe_ratio=1.5, benchmark_sharpe=0.0, skewness=0.0, kurtosis=3.0, sample_length=100
    )
    psr_neg_skew = compute_probabilistic_sharpe_ratio(
        sharpe_ratio=1.5, benchmark_sharpe=0.0, skewness=-1.5, kurtosis=3.0, sample_length=100
    )
    assert psr_neg_skew < psr_normal

    # High excess kurtosis penalizes PSR
    psr_fat_tails = compute_probabilistic_sharpe_ratio(
        sharpe_ratio=1.5, benchmark_sharpe=0.0, skewness=0.0, kurtosis=8.0, sample_length=100
    )
    assert psr_fat_tails < psr_normal


def test_expected_max_sharpe_and_k1_guard() -> None:
    """Verify Expected Maximum Sharpe Ratio across K trials under Extreme Value Theory."""
    # K=1 trial collapses to mean_sharpe without crashing (probit singularity guard)
    e_max_k1 = compute_expected_max_sharpe(n_trials=1.0, var_sharpe=0.50, mean_sharpe=0.0)
    assert e_max_k1 == 0.0

    # Zero variance collapses to mean_sharpe
    e_max_zero_var = compute_expected_max_sharpe(n_trials=100.0, var_sharpe=0.0, mean_sharpe=0.25)
    assert e_max_zero_var == 0.25

    # Monotonicity with respect to trial count K
    e_max_10 = compute_expected_max_sharpe(n_trials=10.0, var_sharpe=0.25, mean_sharpe=0.0)
    e_max_100 = compute_expected_max_sharpe(n_trials=100.0, var_sharpe=0.25, mean_sharpe=0.0)
    e_max_1000 = compute_expected_max_sharpe(n_trials=1000.0, var_sharpe=0.25, mean_sharpe=0.0)

    assert 0.0 < e_max_10 < e_max_100 < e_max_1000

    # Monotonicity with respect to cross-trial variance V
    e_max_low_var = compute_expected_max_sharpe(n_trials=50.0, var_sharpe=0.10, mean_sharpe=0.0)
    e_max_high_var = compute_expected_max_sharpe(n_trials=50.0, var_sharpe=0.50, mean_sharpe=0.0)
    assert e_max_low_var < e_max_high_var


def test_min_backtest_length_piecewise() -> None:
    """Verify Minimum Backtest Length (MinBTL) piecewise formulation."""
    # Invalid confidence level
    with pytest.raises(ValueError, match="significance_level must be in"):
        compute_min_backtest_length(sharpe_ratio=1.0, significance_level=0.50)

    # Underperforming benchmark strictly yields +infinity
    min_btl_losing = compute_min_backtest_length(sharpe_ratio=-0.2, benchmark_sharpe=0.0)
    assert min_btl_losing == float("inf")

    min_btl_equal = compute_min_backtest_length(sharpe_ratio=0.5, benchmark_sharpe=0.5)
    assert min_btl_equal == float("inf")

    # Winning strategy yields finite positive length
    min_btl_winning = compute_min_backtest_length(
        sharpe_ratio=2.0, benchmark_sharpe=0.0, skewness=0.0, kurtosis=3.0, significance_level=0.95
    )
    assert 1.0 < min_btl_winning < 100.0

    # Higher Sharpe ratio requires fewer observations to establish significance
    min_btl_high_sr = compute_min_backtest_length(sharpe_ratio=3.0, benchmark_sharpe=0.0)
    assert min_btl_high_sr < min_btl_winning

    # Higher confidence level requires more observations
    min_btl_99 = compute_min_backtest_length(
        sharpe_ratio=2.0, benchmark_sharpe=0.0, significance_level=0.99
    )
    assert min_btl_99 > min_btl_winning


def test_effective_trials_spectral_decomposition() -> None:
    """Verify Frobenius participation ratio for effective independent trial count K_eff."""
    # Non-square matrix rejected
    with pytest.raises(ValueError, match="must be square 2D"):
        compute_effective_trials(np.zeros((3, 4)))

    # K = 1 returns 1.0
    assert compute_effective_trials(np.array([[1.0]])) == 1.0

    # Orthogonal trials (Identity matrix) yields K_eff = K
    k = 5
    identity_corr = np.eye(k)
    k_eff_ortho = compute_effective_trials(identity_corr)
    assert np.isclose(k_eff_ortho, float(k))

    # Collinear / identical trials (all ones) yields K_eff = 1.0
    collinear_corr = np.ones((k, k))
    k_eff_collinear = compute_effective_trials(collinear_corr)
    assert np.isclose(k_eff_collinear, 1.0)

    # Moderate correlation: 1.0 < K_eff < K
    partial_corr = np.full((k, k), 0.5)
    np.fill_diagonal(partial_corr, 1.0)
    k_eff_partial = compute_effective_trials(partial_corr)
    assert 1.0 < k_eff_partial < float(k)


def test_false_discovery_rate_adjustments() -> None:
    """Verify Benjamini-Hochberg (BH) and Benjamini-Yekutieli (BY) FDR adjustments."""
    # Empty and single-element inputs
    adj_empty, mask_empty = adjust_p_values_fdr([])
    assert len(adj_empty) == 0 and len(mask_empty) == 0

    adj_single, mask_single = adjust_p_values_fdr([0.02], q_threshold=0.05)
    assert np.isclose(adj_single[0], 0.02) and bool(mask_single[0]) is True

    # Multi-p-value cohort
    raw_p = np.array([0.001, 0.01, 0.04, 0.20, 0.80])
    adj_bh, mask_bh = adjust_p_values_fdr(raw_p, method="bh", q_threshold=0.05)
    adj_by, mask_by = adjust_p_values_fdr(raw_p, method="by", q_threshold=0.05)

    # Adjusted p-values are >= raw p-values and monotonic with respect to rank
    assert np.all(adj_bh >= raw_p)
    assert np.all(adj_by >= adj_bh)  # BY is strictly more conservative than BH

    # Discovery masks properly identify significant entries
    assert bool(mask_bh[0]) is True
    assert bool(mask_bh[-1]) is False


def test_deflated_sharpe_engine_single_strategy_evaluation() -> None:
    """Verify DeflatedSharpeEngine evaluating a single strategy return series."""
    cfg = DSRConfig(significance_level=0.95, annualization_factor=252.0, min_sample_length=20)
    engine = DeflatedSharpeEngine(config=cfg)

    # Length < min_sample_length rejected
    with pytest.raises(ValueError, match="less than min_sample_length"):
        engine.evaluate_strategy(returns=np.ones(15))

    # Zero-variance flatline returns
    flatline_res = engine.evaluate_strategy(returns=np.zeros(30))
    assert flatline_res.sharpe_ratio == 0.0
    assert flatline_res.status == "ZERO_VARIANCE_FLATLINE"
    assert flatline_res.is_statistically_significant is False

    # High-performance genuine strategy
    rng = np.random.default_rng(123)
    daily_returns = rng.normal(loc=0.002, scale=0.01, size=252)  # ~SR 3.17 annualized

    res_single = engine.evaluate_strategy(returns=daily_returns, n_trials=1, var_trials=0.0)
    assert res_single.sharpe_ratio > 2.5
    assert res_single.probabilistic_sharpe_ratio > 0.99
    assert res_single.deflated_sharpe_ratio > 0.99
    assert res_single.is_statistically_significant is True
    assert res_single.status == "CERTIFIED_SIGNIFICANT"

    # Strategy tested under massive multiple testing (K=500, high variance)
    res_multi = engine.evaluate_strategy(
        returns=daily_returns, n_trials=500, var_trials=1.5, mean_trials=1.0
    )
    # Higher expected maximum increases hurdle, deflating DSR
    assert res_multi.expected_max_sharpe > 2.0
    assert res_multi.deflated_sharpe_ratio < res_single.deflated_sharpe_ratio


def test_deflated_sharpe_engine_cpcv_integration() -> None:
    """Verify seamless integration consuming Step 4 CPCV path Sharpe ratios."""
    engine = DeflatedSharpeEngine(DSRConfig(min_sample_length=30))

    # CPCV path Sharpes from Step 4
    cpcv_paths_sr = np.array([1.2, 1.4, 1.1, 1.3, 1.5, 1.25])

    # Sized trade returns from Step 5
    rng = np.random.default_rng(99)
    trade_returns = rng.normal(loc=0.0015, scale=0.012, size=150)

    # Length < 2 paths rejected
    with pytest.raises(ValueError, match="must have at least 2 path observations"):
        engine.evaluate_cpcv_results(cpcv_sharpes=[1.5], strategy_returns=trade_returns)

    dsr_result = engine.evaluate_cpcv_results(
        cpcv_sharpes=cpcv_paths_sr,
        strategy_returns=trade_returns,
        de_correlate_variance=True,
        cpcv_k_split=2,
        cpcv_n_splits=6,
    )

    assert isinstance(dsr_result, DSRResult)
    assert dsr_result.effective_trials == float(len(cpcv_paths_sr))
    assert dsr_result.expected_max_sharpe > 1.0
    assert 0.0 <= dsr_result.deflated_sharpe_ratio <= 1.0
    assert dsr_result.sample_length == 150


def test_deflated_sharpe_engine_cohort_evaluation() -> None:
    """Verify cohort evaluation with spectral K_eff and FDR screening."""
    engine = DeflatedSharpeEngine(DSRConfig(min_sample_length=50, fdr_q=0.05))

    rng = np.random.default_rng(42)
    t_bars = 200

    # Construct 4 candidate return streams:
    # Model 0: High alpha (genuine)
    # Model 1: High alpha correlated with Model 0
    # Model 2: Pure noise
    # Model 3: Negative alpha (losing)
    ret_0 = rng.normal(loc=0.004, scale=0.01, size=t_bars)
    ret_1 = ret_0 + rng.normal(loc=0.0, scale=0.002, size=t_bars)  # Highly correlated with 0
    ret_2 = rng.normal(loc=0.0, scale=0.01, size=t_bars)
    ret_3 = rng.normal(loc=-0.003, scale=0.01, size=t_bars)

    cohort_returns = [ret_0, ret_1, ret_2, ret_3]

    results = engine.evaluate_cohort(cohort_returns)

    assert len(results) == 4
    # Correlated candidates yield K_eff strictly less than 4.0
    assert results[0].effective_trials < 4.0
    assert results[0].effective_trials >= 1.0

    # Model 0 & 1 should pass cohort FDR certification
    assert results[0].is_statistically_significant is True
    assert results[0].status == "COHORT_FDR_CERTIFIED"

    # Noise and losing models should fail certification
    assert results[2].is_statistically_significant is False
    assert results[3].is_statistically_significant is False
