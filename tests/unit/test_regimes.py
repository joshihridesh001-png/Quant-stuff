"""Unit tests for Causal Bayesian Jump-Regime Estimator and Thermodynamic Ambiguity Module.

Purpose:
    Validates all mathematical invariants, boundary conditions, CUSUM jump detection,
    OAS covariance conditioning, dwell time hysteresis, and temperature calibration
    for src/quant/analytics/regimes.py.

Invariants Verified:
    1. Configuration parameter bounds and defensive error handling (ERR-GAME-REGIME-PARAM).
    2. Simplex probability property: sum(pi_t) == 1.0 and pi_t >= 0.
    3. Positive-definite covariance: all eigenvalues >= shrinkage_floor > 0.
    4. CUSUM jump detection: fires on cumulative standardized deviations and shifts to panic.
    5. Dwell time hysteresis: prevents rapid oscillation across adjacent bars.
    6. Temperature bounds: beta_t strictly clamped within [min_temp, max_temp].
    7. Dimensionality validation (ERR-GAME-REGIME-DIM and ERR-GAME-REGIME-SAMPLE).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from quant.analytics.regimes import (
    ERR_REGIME_DIM,
    ERR_REGIME_PARAM,
    ERR_REGIME_SAMPLE,
    CausalBayesianRegimeFilter,
    CUSUMJumpDetector,
    OASCovarianceEstimator,
    RegimeConfig,
    RegimeEstimationResult,
)


class TestRegimeConfig:
    """Test suite for configuration validation and boundary invariants."""

    def test_default_config_valid(self) -> None:
        """Verify default configuration instantiates without error."""
        config = RegimeConfig()
        assert config.n_regimes == 3
        assert config.cusum_threshold == 3.0
        assert config.cusum_drift == 0.5
        assert config.min_dwell_bars == 3
        assert config.shrinkage_floor == 1e-5

    def test_invalid_n_regimes(self) -> None:
        """Verify n_regimes < 2 raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(n_regimes=1)

    def test_invalid_cusum_threshold(self) -> None:
        """Verify non-positive cusum_threshold raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(cusum_threshold=0.0)
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(cusum_threshold=-1.0)

    def test_invalid_cusum_drift(self) -> None:
        """Verify negative cusum_drift raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(cusum_drift=-0.1)

    def test_invalid_min_dwell(self) -> None:
        """Verify min_dwell_bars < 1 raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(min_dwell_bars=0)

    def test_invalid_shrinkage_floor(self) -> None:
        """Verify non-positive shrinkage floor raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(shrinkage_floor=0.0)

    def test_invalid_confidence_level(self) -> None:
        """Verify confidence level outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(confidence_level=0.0)
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(confidence_level=1.0)

    def test_invalid_temperature_bounds(self) -> None:
        """Verify min_temp >= max_temp or non-positive min_temp raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(min_temp=-0.1)
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(min_temp=5.0, max_temp=2.0)

    def test_invalid_window_size(self) -> None:
        """Verify window_size < 10 raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(window_size=5)

    def test_invalid_panic_shock_prior(self) -> None:
        """Verify panic_shock_prior outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(panic_shock_prior=0.0)
        with pytest.raises(ValueError, match=ERR_REGIME_PARAM):
            RegimeConfig(panic_shock_prior=1.5)


class TestCUSUMJumpDetector:
    """Test suite for the two-sided CUSUM shock detector and dwell time hysteresis."""

    def test_positive_shock_detection(self) -> None:
        """Verify cumulative positive returns trigger positive shock alarm."""
        detector = CUSUMJumpDetector(threshold=3.0, drift=0.5, min_dwell_bars=2)
        s_pos, s_neg = 0.0, 0.0

        # Bar 1: standardized return +2.0 -> s_pos = 1.5
        alarm, is_neg, s_pos, s_neg = detector.update(
            2.0, current_dwell=5, s_pos=s_pos, s_neg=s_neg
        )
        assert not alarm
        assert s_pos == 1.5
        assert s_neg == 0.0

        # Bar 2: standardized return +2.5 -> s_pos = 1.5 + 2.5 - 0.5 = 3.5 >= 3.0 -> alarm!
        alarm, is_neg, s_pos, s_neg = detector.update(
            2.5, current_dwell=5, s_pos=s_pos, s_neg=s_neg
        )
        assert alarm
        assert not is_neg
        assert s_pos == 0.0  # Reset after alarm

    def test_negative_shock_detection(self) -> None:
        """Verify cumulative negative returns trigger downward panic alarm."""
        detector = CUSUMJumpDetector(threshold=3.0, drift=0.5, min_dwell_bars=2)
        s_pos, s_neg = 0.0, 0.0

        # Severe selloff innovation: -4.0 -> s_neg = 0 + 4.0 - 0.5 = 3.5 >= 3.0
        alarm, is_neg, s_pos, s_neg = detector.update(
            -4.0, current_dwell=5, s_pos=s_pos, s_neg=s_neg
        )
        assert alarm
        assert is_neg
        assert s_neg == 0.0

    def test_dwell_time_hysteresis_suppression(self) -> None:
        """Verify CUSUM alarm is suppressed if dwell time < min_dwell_bars."""
        detector = CUSUMJumpDetector(threshold=3.0, drift=0.5, min_dwell_bars=3)
        s_pos, s_neg = 0.0, 0.0

        # Shock exceeds threshold, but current_dwell is only 1 (< 3)
        alarm, is_neg, s_pos, s_neg = detector.update(
            -4.0, current_dwell=1, s_pos=s_pos, s_neg=s_neg
        )
        assert not alarm
        assert s_neg == 3.5  # Accumulated, but suppressed from firing

        # When dwell reaches 3, next shock fires immediately
        alarm, is_neg, s_pos, s_neg = detector.update(
            -0.1, current_dwell=3, s_pos=s_pos, s_neg=s_neg
        )
        assert alarm
        assert is_neg


class TestOASCovarianceEstimator:
    """Test suite for Oracle Approximating Shrinkage and spectral floor projection."""

    def test_positive_definiteness_and_symmetry(self) -> None:
        """Verify OAS covariance is strictly symmetric and positive-definite."""
        rng = np.random.RandomState(42)
        returns = rng.randn(100, 5) * 0.02
        estimator = OASCovarianceEstimator(shrinkage_floor=1e-5)
        cov = estimator.fit_covariance(returns)

        assert cov.shape == (5, 5)
        # Symmetry check
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

        # Eigenvalue check
        eigenvalues = scipy.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= 1e-5)

    def test_collinear_degenerate_data(self) -> None:
        """Verify perfectly collinear data is regularized above eigenvalue floor."""
        # 3 assets where asset 2 and 3 are exact multiples of asset 1
        base = np.linspace(-0.05, 0.05, 50)
        returns = np.column_stack([base, base * 2.0, base * -1.5])
        estimator = OASCovarianceEstimator(shrinkage_floor=1e-4)
        cov = estimator.fit_covariance(returns)

        eigenvalues = scipy.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= 1e-4)
        assert np.all(np.isfinite(cov))

    def test_small_sample_fallback(self) -> None:
        """Verify sample length T < 2 yields valid diagonal matrix."""
        estimator = OASCovarianceEstimator(shrinkage_floor=1e-5)
        returns = np.array([[0.01, -0.02, 0.03]])
        cov = estimator.fit_covariance(returns)

        assert cov.shape == (3, 3)
        assert np.all(np.diag(cov) >= 1e-5)

    def test_zero_samples_fallback(self) -> None:
        """Verify empty sample window returns scaled identity matrix."""
        estimator = OASCovarianceEstimator(shrinkage_floor=1e-5)
        returns = np.empty((0, 4))
        cov = estimator.fit_covariance(returns)

        assert cov.shape == (4, 4)
        np.testing.assert_allclose(cov, np.eye(4), atol=1e-12)


class TestCausalBayesianRegimeFilter:
    """Test suite for the full online causal Bayesian regime filtering pipeline."""

    def test_simplex_and_temperature_invariants(self) -> None:
        """Verify probabilities satisfy simplex constraint and beta is bounded."""
        config = RegimeConfig(n_regimes=3, window_size=30)
        filt = CausalBayesianRegimeFilter(config=config)

        rng = np.random.RandomState(123)
        for _ in range(50):
            ret = rng.randn(4) * 0.01
            vol = float(np.std(ret) + 0.005)
            res = filt.step(ret, vol)

            assert isinstance(res, RegimeEstimationResult)
            # Simplex probability check
            assert len(res.probabilities) == 3
            assert np.all(res.probabilities >= 0.0)
            assert pytest.approx(np.sum(res.probabilities), abs=1e-10) == 1.0
            assert res.active_regime == int(np.argmax(res.probabilities))

            # Covariance positive definiteness
            for cov in res.regime_covariances:
                eigs = scipy.linalg.eigvalsh(cov)
                assert np.all(eigs >= config.shrinkage_floor)

            # Temperature bounds check
            assert config.min_temp <= res.temperature <= config.max_temp

    def test_instant_panic_shock_response(self) -> None:
        """Verify severe downward market shock immediately inflates panic probability."""
        config = RegimeConfig(
            n_regimes=3,
            cusum_threshold=2.5,
            min_dwell_bars=1,
            panic_shock_prior=0.90,
        )
        filt = CausalBayesianRegimeFilter(config=config)

        # Feed quiet calm data first to establish baseline
        rng = np.random.RandomState(42)
        for _ in range(25):
            ret = rng.randn(3) * 0.002
            filt.step(ret, 0.005)

        # Inject extreme flash crash (-6.0% with low normal volatility)
        crash_ret = np.array([-0.06, -0.055, -0.062])
        res_crash = filt.step(crash_ret, 0.005)

        # Verify CUSUM alarm triggered and panic regime (index 2) dominates
        assert res_crash.cusum_alarm
        assert res_crash.active_regime == 2
        assert res_crash.probabilities[2] > 0.75

    def test_fit_predict_batch_causal_evaluation(self) -> None:
        """Verify fit_predict processes historical matrix and respects dimensionality."""
        filt = CausalBayesianRegimeFilter()
        rng = np.random.RandomState(99)
        returns = rng.randn(40, 3) * 0.015

        results = filt.fit_predict(returns)
        assert len(results) == 40
        assert all(isinstance(r, RegimeEstimationResult) for r in results)

    def test_fit_predict_error_handling(self) -> None:
        """Verify invalid inputs raise deterministic diagnostic errors."""
        filt = CausalBayesianRegimeFilter()

        # 1D array instead of 2D
        with pytest.raises(ValueError, match=ERR_REGIME_DIM):
            filt.fit_predict(np.array([0.01, 0.02]))

        # Less than 2 rows
        with pytest.raises(ValueError, match=ERR_REGIME_SAMPLE):
            filt.fit_predict(np.array([[0.01, 0.02]]))

        # Mismatched volatility series length
        with pytest.raises(ValueError, match=ERR_REGIME_DIM):
            filt.fit_predict(np.zeros((10, 2)), volatility_series=np.zeros(5))
