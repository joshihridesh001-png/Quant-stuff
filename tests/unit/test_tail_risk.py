"""Unit tests for Semi-Parametric EVT Tail Risk domain entities, configuration, and invariants.

Validates Invariants:
- INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha within 1e-10 tolerance).
- INV-TR-002: Fréchet Tail Stability & Infinite Variance Tripwire (shape_xi in [0.001, 0.999], shape_xi >= 1.0 -> InfiniteVarianceException).
- Non-finite input protection (math.isfinite check on all scalars; reject NaN/Inf with DegenerateTailRiskException).
"""

from __future__ import annotations

import inspect
import sys
import time
from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest

from quant.analytics.tail_risk import (
    ERR_TR_DEGENERATE,
    ERR_TR_INFINITE_VARIANCE,
    ERR_TR_INVALID_CONFIG,
    ERR_TR_ORDERING,
    DegenerateTailRiskException,
    EVTTailParameters,
    InfiniteVarianceException,
    InvalidTailRiskInputException,
    ProbabilityWeightedMomentsEstimator,
    TailRiskConfig,
    TailRiskError,
    TailRiskMetrics,
)


class TestTailRiskExceptions:
    """Validate domain exception hierarchy, polymorphism, and error codes."""

    def test_exception_inheritance_tree(self) -> None:
        """Verify that all custom tail risk exceptions inherit from TailRiskError."""
        assert issubclass(TailRiskError, Exception)
        assert issubclass(DegenerateTailRiskException, TailRiskError)
        assert issubclass(InfiniteVarianceException, TailRiskError)
        assert issubclass(InvalidTailRiskInputException, TailRiskError)

    def test_exception_polymorphic_handling(self) -> None:
        """Verify exceptions can be caught polymorphically as TailRiskError."""
        degen = DegenerateTailRiskException(ERR_TR_DEGENERATE)
        inf_var = InfiniteVarianceException(ERR_TR_INFINITE_VARIANCE)
        invalid_in = InvalidTailRiskInputException(ERR_TR_INVALID_CONFIG)
        ordering_degen = DegenerateTailRiskException(ERR_TR_ORDERING)

        assert isinstance(degen, TailRiskError)
        assert isinstance(inf_var, TailRiskError)
        assert isinstance(invalid_in, TailRiskError)
        assert isinstance(ordering_degen, TailRiskError)
        assert str(degen) == ERR_TR_DEGENERATE
        assert str(inf_var) == ERR_TR_INFINITE_VARIANCE
        assert str(invalid_in) == ERR_TR_INVALID_CONFIG
        assert str(ordering_degen) == ERR_TR_ORDERING


class TestTailRiskConfig:
    """Validate TailRiskConfig defaults, immutability, range checks, and hierarchies."""

    def test_valid_default_config(self) -> None:
        """Verify institutional default values."""
        cfg = TailRiskConfig()
        assert cfg.confidence_level == 0.99
        assert cfg.threshold_k == 1.645
        assert cfg.min_observations_evt == 250
        assert cfg.min_observations_student_t == 30
        assert cfg.min_exceedances_evt == 15
        assert cfg.tail_index_lower_bound == 0.001
        assert cfg.tail_index_upper_bound == 0.999
        assert cfg.rolling_window_size == 500

    def test_frozen_immutability(self) -> None:
        """Verify TailRiskConfig is strictly immutable (frozen=True)."""
        cfg = TailRiskConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.confidence_level = 0.95  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            cfg.rolling_window_size = 1000  # type: ignore[misc]

    @pytest.mark.parametrize("invalid_alpha", [-0.1, 0.0, 0.50, 1.0, 1.05])
    def test_confidence_level_bounds(self, invalid_alpha: float) -> None:
        """Confidence level alpha must strictly satisfy 0.50 < alpha < 1.0."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(confidence_level=invalid_alpha)

    @pytest.mark.parametrize("invalid_k", [-1.0, 0.0])
    def test_threshold_k_strictly_positive(self, invalid_k: float) -> None:
        """Threshold multiplier k must be strictly positive."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(threshold_k=invalid_k)

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            (-0.01, 0.5),
            (0.0, 0.5),
            (0.5, 0.5),
            (0.8, 0.2),
            (0.2, 1.0),
            (0.2, 1.1),
        ],
    )
    def test_tail_index_bounds(self, lower: float, upper: float) -> None:
        """Tail index bounds must satisfy 0.0 < lower < upper < 1.0."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(tail_index_lower_bound=lower, tail_index_upper_bound=upper)

    def test_observation_hierarchy_student_t_vs_evt(self) -> None:
        """min_observations_student_t must be strictly less than min_observations_evt."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(min_observations_student_t=250, min_observations_evt=250)
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(min_observations_student_t=300, min_observations_evt=250)

    def test_observation_hierarchy_evt_vs_window(self) -> None:
        """min_observations_evt must be <= rolling_window_size."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(min_observations_evt=600, rolling_window_size=500)

    @pytest.mark.parametrize("min_obs", [0, -1])
    def test_minimum_observations_strictly_positive(self, min_obs: int) -> None:
        """Observation counts must be at least 1."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(min_observations_student_t=min_obs)
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(min_exceedances_evt=min_obs)

    @pytest.mark.parametrize(
        "field_name",
        ["confidence_level", "threshold_k", "tail_index_lower_bound", "tail_index_upper_bound"],
    )
    @pytest.mark.parametrize("non_finite_val", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_floats(self, field_name: str, non_finite_val: float) -> None:
        """Reject NaN and infinite floats with DegenerateTailRiskException."""
        kwargs = {field_name: non_finite_val}
        with pytest.raises(DegenerateTailRiskException):
            TailRiskConfig(**kwargs)

    @pytest.mark.parametrize(
        ("field_name", "bad_val"),
        [
            ("confidence_level", "0.99"),
            ("threshold_k", None),
            ("min_observations_evt", 250.5),
            ("min_observations_evt", True),
            ("rolling_window_size", "500"),
        ],
    )
    def test_rejects_invalid_types(self, field_name: str, bad_val: object) -> None:
        """Reject invalid types with InvalidTailRiskInputException."""
        kwargs = {field_name: bad_val}
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskConfig(**kwargs)

    def test_clamp_tail_index_behavior(self) -> None:
        """Verify tail index clamping and infinite variance tripwire."""
        cfg = TailRiskConfig(tail_index_lower_bound=0.001, tail_index_upper_bound=0.999)

        # Within bounds -> untouched
        assert cfg.clamp_tail_index(0.50) == 0.50
        # Below lower bound -> clamped to lower bound
        assert cfg.clamp_tail_index(0.0001) == 0.001
        # Above upper bound but < 1.0 -> clamped to upper bound
        assert cfg.clamp_tail_index(0.9995) == 0.999
        # Exact boundary
        assert cfg.clamp_tail_index(0.001) == 0.001
        assert cfg.clamp_tail_index(0.999) == 0.999

        # Infinite variance tripwire (xi >= 1.0)
        with pytest.raises(InfiniteVarianceException):
            cfg.clamp_tail_index(1.0)
        with pytest.raises(InfiniteVarianceException):
            cfg.clamp_tail_index(1.50)

        # Non-finite values
        with pytest.raises(DegenerateTailRiskException):
            cfg.clamp_tail_index(float("nan"))
        with pytest.raises(DegenerateTailRiskException):
            cfg.clamp_tail_index(float("inf"))


class TestEVTTailParameters:
    """Validate EVTTailParameters data container, bounds, and invariants."""

    def test_valid_instantiation(self) -> None:
        """Verify standard instantiation across supported estimation methods."""
        params_pwm = EVTTailParameters(
            threshold_u=0.02,
            shape_xi=0.25,
            scale_beta=0.015,
            num_exceedances=25,
            total_observations=500,
            method="EVT_PWM",
        )
        assert params_pwm.threshold_u == 0.02
        assert params_pwm.shape_xi == 0.25
        assert params_pwm.scale_beta == 0.015
        assert params_pwm.num_exceedances == 25
        assert params_pwm.total_observations == 500
        assert params_pwm.method == "EVT_PWM"

        params_t = EVTTailParameters(
            threshold_u=0.0,
            shape_xi=0.15,
            scale_beta=0.02,
            num_exceedances=15,
            total_observations=100,
            method="STUDENT_T",
        )
        assert params_t.method == "STUDENT_T"

        params_emp = EVTTailParameters(
            threshold_u=0.03,
            shape_xi=0.0,
            scale_beta=0.0,
            num_exceedances=5,
            total_observations=25,
            method="EMPIRICAL",
        )
        assert params_emp.method == "EMPIRICAL"
        assert params_emp.scale_beta == 0.0

    def test_frozen_immutability(self) -> None:
        """Verify EVTTailParameters is frozen and cannot be mutated."""
        params = EVTTailParameters(
            threshold_u=0.02,
            shape_xi=0.25,
            scale_beta=0.015,
            num_exceedances=25,
            total_observations=500,
            method="EVT_PWM",
        )
        with pytest.raises(FrozenInstanceError):
            params.shape_xi = 0.30  # type: ignore[misc]

    def test_invalid_method(self) -> None:
        """Verify rejection of unrecognized estimation methods."""
        with pytest.raises(InvalidTailRiskInputException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=0.25,
                scale_beta=0.015,
                num_exceedances=25,
                total_observations=500,
                method="MAXIMUM_LIKELIHOOD",
            )

    @pytest.mark.parametrize("xi_breach", [1.0, 1.0001, 1.5, 2.0])
    def test_inv_tr_002_infinite_variance_tripwire(self, xi_breach: float) -> None:
        """INV-TR-002: shape_xi >= 1.0 must trigger InfiniteVarianceException."""
        with pytest.raises(InfiniteVarianceException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=xi_breach,
                scale_beta=0.015,
                num_exceedances=25,
                total_observations=500,
                method="EVT_PWM",
            )

    def test_negative_shape_xi(self) -> None:
        """shape_xi < 0.0 must raise DegenerateTailRiskException."""
        with pytest.raises(DegenerateTailRiskException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=-0.1,
                scale_beta=0.015,
                num_exceedances=25,
                total_observations=500,
                method="EVT_PWM",
            )

    @pytest.mark.parametrize("method", ["EVT_PWM", "STUDENT_T"])
    @pytest.mark.parametrize("bad_scale", [0.0, -0.01, -1.0])
    def test_scale_beta_strictly_positive_for_parametric(
        self, method: str, bad_scale: float
    ) -> None:
        """scale_beta must be > 0.0 for EVT_PWM and STUDENT_T."""
        with pytest.raises(DegenerateTailRiskException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=0.25,
                scale_beta=bad_scale,
                num_exceedances=25,
                total_observations=500,
                method=method,
            )

    def test_scale_beta_negative_for_empirical(self) -> None:
        """scale_beta must be >= 0.0 for EMPIRICAL (0.0 allowed, negative rejected)."""
        with pytest.raises(DegenerateTailRiskException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=0.0,
                scale_beta=-0.001,
                num_exceedances=5,
                total_observations=25,
                method="EMPIRICAL",
            )

    def test_observation_count_bounds(self) -> None:
        """num_exceedances and total_observations must satisfy 0 <= N_u <= n."""
        # Negative observations
        with pytest.raises(InvalidTailRiskInputException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=0.25,
                scale_beta=0.015,
                num_exceedances=25,
                total_observations=-1,
                method="EVT_PWM",
            )
        # Negative exceedances
        with pytest.raises(InvalidTailRiskInputException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=0.25,
                scale_beta=0.015,
                num_exceedances=-1,
                total_observations=500,
                method="EVT_PWM",
            )
        # Exceedances exceed total observations
        with pytest.raises(InvalidTailRiskInputException):
            EVTTailParameters(
                threshold_u=0.02,
                shape_xi=0.25,
                scale_beta=0.015,
                num_exceedances=501,
                total_observations=500,
                method="EVT_PWM",
            )

    @pytest.mark.parametrize("field_name", ["threshold_u", "shape_xi", "scale_beta"])
    @pytest.mark.parametrize("non_finite_val", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_scalars(self, field_name: str, non_finite_val: float) -> None:
        """Reject NaN and infinite floats with DegenerateTailRiskException."""
        kwargs: dict[str, object] = {
            "threshold_u": 0.02,
            "shape_xi": 0.25,
            "scale_beta": 0.015,
            "num_exceedances": 25,
            "total_observations": 500,
            "method": "EVT_PWM",
        }
        kwargs[field_name] = non_finite_val
        with pytest.raises(DegenerateTailRiskException):
            EVTTailParameters(**kwargs)

    @pytest.mark.parametrize(
        ("field_name", "bad_val"),
        [
            ("threshold_u", "0.02"),
            ("shape_xi", None),
            ("scale_beta", True),
            ("num_exceedances", 25.0),
            ("num_exceedances", False),
            ("total_observations", "500"),
            ("method", 123),
        ],
    )
    def test_rejects_invalid_types(self, field_name: str, bad_val: object) -> None:
        """Reject invalid types with InvalidTailRiskInputException."""
        kwargs: dict[str, object] = {
            "threshold_u": 0.02,
            "shape_xi": 0.25,
            "scale_beta": 0.015,
            "num_exceedances": 25,
            "total_observations": 500,
            "method": "EVT_PWM",
        }
        kwargs[field_name] = bad_val
        with pytest.raises(InvalidTailRiskInputException):
            EVTTailParameters(**kwargs)


class TestTailRiskMetrics:
    """Validate TailRiskMetrics container and INV-TR-001 Coherent Risk Ordering."""

    @pytest.fixture
    def valid_tail_parameters(self) -> EVTTailParameters:
        return EVTTailParameters(
            threshold_u=0.02,
            shape_xi=0.25,
            scale_beta=0.015,
            num_exceedances=25,
            total_observations=500,
            method="EVT_PWM",
        )

    def test_valid_instantiation(self, valid_tail_parameters: EVTTailParameters) -> None:
        """Verify normal instantiation when CVaR >= VaR."""
        metrics = TailRiskMetrics(
            var_alpha=0.045,
            cvar_alpha=0.065,
            confidence_level=0.99,
            tail_parameters=valid_tail_parameters,
            step_index=10,
        )
        assert metrics.var_alpha == 0.045
        assert metrics.cvar_alpha == 0.065
        assert metrics.confidence_level == 0.99
        assert metrics.tail_parameters == valid_tail_parameters
        assert metrics.step_index == 10

    def test_frozen_immutability(self, valid_tail_parameters: EVTTailParameters) -> None:
        """Verify TailRiskMetrics is frozen."""
        metrics = TailRiskMetrics(
            var_alpha=0.045,
            cvar_alpha=0.065,
            confidence_level=0.99,
            tail_parameters=valid_tail_parameters,
            step_index=10,
        )
        with pytest.raises(FrozenInstanceError):
            metrics.var_alpha = 0.05  # type: ignore[misc]

    def test_inv_tr_001_coherent_risk_ordering(
        self, valid_tail_parameters: EVTTailParameters
    ) -> None:
        """INV-TR-001: cvar_alpha >= var_alpha within 1e-10 tolerance."""
        var = 0.050

        # Case 1: Strictly greater -> Valid
        metrics_strict = TailRiskMetrics(
            var_alpha=var,
            cvar_alpha=0.055,
            confidence_level=0.99,
            tail_parameters=valid_tail_parameters,
            step_index=1,
        )
        assert metrics_strict.cvar_alpha > metrics_strict.var_alpha

        # Case 2: Identical -> Valid
        metrics_equal = TailRiskMetrics(
            var_alpha=var,
            cvar_alpha=var,
            confidence_level=0.99,
            tail_parameters=valid_tail_parameters,
            step_index=1,
        )
        assert metrics_equal.cvar_alpha == metrics_equal.var_alpha

        # Case 3: Marginally smaller but within 1e-10 tolerance -> Valid
        metrics_tol = TailRiskMetrics(
            var_alpha=var,
            cvar_alpha=var - 1e-11,
            confidence_level=0.99,
            tail_parameters=valid_tail_parameters,
            step_index=1,
        )
        assert metrics_tol.cvar_alpha < metrics_tol.var_alpha

        # Case 4: Smaller beyond 1e-10 tolerance -> Raises DegenerateTailRiskException
        with pytest.raises(DegenerateTailRiskException, match="INV-TR-001"):
            TailRiskMetrics(
                var_alpha=var,
                cvar_alpha=var - 2e-10,
                confidence_level=0.99,
                tail_parameters=valid_tail_parameters,
                step_index=1,
            )

        # Case 5: Significantly smaller (e.g. CVaR = 0.03, VaR = 0.05) -> Raises DegenerateTailRiskException
        with pytest.raises(DegenerateTailRiskException, match="INV-TR-001"):
            TailRiskMetrics(
                var_alpha=0.05,
                cvar_alpha=0.03,
                confidence_level=0.99,
                tail_parameters=valid_tail_parameters,
                step_index=1,
            )

    @pytest.mark.parametrize("bad_alpha", [-0.1, 0.0, 0.50, 1.0, 1.05])
    def test_confidence_level_bounds(
        self, valid_tail_parameters: EVTTailParameters, bad_alpha: float
    ) -> None:
        """confidence_level must satisfy 0.50 < alpha < 1.0."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskMetrics(
                var_alpha=0.04,
                cvar_alpha=0.06,
                confidence_level=bad_alpha,
                tail_parameters=valid_tail_parameters,
                step_index=0,
            )

    def test_step_index_bounds(self, valid_tail_parameters: EVTTailParameters) -> None:
        """step_index must be >= 0."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskMetrics(
                var_alpha=0.04,
                cvar_alpha=0.06,
                confidence_level=0.99,
                tail_parameters=valid_tail_parameters,
                step_index=-1,
            )

    def test_tail_parameters_type_enforcement(self) -> None:
        """tail_parameters must be an instance of EVTTailParameters."""
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskMetrics(
                var_alpha=0.04,
                cvar_alpha=0.06,
                confidence_level=0.99,
                tail_parameters=cast(EVTTailParameters, "not_parameters"),
                step_index=0,
            )

    @pytest.mark.parametrize("field_name", ["var_alpha", "cvar_alpha", "confidence_level"])
    @pytest.mark.parametrize("non_finite_val", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_scalars(
        self,
        valid_tail_parameters: EVTTailParameters,
        field_name: str,
        non_finite_val: float,
    ) -> None:
        """Reject NaN and infinite floats with DegenerateTailRiskException."""
        kwargs: dict[str, object] = {
            "var_alpha": 0.045,
            "cvar_alpha": 0.065,
            "confidence_level": 0.99,
            "tail_parameters": valid_tail_parameters,
            "step_index": 0,
        }
        kwargs[field_name] = non_finite_val
        with pytest.raises(DegenerateTailRiskException):
            TailRiskMetrics(**kwargs)

    @pytest.mark.parametrize(
        ("field_name", "bad_val"),
        [
            ("var_alpha", "0.045"),
            ("cvar_alpha", None),
            ("confidence_level", True),
            ("step_index", 1.5),
            ("step_index", False),
        ],
    )
    def test_rejects_invalid_types(
        self,
        valid_tail_parameters: EVTTailParameters,
        field_name: str,
        bad_val: object,
    ) -> None:
        """Reject invalid types with InvalidTailRiskInputException."""
        kwargs: dict[str, object] = {
            "var_alpha": 0.045,
            "cvar_alpha": 0.065,
            "confidence_level": 0.99,
            "tail_parameters": valid_tail_parameters,
            "step_index": 0,
        }
        kwargs[field_name] = bad_val
        with pytest.raises(InvalidTailRiskInputException):
            TailRiskMetrics(**kwargs)


class TestProbabilityWeightedMomentsEstimator:
    """Validate closed-form Probability Weighted Moments (PWM) parameter estimation and bounds."""

    def test_compute_dynamic_threshold_analytical(self) -> None:
        """Verify dynamic threshold calculation against analytical mean + k * std."""
        losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
        mu = float(np.mean(losses))
        sigma = float(np.std(losses))

        # Default k = 1.645
        expected_u = mu + 1.645 * sigma
        u = ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(losses)
        assert abs(u - expected_u) < 1e-12

        # Custom k = 2.0
        expected_u_k2 = mu + 2.0 * sigma
        u_k2 = ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
            losses, threshold_k=2.0
        )
        assert abs(u_k2 - expected_u_k2) < 1e-12

    def test_compute_dynamic_threshold_validation(self) -> None:
        """Verify defensive boundary checks on dynamic threshold computation."""
        # Empty array
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                np.array([], dtype=np.float64)
            )

        # Non-finite values
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                np.array([0.01, float("nan")])
            )
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                np.array([0.01, float("inf")])
            )

        # Non-1D array
        with pytest.raises(InvalidTailRiskInputException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(np.zeros((2, 2)))

        # Non-array input
        with pytest.raises(InvalidTailRiskInputException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                cast(np.ndarray, [0.01, 0.02])
            )

        # Invalid threshold_k
        valid_losses = np.array([0.01, 0.02, 0.03])
        with pytest.raises(InvalidTailRiskInputException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                valid_losses, threshold_k=0.0
            )
        with pytest.raises(InvalidTailRiskInputException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                valid_losses, threshold_k=-1.0
            )
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                valid_losses, threshold_k=float("nan")
            )

    def test_extract_exceedances_positive_and_zero(self) -> None:
        """Verify exceedance extraction with positive exceedances and zero exceedances."""
        losses = np.array([0.01, 0.02, 0.05, 0.08, 0.12], dtype=np.float64)
        threshold_u = 0.04

        # Positive exceedances: 0.05 - 0.04, 0.08 - 0.04, 0.12 - 0.04
        y = ProbabilityWeightedMomentsEstimator.extract_exceedances(losses, threshold_u)
        assert len(y) == 3
        np.testing.assert_allclose(y, [0.01, 0.04, 0.08], atol=1e-12)
        assert np.all(y > 0.0)

        # Zero exceedances (threshold exceeds all losses)
        high_threshold = 0.20
        y_zero = ProbabilityWeightedMomentsEstimator.extract_exceedances(losses, high_threshold)
        assert len(y_zero) == 0
        assert y_zero.dtype == np.float64

    def test_extract_exceedances_validation(self) -> None:
        """Verify defensive boundary checks on exceedance extraction."""
        # Non-1D array
        with pytest.raises(InvalidTailRiskInputException):
            ProbabilityWeightedMomentsEstimator.extract_exceedances(np.zeros((2, 2)), 0.05)

        # Non-array input
        with pytest.raises(InvalidTailRiskInputException):
            ProbabilityWeightedMomentsEstimator.extract_exceedances(
                cast(np.ndarray, [0.01, 0.05]), 0.02
            )

        # Non-finite losses
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.extract_exceedances(
                np.array([0.01, float("nan")]), 0.05
            )

        # Non-finite threshold_u
        valid_losses = np.array([0.01, 0.02, 0.03])
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.extract_exceedances(valid_losses, float("nan"))
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.extract_exceedances(valid_losses, float("inf"))

    def test_pwm_parameter_recovery_synthetic_gpd(self) -> None:
        """Verify PWM parameter estimation accuracy against synthetic samples from known GPD."""
        np.random.seed(42)
        xi_true = 0.20
        beta_true = 0.05
        n_exceedances = 25000
        total_obs = 50000
        threshold_u = 0.02

        # Draw inverse-CDF GPD: y = (beta / xi) * ((1 - U)^(-xi) - 1)
        u_rand = np.random.uniform(0.0, 1.0, n_exceedances)
        exceedances = (beta_true / xi_true) * ((1.0 - u_rand) ** (-xi_true) - 1.0)

        params = ProbabilityWeightedMomentsEstimator.fit(
            exceedances=exceedances,
            total_observations=total_obs,
            threshold_u=threshold_u,
        )

        assert params.method == "EVT_PWM"
        assert params.num_exceedances == n_exceedances
        assert params.total_observations == total_obs
        assert params.threshold_u == threshold_u

        # Theoretical parameter recovery tolerance
        assert abs(params.shape_xi - xi_true) < 0.02
        assert abs(params.scale_beta - beta_true) < 0.005

    def test_pwm_exponential_fallback(self) -> None:
        """Verify exponential fallback when M0 - 2*M1 <= 0 or xi <= 0."""
        cfg = TailRiskConfig()

        # Case 1: Identical exceedances (zero dispersion, xi <= 0)
        exceedances_flat = np.full(50, 0.05, dtype=np.float64)
        params_flat = ProbabilityWeightedMomentsEstimator.fit(
            exceedances=exceedances_flat,
            total_observations=100,
            threshold_u=0.02,
            config=cfg,
        )
        assert params_flat.shape_xi == cfg.tail_index_lower_bound
        assert params_flat.scale_beta >= 1e-8

        # Case 2: Near-zero exceedances (denom <= 1e-12)
        exceedances_tiny = np.full(50, 1e-14, dtype=np.float64)
        params_tiny = ProbabilityWeightedMomentsEstimator.fit(
            exceedances=exceedances_tiny,
            total_observations=100,
            threshold_u=0.02,
            config=cfg,
        )
        assert params_tiny.shape_xi == cfg.tail_index_lower_bound
        assert params_tiny.scale_beta == 1e-8

    def test_pwm_infinite_variance_tripwire(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """INV-TR-002: Raise InfiniteVarianceException if xi >= 1.0."""
        # Monkeypatch dot product so that M1 = 0.0, forcing xi = 2.0 - 1.0 = 1.0 >= 1.0
        exceedances = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
        monkeypatch.setattr(np, "dot", lambda *args, **kwargs: 0.0)

        with pytest.raises(InfiniteVarianceException, match="INV-TR-002"):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=exceedances,
                total_observations=100,
                threshold_u=0.01,
            )

    def test_pwm_zero_iterative_solvers(self) -> None:
        """Rule 4.3: Ensure zero iterative numerical solvers (scipy.optimize strictly banned)."""
        import quant.analytics.tail_risk as tr_mod

        src = inspect.getsource(tr_mod)
        assert "import scipy.optimize" not in src
        assert "from scipy.optimize" not in src
        assert not hasattr(tr_mod, "optimize")
        assert not hasattr(tr_mod, "scipy")

    def test_pwm_execution_latency_sla(self) -> None:
        """INV-TR-006: Execution latency SLA <= 0.02ms for Nu = 500 without profiling overhead."""
        np.random.seed(42)
        exceedances = np.random.exponential(scale=0.02, size=500)
        threshold_u = 0.02

        # Warm up
        for _ in range(30):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=exceedances,
                total_observations=1000,
                threshold_u=threshold_u,
            )

        # Timed benchmark without profiler/coverage tracing overhead (INV-TR-006)
        # Batched runs eliminate timer quantization error on Windows
        batch_size = 20
        num_batches = 10
        old_trace = sys.gettrace()
        try:
            sys.settrace(None)
            latencies: list[float] = []
            for _ in range(num_batches):
                t0 = time.perf_counter()
                for _ in range(batch_size):
                    ProbabilityWeightedMomentsEstimator.fit(
                        exceedances=exceedances,
                        total_observations=1000,
                        threshold_u=threshold_u,
                    )
                latencies.append(((time.perf_counter() - t0) / batch_size) * 1000.0)
        finally:
            sys.settrace(old_trace)

        median_latency = float(np.median(latencies))
        assert median_latency <= 0.02, f"Latency SLA violated: {median_latency:.4f}ms > 0.02ms"

    def test_pwm_fit_validation(self) -> None:
        """Verify input validation on ProbabilityWeightedMomentsEstimator.fit."""
        valid_exceedances = np.array([0.01, 0.02, 0.03], dtype=np.float64)

        # Empty exceedances
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=np.array([], dtype=np.float64),
                total_observations=100,
                threshold_u=0.01,
            )

        # Negative exceedances
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=np.array([0.01, -0.02, 0.03]),
                total_observations=100,
                threshold_u=0.01,
            )

        # Non-finite exceedances
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=np.array([0.01, float("nan")]),
                total_observations=100,
                threshold_u=0.01,
            )

        # Invalid total_observations (< Nu)
        with pytest.raises(InvalidTailRiskInputException):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=valid_exceedances,
                total_observations=2,  # len is 3
                threshold_u=0.01,
            )

        # Non-finite threshold_u
        with pytest.raises(DegenerateTailRiskException):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=valid_exceedances,
                total_observations=10,
                threshold_u=float("nan"),
            )

    def test_slots_and_adaptive_tolerance_improvements(self) -> None:
        """Verify Task 1 minor improvements: slots=True and scale-adaptive tolerance."""
        cfg = TailRiskConfig()
        params = EVTTailParameters(
            threshold_u=0.02,
            shape_xi=0.25,
            scale_beta=0.015,
            num_exceedances=25,
            total_observations=500,
            method="EVT_PWM",
        )
        metrics = TailRiskMetrics(
            var_alpha=100.0,
            cvar_alpha=100.0,
            confidence_level=0.99,
            tail_parameters=params,
            step_index=0,
        )

        # Verify slots=True (has __slots__, no __dict__)
        assert hasattr(cfg, "__slots__")
        assert not hasattr(cfg, "__dict__")
        assert hasattr(params, "__slots__")
        assert not hasattr(params, "__dict__")
        assert hasattr(metrics, "__slots__")
        assert not hasattr(metrics, "__dict__")

        # Verify scale-adaptive tolerance: for var_alpha = 100.0,
        # tolerance is max(1e-10, 1e-9 * 100.0) = 1e-7.
        # cvar_alpha slightly smaller by 5e-8 (within 1e-7 tolerance) -> valid
        metrics_scaled = TailRiskMetrics(
            var_alpha=100.0,
            cvar_alpha=100.0 - 5e-8,
            confidence_level=0.99,
            tail_parameters=params,
            step_index=1,
        )
        assert metrics_scaled.cvar_alpha < metrics_scaled.var_alpha

        # cvar_alpha smaller by 2e-7 (> 1e-7 tolerance) -> raises DegenerateTailRiskException
        with pytest.raises(DegenerateTailRiskException, match="INV-TR-001"):
            TailRiskMetrics(
                var_alpha=100.0,
                cvar_alpha=100.0 - 2e-7,
                confidence_level=0.99,
                tail_parameters=params,
                step_index=1,
            )
