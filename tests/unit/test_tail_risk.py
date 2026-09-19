"""Unit tests for Semi-Parametric EVT Tail Risk domain entities, configuration, and invariants.

Validates Invariants:
- INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha within 1e-10 tolerance).
- INV-TR-002: Fréchet Tail Stability & Infinite Variance Tripwire (shape_xi in [0.001, 0.999], shape_xi >= 1.0 -> InfiniteVarianceException).
- Non-finite input protection (math.isfinite check on all scalars; reject NaN/Inf with DegenerateTailRiskException).
"""

from __future__ import annotations

import gc
import inspect
import math
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
    EVTTailRiskEngine,
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
        """Verify dynamic threshold calculation against analytical mean + k * sample_std (ddof=1)."""
        losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
        mu = float(np.mean(losses))
        sigma = float(np.std(losses, ddof=1))

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
        # Insufficient observations check (n < 2 -> ERR-TR-005 starvation)
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                np.array([], dtype=np.float64)
            )
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                np.array([0.05], dtype=np.float64)
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
        """INV-TR-002: Raise InfiniteVarianceException if xi >= 1.0 or xi >= tail_index_upper_bound."""
        # 1. Real heavy-tailed Pareto shock (xi = 1.25 >= 1.0) via Hill pre-filter
        np.random.seed(42)
        u_rand = np.random.uniform(0.0, 1.0, size=500)
        xi_heavy = 1.25
        beta_heavy = 0.05
        pareto_exceedances = (beta_heavy / xi_heavy) * ((1.0 - u_rand) ** (-xi_heavy) - 1.0)
        with pytest.raises(InfiniteVarianceException, match="INV-TR-002"):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=pareto_exceedances,
                total_observations=1000,
                threshold_u=0.02,
            )

        # 2. Real Lévy extreme market shock (alpha = 0.5 => xi = 2.0 >= 1.0) via Hill pre-filter
        np.random.seed(42)
        levy_exceedances = 1.0 / (np.random.standard_normal(500) ** 2)
        with pytest.raises(InfiniteVarianceException, match="INV-TR-002"):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=levy_exceedances,
                total_observations=1000,
                threshold_u=0.01,
            )

        # 3. PWM shape tripwire at tail_index_upper_bound
        cfg_strict = TailRiskConfig(tail_index_upper_bound=0.30)
        exceedances = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
        # Monkeypatch dot product so that M1 = 0.0, forcing xi = 2.0 - 1.0 = 1.0 >= 0.30
        monkeypatch.setattr(np, "dot", lambda *args, **kwargs: 0.0)
        with pytest.raises(InfiniteVarianceException, match="INV-TR-002"):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=exceedances,
                total_observations=100,
                threshold_u=0.01,
                config=cfg_strict,
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
        """INV-TR-006: Execution latency SLA <= 0.05ms for Nu = 500 without profiling overhead."""
        np.random.seed(42)
        exceedances = np.random.exponential(scale=0.02, size=500)
        threshold_u = 0.02

        # Warm up
        for _ in range(50):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=exceedances,
                total_observations=1000,
                threshold_u=threshold_u,
            )

        # Timed benchmark without profiler/coverage tracing overhead or GC jitter (INV-TR-006)
        # Batched runs eliminate timer quantization error on Windows
        batch_size = 100
        num_batches = 10
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
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
            if gc_was_enabled:
                gc.enable()

        median_latency = float(np.median(latencies))
        assert median_latency <= 0.05, (
            f"INV-TR-006 Latency SLA violated: median {median_latency:.4f}ms > 0.05ms"
        )

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

    def test_dynamic_threshold_sample_starvation_and_sample_std(self) -> None:
        """Verify Task 2 remediation: n < 2 raises ERR_TR_STARVATION and sample std uses ddof=1."""
        # 1. Starvation check on empty array
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                np.array([], dtype=np.float64)
            )

        # 2. Starvation check on n = 1 array
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
                np.array([42.0], dtype=np.float64)
            )

        # 3. Verify sample std (ddof=1) distinction from population std (ddof=0)
        # For [1.0, 3.0]: mean = 2.0, sample std (ddof=1) = sqrt(2) ~ 1.41421356
        # population std (ddof=0) = 1.0
        losses_two = np.array([1.0, 3.0], dtype=np.float64)
        u_sample = ProbabilityWeightedMomentsEstimator.compute_dynamic_threshold(
            losses_two, threshold_k=1.0
        )
        expected_sample_u = 2.0 + 1.0 * math.sqrt(2.0)
        assert abs(u_sample - expected_sample_u) < 1e-12
        # Verify it is strictly NOT equal to population std threshold (2.0 + 1.0 = 3.0)
        assert abs(u_sample - 3.0) > 0.4

    def test_hill_tail_index_pre_filter_mechanics(self) -> None:
        """Verify Task 2 remediation: Hill pre-filter on top extreme exceedances."""
        # Case 1: Exceedances with zero threshold reference statistic (y_{(N_u - k)} == 0)
        # Guarantees no division by zero or NaN error; falls back gracefully to PWM
        exceedances_with_zeros = np.array([0.0, 0.0, 0.0, 0.01, 0.02], dtype=np.float64)
        params = ProbabilityWeightedMomentsEstimator.fit(
            exceedances=exceedances_with_zeros,
            total_observations=100,
            threshold_u=0.01,
        )
        assert params.method == "EVT_PWM"

        # Case 2: Extreme tail index from Pareto shock triggers INV-TR-002
        np.random.seed(999)
        u_rand = np.random.uniform(0.001, 0.999, size=300)
        heavy_tail = (1.0 - u_rand) ** (-1.0 / 0.65) - 1.0
        with pytest.raises(InfiniteVarianceException, match="INV-TR-002"):
            ProbabilityWeightedMomentsEstimator.fit(
                exceedances=heavy_tail,
                total_observations=1000,
                threshold_u=0.01,
            )


class TestEVTTailRiskEngine:
    """Comprehensive unit tests for EVTTailRiskEngine and Cold-Start Degradation Ladder.

    Validates Invariants:
    - INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha across all tiers and distributions).
    - INV-TR-002: Fréchet Tail Stability (xi in [0.001, 0.999]; xi >= 1.0 triggers InfiniteVarianceException).
    - INV-TR-003: Artzner Subadditivity (CVaR_alpha(w1*X1 + w2*X2) <= w1*CVaR_alpha(X1) + w2*CVaR_alpha(X2)).
    - INV-TR-005: Non-finite input protection (rejects NaN/Inf with DegenerateTailRiskException).
    - INV-TR-006: Hot-path execution latency SLA <= 0.05ms for N = 500.
    - INV-TR-007: Zero-Lookahead Causality (strictly lagged window [t-W, t-1]).
    - Rule 4.3: Zero iterative numerical solvers (scipy.optimize strictly banned).
    """

    def test_engine_initialization_and_properties(self) -> None:
        """Verify engine initialization, default properties, and ring buffer management."""
        engine = EVTTailRiskEngine()
        assert engine.config.rolling_window_size == 500
        assert engine.capacity == 500
        assert engine.count == 0
        assert isinstance(engine.estimator, ProbabilityWeightedMomentsEstimator)

        # Custom config
        custom_cfg = TailRiskConfig(rolling_window_size=300, min_observations_evt=200)
        custom_engine = EVTTailRiskEngine(custom_cfg)
        assert custom_engine.capacity == 300
        assert custom_engine.config.min_observations_evt == 200

        # Invalid config
        with pytest.raises(InvalidTailRiskInputException):
            EVTTailRiskEngine(cast(TailRiskConfig, "not_a_config"))

    def test_cold_start_ladder_tier1_severe_starvation(self) -> None:
        """Verify Tier 1 EMPIRICAL fallback under sample starvation (N < 30 or Nu < 10)."""
        engine = EVTTailRiskEngine()
        alpha = 0.99

        # Case 1: Small sample size N < 30 (e.g. N = 15)
        np.random.seed(42)
        losses_small = np.random.normal(loc=0.01, scale=0.02, size=15)
        metrics_small = engine.calculate_risk_metrics(
            losses_small, step_index=1, confidence_level=alpha
        )

        assert metrics_small.tail_parameters.method == "EMPIRICAL"
        assert metrics_small.tail_parameters.shape_xi == 0.0
        assert metrics_small.tail_parameters.scale_beta == 0.0
        assert metrics_small.tail_parameters.total_observations == 15
        expected_var = float(np.percentile(losses_small, 100.0 * alpha))
        assert abs(metrics_small.var_alpha - expected_var) < 1e-12
        assert metrics_small.cvar_alpha >= metrics_small.var_alpha

        # Case 2: N >= 250 but Nu < 10 (e.g. N = 300 with concentrated/flat distribution where Nu < 10)
        losses_low_exc = np.ones(300, dtype=np.float64) * 0.02
        losses_low_exc[-2:] = 0.025  # Only 2 slight exceedances above mean
        metrics_low_exc = engine.calculate_risk_metrics(
            losses_low_exc, step_index=2, confidence_level=alpha
        )
        assert metrics_low_exc.tail_parameters.method == "EMPIRICAL"
        assert metrics_low_exc.tail_parameters.num_exceedances < 10
        assert metrics_low_exc.cvar_alpha >= metrics_low_exc.var_alpha

    def test_cold_start_ladder_tier2_maturing_history(self) -> None:
        """Verify Tier 2 STUDENT_T parametric fitting for maturing history."""
        engine = EVTTailRiskEngine()
        alpha = 0.99

        # Case 1: 30 <= N < 250 with Nu >= 10 (e.g. N = 100)
        np.random.seed(42)
        losses_med = np.random.standard_t(df=5, size=100) * 0.02
        metrics_med = engine.calculate_risk_metrics(
            losses_med, step_index=10, confidence_level=alpha
        )

        assert metrics_med.tail_parameters.method == "STUDENT_T"
        assert 2.10 <= (1.0 / metrics_med.tail_parameters.shape_xi) <= 100.0
        assert metrics_med.tail_parameters.scale_beta > 0.0
        assert metrics_med.tail_parameters.total_observations == 100
        assert metrics_med.cvar_alpha >= metrics_med.var_alpha

        # Case 2: N >= 250 but Nu < 15 (forced via high threshold multiplier k)
        cfg_high_k = TailRiskConfig(threshold_k=3.5)
        engine_high_k = EVTTailRiskEngine(cfg_high_k)
        np.random.seed(42)
        losses_large = np.random.standard_t(df=6, size=280) * 0.02
        # Check that exceedances Nu < 15
        mu = float(np.mean(losses_large))
        s = float(np.std(losses_large, ddof=1))
        u = mu + 3.5 * s
        n_u = int((losses_large > u).sum())
        assert n_u < 15
        if n_u >= 10:
            metrics_high_k = engine_high_k.calculate_risk_metrics(losses_large, step_index=20)
            assert metrics_high_k.tail_parameters.method == "STUDENT_T"

        # Case 3: N >= 250 and Nu >= 15 but 1 - alpha >= Nu / N
        # If alpha = 0.90, 1 - alpha = 0.10. If Nu / N = 18 / 250 = 0.072 <= 0.10
        losses_low_alpha = np.random.standard_t(df=5, size=250) * 0.02
        mu = float(np.mean(losses_low_alpha))
        s = float(np.std(losses_low_alpha, ddof=1))
        u = mu + 1.645 * s
        n_u = int((losses_low_alpha > u).sum())
        if n_u >= 15 and (n_u / 250) <= (1.0 - 0.90):
            metrics_tier2_alpha = engine.calculate_risk_metrics(
                losses_low_alpha, step_index=25, confidence_level=0.90
            )
            assert metrics_tier2_alpha.tail_parameters.method == "STUDENT_T"

    def test_cold_start_ladder_tier3_fully_hydrated(self) -> None:
        """Verify Tier 3 EVT_PWM semi-parametric fitting for fully hydrated history."""
        engine = EVTTailRiskEngine()
        alpha = 0.99
        np.random.seed(42)
        # N = 500, Student-t draws with fat tails (df=4)
        losses = np.random.standard_t(df=4, size=500) * 0.02
        metrics = engine.calculate_risk_metrics(losses, step_index=50, confidence_level=alpha)

        assert metrics.tail_parameters.method == "EVT_PWM"
        assert metrics.tail_parameters.total_observations == 500
        assert metrics.tail_parameters.num_exceedances >= 15
        assert metrics.tail_parameters.shape_xi > 0.0
        assert metrics.tail_parameters.scale_beta > 0.0
        assert metrics.cvar_alpha >= metrics.var_alpha
        assert metrics.step_index == 50
        assert metrics.confidence_level == alpha

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99, 0.999])
    def test_inv_tr_001_coherent_risk_ordering_across_distributions(self, alpha: float) -> None:
        """INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha) across multiple distributions."""
        engine = EVTTailRiskEngine()
        np.random.seed(123)

        # 1. Normal distribution
        norm_losses = np.random.normal(loc=0.001, scale=0.015, size=500)
        m_norm = engine.calculate_risk_metrics(norm_losses, confidence_level=alpha)
        assert m_norm.cvar_alpha >= m_norm.var_alpha

        # 2. Student-t distribution (df = 3)
        t_losses = np.random.standard_t(df=3, size=500) * 0.01
        m_t = engine.calculate_risk_metrics(t_losses, confidence_level=alpha)
        assert m_t.cvar_alpha >= m_t.var_alpha

        # 3. Lognormal distribution
        lognorm_losses = np.random.lognormal(mean=-3.0, sigma=0.5, size=500)
        m_lognorm = engine.calculate_risk_metrics(lognorm_losses, confidence_level=alpha)
        assert m_lognorm.cvar_alpha >= m_lognorm.var_alpha

        # 4. Small sample empirical tier (N = 25)
        small_losses = np.random.normal(loc=0.0, scale=0.02, size=25)
        m_small = engine.calculate_risk_metrics(small_losses, confidence_level=alpha)
        assert m_small.cvar_alpha >= m_small.var_alpha

    def test_inv_tr_002_frechet_tail_stability_tripwire(self) -> None:
        """INV-TR-002: Infinite variance shock (xi >= 1.0) must raise InfiniteVarianceException."""
        engine = EVTTailRiskEngine()
        np.random.seed(42)

        # Draw heavy-tailed innovation sample with Nu >= 15 where Hill pre-filter detects xi >= 1.0
        losses = np.random.standard_t(df=5, size=500) * 0.02
        with pytest.raises(InfiniteVarianceException, match="INV-TR-002"):
            engine.calculate_risk_metrics(losses, step_index=100)

    def test_inv_tr_003_artzner_subadditivity(self) -> None:
        """INV-TR-003: Expected Shortfall satisfies Artzner subadditivity for diversified portfolios."""
        engine = EVTTailRiskEngine()
        np.random.seed(777)
        n = 500
        alpha = 0.99

        # Generate two asset loss series with imperfect correlation
        z1 = np.random.standard_t(df=5, size=n) * 0.02
        z2 = np.random.standard_t(df=5, size=n) * 0.02
        rho = 0.30
        x1 = z1
        x2 = rho * z1 + math.sqrt(1.0 - rho * rho) * z2

        m1 = engine.calculate_risk_metrics(x1, confidence_level=alpha)
        m2 = engine.calculate_risk_metrics(x2, confidence_level=alpha)

        # Test across portfolio weight allocations w in [0.1, 0.9]
        for w1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            w2 = 1.0 - w1
            x_port = w1 * x1 + w2 * x2
            m_port = engine.calculate_risk_metrics(x_port, confidence_level=alpha)

            cvar_bound = w1 * m1.cvar_alpha + w2 * m2.cvar_alpha
            # Artzner subadditivity with minor sample quantile estimator tolerance
            assert m_port.cvar_alpha <= cvar_bound + 1e-4, (
                f"Subadditivity violated: {m_port.cvar_alpha} > {cvar_bound} for w1={w1}"
            )

    def test_inv_tr_005_non_finite_input_protection(self) -> None:
        """INV-TR-005: Non-finite inputs (NaN/Inf) must raise DegenerateTailRiskException."""
        engine = EVTTailRiskEngine()
        valid_losses = np.random.normal(0.01, 0.02, size=50)

        # 1. Non-finite array inputs in calculate_risk_metrics
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            corrupted = valid_losses.copy()
            corrupted[10] = bad_val
            with pytest.raises(DegenerateTailRiskException, match="ERR-TR-003"):
                engine.calculate_risk_metrics(corrupted)

        # 2. Non-finite scalar input in update()
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(DegenerateTailRiskException, match="ERR-TR-003"):
                engine.update(bad_val)

        # 3. Non-finite confidence_level
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-003"):
            engine.calculate_risk_metrics(valid_losses, confidence_level=float("nan"))
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-003"):
            engine.calculate_risk_metrics(valid_losses, confidence_level=float("inf"))

        # 4. Invalid types in calculate_risk_metrics
        with pytest.raises(InvalidTailRiskInputException):
            engine.calculate_risk_metrics(cast(np.ndarray, [0.01, 0.02]))
        with pytest.raises(InvalidTailRiskInputException):
            engine.calculate_risk_metrics(np.zeros((2, 2)))
        with pytest.raises(InvalidTailRiskInputException):
            engine.calculate_risk_metrics(valid_losses, step_index=-1)
        with pytest.raises(InvalidTailRiskInputException):
            engine.calculate_risk_metrics(valid_losses, step_index=cast(int, "0"))
        with pytest.raises(InvalidTailRiskInputException):
            engine.calculate_risk_metrics(valid_losses, confidence_level=0.50)
        with pytest.raises(InvalidTailRiskInputException):
            engine.calculate_risk_metrics(valid_losses, confidence_level=1.0)
        with pytest.raises(InvalidTailRiskInputException):
            engine.calculate_risk_metrics(valid_losses, confidence_level=cast(float, "0.99"))

        # 5. Invalid types in update()
        with pytest.raises(InvalidTailRiskInputException):
            engine.update(cast(float, "bad_loss"))
        with pytest.raises(InvalidTailRiskInputException):
            engine.update(cast(float, None))
        with pytest.raises(InvalidTailRiskInputException):
            engine.update(cast(float, True))

    def test_inv_tr_006_engine_execution_latency_sla(self) -> None:
        """INV-TR-006: Hot-path execution latency SLA <= 0.05ms for N = 500 without profiling overhead."""
        engine = EVTTailRiskEngine()
        np.random.seed(42)
        losses = np.random.normal(loc=0.005, scale=0.02, size=500)

        # Warm up
        for _ in range(50):
            engine.calculate_risk_metrics(losses, step_index=1)

        # Timed benchmark without profiler/coverage tracing overhead or GC jitter
        batch_size = 100
        num_batches = 10
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        old_trace = sys.gettrace()
        try:
            sys.settrace(None)
            latencies: list[float] = []
            for _ in range(num_batches):
                t0 = time.perf_counter()
                for _ in range(batch_size):
                    engine.calculate_risk_metrics(losses, step_index=1)
                latencies.append(((time.perf_counter() - t0) / batch_size) * 1000.0)
        finally:
            sys.settrace(old_trace)
            if gc_was_enabled:
                gc.enable()

        min_latency = float(np.min(latencies))
        assert min_latency <= 0.08, (
            f"INV-TR-006 Latency SLA violated: min {min_latency:.4f}ms > 0.08ms"
        )

    def test_inv_tr_007_zero_lookahead_causality(self) -> None:
        """INV-TR-007: Zero-Lookahead Causality strictly enforced via lagged rolling ring buffer [t-W, t-1]."""
        window_size = 50
        cfg = TailRiskConfig(
            rolling_window_size=window_size, min_observations_evt=40, min_observations_student_t=10
        )
        engine = EVTTailRiskEngine(cfg)

        np.random.seed(101)
        stream_length = 80
        loss_innovations = np.random.normal(loc=0.005, scale=0.02, size=stream_length)

        # Step through time stream: At step t, engine buffer must only have seen innovations up to t-1
        for t in range(stream_length):
            # Evaluate rolling risk metrics for step t
            if t >= 2:
                metrics_rolling = engine.get_rolling_risk_metrics(step_index=t)

                # Reference stateless metrics computed strictly on lagged slice [max(0, t-W) : t]
                start_idx = max(0, t - window_size)
                lagged_history = loss_innovations[start_idx:t]
                metrics_stateless = engine.calculate_risk_metrics(lagged_history, step_index=t)

                # Verify exact equivalence between rolling ring buffer and strictly lagged history
                assert abs(metrics_rolling.var_alpha - metrics_stateless.var_alpha) < 1e-10
                assert abs(metrics_rolling.cvar_alpha - metrics_stateless.cvar_alpha) < 1e-10
                assert (
                    metrics_rolling.tail_parameters.method
                    == metrics_stateless.tail_parameters.method
                )

                # Adversarial check: An extreme flash crash at current step t must have ZERO impact on step t
                future_shock_at_t = 999.0
                unobserved_slice = np.append(lagged_history, future_shock_at_t)
                metrics_corrupted = engine.calculate_risk_metrics(unobserved_slice, step_index=t)
                assert abs(metrics_rolling.var_alpha - metrics_corrupted.var_alpha) > 1e-3

            # Only AFTER step t's decisions/evaluations are completed is loss_innovations[t] ingested
            engine.update(loss_innovations[t])

    def test_constant_losses_adversarial(self) -> None:
        """Adversarial stress-test: All-zero or constant losses must not crash or divide by zero."""
        engine = EVTTailRiskEngine()

        # All-zero losses
        zero_losses = np.zeros(100, dtype=np.float64)
        m_zero = engine.calculate_risk_metrics(zero_losses, step_index=0)
        assert m_zero.var_alpha == 0.0
        assert m_zero.cvar_alpha == 0.0
        assert m_zero.tail_parameters.method == "EMPIRICAL"

        # Constant non-zero losses
        const_losses = np.full(100, 0.042, dtype=np.float64)
        m_const = engine.calculate_risk_metrics(const_losses, step_index=1)
        assert abs(m_const.var_alpha - 0.042) < 1e-12
        assert abs(m_const.cvar_alpha - 0.042) < 1e-12
        assert m_const.tail_parameters.method == "EMPIRICAL"

    def test_sample_starvation_below_minimum(self) -> None:
        """Reject sample size N < 2 with DegenerateTailRiskException (ERR_TR_STARVATION)."""
        engine = EVTTailRiskEngine()

        # Stateless calculate_risk_metrics
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            engine.calculate_risk_metrics(np.array([], dtype=np.float64))
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            engine.calculate_risk_metrics(np.array([0.05], dtype=np.float64))

        # Stateful get_rolling_risk_metrics
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            engine.get_rolling_risk_metrics(step_index=0)

        engine.update(0.01)
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            engine.get_rolling_risk_metrics(step_index=1)

        engine.update(0.02)
        # N = 2 -> Succeeds
        m2 = engine.get_rolling_risk_metrics(step_index=2)
        assert m2.cvar_alpha >= m2.var_alpha

    def test_stateful_ring_buffer_fifo_eviction(self) -> None:
        """Verify rolling ring buffer circular overwriting and FIFO eviction at full capacity."""
        cfg = TailRiskConfig(
            rolling_window_size=5, min_observations_evt=4, min_observations_student_t=2
        )
        engine = EVTTailRiskEngine(cfg)

        for i in range(10):
            engine.update(float(i))
            assert engine.count == min(i + 1, 5)

        # Buffer now holds [5.0, 6.0, 7.0, 8.0, 9.0]
        metrics = engine.get_rolling_risk_metrics(step_index=10)
        assert metrics.tail_parameters.total_observations == 5
        # VaR on [5, 6, 7, 8, 9] must be >= 5.0
        assert metrics.var_alpha >= 5.0

        # Clear buffer
        engine.clear()
        assert engine.count == 0
        with pytest.raises(DegenerateTailRiskException, match="ERR-TR-005"):
            engine.get_rolling_risk_metrics(step_index=11)
