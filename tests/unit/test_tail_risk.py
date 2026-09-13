"""Unit tests for Semi-Parametric EVT Tail Risk domain entities, configuration, and invariants.

Validates Invariants:
- INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha within 1e-10 tolerance).
- INV-TR-002: Fréchet Tail Stability & Infinite Variance Tripwire (shape_xi in [0.001, 0.999], shape_xi >= 1.0 -> InfiniteVarianceException).
- Non-finite input protection (math.isfinite check on all scalars; reject NaN/Inf with DegenerateTailRiskException).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

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
            EVTTailParameters(**kwargs)  # type: ignore[arg-type]

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
            EVTTailParameters(**kwargs)  # type: ignore[arg-type]


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
                tail_parameters="not_parameters",  # type: ignore[arg-type]
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
            TailRiskMetrics(**kwargs)  # type: ignore[arg-type]

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
            TailRiskMetrics(**kwargs)  # type: ignore[arg-type]
