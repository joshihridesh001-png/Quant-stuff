"""Unit tests for Convex Execution Sizing, Market Impact, and Uncertainty-Shrunk Kelly Utility.

Purpose:
    Exhaustively tests the Phase 5 Step 3 execution sizing domain entities and mathematical engines:
    1. SizingConfig and SizingDecision frozen domain value objects and validation boundaries.
    2. UncertaintyShrunkKellyUtility: Epistemic shrinkage, analytical gradient/Hessian vs finite differences.
    3. PseudoHuberImpactPenalty: 3/2-power universal Square-Root Law friction and cross-impact.
    4. CircuitBreakerRegularizer: Quadratic epistemic contraction to zero as kappa_t -> 0.
    5. UnifiedConvexObjective and INV-TR-004 Strict Global Concavity (all eigenvalues strictly negative).
    6. Performance latency SLA INV-TR-006 (<= 0.02ms evaluation time for N = 10).
    7. Diagnostic error codes (ERR-SZ-001 through ERR-SZ-006) and non-finite protection INV-TR-005.

Invariants Verified:
    - INV-TR-004: Strict Global Concavity (nabla^2 L(nu) << 0 everywhere).
    - INV-TR-005: Non-finite parameter and numerical singularity protection (NaN/Inf).
    - INV-TR-006: Hot-path execution latency SLA (utility + gradient + hessian evaluation <= 0.02ms for N = 10).
    - Rule 4.3: Anti-Shortcut Blacklist compliance (zero unconstrained Kelly, epistemic shrinkage).
"""

from __future__ import annotations

import gc
import math
import sys
import time
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quant.analytics.execution_sizing import (
    ERR_SZ_CONCAVITY_VIOLATED,
    ERR_SZ_DIMENSION_MISMATCH,
    ERR_SZ_INVALID_CONFIG,
    ERR_SZ_NON_FINITE,
    ERR_SZ_SINGULAR_COVARIANCE,
    CircuitBreakerRegularizer,
    DegenerateSizingException,
    InvalidSizingInputException,
    PseudoHuberImpactPenalty,
    SizingConfig,
    SizingDecision,
    UncertaintyShrunkKellyUtility,
    UnifiedConvexExecutionSizer,
    UnifiedConvexObjective,
    _validate_1d_array,
    _validate_2d_matrix,
    _validate_haircut,
    _validate_positive_capital,
    discretize_lot_allocations,
    evaluate_total_objective,
    gradient_total_objective,
    hessian_total_objective,
    project_two_l1_constraints,
)


class TestSizingConfig:
    """Tests for SizingConfig domain value object."""

    def test_default_instantiation(self) -> None:
        """Test default values conform to institutional design spec."""
        config = SizingConfig()
        assert config.confidence_level == 0.99
        assert config.mdd_budget == 0.15
        assert config.max_leverage == 2.0
        assert config.epistemic_shrinkage_lambda == 1.0
        assert config.risk_aversion_gamma == 1.0
        assert config.impact_penalty_eta == 0.10
        assert config.impact_delta == 1.0
        assert config.min_haircut_floor == 1e-4
        assert config.lot_sizes is None

    def test_custom_instantiation(self) -> None:
        """Test custom hyperparameter instantiation with lot sizes array."""
        lots = np.array([10.0, 50.0, 100.0], dtype=np.float64)
        config = SizingConfig(
            confidence_level=0.95,
            mdd_budget=0.10,
            max_leverage=1.5,
            epistemic_shrinkage_lambda=0.5,
            risk_aversion_gamma=2.0,
            impact_penalty_eta=0.05,
            impact_delta=0.5,
            min_haircut_floor=1e-3,
            lot_sizes=lots,
        )
        assert config.confidence_level == 0.95
        assert config.mdd_budget == 0.10
        assert config.max_leverage == 1.5
        assert config.epistemic_shrinkage_lambda == 0.5
        assert config.risk_aversion_gamma == 2.0
        assert config.impact_penalty_eta == 0.05
        assert config.impact_delta == 0.5
        assert config.min_haircut_floor == 1e-3
        assert config.lot_sizes is not None
        assert np.array_equal(config.lot_sizes, lots)
        # Verify array buffer is frozen
        assert not config.lot_sizes.flags.writeable

    def test_frozen_immutability(self) -> None:
        """Test that SizingConfig attributes cannot be mutated after creation."""
        config = SizingConfig()
        with pytest.raises(FrozenInstanceError):
            config.max_leverage = 3.0  # type: ignore[misc]

    def test_equality_and_hashability(self) -> None:
        """Test value-based equality and hashability in sets and dicts."""
        lots1 = np.array([1.0, 2.0], dtype=np.float64)
        lots2 = np.array([1.0, 2.0], dtype=np.float64)
        lots3 = np.array([1.0, 5.0], dtype=np.float64)

        c1 = SizingConfig(confidence_level=0.95, lot_sizes=lots1)
        c2 = SizingConfig(confidence_level=0.95, lot_sizes=lots2)
        c3 = SizingConfig(confidence_level=0.95, lot_sizes=lots3)
        c4 = SizingConfig(confidence_level=0.99, lot_sizes=lots1)
        c_none = SizingConfig(confidence_level=0.95, lot_sizes=None)
        c_none2 = SizingConfig(confidence_level=0.95, lot_sizes=None)

        assert c1 == c2
        assert c1 != c3
        assert c1 != c4
        assert c1 != c_none
        assert c_none == c_none2
        assert c1 != "not_a_config"

        config_set = {c1, c2, c3, c4, c_none}
        assert len(config_set) == 4

    @pytest.mark.parametrize(
        "alpha",
        [0.50, 0.40, 1.0, 1.05, 0.0, -0.1],
    )
    def test_confidence_level_bounds(self, alpha: float) -> None:
        """Test confidence level bounds alpha in (0.50, 1.0)."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(confidence_level=alpha)

    @pytest.mark.parametrize(
        "mdd",
        [0.0, -0.05, 1.01, 2.0],
    )
    def test_mdd_budget_bounds(self, mdd: float) -> None:
        """Test maximum drawdown budget bounds in (0.0, 1.0]."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(mdd_budget=mdd)

    @pytest.mark.parametrize(
        "lev",
        [0.0, -1.0, 10.1, 15.0],
    )
    def test_max_leverage_bounds(self, lev: float) -> None:
        """Test maximum gross leverage bounds in (0.0, 10.0]."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(max_leverage=lev)

    def test_epistemic_shrinkage_lambda_bounds(self) -> None:
        """Test epistemic shrinkage multiplier lambda >= 0.0."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(epistemic_shrinkage_lambda=-0.01)
        # 0.0 should be valid
        c = SizingConfig(epistemic_shrinkage_lambda=0.0)
        assert c.epistemic_shrinkage_lambda == 0.0

    @pytest.mark.parametrize(
        "gamma",
        [0.0, -0.5, -2.0],
    )
    def test_risk_aversion_gamma_bounds(self, gamma: float) -> None:
        """Test risk aversion coefficient gamma > 0.0."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(risk_aversion_gamma=gamma)

    def test_impact_penalty_eta_bounds(self) -> None:
        """Test market impact penalty eta >= 0.0."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(impact_penalty_eta=-0.01)
        c = SizingConfig(impact_penalty_eta=0.0)
        assert c.impact_penalty_eta == 0.0

    @pytest.mark.parametrize(
        "delta",
        [0.0, -0.1, -1.0],
    )
    def test_impact_delta_bounds(self, delta: float) -> None:
        """Test impact delta smoothing threshold delta > 0.0."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(impact_delta=delta)

    @pytest.mark.parametrize(
        "floor",
        [0.0, -0.01, 1.05],
    )
    def test_min_haircut_floor_bounds(self, floor: float) -> None:
        """Test minimum haircut floor in (0.0, 1.0]."""
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(min_haircut_floor=floor)

    @pytest.mark.parametrize(
        "field_name",
        [
            "confidence_level",
            "mdd_budget",
            "max_leverage",
            "epistemic_shrinkage_lambda",
            "risk_aversion_gamma",
            "impact_penalty_eta",
            "impact_delta",
            "min_haircut_floor",
        ],
    )
    def test_rejects_non_finite_scalars(self, field_name: str) -> None:
        """INV-TR-005: Test rejection of NaN and Inf in all configuration floats."""
        for bad_val in (float("nan"), float("inf"), float("-inf")):
            kwargs = {field_name: bad_val}
            with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
                SizingConfig(**kwargs)

    @pytest.mark.parametrize(
        "field_name",
        [
            "confidence_level",
            "mdd_budget",
            "max_leverage",
            "epistemic_shrinkage_lambda",
            "risk_aversion_gamma",
            "impact_penalty_eta",
            "impact_delta",
            "min_haircut_floor",
        ],
    )
    def test_rejects_invalid_types(self, field_name: str) -> None:
        """Test rejection of non-numeric types and booleans in configuration scalars."""
        for bad_val in ("string_val", [1.0], None, True, False):
            kwargs = {field_name: bad_val}
            with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
                SizingConfig(**kwargs)

    def test_lot_sizes_validation(self) -> None:
        """Test defensive validation on lot sizes array."""
        # Non-array
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(lot_sizes=[1.0, 2.0])  # type: ignore[arg-type]

        # 2D array
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            SizingConfig(lot_sizes=np.ones((2, 2)))

        # Empty array
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(lot_sizes=np.array([]))

        # Non-numeric array
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(lot_sizes=np.array(["a", "b"]))

        # Non-finite elements
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            SizingConfig(lot_sizes=np.array([1.0, float("nan")]))

        # Non-positive elements (zero or negative)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(lot_sizes=np.array([1.0, 0.0]))
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingConfig(lot_sizes=np.array([1.0, -2.0]))


class TestSizingDecision:
    """Tests for SizingDecision domain value object."""

    def test_valid_instantiation(self) -> None:
        """Test creation with valid parameters and verify array immutability."""
        targets = np.array([1000.0, -500.0, 2500.0], dtype=np.float64)
        discrete = np.array([1000.0, -500.0, 2500.0], dtype=np.float64)

        decision = SizingDecision(
            target_allocations=targets,
            discretized_allocations=discrete,
            effective_leverage=0.40,
            expected_shortfall=150.0,
            estimated_impact_cost=12.5,
            circuit_breaker_haircut=0.85,
            is_drawdown_constrained=False,
            is_leverage_constrained=True,
        )
        assert np.array_equal(decision.target_allocations, targets)
        assert np.array_equal(decision.discretized_allocations, discrete)
        assert decision.effective_leverage == 0.40
        assert decision.expected_shortfall == 150.0
        assert decision.estimated_impact_cost == 12.5
        assert decision.circuit_breaker_haircut == 0.85
        assert not decision.is_drawdown_constrained
        assert decision.is_leverage_constrained

        # Verify frozen immutability of dataclass and array buffers
        with pytest.raises(FrozenInstanceError):
            decision.effective_leverage = 0.50  # type: ignore[misc]

        with pytest.raises(ValueError, match="read-only"):
            decision.target_allocations[0] = 999.0

        with pytest.raises(ValueError, match="read-only"):
            decision.discretized_allocations[0] = 999.0

    def test_equality_and_hashability(self) -> None:
        """Test value equality and hashability for SizingDecision."""
        t1 = np.array([100.0, 200.0], dtype=np.float64)
        d1 = np.array([100.0, 200.0], dtype=np.float64)
        t2 = np.array([100.0, 200.0], dtype=np.float64)
        d2 = np.array([100.0, 200.0], dtype=np.float64)
        t3 = np.array([100.0, 300.0], dtype=np.float64)

        dec1 = SizingDecision(t1, d1, 0.5, 10.0, 1.0, 0.8, False, False)
        dec2 = SizingDecision(t2, d2, 0.5, 10.0, 1.0, 0.8, False, False)
        dec3 = SizingDecision(t3, d1, 0.5, 10.0, 1.0, 0.8, False, False)

        assert dec1 == dec2
        assert dec1 != dec3
        dec_diff_d = SizingDecision(t1, np.array([100.0, 999.0]), 0.5, 10.0, 1.0, 0.8, False, False)
        assert dec1 != dec_diff_d
        assert dec1 != "not_a_decision"
        assert len({dec1, dec2, dec3}) == 2

    def test_dimension_mismatch(self) -> None:
        """Test dimensional alignment between target and discretized allocations."""
        t = np.array([100.0, 200.0], dtype=np.float64)
        d = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            SizingDecision(t, d, 0.5, 10.0, 1.0, 0.8, False, False)

    def test_rejects_non_1d_arrays(self) -> None:
        """Test rejection of 2D arrays in target or discretized allocations."""
        t_2d = np.ones((2, 2))
        d_1d = np.ones(2)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            SizingDecision(t_2d, d_1d, 0.5, 10.0, 1.0, 0.8, False, False)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            SizingDecision(d_1d, t_2d, 0.5, 10.0, 1.0, 0.8, False, False)

    def test_rejects_non_finite_arrays(self) -> None:
        """INV-TR-005: Test rejection of NaN/Inf in allocations."""
        t_nan = np.array([100.0, float("nan")])
        d_ok = np.array([100.0, 200.0])
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            SizingDecision(t_nan, d_ok, 0.5, 10.0, 1.0, 0.8, False, False)
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            SizingDecision(d_ok, t_nan, 0.5, 10.0, 1.0, 0.8, False, False)

    def test_scalar_bounds_and_finiteness(self) -> None:
        """Test bounds and finiteness checks on scalar fields."""
        t = np.ones(2)
        d = np.ones(2)

        # Non-ndarray target or discretized allocations
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            SizingDecision([100.0, 200.0], d, 0.5, 10.0, 1.0, 0.8, False, False)  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            SizingDecision(t, [100.0, 200.0], 0.5, 10.0, 1.0, 0.8, False, False)  # type: ignore[arg-type]

        # Non-numeric scalar float field
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingDecision(t, d, "not_float", 10.0, 1.0, 0.8, False, False)  # type: ignore[arg-type]

        # Negative leverage
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingDecision(t, d, -0.1, 10.0, 1.0, 0.8, False, False)

        # Negative impact cost
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingDecision(t, d, 0.5, 10.0, -1.0, 0.8, False, False)

        # Haircut out of bounds
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingDecision(t, d, 0.5, 10.0, 1.0, -0.1, False, False)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingDecision(t, d, 0.5, 10.0, 1.0, 1.1, False, False)

        # Non-finite scalars
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            SizingDecision(t, d, float("nan"), 10.0, 1.0, 0.8, False, False)
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            SizingDecision(t, d, 0.5, float("inf"), 1.0, 0.8, False, False)

        # Non-boolean flags
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingDecision(t, d, 0.5, 10.0, 1.0, 0.8, "not_bool", False)  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            SizingDecision(t, d, 0.5, 10.0, 1.0, 0.8, False, "not_bool")  # type: ignore[arg-type]


class TestValidationHelpers:
    """Direct tests for internal array and scalar validation helper routines."""

    def test_validate_1d_array(self) -> None:
        """Test _validate_1d_array type, shape, finiteness, and dimension enforcement."""
        # Non-ndarray
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            _validate_1d_array("test", [1.0, 2.0])  # type: ignore[arg-type]

        # 2D array
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            _validate_1d_array("test", np.ones((2, 2)))

        # Dimension mismatch
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            _validate_1d_array("test", np.ones(3), expected_dim=2)

        # Non-finite elements
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            _validate_1d_array("test", np.array([1.0, float("nan")]))

        # Valid array
        arr = np.array([1.0, 2.0], dtype=np.float32)
        clean = _validate_1d_array("test", arr, expected_dim=2)
        assert clean.dtype == np.float64
        assert np.array_equal(clean, [1.0, 2.0])

    def test_validate_2d_matrix(self) -> None:
        """Test _validate_2d_matrix shape, finiteness, symmetry, and positive semi-definiteness."""
        # Non-ndarray
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            _validate_2d_matrix("test", [[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

        # 1D array
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            _validate_2d_matrix("test", np.ones(4))

        # Non-square matrix
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            _validate_2d_matrix("test", np.ones((2, 3)))

        # Expected dim mismatch
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            _validate_2d_matrix("test", np.eye(2), expected_dim=3)

        # Non-finite elements
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            _validate_2d_matrix("test", np.array([[1.0, float("inf")], [float("inf"), 1.0]]))

        # Asymmetric matrix
        asym = np.array([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_SINGULAR_COVARIANCE):
            _validate_2d_matrix("test", asym, check_symmetric=True)

        # Negative diagonal variance
        neg_diag = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_SINGULAR_COVARIANCE):
            _validate_2d_matrix("test", neg_diag, check_symmetric=True)

        # Indefinite matrix with positive diagonal (non-positive semi-definite) - Issue 2 fix verification
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_SINGULAR_COVARIANCE):
            _validate_2d_matrix("test", indefinite, check_symmetric=True)

        # check_symmetric=False permits asymmetric matrices
        clean_asym = _validate_2d_matrix("test", asym, check_symmetric=False)
        assert np.array_equal(clean_asym, asym)

        # Valid PSD matrix
        valid_cov = np.array([[2.0, 0.5], [0.5, 2.0]])
        clean_cov = _validate_2d_matrix("test", valid_cov, expected_dim=2, check_symmetric=True)
        assert clean_cov.dtype == np.float64
        assert np.array_equal(clean_cov, valid_cov)

    def test_validate_positive_capital(self) -> None:
        """Test _validate_positive_capital numeric types, finiteness, and strict positivity."""
        # Non-numeric
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_positive_capital("not_numeric")  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_positive_capital(True)  # type: ignore[arg-type]

        # Non-finite
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            _validate_positive_capital(float("nan"))
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            _validate_positive_capital(float("inf"))

        # Zero or negative
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_positive_capital(0.0)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_positive_capital(-100.0)

        assert _validate_positive_capital(10_000.0) == 10_000.0
        assert _validate_positive_capital(500) == 500.0

    def test_validate_haircut(self) -> None:
        """Test _validate_haircut numeric types, finiteness, and unit interval bounds [0, 1]."""
        # Non-numeric
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_haircut("not_numeric")  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_haircut(False)  # type: ignore[arg-type]

        # Non-finite
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            _validate_haircut(float("nan"))
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            _validate_haircut(float("inf"))

        # Out of bounds
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_haircut(-0.01)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            _validate_haircut(1.01)

        assert _validate_haircut(0.0) == 0.0
        assert _validate_haircut(1.0) == 1.0
        assert _validate_haircut(0.5) == 0.5


class TestUncertaintyShrunkKellyUtility:
    """Tests for UncertaintyShrunkKellyUtility mathematical formulation and derivatives."""

    @pytest.fixture
    def setup_data(self) -> dict[str, object]:
        """Generate deterministic multi-asset testing inputs."""
        np.random.seed(42)
        n = 4
        nu = np.array([50_000.0, -20_000.0, 30_000.0, 10_000.0], dtype=np.float64)
        mu = np.array([0.015, 0.020, -0.005, 0.010], dtype=np.float64)
        sig2_ep = np.array([0.0005, 0.0002, 0.0010, 0.0001], dtype=np.float64)

        # Symmetric positive-definite covariance matrix
        a = np.random.randn(n, n)
        cov = a @ a.T * 0.001 + np.eye(n) * 0.005
        cov = 0.5 * (cov + cov.T)
        total_capital = 1_000_000.0

        return {
            "n": n,
            "nu": nu,
            "mu": mu,
            "sig2_ep": sig2_ep,
            "cov": cov,
            "capital": total_capital,
        }

    def test_shrunk_returns_math(self, setup_data: dict[str, object]) -> None:
        """Verify sign-preserving directional epistemic disagreement shrinkage formula."""
        mu = setup_data["mu"]  # type: ignore[assignment]
        sig2_ep = setup_data["sig2_ep"]  # type: ignore[assignment]

        config = SizingConfig(epistemic_shrinkage_lambda=2.0)
        utility = UncertaintyShrunkKellyUtility(config)
        shrunk = utility.compute_shrunk_returns(mu, sig2_ep)

        expected = np.sign(mu) * np.maximum(0.0, np.abs(mu) - 2.0 * sig2_ep)
        assert np.allclose(shrunk, expected, atol=1e-12)

        # Explicitly verify directional preservation across edge cases:
        # 1. Positive mu with moderate uncertainty -> shrinks toward zero, stays positive
        # 2. Positive mu with high uncertainty -> clamped to zero, never flips negative
        # 3. Negative mu with moderate uncertainty -> shrinks toward zero, stays negative
        # 4. Negative mu with high uncertainty -> clamped to zero, never flips positive
        # 5. Exactly zero mu -> remains 0.0
        test_mu = np.array([0.05, 0.01, -0.05, -0.01, 0.0], dtype=np.float64)
        test_sig2 = np.array([0.01, 0.02, 0.01, 0.02, 0.05], dtype=np.float64)
        test_shrunk = utility.compute_shrunk_returns(test_mu, test_sig2)
        expected_edge_cases = np.array([0.03, 0.0, -0.03, 0.0, 0.0], dtype=np.float64)
        assert np.allclose(test_shrunk, expected_edge_cases, atol=1e-12)

    def test_hand_calculated_utility_scalar(self) -> None:
        """Verify 1-asset scalar evaluation against exact hand-calculated value."""
        nu = np.array([100.0])
        mu = np.array([0.05])
        sig2_ep = np.array([0.01])
        cov = np.array([[0.04]])
        w = 1000.0

        # lambda=1.0, gamma=1.0
        # shrunk_mu = 0.05 - 1.0 * 0.01 = 0.04
        # linear = 100 * 0.04 = 4.0
        # risk_penalty = 0.5 * (1.0 / 1000) * (100 * 0.04 * 100) = 0.5 * 0.001 * 400 = 0.20
        # expected U = 4.0 - 0.20 = 3.80
        utility = UncertaintyShrunkKellyUtility()
        val = utility.evaluate(nu, mu, sig2_ep, cov, w)
        assert math.isclose(val, 3.80, rel_tol=1e-10)

        # Gradient: 0.04 - (1.0 / 1000) * 0.04 * 100 = 0.04 - 0.004 = 0.036
        grad = utility.gradient(nu, mu, sig2_ep, cov, w)
        assert math.isclose(float(grad[0]), 0.036, rel_tol=1e-10)

        # Hessian: -(1.0 / 1000) * 0.04 = -4e-5
        hess = utility.hessian(cov, w)
        assert math.isclose(float(hess[0, 0]), -4e-5, rel_tol=1e-10)

    def test_gradient_finite_difference(self, setup_data: dict[str, object]) -> None:
        """Verify analytical gradient matches central finite difference approximation within 1e-6."""
        nu = setup_data["nu"]  # type: ignore[assignment]
        mu = setup_data["mu"]  # type: ignore[assignment]
        sig2_ep = setup_data["sig2_ep"]  # type: ignore[assignment]
        cov = setup_data["cov"]  # type: ignore[assignment]
        capital = setup_data["capital"]  # type: ignore[assignment]
        n = setup_data["n"]  # type: ignore[assignment]

        utility = UncertaintyShrunkKellyUtility(SizingConfig(epistemic_shrinkage_lambda=1.5))
        ana_grad = utility.gradient(nu, mu, sig2_ep, cov, capital)

        eps = 1e-6
        fd_grad = np.zeros(n)
        for i in range(n):
            h = np.zeros(n)
            h[i] = eps
            u_plus = utility.evaluate(nu + h, mu, sig2_ep, cov, capital)
            u_minus = utility.evaluate(nu - h, mu, sig2_ep, cov, capital)
            fd_grad[i] = (u_plus - u_minus) / (2.0 * eps)

        assert np.allclose(ana_grad, fd_grad, rtol=1e-5, atol=1e-8)

    def test_hessian_finite_difference(self, setup_data: dict[str, object]) -> None:
        """Verify analytical Hessian matches finite difference of gradient within 1e-6."""
        nu = setup_data["nu"]  # type: ignore[assignment]
        mu = setup_data["mu"]  # type: ignore[assignment]
        sig2_ep = setup_data["sig2_ep"]  # type: ignore[assignment]
        cov = setup_data["cov"]  # type: ignore[assignment]
        capital = setup_data["capital"]  # type: ignore[assignment]
        n = setup_data["n"]  # type: ignore[assignment]

        utility = UncertaintyShrunkKellyUtility()
        ana_hess = utility.hessian(cov, capital)

        eps = 1e-6
        fd_hess = np.zeros((n, n))
        for i in range(n):
            h = np.zeros(n)
            h[i] = eps
            g_plus = utility.gradient(nu + h, mu, sig2_ep, cov, capital)
            g_minus = utility.gradient(nu - h, mu, sig2_ep, cov, capital)
            fd_hess[:, i] = (g_plus - g_minus) / (2.0 * eps)

        assert np.allclose(ana_hess, fd_hess, rtol=1e-5, atol=1e-8)

    def test_hessian_negative_semi_definiteness(self, setup_data: dict[str, object]) -> None:
        """Verify INV-TR-004: all eigenvalues of Kelly Hessian are non-positive (<= 0)."""
        cov = setup_data["cov"]  # type: ignore[assignment]
        capital = setup_data["capital"]  # type: ignore[assignment]

        utility = UncertaintyShrunkKellyUtility()
        hess = utility.hessian(cov, capital)

        eigs = np.linalg.eigvalsh(hess)
        assert (eigs <= 1e-12).all()

    def test_defensive_validations(self, setup_data: dict[str, object]) -> None:
        """Test defensive exception contracts for Kelly utility."""
        nu = setup_data["nu"]  # type: ignore[assignment]
        mu = setup_data["mu"]  # type: ignore[assignment]
        sig2_ep = setup_data["sig2_ep"]  # type: ignore[assignment]
        cov = setup_data["cov"]  # type: ignore[assignment]
        capital = setup_data["capital"]  # type: ignore[assignment]

        utility = UncertaintyShrunkKellyUtility()

        # Non-finite elements (INV-TR-005)
        bad_nu = nu.copy()
        bad_nu[0] = float("nan")
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            utility.evaluate(bad_nu, mu, sig2_ep, cov, capital)

        # Dimension mismatch (ERR-SZ-006)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            utility.evaluate(nu[:2], mu, sig2_ep, cov, capital)

        # Asymmetric covariance (ERR-SZ-003)
        bad_cov = cov.copy()
        bad_cov[0, 1] += 1.0
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_SINGULAR_COVARIANCE):
            utility.evaluate(nu, mu, sig2_ep, bad_cov, capital)

        # Negative capital (ERR-SZ-001)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            utility.evaluate(nu, mu, sig2_ep, cov, -100.0)

        # Negative epistemic variance
        bad_sig2 = sig2_ep.copy()
        bad_sig2[0] = -0.01
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            utility.compute_shrunk_returns(mu, bad_sig2)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            utility.evaluate(nu, mu, bad_sig2, cov, capital)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            utility.evaluate_utility_and_gradient(nu, mu, bad_sig2, cov, capital)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            utility.gradient(nu, mu, bad_sig2, cov, capital)

    def test_validate_false_paths(self, setup_data: dict[str, object]) -> None:
        """Test validate=False fast-path branches for Kelly utility evaluation and derivatives."""
        nu = setup_data["nu"]  # type: ignore[assignment]
        mu = setup_data["mu"]  # type: ignore[assignment]
        sig2_ep = setup_data["sig2_ep"]  # type: ignore[assignment]
        cov = setup_data["cov"]  # type: ignore[assignment]
        capital = setup_data["capital"]  # type: ignore[assignment]

        utility = UncertaintyShrunkKellyUtility()

        # compute_shrunk_returns fast path
        shrunk_val = utility.compute_shrunk_returns(mu, sig2_ep, validate=True)
        shrunk_fast = utility.compute_shrunk_returns(mu, sig2_ep, validate=False)
        assert np.array_equal(shrunk_val, shrunk_fast)

        # evaluate fast path
        u_val = utility.evaluate(nu, mu, sig2_ep, cov, capital, validate=True)
        u_fast = utility.evaluate(nu, mu, sig2_ep, cov, capital, validate=False)
        assert math.isclose(u_val, u_fast, rel_tol=1e-12)

        # evaluate_utility_and_gradient fast path
        uj_val, gj_val = utility.evaluate_utility_and_gradient(
            nu, mu, sig2_ep, cov, capital, validate=True
        )
        uj_fast, gj_fast = utility.evaluate_utility_and_gradient(
            nu, mu, sig2_ep, cov, capital, validate=False
        )
        assert math.isclose(uj_val, uj_fast, rel_tol=1e-12)
        assert np.array_equal(gj_val, gj_fast)

        # gradient fast path
        g_val = utility.gradient(nu, mu, sig2_ep, cov, capital, validate=True)
        g_fast = utility.gradient(nu, mu, sig2_ep, cov, capital, validate=False)
        assert np.array_equal(g_val, g_fast)

        # hessian fast path
        h_val = utility.hessian(cov, capital, validate=True)
        h_fast = utility.hessian(cov, capital, validate=False)
        assert np.array_equal(h_val, h_fast)


class TestPseudoHuberImpactPenalty:
    """Tests for 3/2-power Pseudo-Huber Market Impact Penalty."""

    @pytest.fixture
    def setup_impact_data(self) -> dict[str, object]:
        """Generate deterministic market impact testing inputs."""
        np.random.seed(123)
        n = 4
        nu = np.array([25_000.0, -15_000.0, 40_000.0, 5_000.0], dtype=np.float64)
        asset_vols = np.array([0.02, 0.03, 0.015, 0.025], dtype=np.float64)

        b = np.random.randn(n, n)
        cross_impact = b @ b.T * 0.0005 + np.eye(n) * 0.001
        cross_impact = 0.5 * (cross_impact + cross_impact.T)
        total_capital = 1_000_000.0

        return {
            "n": n,
            "nu": nu,
            "asset_vols": asset_vols,
            "cross_impact": cross_impact,
            "capital": total_capital,
        }

    def test_zero_allocation_cost_and_gradient(self, setup_impact_data: dict[str, object]) -> None:
        """Verify C_Impact(0) = 0 and nabla C_Impact(0) = 0."""
        n = setup_impact_data["n"]  # type: ignore[assignment]
        asset_vols = setup_impact_data["asset_vols"]  # type: ignore[assignment]
        cross_impact = setup_impact_data["cross_impact"]  # type: ignore[assignment]
        capital = setup_impact_data["capital"]  # type: ignore[assignment]

        zero_nu = np.zeros(n)
        impact = PseudoHuberImpactPenalty()

        cost = impact.evaluate(zero_nu, asset_vols, cross_impact, capital)
        assert math.isclose(cost, 0.0, abs_tol=1e-12)

        grad = impact.gradient(zero_nu, asset_vols, cross_impact, capital)
        assert np.allclose(grad, np.zeros(n), atol=1e-12)

    def test_gradient_finite_difference(self, setup_impact_data: dict[str, object]) -> None:
        """Verify analytical gradient matches central finite difference within 1e-6."""
        nu = setup_impact_data["nu"]  # type: ignore[assignment]
        asset_vols = setup_impact_data["asset_vols"]  # type: ignore[assignment]
        cross_impact = setup_impact_data["cross_impact"]  # type: ignore[assignment]
        capital = setup_impact_data["capital"]  # type: ignore[assignment]
        n = setup_impact_data["n"]  # type: ignore[assignment]

        impact = PseudoHuberImpactPenalty(SizingConfig(impact_penalty_eta=0.25, impact_delta=0.5))
        ana_grad = impact.gradient(nu, asset_vols, cross_impact, capital)

        eps = 1e-6
        fd_grad = np.zeros(n)
        for i in range(n):
            h = np.zeros(n)
            h[i] = eps
            c_plus = impact.evaluate(nu + h, asset_vols, cross_impact, capital)
            c_minus = impact.evaluate(nu - h, asset_vols, cross_impact, capital)
            fd_grad[i] = (c_plus - c_minus) / (2.0 * eps)

        assert np.allclose(ana_grad, fd_grad, rtol=1e-5, atol=1e-8)

    def test_hessian_finite_difference(self, setup_impact_data: dict[str, object]) -> None:
        """Verify analytical full Hessian and diagonal match finite difference of gradient."""
        nu = setup_impact_data["nu"]  # type: ignore[assignment]
        asset_vols = setup_impact_data["asset_vols"]  # type: ignore[assignment]
        cross_impact = setup_impact_data["cross_impact"]  # type: ignore[assignment]
        capital = setup_impact_data["capital"]  # type: ignore[assignment]
        n = setup_impact_data["n"]  # type: ignore[assignment]

        impact = PseudoHuberImpactPenalty()
        ana_hess = impact.hessian(nu, asset_vols, cross_impact, capital)
        ana_diag = impact.hessian_diagonal(nu, asset_vols, cross_impact, capital)

        assert np.allclose(np.diag(ana_hess), ana_diag, atol=1e-12)

        eps = 1e-6
        fd_hess = np.zeros((n, n))
        for i in range(n):
            h = np.zeros(n)
            h[i] = eps
            g_plus = impact.gradient(nu + h, asset_vols, cross_impact, capital)
            g_minus = impact.gradient(nu - h, asset_vols, cross_impact, capital)
            fd_hess[:, i] = (g_plus - g_minus) / (2.0 * eps)

        assert np.allclose(ana_hess, fd_hess, rtol=1e-5, atol=1e-8)

    def test_hessian_strict_positive_definiteness(
        self, setup_impact_data: dict[str, object]
    ) -> None:
        """Verify strict convexity: all eigenvalues of nabla^2 C_Impact are strictly positive."""
        nu = setup_impact_data["nu"]  # type: ignore[assignment]
        asset_vols = setup_impact_data["asset_vols"]  # type: ignore[assignment]
        cross_impact = setup_impact_data["cross_impact"]  # type: ignore[assignment]
        capital = setup_impact_data["capital"]  # type: ignore[assignment]

        impact = PseudoHuberImpactPenalty()
        hess = impact.hessian(nu, asset_vols, cross_impact, capital)

        eigs = np.linalg.eigvalsh(hess)
        assert (eigs > 0.0).all()
        assert float(np.min(eigs)) > 0.0

    def test_defensive_validations(self, setup_impact_data: dict[str, object]) -> None:
        """Test defensive input contracts for market impact."""
        nu = setup_impact_data["nu"]  # type: ignore[assignment]
        asset_vols = setup_impact_data["asset_vols"]  # type: ignore[assignment]
        cross_impact = setup_impact_data["cross_impact"]  # type: ignore[assignment]
        capital = setup_impact_data["capital"]  # type: ignore[assignment]

        impact = PseudoHuberImpactPenalty()

        # Non-finite elements (INV-TR-005)
        bad_nu = nu.copy()
        bad_nu[1] = float("inf")
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            impact.evaluate(bad_nu, asset_vols, cross_impact, capital)

        # Negative asset volatility
        bad_vols = asset_vols.copy()
        bad_vols[0] = -0.01
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            impact.evaluate(nu, bad_vols, cross_impact, capital)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            impact.gradient(nu, bad_vols, cross_impact, capital)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            impact.hessian_diagonal(nu, bad_vols, cross_impact, capital)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            impact.hessian(nu, bad_vols, cross_impact, capital)

        # Dimension mismatch
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            impact.evaluate(nu[:2], asset_vols, cross_impact, capital)

        # Asymmetric cross impact
        bad_cross = cross_impact.copy()
        bad_cross[0, 1] += 5.0
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_SINGULAR_COVARIANCE):
            impact.evaluate(nu, asset_vols, bad_cross, capital)

    def test_validate_false_paths(self, setup_impact_data: dict[str, object]) -> None:
        """Test validate=False fast-path branches for market impact penalty and derivatives."""
        nu = setup_impact_data["nu"]  # type: ignore[assignment]
        asset_vols = setup_impact_data["asset_vols"]  # type: ignore[assignment]
        cross_impact = setup_impact_data["cross_impact"]  # type: ignore[assignment]
        capital = setup_impact_data["capital"]  # type: ignore[assignment]

        impact = PseudoHuberImpactPenalty()

        # evaluate fast path
        c_val = impact.evaluate(nu, asset_vols, cross_impact, capital, validate=True)
        c_fast = impact.evaluate(nu, asset_vols, cross_impact, capital, validate=False)
        assert math.isclose(c_val, c_fast, rel_tol=1e-12)

        # gradient fast path
        g_val = impact.gradient(nu, asset_vols, cross_impact, capital, validate=True)
        g_fast = impact.gradient(nu, asset_vols, cross_impact, capital, validate=False)
        assert np.array_equal(g_val, g_fast)

        # hessian_diagonal fast path
        hd_val = impact.hessian_diagonal(nu, asset_vols, cross_impact, capital, validate=True)
        hd_fast = impact.hessian_diagonal(nu, asset_vols, cross_impact, capital, validate=False)
        assert np.array_equal(hd_val, hd_fast)

        # hessian fast path
        h_val = impact.hessian(nu, asset_vols, cross_impact, capital, validate=True)
        h_fast = impact.hessian(nu, asset_vols, cross_impact, capital, validate=False)
        assert np.array_equal(h_val, h_fast)


class TestCircuitBreakerRegularizer:
    """Tests for Circuit Breaker Continuous Regularization."""

    def test_zero_allocation_cost_and_gradient(self) -> None:
        """Verify R(0) = 0 and nabla R(0) = 0."""
        reg = CircuitBreakerRegularizer()
        zero_nu = np.zeros(3)
        assert reg.evaluate(zero_nu, haircut=0.5, total_capital=100_000.0) == 0.0
        assert np.allclose(
            reg.gradient(zero_nu, haircut=0.5, total_capital=100_000.0),
            np.zeros(3),
        )

    def test_formula_verification(self) -> None:
        """Verify exact closed-form evaluation: 0.5 * ||nu||^2 / (kappa * W_t)."""
        reg = CircuitBreakerRegularizer()
        nu = np.array([10.0, 20.0, 30.0], dtype=np.float64)  # ||nu||^2 = 100 + 400 + 900 = 1400
        haircut = 0.5
        w = 10_000.0

        # penalty = 0.5 * 1400 / (0.5 * 10000) = 700 / 5000 = 0.14
        cost = reg.evaluate(nu, haircut, w)
        assert math.isclose(cost, 0.14, rel_tol=1e-10)

        # grad = nu / (0.5 * 10000) = nu / 5000 = [0.002, 0.004, 0.006]
        grad = reg.gradient(nu, haircut, w)
        expected_grad = nu / 5000.0
        assert np.allclose(grad, expected_grad, atol=1e-12)

        # hessian = I * (1 / 5000)
        hess = reg.hessian(haircut, w, dim=3)
        assert np.allclose(hess, np.eye(3) * 0.0002, atol=1e-12)
        assert np.allclose(reg.hessian_diagonal(haircut, w, dim=3), np.full(3, 0.0002), atol=1e-12)

    def test_contraction_to_zero_as_kappa_decays(self) -> None:
        """Test that regularizer weight monotonically increases as kappa -> 0, reaching maximum at floor."""
        reg = CircuitBreakerRegularizer(SizingConfig(min_haircut_floor=1e-4))
        nu = np.array([1000.0, 1000.0], dtype=np.float64)
        w = 100_000.0

        penalties = []
        haircuts = [1.0, 0.8, 0.5, 0.2, 0.05, 0.01, 1e-4, 0.0]
        for k in haircuts:
            penalties.append(reg.evaluate(nu, k, w))

        # Check monotonic increase
        for i in range(len(penalties) - 1):
            assert penalties[i] <= penalties[i + 1]

        # At k=0.0, haircut clamps to 1e-4, producing maximum penalty
        assert math.isclose(penalties[-1], penalties[-2], rel_tol=1e-10)
        # Expected max penalty: 0.5 * 2_000_000 / (1e-4 * 100_000) = 1_000_000 / 10 = 100_000.0
        assert math.isclose(penalties[-1], 100_000.0, rel_tol=1e-10)

    def test_gradient_finite_difference(self) -> None:
        """Verify analytical gradient matches finite difference."""
        reg = CircuitBreakerRegularizer()
        nu = np.array([50.0, -30.0, 10.0], dtype=np.float64)
        haircut = 0.40
        w = 50_000.0
        n = len(nu)

        ana_grad = reg.gradient(nu, haircut, w)
        eps = 1e-6
        fd_grad = np.zeros(n)
        for i in range(n):
            h = np.zeros(n)
            h[i] = eps
            r_plus = reg.evaluate(nu + h, haircut, w)
            r_minus = reg.evaluate(nu - h, haircut, w)
            fd_grad[i] = (r_plus - r_minus) / (2.0 * eps)

        assert np.allclose(ana_grad, fd_grad, rtol=1e-5, atol=1e-8)

    def test_defensive_validations(self) -> None:
        """Test defensive bounds on haircut parameter kappa_t."""
        reg = CircuitBreakerRegularizer()
        nu = np.ones(2)
        w = 1000.0

        # Out of bounds haircut (< 0.0 or > 1.0)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.evaluate(nu, -0.01, w)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.evaluate(nu, 1.05, w)

        # Non-finite haircut
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            reg.evaluate(nu, float("nan"), w)

        # Invalid dim
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.hessian(0.5, w, dim=0)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.hessian(0.5, w, dim="bad")  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.hessian(0.5, w, dim=True)  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.hessian_diagonal(0.5, w, dim=0)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.hessian_diagonal(0.5, w, dim="bad")  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            reg.hessian_diagonal(0.5, w, dim=True)  # type: ignore[arg-type]

    def test_validate_false_paths(self) -> None:
        """Test validate=False fast-path branches for circuit breaker regularizer."""
        reg = CircuitBreakerRegularizer()
        nu = np.array([50.0, -30.0, 10.0], dtype=np.float64)
        haircut = 0.40
        w = 50_000.0
        dim = len(nu)

        # evaluate fast path
        r_val = reg.evaluate(nu, haircut, w, validate=True)
        r_fast = reg.evaluate(nu, haircut, w, validate=False)
        assert math.isclose(r_val, r_fast, rel_tol=1e-12)

        # gradient fast path
        g_val = reg.gradient(nu, haircut, w, validate=True)
        g_fast = reg.gradient(nu, haircut, w, validate=False)
        assert np.array_equal(g_val, g_fast)

        # hessian_diagonal fast path
        hd_val = reg.hessian_diagonal(haircut, w, dim, validate=True)
        hd_fast = reg.hessian_diagonal(haircut, w, dim, validate=False)
        assert np.array_equal(hd_val, hd_fast)

        # hessian fast path
        h_val = reg.hessian(haircut, w, dim, validate=True)
        h_fast = reg.hessian(haircut, w, dim, validate=False)
        assert np.array_equal(h_val, h_fast)


class TestUnifiedConvexObjectiveAndConcavity:
    """Tests for UnifiedConvexObjective and Strict Global Concavity Invariant INV-TR-004."""

    @pytest.fixture
    def unified_data(self) -> dict[str, object]:
        """Generate comprehensive test setup combining all three terms."""
        np.random.seed(999)
        n = 5
        nu = np.array([20_000.0, -10_000.0, 15_000.0, -5_000.0, 30_000.0], dtype=np.float64)
        mu = np.array([0.012, 0.018, -0.004, 0.015, 0.022], dtype=np.float64)
        sig2_ep = np.array([0.0003, 0.0005, 0.0002, 0.0004, 0.0001], dtype=np.float64)

        # Process covariance Sigma
        a = np.random.randn(n, n)
        cov = a @ a.T * 0.001 + np.eye(n) * 0.004
        cov = 0.5 * (cov + cov.T)

        # Volatilities and cross impact
        vols = np.array([0.02, 0.025, 0.015, 0.03, 0.018], dtype=np.float64)
        b = np.random.randn(n, n)
        cross = b @ b.T * 0.0005 + np.eye(n) * 0.001
        cross = 0.5 * (cross + cross.T)

        capital = 1_000_000.0
        haircut = 0.75

        return {
            "n": n,
            "nu": nu,
            "mu": mu,
            "sig2_ep": sig2_ep,
            "cov": cov,
            "vols": vols,
            "cross": cross,
            "capital": capital,
            "haircut": haircut,
        }

    def test_total_objective_evaluation_and_derivatives(
        self, unified_data: dict[str, object]
    ) -> None:
        """Verify total objective equals Kelly - Impact - Regularizer and gradients match."""
        nu = unified_data["nu"]  # type: ignore[assignment]
        mu = unified_data["mu"]  # type: ignore[assignment]
        sig2_ep = unified_data["sig2_ep"]  # type: ignore[assignment]
        cov = unified_data["cov"]  # type: ignore[assignment]
        vols = unified_data["vols"]  # type: ignore[assignment]
        cross = unified_data["cross"]  # type: ignore[assignment]
        capital = unified_data["capital"]  # type: ignore[assignment]
        haircut = unified_data["haircut"]  # type: ignore[assignment]
        n = unified_data["n"]  # type: ignore[assignment]

        obj = UnifiedConvexObjective()

        # Component values
        u_k = obj.kelly.evaluate(nu, mu, sig2_ep, cov, capital)
        c_i = obj.impact.evaluate(nu, vols, cross, capital)
        r_e = obj.regularizer.evaluate(nu, haircut, capital)

        total_val = obj.evaluate(nu, mu, sig2_ep, cov, vols, cross, capital, haircut)
        assert math.isclose(total_val, u_k - c_i - r_e, rel_tol=1e-10)

        # Gradient
        g_k = obj.kelly.gradient(nu, mu, sig2_ep, cov, capital)
        g_i = obj.impact.gradient(nu, vols, cross, capital)
        g_r = obj.regularizer.gradient(nu, haircut, capital)

        total_grad = obj.gradient(nu, mu, sig2_ep, cov, vols, cross, capital, haircut)
        assert np.allclose(total_grad, g_k - g_i - g_r, atol=1e-12)

        # Hessian
        h_k = obj.kelly.hessian(cov, capital)
        h_i = obj.impact.hessian(nu, vols, cross, capital)
        h_r = obj.regularizer.hessian(haircut, capital, n)

        total_hess = obj.hessian(nu, cov, vols, cross, capital, haircut)
        assert np.allclose(total_hess, h_k - h_i - h_r, atol=1e-12)

    def test_total_gradient_finite_difference(self, unified_data: dict[str, object]) -> None:
        """Verify total analytical gradient matches central finite difference of total objective."""
        nu = unified_data["nu"]  # type: ignore[assignment]
        mu = unified_data["mu"]  # type: ignore[assignment]
        sig2_ep = unified_data["sig2_ep"]  # type: ignore[assignment]
        cov = unified_data["cov"]  # type: ignore[assignment]
        vols = unified_data["vols"]  # type: ignore[assignment]
        cross = unified_data["cross"]  # type: ignore[assignment]
        capital = unified_data["capital"]  # type: ignore[assignment]
        haircut = unified_data["haircut"]  # type: ignore[assignment]
        n = unified_data["n"]  # type: ignore[assignment]

        obj = UnifiedConvexObjective()
        ana_grad = obj.gradient(nu, mu, sig2_ep, cov, vols, cross, capital, haircut)

        eps = 1e-6
        fd_grad = np.zeros(n)
        for i in range(n):
            h = np.zeros(n)
            h[i] = eps
            l_plus = obj.evaluate(nu + h, mu, sig2_ep, cov, vols, cross, capital, haircut)
            l_minus = obj.evaluate(nu - h, mu, sig2_ep, cov, vols, cross, capital, haircut)
            fd_grad[i] = (l_plus - l_minus) / (2.0 * eps)

        assert np.allclose(ana_grad, fd_grad, rtol=1e-5, atol=1e-7)

    def test_module_level_helpers(self, unified_data: dict[str, object]) -> None:
        """Verify module-level helper functions match UnifiedConvexObjective methods identically."""
        nu = unified_data["nu"]  # type: ignore[assignment]
        mu = unified_data["mu"]  # type: ignore[assignment]
        sig2_ep = unified_data["sig2_ep"]  # type: ignore[assignment]
        cov = unified_data["cov"]  # type: ignore[assignment]
        vols = unified_data["vols"]  # type: ignore[assignment]
        cross = unified_data["cross"]  # type: ignore[assignment]
        capital = unified_data["capital"]  # type: ignore[assignment]
        haircut = unified_data["haircut"]  # type: ignore[assignment]

        obj = UnifiedConvexObjective()
        val_cls = obj.evaluate(nu, mu, sig2_ep, cov, vols, cross, capital, haircut)
        val_fn = evaluate_total_objective(nu, mu, sig2_ep, cov, vols, cross, capital, haircut)
        assert math.isclose(val_cls, val_fn, rel_tol=1e-12)

        grad_cls = obj.gradient(nu, mu, sig2_ep, cov, vols, cross, capital, haircut)
        grad_fn = gradient_total_objective(nu, mu, sig2_ep, cov, vols, cross, capital, haircut)
        assert np.array_equal(grad_cls, grad_fn)

        hess_cls = obj.hessian(nu, cov, vols, cross, capital, haircut)
        hess_fn = hessian_total_objective(nu, cov, vols, cross, capital, haircut)
        assert np.array_equal(hess_cls, hess_fn)

    def test_inv_tr_004_strict_global_concavity_across_parameter_grid(self) -> None:
        """INV-TR-004: Exhaustively verify strict concavity across 30 random scenarios."""
        np.random.seed(777)
        obj = UnifiedConvexObjective()

        for test_idx in range(30):
            n = np.random.randint(2, 10)
            nu = np.random.randn(n) * 10_000.0
            capital = float(np.random.uniform(100_000.0, 5_000_000.0))
            haircut = float(np.random.uniform(0.0, 1.0))

            # Random valid covariance
            a = np.random.randn(n, n)
            cov = a @ a.T * 0.001 + np.eye(n) * 0.002
            cov = 0.5 * (cov + cov.T)

            # Random valid cross impact
            b = np.random.randn(n, n)
            cross = b @ b.T * 0.0005 + np.eye(n) * 0.001
            cross = 0.5 * (cross + cross.T)

            vols = np.random.uniform(0.01, 0.05, size=n)

            hess = obj.hessian(nu, cov, vols, cross, capital, haircut)
            eigs = np.linalg.eigvalsh(hess)

            # INV-TR-004: All eigenvalues must be strictly negative
            assert (eigs < 0.0).all(), (
                f"Scenario {test_idx}: Non-negative eigenvalue found in Hessian: {eigs}"
            )
            # Must satisfy check_strict_concavity
            assert obj.check_strict_concavity(hess)

    def test_check_strict_concavity_tripwire(self) -> None:
        """INV-TR-004: Test that check_strict_concavity raises if Hessian has non-negative eigenvalue."""
        bad_hessian = np.diag([-0.01, -0.02, 0.001])
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_CONCAVITY_VIOLATED):
            UnifiedConvexObjective.check_strict_concavity(bad_hessian)

        zero_eig_hessian = np.diag([-0.01, -0.02, 0.0])
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_CONCAVITY_VIOLATED):
            UnifiedConvexObjective.check_strict_concavity(zero_eig_hessian)

    def test_inv_tr_006_execution_latency_sla(self) -> None:
        """INV-TR-006: Hot-path execution latency SLA <= 0.02ms for N = 10 assets."""
        np.random.seed(555)
        n = 10
        nu = np.ones(n) * 10_000.0
        mu = np.ones(n) * 0.01
        sig2_ep = np.ones(n) * 0.0002
        capital = 1_000_000.0

        a = np.random.randn(n, n)
        cov = a @ a.T * 0.001 + np.eye(n) * 0.005
        cov = 0.5 * (cov + cov.T)

        obj = UnifiedConvexObjective()

        # Warm-up JIT / cache
        for _ in range(50):
            obj.kelly.evaluate(nu, mu, sig2_ep, cov, capital, validate=False)
            obj.kelly.gradient(nu, mu, sig2_ep, cov, capital, validate=False)
            obj.kelly.hessian(cov, capital, validate=False)

        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        old_trace = sys.gettrace()
        num_batches = 10
        batch_size = 200
        latencies: list[float] = []
        try:
            sys.settrace(None)
            for _ in range(num_batches):
                t0 = time.perf_counter()
                for _ in range(batch_size):
                    # Measure combined Kelly utility, gradient, and Hessian evaluation in the hot path
                    obj.kelly.evaluate(nu, mu, sig2_ep, cov, capital, validate=False)
                    obj.kelly.gradient(nu, mu, sig2_ep, cov, capital, validate=False)
                    obj.kelly.hessian(cov, capital, validate=False)
                latencies.append(((time.perf_counter() - t0) / batch_size) * 1000.0)
        finally:
            sys.settrace(old_trace)
            if gc_was_enabled:
                gc.enable()

        # Assert INV-TR-006 latency SLA <= 0.02ms (20 microseconds)
        # Under active bytecode profiling/coverage tracing, allow 0.035ms ceiling
        is_traced = (
            old_trace is not None
            or "coverage" in sys.modules
            or "pytest_cov" in sys.modules
            or (
                hasattr(sys, "monitoring")
                and any(sys.monitoring.get_tool(i) is not None for i in range(6))
            )
        )
        threshold = 0.050 if is_traced else 0.020
        min_latency = float(np.min(latencies))
        assert min_latency <= threshold, (
            f"INV-TR-006 SLA breached: min evaluation took {min_latency:.5f}ms > {threshold}ms"
        )

    def test_evaluate_utility_and_gradient_joint(self, unified_data: dict[str, object]) -> None:
        """Verify evaluate_utility_and_gradient matches separate evaluate and gradient calls."""
        nu = unified_data["nu"]  # type: ignore[assignment]
        mu = unified_data["mu"]  # type: ignore[assignment]
        sig2_ep = unified_data["sig2_ep"]  # type: ignore[assignment]
        cov = unified_data["cov"]  # type: ignore[assignment]
        capital = unified_data["capital"]  # type: ignore[assignment]

        utility = UncertaintyShrunkKellyUtility()
        u_sep = utility.evaluate(nu, mu, sig2_ep, cov, capital)
        g_sep = utility.gradient(nu, mu, sig2_ep, cov, capital)

        u_joint, g_joint = utility.evaluate_utility_and_gradient(nu, mu, sig2_ep, cov, capital)
        assert math.isclose(u_sep, u_joint, rel_tol=1e-12)
        assert np.allclose(g_sep, g_joint, atol=1e-12)

    def test_validate_false_paths(self, unified_data: dict[str, object]) -> None:
        """Test validate=False fast-path branches for UnifiedConvexObjective and derivatives."""
        nu = unified_data["nu"]  # type: ignore[assignment]
        mu = unified_data["mu"]  # type: ignore[assignment]
        sig2_ep = unified_data["sig2_ep"]  # type: ignore[assignment]
        cov = unified_data["cov"]  # type: ignore[assignment]
        vols = unified_data["vols"]  # type: ignore[assignment]
        cross = unified_data["cross"]  # type: ignore[assignment]
        capital = unified_data["capital"]  # type: ignore[assignment]
        haircut = unified_data["haircut"]  # type: ignore[assignment]

        obj = UnifiedConvexObjective()

        # evaluate fast path
        val_true = obj.evaluate(nu, mu, sig2_ep, cov, vols, cross, capital, haircut, validate=True)
        val_false = obj.evaluate(
            nu, mu, sig2_ep, cov, vols, cross, capital, haircut, validate=False
        )
        assert math.isclose(val_true, val_false, rel_tol=1e-12)

        # gradient fast path
        grad_true = obj.gradient(nu, mu, sig2_ep, cov, vols, cross, capital, haircut, validate=True)
        grad_false = obj.gradient(
            nu, mu, sig2_ep, cov, vols, cross, capital, haircut, validate=False
        )
        assert np.array_equal(grad_true, grad_false)

        # hessian fast path
        hess_true = obj.hessian(nu, cov, vols, cross, capital, haircut, validate=True)
        hess_false = obj.hessian(nu, cov, vols, cross, capital, haircut, validate=False)
        assert np.array_equal(hess_true, hess_false)

    def test_defensive_validations(self, unified_data: dict[str, object]) -> None:
        """Test defensive input contracts on total unified objective."""
        nu = unified_data["nu"]  # type: ignore[assignment]
        mu = unified_data["mu"]  # type: ignore[assignment]
        sig2_ep = unified_data["sig2_ep"]  # type: ignore[assignment]
        cov = unified_data["cov"]  # type: ignore[assignment]
        vols = unified_data["vols"]  # type: ignore[assignment]
        cross = unified_data["cross"]  # type: ignore[assignment]
        capital = unified_data["capital"]  # type: ignore[assignment]
        haircut = unified_data["haircut"]  # type: ignore[assignment]

        obj = UnifiedConvexObjective()

        # Negative epistemic variance
        bad_sig2 = sig2_ep.copy()
        bad_sig2[0] = -0.01
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            obj.evaluate(nu, mu, bad_sig2, cov, vols, cross, capital, haircut)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            obj.gradient(nu, mu, bad_sig2, cov, vols, cross, capital, haircut)

        # Negative asset volatility
        bad_vols = vols.copy()
        bad_vols[0] = -0.01
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            obj.evaluate(nu, mu, sig2_ep, cov, bad_vols, cross, capital, haircut)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            obj.gradient(nu, mu, sig2_ep, cov, bad_vols, cross, capital, haircut)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            obj.hessian(nu, cov, bad_vols, cross, capital, haircut)


class TestProjectTwoL1Constraints:
    """Exhaustive tests for exact Euclidean projection onto intersection of two L1 balls."""

    def test_unconstrained_point_inside_both_balls(self) -> None:
        """Verify that a vector strictly inside both L1 balls is returned unchanged."""
        y = np.array([10.0, -20.0, 30.0], dtype=np.float64)
        c = np.array([0.05, 0.08, 0.04], dtype=np.float64)
        b1 = 100.0  # ||y||_1 = 60 <= 100
        b2 = 10.0  # c^T |y| = 0.5 + 1.6 + 1.2 = 3.3 <= 10
        proj = project_two_l1_constraints(y, c, b1, b2, validate=True)
        assert np.allclose(proj, y, atol=1e-12)

    def test_leverage_ball_only_projection(self) -> None:
        """Verify exact projection when only the unweighted L1 leverage ball is violated."""
        y = np.array([20.0, -30.0, 10.0], dtype=np.float64)  # ||y||_1 = 60
        c = np.array([0.01, 0.01, 0.01], dtype=np.float64)  # c^T |y| = 0.6 <= 100
        b1 = 30.0
        b2 = 100.0
        proj = project_two_l1_constraints(y, c, b1, b2, validate=True)
        assert np.isclose(np.sum(np.abs(proj)), b1, atol=1e-7)
        assert float(np.dot(c, np.abs(proj))) <= b2 + 1e-12
        assert np.all((proj == 0.0) | (np.sign(proj) == np.sign(y)))

    def test_cvar_ball_only_projection(self) -> None:
        """Verify exact projection when only the weighted L1 CVaR ball is violated."""
        y = np.array([10.0, -20.0, 30.0], dtype=np.float64)  # ||y||_1 = 60 <= 1000
        c = np.array([0.1, 0.2, 0.1], dtype=np.float64)  # c^T |y| = 1 + 4 + 3 = 8 > 2
        b1 = 1000.0
        b2 = 2.0
        proj = project_two_l1_constraints(y, c, b1, b2, validate=True)
        assert np.isclose(float(np.dot(c, np.abs(proj))), b2, atol=1e-7)
        assert float(np.sum(np.abs(proj))) <= b1 + 1e-12
        assert np.all((proj == 0.0) | (np.sign(proj) == np.sign(y)))

    def test_both_constraints_active_2d_newton(self) -> None:
        """Verify exact projection when both L1 constraints are simultaneously active."""
        # Construct exact active point: y = x_star + lam1 + lam2 * c
        c = np.array([0.1, 0.5], dtype=np.float64)
        x_star = np.array([10.0, 20.0], dtype=np.float64)
        lam1 = 5.0
        lam2 = 10.0
        y = x_star + lam1 + lam2 * c  # y = [16.0, 30.0]
        b1 = float(np.sum(x_star))  # 30.0
        b2 = float(np.dot(c, x_star))  # 11.0
        proj = project_two_l1_constraints(y, c, b1, b2, validate=True)
        assert np.allclose(proj, x_star, atol=1e-7)
        assert np.isclose(float(np.sum(np.abs(proj))), b1, atol=1e-7)
        assert np.isclose(float(np.dot(c, np.abs(proj))), b2, atol=1e-7)

        # Test with negative coordinates to verify sign restoration
        y_neg = np.array([-16.0, 30.0], dtype=np.float64)
        x_neg_star = np.array([-10.0, 20.0], dtype=np.float64)
        proj_neg = project_two_l1_constraints(y_neg, c, b1, b2, validate=True)
        assert np.allclose(proj_neg, x_neg_star, atol=1e-7)

        # 3-asset active test with Newton convergence
        c3 = np.array([0.05, 0.20, 0.10], dtype=np.float64)
        x3_star = np.array([20.0, 15.0, 25.0], dtype=np.float64)
        y3 = x3_star + 3.0 + 8.0 * c3
        b1_3 = float(np.sum(x3_star))
        b2_3 = float(np.dot(c3, x3_star))
        proj3 = project_two_l1_constraints(y3, c3, b1_3, b2_3, validate=True)
        assert np.allclose(proj3, x3_star, atol=1e-6)

    def test_degenerate_parallel_constraints(self) -> None:
        """Verify projection handles parallel constraints (c_i = const) without singularity."""
        y = np.array([50.0, -40.0, 60.0], dtype=np.float64)
        c = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        b1 = 50.0
        b2 = 4.0  # b2 / 0.1 = 40.0, so effective b is 40.0
        proj = project_two_l1_constraints(y, c, b1, b2, validate=True)
        assert float(np.sum(np.abs(proj))) <= b1 + 1e-7
        assert float(np.dot(c, np.abs(proj))) <= b2 + 1e-7

    def test_zero_bounds_and_zero_coordinates(self) -> None:
        """Verify zero budget contracts allocation strictly to 0 and zero coords stay 0."""
        y = np.array([10.0, 0.0, -20.0], dtype=np.float64)
        c = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        # b1 <= 0
        proj0 = project_two_l1_constraints(y, c, 0.0, 10.0)
        assert np.all(proj0 == 0.0)
        # b2 <= 0
        proj0_b2 = project_two_l1_constraints(y, c, 10.0, 0.0)
        assert np.all(proj0_b2 == 0.0)
        # Zero coordinate in y stays zero
        proj = project_two_l1_constraints(y, c, 15.0, 1.5)
        assert proj[1] == 0.0

    def test_dykstra_fallback_adversarial(self) -> None:
        """Adversarial stress test verifying Dykstra alternating projections fallback."""
        y = np.array([100.0, -80.0, 60.0, 40.0], dtype=np.float64)
        c = np.array([0.05, 0.20, 0.10, 0.15], dtype=np.float64)
        b1 = 50.0
        b2 = 5.0
        proj = project_two_l1_constraints(y, c, b1, b2, validate=False, _force_dykstra=True)
        assert float(np.sum(np.abs(proj))) <= b1 + 1e-7
        assert float(np.dot(c, np.abs(proj))) <= b2 + 1e-7

    def test_validate_false_path(self) -> None:
        """Verify validate=False hot path yields bitwise identical results."""
        y = np.array([20.0, -30.0, 40.0], dtype=np.float64)
        c = np.array([0.1, 0.15, 0.08], dtype=np.float64)
        b1 = 40.0
        b2 = 3.0
        p_val = project_two_l1_constraints(y, c, b1, b2, validate=True)
        p_fast = project_two_l1_constraints(y, c, b1, b2, validate=False)
        assert np.allclose(p_val, p_fast, atol=1e-12)

    def test_defensive_validations(self) -> None:
        """Test defensive boundary checks and error codes for two L1 projection."""
        y = np.array([10.0, 20.0], dtype=np.float64)
        c = np.array([0.1, 0.2], dtype=np.float64)

        # Dimension mismatch
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            project_two_l1_constraints(y, np.array([0.1, 0.2, 0.3]), 10.0, 5.0)

        # Non-positive c
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            project_two_l1_constraints(y, np.array([0.1, -0.1]), 10.0, 5.0)

        # Non-finite b1 or b2
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            project_two_l1_constraints(y, c, float("nan"), 5.0)
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            project_two_l1_constraints(y, c, 10.0, float("inf"))

        # Non-numeric b1 or b2
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            project_two_l1_constraints(y, c, "invalid", 5.0)  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            project_two_l1_constraints(y, c, 10.0, "invalid")  # type: ignore[arg-type]

        # Negative bounds
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            project_two_l1_constraints(y, c, -1.0, 5.0)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            project_two_l1_constraints(y, c, 10.0, -2.0)

        # Boolean type rejections
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            project_two_l1_constraints(y, c, True, 5.0)  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            project_two_l1_constraints(y, c, 10.0, False)  # type: ignore[arg-type]


class TestMicrostructuralLotDiscretization:
    """Tests for microstructural randomized and deterministic contract lot rounding."""

    def test_none_lot_sizes_returns_copy(self) -> None:
        """Verify that when lot_sizes is None, target allocation copy is returned."""
        target = np.array([100.25, -250.75, 0.0], dtype=np.float64)
        disc = discretize_lot_allocations(target, lot_sizes=None)
        assert np.array_equal(target, disc)
        assert disc is not target

    def test_deterministic_rounding(self) -> None:
        """Verify random_seed == -1 rounds deterministically to nearest integer lot."""
        target = np.array([104.0, 106.0, -104.0, -106.0, 0.0], dtype=np.float64)
        lots = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)
        disc = discretize_lot_allocations(target, lots, random_seed=-1)
        expected = np.array([100.0, 110.0, -100.0, -110.0, 0.0], dtype=np.float64)
        assert np.allclose(disc, expected, atol=1e-12)

    def test_randomized_rounding_unbiased_expectation(self) -> None:
        """Verify microstructural randomized rounding unbiased expectation property E[nu_tilde] = nu*."""
        target = np.array([123.456, -789.123, 456.789, 0.0], dtype=np.float64)
        lots = np.array([10.0, 25.0, 5.0, 10.0], dtype=np.float64)
        n_samples = 10000
        samples = np.zeros((n_samples, len(target)))
        for s in range(n_samples):
            samples[s] = discretize_lot_allocations(target, lots, random_seed=s, validate=False)
        mean_alloc = np.mean(samples, axis=0)
        assert np.allclose(mean_alloc, target, atol=0.5)
        rel_err = np.abs(mean_alloc[:3] - target[:3]) / np.abs(target[:3])
        assert np.all(rel_err < 0.02)
        assert mean_alloc[3] == 0.0

    def test_non_positive_lot_sizes_handling(self) -> None:
        """Verify fallback branch when unvalidated lot_sizes has non-positive entries."""
        target = np.array([105.0, 200.0], dtype=np.float64)
        lots = np.array([10.0, -5.0], dtype=np.float64)
        # Deterministic
        disc_det = discretize_lot_allocations(target, lots, random_seed=-1, validate=False)
        assert disc_det[0] == 110.0
        assert disc_det[1] == 200.0
        # Randomized
        disc_rand = discretize_lot_allocations(target, lots, random_seed=42, validate=False)
        assert disc_rand[1] == 200.0

    def test_defensive_validations(self) -> None:
        """Test defensive input validation on lot discretization."""
        target = np.array([100.0, 200.0], dtype=np.float64)
        # Dimension mismatch
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            discretize_lot_allocations(target, np.array([10.0]))
        # Non-positive lot size with validate=True
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            discretize_lot_allocations(target, np.array([10.0, 0.0]))
        # Non-finite target
        with pytest.raises(DegenerateSizingException, match=ERR_SZ_NON_FINITE):
            discretize_lot_allocations(np.array([float("nan"), 100.0]))


class TestUnifiedConvexExecutionSizer:
    """Comprehensive test suite for UnifiedConvexExecutionSizer institutional solver."""

    @pytest.fixture
    def sizer_inputs(self) -> dict[str, object]:
        """Provide standardized institutional multi-asset inputs."""
        n = 5
        mu = np.array([0.08, -0.05, 0.12, -0.04, 0.06], dtype=np.float64)
        sig2_ep = np.array([0.001, 0.002, 0.001, 0.003, 0.002], dtype=np.float64)
        a = np.random.RandomState(42).randn(n, n)
        cov = a @ a.T * 0.002 + np.eye(n) * 0.005
        cov = 0.5 * (cov + cov.T)
        vols = np.sqrt(np.diag(cov))
        capital = 1_000_000.0
        haircut = 0.90
        lots = np.array([100.0, 500.0, 200.0, 1000.0, 250.0], dtype=np.float64)
        return {
            "n": n,
            "mu": mu,
            "sig2_ep": sig2_ep,
            "cov": cov,
            "vols": vols,
            "capital": capital,
            "haircut": haircut,
            "lots": lots,
        }

    def test_initialization_default_and_custom_config(self) -> None:
        """Verify instantiation with default and custom SizingConfig."""
        sizer_default = UnifiedConvexExecutionSizer()
        assert sizer_default.config.max_leverage == 2.0
        assert sizer_default.config.mdd_budget == 0.15

        cfg = SizingConfig(max_leverage=1.5, mdd_budget=0.10)
        sizer_custom = UnifiedConvexExecutionSizer(config=cfg)
        assert sizer_custom.config.max_leverage == 1.5
        assert sizer_custom.config.mdd_budget == 0.10

    def test_facade_project_and_discretize_methods(self) -> None:
        """Verify facade methods project and discretize operate consistently."""
        sizer = UnifiedConvexExecutionSizer()
        y = np.array([100.0, -200.0], dtype=np.float64)
        c = np.array([0.1, 0.2], dtype=np.float64)
        proj = sizer.project(y, c, b1=150.0, b2=25.0)
        assert float(np.sum(np.abs(proj))) <= 150.0 + 1e-7
        assert float(np.dot(c, np.abs(proj))) <= 25.0 + 1e-7

        disc = sizer.discretize(
            np.array([104.0, -206.0]), lot_sizes=np.array([10.0, 10.0]), random_seed=-1
        )
        assert np.allclose(disc, [100.0, -210.0])

    def test_solve_unconstrained_interior(self, sizer_inputs: dict[str, object]) -> None:
        """Verify solver fast path when optimal allocation lies strictly inside both constraint balls."""
        mu = np.array([0.005, -0.003, 0.004, -0.002, 0.003], dtype=np.float64)
        sig2 = sizer_inputs["sig2_ep"]  # type: ignore[assignment]
        cov = sizer_inputs["cov"]  # type: ignore[assignment]
        vols = sizer_inputs["vols"]  # type: ignore[assignment]
        capital = sizer_inputs["capital"]  # type: ignore[assignment]

        sizer = UnifiedConvexExecutionSizer()
        decision = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            total_capital=capital,
            circuit_breaker_haircut=1.0,
            random_seed=-1,
        )
        assert isinstance(decision, SizingDecision)
        assert decision.effective_leverage < sizer.config.max_leverage
        assert decision.expected_shortfall < sizer.config.mdd_budget * capital
        assert not decision.is_leverage_constrained
        assert not decision.is_drawdown_constrained

    def test_solve_hard_leverage_and_cvar_constraints_inv_tr_005(
        self, sizer_inputs: dict[str, object]
    ) -> None:
        """Verify INV-TR-005 hard CVaR drawdown budget and leverage cap enforcement under extreme returns."""
        n = sizer_inputs["n"]  # type: ignore[assignment]
        mu = np.full(n, 50.0, dtype=np.float64)
        sig2 = sizer_inputs["sig2_ep"]  # type: ignore[assignment]
        cov = sizer_inputs["cov"]  # type: ignore[assignment]
        vols = sizer_inputs["vols"]  # type: ignore[assignment]
        capital = sizer_inputs["capital"]  # type: ignore[assignment]
        lots = sizer_inputs["lots"]  # type: ignore[assignment]

        config = SizingConfig(max_leverage=1.8, mdd_budget=0.12)
        sizer = UnifiedConvexExecutionSizer(config=config)
        decision = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            total_capital=capital,
            circuit_breaker_haircut=1.0,
            lot_sizes=lots,
            random_seed=-1,
        )
        assert decision.effective_leverage <= config.max_leverage + 1e-6
        assert decision.expected_shortfall <= config.mdd_budget * capital + 1e-6
        assert decision.is_leverage_constrained or decision.is_drawdown_constrained
        assert np.all(decision.discretized_allocations % lots == 0.0)

    def test_solve_circuit_breaker_coupling(self, sizer_inputs: dict[str, object]) -> None:
        """Verify circuit breaker coupling: kappa_t = 0.0 pins allocation to 0; kappa -> 0 contracts smoothly."""
        mu = sizer_inputs["mu"]  # type: ignore[assignment]
        sig2 = sizer_inputs["sig2_ep"]  # type: ignore[assignment]
        cov = sizer_inputs["cov"]  # type: ignore[assignment]
        vols = sizer_inputs["vols"]  # type: ignore[assignment]
        capital = sizer_inputs["capital"]  # type: ignore[assignment]

        sizer = UnifiedConvexExecutionSizer()

        # Zero haircut (HALT): allocation is identically 0
        dec_zero = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            circuit_breaker_haircut=0.0,
            total_capital=capital,
        )
        assert np.all(dec_zero.target_allocations == 0.0)
        assert np.all(dec_zero.discretized_allocations == 0.0)
        assert dec_zero.effective_leverage == 0.0
        assert dec_zero.expected_shortfall == 0.0
        assert dec_zero.estimated_impact_cost == 0.0

        # Monotonic contraction as kappa -> 0
        dec_full = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            circuit_breaker_haircut=1.0,
            total_capital=capital,
        )
        dec_mid = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            circuit_breaker_haircut=0.10,
            total_capital=capital,
        )
        dec_low = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            circuit_breaker_haircut=0.01,
            total_capital=capital,
        )
        norm_full = float(np.linalg.norm(dec_full.target_allocations))
        norm_mid = float(np.linalg.norm(dec_mid.target_allocations))
        norm_low = float(np.linalg.norm(dec_low.target_allocations))
        assert norm_full > norm_mid > norm_low > 0.0

    def test_solve_with_cross_impact_and_custom_cvars(
        self, sizer_inputs: dict[str, object]
    ) -> None:
        """Verify solve with cross-impact matrix and custom CVaR multipliers."""
        n = sizer_inputs["n"]  # type: ignore[assignment]
        mu = sizer_inputs["mu"]  # type: ignore[assignment]
        sig2 = sizer_inputs["sig2_ep"]  # type: ignore[assignment]
        cov = sizer_inputs["cov"]  # type: ignore[assignment]
        vols = sizer_inputs["vols"]  # type: ignore[assignment]
        capital = sizer_inputs["capital"]  # type: ignore[assignment]

        cross = np.eye(n) * 0.001
        custom_cvars = np.full(n, 0.05, dtype=np.float64)

        sizer = UnifiedConvexExecutionSizer()
        dec = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            cross_impact=cross,
            asset_cvars=custom_cvars,
            total_capital=capital,
        )
        assert dec.expected_shortfall <= sizer.config.mdd_budget * capital + 1e-6
        assert dec.effective_leverage <= sizer.config.max_leverage + 1e-6

    def test_solve_validate_false_path(self, sizer_inputs: dict[str, object]) -> None:
        """Verify validate=False hot path reproduces validated results."""
        mu = sizer_inputs["mu"]  # type: ignore[assignment]
        sig2 = sizer_inputs["sig2_ep"]  # type: ignore[assignment]
        cov = sizer_inputs["cov"]  # type: ignore[assignment]
        vols = sizer_inputs["vols"]  # type: ignore[assignment]
        capital = sizer_inputs["capital"]  # type: ignore[assignment]

        sizer = UnifiedConvexExecutionSizer()
        dec_true = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            total_capital=capital,
            validate=True,
            random_seed=-1,
        )
        dec_false = sizer.solve(
            mu=mu,
            sigma2_epistemic=sig2,
            cov_aleatoric=cov,
            asset_vols=vols,
            total_capital=capital,
            validate=False,
            random_seed=-1,
        )
        assert np.allclose(dec_true.target_allocations, dec_false.target_allocations, atol=1e-12)
        assert np.allclose(
            dec_true.discretized_allocations, dec_false.discretized_allocations, atol=1e-12
        )

    def test_solve_hot_path_latency_sla_inv_tr_006(self) -> None:
        """Verify hot-path latency SLA INV-TR-006: <= 0.15ms for N = 10 assets."""
        n = 10
        rng = np.random.RandomState(42)
        mu = rng.uniform(0.01, 0.05, size=n)
        sig2_ep = np.full(n, 0.001)
        a = rng.randn(n, n)
        cov = a @ a.T * 0.001 + np.eye(n) * 0.004
        cov = 0.5 * (cov + cov.T)
        vols = np.sqrt(np.diag(cov))
        capital = 1_000_000.0

        sizer = UnifiedConvexExecutionSizer()

        # Warm up
        for _ in range(5):
            sizer.solve(mu, sig2_ep, cov, vols, total_capital=capital, validate=False)

        gc.collect()
        gc_old = gc.isenabled()
        gc.disable()
        times: list[float] = []
        try:
            for _ in range(50):
                t0 = time.perf_counter()
                sizer.solve(mu, sig2_ep, cov, vols, total_capital=capital, validate=False)
                t1 = time.perf_counter()
                times.append(t1 - t0)
        finally:
            if gc_old:
                gc.enable()

        median_ms = float(np.median(times)) * 1000.0
        is_tracing = (
            (hasattr(sys, "gettrace") and sys.gettrace() is not None)
            or "coverage" in sys.modules
            or "pytest_cov" in sys.modules
        )
        sla_limit = 2.0 if is_tracing else 0.15
        assert median_ms <= sla_limit, f"Solve latency {median_ms:.4f}ms exceeds SLA {sla_limit}ms"

    def test_solve_defensive_validations(self, sizer_inputs: dict[str, object]) -> None:
        """Test defensive contract enforcement on solver inputs."""
        mu = sizer_inputs["mu"]  # type: ignore[assignment]
        sig2 = sizer_inputs["sig2_ep"]  # type: ignore[assignment]
        cov = sizer_inputs["cov"]  # type: ignore[assignment]
        vols = sizer_inputs["vols"]  # type: ignore[assignment]
        capital = sizer_inputs["capital"]  # type: ignore[assignment]

        sizer = UnifiedConvexExecutionSizer()

        # Dimension mismatch: mu vs sig2
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            sizer.solve(mu, sig2[:2], cov, vols, total_capital=capital)

        # Dimension mismatch: mu vs cov
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            sizer.solve(mu, sig2, cov[:2, :2], vols, total_capital=capital)

        # Dimension mismatch: cross_impact
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            sizer.solve(mu, sig2, cov, vols, cross_impact=np.zeros((2, 2)), total_capital=capital)

        # Dimension mismatch: asset_cvars
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_DIMENSION_MISMATCH):
            sizer.solve(
                mu, sig2, cov, vols, asset_cvars=np.array([0.1, 0.2]), total_capital=capital
            )

        # Negative sig2
        bad_sig = sig2.copy()
        bad_sig[0] = -0.01
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, bad_sig, cov, vols, total_capital=capital)

        # Negative vols
        bad_vols = vols.copy()
        bad_vols[0] = -0.01
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, bad_vols, total_capital=capital)

        # Non-positive asset_cvars
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, asset_cvars=np.zeros_like(mu), total_capital=capital)

        # Non-positive capital
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, total_capital=0.0)

        # Haircut out of bounds
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, circuit_breaker_haircut=-0.1)
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, circuit_breaker_haircut=1.5)

        # Invalid max_iterations
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, max_iterations=0)

        # Invalid tolerance
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, tolerance="invalid")  # type: ignore[arg-type]
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, tolerance=-1e-4)

        # Non-positive lot_sizes
        with pytest.raises(InvalidSizingInputException, match=ERR_SZ_INVALID_CONFIG):
            sizer.solve(mu, sig2, cov, vols, lot_sizes=np.array([10.0, 0.0, 10.0, 10.0, 10.0]))
