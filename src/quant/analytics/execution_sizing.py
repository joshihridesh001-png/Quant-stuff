"""Unified Convex Execution Sizing & Risk-Calibrated Calibration Subsystem.

Purpose:
    Provides institutional-grade, mathematically rigorous portfolio execution sizing calibration
    enforcing strict global concavity (INV-TR-004) and sub-0.02ms execution latency (INV-TR-006):
    1. Uncertainty-Shrunk Kelly Utility: Dynamically penalizes expected return forecasts by
       epistemic model disagreement variance, eliminating overbetting cliffs (Rule 4.3).
    2. Generalized Pseudo-Huber Market Impact Penalty: 3/2-power non-linear execution friction
       and permanent cross-impact tensor ensuring universal Square-Root Law slippage.
    3. Circuit Breaker Regularizer: Continuous quadratic contraction smoothly pinning dollar
       allocations to zero as macro/entropy shocks escalate.
    4. Unified Convex Objective: Strictly concave total allocation objective L(nu) guaranteeing
       a unique global maximizer without local extrema or solver stalls.

Dependencies:
    - math: Scalar finiteness and arithmetic operations.
    - dataclasses: Immutable frozen domain value containers.
    - typing: Static typing annotations and Final constants.
    - numpy: Vectorized linear algebra and array transforms.

Structural Relationship:
    - Ingests: Expected returns mu and epistemic variance sigma2_epistemic from RD-DMA (ensemble.py),
      continuous logistic haircut kappa_t from circuit breakers (circuit_breakers.py),
      and cross-impact tensor Lambda_cross from market impact (market_impact.py).
    - Emits: SizingConfig, SizingDecision, analytical utility, gradient, and Hessian tensors,
      project_two_l1_constraints, discretize_lot_allocations, UnifiedConvexExecutionSizer.
    - Consumed by: Portfolio execution router, risk management, and order sizing pipeline.

Invariants Enforced:
    - INV-TR-004 (Strict Global Concavity): nabla^2 L(nu) << 0 everywhere; all eigenvalues of the
      total Hessian are strictly negative, guaranteeing a unique global allocation.
    - INV-TR-005 (Hard Drawdown Budget & Gross Leverage Cap): CVaR_alpha(nu) <= MDD_budget * W_t and
      ||nu||_1 <= L_max * W_t are strictly enforced across continuous and expected discrete space.
    - INV-TR-006 (Hot-Path Execution Latency SLA): Full execution sizing solve completes in
      <= 0.15ms for N = 10 assets.
    - Rule 4.3: Zero scipy.optimize or black-box solvers; deterministic projected gradient / KKT solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

# Diagnostic Fault Vector Constants (Rule 2: Deterministic Diagnostics)
ERR_SZ_INVALID_CONFIG: Final[str] = (
    "ERR-SZ-001: Invalid hyperparameter configuration or parameter bounds"
)
ERR_SZ_NON_FINITE: Final[str] = "ERR-SZ-002: Non-finite input or numerical singularity"
ERR_SZ_SINGULAR_COVARIANCE: Final[str] = (
    "ERR-SZ-003: Covariance matrix is singular, asymmetric, or non-positive semi-definite"
)
ERR_SZ_INFEASIBLE: Final[str] = "ERR-SZ-004: Infeasible sizing constraints or allocation bounds"
ERR_SZ_CONCAVITY_VIOLATED: Final[str] = "ERR-SZ-005: Strict global concavity violated"
ERR_SZ_DIMENSION_MISMATCH: Final[str] = (
    "ERR-SZ-006: Dimensional mismatch across allocation, return, or covariance arrays"
)


class SizingError(Exception):
    """Base exception for all execution sizing and allocation calibration errors."""


class DegenerateSizingException(SizingError):
    """Raised when non-finite inputs (NaN/Inf) or numerical singularities are detected."""


class InfeasibleSizingException(SizingError):
    """Raised when sizing constraints or allocation bounds cannot be satisfied."""


class InvalidSizingInputException(SizingError):
    """Raised on invalid hyperparameter configurations, invalid types, or dimensional mismatches."""


@dataclass(frozen=True, slots=True)
class SizingConfig:
    """Hyperparameter configuration container for convex execution sizing calibration.

    Attributes:
        confidence_level: Downside tail risk confidence level alpha in (0.50, 1.0), default 0.99.
        mdd_budget: Maximum portfolio drawdown budget fraction in (0.0, 1.0], default 0.15.
        max_leverage: Maximum gross portfolio leverage ceiling in (0.0, 10.0], default 2.0.
        epistemic_shrinkage_lambda: Epistemic disagreement shrinkage multiplier >= 0.0, default 1.0.
        risk_aversion_gamma: Risk aversion coefficient gamma > 0.0, default 1.0.
        impact_penalty_eta: Non-linear 3/2-power market impact penalty scale >= 0.0, default 0.10.
        impact_delta: Pseudo-Huber transition smoothing threshold delta > 0.0, default 1.0.
        min_haircut_floor: Minimum circuit breaker haircut floor kappa_floor > 0.0, default 1e-4.
        lot_sizes: Optional 1D array of positive contract lot sizes for microstructure rounding.
    """

    confidence_level: float = 0.99
    mdd_budget: float = 0.15
    max_leverage: float = 2.0
    epistemic_shrinkage_lambda: float = 1.0
    risk_aversion_gamma: float = 1.0
    impact_penalty_eta: float = 0.10
    impact_delta: float = 1.0
    min_haircut_floor: float = 1e-4
    lot_sizes: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate invariant boundaries and types upon instantiation."""
        # Functional Purpose: Validate sizing hyperparameters and enforce domain contracts.
        # Explicit Dependency Tracking: All scalar fields and optional lot_sizes array.
        # Structural Relationship: Configures all sizing utility, penalty, and optimizer engines.
        # Defensive Invariant: All floats must be finite; domains bounded as specified.

        # 1. Validate scalar float typing and finiteness
        float_fields = (
            "confidence_level",
            "mdd_budget",
            "max_leverage",
            "epistemic_shrinkage_lambda",
            "risk_aversion_gamma",
            "impact_penalty_eta",
            "impact_delta",
            "min_haircut_floor",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateSizingException(
                    f"{ERR_SZ_NON_FINITE}: {field_name} must be a finite float, got {val}"
                )

        # 2. Confidence level bounds (0.50 < alpha < 1.0)
        if not (0.50 < self.confidence_level < 1.0):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: confidence_level must be in (0.50, 1.0), got {self.confidence_level}"
            )

        # 3. MDD budget bounds (0.0 < mdd_budget <= 1.0)
        if not (0.0 < self.mdd_budget <= 1.0):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: mdd_budget must be in (0.0, 1.0], got {self.mdd_budget}"
            )

        # 4. Gross leverage ceiling bounds (0.0 < max_leverage <= 10.0)
        if not (0.0 < self.max_leverage <= 10.0):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: max_leverage must be in (0.0, 10.0], got {self.max_leverage}"
            )

        # 5. Epistemic shrinkage multiplier bounds (lambda >= 0.0)
        if self.epistemic_shrinkage_lambda < 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: epistemic_shrinkage_lambda must be >= 0.0, got {self.epistemic_shrinkage_lambda}"
            )

        # 6. Risk aversion bounds (gamma > 0.0)
        if self.risk_aversion_gamma <= 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: risk_aversion_gamma must be strictly positive, got {self.risk_aversion_gamma}"
            )

        # 7. Impact penalty eta bounds (eta >= 0.0)
        if self.impact_penalty_eta < 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: impact_penalty_eta must be >= 0.0, got {self.impact_penalty_eta}"
            )

        # 8. Impact delta smoothing threshold bounds (delta > 0.0)
        if self.impact_delta <= 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: impact_delta must be strictly positive, got {self.impact_delta}"
            )

        # 9. Minimum haircut floor bounds (0.0 < min_haircut_floor <= 1.0)
        if not (0.0 < self.min_haircut_floor <= 1.0):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: min_haircut_floor must be in (0.0, 1.0], got {self.min_haircut_floor}"
            )

        # 10. Optional lot sizes array validation
        if self.lot_sizes is not None:
            if not isinstance(self.lot_sizes, np.ndarray):
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: lot_sizes must be a numpy ndarray, got {type(self.lot_sizes).__name__}"
                )
            if self.lot_sizes.ndim != 1:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_DIMENSION_MISMATCH}: lot_sizes must be 1D, got shape {self.lot_sizes.shape}"
                )
            if self.lot_sizes.size < 1:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: lot_sizes array cannot be empty"
                )
            if not np.issubdtype(self.lot_sizes.dtype, np.number):
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: lot_sizes elements must be numeric, got {self.lot_sizes.dtype}"
                )
            if not np.isfinite(self.lot_sizes).all():
                raise DegenerateSizingException(
                    f"{ERR_SZ_NON_FINITE}: lot_sizes contains non-finite elements (NaN or Inf)"
                )
            if not (self.lot_sizes > 0.0).all():
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: all lot_sizes elements must be strictly positive (> 0.0)"
                )
            self.lot_sizes.flags.writeable = False

    def __eq__(self, other: object) -> bool:
        """Evaluate value equality across scalar fields and optional lot_sizes array."""
        if not isinstance(other, SizingConfig):
            return False
        for field_name in (
            "confidence_level",
            "mdd_budget",
            "max_leverage",
            "epistemic_shrinkage_lambda",
            "risk_aversion_gamma",
            "impact_penalty_eta",
            "impact_delta",
            "min_haircut_floor",
        ):
            if getattr(self, field_name) != getattr(other, field_name):
                return False
        if (self.lot_sizes is None) != (other.lot_sizes is None):
            return False
        if self.lot_sizes is not None and other.lot_sizes is not None:
            return bool(np.array_equal(self.lot_sizes, other.lot_sizes))
        return True

    def __hash__(self) -> int:
        """Compute deterministic hash for immutable configuration container."""
        lot_tuple = tuple(self.lot_sizes.tolist()) if self.lot_sizes is not None else None
        return hash(
            (
                self.confidence_level,
                self.mdd_budget,
                self.max_leverage,
                self.epistemic_shrinkage_lambda,
                self.risk_aversion_gamma,
                self.impact_penalty_eta,
                self.impact_delta,
                self.min_haircut_floor,
                lot_tuple,
            )
        )


@dataclass(frozen=True, slots=True)
class SizingDecision:
    """Immutable domain container representing calibrated execution sizing allocations.

    Attributes:
        target_allocations: Continuous dollar allocation vector nu in R^N.
        discretized_allocations: Lot-rounded integer-executable dollar allocation vector nu_tilde in R^N.
        effective_leverage: Total gross portfolio leverage ||nu||_1 / W_t >= 0.0.
        expected_shortfall: Portfolio Expected Shortfall CVaR_alpha in dollar or fraction terms.
        estimated_impact_cost: Total non-linear execution friction cost >= 0.0.
        circuit_breaker_haircut: Active continuous circuit breaker multiplier kappa_t in [0.0, 1.0].
        is_drawdown_constrained: True if the Rockafellar-Uryasev CVaR MDD budget bound is active.
        is_leverage_constrained: True if the gross leverage ceiling bound is active.
    """

    target_allocations: np.ndarray
    discretized_allocations: np.ndarray
    effective_leverage: float
    expected_shortfall: float
    estimated_impact_cost: float
    circuit_breaker_haircut: float
    is_drawdown_constrained: bool
    is_leverage_constrained: bool

    def __post_init__(self) -> None:
        """Validate invariant boundaries and array contracts upon instantiation."""
        # Functional Purpose: Verify consistency and defensive invariants of sizing decision.
        # Explicit Dependency Tracking: target_allocations, discretized_allocations, scalars.
        # Structural Relationship: Emitted to order management and execution routing systems.
        # Defensive Invariant: Matching 1D arrays, finite numbers, non-negative costs and leverage.

        # 1. Validate target_allocations array
        if not isinstance(self.target_allocations, np.ndarray):
            raise InvalidSizingInputException(
                f"{ERR_SZ_DIMENSION_MISMATCH}: target_allocations must be a numpy ndarray, got {type(self.target_allocations).__name__}"
            )
        if self.target_allocations.ndim != 1:
            raise InvalidSizingInputException(
                f"{ERR_SZ_DIMENSION_MISMATCH}: target_allocations must be 1D, got shape {self.target_allocations.shape}"
            )
        if not np.isfinite(self.target_allocations).all():
            raise DegenerateSizingException(
                f"{ERR_SZ_NON_FINITE}: target_allocations contains non-finite elements (NaN or Inf)"
            )

        # 2. Validate discretized_allocations array
        if not isinstance(self.discretized_allocations, np.ndarray):
            raise InvalidSizingInputException(
                f"{ERR_SZ_DIMENSION_MISMATCH}: discretized_allocations must be a numpy ndarray, got {type(self.discretized_allocations).__name__}"
            )
        if self.discretized_allocations.ndim != 1:
            raise InvalidSizingInputException(
                f"{ERR_SZ_DIMENSION_MISMATCH}: discretized_allocations must be 1D, got shape {self.discretized_allocations.shape}"
            )
        if not np.isfinite(self.discretized_allocations).all():
            raise DegenerateSizingException(
                f"{ERR_SZ_NON_FINITE}: discretized_allocations contains non-finite elements (NaN or Inf)"
            )

        # 3. Dimensional alignment between target and discretized allocations
        if self.target_allocations.shape != self.discretized_allocations.shape:
            raise InvalidSizingInputException(
                f"{ERR_SZ_DIMENSION_MISMATCH}: target_allocations shape {self.target_allocations.shape} "
                f"does not match discretized_allocations shape {self.discretized_allocations.shape}"
            )

        # Freeze array buffers to preserve true immutability
        self.target_allocations.flags.writeable = False
        self.discretized_allocations.flags.writeable = False

        # 4. Validate scalar float typing and finiteness
        float_fields = (
            "effective_leverage",
            "expected_shortfall",
            "estimated_impact_cost",
            "circuit_breaker_haircut",
        )
        for field_name in float_fields:
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateSizingException(
                    f"{ERR_SZ_NON_FINITE}: {field_name} must be a finite float, got {val}"
                )

        # 5. Effective leverage bounds (>= 0.0)
        if self.effective_leverage < 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: effective_leverage must be >= 0.0, got {self.effective_leverage}"
            )

        # 6. Estimated impact cost bounds (>= 0.0)
        if self.estimated_impact_cost < 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: estimated_impact_cost must be >= 0.0, got {self.estimated_impact_cost}"
            )

        # 7. Circuit breaker haircut bounds (0.0 <= kappa <= 1.0)
        if not (0.0 <= self.circuit_breaker_haircut <= 1.0):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: circuit_breaker_haircut must be in [0.0, 1.0], got {self.circuit_breaker_haircut}"
            )

        # 8. Boolean flag type verification
        if not isinstance(self.is_drawdown_constrained, bool):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: is_drawdown_constrained must be a bool, got {type(self.is_drawdown_constrained).__name__}"
            )
        if not isinstance(self.is_leverage_constrained, bool):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: is_leverage_constrained must be a bool, got {type(self.is_leverage_constrained).__name__}"
            )

    def __eq__(self, other: object) -> bool:
        """Evaluate value equality across arrays and scalar decision fields."""
        if not isinstance(other, SizingDecision):
            return False
        if not np.array_equal(self.target_allocations, other.target_allocations):
            return False
        if not np.array_equal(self.discretized_allocations, other.discretized_allocations):
            return False
        return (
            self.effective_leverage == other.effective_leverage
            and self.expected_shortfall == other.expected_shortfall
            and self.estimated_impact_cost == other.estimated_impact_cost
            and self.circuit_breaker_haircut == other.circuit_breaker_haircut
            and self.is_drawdown_constrained == other.is_drawdown_constrained
            and self.is_leverage_constrained == other.is_leverage_constrained
        )

    def __hash__(self) -> int:
        """Compute deterministic hash for immutable decision container."""
        return hash(
            (
                tuple(self.target_allocations.tolist()),
                tuple(self.discretized_allocations.tolist()),
                self.effective_leverage,
                self.expected_shortfall,
                self.estimated_impact_cost,
                self.circuit_breaker_haircut,
                self.is_drawdown_constrained,
                self.is_leverage_constrained,
            )
        )


_DEFAULT_SIZING_CONFIG: Final[SizingConfig] = SizingConfig()


def _validate_1d_array(
    name: str,
    arr: np.ndarray,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Validate 1D NumPy array typing, shape, and finiteness."""
    # Functional Purpose: Provide microsecond input sanitation for vectors in the sizing hot path.
    # Explicit Dependency Tracking: arr, name, expected_dim.
    # Structural Relationship: Called by Kelly, Impact, and Regularizer evaluation methods.
    # Defensive Invariant: 1D array, finite float64 values, dimensional match.
    if not isinstance(arr, np.ndarray):
        raise InvalidSizingInputException(
            f"{ERR_SZ_DIMENSION_MISMATCH}: {name} must be a numpy ndarray, got {type(arr).__name__}"
        )
    if arr.ndim != 1:
        raise InvalidSizingInputException(
            f"{ERR_SZ_DIMENSION_MISMATCH}: {name} must be 1-dimensional, got shape {arr.shape}"
        )
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise InvalidSizingInputException(
            f"{ERR_SZ_DIMENSION_MISMATCH}: {name} length ({arr.shape[0]}) does not match expected dimension ({expected_dim})"
        )
    if not np.isfinite(arr).all():
        raise DegenerateSizingException(
            f"{ERR_SZ_NON_FINITE}: {name} contains non-finite elements (NaN or Inf)"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_2d_matrix(
    name: str,
    mat: np.ndarray,
    expected_dim: int | None = None,
    check_symmetric: bool = True,
) -> np.ndarray:
    """Validate 2D square matrix typing, dimensions, finiteness, and symmetry."""
    # Functional Purpose: Provide microsecond validation for covariance and cross-impact matrices.
    # Explicit Dependency Tracking: mat, name, expected_dim, check_symmetric.
    # Structural Relationship: Called by Kelly and Impact evaluation methods.
    # Defensive Invariant: 2D square matrix, finite values, symmetry within 1e-8.
    if not isinstance(mat, np.ndarray):
        raise InvalidSizingInputException(
            f"{ERR_SZ_DIMENSION_MISMATCH}: {name} must be a numpy ndarray, got {type(mat).__name__}"
        )
    if mat.ndim != 2:
        raise InvalidSizingInputException(
            f"{ERR_SZ_DIMENSION_MISMATCH}: {name} must be 2-dimensional, got shape {mat.shape}"
        )
    if mat.shape[0] != mat.shape[1]:
        raise InvalidSizingInputException(
            f"{ERR_SZ_DIMENSION_MISMATCH}: {name} must be square matrix, got shape {mat.shape}"
        )
    if expected_dim is not None and mat.shape[0] != expected_dim:
        raise InvalidSizingInputException(
            f"{ERR_SZ_DIMENSION_MISMATCH}: {name} dimension ({mat.shape[0]}) does not match expected ({expected_dim})"
        )
    if not np.isfinite(mat).all():
        raise DegenerateSizingException(
            f"{ERR_SZ_NON_FINITE}: {name} contains non-finite elements (NaN or Inf)"
        )
    if check_symmetric:
        if float(np.max(np.abs(mat - mat.T))) > 1e-8:
            raise DegenerateSizingException(
                f"{ERR_SZ_SINGULAR_COVARIANCE}: {name} is not symmetric within 1e-8 tolerance"
            )
        if float(np.min(np.diag(mat))) < -1e-8:
            raise DegenerateSizingException(
                f"{ERR_SZ_SINGULAR_COVARIANCE}: {name} contains negative diagonal variances"
            )
        eigvals = np.linalg.eigvalsh(mat)
        if float(np.min(eigvals)) < -1e-8:
            raise DegenerateSizingException(
                f"{ERR_SZ_SINGULAR_COVARIANCE}: {name} is not positive semi-definite (min eigenvalue {float(np.min(eigvals)):.6e} < -1e-8)"
            )
    return np.ascontiguousarray(mat, dtype=np.float64)


def _validate_positive_capital(total_capital: float) -> float:
    """Validate total portfolio capital finiteness and strict positivity."""
    # Functional Purpose: Ensure portfolio capital W_t is finite and strictly positive.
    # Explicit Dependency Tracking: total_capital scalar.
    # Structural Relationship: Normalizes dollar exposures in Kelly and Impact formulations.
    # Defensive Invariant: total_capital > 0.0 and math.isfinite.
    if not (isinstance(total_capital, (int, float)) and not isinstance(total_capital, bool)):
        raise InvalidSizingInputException(
            f"{ERR_SZ_INVALID_CONFIG}: total_capital must be numeric, got {type(total_capital).__name__}"
        )
    if not math.isfinite(total_capital):
        raise DegenerateSizingException(
            f"{ERR_SZ_NON_FINITE}: total_capital must be a finite float, got {total_capital}"
        )
    if total_capital <= 0.0:
        raise InvalidSizingInputException(
            f"{ERR_SZ_INVALID_CONFIG}: total_capital must be strictly positive, got {total_capital}"
        )
    return float(total_capital)


def _validate_haircut(haircut: float) -> float:
    """Validate circuit breaker continuous haircut parameter kappa_t."""
    # Functional Purpose: Verify circuit breaker multiplier bounds kappa_t in [0.0, 1.0].
    # Explicit Dependency Tracking: haircut scalar.
    # Structural Relationship: Passed from CircuitBreakerOverlayEngine to regularizer.
    # Defensive Invariant: 0.0 <= haircut <= 1.0 and math.isfinite.
    if not (isinstance(haircut, (int, float)) and not isinstance(haircut, bool)):
        raise InvalidSizingInputException(
            f"{ERR_SZ_INVALID_CONFIG}: haircut must be numeric, got {type(haircut).__name__}"
        )
    if not math.isfinite(haircut):
        raise DegenerateSizingException(
            f"{ERR_SZ_NON_FINITE}: haircut must be a finite float, got {haircut}"
        )
    if not (0.0 <= haircut <= 1.0):
        raise InvalidSizingInputException(
            f"{ERR_SZ_INVALID_CONFIG}: haircut must be in [0.0, 1.0], got {haircut}"
        )
    return float(haircut)


class UncertaintyShrunkKellyUtility:
    """Uncertainty-Shrunk Kelly Utility Engine.

    Purpose:
        Evaluates quadratic Kelly portfolio utility dynamically penalized by epistemic
        model disagreement variance (sigma2_epistemic), preventing overbetting cliffs
        under parameter estimation uncertainty (codified under Rule 4.3).

    Mathematical Foundation:
        Shrunk Expected Return Vector:
            mu_shrunk = mu - lambda_shrink * sigma2_epistemic

        Utility Functional:
            U_Kelly(nu) = nu^T mu_shrunk - (gamma_risk / (2 * W_t)) * nu^T Sigma_aleatoric * nu

        Analytical Gradient:
            nabla U_Kelly(nu) = mu_shrunk - (gamma_risk / W_t) * Sigma_aleatoric * nu

        Analytical Hessian:
            nabla^2 U_Kelly(nu) = -(gamma_risk / W_t) * Sigma_aleatoric  (<= 0 everywhere)

    Invariants Enforced:
        - INV-TR-004: Strict Global Concavity (nabla^2 U <= 0).
        - INV-TR-005: Non-finite input protection (NaN/Inf raises DegenerateSizingException).
        - INV-TR-006: Sub-0.02ms hot-path execution latency SLA.
        - Rule 4.3: Zero unconstrained Kelly betting; dynamic epistemic uncertainty shrinkage.
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        """Initialize utility engine with optional institutional SizingConfig."""
        self.config: SizingConfig = config if config is not None else _DEFAULT_SIZING_CONFIG
        self._lambda: float = float(self.config.epistemic_shrinkage_lambda)
        self._gamma: float = float(self.config.risk_aversion_gamma)

    def compute_shrunk_returns(
        self,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        validate: bool = True,
    ) -> np.ndarray:
        """Compute sign-preserving epistemic-uncertainty-shrunk expected returns vector.

        Formula:
            mu_shrunk = sign(mu) * max(0.0, abs(mu) - lambda_shrink * sigma2_epistemic)

        Args:
            mu: Expected return vector (N,) from RD-DMA.
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            validate: If True, executes defensive input validation.

        Returns:
            Shrunk expected return vector (N,) as float64.
        """
        # Functional Purpose: Penalize assets with high model disagreement, shrinking toward cash (0.0).
        # Explicit Dependency Tracking: mu, sigma2_epistemic, self.config.epistemic_shrinkage_lambda.
        # Structural Relationship: Feeds linear term in Kelly quadratic utility and gradient.
        # Defensive Invariant: Non-negative epistemic variances, dimensional match, sign preservation.
        if validate:
            mu_clean = _validate_1d_array("mu", mu)
            n_assets = mu_clean.shape[0]
            sig2_clean = _validate_1d_array("sigma2_epistemic", sigma2_epistemic, n_assets)
            if float(np.min(sig2_clean)) < -1e-12:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: sigma2_epistemic must be non-negative (>= 0.0)"
                )
        else:
            mu_clean = mu
            sig2_clean = sigma2_epistemic

        shrinkage_magnitude = np.maximum(
            0.0,
            np.abs(mu_clean) - self.config.epistemic_shrinkage_lambda * sig2_clean,
        )
        return np.asarray(
            np.sign(mu_clean) * shrinkage_magnitude,
            dtype=np.float64,
        )

    def evaluate(
        self,
        nu: np.ndarray,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        cov_aleatoric: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> float:
        """Evaluate Uncertainty-Shrunk Kelly quadratic utility scalar U_Kelly(nu).

        Args:
            nu: Target dollar allocation vector (N,).
            mu: Expected return vector (N,).
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            cov_aleatoric: Aleatoric process covariance matrix (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Scalar utility value as float.
        """
        # Functional Purpose: Evaluate quadratic Kelly utility penalized by epistemic variance.
        # Explicit Dependency Tracking: nu, mu, sigma2_epistemic, cov_aleatoric, total_capital.
        # Structural Relationship: Evaluates linear-quadratic portion of execution sizing objective.
        # Defensive Invariant: INV-TR-004 concavity, INV-TR-005 non-finite data rejection.
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            mu_clean = _validate_1d_array("mu", mu, n_assets)
            sig2_clean = _validate_1d_array("sigma2_epistemic", sigma2_epistemic, n_assets)
            if float(np.min(sig2_clean)) < -1e-12:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: sigma2_epistemic must be non-negative (>= 0.0)"
                )
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric, n_assets)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            mu_clean = mu
            sig2_clean = sigma2_epistemic
            cov_clean = cov_aleatoric
            w_clean = total_capital

        shrunk_mu = np.sign(mu_clean) * np.maximum(
            0.0, np.abs(mu_clean) - self._lambda * sig2_clean
        )
        linear_term = float(np.dot(nu_clean, shrunk_mu))
        cov_nu = cov_clean @ nu_clean
        risk_penalty = 0.5 * (self._gamma / w_clean) * float(np.dot(nu_clean, cov_nu))
        return linear_term - risk_penalty

    def evaluate_utility_and_gradient(
        self,
        nu: np.ndarray,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        cov_aleatoric: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> tuple[float, np.ndarray]:
        """Evaluate Kelly utility scalar and gradient vector simultaneously in a single pass.

        Args:
            nu: Target dollar allocation vector (N,).
            mu: Expected return vector (N,).
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            cov_aleatoric: Aleatoric process covariance matrix (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Tuple of (scalar utility float, analytical gradient array).
        """
        # Functional Purpose: Single-pass joint evaluation of utility and gradient for line search.
        # Explicit Dependency Tracking: nu, mu, sigma2_epistemic, cov_aleatoric, total_capital.
        # Structural Relationship: Ingested by Armijo backtracking and Newton step solvers.
        # Defensive Invariant: INV-TR-004 concavity, INV-TR-005 non-finite data rejection.
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            mu_clean = _validate_1d_array("mu", mu, n_assets)
            sig2_clean = _validate_1d_array("sigma2_epistemic", sigma2_epistemic, n_assets)
            if float(np.min(sig2_clean)) < -1e-12:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: sigma2_epistemic must be non-negative (>= 0.0)"
                )
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric, n_assets)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            mu_clean = mu
            sig2_clean = sigma2_epistemic
            cov_clean = cov_aleatoric
            w_clean = total_capital

        shrunk_mu = np.sign(mu_clean) * np.maximum(
            0.0, np.abs(mu_clean) - self._lambda * sig2_clean
        )
        cov_nu = cov_clean @ nu_clean
        gamma_w = self._gamma / w_clean
        scaled_cov_nu = gamma_w * cov_nu
        u_val = float(np.dot(nu_clean, shrunk_mu)) - 0.5 * float(np.dot(nu_clean, scaled_cov_nu))
        grad = np.asarray(shrunk_mu - scaled_cov_nu, dtype=np.float64)
        return u_val, grad

    def gradient(
        self,
        nu: np.ndarray,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        cov_aleatoric: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate analytical gradient vector nabla U_Kelly(nu).

        Formula:
            nabla U_Kelly = mu_shrunk - (gamma_risk / W_t) * Sigma_aleatoric * nu

        Args:
            nu: Target dollar allocation vector (N,).
            mu: Expected return vector (N,).
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            cov_aleatoric: Aleatoric process covariance matrix (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Analytical gradient vector (N,) as float64.
        """
        # Functional Purpose: Evaluate closed-form gradient for Kelly utility term.
        # Explicit Dependency Tracking: nu, mu, sigma2_epistemic, cov_aleatoric, total_capital.
        # Structural Relationship: Ingested by projected gradient and Newton-Raphson solvers.
        # Defensive Invariant: Dimensions align to N; returns finite float64 array.
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            mu_clean = _validate_1d_array("mu", mu, n_assets)
            sig2_clean = _validate_1d_array("sigma2_epistemic", sigma2_epistemic, n_assets)
            if float(np.min(sig2_clean)) < -1e-12:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: sigma2_epistemic must be non-negative (>= 0.0)"
                )
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric, n_assets)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            mu_clean = mu
            sig2_clean = sigma2_epistemic
            cov_clean = cov_aleatoric
            w_clean = total_capital

        shrunk_mu = np.sign(mu_clean) * np.maximum(
            0.0, np.abs(mu_clean) - self._lambda * sig2_clean
        )
        cov_nu = cov_clean @ nu_clean
        return np.asarray(
            shrunk_mu - (self._gamma / w_clean) * cov_nu,
            dtype=np.float64,
        )

    def hessian(
        self,
        cov_aleatoric: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate analytical Hessian matrix nabla^2 U_Kelly(nu).

        Formula:
            nabla^2 U_Kelly = -(gamma_risk / W_t) * Sigma_aleatoric

        Args:
            cov_aleatoric: Aleatoric process covariance matrix (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Strictly negative-semi-definite Hessian matrix (N, N) as float64.
        """
        # Functional Purpose: Evaluate analytical Hessian matrix for Newton-Raphson steps.
        # Explicit Dependency Tracking: cov_aleatoric, total_capital, config.
        # Structural Relationship: Ingested by Hessian assembler in UnifiedConvexObjective.
        # Defensive Invariant: INV-TR-004: Negative semi-definite (all eigenvalues <= 0.0).
        if validate:
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric)
            w_clean = _validate_positive_capital(total_capital)
        else:
            cov_clean = cov_aleatoric
            w_clean = total_capital

        gamma_w = self._gamma / w_clean
        return np.asarray(
            -gamma_w * cov_clean,
            dtype=np.float64,
        )


class PseudoHuberImpactPenalty:
    """3/2-Power Generalized Pseudo-Huber Market Impact Penalty Engine.

    Purpose:
        Models non-linear execution friction according to the universal Square-Root Law
        of price impact (Psi'_{3/2}(u) ~ sqrt(u)) coupled with permanent cross-asset impact
        (Lambda_cross), while maintaining strict global convexity and C^infinity smoothness.

    Mathematical Foundation:
        Total Impact Cost Functional:
            C_Impact(nu) = (1 / (2 * W_t)) * nu^T Lambda_cross * nu
                         + eta * sum_{i=1}^N sigma_i * [(nu_i^2 + delta^2)^{3/4} - delta^{1.5}]

        Analytical Gradient:
            nabla C_Impact(nu) = (1 / W_t) * Lambda_cross * nu
                               + (3/2) * eta * sigma * nu * (nu^2 + delta^2)^{-1/4}

        Analytical Hessian Diagonal:
            d^2 C / d nu_i^2 = (Lambda_ii / W_t)
                             + (3/2) * eta * sigma_i * (nu_i^2 + delta^2)^{-5/4} * (0.5 * nu_i^2 + delta^2) > 0

        Full Analytical Hessian Matrix:
            nabla^2 C_Impact(nu) = (1 / W_t) * Lambda_cross + diag(H_diag_transient) > 0

    Invariants Enforced:
        - Strict Global Convexity: nabla^2 C_Impact(nu) > 0 everywhere for all nu in R^N.
        - Zero Gradient at Rest: C_Impact(0) = 0 and nabla C_Impact(0) = 0.
        - INV-TR-005: Non-finite parameter protection (NaN/Inf raises DegenerateSizingException).
        - INV-TR-006: Sub-0.02ms hot-path execution latency SLA.
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        """Initialize Pseudo-Huber penalty engine with optional SizingConfig."""
        self.config: SizingConfig = config if config is not None else _DEFAULT_SIZING_CONFIG
        self._delta_sq: float = self.config.impact_delta**2
        self._delta_15: float = self.config.impact_delta**1.5
        self._one_and_half_eta: float = 1.5 * self.config.impact_penalty_eta

    def evaluate(
        self,
        nu: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> float:
        """Evaluate total execution friction penalty scalar C_Impact(nu).

        Args:
            nu: Dollar allocation vector (N,).
            asset_vols: Asset volatility vector sigma (N,).
            cross_impact: Symmetric positive-definite cross-impact matrix Lambda_cross (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Scalar execution friction penalty as float (>= 0.0).
        """
        # Functional Purpose: Evaluate combined permanent and 3/2-power transient impact cost.
        # Explicit Dependency Tracking: nu, asset_vols, cross_impact, total_capital, config.
        # Structural Relationship: Deducted from Kelly utility in execution sizing objective.
        # Defensive Invariant: Returns non-negative finite cost; C(0) = 0.
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            vols_clean = asset_vols
            cross_clean = cross_impact
            w_clean = total_capital

        # 1. Permanent cross-impact cost: 0.5 * nu^T * Lambda_cross * nu / W_t
        cross_nu = cross_clean @ nu_clean
        perm_cost = 0.5 * (1.0 / w_clean) * float(np.dot(nu_clean, cross_nu))

        # 2. 3/2-power transient pseudo-Huber cost: eta * sum(sigma_i * (inner^{0.75} - delta^{1.5}))
        nu_sq = nu_clean * nu_clean
        inner = nu_sq + self._delta_sq
        q = np.sqrt(np.sqrt(inner))  # inner^0.25 via double square root for hardware SIMD speed
        psi = (q * q * q) - self._delta_15
        trans_cost = self.config.impact_penalty_eta * float(np.sum(vols_clean * psi))

        return perm_cost + trans_cost

    def gradient(
        self,
        nu: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate analytical gradient vector nabla C_Impact(nu).

        Formula:
            nabla C_Impact = (1 / W_t) * Lambda_cross * nu + 1.5 * eta * sigma * nu * (nu^2 + delta^2)^{-1/4}

        Args:
            nu: Dollar allocation vector (N,).
            asset_vols: Asset volatility vector sigma (N,).
            cross_impact: Symmetric cross-impact matrix Lambda_cross (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Analytical gradient vector (N,) as float64.
        """
        # Functional Purpose: Evaluate exact gradient of market impact penalty.
        # Explicit Dependency Tracking: nu, asset_vols, cross_impact, total_capital, config.
        # Structural Relationship: Subtracted from objective gradient in optimization solver.
        # Defensive Invariant: nabla C(0) == 0; returns finite float64 array of shape (N,).
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            vols_clean = asset_vols
            cross_clean = cross_impact
            w_clean = total_capital

        cross_nu = cross_clean @ nu_clean
        perm_grad = (1.0 / w_clean) * cross_nu

        nu_sq = nu_clean * nu_clean
        inner = nu_sq + self._delta_sq
        q = np.sqrt(np.sqrt(inner))  # inner^0.25
        inv_q = 1.0 / q
        trans_grad = self._one_and_half_eta * vols_clean * nu_clean * inv_q

        return np.asarray(perm_grad + trans_grad, dtype=np.float64)

    def hessian_diagonal(
        self,
        nu: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate diagonal vector of analytical Hessian matrix diag(nabla^2 C_Impact(nu)).

        Formula:
            d^2 C / d nu_i^2 = (Lambda_ii / W_t) + 1.5 * eta * sigma_i * (inner)^{-5/4} * (0.5 * nu_i^2 + delta^2)

        Args:
            nu: Dollar allocation vector (N,).
            asset_vols: Asset volatility vector sigma (N,).
            cross_impact: Symmetric cross-impact matrix Lambda_cross (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            1D array of diagonal Hessian elements (N,) as float64 (> 0.0).
        """
        # Functional Purpose: Evaluate diagonal Hessian entries for coordinate updates / preconditioners.
        # Explicit Dependency Tracking: nu, asset_vols, cross_impact, total_capital, config.
        # Structural Relationship: Ingested by diagonal scaling and quasi-Newton approximations.
        # Defensive Invariant: All diagonal entries strictly positive (> 0.0).
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            vols_clean = asset_vols
            cross_clean = cross_impact
            w_clean = total_capital

        perm_diag = np.diag(cross_clean) / w_clean

        nu_sq = nu_clean * nu_clean
        inner = nu_sq + self._delta_sq
        q = np.sqrt(np.sqrt(inner))  # inner^0.25
        # inner^{-5/4} = 1.0 / (inner * q)
        inv_inner_125 = 1.0 / (inner * q)
        num = 0.5 * nu_sq + self._delta_sq
        trans_diag = self._one_and_half_eta * vols_clean * inv_inner_125 * num

        return np.asarray(perm_diag + trans_diag, dtype=np.float64)

    def hessian(
        self,
        nu: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray,
        total_capital: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate full analytical Hessian matrix nabla^2 C_Impact(nu).

        Formula:
            nabla^2 C_Impact = (1 / W_t) * Lambda_cross + diag(transient_hessian_diagonal)

        Args:
            nu: Dollar allocation vector (N,).
            asset_vols: Asset volatility vector sigma (N,).
            cross_impact: Symmetric cross-impact matrix Lambda_cross (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Strictly positive-definite Hessian matrix (N, N) as float64.
        """
        # Functional Purpose: Evaluate full analytical Hessian matrix for Newton-Raphson iterations.
        # Explicit Dependency Tracking: nu, asset_vols, cross_impact, total_capital, config.
        # Structural Relationship: Ingested by Newton step solver in UnifiedConvexExecutionSizer.
        # Defensive Invariant: Strictly positive definite: nabla^2 C > 0 everywhere.
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            vols_clean = asset_vols
            cross_clean = cross_impact
            w_clean = total_capital

        h = cross_clean / w_clean

        nu_sq = nu_clean * nu_clean
        inner = nu_sq + self._delta_sq
        q = np.sqrt(np.sqrt(inner))  # inner^0.25
        inv_inner_125 = 1.0 / (inner * q)
        num = 0.5 * nu_sq + self._delta_sq
        trans_diag = self._one_and_half_eta * vols_clean * inv_inner_125 * num

        np.fill_diagonal(h, h.diagonal() + trans_diag)
        return np.asarray(h, dtype=np.float64)


class CircuitBreakerRegularizer:
    """Circuit Breaker Continuous Haircut Regularization Engine.

    Purpose:
        Enforces continuous epistemic contraction on portfolio allocations:
            R_Epistemic(nu) = (1 / (2 * max(kappa_t, kappa_floor) * W_t)) * ||nu||^2_2

        As macro/entropy shock Xi_t -> 1.0, continuous haircut kappa_t -> 0.0, driving
        the quadratic regularization penalty to +infinity (clamped by kappa_floor),
        smoothly and unconditionally pinning target allocations to zero.

    Mathematical Foundation:
        Let kappa_eff = max(kappa_t, kappa_floor).
        Weight scalar:
            w_reg = 1 / (kappa_eff * W_t)

        Regularizer Value:
            R_Epistemic(nu) = 0.5 * w_reg * ||nu||^2_2

        Analytical Gradient:
            nabla R_Epistemic(nu) = w_reg * nu

        Analytical Hessian:
            nabla^2 R_Epistemic(nu) = w_reg * I_N  (> 0 everywhere)

    Invariants Enforced:
        - Contraction to Zero: As kappa_t -> 0.0, penalty w_reg reaches maximum 1 / (kappa_floor * W_t).
        - Strictly Positive Definite: nabla^2 R_Epistemic = w_reg * I_N > 0.
        - INV-TR-005: Non-finite input protection.
        - INV-TR-006: Hot-path latency SLA.
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        """Initialize regularizer with optional institutional SizingConfig."""
        self.config: SizingConfig = config if config is not None else _DEFAULT_SIZING_CONFIG

    def effective_haircut(self, haircut: float) -> float:
        """Evaluate floored circuit breaker haircut multiplier max(kappa_t, kappa_floor)."""
        # Functional Purpose: Apply defensive lower bound floor on haircut to avoid division by zero.
        # Explicit Dependency Tracking: haircut, self.config.min_haircut_floor.
        # Structural Relationship: Used in regularization weight computation.
        # Defensive Invariant: Returns float in [min_haircut_floor, 1.0].
        clean_haircut = _validate_haircut(haircut)
        return max(clean_haircut, self.config.min_haircut_floor)

    def evaluate(
        self,
        nu: np.ndarray,
        haircut: float,
        total_capital: float,
        validate: bool = True,
    ) -> float:
        """Evaluate quadratic circuit breaker regularization penalty scalar R_Epistemic(nu).

        Args:
            nu: Dollar allocation vector (N,).
            haircut: Circuit breaker continuous haircut kappa_t in [0.0, 1.0].
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Scalar penalty value as float (>= 0.0).
        """
        # Functional Purpose: Evaluate quadratic penalty crushing allocations under distress.
        # Explicit Dependency Tracking: nu, haircut, total_capital, config.min_haircut_floor.
        # Structural Relationship: Deducted from Kelly utility in execution sizing objective.
        # Defensive Invariant: Returns non-negative finite float; R(0) = 0.
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            eff_k = self.effective_haircut(haircut)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            eff_k = max(haircut, self.config.min_haircut_floor)
            w_clean = total_capital

        inv_kw = 1.0 / (eff_k * w_clean)
        return 0.5 * inv_kw * float(np.dot(nu_clean, nu_clean))

    def gradient(
        self,
        nu: np.ndarray,
        haircut: float,
        total_capital: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate analytical gradient vector nabla R_Epistemic(nu).

        Formula:
            nabla R_Epistemic = (1 / (max(kappa_t, kappa_floor) * W_t)) * nu

        Args:
            nu: Dollar allocation vector (N,).
            haircut: Circuit breaker continuous haircut kappa_t in [0.0, 1.0].
            total_capital: Total portfolio equity W_t > 0.0.
            validate: If True, executes input validation.

        Returns:
            Analytical gradient vector (N,) as float64.
        """
        # Functional Purpose: Evaluate gradient of quadratic circuit breaker penalty.
        # Explicit Dependency Tracking: nu, haircut, total_capital, config.min_haircut_floor.
        # Structural Relationship: Subtracted from objective gradient in optimization solver.
        # Defensive Invariant: nabla R(0) == 0; returns finite float64 array of shape (N,).
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            eff_k = self.effective_haircut(haircut)
            w_clean = _validate_positive_capital(total_capital)
        else:
            nu_clean = nu
            eff_k = max(haircut, self.config.min_haircut_floor)
            w_clean = total_capital

        inv_kw = 1.0 / (eff_k * w_clean)
        return np.asarray(inv_kw * nu_clean, dtype=np.float64)

    def hessian_diagonal(
        self,
        haircut: float,
        total_capital: float,
        dim: int,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate diagonal vector of analytical Hessian matrix diag(nabla^2 R_Epistemic).

        Formula:
            d^2 R / d nu_i^2 = 1 / (max(kappa_t, kappa_floor) * W_t) for i = 1, ..., N

        Args:
            haircut: Circuit breaker continuous haircut kappa_t in [0.0, 1.0].
            total_capital: Total portfolio equity W_t > 0.0.
            dim: Asset universe dimension N >= 1.
            validate: If True, executes input validation.

        Returns:
            1D array of diagonal Hessian elements (dim,) as float64 (> 0.0).
        """
        # Functional Purpose: Evaluate diagonal Hessian entries for regularizer.
        # Explicit Dependency Tracking: haircut, total_capital, dim.
        # Structural Relationship: Ingested by diagonal preconditioning in Newton solver.
        # Defensive Invariant: All entries strictly positive (> 0.0).
        if validate:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: dim must be integer >= 1, got {dim}"
                )
            eff_k = self.effective_haircut(haircut)
            w_clean = _validate_positive_capital(total_capital)
        else:
            eff_k = max(haircut, self.config.min_haircut_floor)
            w_clean = total_capital

        inv_kw = 1.0 / (eff_k * w_clean)
        return np.full(dim, inv_kw, dtype=np.float64)

    def hessian(
        self,
        haircut: float,
        total_capital: float,
        dim: int,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate full analytical Hessian matrix nabla^2 R_Epistemic.

        Formula:
            nabla^2 R_Epistemic = (1 / (max(kappa_t, kappa_floor) * W_t)) * I_N

        Args:
            haircut: Circuit breaker continuous haircut kappa_t in [0.0, 1.0].
            total_capital: Total portfolio equity W_t > 0.0.
            dim: Asset universe dimension N >= 1.
            validate: If True, executes input validation.

        Returns:
            Strictly positive-definite diagonal Hessian matrix (dim, dim) as float64.
        """
        # Functional Purpose: Evaluate full analytical Hessian matrix for regularizer.
        # Explicit Dependency Tracking: haircut, total_capital, dim.
        # Structural Relationship: Added to objective Hessian in Newton step solver.
        # Defensive Invariant: Strictly positive definite; scaled identity matrix.
        if validate:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: dim must be integer >= 1, got {dim}"
                )
            eff_k = self.effective_haircut(haircut)
            w_clean = _validate_positive_capital(total_capital)
        else:
            eff_k = max(haircut, self.config.min_haircut_floor)
            w_clean = total_capital

        inv_kw = 1.0 / (eff_k * w_clean)
        return np.eye(dim, dtype=np.float64) * inv_kw


class UnifiedConvexObjective:
    """Unified Strictly Concave Execution Sizing Objective Evaluator.

    Purpose:
        Combines Uncertainty-Shrunk Kelly Utility, Generalized Pseudo-Huber Market Impact,
        and Circuit Breaker Regularization into a unified strictly concave allocation functional:
            L(nu) = U_Kelly(nu) - C_Impact(nu) - R_Epistemic(nu)

        Analytical Gradient:
            nabla L(nu) = nabla U_Kelly(nu) - nabla C_Impact(nu) - nabla R_Epistemic(nu)

        Analytical Hessian:
            nabla^2 L(nu) = nabla^2 U_Kelly(nu) - nabla^2 C_Impact(nu) - nabla^2 R_Epistemic(nu)

    Concavity Guarantee (INV-TR-004):
        - nabla^2 U_Kelly = -(gamma / W_t) * Sigma <= 0
        - -nabla^2 C_Impact = -(1 / W_t) * Lambda_cross - diag(H_transient) < 0
        - -nabla^2 R_Epistemic = -(1 / (kappa_eff * W_t)) * I_N < 0
        Therefore:
            nabla^2 L(nu) << 0 everywhere in R^N.
        All eigenvalues of nabla^2 L(nu) are strictly negative (lambda_max <= -1 / (kappa_eff * W_t) < 0).
        This mathematically guarantees that the unconstrained and convex-constrained maximizer
        nu* is unique and globally optimal without saddle points or local traps.

    Invariants Enforced:
        - INV-TR-004: Strict Global Concavity (all eigenvalues strictly negative).
        - INV-TR-005: Non-finite input protection (NaN/Inf raises DegenerateSizingException).
        - INV-TR-006: Hot-path latency SLA <= 0.02ms for N = 10.
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        """Initialize unified objective evaluator with optional SizingConfig."""
        self.config: SizingConfig = config if config is not None else _DEFAULT_SIZING_CONFIG
        self.kelly: UncertaintyShrunkKellyUtility = UncertaintyShrunkKellyUtility(self.config)
        self.impact: PseudoHuberImpactPenalty = PseudoHuberImpactPenalty(self.config)
        self.regularizer: CircuitBreakerRegularizer = CircuitBreakerRegularizer(self.config)

    def evaluate(
        self,
        nu: np.ndarray,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        cov_aleatoric: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray,
        total_capital: float,
        haircut: float,
        validate: bool = True,
    ) -> float:
        """Evaluate total execution sizing objective scalar L(nu).

        Formula:
            L(nu) = U_Kelly(nu) - C_Impact(nu) - R_Epistemic(nu)

        Args:
            nu: Dollar allocation vector (N,).
            mu: Expected return vector (N,).
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            cov_aleatoric: Aleatoric process covariance matrix (N, N).
            asset_vols: Asset volatility vector sigma (N,).
            cross_impact: Symmetric cross-impact matrix Lambda_cross (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            haircut: Circuit breaker continuous haircut kappa_t in [0.0, 1.0].
            validate: If True, executes input validation once.

        Returns:
            Scalar objective value as float.
        """
        # Functional Purpose: Compute total concave objective for line search and optimality checks.
        # Explicit Dependency Tracking: All input vectors, matrices, capital, haircut, components.
        # Structural Relationship: Master objective function maximized by UnifiedConvexExecutionSizer.
        # Defensive Invariant: INV-TR-004 concavity, INV-TR-005 non-finite data rejection.
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            mu_clean = _validate_1d_array("mu", mu, n_assets)
            sig2_clean = _validate_1d_array("sigma2_epistemic", sigma2_epistemic, n_assets)
            if float(np.min(sig2_clean)) < -1e-12:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: sigma2_epistemic must be non-negative (>= 0.0)"
                )
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric, n_assets)
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            w_clean = _validate_positive_capital(total_capital)
            k_clean = _validate_haircut(haircut)
        else:
            nu_clean = nu
            mu_clean = mu
            sig2_clean = sigma2_epistemic
            cov_clean = cov_aleatoric
            vols_clean = asset_vols
            cross_clean = cross_impact
            w_clean = total_capital
            k_clean = haircut

        u_k = self.kelly.evaluate(
            nu_clean, mu_clean, sig2_clean, cov_clean, w_clean, validate=False
        )
        c_i = self.impact.evaluate(nu_clean, vols_clean, cross_clean, w_clean, validate=False)
        r_e = self.regularizer.evaluate(nu_clean, k_clean, w_clean, validate=False)
        return u_k - c_i - r_e

    def gradient(
        self,
        nu: np.ndarray,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        cov_aleatoric: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray,
        total_capital: float,
        haircut: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate total analytical gradient vector nabla L(nu).

        Formula:
            nabla L(nu) = nabla U_Kelly(nu) - nabla C_Impact(nu) - nabla R_Epistemic(nu)

        Args:
            nu: Dollar allocation vector (N,).
            mu: Expected return vector (N,).
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            cov_aleatoric: Aleatoric process covariance matrix (N, N).
            asset_vols: Asset volatility vector sigma (N,).
            cross_impact: Symmetric cross-impact matrix Lambda_cross (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            haircut: Circuit breaker continuous haircut kappa_t in [0.0, 1.0].
            validate: If True, executes input validation once.

        Returns:
            Analytical gradient vector (N,) as float64.
        """
        # Functional Purpose: Evaluate total gradient vector for ascent search directions.
        # Explicit Dependency Tracking: All inputs, components.
        # Structural Relationship: Ingested by projected gradient and Newton solver steps.
        # Defensive Invariant: Returns finite float64 array of shape (N,).
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            mu_clean = _validate_1d_array("mu", mu, n_assets)
            sig2_clean = _validate_1d_array("sigma2_epistemic", sigma2_epistemic, n_assets)
            if float(np.min(sig2_clean)) < -1e-12:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: sigma2_epistemic must be non-negative (>= 0.0)"
                )
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric, n_assets)
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            w_clean = _validate_positive_capital(total_capital)
            k_clean = _validate_haircut(haircut)
        else:
            nu_clean = nu
            mu_clean = mu
            sig2_clean = sigma2_epistemic
            cov_clean = cov_aleatoric
            vols_clean = asset_vols
            cross_clean = cross_impact
            w_clean = total_capital
            k_clean = haircut

        g_k = self.kelly.gradient(
            nu_clean, mu_clean, sig2_clean, cov_clean, w_clean, validate=False
        )
        g_i = self.impact.gradient(nu_clean, vols_clean, cross_clean, w_clean, validate=False)
        g_r = self.regularizer.gradient(nu_clean, k_clean, w_clean, validate=False)
        return np.asarray(g_k - g_i - g_r, dtype=np.float64)

    def hessian(
        self,
        nu: np.ndarray,
        cov_aleatoric: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray,
        total_capital: float,
        haircut: float,
        validate: bool = True,
    ) -> np.ndarray:
        """Evaluate total analytical Hessian matrix nabla^2 L(nu).

        Formula:
            nabla^2 L(nu) = nabla^2 U_Kelly(nu) - nabla^2 C_Impact(nu) - nabla^2 R_Epistemic(nu)

        Args:
            nu: Dollar allocation vector (N,).
            cov_aleatoric: Aleatoric process covariance matrix (N, N).
            asset_vols: Asset volatility vector sigma (N,).
            cross_impact: Symmetric cross-impact matrix Lambda_cross (N, N).
            total_capital: Total portfolio equity W_t > 0.0.
            haircut: Circuit breaker continuous haircut kappa_t in [0.0, 1.0].
            validate: If True, executes input validation once.

        Returns:
            Strictly negative-definite Hessian matrix (N, N) as float64.
        """
        # Functional Purpose: Evaluate total analytical Hessian matrix for Newton iterations.
        # Explicit Dependency Tracking: All inputs, components.
        # Structural Relationship: Ingested by Newton step solver; verified for INV-TR-004 concavity.
        # Defensive Invariant: INV-TR-004: All eigenvalues strictly negative (< 0.0).
        if validate:
            nu_clean = _validate_1d_array("nu", nu)
            n_assets = nu_clean.shape[0]
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric, n_assets)
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            w_clean = _validate_positive_capital(total_capital)
            k_clean = _validate_haircut(haircut)
        else:
            nu_clean = nu
            cov_clean = cov_aleatoric
            vols_clean = asset_vols
            cross_clean = cross_impact
            w_clean = total_capital
            k_clean = haircut

        h_k = self.kelly.hessian(cov_clean, w_clean, validate=False)
        h_i = self.impact.hessian(nu_clean, vols_clean, cross_clean, w_clean, validate=False)
        dim = nu_clean.shape[0]
        h_r = self.regularizer.hessian(k_clean, w_clean, dim, validate=False)
        return np.asarray(h_k - h_i - h_r, dtype=np.float64)

    @staticmethod
    def check_strict_concavity(hessian: np.ndarray, tol: float = 1e-12) -> bool:
        """Verify invariant INV-TR-004: all eigenvalues of Hessian are strictly negative.

        Args:
            hessian: Total objective Hessian matrix (N, N).
            tol: Maximum allowed eigenvalue bound (< -tol).

        Returns:
            True if all eigenvalues are strictly negative.

        Raises:
            DegenerateSizingException: If any eigenvalue >= -tol, violating INV-TR-004.
        """
        # Functional Purpose: Verify INV-TR-004 strict global concavity via spectral decomposition.
        # Explicit Dependency Tracking: hessian, tol.
        # Structural Relationship: Gatekeeper for convex optimization guarantees.
        # Defensive Invariant: INV-TR-004: lambda_max(Hessian) < -tol.
        eigs = np.linalg.eigvalsh(hessian)
        max_eig = float(np.max(eigs))
        if max_eig >= -tol:
            raise DegenerateSizingException(
                f"{ERR_SZ_CONCAVITY_VIOLATED}: INV-TR-004 violated: maximum eigenvalue "
                f"of total Hessian is {max_eig} >= {-tol} (must be strictly negative)."
            )
        return True


def evaluate_total_objective(
    nu: np.ndarray,
    mu: np.ndarray,
    sigma2_epistemic: np.ndarray,
    cov_aleatoric: np.ndarray,
    asset_vols: np.ndarray,
    cross_impact: np.ndarray,
    total_capital: float,
    haircut: float,
    config: SizingConfig | None = None,
    validate: bool = True,
) -> float:
    """Module-level helper to evaluate total execution sizing objective L(nu)."""
    # Functional Purpose: Functional facade for objective evaluation.
    # Explicit Dependency Tracking: All sizing inputs and optional SizingConfig.
    # Structural Relationship: Directly called by external solvers and benchmark tests.
    # Defensive Invariant: INV-TR-004, INV-TR-005.
    evaluator = UnifiedConvexObjective(config)
    return evaluator.evaluate(
        nu=nu,
        mu=mu,
        sigma2_epistemic=sigma2_epistemic,
        cov_aleatoric=cov_aleatoric,
        asset_vols=asset_vols,
        cross_impact=cross_impact,
        total_capital=total_capital,
        haircut=haircut,
        validate=validate,
    )


def gradient_total_objective(
    nu: np.ndarray,
    mu: np.ndarray,
    sigma2_epistemic: np.ndarray,
    cov_aleatoric: np.ndarray,
    asset_vols: np.ndarray,
    cross_impact: np.ndarray,
    total_capital: float,
    haircut: float,
    config: SizingConfig | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Module-level helper to evaluate total analytical gradient nabla L(nu)."""
    # Functional Purpose: Functional facade for gradient evaluation.
    # Explicit Dependency Tracking: All sizing inputs and optional SizingConfig.
    # Structural Relationship: Directly called by external solvers and benchmark tests.
    # Defensive Invariant: Dimensions align to N; finite float64 array.
    evaluator = UnifiedConvexObjective(config)
    return evaluator.gradient(
        nu=nu,
        mu=mu,
        sigma2_epistemic=sigma2_epistemic,
        cov_aleatoric=cov_aleatoric,
        asset_vols=asset_vols,
        cross_impact=cross_impact,
        total_capital=total_capital,
        haircut=haircut,
        validate=validate,
    )


def hessian_total_objective(
    nu: np.ndarray,
    cov_aleatoric: np.ndarray,
    asset_vols: np.ndarray,
    cross_impact: np.ndarray,
    total_capital: float,
    haircut: float,
    config: SizingConfig | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Module-level helper to evaluate total analytical Hessian nabla^2 L(nu)."""
    # Functional Purpose: Functional facade for Hessian evaluation.
    # Explicit Dependency Tracking: All sizing inputs and optional SizingConfig.
    # Structural Relationship: Directly called by external solvers and benchmark tests.
    # Defensive Invariant: INV-TR-004 strict negative definiteness.
    evaluator = UnifiedConvexObjective(config)
    return evaluator.hessian(
        nu=nu,
        cov_aleatoric=cov_aleatoric,
        asset_vols=asset_vols,
        cross_impact=cross_impact,
        total_capital=total_capital,
        haircut=haircut,
        validate=validate,
    )


def project_two_l1_constraints(
    y: np.ndarray,
    c: np.ndarray,
    b1: float,
    b2: float,
    validate: bool = True,
    _force_dykstra: bool = False,
) -> np.ndarray:
    """Exact O(N log N) Euclidean projection onto intersection of L1 and weighted L1 balls.

    Solves:
        argmin_{nu in R^N} 0.5 * ||nu - y||_2^2
        subject to:
            ||nu||_1 <= b1                              (Gross Leverage Bound)
            sum_{i=1}^N c_i * |nu_i| <= b2              (Weighted CVaR Drawdown Budget)

    Mathematical Foundation:
        By coordinate sign symmetry, sign(nu_i*) = sign(y_i) for all i in {1, ..., N}.
        Let a = |y| >= 0. The problem decouples into non-negative projection:
            argmin_{x >= 0} 0.5 * ||x - a||_2^2
            subject to:
                sum_{i=1}^N x_i <= b1
                sum_{i=1}^N c_i * x_i <= b2

        1. Unconstrained: If sum(a) <= b1 and sum(c * a) <= b2, return y.
        2. Single constraint 1 (L1 leverage ball):
           Find lambda_1 > 0 such that sum(max(0, a_i - lambda_1)) = b1 via O(N log N) sorting.
           If sum(c * x1) <= b2, return sign(y) * x1.
        3. Single constraint 2 (Weighted L1 CVaR ball):
           Find lambda_2 > 0 such that sum(c_i * max(0, a_i - lambda_2 * c_i)) = b2 via sorting ratios a_i / c_i.
           If sum(x2) <= b1, return sign(y) * x2.
        4. Both constraints strictly active:
           KKT conditions require (lambda_1, lambda_2) > (0, 0) satisfying:
               f_1(lambda_1, lambda_2) = sum_{i in S} (a_i - lambda_1 - lambda_2 * c_i) = b1
               f_2(lambda_1, lambda_2) = sum_{i in S} c_i * (a_i - lambda_1 - lambda_2 * c_i) = b2
           where S = {i : a_i - lambda_1 - lambda_2 * c_i > 0}.
           Solved via Semismooth 2D Newton iteration on (lambda_1, lambda_2), with Dykstra
           alternating projections as an adversarial fallback.
        5. Hard Invariant Defense (INV-TR-005):
           Zero-leakage scaling ensures ||nu*||_1 <= b1 and c^T |nu*| <= b2 to exact float bounds.
    """
    # Functional Purpose: Project unconstrained or ascent point onto feasible convex set K.
    # Explicit Dependency Tracking: y, c, b1, b2, validation helpers.
    # Structural Relationship: Inner projection step of Barzilai-Borwein Spectral Projected Gradient.
    # Defensive Invariant: INV-TR-005: Output strictly satisfies ||nu||_1 <= b1 and c^T |nu| <= b2.
    if validate:
        y_clean = _validate_1d_array("y", y)
        n = y_clean.shape[0]
        c_clean = _validate_1d_array("c", c, n)
        if float(np.min(c_clean)) <= 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: c elements must be strictly positive (> 0.0)"
            )
        if not (isinstance(b1, (int, float)) and not isinstance(b1, bool)):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: b1 must be numeric, got {type(b1).__name__}"
            )
        if not math.isfinite(b1):
            raise DegenerateSizingException(
                f"{ERR_SZ_NON_FINITE}: b1 must be a finite float, got {b1}"
            )
        if b1 < 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: b1 must be a non-negative float, got {b1}"
            )
        if not (isinstance(b2, (int, float)) and not isinstance(b2, bool)):
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: b2 must be numeric, got {type(b2).__name__}"
            )
        if not math.isfinite(b2):
            raise DegenerateSizingException(
                f"{ERR_SZ_NON_FINITE}: b2 must be a finite float, got {b2}"
            )
        if b2 < 0.0:
            raise InvalidSizingInputException(
                f"{ERR_SZ_INVALID_CONFIG}: b2 must be a non-negative float, got {b2}"
            )
    else:
        y_clean = y
        n = y_clean.shape[0]
        c_clean = c

    b1_val = float(b1)
    b2_val = float(b2)

    # Degenerate bounds collapsed to zero
    if b1_val <= 0.0 or b2_val <= 0.0:
        return np.zeros(n, dtype=np.float64)

    a = np.abs(y_clean)
    sum_a = float(np.sum(a))
    sum_ca = float(np.dot(c_clean, a))

    # Step 0: Unconstrained feasibility
    if sum_a <= b1_val and sum_ca <= b2_val:
        return y_clean.copy()

    # Step 1: L1 leverage ball projection
    s = sorted(a.tolist(), reverse=True)
    cum = 0.0
    th1 = 0.0
    for j in range(n):
        cum += s[j]
        t = (cum - b1_val) / (j + 1)
        if s[j] - t > 0:
            th1 = t
    x1 = np.maximum(0.0, a - th1)
    if float(np.dot(c_clean, x1)) <= b2_val + 1e-12:
        res1 = np.copysign(x1, y_clean)
        res1[y_clean == 0.0] = 0.0
        cvar1 = float(np.dot(c_clean, np.abs(res1)))
        if cvar1 > b2_val:
            res1 *= b2_val / cvar1
        l1_1 = float(np.sum(np.abs(res1)))
        if l1_1 > b1_val:
            res1 *= b1_val / l1_1
        return np.ascontiguousarray(res1, dtype=np.float64)

    # Step 2: Weighted L1 CVaR drawdown projection
    r_pairs = sorted([(a[i] / c_clean[i], i) for i in range(n)], reverse=True)
    cum_cy = 0.0
    cum_c2 = 0.0
    lam2 = 0.0
    for k in range(n):
        idx_k = r_pairs[k][1]
        cum_cy += c_clean[idx_k] * a[idx_k]
        cum_c2 += c_clean[idx_k] * c_clean[idx_k]
        l_cand = (cum_cy - b2_val) / cum_c2
        if l_cand < r_pairs[k][0] and (k == n - 1 or l_cand >= r_pairs[k + 1][0]):
            lam2 = l_cand
            break
    x2 = np.maximum(0.0, a - lam2 * c_clean)
    if float(np.sum(x2)) <= b1_val + 1e-12:
        res2 = np.copysign(x2, y_clean)
        res2[y_clean == 0.0] = 0.0
        cvar2 = float(np.dot(c_clean, np.abs(res2)))
        if cvar2 > b2_val:
            res2 *= b2_val / cvar2
        l1_2 = float(np.sum(np.abs(res2)))
        if l1_2 > b1_val:
            res2 *= b1_val / l1_2
        return np.ascontiguousarray(res2, dtype=np.float64)

    # Step 3: Both constraints strictly active - 2D Semismooth Newton method
    lam1 = 0.0
    lam2_curr = 0.0
    solved = False
    x_opt = x2
    if not _force_dykstra:
        for _ in range(15):
            x = np.maximum(0.0, a - lam1 - lam2_curr * c_clean)
            mask = x > 0
            k_act = int(np.sum(mask))
            if k_act == 0:
                break
            a_m = a[mask]
            c_m = c_clean[mask]
            Sa = float(np.sum(a_m))
            Sc = float(np.sum(c_m))
            Sca = float(np.dot(c_m, a_m))
            Scc = float(np.dot(c_m, c_m))
            det = k_act * Scc - Sc * Sc
            if det < 1e-12:
                if float(np.ptp(c_clean)) < 1e-12:
                    # Genuinely parallel constraints across entire portfolio: b_eff = min(b1, b2 / mean(c))
                    mean_c = Sc / k_act
                    b_eff = min(b1_val, b2_val / mean_c)
                    cum_deg = 0.0
                    th_deg = 0.0
                    for j in range(n):
                        cum_deg += s[j]
                        t = (cum_deg - b_eff) / (j + 1)
                        if s[j] - t > 0:
                            th_deg = t
                    x_opt = np.maximum(0.0, a - th_deg)
                    solved = True
                    break
                else:
                    # Single-asset active set or localized tie on heterogeneous universe: fall through to Dykstra
                    break
            lam1_new = (Scc * (Sa - b1_val) - Sc * (Sca - b2_val)) / det
            lam2_new = (k_act * (Sca - b2_val) - Sc * (Sa - b1_val)) / det
            if lam1_new >= -1e-12 and lam2_new >= -1e-12:
                x_new = np.maximum(0.0, a - max(0.0, lam1_new) - max(0.0, lam2_new) * c_clean)
                if np.array_equal(x_new > 0, mask):
                    x_opt = x_new
                    solved = True
                    break
            lam1 = max(0.0, lam1_new)
            lam2_curr = max(0.0, lam2_new)

    # Step 4: Dykstra Alternating Projections adversarial fallback
    if not solved:
        x_d = a.copy()
        p_corr = np.zeros_like(x_d)
        q_corr = np.zeros_like(x_d)
        for _ in range(40):
            y1 = x_d + p_corr
            y1_pos = np.maximum(0.0, y1)
            if float(np.sum(y1_pos)) > b1_val:
                s1 = sorted(y1_pos.tolist(), reverse=True)
                cum1 = 0.0
                th1_d = 0.0
                for j in range(n):
                    cum1 += s1[j]
                    t = (cum1 - b1_val) / (j + 1)
                    if s1[j] - t > 0:
                        th1_d = t
                y_step = np.maximum(0.0, y1_pos - th1_d)
            else:
                y_step = y1_pos
            p_corr = y1 - y_step

            y2 = y_step + q_corr
            y2_pos = np.maximum(0.0, y2)
            if float(np.dot(c_clean, y2_pos)) > b2_val:
                r2 = sorted([(y2_pos[i] / c_clean[i], i) for i in range(n)], reverse=True)
                cum_cy2 = 0.0
                cum_c22 = 0.0
                lam2_d = 0.0
                for k in range(n):
                    idx_k = r2[k][1]
                    cum_cy2 += c_clean[idx_k] * y2_pos[idx_k]
                    cum_c22 += c_clean[idx_k] * c_clean[idx_k]
                    l_cand = (cum_cy2 - b2_val) / cum_c22
                    if l_cand < r2[k][0] and (k == n - 1 or l_cand >= r2[k + 1][0]):
                        lam2_d = l_cand
                        break
                x_step = np.maximum(0.0, y2_pos - lam2_d * c_clean)
            else:
                x_step = y2_pos
            q_corr = y2 - x_step

            if float(np.max(np.abs(x_step - x_d))) < 1e-7:
                x_opt = x_step
                break
            x_d = x_step
        else:
            x_opt = x_d

    res = np.copysign(x_opt, y_clean)
    res[y_clean == 0.0] = 0.0
    cvar_res = float(np.dot(c_clean, np.abs(res)))
    if cvar_res > b2_val:
        res *= b2_val / cvar_res
    l1_res = float(np.sum(np.abs(res)))
    if l1_res > b1_val:
        res *= b1_val / l1_res
    return np.ascontiguousarray(res, dtype=np.float64)


def discretize_lot_allocations(
    target_allocations: np.ndarray,
    lot_sizes: np.ndarray | None = None,
    random_seed: int | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Discretize continuous dollar allocations into microstructure contract lot sizes.

    Mathematical Foundation:
        For each asset i, given minimum contract lot size Delta nu_i > 0:
            lots_i = |nu_i*| / Delta nu_i
            base_lots_i = floor(lots_i)
            prob_ceil_i = lots_i - base_lots_i in [0, 1)

        If random_seed == -1:
            B_i = 1.0 if prob_ceil_i >= 0.5 else 0.0  (deterministic nearest integer lot)
        Else:
            B_i ~ Bernoulli(prob_ceil_i)

        nu_tilde_i = sign(nu_i*) * (base_lots_i + B_i) * Delta nu_i

    Property:
        E[nu_tilde_i] = nu_i*  (unbiased expectation preserving risk budget in expectation)
    """
    # Functional Purpose: Map continuous dollar sizing into executable discrete exchange lots.
    # Explicit Dependency Tracking: target_allocations, lot_sizes, random_seed.
    # Structural Relationship: Emitted in SizingDecision.discretized_allocations.
    # Defensive Invariant: Unbiased expectation E[nu_tilde] = nu*; matches shape and finiteness.
    if validate:
        target_clean = _validate_1d_array("target_allocations", target_allocations)
        n = target_clean.shape[0]
        if lot_sizes is not None:
            lots_clean = _validate_1d_array("lot_sizes", lot_sizes, n)
            if float(np.min(lots_clean)) <= 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: lot_sizes elements must be strictly positive (> 0.0)"
                )
        else:
            lots_clean = None
    else:
        target_clean = target_allocations
        lots_clean = lot_sizes

    if lots_clean is None:
        return target_clean.copy()

    n = target_clean.shape[0]
    # Filter non-positive lot sizes (if any passed unvalidated)
    valid_mask = lots_clean > 0.0
    if not np.all(valid_mask):
        discretized = target_clean.copy()
        for i in range(n):
            if lots_clean[i] > 0.0:
                ratio = abs(target_clean[i]) / lots_clean[i]
                b_lots = math.floor(ratio)
                p_ceil = ratio - b_lots
                if random_seed == -1:
                    b_int = 1.0 if p_ceil >= 0.5 else 0.0
                else:
                    rng = np.random.default_rng(random_seed)
                    b_int = 1.0 if rng.random() < p_ceil else 0.0
                sgn_scalar = (
                    1.0 if target_clean[i] > 0.0 else (-1.0 if target_clean[i] < 0.0 else 0.0)
                )
                discretized[i] = sgn_scalar * (b_lots + b_int) * lots_clean[i]
        return np.ascontiguousarray(discretized, dtype=np.float64)

    lots_ratio = np.abs(target_clean) / lots_clean
    base_lots = np.floor(lots_ratio)
    prob_ceil = lots_ratio - base_lots

    if random_seed == -1:
        B = np.where(prob_ceil >= 0.5, 1.0, 0.0)
    else:
        rng = np.random.default_rng(random_seed)
        B = (rng.random(n) < prob_ceil).astype(np.float64)

    sgn_arr = np.sign(target_clean)
    discretized = sgn_arr * (base_lots + B) * lots_clean
    discretized[target_clean == 0.0] = 0.0
    return np.ascontiguousarray(discretized, dtype=np.float64)


class UnifiedConvexExecutionSizer:
    """Institutional Unified Convex Execution Sizer & Allocation Calibration Engine.

    Purpose:
        Calibrates optimal dollar execution allocations nu* in R^N by solving the strictly
        concave optimization problem maximizing epistemic-shrunk Kelly utility penalized by
        3/2-power pseudo-Huber market impact and continuous circuit breaker regularization,
        subject to hard Expected Shortfall (CVaR) drawdown budgets and gross leverage caps (INV-TR-005).

    Mathematical Formulation:
        Maximize:
            L(nu) = U_Kelly(nu) - C_Impact(nu) - R_Epistemic(nu)
        Subject to:
            ||nu||_1 <= L_max * W_t                     (Gross Leverage Ceiling)
            c^T |nu| <= MDD_budget * W_t                (Rockafellar-Uryasev CVaR Budget)

    Algorithmic Guarantees:
        - Strict Global Concavity (INV-TR-004): nabla^2 L(nu) << 0 everywhere.
        - Hard Drawdown Budget & Leverage Bound (INV-TR-005): Constraints strictly respected.
        - Hot-Path Latency SLA (INV-TR-006): Sub-0.15ms execution time for N = 10 assets.
        - Zero Iterative Black-Box Solvers (Rule 4.3): Pure NumPy closed-form KKT / SPG solver.
        - Microstructural Lot Discretization: Preserves risk budget in expectation (E[nu_tilde] = nu*).
    """

    def __init__(self, config: SizingConfig | None = None) -> None:
        """Initialize unified execution sizer with optional institutional SizingConfig."""
        # Functional Purpose: Initialize component engines with institutional configuration.
        # Explicit Dependency Tracking: config, SizingConfig default fallback.
        # Structural Relationship: Encapsulates Kelly, Impact, Regularizer, and Unified Objective engines.
        # Defensive Invariant: Valid immutable SizingConfig.
        self.config: SizingConfig = config if config is not None else _DEFAULT_SIZING_CONFIG
        self.kelly: UncertaintyShrunkKellyUtility = UncertaintyShrunkKellyUtility(self.config)
        self.impact: PseudoHuberImpactPenalty = PseudoHuberImpactPenalty(self.config)
        self.regularizer: CircuitBreakerRegularizer = CircuitBreakerRegularizer(self.config)
        self.objective: UnifiedConvexObjective = UnifiedConvexObjective(self.config)

    def project(
        self,
        y: np.ndarray,
        c: np.ndarray,
        b1: float,
        b2: float,
        validate: bool = True,
        _force_dykstra: bool = False,
    ) -> np.ndarray:
        """Project candidate dollar vector onto intersection of leverage and CVaR balls."""
        # Functional Purpose: Project vector onto feasible set K using institutional parameters.
        # Explicit Dependency Tracking: y, c, b1, b2, project_two_l1_constraints.
        # Structural Relationship: Facade method for exact convex projection.
        # Defensive Invariant: INV-TR-005: Returned vector lies strictly inside K.
        return project_two_l1_constraints(
            y=y, c=c, b1=b1, b2=b2, validate=validate, _force_dykstra=_force_dykstra
        )

    def discretize(
        self,
        target_allocations: np.ndarray,
        lot_sizes: np.ndarray | None = None,
        random_seed: int | None = None,
        validate: bool = True,
    ) -> np.ndarray:
        """Discretize continuous dollar allocations into exchange-compliant lots."""
        # Functional Purpose: Map continuous dollar sizing into executable integer lots.
        # Explicit Dependency Tracking: target_allocations, lot_sizes, random_seed.
        # Structural Relationship: Facade method for microstructural randomized rounding.
        # Defensive Invariant: Unbiased expectation E[nu_tilde] = nu*.
        effective_lots = lot_sizes if lot_sizes is not None else self.config.lot_sizes
        return discretize_lot_allocations(
            target_allocations=target_allocations,
            lot_sizes=effective_lots,
            random_seed=random_seed,
            validate=validate,
        )

    def solve(
        self,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        cov_aleatoric: np.ndarray,
        asset_vols: np.ndarray,
        cross_impact: np.ndarray | None = None,
        circuit_breaker_haircut: float = 1.0,
        asset_cvars: np.ndarray | None = None,
        total_capital: float = 1_000_000.0,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        lot_sizes: np.ndarray | None = None,
        random_seed: int | None = None,
        validate: bool = True,
    ) -> SizingDecision:
        """Solve unified convex execution sizing problem with hard CVaR and leverage constraints.

        Args:
            mu: Expected return vector (N,).
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            cov_aleatoric: Aleatoric covariance matrix (N, N).
            asset_vols: Asset volatility vector (N,).
            cross_impact: Optional symmetric cross-impact matrix (N, N). Defaults to zero.
            circuit_breaker_haircut: Continuous haircut kappa_t in [0.0, 1.0].
            asset_cvars: Optional downside CVaR multiplier vector c in R^N. Defaults to 2.33 * sigma_i.
            total_capital: Total portfolio equity W_t > 0.0.
            max_iterations: Maximum iterations for projected gradient solver.
            tolerance: Infinity-norm convergence tolerance > 0.0.
            lot_sizes: Optional contract lot sizes for microstructure discretization.
            random_seed: Optional RNG seed (-1 for deterministic rounding).
            validate: If True, validates inputs defensively.

        Returns:
            Frozen SizingDecision containing optimal continuous and discretized allocations.
        """
        # Functional Purpose: Formulate and solve strictly concave execution sizing optimization.
        # Explicit Dependency Tracking: All input arrays, scalars, SizingConfig, and components.
        # Structural Relationship: Primary entrypoint consumed by order generator and execution router.
        # Defensive Invariant: INV-TR-004 concavity, INV-TR-005 hard bounds, INV-TR-006 latency SLA.
        if validate:
            mu_clean = _validate_1d_array("mu", mu)
            n_assets = mu_clean.shape[0]
            sig2_clean = _validate_1d_array("sigma2_epistemic", sigma2_epistemic, n_assets)
            if float(np.min(sig2_clean)) < -1e-12:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: sigma2_epistemic must be non-negative (>= 0.0)"
                )
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric, n_assets)
            vols_clean = _validate_1d_array("asset_vols", asset_vols, n_assets)
            if float(np.min(vols_clean)) < 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: asset_vols must be non-negative (>= 0.0)"
                )
            if cross_impact is not None:
                cross_clean = _validate_2d_matrix("cross_impact", cross_impact, n_assets)
            else:
                cross_clean = np.zeros((n_assets, n_assets), dtype=np.float64)

            w_clean = _validate_positive_capital(total_capital)
            k_clean = _validate_haircut(circuit_breaker_haircut)

            if asset_cvars is not None:
                cvars_clean = _validate_1d_array("asset_cvars", asset_cvars, n_assets)
                if float(np.min(cvars_clean)) <= 0.0:
                    raise InvalidSizingInputException(
                        f"{ERR_SZ_INVALID_CONFIG}: asset_cvars elements must be strictly positive (> 0.0)"
                    )
            else:
                cvars_clean = np.where(vols_clean > 0.0, 2.33 * vols_clean, 1.0)

            if max_iterations < 1:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: max_iterations must be >= 1, got {max_iterations}"
                )
            if not (isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)):
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: tolerance must be numeric, got {type(tolerance).__name__}"
                )
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise InvalidSizingInputException(
                    f"{ERR_SZ_INVALID_CONFIG}: tolerance must be finite and strictly positive, got {tolerance}"
                )

            effective_lots = lot_sizes if lot_sizes is not None else self.config.lot_sizes
            if effective_lots is not None:
                effective_lots = _validate_1d_array("lot_sizes", effective_lots, n_assets)
                if float(np.min(effective_lots)) <= 0.0:
                    raise InvalidSizingInputException(
                        f"{ERR_SZ_INVALID_CONFIG}: lot_sizes elements must be strictly positive (> 0.0)"
                    )
        else:
            mu_clean = mu
            n_assets = mu_clean.shape[0]
            sig2_clean = sigma2_epistemic
            cov_clean = cov_aleatoric
            vols_clean = asset_vols
            cross_clean = (
                cross_impact
                if cross_impact is not None
                else np.zeros((n_assets, n_assets), dtype=np.float64)
            )
            w_clean = total_capital
            k_clean = circuit_breaker_haircut
            cvars_clean = (
                asset_cvars
                if asset_cvars is not None
                else np.where(vols_clean > 0.0, 2.33 * vols_clean, 1.0)
            )
            effective_lots = lot_sizes if lot_sizes is not None else self.config.lot_sizes

        # Circuit Breaker HALT Defense: When kappa_t <= 0.0, freeze allocations strictly to 0
        if k_clean <= 0.0:
            zeros = np.zeros(n_assets, dtype=np.float64)
            return SizingDecision(
                target_allocations=zeros.copy(),
                discretized_allocations=zeros.copy(),
                effective_leverage=0.0,
                expected_shortfall=0.0,
                estimated_impact_cost=0.0,
                circuit_breaker_haircut=0.0,
                is_drawdown_constrained=False,
                is_leverage_constrained=False,
            )

        b_leverage = float(self.config.max_leverage * w_clean)
        b_cvar = float(self.config.mdd_budget * w_clean)

        # 1. Shrunk expected returns vector (Rule 4.3 directional epistemic shrinkage)
        shrunk_mu = self.kelly.compute_shrunk_returns(mu_clean, sig2_clean, validate=False)

        # 2. Precompute combined quadratic Hessian matrix Q = (gamma / W) * Sigma + (1 / W) * Lambda + (1 / (kappa_eff * W)) * I
        k_eff = self.regularizer.effective_haircut(k_clean)
        w_reg = 1.0 / (k_eff * w_clean)
        gamma_w = float(self.config.risk_aversion_gamma) / w_clean
        inv_w = 1.0 / w_clean

        Q = gamma_w * cov_clean + inv_w * cross_clean
        np.fill_diagonal(Q, Q.diagonal() + w_reg)

        # 3. Pseudo-Huber impact constants
        eta = float(self.config.impact_penalty_eta)
        delta_sq = float(self.config.impact_delta**2)
        delta_15 = float(self.config.impact_delta**1.5)
        one_and_half_eta = 1.5 * eta
        one_and_half_eta_vols = one_and_half_eta * vols_clean

        # 4. High-speed unconstrained Newton candidate
        nu_cand = np.zeros(n_assets, dtype=np.float64)
        neg_H = np.empty_like(Q)
        diag_idx = np.arange(n_assets)
        newton_tol = max(tolerance, 1e-4)

        for _ in range(4):
            q_x = Q @ nu_cand
            nu_sq = nu_cand * nu_cand
            inner = nu_sq + delta_sq
            q = np.sqrt(np.sqrt(inner))
            huber_grad = one_and_half_eta_vols * nu_cand / q
            grad_L = shrunk_mu - q_x - huber_grad

            inv_inner_54 = 1.0 / (inner * q)
            h_diag = one_and_half_eta_vols * inv_inner_54 * (0.5 * nu_sq + delta_sq)
            np.copyto(neg_H, Q)
            neg_H[diag_idx, diag_idx] += h_diag

            delta_nu = np.linalg.solve(neg_H, grad_L)
            nu_cand += delta_nu
            if float(np.max(np.abs(delta_nu))) < newton_tol:
                break

        # 5. Check unconstrained candidate feasibility
        abs_cand = np.abs(nu_cand)
        cand_l1 = float(np.sum(abs_cand))
        cand_cvar = float(np.dot(cvars_clean, abs_cand))

        if cand_l1 <= b_leverage and cand_cvar <= b_cvar:
            nu_star = nu_cand
        else:
            # 6. Boundary Barzilai-Borwein Spectral Projected Gradient (SPG)
            nu_curr = project_two_l1_constraints(
                nu_cand, cvars_clean, b_leverage, b_cvar, validate=False
            )

            def _eval_local(x: np.ndarray) -> tuple[float, np.ndarray]:
                q_x = Q @ x
                nu_sq_loc = x * x
                inner_loc = nu_sq_loc + delta_sq
                q_loc = np.sqrt(np.sqrt(inner_loc))
                h_grad = one_and_half_eta_vols * x / q_loc
                g = shrunk_mu - q_x - h_grad
                psi = (q_loc * q_loc * q_loc) - delta_15
                trans_cost = eta * float(np.sum(vols_clean * psi))
                obj = float(np.dot(x, shrunk_mu)) - 0.5 * float(np.dot(x, q_x)) - trans_cost
                return obj, g

            L_curr, g_curr = _eval_local(nu_curr)
            alpha = 25.0

            for _ in range(max_iterations):
                nu_trial = project_two_l1_constraints(
                    nu_curr + alpha * g_curr, cvars_clean, b_leverage, b_cvar, validate=False
                )
                d = nu_trial - nu_curr
                norm_d = float(np.max(np.abs(d)))
                if norm_d < tolerance:
                    nu_curr = nu_trial
                    break

                g_dot_d = float(np.dot(g_curr, d))
                beta = 1.0
                nu_next = nu_trial
                L_next = L_curr
                g_next = g_curr
                for _ in range(10):
                    nu_cand_step = nu_curr + beta * d
                    L_cand, g_cand = _eval_local(nu_cand_step)
                    if L_cand >= L_curr + 1e-4 * beta * g_dot_d:
                        nu_next = nu_cand_step
                        L_next = L_cand
                        g_next = g_cand
                        break
                    beta *= 0.5

                s = nu_next - nu_curr
                y_diff = g_curr - g_next
                sy = float(np.dot(s, y_diff))
                if sy > 1e-12:
                    alpha = float(np.clip(float(np.dot(s, s)) / sy, 1e-2, 1e4))
                else:
                    alpha = 25.0

                diff_step = float(np.max(np.abs(nu_next - nu_curr)))
                nu_curr = nu_next
                g_curr = g_next
                L_curr = L_next

                if diff_step < tolerance:
                    break

            nu_star = nu_curr

        # Final hard projection safety clamp (INV-TR-005)
        nu_star = project_two_l1_constraints(
            nu_star, cvars_clean, b_leverage, b_cvar, validate=False
        )

        abs_star = np.abs(nu_star)
        final_l1 = float(np.sum(abs_star))
        final_cvar = float(np.dot(cvars_clean, abs_star))

        # Active constraint threshold: within 1e-4 * W_t of boundary
        tol_bound = 1e-4 * w_clean
        is_leverage_constrained = bool((b_leverage - final_l1) <= tol_bound)
        is_drawdown_constrained = bool((b_cvar - final_cvar) <= tol_bound)

        # Microstructural lot discretization preserving risk budget in expectation
        nu_tilde = discretize_lot_allocations(
            target_allocations=nu_star,
            lot_sizes=effective_lots,
            random_seed=random_seed,
            validate=False,
        )

        if cross_impact is not None and np.any(cross_clean != 0.0):
            impact_cost = float(
                self.impact.evaluate(
                    nu=nu_star,
                    asset_vols=vols_clean,
                    cross_impact=cross_clean,
                    total_capital=w_clean,
                    validate=False,
                )
            )
        else:
            nu_sq_star = nu_star * nu_star
            q_star = np.sqrt(np.sqrt(nu_sq_star + delta_sq))
            psi_star = (q_star * q_star * q_star) - delta_15
            impact_cost = eta * float(np.dot(vols_clean, psi_star))

        return SizingDecision(
            target_allocations=nu_star,
            discretized_allocations=nu_tilde,
            effective_leverage=final_l1 / w_clean,
            expected_shortfall=final_cvar,
            estimated_impact_cost=impact_cost,
            circuit_breaker_haircut=k_clean,
            is_drawdown_constrained=is_drawdown_constrained,
            is_leverage_constrained=is_leverage_constrained,
        )
