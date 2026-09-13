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
    - Emits: SizingConfig, SizingDecision, analytical utility, gradient, and Hessian tensors.
    - Consumed by: UnifiedConvexExecutionSizer in Phase 5 Step 3 Task 5.

Invariants Enforced:
    - INV-TR-004 (Strict Global Concavity): nabla^2 L(nu) << 0 everywhere; all eigenvalues of the
      total Hessian are strictly negative, guaranteeing a unique global allocation.
    - INV-TR-005 (Non-Finite Parameter Protection): Rejects NaN and Inf inputs immediately with
      DegenerateSizingException (ERR-SZ-002).
    - INV-TR-006 (Hot-Path Execution Latency SLA): Combined utility, gradient, and Hessian evaluation
      completes in <= 0.02ms for N = 10 assets.
    - Rule 4.3: Zero unconstrained Kelly betting; dynamic epistemic shrinkage; 3/2-power impact.
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

    def compute_shrunk_returns(
        self,
        mu: np.ndarray,
        sigma2_epistemic: np.ndarray,
        validate: bool = True,
    ) -> np.ndarray:
        """Compute epistemic-uncertainty-shrunk expected returns vector.

        Formula:
            mu_shrunk = mu - lambda_shrink * sigma2_epistemic

        Args:
            mu: Expected return vector (N,) from RD-DMA.
            sigma2_epistemic: Epistemic disagreement variance vector (N,).
            validate: If True, executes defensive input validation.

        Returns:
            Shrunk expected return vector (N,) as float64.
        """
        # Functional Purpose: Penalize assets with high model disagreement.
        # Explicit Dependency Tracking: mu, sigma2_epistemic, self.config.epistemic_shrinkage_lambda.
        # Structural Relationship: Feeds linear term in Kelly quadratic utility and gradient.
        # Defensive Invariant: Non-negative epistemic variances, dimensional match.
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

        return np.asarray(
            mu_clean - self.config.epistemic_shrinkage_lambda * sig2_clean,
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

        shrunk_mu = mu_clean - self.config.epistemic_shrinkage_lambda * sig2_clean
        linear_term = float(np.dot(nu_clean, shrunk_mu))
        cov_nu = cov_clean @ nu_clean
        risk_penalty = (
            0.5 * (self.config.risk_aversion_gamma / w_clean) * float(np.dot(nu_clean, cov_nu))
        )
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

        shrunk_mu = mu_clean - self.config.epistemic_shrinkage_lambda * sig2_clean
        cov_nu = cov_clean @ nu_clean
        gamma_w = self.config.risk_aversion_gamma / w_clean
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

        shrunk_mu = mu_clean - self.config.epistemic_shrinkage_lambda * sig2_clean
        cov_nu = cov_clean @ nu_clean
        return np.asarray(
            shrunk_mu - (self.config.risk_aversion_gamma / w_clean) * cov_nu,
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
            Negative semi-definite Hessian matrix (N, N) as float64.
        """
        # Functional Purpose: Evaluate closed-form Hessian matrix for Kelly utility term.
        # Explicit Dependency Tracking: cov_aleatoric, total_capital, self.config.risk_aversion_gamma.
        # Structural Relationship: Ingested by Newton step calculation in convex optimizer.
        # Defensive Invariant: INV-TR-004 negative semi-definiteness: nabla^2 U <= 0.
        if validate:
            cov_clean = _validate_2d_matrix("cov_aleatoric", cov_aleatoric)
            w_clean = _validate_positive_capital(total_capital)
        else:
            cov_clean = cov_aleatoric
            w_clean = total_capital

        return np.asarray(
            -(self.config.risk_aversion_gamma / w_clean) * cov_clean,
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
