"""Semi-Parametric Extreme Value Theory (EVT-POT) Tail Risk Subsystem.

Purpose:
    Provides institutional-grade, mathematically rigorous tail risk modeling for extreme
    negative portfolio return innovations:
    1. Peaks-Over-Threshold (POT) Generalized Pareto Distribution (GPD): Closed-form algebraic
       parameter estimation (xi, beta) eliminating iterative numerical solvers in the hot path.
    2. Coherent Expected Shortfall (CVaR): Coherent, subadditive downside tail risk strictly
       bounding Value-at-Risk from above (INV-TR-001).
    3. Fréchet Tail Stability & Infinite Variance Tripwire: Detects theoretical infinite
       variance (xi >= 1.0) and triggers emergency execution halts (INV-TR-002).
    4. Cold-Start Degradation Ladder: Seamless transition across Empirical Quantiles (N < 30),
       Student-t Method-of-Moments (30 <= N < 250), and Full EVT-GPD (N >= 250).

Dependencies:
    - math: Scalar finiteness and arithmetic operations.
    - dataclasses: Immutable frozen domain containers.
    - numpy: Vectorized numerical bounds and array operations.

Structural Relationship:
    - Ingests: Strictly lagged rolling loss innovations X_tau = -r_tau from portfolio bars.
    - Emits: EVTTailParameters, TailRiskMetrics (VaR_alpha, CVaR_alpha).
    - Consumed by: Unified convex execution sizer (execution_sizing.py) and circuit breaker
      overlay engine (circuit_breakers.py).

Invariants Enforced:
    - INV-TR-001 (Coherent Risk Ordering): cvar_alpha >= var_alpha within 1e-10 tolerance.
    - INV-TR-002 (Fréchet Tail Stability & Infinite Variance Tripwire): Tail index xi in
      [0.001, 0.999]. If xi >= 1.0, raise InfiniteVarianceException demanding emergency HALT.
    - INV-TR-005 (Non-Finite Input Data Protection): Immediate defensive failure on non-finite
      values (NaN / Inf) with DegenerateTailRiskException.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

# Diagnostic Fault Vector Constants (Rule 2: Deterministic Diagnostics)
ERR_TR_ORDERING: Final[str] = "ERR-TR-001: Coherent Risk Ordering violated (cvar_alpha < var_alpha)"
ERR_TR_INFINITE_VARIANCE: Final[str] = (
    "ERR-TR-002: Infinite theoretical variance tripwire (shape_xi >= 1.0)"
)
ERR_TR_DEGENERATE: Final[str] = "ERR-TR-003: Non-finite input, negative scale, or degenerate bounds"
ERR_TR_INVALID_CONFIG: Final[str] = (
    "ERR-TR-004: Invalid hyperparameter configuration or hierarchy violation"
)
ERR_TR_STARVATION: Final[str] = "ERR-TR-005: Insufficient observation history or buffer starvation"
ERR_TR_LATENCY: Final[str] = "ERR-TR-006: Tail risk evaluation latency violation or execution error"

VALID_TAIL_METHODS: Final[tuple[str, ...]] = ("EVT_PWM", "STUDENT_T", "EMPIRICAL")


class TailRiskError(Exception):
    """Base exception for all EVT tail risk and Expected Shortfall errors."""


class DegenerateTailRiskException(TailRiskError):
    """Raised when non-finite inputs (NaN/Inf), negative scale, or degenerate bounds are detected."""


class InfiniteVarianceException(TailRiskError):
    """Raised when tail shape index xi >= 1.0, indicating infinite theoretical variance demanding emergency HALT."""


class InvalidTailRiskInputException(TailRiskError):
    """Raised on invalid hyperparameter configurations, invalid types, or hierarchy contract violations."""


@dataclass(frozen=True)
class TailRiskConfig:
    """Hyperparameter configuration container for EVT-GPD tail risk estimation.

    Attributes:
        confidence_level: Downside tail confidence level alpha in (0.50, 1.0), default 0.99.
        threshold_k: Dynamic high-threshold standard deviation multiplier k > 0.0, default 1.645.
        min_observations_evt: Minimum sample size required for EVT-GPD modeling, default 250.
        min_observations_student_t: Minimum sample size for Student-t modeling, default 30.
        min_exceedances_evt: Minimum count of threshold exceedances N_u for EVT fit, default 15.
        tail_index_lower_bound: Lower clamp barrier for Fréchet tail shape index xi, default 0.001.
        tail_index_upper_bound: Upper clamp barrier for Fréchet tail shape index xi, default 0.999.
        rolling_window_size: Rolling historical buffer capacity W >= min_observations_evt, default 500.
    """

    confidence_level: float = 0.99
    threshold_k: float = 1.645
    min_observations_evt: int = 250
    min_observations_student_t: int = 30
    min_exceedances_evt: int = 15
    tail_index_lower_bound: float = 0.001
    tail_index_upper_bound: float = 0.999
    rolling_window_size: int = 500

    def __post_init__(self) -> None:
        """Validate invariant boundaries and threshold hierarchies upon instantiation."""
        # 1. Float field typing and finiteness verification
        for field_name in (
            "confidence_level",
            "threshold_k",
            "tail_index_lower_bound",
            "tail_index_upper_bound",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateTailRiskException(
                    f"{ERR_TR_DEGENERATE}: {field_name} must be a finite float, got {val}"
                )

        # 2. Integer field typing verification
        for field_name in (
            "min_observations_evt",
            "min_observations_student_t",
            "min_exceedances_evt",
            "rolling_window_size",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, int) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be an integer, got {type(val).__name__}"
                )

        # 3. Confidence level bounds (0.50 < alpha < 1.0)
        if not (0.50 < self.confidence_level < 1.0):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: confidence_level must be in (0.50, 1.0), got {self.confidence_level}"
            )

        # 4. Threshold multiplier bounds (k > 0.0)
        if self.threshold_k <= 0.0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: threshold_k must be strictly positive, got {self.threshold_k}"
            )

        # 5. Tail index bounds (0.0 < lower < upper < 1.0)
        if not (0.0 < self.tail_index_lower_bound < self.tail_index_upper_bound < 1.0):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: tail index bounds must satisfy 0.0 < lower < upper < 1.0, "
                f"got lower={self.tail_index_lower_bound}, upper={self.tail_index_upper_bound}"
            )

        # 6. Minimum observation counts must be strictly positive
        if self.min_observations_student_t < 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_observations_student_t must be >= 1, got {self.min_observations_student_t}"
            )
        if self.min_exceedances_evt < 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_exceedances_evt must be >= 1, got {self.min_exceedances_evt}"
            )

        # 7. Hierarchy contract: min_student_t < min_evt <= rolling_window_size
        if self.min_observations_student_t >= self.min_observations_evt:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_observations_student_t ({self.min_observations_student_t}) "
                f"must be strictly less than min_observations_evt ({self.min_observations_evt})"
            )
        if self.min_observations_evt > self.rolling_window_size:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_observations_evt ({self.min_observations_evt}) "
                f"cannot exceed rolling_window_size ({self.rolling_window_size})"
            )

    def clamp_tail_index(self, xi: float) -> float:
        """Clamp estimated tail index into stable Fréchet bounds [lower, upper].

        INV-TR-002: If xi >= 1.0, raises InfiniteVarianceException demanding emergency HALT.
        Otherwise clamps xi to [tail_index_lower_bound, tail_index_upper_bound].

        Args:
            xi: Raw estimated tail shape index.

        Returns:
            Clamped tail shape index strictly contained in [lower_bound, upper_bound].

        Raises:
            InvalidTailRiskInputException: If xi is not numeric.
            DegenerateTailRiskException: If xi is non-finite (NaN / Inf).
            InfiniteVarianceException: If xi >= 1.0 (INV-TR-002 infinite variance tripwire).
        """
        # Functional Purpose: Protect against explosive tail extrapolations and tripwire infinite variance.
        # Explicit Dependency Tracking: self.tail_index_lower_bound, self.tail_index_upper_bound.
        # Structural Relationship: Called by EVTTailRiskEngine during parameter calibration.
        # Defensive Invariant: INV-TR-002 Fréchet stability and infinite theoretical variance rejection.
        if not (isinstance(xi, (int, float)) and not isinstance(xi, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: xi must be numeric, got {type(xi).__name__}"
            )
        if not math.isfinite(xi):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: Cannot clamp non-finite tail index: {xi}"
            )
        if xi >= 1.0:
            raise InfiniteVarianceException(
                f"{ERR_TR_INFINITE_VARIANCE}: Estimated shape_xi={xi} >= 1.0 violates INV-TR-002. "
                "Theoretical variance is infinite; emergency HALT required."
            )
        return float(np.clip(xi, self.tail_index_lower_bound, self.tail_index_upper_bound))


@dataclass(frozen=True)
class EVTTailParameters:
    """Extreme Value Theory (EVT) Generalized Pareto Distribution (GPD) parameter container.

    Attributes:
        threshold_u: Dynamic high threshold u_t above which excess losses are evaluated.
        shape_xi: Fréchet tail shape parameter xi in [0.0, 1.0). If xi >= 1.0, raises InfiniteVarianceException.
        scale_beta: GPD scale parameter beta > 0.0 (or beta >= 0.0 for empirical fallback).
        num_exceedances: Count of observations strictly exceeding threshold u (N_u >= 0).
        total_observations: Total count of historical return innovations in sample (n >= N_u).
        method: Parameter estimation methodology ("EVT_PWM", "STUDENT_T", "EMPIRICAL").
    """

    threshold_u: float
    shape_xi: float
    scale_beta: float
    num_exceedances: int
    total_observations: int
    method: str

    def __post_init__(self) -> None:
        """Validate invariant boundaries and type integrity upon instantiation."""
        # 1. Float field typing and finiteness verification
        for field_name in ("threshold_u", "shape_xi", "scale_beta"):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateTailRiskException(
                    f"{ERR_TR_DEGENERATE}: {field_name} must be a finite float, got {val}"
                )

        # 2. Integer field typing verification
        for field_name in ("num_exceedances", "total_observations"):
            val = getattr(self, field_name)
            if not (isinstance(val, int) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be an integer, got {type(val).__name__}"
                )

        # 3. Method validation
        if not isinstance(self.method, str):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: method must be a string, got {type(self.method).__name__}"
            )
        if self.method not in VALID_TAIL_METHODS:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: method must be one of {VALID_TAIL_METHODS}, got '{self.method}'"
            )

        # 4. INV-TR-002: Fréchet Tail Stability & Infinite Variance Tripwire
        if self.shape_xi >= 1.0:
            raise InfiniteVarianceException(
                f"{ERR_TR_INFINITE_VARIANCE}: shape_xi={self.shape_xi} >= 1.0 violates INV-TR-002. "
                "Theoretical variance is infinite; emergency HALT required."
            )
        if self.shape_xi < 0.0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: shape_xi must be non-negative, got {self.shape_xi}"
            )

        # 5. Scale parameter beta validation
        if self.scale_beta < 0.0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: scale_beta must be non-negative, got {self.scale_beta}"
            )
        if self.method != "EMPIRICAL" and self.scale_beta <= 0.0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: scale_beta must be strictly positive for method '{self.method}', "
                f"got {self.scale_beta}"
            )

        # 6. Exceedance and observation count validation (0 <= N_u <= n)
        if self.total_observations < 0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: total_observations must be >= 0, got {self.total_observations}"
            )
        if self.num_exceedances < 0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: num_exceedances must be >= 0, got {self.num_exceedances}"
            )
        if self.num_exceedances > self.total_observations:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: num_exceedances ({self.num_exceedances}) "
                f"cannot exceed total_observations ({self.total_observations})"
            )


@dataclass(frozen=True)
class TailRiskMetrics:
    """Coherent tail risk metrics container holding Value-at-Risk and Expected Shortfall.

    Attributes:
        var_alpha: Value-at-Risk at confidence level alpha (loss quantile).
        cvar_alpha: Expected Shortfall (CVaR) at confidence level alpha (mean excess loss).
        confidence_level: Confidence level alpha in (0.50, 1.0).
        tail_parameters: Underlying EVTTailParameters calibrated for this evaluation.
        step_index: Sequential bar or execution step index >= 0.
    """

    var_alpha: float
    cvar_alpha: float
    confidence_level: float
    tail_parameters: EVTTailParameters
    step_index: int

    def __post_init__(self) -> None:
        """Validate invariant boundaries and coherent risk ordering (INV-TR-001)."""
        # 1. Float field typing and finiteness verification
        for field_name in ("var_alpha", "cvar_alpha", "confidence_level"):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateTailRiskException(
                    f"{ERR_TR_DEGENERATE}: {field_name} must be a finite float, got {val}"
                )

        # 2. Tail parameters type verification
        if not isinstance(self.tail_parameters, EVTTailParameters):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: tail_parameters must be an instance of EVTTailParameters, "
                f"got {type(self.tail_parameters).__name__}"
            )

        # 3. Step index typing and bounds verification
        if not (isinstance(self.step_index, int) and not isinstance(self.step_index, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: step_index must be an integer, got {type(self.step_index).__name__}"
            )
        if self.step_index < 0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: step_index must be >= 0, got {self.step_index}"
            )

        # 4. Confidence level bounds (0.50 < alpha < 1.0)
        if not (0.50 < self.confidence_level < 1.0):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: confidence_level must be in (0.50, 1.0), got {self.confidence_level}"
            )

        # 5. INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha within 1e-10 tolerance)
        # Functional Purpose: Enforce Artzner's coherence axiom; CVaR must strictly bound VaR from above.
        # Explicit Dependency Tracking: self.cvar_alpha, self.var_alpha.
        # Structural Relationship: Ingested by convex execution sizer for hard drawdown constraints.
        # Defensive Invariant: INV-TR-001 cvar_alpha >= var_alpha - 1e-10.
        if self.cvar_alpha < self.var_alpha - 1e-10:
            raise DegenerateTailRiskException(
                f"{ERR_TR_ORDERING}: INV-TR-001 violated: cvar_alpha ({self.cvar_alpha}) < "
                f"var_alpha ({self.var_alpha}) beyond 1e-10 numerical tolerance."
            )
