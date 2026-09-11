"""Regime-Conditioned Dynamic Model Averaging (RD-DMA) subsystem.

Provides domain entities, configuration, state representations, and invariant contracts
for institutional-grade ensemble aggregation of quantitative strategies.

Invariants Enforced:
- INV-ENS-001 (Strict Simplex Conservation): Model weights must sum to 1.0 +/- 1e-10
  with all components strictly positive.
- INV-ENS-002 (Variance Positivity & Additivity): sigma^2_total = sigma^2_aleatoric + sigma^2_epistemic
  with sigma^2_aleatoric > 0 and sigma^2_epistemic >= 0.
- INV-ENS-003 (Bounded Adaptive Forgetting): alpha_t in [alpha_min, alpha_max] subset of (0, 1).
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np


class EnsembleError(Exception):
    """Base exception for all ensemble modeling and invariant violations."""


class DegenerateEnsembleException(EnsembleError):
    """Raised when numerical collapse, NaN/Inf, variance breach, or degeneracy occurs."""


class InvalidPredictionException(EnsembleError):
    """Raised when prediction payloads or dimension alignments are invalid."""


@dataclass(frozen=True)
class EnsembleConfig:
    """Hyperparameter configuration for Regime-Conditioned Dynamic Model Averaging.

    Enforces Invariant INV-ENS-003 and defensive boundary validations.

    Attributes:
        base_forgetting_factor: Baseline memory forgetting factor alpha_0 in (0.80, 0.999).
        volatility_sensitivity: Sensitivity coefficient kappa_alpha >= 0.0.
        min_forgetting_factor: Lower forgetting bound alpha_min in (0.50, base_forgetting_factor).
        max_forgetting_factor: Upper forgetting bound alpha_max in (base_forgetting_factor, 1.0).
        orthogonality_penalty: SVD subspace correlation penalty lambda_ortho >= 0.0.
        downside_penalty: Asymmetric downside loss multiplier gamma_down >= 0.0.
        turnover_damping: L1 turnover regularization weight lambda_churn in [0.0, 1.0).
        ridge_shrinkage: Tikhonov correlation shrinkage delta_ridge in (0.0, 1.0).
        ambiguity_shrinkage_cap: Thermodynamic shrinkage upper bound kappa_shrink in [0.0, 1.0].
        min_weight_floor: Minimum Laplace probability floor eps_floor > 0.0.
        temperature: Optimization entropy temperature tau > 0.0.
        mirror_descent_lr: Entropic Mirror Descent step size eta > 0.0.
        mirror_descent_max_iter: Maximum mirror descent iterations >= 1.
        mirror_descent_tol: Convergence tolerance for mirror descent > 0.0.
    """

    base_forgetting_factor: float = 0.96
    volatility_sensitivity: float = 0.50
    min_forgetting_factor: float = 0.85
    max_forgetting_factor: float = 0.99
    orthogonality_penalty: float = 0.25
    downside_penalty: float = 2.50
    turnover_damping: float = 0.15
    ridge_shrinkage: float = 0.05
    ambiguity_shrinkage_cap: float = 0.50
    min_weight_floor: float = 1e-5
    temperature: float = 1.0
    mirror_descent_lr: float = 0.50
    mirror_descent_max_iter: int = 10
    mirror_descent_tol: float = 1e-6

    def __post_init__(self) -> None:
        """Validate invariant boundary contracts on hyperparameters."""
        # Finiteness check on all float attributes
        for field_name in (
            "base_forgetting_factor",
            "volatility_sensitivity",
            "min_forgetting_factor",
            "max_forgetting_factor",
            "orthogonality_penalty",
            "downside_penalty",
            "turnover_damping",
            "ridge_shrinkage",
            "ambiguity_shrinkage_cap",
            "min_weight_floor",
            "temperature",
            "mirror_descent_lr",
            "mirror_descent_tol",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise EnsembleError(f"{field_name} must be a finite float, got {val}")

        # INV-ENS-003 bounds & relations
        if self.min_forgetting_factor >= self.max_forgetting_factor:
            raise EnsembleError(
                f"min_forgetting_factor ({self.min_forgetting_factor}) must be strictly less than "
                f"max_forgetting_factor ({self.max_forgetting_factor})"
            )
        if not (0.80 < self.base_forgetting_factor < 0.999):
            raise EnsembleError(
                f"base_forgetting_factor must be in (0.80, 0.999), got {self.base_forgetting_factor}"
            )
        if not (0.50 < self.min_forgetting_factor < self.base_forgetting_factor):
            raise EnsembleError(
                "min_forgetting_factor must be in (0.50, base_forgetting_factor), "
                f"got min={self.min_forgetting_factor}, base={self.base_forgetting_factor}"
            )
        if not (self.base_forgetting_factor < self.max_forgetting_factor < 1.0):
            raise EnsembleError(
                "max_forgetting_factor must be in (base_forgetting_factor, 1.0), "
                f"got max={self.max_forgetting_factor}, base={self.base_forgetting_factor}"
            )

        # Non-negative penalties
        if self.volatility_sensitivity < 0.0:
            raise EnsembleError(
                f"volatility_sensitivity must be >= 0.0, got {self.volatility_sensitivity}"
            )
        if self.orthogonality_penalty < 0.0:
            raise EnsembleError(
                f"orthogonality_penalty must be >= 0.0, got {self.orthogonality_penalty}"
            )
        if self.downside_penalty < 0.0:
            raise EnsembleError(f"downside_penalty must be >= 0.0, got {self.downside_penalty}")

        # Turnover damping in [0.0, 1.0)
        if not (0.0 <= self.turnover_damping < 1.0):
            raise EnsembleError(
                f"turnover_damping must be in [0.0, 1.0), got {self.turnover_damping}"
            )

        # Ridge shrinkage in (0.0, 1.0)
        if not (0.0 < self.ridge_shrinkage < 1.0):
            raise EnsembleError(
                f"ridge_shrinkage must be in (0.0, 1.0), got {self.ridge_shrinkage}"
            )

        # Ambiguity shrinkage cap in [0.0, 1.0]
        if not (0.0 <= self.ambiguity_shrinkage_cap <= 1.0):
            raise EnsembleError(
                f"ambiguity_shrinkage_cap must be in [0.0, 1.0], got {self.ambiguity_shrinkage_cap}"
            )

        # Strictly positive parameters
        if self.min_weight_floor <= 0.0:
            raise EnsembleError(f"min_weight_floor must be > 0.0, got {self.min_weight_floor}")
        if self.temperature <= 0.0:
            raise EnsembleError(f"temperature must be > 0.0, got {self.temperature}")
        if self.mirror_descent_lr <= 0.0:
            raise EnsembleError(f"mirror_descent_lr must be > 0.0, got {self.mirror_descent_lr}")
        if self.mirror_descent_max_iter < 1:
            raise EnsembleError(
                f"mirror_descent_max_iter must be >= 1, got {self.mirror_descent_max_iter}"
            )
        if self.mirror_descent_tol <= 0.0:
            raise EnsembleError(f"mirror_descent_tol must be > 0.0, got {self.mirror_descent_tol}")


@dataclass(frozen=True)
class EnsembleState:
    """Immutable state snapshot tracking online dynamic model averaging across time.

    Enforces Invariant INV-ENS-001 (Strict Simplex Conservation) and dimensional alignment.

    Attributes:
        step_index: Monotonically increasing bar counter t >= 0.
        weights: Active composite model weights vector w_t (shape: (K,), sum=1.0, w_k > 0).
        regime_conditional_posteriors: Matrix of regime-conditioned weights (shape: (3, K), rows sum=1.0).
        cumulative_losses: Historical accumulated asymmetric downside losses (shape: (K,)).
        effective_models: Effective number of constituent models K_eff in [1.0, K].
        mean_realized_volatility: Exponential moving average baseline volatility sigma_bar > 0.0.
        last_ambiguity_temperature: Most recent thermodynamic temperature beta_t > 0.0.
    """

    step_index: int
    weights: np.ndarray
    regime_conditional_posteriors: np.ndarray
    cumulative_losses: np.ndarray
    effective_models: float
    mean_realized_volatility: float
    last_ambiguity_temperature: float

    def __post_init__(self) -> None:
        """Validate invariant contracts on state variables."""
        if self.step_index < 0:
            raise EnsembleError(f"step_index must be >= 0, got {self.step_index}")

        # Weights validation: INV-ENS-001
        if not isinstance(self.weights, np.ndarray) or self.weights.ndim != 1:
            raise EnsembleError("weights must be a 1D numpy array")
        k_models = len(self.weights)
        if k_models < 1:
            raise EnsembleError("weights must contain at least 1 model")
        if not np.all(np.isfinite(self.weights)):
            raise DegenerateEnsembleException("weights must contain only finite values")
        if np.any(self.weights <= 0.0):
            raise EnsembleError("INV-ENS-001 violation: weights must be strictly positive")
        weights_sum = float(np.sum(self.weights))
        if not math.isclose(weights_sum, 1.0, abs_tol=1e-10):
            raise EnsembleError(
                f"INV-ENS-001 violation: weights sum to {weights_sum}, expected 1.0 +/- 1e-10"
            )

        # Regime conditional posteriors validation
        if not isinstance(self.regime_conditional_posteriors, np.ndarray):
            raise EnsembleError("regime_conditional_posteriors must be a numpy array")
        if self.regime_conditional_posteriors.shape != (3, k_models):
            raise EnsembleError(
                f"regime_conditional_posteriors shape must be (3, {k_models}), "
                f"got {self.regime_conditional_posteriors.shape}"
            )
        if not np.all(np.isfinite(self.regime_conditional_posteriors)):
            raise DegenerateEnsembleException(
                "regime_conditional_posteriors must contain only finite values"
            )
        if np.any(self.regime_conditional_posteriors < 0.0):
            raise EnsembleError("regime_conditional_posteriors elements must be non-negative")
        for row_idx in range(3):
            row_sum = float(np.sum(self.regime_conditional_posteriors[row_idx, :]))
            if not math.isclose(row_sum, 1.0, abs_tol=1e-10):
                raise EnsembleError(
                    f"regime_conditional_posteriors row {row_idx} must sum to 1.0 +/- 1e-10, got {row_sum}"
                )

        # Cumulative losses validation
        if not isinstance(self.cumulative_losses, np.ndarray):
            raise EnsembleError("cumulative_losses must be a numpy array")
        if self.cumulative_losses.shape != (k_models,):
            raise EnsembleError(
                f"cumulative_losses shape must be ({k_models},), got {self.cumulative_losses.shape}"
            )
        if not np.all(np.isfinite(self.cumulative_losses)):
            raise DegenerateEnsembleException("cumulative_losses must contain only finite values")

        # Effective models bounds check [1.0, K]
        if not math.isfinite(self.effective_models):
            raise DegenerateEnsembleException("effective_models must be finite")
        if self.effective_models < 1.0 - 1e-6 or self.effective_models > float(k_models) + 1e-6:
            raise EnsembleError(
                f"effective_models must be in [1.0, {k_models}], got {self.effective_models}"
            )

        # Volatility baseline check
        if not (
            math.isfinite(self.mean_realized_volatility) and self.mean_realized_volatility > 0.0
        ):
            raise EnsembleError(
                f"mean_realized_volatility must be > 0.0, got {self.mean_realized_volatility}"
            )

        # Ambiguity temperature check
        if not (
            math.isfinite(self.last_ambiguity_temperature) and self.last_ambiguity_temperature > 0.0
        ):
            raise EnsembleError(
                f"last_ambiguity_temperature must be > 0.0, got {self.last_ambiguity_temperature}"
            )


@dataclass(frozen=True)
class EnsemblePrediction:
    """Probabilistic prediction payload emitted by the RD-DMA engine at bar t.

    Enforces Invariant INV-ENS-002 (Law of Total Variance decomposition).

    Attributes:
        point_prediction: Aggregated return prediction mu_hat_t.
        aleatoric_variance: Expected downside process variance sigma^2_aleatoric > 0.0.
        epistemic_variance: Strategy disagreement variance sigma^2_epistemic >= 0.0.
        total_variance: Total predictive variance sigma^2_total = aleatoric + epistemic.
        model_weights: Posterior model allocation vector w_t (shape: (K,), sum=1.0).
        regime_probabilities: Predictive forward regime distribution p_{t+1|t} (shape: (3,), sum=1.0).
        effective_models: Effective number of models K_eff in [1.0, K].
        volatility_forgetting_factor: Applied volatility-adaptive forgetting factor alpha_t.
        ambiguity_shrinkage_weight: Applied thermodynamic shrinkage weight lambda_beta.
    """

    point_prediction: float
    aleatoric_variance: float
    epistemic_variance: float
    total_variance: float
    model_weights: np.ndarray
    regime_probabilities: np.ndarray
    effective_models: float
    volatility_forgetting_factor: float
    ambiguity_shrinkage_weight: float

    def __post_init__(self) -> None:
        """Validate invariant contracts on predictive emission."""
        # Finiteness checks on scalar attributes
        for field_name in (
            "point_prediction",
            "aleatoric_variance",
            "epistemic_variance",
            "total_variance",
            "effective_models",
            "volatility_forgetting_factor",
            "ambiguity_shrinkage_weight",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise DegenerateEnsembleException(f"{field_name} must be a finite float, got {val}")

        # Invariant INV-ENS-002: Variance positivity and additivity
        if self.aleatoric_variance <= 0.0:
            raise DegenerateEnsembleException(
                f"INV-ENS-002 violation: aleatoric_variance must be > 0.0, got {self.aleatoric_variance}"
            )
        if self.epistemic_variance < 0.0:
            raise DegenerateEnsembleException(
                f"INV-ENS-002 violation: epistemic_variance must be >= 0.0, got {self.epistemic_variance}"
            )
        expected_total = self.aleatoric_variance + self.epistemic_variance
        if not math.isclose(self.total_variance, expected_total, rel_tol=1e-7, abs_tol=1e-10):
            raise DegenerateEnsembleException(
                f"INV-ENS-002 violation: total_variance ({self.total_variance}) != "
                f"aleatoric ({self.aleatoric_variance}) + epistemic ({self.epistemic_variance})"
            )

        # Model weights validation
        if not isinstance(self.model_weights, np.ndarray) or self.model_weights.ndim != 1:
            raise DegenerateEnsembleException("model_weights must be a 1D numpy array")
        k_models = len(self.model_weights)
        if k_models < 1:
            raise DegenerateEnsembleException("model_weights must have length >= 1")
        if not np.all(np.isfinite(self.model_weights)):
            raise DegenerateEnsembleException("model_weights must contain only finite values")
        if np.any(self.model_weights <= 0.0):
            raise DegenerateEnsembleException(
                "INV-ENS-001 violation: model_weights must be strictly positive"
            )
        weights_sum = float(np.sum(self.model_weights))
        if not math.isclose(weights_sum, 1.0, abs_tol=1e-10):
            raise DegenerateEnsembleException(
                f"INV-ENS-001 violation: model_weights sum to {weights_sum}, expected 1.0 +/- 1e-10"
            )

        # Regime probabilities validation
        if not isinstance(
            self.regime_probabilities, np.ndarray
        ) or self.regime_probabilities.shape != (3,):
            raise DegenerateEnsembleException("regime_probabilities must have shape (3,)")
        if not np.all(np.isfinite(self.regime_probabilities)):
            raise DegenerateEnsembleException(
                "regime_probabilities must contain only finite values"
            )
        if np.any(self.regime_probabilities < 0.0):
            raise DegenerateEnsembleException("regime_probabilities elements must be non-negative")
        regime_sum = float(np.sum(self.regime_probabilities))
        if not math.isclose(regime_sum, 1.0, abs_tol=1e-10):
            raise DegenerateEnsembleException(
                f"regime_probabilities sum to {regime_sum}, expected 1.0 +/- 1e-10"
            )

        # Effective models bounds check
        if self.effective_models < 1.0 - 1e-6 or self.effective_models > float(k_models) + 1e-6:
            raise DegenerateEnsembleException(
                f"effective_models must be in [1.0, {k_models}], got {self.effective_models}"
            )

        # Adaptive forgetting factor bounds check
        if not (0.0 < self.volatility_forgetting_factor < 1.0):
            raise DegenerateEnsembleException(
                f"volatility_forgetting_factor must be in (0.0, 1.0), got {self.volatility_forgetting_factor}"
            )

        # Ambiguity shrinkage weight bounds check
        if not (0.0 <= self.ambiguity_shrinkage_weight <= 1.0):
            raise DegenerateEnsembleException(
                f"ambiguity_shrinkage_weight must be in [0.0, 1.0], got {self.ambiguity_shrinkage_weight}"
            )


class VolatilityAdaptiveForgetting:
    """Dynamically computes volatility-adaptive forgetting factors alpha_t.

    Enforces Invariant INV-ENS-003 by strictly clamping memory decay factors
    to [alpha_min, alpha_max] based on normalized volatility deviations from a
    running exponential moving average baseline.

    In calm regimes (sigma_t < sigma_bar), alpha_t approaches alpha_max (expanding
    effective memory window). In panic shocks (sigma_t >> sigma_bar), alpha_t contracts
    toward alpha_min (accelerating memory decay to quickly adapt to structural shifts).
    """

    def __init__(
        self,
        config: EnsembleConfig,
        initial_mean_vol: float = 0.01,
        vol_smoothing: float = 0.10,
    ) -> None:
        """Initialize the volatility-adaptive forgetting tracker.

        Args:
            config: EnsembleConfig containing base, min, max forgetting factors and sensitivity.
            initial_mean_vol: Initial baseline volatility sigma_bar_0 > 0.0.
            vol_smoothing: Exponential moving average update rate beta in (0.0, 1.0].

        Raises:
            EnsembleError: If initial_mean_vol <= 0 or vol_smoothing not in (0, 1].
            DegenerateEnsembleException: If initial_mean_vol or vol_smoothing is non-finite.
        """
        if not isinstance(config, EnsembleConfig):
            raise EnsembleError(f"config must be an EnsembleConfig, got {type(config)}")

        if not (isinstance(initial_mean_vol, (int, float)) and math.isfinite(initial_mean_vol)):
            raise DegenerateEnsembleException(
                f"initial_mean_vol must be a finite float, got {initial_mean_vol}"
            )
        if initial_mean_vol <= 0.0:
            raise EnsembleError(f"initial_mean_vol must be > 0.0, got {initial_mean_vol}")

        if not (isinstance(vol_smoothing, (int, float)) and math.isfinite(vol_smoothing)):
            raise DegenerateEnsembleException(
                f"vol_smoothing must be a finite float, got {vol_smoothing}"
            )
        if not (0.0 < vol_smoothing <= 1.0):
            raise EnsembleError(f"vol_smoothing must be in (0, 1], got {vol_smoothing}")

        self._config = config
        self._mean_realized_vol: float = float(initial_mean_vol)
        self._smoothing: float = float(vol_smoothing)

    @property
    def mean_realized_volatility(self) -> float:
        """Current exponential moving average baseline volatility sigma_bar."""
        return self._mean_realized_vol

    def compute_alpha(self, realized_vol: float) -> float:
        """Compute volatility-adapted forgetting factor alpha_t and update baseline volatility.

        Args:
            realized_vol: Contemporaneous realized volatility observation sigma_t > 0.0.

        Returns:
            alpha_t: Memory forgetting factor strictly bounded in [alpha_min, alpha_max].

        Raises:
            DegenerateEnsembleException: If realized_vol is non-finite (NaN/Inf).
            EnsembleError: If realized_vol <= 0.0.
        """
        if not (isinstance(realized_vol, (int, float)) and math.isfinite(realized_vol)):
            raise DegenerateEnsembleException(
                f"realized_vol must be a finite float, got {realized_vol}"
            )
        if realized_vol <= 0.0:
            raise EnsembleError(f"realized_vol must be > 0.0, got {realized_vol}")

        # Normalized volatility deviation Delta sigma = (sigma_t - sigma_bar) / sigma_bar
        delta_sigma = (realized_vol - self._mean_realized_vol) / self._mean_realized_vol

        # Dynamic alpha formulation: alpha_t = clip(alpha_0 - kappa_alpha * Delta sigma, alpha_min, alpha_max)
        raw_alpha = (
            self._config.base_forgetting_factor - self._config.volatility_sensitivity * delta_sigma
        )
        alpha_t = float(
            np.clip(
                raw_alpha,
                self._config.min_forgetting_factor,
                self._config.max_forgetting_factor,
            )
        )

        # Running baseline update: sigma_bar <- (1 - beta) * sigma_bar + beta * sigma_t
        self._mean_realized_vol = (1.0 - self._smoothing) * self._mean_realized_vol + (
            self._smoothing * float(realized_vol)
        )

        return alpha_t


class AsymmetricDownsideLossScorer:
    """Asymmetric downside prediction loss scorer and downside semi-variance evaluator.

    Evaluates strategy prediction errors with asymmetric downside penalization:
        l_{t, k} = (y_t - y_tilde_k)^2 + gamma_down * max(0, -y_t * y_tilde_k)
    where y_tilde_k = y_hat_k / sqrt(H_k) standardizes across heterogeneous forecast horizons.

    Also evaluates empirical downside semi-variance:
        sigma^2_{k, down} = (1/N) * sum_{i=1}^N min(0.0, r_i - target)^2
    subject to a strictly positive defensive variance floor (>= 1e-8).
    """

    def __init__(self, downside_penalty: float = 2.50) -> None:
        """Initialize the asymmetric downside loss scorer.

        Args:
            downside_penalty: Asymmetric penalty multiplier gamma_down >= 0.0.

        Raises:
            EnsembleError: If downside_penalty < 0.0 or is non-finite / invalid.
        """
        if not (isinstance(downside_penalty, (int, float)) and math.isfinite(downside_penalty)):
            raise EnsembleError(f"downside_penalty must be a finite float, got {downside_penalty}")
        if downside_penalty < 0.0:
            raise EnsembleError(f"downside_penalty must be >= 0.0, got {downside_penalty}")
        self._downside_penalty: float = float(downside_penalty)

    @property
    def downside_penalty(self) -> float:
        """Asymmetric downside loss penalty multiplier gamma_down."""
        return self._downside_penalty

    def compute_losses(
        self,
        predictions: np.ndarray,
        realized_return: float,
        forecast_horizons: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute asymmetric downside prediction losses for strategy forecast candidates.

        Args:
            predictions: 1D array of nominal strategy return forecasts (shape: (K,)).
            realized_return: Contemporaneous realized benchmark/market return y_t.
            forecast_horizons: Optional 1D array of model forecast horizons H_k >= 1.0 (shape: (K,)).

        Returns:
            losses: 1D array of non-negative prediction losses (shape: (K,)).

        Raises:
            InvalidPredictionException: If predictions or horizons have invalid shape/dimensions
                or horizons < 1.0.
            DegenerateEnsembleException: If predictions, realized_return, or horizons contain NaN/Inf.
        """
        if not isinstance(predictions, np.ndarray) or predictions.ndim != 1:
            raise InvalidPredictionException("predictions must be a 1D numpy array")
        if len(predictions) < 1:
            raise InvalidPredictionException("predictions must contain at least 1 model prediction")
        if not np.all(np.isfinite(predictions)):
            raise DegenerateEnsembleException("predictions must contain only finite values")

        if not (isinstance(realized_return, (int, float)) and math.isfinite(realized_return)):
            raise DegenerateEnsembleException(
                f"realized_return must be a finite float, got {realized_return}"
            )

        if forecast_horizons is not None:
            if not isinstance(forecast_horizons, np.ndarray) or forecast_horizons.ndim != 1:
                raise InvalidPredictionException("forecast_horizons must be a 1D numpy array")
            if forecast_horizons.shape != predictions.shape:
                raise InvalidPredictionException(
                    f"forecast_horizons shape {forecast_horizons.shape} must match predictions shape {predictions.shape}"
                )
            if not np.all(np.isfinite(forecast_horizons)):
                raise DegenerateEnsembleException(
                    "forecast_horizons must contain only finite values"
                )
            if np.any(forecast_horizons < 1.0):
                raise InvalidPredictionException("forecast_horizons elements must be >= 1.0")
            scaled_predictions: np.ndarray = predictions / np.sqrt(forecast_horizons)
        else:
            scaled_predictions = predictions.astype(np.float64, copy=False)

        squared_error = (realized_return - scaled_predictions) ** 2
        sign_product = -float(realized_return) * scaled_predictions
        downside_penalty = self._downside_penalty * np.maximum(0.0, sign_product)

        losses = squared_error + downside_penalty
        losses = np.maximum(0.0, losses)
        return losses.astype(np.float64)

    def compute_downside_semi_variance(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        min_variance_floor: float = 1e-8,
    ) -> float:
        """Compute empirical downside semi-variance for a strategy return series.

        Args:
            returns: 1D array of historical strategy returns (length >= 1).
            target_return: Minimum acceptable return threshold (MAR), default 0.0.
            min_variance_floor: Strictly positive defensive variance floor, default 1e-8.

        Returns:
            downside_semi_variance: Empirical downside semi-variance clamped to >= min_variance_floor.

        Raises:
            InvalidPredictionException: If returns is not 1D or is empty.
            DegenerateEnsembleException: If returns, target_return, or min_variance_floor contain NaN/Inf.
            EnsembleError: If min_variance_floor <= 0.0.
        """
        if not isinstance(returns, np.ndarray) or returns.ndim != 1:
            raise InvalidPredictionException("returns must be a 1D numpy array")
        if len(returns) < 1:
            raise InvalidPredictionException("returns must contain at least 1 observation")
        if not np.all(np.isfinite(returns)):
            raise DegenerateEnsembleException("returns must contain only finite values")

        if not (isinstance(target_return, (int, float)) and math.isfinite(target_return)):
            raise DegenerateEnsembleException(
                f"target_return must be a finite float, got {target_return}"
            )

        if not (isinstance(min_variance_floor, (int, float)) and math.isfinite(min_variance_floor)):
            raise DegenerateEnsembleException(
                f"min_variance_floor must be a finite float, got {min_variance_floor}"
            )
        if min_variance_floor <= 0.0:
            raise EnsembleError(f"min_variance_floor must be > 0.0, got {min_variance_floor}")

        deviations = np.minimum(0.0, returns - target_return)
        semi_variance = float(np.mean(deviations**2))
        return float(max(semi_variance, float(min_variance_floor)))

    def compute_cohort_downside_variances(
        self,
        return_matrix: np.ndarray,
        target_return: float = 0.0,
        min_variance_floor: float = 1e-8,
    ) -> np.ndarray:
        """Compute downside semi-variances across a cohort of strategy return series.

        Args:
            return_matrix: 2D array of historical returns (shape: (N, K)) where N is time bars
                and K is strategy models.
            target_return: Minimum acceptable return threshold (MAR), default 0.0.
            min_variance_floor: Strictly positive defensive variance floor, default 1e-8.

        Returns:
            cohort_downside_variances: 1D array of downside semi-variances (shape: (K,)).

        Raises:
            InvalidPredictionException: If return_matrix is not 2D or is empty.
            DegenerateEnsembleException: If return_matrix, target_return, or min_variance_floor contain NaN/Inf.
            EnsembleError: If min_variance_floor <= 0.0.
        """
        if not isinstance(return_matrix, np.ndarray) or return_matrix.ndim != 2:
            raise InvalidPredictionException("return_matrix must be a 2D numpy array")
        if return_matrix.shape[0] < 1 or return_matrix.shape[1] < 1:
            raise InvalidPredictionException(
                f"return_matrix must have shape (N, K) with N >= 1, K >= 1, got {return_matrix.shape}"
            )
        if not np.all(np.isfinite(return_matrix)):
            raise DegenerateEnsembleException("return_matrix must contain only finite values")

        if not (isinstance(target_return, (int, float)) and math.isfinite(target_return)):
            raise DegenerateEnsembleException(
                f"target_return must be a finite float, got {target_return}"
            )

        if not (isinstance(min_variance_floor, (int, float)) and math.isfinite(min_variance_floor)):
            raise DegenerateEnsembleException(
                f"min_variance_floor must be a finite float, got {min_variance_floor}"
            )
        if min_variance_floor <= 0.0:
            raise EnsembleError(f"min_variance_floor must be > 0.0, got {min_variance_floor}")

        deviations = np.minimum(0.0, return_matrix - target_return)
        col_vars: np.ndarray = np.mean(deviations**2, axis=0)
        clamped_vars = np.maximum(col_vars, float(min_variance_floor))
        return cast(np.ndarray, clamped_vars.astype(np.float64))


def predict_forward_regime_prior(
    current_regime_probs: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """Project current regime probabilities forward via Markov transition matrix.

    Computes p_{t+1|t} = P_trans^T * p_t, projecting forward probability distribution
    over regimes {0: Absorption, 1: Momentum, 2: Panic} for predictive model weighting.

    Args:
        current_regime_probs: 1D array of shape (3,) summing to 1.0 +/- 1e-10 with p_i >= 0.
        transition_matrix: 2D array of shape (3, 3) where each row sums to 1.0 +/- 1e-10.

    Returns:
        forward_regime_probs: 1D array of shape (3,) summing strictly to 1.0.

    Raises:
        InvalidPredictionException: If inputs fail shape, non-negativity, or simplex constraints.
        DegenerateEnsembleException: If inputs contain NaN or Inf values.
    """
    # Defensive type and shape validations for current_regime_probs
    if not isinstance(current_regime_probs, np.ndarray) or current_regime_probs.ndim != 1:
        raise InvalidPredictionException("current_regime_probs must be a 1D numpy array")
    if current_regime_probs.shape != (3,):
        raise InvalidPredictionException(
            f"current_regime_probs must have shape (3,), got {current_regime_probs.shape}"
        )
    if not np.all(np.isfinite(current_regime_probs)):
        raise DegenerateEnsembleException("current_regime_probs must contain only finite values")
    if np.any(current_regime_probs < 0.0):
        raise InvalidPredictionException("current_regime_probs elements must be non-negative")
    p_sum = float(np.sum(current_regime_probs))
    if not math.isclose(p_sum, 1.0, abs_tol=1e-10):
        raise InvalidPredictionException(
            f"current_regime_probs must sum to 1.0 +/- 1e-10, got {p_sum}"
        )

    # Defensive type and shape validations for transition_matrix
    if not isinstance(transition_matrix, np.ndarray) or transition_matrix.ndim != 2:
        raise InvalidPredictionException("transition_matrix must be a 2D numpy array")
    if transition_matrix.shape != (3, 3):
        raise InvalidPredictionException(
            f"transition_matrix must have shape (3, 3), got {transition_matrix.shape}"
        )
    if not np.all(np.isfinite(transition_matrix)):
        raise DegenerateEnsembleException("transition_matrix must contain only finite values")
    if np.any(transition_matrix < 0.0):
        raise InvalidPredictionException("transition_matrix elements must be non-negative")
    for row_idx in range(3):
        row_sum = float(np.sum(transition_matrix[row_idx, :]))
        if not math.isclose(row_sum, 1.0, abs_tol=1e-10):
            raise InvalidPredictionException(
                f"transition_matrix row {row_idx} must sum to 1.0 +/- 1e-10, got {row_sum}"
            )

    # Compute forward projection: p_{t+1|t} = P_trans^T * p_t
    forward_p: np.ndarray = transition_matrix.T @ current_regime_probs
    forward_p = np.clip(forward_p, 0.0, 1.0)
    norm_sum = float(np.sum(forward_p))
    if not (math.isfinite(norm_sum) and norm_sum > 0.0):
        raise DegenerateEnsembleException(
            "Forward regime probabilities sum collapsed to zero or non-finite"
        )
    forward_p = forward_p / norm_sum

    return forward_p


def update_regime_posteriors(
    posteriors: np.ndarray,
    alpha_t: float,
    forward_regime_probs: np.ndarray,
    alpha_min: float = 0.50,
    alpha_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute tempered regime-conditioned model posteriors and composite prior.

    For each regime j in {0, 1, 2}, tempers historical model posteriors by memory factor alpha_t
    using log-space max-shift for guaranteed numerical stability under extreme likelihoods:
        ln pi_{j, k} = alpha_t * ln(pi_{j, k} + 1e-30)
        pi_{j, k}^tempered = exp(ln pi_{j, k} - max_m ln pi_{j, m}) / sum_l exp(ln pi_{j, l} - max_m ln pi_{j, m})

    Blends the tempered regime posteriors with predictive forward regime probabilities:
        bar{pi}_k = sum_{j=0}^2 p_{t+1|t, j} * pi_{j, k}^tempered

    Args:
        posteriors: Matrix of regime-conditioned model probabilities of shape (3, K),
            where each row sums to 1.0 +/- 1e-10 with non-negative elements.
        alpha_t: Memory forgetting factor strictly within [alpha_min, alpha_max] and (0.0, 1.0).
        forward_regime_probs: Predictive forward regime vector p_{t+1|t} of shape (3,).
        alpha_min: Lower bound for alpha_t validation (defaults to 0.50).
        alpha_max: Upper bound for alpha_t validation (defaults to 1.0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - updated_posteriors: Tempered posteriors matrix of shape (3, K), strictly normalized per row.
            - composite_prior: Blended prior distribution vector of shape (K,), strictly normalized.

    Raises:
        InvalidPredictionException: If posteriors or forward_regime_probs fail dimensions,
            non-negativity, or simplex constraints.
        DegenerateEnsembleException: If inputs contain NaN/Inf or alpha_t violates bounds.
    """
    # Defensive validation on posteriors
    if not isinstance(posteriors, np.ndarray) or posteriors.ndim != 2:
        raise InvalidPredictionException("posteriors must be a 2D numpy array")
    if posteriors.shape[0] != 3 or posteriors.shape[1] < 1:
        raise InvalidPredictionException(
            f"posteriors must have shape (3, K) with K >= 1, got {posteriors.shape}"
        )
    if not np.all(np.isfinite(posteriors)):
        raise DegenerateEnsembleException("posteriors must contain only finite values")
    if np.any(posteriors < 0.0):
        raise InvalidPredictionException("posteriors elements must be non-negative")
    for row_idx in range(3):
        row_sum = float(np.sum(posteriors[row_idx, :]))
        if not math.isclose(row_sum, 1.0, abs_tol=1e-10):
            raise InvalidPredictionException(
                f"posteriors row {row_idx} must sum to 1.0 +/- 1e-10, got {row_sum}"
            )

    # Defensive validation on alpha_t
    if not (isinstance(alpha_t, (int, float)) and math.isfinite(alpha_t)):
        raise DegenerateEnsembleException(f"alpha_t must be a finite float, got {alpha_t}")
    if not (alpha_min <= alpha_t <= alpha_max and 0.0 < alpha_t < 1.0):
        raise DegenerateEnsembleException(
            f"alpha_t must be in [{alpha_min}, {alpha_max}] and (0, 1), got {alpha_t}"
        )

    # Defensive validation on forward_regime_probs
    if not isinstance(forward_regime_probs, np.ndarray) or forward_regime_probs.ndim != 1:
        raise InvalidPredictionException("forward_regime_probs must be a 1D numpy array")
    if forward_regime_probs.shape != (3,):
        raise InvalidPredictionException(
            f"forward_regime_probs must have shape (3,), got {forward_regime_probs.shape}"
        )
    if not np.all(np.isfinite(forward_regime_probs)):
        raise DegenerateEnsembleException("forward_regime_probs must contain only finite values")
    if np.any(forward_regime_probs < 0.0):
        raise InvalidPredictionException("forward_regime_probs elements must be non-negative")
    f_sum = float(np.sum(forward_regime_probs))
    if not math.isclose(f_sum, 1.0, abs_tol=1e-10):
        raise InvalidPredictionException(
            f"forward_regime_probs must sum to 1.0 +/- 1e-10, got {f_sum}"
        )

    # Tempered prior per regime j in {0, 1, 2} using log-space max-shift:
    # ln pi_{j, k} = alpha_t * ln(pi_{j, k} + 1e-30)
    # pi_{j, k}^tempered = exp(ln pi_{j, k} - max_m ln pi_{j, m}) / sum_l exp(ln pi_{j, l} - max_m ln pi_{j, m})
    log_pi = alpha_t * np.log(posteriors + 1e-30)
    max_log_pi = np.max(log_pi, axis=1, keepdims=True)
    exp_shifted = np.exp(log_pi - max_log_pi)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    updated_posteriors: np.ndarray = exp_shifted / sum_exp

    # Strictly enforce simplex normalization per row
    for r in range(3):
        r_sum = float(np.sum(updated_posteriors[r, :]))
        if not (math.isfinite(r_sum) and r_sum > 0.0):
            raise DegenerateEnsembleException(f"updated_posteriors row {r} collapsed")
        updated_posteriors[r, :] /= r_sum

    # Composite prior: bar{pi}_k = sum_{j=0}^2 p_{t+1|t, j} * pi_{j, k}^tempered
    composite_prior: np.ndarray = forward_regime_probs @ updated_posteriors
    composite_prior = np.clip(composite_prior, 0.0, 1.0)
    comp_sum = float(np.sum(composite_prior))
    if not (math.isfinite(comp_sum) and comp_sum > 0.0):
        raise DegenerateEnsembleException(
            "composite_prior collapsed to non-finite or non-positive sum"
        )
    composite_prior = composite_prior / comp_sum

    return updated_posteriors, composite_prior


class TikhonovCorrelationEstimator:
    """Tikhonov-regularized pairwise correlation estimator with flatline model defense.

    Computes empirical pairwise prediction/return correlation matrix C_hat in R^{K x K},
    defensively handles zero-variance/flatline models (clamping standard deviation sigma_k >= 1e-8
    and setting pairwise correlations for flatline models to 0.0 with diagonal 1.0),
    and applies Tikhonov ridge shrinkage:
        C_t = (1 - delta) * C_hat + delta * I_K
    with delta in (0.0, 1.0) (default 0.05), guaranteeing C_t is strictly positive definite
    with lambda_min >= delta.
    """

    def __init__(
        self,
        ridge_shrinkage: float = 0.05,
        min_std_dev: float = 1e-8,
    ) -> None:
        """Initialize the Tikhonov correlation estimator.

        Args:
            ridge_shrinkage: Ridge regularization parameter delta in (0.0, 1.0).
            min_std_dev: Defensive standard deviation floor sigma_min > 0.0 for flatline detection.

        Raises:
            EnsembleError: If ridge_shrinkage not in (0, 1) or min_std_dev <= 0.
            DegenerateEnsembleException: If parameters contain NaN or Inf.
        """
        if not (isinstance(ridge_shrinkage, (int, float)) and math.isfinite(ridge_shrinkage)):
            raise DegenerateEnsembleException(
                f"ridge_shrinkage must be a finite float, got {ridge_shrinkage}"
            )
        if not (0.0 < ridge_shrinkage < 1.0):
            raise EnsembleError(f"ridge_shrinkage must be in (0.0, 1.0), got {ridge_shrinkage}")

        if not (isinstance(min_std_dev, (int, float)) and math.isfinite(min_std_dev)):
            raise DegenerateEnsembleException(
                f"min_std_dev must be a finite float, got {min_std_dev}"
            )
        if min_std_dev <= 0.0:
            raise EnsembleError(f"min_std_dev must be > 0.0, got {min_std_dev}")

        self._ridge_shrinkage: float = float(ridge_shrinkage)
        self._min_std_dev: float = float(min_std_dev)

    @property
    def ridge_shrinkage(self) -> float:
        """Ridge shrinkage regularizer delta."""
        return self._ridge_shrinkage

    @property
    def min_std_dev(self) -> float:
        """Defensive standard deviation floor sigma_min."""
        return self._min_std_dev

    def compute_correlation_matrix(self, predictions: np.ndarray) -> np.ndarray:
        """Compute Tikhonov-regularized pairwise correlation matrix from predictions or returns.

        Args:
            predictions: 2D array of historical model predictions or returns of shape (N, K)
                where N >= 1 is sample size / time bars and K >= 1 is number of models.

        Returns:
            C: Regularized correlation matrix of shape (K, K), symmetric with unit diagonal
                and minimum eigenvalue lambda_min >= ridge_shrinkage.

        Raises:
            InvalidPredictionException: If predictions is not a 2D array or is empty.
            DegenerateEnsembleException: If predictions contains NaN or Inf.
        """
        if not isinstance(predictions, np.ndarray) or predictions.ndim != 2:
            raise InvalidPredictionException("predictions must be a 2D numpy array")
        n_samples, k_models = predictions.shape
        if n_samples < 1 or k_models < 1:
            raise InvalidPredictionException(
                f"predictions must have shape (N, K) with N >= 1, K >= 1, got {predictions.shape}"
            )
        if not np.all(np.isfinite(predictions)):
            raise DegenerateEnsembleException("predictions must contain only finite values")

        if n_samples == 1:
            # Single observation: zero sample variance for all models across time.
            # All models are flatline. Return identity matrix.
            return np.eye(k_models, dtype=np.float64)

        # Compute column means and standard deviations
        means = np.mean(predictions, axis=0)
        centered = predictions - means
        stds = np.std(predictions, axis=0)

        # Identify flatline / zero-variance models
        is_flatline = stds < self._min_std_dev

        # If all models are flatline, return identity matrix
        if np.all(is_flatline):
            return np.eye(k_models, dtype=np.float64)

        # Standardize non-flatline columns to unit vectors
        z = np.zeros_like(centered, dtype=np.float64)
        for k in range(k_models):
            if not is_flatline[k]:
                col_norm = float(np.linalg.norm(centered[:, k]))
                if col_norm >= self._min_std_dev:
                    z[:, k] = centered[:, k] / col_norm

        # Empirical correlation Gram matrix
        c_emp = z.T @ z

        # Defensive handling for flatline models:
        # Zero off-diagonals and set unit diagonal
        for k in range(k_models):
            if is_flatline[k]:
                c_emp[k, :] = 0.0
                c_emp[:, k] = 0.0
                c_emp[k, k] = 1.0

        # Enforce exact [-1, 1] range, unit diagonal, and symmetry
        c_emp = np.clip(c_emp, -1.0, 1.0)
        np.fill_diagonal(c_emp, 1.0)
        c_emp = 0.5 * (c_emp + c_emp.T)

        # Tikhonov regularizer / ridge shrinkage:
        # C_t = (1 - delta) * C_hat + delta * I_K
        eye = np.eye(k_models, dtype=np.float64)
        c_reg = (1.0 - self._ridge_shrinkage) * c_emp + self._ridge_shrinkage * eye
        np.fill_diagonal(c_reg, 1.0)
        c_reg = 0.5 * (c_reg + c_reg.T)

        return c_reg.astype(np.float64)

    def regularize_correlation_matrix(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Apply Tikhonov ridge shrinkage directly to an existing correlation matrix.

        Args:
            correlation_matrix: 2D square correlation matrix of shape (K, K).

        Returns:
            C: Regularized correlation matrix with lambda_min >= ridge_shrinkage.

        Raises:
            InvalidPredictionException: If matrix is not 2D square.
            DegenerateEnsembleException: If matrix contains NaN or Inf.
        """
        if not isinstance(correlation_matrix, np.ndarray) or correlation_matrix.ndim != 2:
            raise InvalidPredictionException("correlation_matrix must be a 2D numpy array")
        if (
            correlation_matrix.shape[0] != correlation_matrix.shape[1]
            or correlation_matrix.shape[0] < 1
        ):
            raise InvalidPredictionException(
                f"correlation_matrix must be square (K, K), got {correlation_matrix.shape}"
            )
        if not np.all(np.isfinite(correlation_matrix)):
            raise DegenerateEnsembleException("correlation_matrix must contain only finite values")

        k_models = correlation_matrix.shape[0]
        c_sym = 0.5 * (correlation_matrix + correlation_matrix.T)
        c_sym = np.clip(c_sym, -1.0, 1.0)
        np.fill_diagonal(c_sym, 1.0)

        eye = np.eye(k_models, dtype=np.float64)
        c_reg = (1.0 - self._ridge_shrinkage) * c_sym + self._ridge_shrinkage * eye
        np.fill_diagonal(c_reg, 1.0)
        c_reg = 0.5 * (c_reg + c_reg.T)

        return cast(np.ndarray, c_reg.astype(np.float64))


class OrthogonalityRegularizedSolver:
    """Entropic Mirror Descent optimizer on the probability simplex.

    Solves the clone-penalized quadratic objective:
        min_{w in Delta^K} { w^T s + (lambda_ortho / 2) * w^T C w - tau * H(w) }
    where s is the composite loss/score vector, C is the Tikhonov-regularized correlation matrix,
    and H(w) = -sum w_k ln(w_k) is Shannon entropy.

    Optimizes on the simplex Delta^K via exponentiated gradient descent with Log-Sum-Exp max shift,
    terminates on infinity-norm convergence (< tol) or max_iter iterations, and applies
    Laplace floor smoothing:
        w_k <- (1 - K * eps_floor) * w_k + eps_floor
    strictly upholding Invariant INV-ENS-001 (Strict Simplex Conservation).
    """

    def __init__(
        self,
        orthogonality_penalty: float = 0.25,
        temperature: float = 1.0,
        learning_rate: float = 0.50,
        max_iter: int = 10,
        tol: float = 1e-6,
        min_weight_floor: float = 1e-5,
        config: EnsembleConfig | None = None,
    ) -> None:
        """Initialize the OrthogonalityRegularizedSolver.

        Args:
            orthogonality_penalty: Penalty multiplier lambda_ortho >= 0.0 on model correlation.
            temperature: Entropy temperature tau > 0.0.
            learning_rate: Step size eta > 0.0 for mirror descent.
            max_iter: Maximum mirror descent iterations >= 1.
            tol: Infinity-norm convergence tolerance > 0.0.
            min_weight_floor: Minimum Laplace probability floor eps_floor > 0.0.
            config: Optional EnsembleConfig to inherit default hyperparameters.

        Raises:
            EnsembleError: If any parameter violates bounds or is invalid.
            DegenerateEnsembleException: If parameters contain NaN or Inf.
        """
        if config is not None:
            orthogonality_penalty = config.orthogonality_penalty
            temperature = config.temperature
            learning_rate = config.mirror_descent_lr
            max_iter = config.mirror_descent_max_iter
            tol = config.mirror_descent_tol
            min_weight_floor = config.min_weight_floor

        # Finiteness validations
        for name, val in [
            ("orthogonality_penalty", orthogonality_penalty),
            ("temperature", temperature),
            ("learning_rate", learning_rate),
            ("tol", tol),
            ("min_weight_floor", min_weight_floor),
        ]:
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise DegenerateEnsembleException(f"{name} must be a finite float, got {val}")

        if orthogonality_penalty < 0.0:
            raise EnsembleError(
                f"orthogonality_penalty must be >= 0.0, got {orthogonality_penalty}"
            )
        if temperature <= 0.0:
            raise EnsembleError(f"temperature must be > 0.0, got {temperature}")
        if learning_rate <= 0.0:
            raise EnsembleError(f"learning_rate must be > 0.0, got {learning_rate}")
        if not (isinstance(max_iter, int) and max_iter >= 1):
            raise EnsembleError(f"max_iter must be an integer >= 1, got {max_iter}")
        if tol <= 0.0:
            raise EnsembleError(f"tol must be > 0.0, got {tol}")
        if min_weight_floor <= 0.0:
            raise EnsembleError(f"min_weight_floor must be > 0.0, got {min_weight_floor}")

        self._orthogonality_penalty: float = float(orthogonality_penalty)
        self._temperature: float = float(temperature)
        self._learning_rate: float = float(learning_rate)
        self._max_iter: int = int(max_iter)
        self._tol: float = float(tol)
        self._min_weight_floor: float = float(min_weight_floor)

    @property
    def orthogonality_penalty(self) -> float:
        """Orthogonality penalty lambda_ortho."""
        return self._orthogonality_penalty

    @property
    def temperature(self) -> float:
        """Entropy temperature tau."""
        return self._temperature

    @property
    def learning_rate(self) -> float:
        """Mirror descent step size eta."""
        return self._learning_rate

    @property
    def max_iter(self) -> int:
        """Maximum mirror descent iterations."""
        return self._max_iter

    @property
    def tol(self) -> float:
        """Convergence tolerance."""
        return self._tol

    @property
    def min_weight_floor(self) -> float:
        """Minimum weight floor eps_floor."""
        return self._min_weight_floor

    def solve(
        self,
        scores: np.ndarray,
        correlation_matrix: np.ndarray,
        current_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve for optimal model weights on the simplex via Entropic Mirror Descent.

        Args:
            scores: 1D array of model loss scores s_t of shape (K,).
            correlation_matrix: 2D regularized correlation matrix C_t of shape (K, K).
            current_weights: Optional 1D warm-start weight vector w^{(0)} of shape (K,).
                If None, initialized uniformly to 1/K.

        Returns:
            w_star: Optimal weight allocation vector on Delta^K satisfying INV-ENS-001
                (sum=1.0 +/- 1e-10, w_k >= eps_floor > 0).

        Raises:
            InvalidPredictionException: If inputs fail dimension or simplex alignment.
            DegenerateEnsembleException: If inputs contain NaN or Inf, or optimization collapses.
        """
        # Defensive validation on scores
        if not isinstance(scores, np.ndarray) or scores.ndim != 1:
            raise InvalidPredictionException("scores must be a 1D numpy array")
        k_models = len(scores)
        if k_models < 1:
            raise InvalidPredictionException("scores must contain at least 1 model score")
        if not np.all(np.isfinite(scores)):
            raise DegenerateEnsembleException("scores must contain only finite values")

        # Defensive validation on correlation_matrix
        if not isinstance(correlation_matrix, np.ndarray) or correlation_matrix.ndim != 2:
            raise InvalidPredictionException("correlation_matrix must be a 2D numpy array")
        if correlation_matrix.shape != (k_models, k_models):
            raise InvalidPredictionException(
                f"correlation_matrix shape {correlation_matrix.shape} must match (K, K) with K={k_models}"
            )
        if not np.all(np.isfinite(correlation_matrix)):
            raise DegenerateEnsembleException("correlation_matrix must contain only finite values")

        # Defensive validation on current_weights
        if current_weights is not None:
            if not isinstance(current_weights, np.ndarray) or current_weights.ndim != 1:
                raise InvalidPredictionException("current_weights must be a 1D numpy array")
            if current_weights.shape != (k_models,):
                raise InvalidPredictionException(
                    f"current_weights shape {current_weights.shape} must match scores shape ({k_models},)"
                )
            if not np.all(np.isfinite(current_weights)):
                raise DegenerateEnsembleException("current_weights must contain only finite values")
            if np.any(current_weights < 0.0):
                raise InvalidPredictionException("current_weights elements must be non-negative")
            init_sum = float(np.sum(current_weights))
            if not (math.isfinite(init_sum) and init_sum > 0.0):
                raise InvalidPredictionException("current_weights sum must be strictly positive")
            w = (current_weights / init_sum).astype(np.float64, copy=True)
            w = np.maximum(w, 1e-30)
            w /= float(np.sum(w))
        else:
            w = np.full(k_models, 1.0 / float(k_models), dtype=np.float64)

        step_factor = self._learning_rate / self._temperature
        penalty = self._orthogonality_penalty
        has_penalty = penalty > 0.0

        scaled_scores = -step_factor * scores
        scaled_penalty = -step_factor * penalty

        cw = np.empty(k_models, dtype=np.float64)
        v = np.empty(k_models, dtype=np.float64)

        # Entropic Mirror Descent loop (exponentiated gradient with Log-Sum-Exp shift)
        for _ in range(self._max_iter):
            # Scaled mirror gradient: - (eta / tau) * (s + lambda_ortho * (C @ w))
            if has_penalty:
                np.dot(correlation_matrix, w, out=cw)
                np.multiply(cw, scaled_penalty, out=v)
                v += scaled_scores
            else:
                np.copyto(v, scaled_scores)

            # Mirror step: w_i^{(m+1)} proportional to w_i^{(m)} * exp(scaled_grad_i)
            v -= np.max(v)
            np.exp(v, out=v)
            v *= w
            denom = float(np.sum(v))
            if not (math.isfinite(denom) and denom > 0.0):
                raise DegenerateEnsembleException(
                    "Mirror descent step collapsed to non-finite or non-positive denominator"
                )
            v /= denom

            # Convergence check: L_infinity norm without temporary array allocations
            np.subtract(v, w, out=cw)
            np.abs(cw, out=cw)
            diff = float(np.max(cw))
            np.copyto(w, v)
            if diff < self._tol:
                break

        # Laplace floor regularization:
        # w_k <- (1 - K * eps_floor) * w_k + eps_floor
        eps_floor = self._min_weight_floor
        if eps_floor * float(k_models) >= 1.0:
            eps_floor = 0.001 / float(k_models)
        if eps_floor * float(k_models) >= 1.0:
            eps_floor = 0.5 / float(k_models)

        w = (1.0 - float(k_models) * eps_floor) * w + eps_floor

        # Normalize and enforce INV-ENS-001
        w_sum = float(np.sum(w))
        if not (math.isfinite(w_sum) and w_sum > 0.0):
            raise DegenerateEnsembleException("Laplace smoothing produced invalid sum")
        w = w / w_sum

        if not np.all(w > 0.0):
            raise DegenerateEnsembleException(
                "INV-ENS-001 violation: weights must be strictly positive"
            )
        if not math.isclose(float(np.sum(w)), 1.0, abs_tol=1e-10):
            raise DegenerateEnsembleException(
                f"INV-ENS-001 violation: weights sum to {np.sum(w)}, expected 1.0 +/- 1e-10"
            )

        return w.astype(np.float64)


__all__ = [
    "EnsembleError",
    "DegenerateEnsembleException",
    "InvalidPredictionException",
    "EnsembleConfig",
    "EnsembleState",
    "EnsemblePrediction",
    "VolatilityAdaptiveForgetting",
    "AsymmetricDownsideLossScorer",
    "predict_forward_regime_prior",
    "update_regime_posteriors",
    "TikhonovCorrelationEstimator",
    "OrthogonalityRegularizedSolver",
]
