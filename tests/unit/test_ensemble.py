"""Unit tests for Regime-Conditioned Dynamic Model Averaging (RD-DMA) domain entities.

Validates Invariants:
- INV-ENS-001: Strict Simplex Conservation on weights.
- INV-ENS-002: Variance Positivity and Additivity.
- INV-ENS-003: Bounded Adaptive Forgetting hyperparameter bounds.
"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quant.analytics.ensemble import (
    DegenerateEnsembleException,
    EnsembleConfig,
    EnsembleError,
    EnsemblePrediction,
    EnsembleState,
    InvalidPredictionException,
)


class TestDomainEntitiesAndInvariants:
    """Tests for EnsembleConfig, EnsembleState, EnsemblePrediction, and exceptions."""

    def test_custom_exception_hierarchy(self) -> None:
        """Verify exception inheritance tree."""
        assert issubclass(EnsembleError, Exception)
        assert issubclass(DegenerateEnsembleException, EnsembleError)
        assert issubclass(InvalidPredictionException, EnsembleError)

        err = DegenerateEnsembleException("variance breach")
        assert isinstance(err, EnsembleError)

    def test_ensemble_config_defaults_and_validation(self) -> None:
        """Verify EnsembleConfig defaults and invariant validation in __post_init__."""
        cfg = EnsembleConfig()
        assert cfg.base_forgetting_factor == 0.96
        assert cfg.volatility_sensitivity == 0.50
        assert cfg.min_forgetting_factor == 0.85
        assert cfg.max_forgetting_factor == 0.99
        assert cfg.orthogonality_penalty == 0.25
        assert cfg.downside_penalty == 2.50
        assert cfg.turnover_damping == 0.15
        assert cfg.ridge_shrinkage == 0.05
        assert cfg.ambiguity_shrinkage_cap == 0.50
        assert cfg.min_weight_floor == 1e-5
        assert cfg.temperature == 1.0
        assert cfg.mirror_descent_lr == 0.50
        assert cfg.mirror_descent_max_iter == 10
        assert cfg.mirror_descent_tol == 1e-6

        # Invariant INV-ENS-003 & boundary checks
        # base_forgetting_factor in (0.80, 0.999)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=1.5)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=0.75)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=0.80)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=0.999)

        # min_forgetting_factor > 0.50 and < base
        with pytest.raises(EnsembleError, match="min_forgetting_factor"):
            EnsembleConfig(min_forgetting_factor=0.40)
        with pytest.raises(EnsembleError, match="min_forgetting_factor"):
            EnsembleConfig(min_forgetting_factor=0.98, base_forgetting_factor=0.96)

        # max_forgetting_factor > base and < 1.0
        with pytest.raises(EnsembleError, match="max_forgetting_factor"):
            EnsembleConfig(max_forgetting_factor=0.90, base_forgetting_factor=0.96)
        with pytest.raises(EnsembleError, match="max_forgetting_factor"):
            EnsembleConfig(max_forgetting_factor=1.0)

        # min >= max
        with pytest.raises(EnsembleError):
            EnsembleConfig(min_forgetting_factor=0.98, max_forgetting_factor=0.90)

        # volatility_sensitivity >= 0.0
        with pytest.raises(EnsembleError, match="volatility_sensitivity"):
            EnsembleConfig(volatility_sensitivity=-0.1)

        # orthogonality_penalty >= 0.0
        with pytest.raises(EnsembleError, match="orthogonality_penalty"):
            EnsembleConfig(orthogonality_penalty=-0.1)

        # downside_penalty >= 0.0
        with pytest.raises(EnsembleError, match="downside_penalty"):
            EnsembleConfig(downside_penalty=-0.5)

        # turnover_damping in [0.0, 1.0)
        with pytest.raises(EnsembleError, match="turnover_damping"):
            EnsembleConfig(turnover_damping=-0.01)
        with pytest.raises(EnsembleError, match="turnover_damping"):
            EnsembleConfig(turnover_damping=1.0)
        with pytest.raises(EnsembleError, match="turnover_damping"):
            EnsembleConfig(turnover_damping=1.2)

        # ridge_shrinkage in (0.0, 1.0)
        with pytest.raises(EnsembleError, match="ridge_shrinkage"):
            EnsembleConfig(ridge_shrinkage=0.0)
        with pytest.raises(EnsembleError, match="ridge_shrinkage"):
            EnsembleConfig(ridge_shrinkage=1.0)

        # ambiguity_shrinkage_cap in [0.0, 1.0]
        with pytest.raises(EnsembleError, match="ambiguity_shrinkage_cap"):
            EnsembleConfig(ambiguity_shrinkage_cap=-0.1)
        with pytest.raises(EnsembleError, match="ambiguity_shrinkage_cap"):
            EnsembleConfig(ambiguity_shrinkage_cap=1.05)

        # min_weight_floor > 0.0
        with pytest.raises(EnsembleError, match="min_weight_floor"):
            EnsembleConfig(min_weight_floor=0.0)

        # temperature > 0.0
        with pytest.raises(EnsembleError, match="temperature"):
            EnsembleConfig(temperature=-1.0)

        # mirror_descent_lr > 0.0
        with pytest.raises(EnsembleError, match="mirror_descent_lr"):
            EnsembleConfig(mirror_descent_lr=0.0)

        # mirror_descent_max_iter >= 1
        with pytest.raises(EnsembleError, match="mirror_descent_max_iter"):
            EnsembleConfig(mirror_descent_max_iter=0)

        # mirror_descent_tol > 0.0
        with pytest.raises(EnsembleError, match="mirror_descent_tol"):
            EnsembleConfig(mirror_descent_tol=0.0)

        # Non-finite and type validation
        with pytest.raises(EnsembleError):
            EnsembleConfig(base_forgetting_factor=float("nan"))
        with pytest.raises(EnsembleError):
            EnsembleConfig(volatility_sensitivity=float("inf"))

    def test_ensemble_state_immutability_and_validation(self) -> None:
        """Verify EnsembleState initialization, immutability, and INV-ENS-001."""
        K = 5
        weights = np.full(K, 1.0 / K)
        posteriors = np.full((3, K), 1.0 / K)
        losses = np.zeros(K)
        state = EnsembleState(
            step_index=0,
            weights=weights,
            regime_conditional_posteriors=posteriors,
            cumulative_losses=losses,
            effective_models=float(K),
            mean_realized_volatility=0.015,
            last_ambiguity_temperature=1.0,
        )
        assert state.step_index == 0
        assert state.effective_models == pytest.approx(float(K))
        assert np.isclose(np.sum(state.weights), 1.0)
        assert state.weights.shape == (K,)
        assert state.regime_conditional_posteriors.shape == (3, K)
        assert state.cumulative_losses.shape == (K,)

        # Immutability: frozen dataclass
        with pytest.raises(FrozenInstanceError):
            state.step_index = 1  # type: ignore[misc]

        # Invariant INV-ENS-001: simplex sum breach
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=np.array([0.5, 0.2]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=2.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Invariant INV-ENS-001: weights strictly positive
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=np.array([1.0, 0.0]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=1.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=np.array([1.1, -0.1]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=1.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-array or bad dimension weights
        with pytest.raises(EnsembleError, match="1D numpy array"):
            EnsembleState(
                step_index=0,
                weights=[0.5, 0.5],  # type: ignore[arg-type]
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=2.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(EnsembleError, match="at least 1 model"):
            EnsembleState(
                step_index=0,
                weights=np.array([]),
                regime_conditional_posteriors=np.zeros((3, 0)),
                cumulative_losses=np.zeros(0),
                effective_models=1.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=np.array([np.nan, 1.0]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=2.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # step_index >= 0
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=-1,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Dimension mismatch between posteriors and weights
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=np.full((3, K - 1), 1.0 / (K - 1)),
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-array posteriors
        with pytest.raises(EnsembleError, match="numpy array"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=[[1.0 / K] * K] * 3,  # type: ignore[arg-type]
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-finite posteriors
        bad_nan_posteriors = np.copy(posteriors)
        bad_nan_posteriors[0, 0] = np.nan
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=bad_nan_posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Negative posteriors
        bad_neg_posteriors = np.copy(posteriors)
        bad_neg_posteriors[0, 0] = -0.1
        bad_neg_posteriors[0, 1] += 0.1 + (1.0 / K)
        with pytest.raises(EnsembleError, match="non-negative"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=bad_neg_posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Posterior rows must sum to 1.0
        bad_posteriors = np.copy(posteriors)
        bad_posteriors[0, :] = 0.5
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=bad_posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Dimension mismatch: cumulative_losses
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=np.zeros(K + 1),
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-array cumulative_losses
        with pytest.raises(EnsembleError, match="numpy array"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=list(losses),  # type: ignore[arg-type]
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-finite cumulative_losses
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=np.array([np.inf] + [0.0] * (K - 1)),
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # effective_models bounds [1.0, K]
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=0.5,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K + 1),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float("nan"),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # mean_realized_volatility > 0.0
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.0,
                last_ambiguity_temperature=1.0,
            )

        # last_ambiguity_temperature > 0.0
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=-0.5,
            )

    def test_ensemble_prediction_variance_additivity(self) -> None:
        """Verify EnsemblePrediction and invariant INV-ENS-002 (variance additivity)."""
        pred = EnsemblePrediction(
            point_prediction=0.0025,
            aleatoric_variance=0.0004,
            epistemic_variance=0.0001,
            total_variance=0.0005,
            model_weights=np.array([0.5, 0.5]),
            regime_probabilities=np.array([0.7, 0.2, 0.1]),
            effective_models=2.0,
            volatility_forgetting_factor=0.96,
            ambiguity_shrinkage_weight=0.0,
        )
        # Invariant INV-ENS-002: total == aleatoric + epistemic
        assert pred.total_variance == pytest.approx(
            pred.aleatoric_variance + pred.epistemic_variance
        )
        assert math.isclose(
            pred.total_variance,
            pred.aleatoric_variance + pred.epistemic_variance,
            rel_tol=1e-7,
            abs_tol=1e-10,
        )

        # NaN rejection
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=np.nan,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Inf rejection
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=float("inf"),
                epistemic_variance=0.0001,
                total_variance=float("inf"),
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Invariant INV-ENS-002: aleatoric_variance must be > 0
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0,
                epistemic_variance=0.0001,
                total_variance=0.0001,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Invariant INV-ENS-002: epistemic_variance must be >= 0
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=-0.0001,
                total_variance=0.0003,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Invariant INV-ENS-002: variance additivity breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0010,  # Should be 0.0005
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Model weights type, empty, non-finite
        with pytest.raises(DegenerateEnsembleException, match="1D numpy array"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=[0.5, 0.5],  # type: ignore[arg-type]
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="length >= 1"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=1.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([np.nan, 1.0]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Model weights simplex breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.6, 0.6]),  # sum != 1.0
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Model weights negative component breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([1.1, -0.1]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Regime probabilities shape, non-finite, negative, sum breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.5, 0.5]),  # length != 3
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([np.nan, 0.5, 0.5]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="non-negative"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([-0.1, 0.6, 0.5]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.5, 0.3, 0.1]),  # sum != 1.0
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Effective models bounds check
        with pytest.raises(DegenerateEnsembleException, match="effective_models"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=0.5,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="effective_models"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=3.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # volatility_forgetting_factor in (0.0, 1.0)
        with pytest.raises(DegenerateEnsembleException, match="volatility_forgetting_factor"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.0,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="volatility_forgetting_factor"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=1.0,
                ambiguity_shrinkage_weight=0.0,
            )

        # ambiguity_shrinkage_weight in [0.0, 1.0]
        with pytest.raises(DegenerateEnsembleException, match="ambiguity_shrinkage_weight"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=-0.1,
            )
        with pytest.raises(DegenerateEnsembleException, match="ambiguity_shrinkage_weight"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=1.1,
            )
