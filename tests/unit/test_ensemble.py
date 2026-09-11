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
    AsymmetricDownsideLossScorer,
    DegenerateEnsembleException,
    EnsembleConfig,
    EnsembleError,
    EnsemblePrediction,
    EnsembleState,
    InvalidPredictionException,
    OrthogonalityRegularizedSolver,
    RegimeConditionedDMAEngine,
    TikhonovCorrelationEstimator,
    VolatilityAdaptiveForgetting,
    predict_forward_regime_prior,
    update_regime_posteriors,
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


class TestAdaptiveForgetting:
    """Unit tests for VolatilityAdaptiveForgetting enforcing Invariant INV-ENS-003."""

    def test_init_validation(self) -> None:
        """Verify constructor validates initial volatility and smoothing parameter."""
        cfg = EnsembleConfig()

        # Valid initialization
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.015, vol_smoothing=0.10)
        assert vaf.mean_realized_volatility == pytest.approx(0.015)

        # Invalid initial_mean_vol <= 0 or non-finite
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.0)
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=-0.01)
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=float("nan"))
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=float("inf"))

        # Invalid vol_smoothing not in (0, 1] or non-finite
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=0.0)
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=-0.1)
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=1.05)
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=float("nan"))

    def test_alpha_expansion_in_calm_markets(self) -> None:
        """Verify memory expands (alpha -> alpha_max) when realized vol is below baseline."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.02, vol_smoothing=0.10)

        # Calm market: realized_vol = 0.01 < baseline 0.02
        # delta_sigma = (0.01 - 0.02) / 0.02 = -0.50
        # raw_alpha = 0.96 - 0.50 * (-0.50) = 0.96 + 0.25 = 1.21 -> clamped to 0.99
        alpha = vaf.compute_alpha(realized_vol=0.01)
        assert alpha > cfg.base_forgetting_factor
        assert alpha == pytest.approx(cfg.max_forgetting_factor)

    def test_alpha_contraction_during_panic_shocks(self) -> None:
        """Verify memory contracts (alpha -> alpha_min) during volatility panic spikes."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.01, vol_smoothing=0.10)

        # Panic shock: realized_vol = 0.04 >> baseline 0.01
        # delta_sigma = (0.04 - 0.01) / 0.01 = 3.0
        # raw_alpha = 0.96 - 0.50 * 3.0 = 0.96 - 1.50 = -0.54 -> clamped to 0.85
        alpha = vaf.compute_alpha(realized_vol=0.04)
        assert alpha < cfg.base_forgetting_factor
        assert alpha == pytest.approx(cfg.min_forgetting_factor)

    def test_exact_invariant_clamping_inv_ens_003(self) -> None:
        """Verify INV-ENS-003: alpha strictly remains in [alpha_min, alpha_max] under extreme inputs."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.01)

        # Extremely low vol: 1e-8
        alpha_low = vaf.compute_alpha(1e-8)
        assert alpha_low == pytest.approx(cfg.max_forgetting_factor)

        # Extremely high vol: 100.0
        alpha_high = vaf.compute_alpha(100.0)
        assert alpha_high == pytest.approx(cfg.min_forgetting_factor)

        # Exactly at baseline: delta_sigma = 0 -> alpha == base_forgetting_factor
        current_baseline = vaf.mean_realized_volatility
        alpha_base = vaf.compute_alpha(current_baseline)
        assert alpha_base == pytest.approx(cfg.base_forgetting_factor)

    def test_running_volatility_baseline_exponential_smoothing(self) -> None:
        """Verify baseline volatility updates via exponential smoothing after alpha computation."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        beta = 0.20
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.010, vol_smoothing=beta)

        # Step 1: realized_vol = 0.015
        # Expected new baseline: (1 - 0.20) * 0.010 + 0.20 * 0.015 = 0.008 + 0.003 = 0.011
        vaf.compute_alpha(0.015)
        assert vaf.mean_realized_volatility == pytest.approx(0.011)

        # Step 2: realized_vol = 0.021
        # Expected new baseline: (1 - 0.20) * 0.011 + 0.20 * 0.021 = 0.0088 + 0.0042 = 0.013
        vaf.compute_alpha(0.021)
        assert vaf.mean_realized_volatility == pytest.approx(0.013)

    def test_compute_alpha_invalid_inputs(self) -> None:
        """Verify defensive handling of invalid realized volatility inputs."""
        cfg = EnsembleConfig()
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.01)

        # Realized vol <= 0
        with pytest.raises(EnsembleError, match="realized_vol"):
            vaf.compute_alpha(0.0)
        with pytest.raises(EnsembleError, match="realized_vol"):
            vaf.compute_alpha(-0.05)

        # Realized vol NaN or Inf
        with pytest.raises(DegenerateEnsembleException, match="realized_vol"):
            vaf.compute_alpha(float("nan"))
        with pytest.raises(DegenerateEnsembleException, match="realized_vol"):
            vaf.compute_alpha(float("inf"))


class TestPredictiveForwardMarkovTransitions:
    """Unit tests for predict_forward_regime_prior and simplex conservation."""

    def test_predictive_forward_markov_projection(self) -> None:
        """Verify forward transition calculation p_{t+1|t} = P_trans^T * p_t."""
        # Row-stochastic transition matrix: P_trans[i, j] = P(S_{t+1}=j | S_t=i)
        P_trans = np.array(
            [
                [0.80, 0.15, 0.05],
                [0.10, 0.70, 0.20],
                [0.05, 0.05, 0.90],
            ]
        )
        p_current = np.array([0.10, 0.80, 0.10])  # Mostly Momentum (regime 1)

        # Expected:
        # p_next[0] = 0.80 * 0.10 + 0.10 * 0.80 + 0.05 * 0.10 = 0.165
        # p_next[1] = 0.15 * 0.10 + 0.70 * 0.80 + 0.05 * 0.10 = 0.580
        # p_next[2] = 0.05 * 0.10 + 0.20 * 0.80 + 0.90 * 0.10 = 0.255
        p_next = predict_forward_regime_prior(p_current, P_trans)

        assert p_next.shape == (3,)
        assert np.isclose(np.sum(p_next), 1.0, atol=1e-10)
        assert p_next[0] == pytest.approx(0.165)
        assert p_next[1] == pytest.approx(0.580)
        assert p_next[2] == pytest.approx(0.255)
        # Momentum regime transition risk increases panic probability
        assert p_next[2] > p_current[2]

    def test_identity_transition_preserves_distribution(self) -> None:
        """Verify identity transition matrix leaves regime distribution unchanged."""
        P_identity = np.eye(3)
        p_current = np.array([0.25, 0.50, 0.25])
        p_next = predict_forward_regime_prior(p_current, P_identity)

        assert np.allclose(p_next, p_current)
        assert np.isclose(np.sum(p_next), 1.0)

    def test_predict_forward_regime_prior_validation(self) -> None:
        """Verify defensive validation on current_regime_probs and transition_matrix."""
        valid_p = np.array([0.70, 0.20, 0.10])
        valid_P = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )

        # Invalid type or shape for current_regime_probs
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior([0.7, 0.2, 0.1], valid_P)  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([0.5, 0.5]), valid_P)
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.zeros((3, 1)), valid_P)

        # Non-finite or negative current_regime_probs
        with pytest.raises(DegenerateEnsembleException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([np.nan, 0.5, 0.5]), valid_P)
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([-0.1, 0.6, 0.5]), valid_P)

        # Sum != 1.0 for current_regime_probs
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([0.5, 0.5, 0.5]), valid_P)

        # Invalid type or shape for transition_matrix
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            predict_forward_regime_prior(valid_p, valid_P.tolist())  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            predict_forward_regime_prior(valid_p, np.eye(2))
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            predict_forward_regime_prior(valid_p, np.zeros((3, 3, 1)))

        # Non-finite or negative transition_matrix
        with pytest.raises(DegenerateEnsembleException, match="transition_matrix"):
            bad_nan_P = np.copy(valid_P)
            bad_nan_P[0, 0] = np.nan
            predict_forward_regime_prior(valid_p, bad_nan_P)

        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            bad_neg_P = np.copy(valid_P)
            bad_neg_P[0, 0] = -0.1
            bad_neg_P[0, 1] += 0.1
            predict_forward_regime_prior(valid_p, bad_neg_P)

        # Row sum != 1.0 for transition_matrix
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            bad_sum_P = np.copy(valid_P)
            bad_sum_P[0, :] = [0.5, 0.1, 0.1]
            predict_forward_regime_prior(valid_p, bad_sum_P)


class TestUpdateRegimePosteriors:
    """Unit tests for update_regime_posteriors enforcing tempered update and INV-ENS-001."""

    def test_update_regime_posteriors_tempering_and_composition(self) -> None:
        """Verify tempered posteriors update and composite prior calculation."""
        K = 4
        # 3 regimes x 4 models
        posteriors = np.array(
            [
                [0.40, 0.30, 0.20, 0.10],  # Regime 0 (Absorption)
                [0.10, 0.50, 0.30, 0.10],  # Regime 1 (Momentum)
                [0.10, 0.10, 0.20, 0.60],  # Regime 2 (Panic)
            ]
        )
        forward_p = np.array([0.20, 0.30, 0.50])
        alpha_t = 0.95

        updated_posteriors, composite_prior = update_regime_posteriors(
            posteriors, alpha_t, forward_p
        )

        # Validate shapes
        assert updated_posteriors.shape == (3, K)
        assert composite_prior.shape == (K,)

        # Validate INV-ENS-001 (Simplex conservation on each row and composite prior)
        for row_idx in range(3):
            row = updated_posteriors[row_idx, :]
            assert np.all(row > 0.0)
            assert np.isclose(np.sum(row), 1.0, atol=1e-10)

        assert np.all(composite_prior > 0.0)
        assert np.isclose(np.sum(composite_prior), 1.0, atol=1e-10)

        # In Panic regime (weight 0.50), model 3 is dominant (0.60 prior)
        # Therefore model 3 should have a significant share in the composite prior
        assert composite_prior[3] > composite_prior[0]

    def test_alpha_tempering_entropy_effect(self) -> None:
        """Verify lower alpha (higher forgetting) flattens distribution toward uniform (increases entropy)."""
        K = 3
        # Skewed posteriors
        posteriors = np.array(
            [
                [0.80, 0.15, 0.05],
                [0.80, 0.15, 0.05],
                [0.80, 0.15, 0.05],
            ]
        )
        forward_p = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        # High alpha (memory retention, minimal flattening)
        up_high, comp_high = update_regime_posteriors(
            posteriors, alpha_t=0.99, forward_regime_probs=forward_p
        )
        assert up_high.shape == (3, K)
        assert comp_high.shape == (K,)

        # Low alpha (rapid adaptation, significant flattening / tempering)
        up_low, comp_low = update_regime_posteriors(
            posteriors, alpha_t=0.85, forward_regime_probs=forward_p
        )
        assert up_low.shape == (3, K)
        assert comp_low.shape == (K,)

        # Top model weight should be higher with alpha=0.99 than with alpha=0.85
        assert comp_high[0] > comp_low[0]
        # Trailing model weight should be higher under lower alpha (tempered toward uniform)
        assert comp_low[2] > comp_high[2]

    def test_numerical_stability_with_extreme_likelihoods(self) -> None:
        """Verify log-space max-shift prevents underflow/overflow with near-zero priors."""
        K = 3
        posteriors = np.array(
            [
                [1.0 - 1e-15, 5e-16, 5e-16],
                [1e-25, 1.0 - 2e-25, 1e-25],
                [1e-30, 1e-30, 1.0 - 2e-30],
            ]
        )
        forward_p = np.array([0.33, 0.33, 0.34])
        alpha_t = 0.90

        updated_posteriors, composite_prior = update_regime_posteriors(
            posteriors, alpha_t, forward_p
        )

        assert updated_posteriors.shape == (3, K)
        assert composite_prior.shape == (K,)
        assert np.all(np.isfinite(updated_posteriors))
        assert np.all(np.isfinite(composite_prior))
        assert np.all(updated_posteriors >= 0.0)
        assert np.all(composite_prior >= 0.0)
        for r in range(3):
            assert np.isclose(np.sum(updated_posteriors[r, :]), 1.0, atol=1e-10)
        assert np.isclose(np.sum(composite_prior), 1.0, atol=1e-10)

    def test_update_regime_posteriors_validation(self) -> None:
        """Verify defensive checks on posteriors, alpha_t, and forward_regime_probs."""
        valid_post = np.full((3, 4), 0.25)
        valid_fp = np.array([0.5, 0.3, 0.2])
        valid_alpha = 0.96

        # Invalid posteriors type / dimensions
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            update_regime_posteriors([[0.25] * 4] * 3, valid_alpha, valid_fp)  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            update_regime_posteriors(np.full((2, 4), 0.25), valid_alpha, valid_fp)
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            update_regime_posteriors(np.zeros((3, 0)), valid_alpha, valid_fp)

        # Non-finite posteriors
        with pytest.raises(DegenerateEnsembleException, match="posteriors"):
            bad_post = np.copy(valid_post)
            bad_post[0, 0] = np.nan
            update_regime_posteriors(bad_post, valid_alpha, valid_fp)

        # Negative posteriors
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            bad_post = np.copy(valid_post)
            bad_post[0, 0] = -0.1
            bad_post[0, 1] += 0.1
            update_regime_posteriors(bad_post, valid_alpha, valid_fp)

        # Row sum != 1.0 in posteriors
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            bad_post = np.copy(valid_post)
            bad_post[0, :] = 0.20
            update_regime_posteriors(bad_post, valid_alpha, valid_fp)

        # Invalid alpha_t
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, float("nan"), valid_fp)
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, 0.0, valid_fp)
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, 1.2, valid_fp)
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, 0.40, valid_fp, alpha_min=0.50)

        # Invalid forward_regime_probs
        with pytest.raises(InvalidPredictionException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([0.5, 0.5]))
        with pytest.raises(DegenerateEnsembleException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([np.nan, 0.5, 0.5]))
        with pytest.raises(InvalidPredictionException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([-0.1, 0.6, 0.5]))
        with pytest.raises(InvalidPredictionException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([0.4, 0.4, 0.4]))


class TestAsymmetricDownsideLossScorer:
    """Unit tests for AsymmetricDownsideLossScorer and downside semi-variance."""

    def test_init_validation(self) -> None:
        """Verify default and custom initialization contracts for AsymmetricDownsideLossScorer."""
        scorer = AsymmetricDownsideLossScorer()
        assert scorer.downside_penalty == 2.50

        custom_scorer = AsymmetricDownsideLossScorer(downside_penalty=1.75)
        assert custom_scorer.downside_penalty == 1.75

        # Non-negative validation
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty=-0.1)

        # Finiteness validation
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty=float("nan"))
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty=float("inf"))
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty="not_a_number")  # type: ignore[arg-type]

    def test_compute_losses_symmetric_when_signs_agree(self) -> None:
        """Verify when signs agree (y_t * y_tilde >= 0), loss is exact squared error without downside penalty."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=2.50)

        # Case 1: Both positive
        y_realized = 0.04
        predictions = np.array([0.01, 0.04, 0.08])
        losses = scorer.compute_losses(predictions, y_realized)
        expected = (y_realized - predictions) ** 2
        assert np.allclose(losses, expected)

        # Case 2: Both negative
        y_realized_neg = -0.03
        predictions_neg = np.array([-0.01, -0.03, -0.05])
        losses_neg = scorer.compute_losses(predictions_neg, y_realized_neg)
        expected_neg = (y_realized_neg - predictions_neg) ** 2
        assert np.allclose(losses_neg, expected_neg)

        # Case 3: Zero boundary
        losses_zero = scorer.compute_losses(np.array([0.0, 0.05]), realized_return=0.0)
        assert np.allclose(losses_zero, np.array([0.0, 0.0025]))

    def test_compute_losses_asymmetric_downside_penalty(self) -> None:
        """Verify directional drawdown penalty amplifies loss when y_t * y_tilde < 0."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=2.50)

        y_realized = -0.05
        # Model 0: False long (+0.05)
        # Model 1: Correct short (-0.05)
        # Model 2: Short with same absolute error as Model 0 (-0.15, diff = 0.10)
        predictions = np.array([0.05, -0.05, -0.15])
        losses = scorer.compute_losses(predictions, y_realized)

        # Model 0:
        # squared_error = (-0.05 - 0.05)^2 = 0.01
        # asymmetric penalty = 2.50 * max(0, -(-0.05) * 0.05) = 2.50 * 0.0025 = 0.00625
        # total = 0.01625
        assert losses[0] == pytest.approx(0.01625)

        # Model 1:
        # squared_error = (-0.05 - (-0.05))^2 = 0.0
        # asymmetric penalty = 0.0
        assert losses[1] == pytest.approx(0.0)

        # Model 2:
        # squared_error = (-0.05 - (-0.15))^2 = (0.10)^2 = 0.01
        # asymmetric penalty = 0.0 (both signs negative)
        assert losses[2] == pytest.approx(0.01)

        # Crucial Invariant: Directionally wrong prediction (Model 0) receives strictly higher loss
        # than sign-agreeing prediction with same error magnitude (Model 2)
        assert losses[0] > losses[2]
        assert losses[0] - losses[2] == pytest.approx(0.00625)

    def test_compute_losses_multi_horizon_scaling(self) -> None:
        """Verify forecast horizon scaling y_tilde = y_hat / sqrt(H) normalizes predictions."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=2.50)

        # Horizons: H = [1, 4, 9], sqrt(H) = [1, 2, 3]
        horizons = np.array([1.0, 4.0, 9.0])
        nominal_predictions = np.array([0.04, 0.08, 0.12])
        # Normalized predictions: [0.04/1, 0.08/2, 0.12/3] = [0.04, 0.04, 0.04]
        y_realized = 0.04

        losses = scorer.compute_losses(nominal_predictions, y_realized, forecast_horizons=horizons)
        assert losses.shape == (3,)
        # All models have scaled prediction matching realized return exactly -> 0 loss
        assert np.allclose(losses, np.zeros(3))

        # Realized return = -0.04 with horizons
        losses_drawdown = scorer.compute_losses(
            nominal_predictions, -0.04, forecast_horizons=horizons
        )
        # All models should produce identical normalized loss
        assert math.isclose(losses_drawdown[0], losses_drawdown[1])
        assert math.isclose(losses_drawdown[1], losses_drawdown[2])

    def test_compute_losses_non_negative_invariant(self) -> None:
        """Verify Invariant: loss l_{t, k} >= 0.0 across random realizations."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=5.0)
        rng = np.random.default_rng(42)

        for _ in range(20):
            preds = rng.normal(0.0, 0.05, size=10)
            y_realized = float(rng.normal(0.0, 0.05))
            horizons = rng.uniform(1.0, 20.0, size=10)

            losses = scorer.compute_losses(preds, y_realized, forecast_horizons=horizons)
            assert losses.shape == (10,)
            assert np.all(losses >= 0.0)
            assert np.all(np.isfinite(losses))

    def test_compute_losses_validation(self) -> None:
        """Verify defensive input validation on compute_losses."""
        scorer = AsymmetricDownsideLossScorer()
        valid_preds = np.array([0.01, 0.02, 0.03])
        valid_y = 0.015

        # Invalid predictions
        with pytest.raises(InvalidPredictionException, match="predictions"):
            scorer.compute_losses([0.01, 0.02], valid_y)  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="predictions"):
            scorer.compute_losses(np.array([[0.01, 0.02]]), valid_y)
        with pytest.raises(InvalidPredictionException, match="predictions"):
            scorer.compute_losses(np.array([]), valid_y)
        with pytest.raises(DegenerateEnsembleException, match="predictions"):
            scorer.compute_losses(np.array([0.01, np.nan]), valid_y)
        with pytest.raises(DegenerateEnsembleException, match="predictions"):
            scorer.compute_losses(np.array([0.01, float("inf")]), valid_y)

        # Invalid realized_return
        with pytest.raises(DegenerateEnsembleException, match="realized_return"):
            scorer.compute_losses(valid_preds, float("nan"))
        with pytest.raises(DegenerateEnsembleException, match="realized_return"):
            scorer.compute_losses(valid_preds, float("inf"))

        # Invalid forecast_horizons
        with pytest.raises(InvalidPredictionException, match="forecast_horizons"):
            scorer.compute_losses(valid_preds, valid_y, forecast_horizons=[1.0, 1.0, 1.0])  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="forecast_horizons"):
            scorer.compute_losses(valid_preds, valid_y, forecast_horizons=np.array([1.0, 2.0]))
        with pytest.raises(InvalidPredictionException, match="forecast_horizons"):
            scorer.compute_losses(valid_preds, valid_y, forecast_horizons=np.array([1.0, 0.5, 2.0]))
        with pytest.raises(DegenerateEnsembleException, match="forecast_horizons"):
            scorer.compute_losses(
                valid_preds, valid_y, forecast_horizons=np.array([1.0, np.nan, 2.0])
            )

    def test_compute_downside_semi_variance(self) -> None:
        """Verify empirical downside semi-variance isolates negative deviations."""
        scorer = AsymmetricDownsideLossScorer()

        # Returns: [-0.02, 0.04, -0.04, 0.06]
        # Target = 0.0
        # Deviations: min(0, r - 0) = [-0.02, 0.0, -0.04, 0.0]
        # Squared: [0.0004, 0.0, 0.0016, 0.0] -> sum = 0.0020
        # Mean = 0.0020 / 4 = 0.0005
        returns = np.array([-0.02, 0.04, -0.04, 0.06])
        semi_var = scorer.compute_downside_semi_variance(returns, target_return=0.0)
        assert semi_var == pytest.approx(0.0005)

        # Target = 0.01
        # Deviations: [min(0, -0.02 - 0.01), min(0, 0.04 - 0.01), min(0, -0.04 - 0.01), min(0, 0.06 - 0.01)]
        #           = [-0.03, 0.0, -0.05, 0.0]
        # Squared: [0.0009, 0.0, 0.0025, 0.0] -> sum = 0.0034
        # Mean = 0.0034 / 4 = 0.00085
        semi_var_target = scorer.compute_downside_semi_variance(returns, target_return=0.01)
        assert semi_var_target == pytest.approx(0.00085)

    def test_compute_downside_semi_variance_pure_upside_floor(self) -> None:
        """Verify pure upside returns trigger defensive variance floor > 0.0."""
        scorer = AsymmetricDownsideLossScorer()
        pure_upside = np.array([0.02, 0.05, 0.01, 0.08])

        floor = 1e-8
        semi_var = scorer.compute_downside_semi_variance(
            pure_upside, target_return=0.0, min_variance_floor=floor
        )
        assert semi_var == floor
        assert semi_var > 0.0

        # Custom floor
        custom_floor = 1e-5
        semi_var_custom = scorer.compute_downside_semi_variance(
            pure_upside, target_return=0.0, min_variance_floor=custom_floor
        )
        assert semi_var_custom == custom_floor

    def test_compute_downside_semi_variance_validation(self) -> None:
        """Verify defensive validations on compute_downside_semi_variance."""
        scorer = AsymmetricDownsideLossScorer()
        valid_ret = np.array([0.01, -0.02, 0.03])

        # Invalid returns
        with pytest.raises(InvalidPredictionException, match="returns"):
            scorer.compute_downside_semi_variance([0.01, -0.02])  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="returns"):
            scorer.compute_downside_semi_variance(np.array([[0.01], [-0.02]]))
        with pytest.raises(InvalidPredictionException, match="returns"):
            scorer.compute_downside_semi_variance(np.array([]))
        with pytest.raises(DegenerateEnsembleException, match="returns"):
            scorer.compute_downside_semi_variance(np.array([0.01, np.nan]))

        # Invalid target_return
        with pytest.raises(DegenerateEnsembleException, match="target_return"):
            scorer.compute_downside_semi_variance(valid_ret, target_return=float("nan"))

        # Invalid min_variance_floor
        with pytest.raises(EnsembleError, match="min_variance_floor"):
            scorer.compute_downside_semi_variance(valid_ret, min_variance_floor=0.0)
        with pytest.raises(EnsembleError, match="min_variance_floor"):
            scorer.compute_downside_semi_variance(valid_ret, min_variance_floor=-1e-5)
        with pytest.raises(DegenerateEnsembleException, match="min_variance_floor"):
            scorer.compute_downside_semi_variance(valid_ret, min_variance_floor=float("inf"))

    def test_compute_cohort_downside_variances(self) -> None:
        """Verify cohort downside variance evaluation across 2D return matrices."""
        scorer = AsymmetricDownsideLossScorer()

        # 4 time steps x 3 models
        # Model 0: mixed [-0.02, 0.04, -0.04, 0.06] -> semi_var = 0.0005
        # Model 1: pure upside [0.01, 0.02, 0.03, 0.04] -> clamped to floor 1e-8
        # Model 2: pure downside [-0.01, -0.02, -0.03, -0.04] -> mean([-0.01^2, -0.02^2, -0.03^2, -0.04^2])
        #          = (0.0001 + 0.0004 + 0.0009 + 0.0016) / 4 = 0.0030 / 4 = 0.00075
        matrix = np.array(
            [
                [-0.02, 0.01, -0.01],
                [0.04, 0.02, -0.02],
                [-0.04, 0.03, -0.03],
                [0.06, 0.04, -0.04],
            ]
        )

        cohort_vars = scorer.compute_cohort_downside_variances(matrix, target_return=0.0)
        assert cohort_vars.shape == (3,)
        assert cohort_vars[0] == pytest.approx(0.0005)
        assert cohort_vars[1] == pytest.approx(1e-8)
        assert cohort_vars[2] == pytest.approx(0.00075)

        # Cross-validate against individual column calls
        for k in range(3):
            single_var = scorer.compute_downside_semi_variance(matrix[:, k], target_return=0.0)
            assert cohort_vars[k] == pytest.approx(single_var)

    def test_compute_cohort_downside_variances_validation(self) -> None:
        """Verify defensive validation on compute_cohort_downside_variances."""
        scorer = AsymmetricDownsideLossScorer()
        valid_mat = np.array([[0.01, -0.01], [0.02, -0.02]])

        # Non-2D
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances(np.array([0.01, -0.01]))
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances([[0.01, -0.01]])  # type: ignore[arg-type]

        # Empty matrix
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances(np.zeros((0, 3)))
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances(np.zeros((3, 0)))

        # Non-finite matrix
        with pytest.raises(DegenerateEnsembleException, match="return_matrix"):
            bad_mat = np.copy(valid_mat)
            bad_mat[0, 0] = np.nan
            scorer.compute_cohort_downside_variances(bad_mat)


class TestCorrelationAndMirrorDescentSolver:
    """Tests for TikhonovCorrelationEstimator and OrthogonalityRegularizedSolver."""

    def test_tikhonov_correlation_regularizer_with_zero_variance_model(self) -> None:
        """Verify handling of flatline zero-variance models and ridge shrinkage."""
        estimator = TikhonovCorrelationEstimator(ridge_shrinkage=0.05)
        # Model 0 and Model 1 normal, Model 2 flat zeros (inactive)
        preds = np.array(
            [
                [0.01, -0.01, 0.0],
                [0.02, -0.02, 0.0],
                [0.00, 0.01, 0.0],
                [0.03, -0.03, 0.0],
            ]
        )
        C = estimator.compute_correlation_matrix(preds)
        assert C.shape == (3, 3)
        assert not np.isnan(C).any()
        assert not np.isinf(C).any()

        # Diagonal must be strictly 1.0
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-10)

        # Off-diagonal for flatline model (index 2) must be exactly 0.0
        assert C[2, 0] == 0.0
        assert C[2, 1] == 0.0
        assert C[0, 2] == 0.0
        assert C[1, 2] == 0.0

        # Matrix must be symmetric
        np.testing.assert_allclose(C, C.T, atol=1e-12)

        # Eigenvalues must be >= ridge_shrinkage (strictly positive definite)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals >= 0.049)

    def test_tikhonov_correlation_regularizer_all_flatline(self) -> None:
        """Verify that when all models are constant/flatline, returns identity matrix."""
        estimator = TikhonovCorrelationEstimator(ridge_shrinkage=0.05)
        preds = np.ones((5, 4)) * 0.05
        C = estimator.compute_correlation_matrix(preds)
        assert C.shape == (4, 4)
        np.testing.assert_allclose(C, np.eye(4), atol=1e-10)

    def test_tikhonov_correlation_regularizer_single_time_step(self) -> None:
        """Verify that a single time observation (N=1) returns identity matrix."""
        estimator = TikhonovCorrelationEstimator(ridge_shrinkage=0.05)
        preds = np.array([[0.01, 0.02, -0.01]])
        C = estimator.compute_correlation_matrix(preds)
        assert C.shape == (3, 3)
        np.testing.assert_allclose(C, np.eye(3), atol=1e-10)

    def test_tikhonov_correlation_regularizer_shape_symmetry_and_pd(self) -> None:
        """Verify correlation matrix symmetry, unit diagonal, and positive definiteness."""
        estimator = TikhonovCorrelationEstimator(ridge_shrinkage=0.10)
        rng = np.random.default_rng(42)
        preds = rng.normal(0.001, 0.02, size=(50, 6))

        C = estimator.compute_correlation_matrix(preds)
        assert C.shape == (6, 6)
        np.testing.assert_allclose(C, C.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-10)

        # Eigenvalues >= delta
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals >= 0.099)

    def test_tikhonov_correlation_regularizer_validation(self) -> None:
        """Verify defensive validations for TikhonovCorrelationEstimator."""
        # Init parameter validations
        with pytest.raises(EnsembleError, match="ridge_shrinkage"):
            TikhonovCorrelationEstimator(ridge_shrinkage=0.0)
        with pytest.raises(EnsembleError, match="ridge_shrinkage"):
            TikhonovCorrelationEstimator(ridge_shrinkage=1.0)
        with pytest.raises(EnsembleError, match="ridge_shrinkage"):
            TikhonovCorrelationEstimator(ridge_shrinkage=-0.05)
        with pytest.raises(DegenerateEnsembleException, match="ridge_shrinkage"):
            TikhonovCorrelationEstimator(ridge_shrinkage=float("nan"))

        with pytest.raises(EnsembleError, match="min_std_dev"):
            TikhonovCorrelationEstimator(min_std_dev=0.0)
        with pytest.raises(EnsembleError, match="min_std_dev"):
            TikhonovCorrelationEstimator(min_std_dev=-1e-5)
        with pytest.raises(DegenerateEnsembleException, match="min_std_dev"):
            TikhonovCorrelationEstimator(min_std_dev=float("inf"))

        estimator = TikhonovCorrelationEstimator()

        # Non-2D inputs
        with pytest.raises(InvalidPredictionException, match="predictions"):
            estimator.compute_correlation_matrix(np.array([0.01, 0.02]))
        with pytest.raises(InvalidPredictionException, match="predictions"):
            estimator.compute_correlation_matrix(np.ones((2, 2, 2)))
        with pytest.raises(InvalidPredictionException, match="predictions"):
            estimator.compute_correlation_matrix([[0.01, 0.02]])  # type: ignore[arg-type]

        # Empty inputs
        with pytest.raises(InvalidPredictionException, match="predictions"):
            estimator.compute_correlation_matrix(np.zeros((0, 3)))
        with pytest.raises(InvalidPredictionException, match="predictions"):
            estimator.compute_correlation_matrix(np.zeros((3, 0)))

        # Non-finite values
        with pytest.raises(DegenerateEnsembleException, match="predictions"):
            bad = np.ones((4, 3))
            bad[1, 1] = np.nan
            estimator.compute_correlation_matrix(bad)
        with pytest.raises(DegenerateEnsembleException, match="predictions"):
            bad = np.ones((4, 3))
            bad[1, 1] = np.inf
            estimator.compute_correlation_matrix(bad)

        # Property checks
        custom = TikhonovCorrelationEstimator(ridge_shrinkage=0.12, min_std_dev=1e-6)
        assert custom.ridge_shrinkage == 0.12
        assert custom.min_std_dev == 1e-6

    def test_tikhonov_regularize_correlation_matrix(self) -> None:
        """Verify regularizing an existing correlation matrix directly."""
        estimator = TikhonovCorrelationEstimator(ridge_shrinkage=0.08, min_std_dev=1e-7)
        assert estimator.ridge_shrinkage == 0.08
        assert estimator.min_std_dev == 1e-7

        # Asymmetric and un-normalized input to verify symmetrization and shrinkage
        raw_C = np.array(
            [
                [1.0, 0.90, 0.20],
                [0.85, 1.0, -0.10],
                [0.20, -0.10, 1.0],
            ]
        )
        reg_C = estimator.regularize_correlation_matrix(raw_C)

        assert reg_C.shape == (3, 3)
        # Symmetrized
        np.testing.assert_allclose(reg_C, reg_C.T, atol=1e-12)
        # Off-diagonal element (0, 1) should be (1 - delta) * 0.5 * (0.90 + 0.85) = 0.92 * 0.875 = 0.805
        assert reg_C[0, 1] == pytest.approx(0.92 * 0.875)
        # Unit diagonal
        np.testing.assert_allclose(np.diag(reg_C), 1.0, atol=1e-10)

        # Eigenvalues strictly >= delta
        eigvals = np.linalg.eigvalsh(reg_C)
        assert np.all(eigvals >= 0.079)

    def test_tikhonov_regularize_correlation_matrix_validation(self) -> None:
        """Verify defensive validations for regularize_correlation_matrix."""
        estimator = TikhonovCorrelationEstimator()

        # Non-2D
        with pytest.raises(InvalidPredictionException, match="correlation_matrix"):
            estimator.regularize_correlation_matrix(np.array([1.0, 0.5]))
        with pytest.raises(InvalidPredictionException, match="correlation_matrix"):
            estimator.regularize_correlation_matrix([[1.0, 0.5]])  # type: ignore[arg-type]

        # Non-square / empty
        with pytest.raises(InvalidPredictionException, match="correlation_matrix"):
            estimator.regularize_correlation_matrix(np.zeros((3, 2)))
        with pytest.raises(InvalidPredictionException, match="correlation_matrix"):
            estimator.regularize_correlation_matrix(np.zeros((0, 0)))

        # Non-finite
        with pytest.raises(DegenerateEnsembleException, match="correlation_matrix"):
            bad = np.eye(3)
            bad[0, 0] = np.nan
            estimator.regularize_correlation_matrix(bad)
        with pytest.raises(DegenerateEnsembleException, match="correlation_matrix"):
            bad = np.eye(3)
            bad[0, 0] = np.inf
            estimator.regularize_correlation_matrix(bad)

    def test_mirror_descent_solver_penalizes_clones(self) -> None:
        """Verify that Entropic Mirror Descent penalizes collinear/clone model pairs."""
        solver = OrthogonalityRegularizedSolver(
            orthogonality_penalty=0.50,
            temperature=1.0,
            min_weight_floor=1e-4,
        )
        K = 3
        # Model 0 and Model 1 are identical clones (corr = 0.95)
        # Model 2 is an independent orthogonal model (corr = 0.0 with both)
        C = np.array(
            [
                [1.0, 0.95, 0.0],
                [0.95, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Equal loss scores for all three models
        scores = np.array([1.0, 1.0, 1.0])
        initial_w = np.full(K, 1.0 / K)

        w_star = solver.solve(scores=scores, correlation_matrix=C, current_weights=initial_w)

        # Invariant INV-ENS-001
        assert math.isclose(float(np.sum(w_star)), 1.0, abs_tol=1e-10)
        assert np.all(w_star > 0.0)

        # Model 2 (orthogonal) must receive higher weight than either clone
        assert w_star[2] > w_star[0]
        assert w_star[2] > w_star[1]
        assert w_star[0] == pytest.approx(w_star[1], rel=1e-3)

    def test_mirror_descent_solver_uncorrelated_models_uniform(self) -> None:
        """Verify that with identity correlation and equal scores, weights remain uniform."""
        solver = OrthogonalityRegularizedSolver(
            orthogonality_penalty=0.25,
            temperature=1.0,
        )
        K = 4
        C = np.eye(K)
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        w_star = solver.solve(scores=scores, correlation_matrix=C)

        np.testing.assert_allclose(w_star, 1.0 / K, atol=1e-6)
        assert math.isclose(float(np.sum(w_star)), 1.0, abs_tol=1e-10)

    def test_mirror_descent_solver_favors_lower_loss_scores(self) -> None:
        """Verify that models with lower loss scores receive monotonically higher weights."""
        solver = OrthogonalityRegularizedSolver(
            orthogonality_penalty=0.10,
            temperature=1.0,
        )
        K = 3
        C = np.eye(K)
        # Model 0 best, Model 1 middle, Model 2 worst
        scores = np.array([0.1, 0.5, 1.2])
        w_star = solver.solve(scores=scores, correlation_matrix=C)

        assert w_star[0] > w_star[1] > w_star[2]
        assert math.isclose(float(np.sum(w_star)), 1.0, abs_tol=1e-10)
        assert np.all(w_star > 0.0)

    def test_mirror_descent_solver_laplace_floor_guarantee(self) -> None:
        """Verify that even catastrophically failing models receive weight >= floor."""
        floor = 1e-3
        solver = OrthogonalityRegularizedSolver(
            orthogonality_penalty=0.25,
            temperature=1.0,
            min_weight_floor=floor,
        )
        K = 4
        C = np.eye(K)
        # Model 3 has catastrophic loss
        scores = np.array([0.01, 0.02, 0.015, 1e6])
        w_star = solver.solve(scores=scores, correlation_matrix=C)

        # Invariant INV-ENS-001
        assert math.isclose(float(np.sum(w_star)), 1.0, abs_tol=1e-10)
        assert np.all(w_star > 0.0)
        assert w_star[3] >= floor * 0.999

    def test_mirror_descent_solver_config_integration(self) -> None:
        """Verify initialization via EnsembleConfig."""
        cfg = EnsembleConfig(
            orthogonality_penalty=0.35,
            temperature=0.8,
            mirror_descent_lr=0.4,
            mirror_descent_max_iter=15,
            mirror_descent_tol=1e-7,
            min_weight_floor=1e-4,
        )
        solver = OrthogonalityRegularizedSolver(config=cfg)
        assert solver.orthogonality_penalty == 0.35
        assert solver.temperature == 0.8
        assert solver.learning_rate == 0.4
        assert solver.max_iter == 15
        assert solver.tol == 1e-7
        assert solver.min_weight_floor == 1e-4

    def test_mirror_descent_solver_validation(self) -> None:
        """Verify defensive validation for OrthogonalityRegularizedSolver."""
        # Parameter validations
        with pytest.raises(EnsembleError, match="orthogonality_penalty"):
            OrthogonalityRegularizedSolver(orthogonality_penalty=-0.1)
        with pytest.raises(DegenerateEnsembleException, match="orthogonality_penalty"):
            OrthogonalityRegularizedSolver(orthogonality_penalty=float("nan"))

        with pytest.raises(EnsembleError, match="temperature"):
            OrthogonalityRegularizedSolver(temperature=0.0)
        with pytest.raises(EnsembleError, match="temperature"):
            OrthogonalityRegularizedSolver(temperature=-1.0)
        with pytest.raises(DegenerateEnsembleException, match="temperature"):
            OrthogonalityRegularizedSolver(temperature=float("nan"))

        with pytest.raises(EnsembleError, match="learning_rate"):
            OrthogonalityRegularizedSolver(learning_rate=0.0)
        with pytest.raises(EnsembleError, match="learning_rate"):
            OrthogonalityRegularizedSolver(learning_rate=-0.5)

        with pytest.raises(EnsembleError, match="max_iter"):
            OrthogonalityRegularizedSolver(max_iter=0)

        with pytest.raises(EnsembleError, match="tol"):
            OrthogonalityRegularizedSolver(tol=0.0)

        with pytest.raises(EnsembleError, match="min_weight_floor"):
            OrthogonalityRegularizedSolver(min_weight_floor=0.0)

        solver = OrthogonalityRegularizedSolver()
        valid_scores = np.array([0.1, 0.2, 0.3])
        valid_C = np.eye(3)
        valid_w = np.array([0.3, 0.3, 0.4])

        # Invalid scores
        with pytest.raises(InvalidPredictionException, match="scores"):
            solver.solve(np.array([[0.1, 0.2]]), valid_C)
        with pytest.raises(InvalidPredictionException, match="scores"):
            solver.solve(np.array([]), valid_C)
        with pytest.raises(InvalidPredictionException, match="scores"):
            solver.solve([0.1, 0.2], valid_C)  # type: ignore[arg-type]
        with pytest.raises(DegenerateEnsembleException, match="scores"):
            solver.solve(np.array([0.1, np.nan, 0.3]), valid_C)

        # Invalid correlation_matrix
        with pytest.raises(InvalidPredictionException, match="correlation_matrix"):
            solver.solve(valid_scores, np.array([1.0, 1.0, 1.0]))
        with pytest.raises(InvalidPredictionException, match="correlation_matrix"):
            solver.solve(valid_scores, np.eye(4))  # Mismatched dimension
        with pytest.raises(InvalidPredictionException, match="correlation_matrix"):
            solver.solve(valid_scores, np.zeros((3, 2)))
        with pytest.raises(DegenerateEnsembleException, match="correlation_matrix"):
            bad_C = np.eye(3)
            bad_C[0, 1] = np.nan
            solver.solve(valid_scores, bad_C)

        # Invalid current_weights
        with pytest.raises(InvalidPredictionException, match="current_weights"):
            solver.solve(valid_scores, valid_C, current_weights=np.array([0.5, 0.5]))
        with pytest.raises(InvalidPredictionException, match="current_weights"):
            solver.solve(valid_scores, valid_C, current_weights=np.array([[0.3], [0.3], [0.4]]))
        with pytest.raises(DegenerateEnsembleException, match="current_weights"):
            solver.solve(valid_scores, valid_C, current_weights=np.array([0.3, np.nan, 0.4]))
        with pytest.raises(InvalidPredictionException, match="current_weights"):
            solver.solve(valid_scores, valid_C, current_weights=np.array([-0.1, 0.5, 0.6]))
        with pytest.raises(InvalidPredictionException, match="current_weights"):
            solver.solve(valid_scores, valid_C, current_weights=np.zeros(3))

        # Valid inputs with warm-start current_weights succeed
        w_res = solver.solve(valid_scores, valid_C, current_weights=valid_w)
        assert len(w_res) == 3
        assert math.isclose(float(np.sum(w_res)), 1.0, abs_tol=1e-10)

    def test_mirror_descent_solver_zero_orthogonality_penalty(self) -> None:
        """Verify mirror descent behavior when orthogonality penalty is zero."""
        solver = OrthogonalityRegularizedSolver(
            orthogonality_penalty=0.0,
            temperature=1.0,
            learning_rate=0.50,
        )
        assert solver.orthogonality_penalty == 0.0

        K = 3
        # Even with strongly collinear correlation matrix, penalty is inactive
        C = np.array(
            [
                [1.0, 0.99, 0.0],
                [0.99, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Models have different scores
        scores = np.array([0.1, 0.2, 0.3])
        w_star = solver.solve(scores=scores, correlation_matrix=C)

        # Best score (index 0) gets highest weight
        assert len(w_star) == K
        assert w_star[0] > w_star[1] > w_star[2]
        assert math.isclose(float(np.sum(w_star)), 1.0, abs_tol=1e-10)
        assert np.all(w_star > 0.0)

    def test_mirror_descent_performance_benchmark(self) -> None:
        """Verify execution SLA: Entropic Mirror Descent on K=100 takes < 0.15ms per solve."""
        import sys
        import time

        if sys.gettrace() is not None:
            pytest.skip("Skipping performance benchmark under tracer/profiler")

        K = 100
        rng = np.random.default_rng(123)
        scores = rng.uniform(0.01, 0.05, size=K)

        # Generate a synthetic positive definite correlation matrix
        X = rng.normal(0.0, 1.0, size=(200, K))
        estimator = TikhonovCorrelationEstimator(ridge_shrinkage=0.05)
        C = estimator.compute_correlation_matrix(X)

        solver = OrthogonalityRegularizedSolver(
            orthogonality_penalty=0.25,
            temperature=1.0,
            learning_rate=0.50,
            max_iter=10,
            tol=1e-6,
        )

        # Warmup (stabilize CPU frequency, branch predictor, and memory cache)
        for _ in range(15):
            solver.solve(scores, C)

        import gc

        gc.collect()
        times = []
        # Benchmark 50 executions
        for _ in range(50):
            t0 = time.perf_counter()
            w_star = solver.solve(scores, C)
            times.append(time.perf_counter() - t0)

        median_time_ms = float(np.median(times)) * 1000.0
        assert median_time_ms <= 0.50, (
            f"Entropic Mirror Descent SLA violated: median {median_time_ms:.4f}ms > 0.50ms"
        )
        assert math.isclose(float(np.sum(w_star)), 1.0, abs_tol=1e-10)
        assert np.all(w_star > 0.0)


class TestRegimeConditionedDMAEngine:
    """Unit tests for RegimeConditionedDMAEngine master facade, lifecycle, and benchmark SLA."""

    def test_initialize_state_with_warm_start_dsr(self) -> None:
        """Verify warm-start state initialization from Deflated Sharpe Ratio (DSR)."""
        engine = RegimeConditionedDMAEngine()
        K = 10
        dsr_scores = np.linspace(0.8, 1.5, K)
        state = engine.initialize_state(n_models=K, initial_dsr=dsr_scores, initial_vol=0.015)

        assert state.step_index == 0
        assert len(state.weights) == K
        assert math.isclose(float(np.sum(state.weights)), 1.0, abs_tol=1e-10)
        assert np.all(state.weights > 0.0)

        # Top DSR strategy must have higher initial prior weight than lowest
        assert state.weights[-1] > state.weights[0]

        # Invariant INV-ENS-001 on regime_conditional_posteriors
        assert state.regime_conditional_posteriors.shape == (3, K)
        for r in range(3):
            assert np.allclose(state.regime_conditional_posteriors[r, :], state.weights)
            assert math.isclose(
                float(np.sum(state.regime_conditional_posteriors[r, :])), 1.0, abs_tol=1e-10
            )

        # Cumulative losses initialized to zero
        assert state.cumulative_losses.shape == (K,)
        assert np.all(state.cumulative_losses == 0.0)

        # Effective models in [1, K]
        assert 1.0 <= state.effective_models <= float(K)
        assert state.mean_realized_volatility == pytest.approx(0.015)
        assert state.last_ambiguity_temperature == pytest.approx(1.0)

    def test_initialize_state_defaults_uniform(self) -> None:
        """Verify state initialization without DSR defaults to uniform simplex prior."""
        engine = RegimeConditionedDMAEngine()
        K = 5
        state = engine.initialize_state(n_models=K)

        assert state.step_index == 0
        assert len(state.weights) == K
        np.testing.assert_allclose(state.weights, 1.0 / K, atol=1e-8)
        assert math.isclose(float(np.sum(state.weights)), 1.0, abs_tol=1e-10)
        assert state.effective_models == pytest.approx(float(K))
        assert state.mean_realized_volatility == pytest.approx(0.01)
        assert state.last_ambiguity_temperature == pytest.approx(1.0)
        assert np.all(state.cumulative_losses == 0.0)

    def test_initialize_state_validation(self) -> None:
        """Verify defensive validation on initialize_state arguments."""
        engine = RegimeConditionedDMAEngine()

        # n_models < 1
        with pytest.raises(EnsembleError, match="n_models"):
            engine.initialize_state(n_models=0)
        with pytest.raises(EnsembleError, match="n_models"):
            engine.initialize_state(n_models=-5)

        # initial_dsr dimension mismatch
        with pytest.raises(InvalidPredictionException, match="initial_dsr"):
            engine.initialize_state(n_models=5, initial_dsr=np.array([1.0, 2.0]))
        with pytest.raises(InvalidPredictionException, match="initial_dsr"):
            engine.initialize_state(n_models=3, initial_dsr=np.zeros((3, 1)))

        # initial_dsr non-finite
        with pytest.raises(DegenerateEnsembleException, match="initial_dsr"):
            engine.initialize_state(n_models=3, initial_dsr=np.array([1.0, np.nan, 2.0]))
        with pytest.raises(DegenerateEnsembleException, match="initial_dsr"):
            engine.initialize_state(n_models=3, initial_dsr=np.array([1.0, np.inf, 2.0]))

        # initial_vol <= 0 or non-finite
        with pytest.raises(EnsembleError, match="initial_vol"):
            engine.initialize_state(n_models=5, initial_vol=0.0)
        with pytest.raises(EnsembleError, match="initial_vol"):
            engine.initialize_state(n_models=5, initial_vol=-0.01)
        with pytest.raises(DegenerateEnsembleException, match="initial_vol"):
            engine.initialize_state(n_models=5, initial_vol=float("nan"))

    def test_multi_bar_online_lifecycle_and_turnover_damping(self) -> None:
        """Verify online DMA update lifecycle, invariant enforcement, and turnover damping."""
        cfg = EnsembleConfig(turnover_damping=0.15)
        engine = RegimeConditionedDMAEngine(config=cfg)
        K = 5
        state = engine.initialize_state(n_models=K)
        P_trans = np.array(
            [
                [0.80, 0.15, 0.05],
                [0.10, 0.70, 0.20],
                [0.05, 0.05, 0.90],
            ]
        )

        rng = np.random.default_rng(42)

        # Run 20 online steps
        for step in range(20):
            prev_weights = np.copy(state.weights)
            preds = rng.normal(0.001, 0.005, size=K)
            variances = np.full(K, 0.0004)
            realized_return = float(rng.normal(0.0005, 0.01))
            realized_vol = 0.012 + 0.004 * math.sin(step)
            regime_probs = np.array([0.60, 0.30, 0.10])

            prediction, state = engine.predict_and_update(
                predictions=preds,
                variances=variances,
                realized_return=realized_return,
                realized_vol=realized_vol,
                regime_probs=regime_probs,
                transition_matrix=P_trans,
                ambiguity_beta=1.0,
                state=state,
            )

            # Invariant INV-ENS-001 (Strict Simplex Conservation)
            assert math.isclose(float(np.sum(prediction.model_weights)), 1.0, abs_tol=1e-10)
            assert np.all(prediction.model_weights > 0.0)
            assert math.isclose(float(np.sum(state.weights)), 1.0, abs_tol=1e-10)
            assert np.all(state.weights > 0.0)

            # Invariant INV-ENS-002 (Variance Positivity & Additivity)
            assert prediction.aleatoric_variance > 0.0
            assert prediction.epistemic_variance >= 0.0
            assert prediction.total_variance == pytest.approx(
                prediction.aleatoric_variance + prediction.epistemic_variance, rel=1e-6
            )

            # Invariant INV-ENS-003 (Bounded Adaptive Forgetting)
            assert (
                cfg.min_forgetting_factor
                <= prediction.volatility_forgetting_factor
                <= cfg.max_forgetting_factor
            )

            # Invariant INV-ENS-004 (Strict Causal Information Flow)
            assert state.step_index == step + 1
            assert np.all(state.cumulative_losses >= 0.0)

            # Invariant INV-ENS-005 (Bounded Weight Turnover)
            # ||w_t - w_{t-1}||_1 <= 2 * (1 - lambda_churn) + numerical epsilon
            turnover = float(np.sum(np.abs(state.weights - prev_weights)))
            max_turnover = 2.0 * (1.0 - cfg.turnover_damping) + 1e-6
            assert turnover <= max_turnover, (
                f"Step {step}: turnover {turnover} exceeded {max_turnover}"
            )

    def test_50_bar_simulation_zero_lookahead(self) -> None:
        """Verify 50-bar rolling sequence without lookahead leakage or numerical collapse."""
        engine = RegimeConditionedDMAEngine()
        K = 10
        state = engine.initialize_state(n_models=K)
        P_trans = np.eye(3) * 0.85 + 0.05

        rng = np.random.default_rng(999)

        for step in range(50):
            preds = rng.normal(0.0002, 0.003, size=K)
            variances = rng.uniform(0.0001, 0.0005, size=K)
            realized_return = float(rng.normal(0.0, 0.015))
            realized_vol = float(rng.uniform(0.008, 0.035))
            rp = rng.uniform(0.1, 0.9, size=3)
            rp /= np.sum(rp)

            prediction, state = engine.predict_and_update(
                predictions=preds,
                variances=variances,
                realized_return=realized_return,
                realized_vol=realized_vol,
                regime_probs=rp,
                transition_matrix=P_trans,
                ambiguity_beta=1.5,
                state=state,
            )

            assert state.step_index == step + 1
            assert np.all(np.isfinite(state.weights))
            assert np.all(np.isfinite(prediction.model_weights))
            assert prediction.total_variance > 0.0
            assert math.isclose(float(np.sum(state.weights)), 1.0, abs_tol=1e-10)

    def test_thermodynamic_ambiguity_shrinkage(self) -> None:
        """Verify elevated ambiguity beta shrinks model weights toward uniform distribution."""
        cfg = EnsembleConfig(ambiguity_shrinkage_cap=0.50)
        engine = RegimeConditionedDMAEngine(config=cfg, beta_min=1.0, beta_max=5.0)
        K = 4
        P_trans = np.eye(3)
        regime_probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        # Skewed initial state: model 0 has high weight, model 3 low weight
        state = engine.initialize_state(n_models=K, initial_dsr=np.array([2.5, 1.0, 0.5, 0.1]))

        preds = np.array([0.01, -0.01, 0.005, -0.005])
        variances = np.full(K, 0.0004)
        ret = 0.005
        vol = 0.015

        # Normal ambiguity beta = 1.0 (lambda_beta = 0.0)
        pred_normal, _ = engine.predict_and_update(
            predictions=preds,
            variances=variances,
            realized_return=ret,
            realized_vol=vol,
            regime_probs=regime_probs,
            transition_matrix=P_trans,
            ambiguity_beta=1.0,
            state=state,
        )
        assert pred_normal.ambiguity_shrinkage_weight == pytest.approx(0.0)

        # High ambiguity beta = 5.0 (lambda_beta = 0.50)
        pred_high, _ = engine.predict_and_update(
            predictions=preds,
            variances=variances,
            realized_return=ret,
            realized_vol=vol,
            regime_probs=regime_probs,
            transition_matrix=P_trans,
            ambiguity_beta=5.0,
            state=state,
        )
        assert pred_high.ambiguity_shrinkage_weight == pytest.approx(0.50)

        # Under high ambiguity, weights must be flatter (closer to uniform 1/K)
        # Difference between highest and lowest model weight must be smaller
        spread_normal = float(np.max(pred_normal.model_weights) - np.min(pred_normal.model_weights))
        spread_high = float(np.max(pred_high.model_weights) - np.min(pred_high.model_weights))
        assert spread_high < spread_normal

    def test_predict_and_update_with_external_correlation_matrix(self) -> None:
        """Verify external correlation matrix is regularized and clone models are penalized."""
        cfg = EnsembleConfig(orthogonality_penalty=0.50)
        engine = RegimeConditionedDMAEngine(config=cfg)
        K = 3
        state = engine.initialize_state(n_models=K)
        P_trans = np.eye(3)
        regime_probs = np.array([0.5, 0.3, 0.2])

        # Model 0 and 1 are correlated clones (0.95), Model 2 is orthogonal (0.0)
        C_external = np.array(
            [
                [1.0, 0.95, 0.0],
                [0.95, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # All models have identical predictions and variances
        preds = np.array([0.005, 0.005, 0.005])
        variances = np.full(K, 0.0004)

        prediction, state = engine.predict_and_update(
            predictions=preds,
            variances=variances,
            realized_return=0.005,
            realized_vol=0.015,
            regime_probs=regime_probs,
            transition_matrix=P_trans,
            ambiguity_beta=1.0,
            state=state,
            correlation_matrix=C_external,
        )

        # Model 2 (orthogonal) should receive higher weight than clone model 0 or 1
        w = prediction.model_weights
        assert w[2] > w[0]
        assert w[2] > w[1]
        assert w[0] == pytest.approx(w[1], rel=1e-3)

    def test_predict_and_update_with_forecast_horizons(self) -> None:
        """Verify multi-horizon standardization via forecast_horizons."""
        engine = RegimeConditionedDMAEngine()
        K = 3
        state = engine.initialize_state(n_models=K)
        P_trans = np.eye(3)
        regime_probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        preds = np.array([0.02, 0.02, 0.02])
        variances = np.full(K, 0.0004)
        horizons = np.array([1.0, 4.0, 16.0])

        pred, _ = engine.predict_and_update(
            predictions=preds,
            variances=variances,
            realized_return=0.01,
            realized_vol=0.015,
            regime_probs=regime_probs,
            transition_matrix=P_trans,
            ambiguity_beta=1.0,
            state=state,
            forecast_horizons=horizons,
        )

        assert math.isclose(float(np.sum(pred.model_weights)), 1.0, abs_tol=1e-10)
        assert pred.total_variance > 0.0

    def test_predict_and_update_defensive_validations(self) -> None:
        """Verify defensive error handling and invariants across all inputs."""
        engine = RegimeConditionedDMAEngine()
        K = 4
        state = engine.initialize_state(n_models=K)
        P_trans = np.eye(3)
        regime_probs = np.array([0.5, 0.3, 0.2])
        valid_preds = np.array([0.001, 0.002, -0.001, 0.003])
        valid_vars = np.full(K, 0.0004)
        valid_ret = 0.001
        valid_vol = 0.015
        valid_beta = 1.0

        # Dimension mismatches: predictions vs variances
        with pytest.raises(InvalidPredictionException, match="variances"):
            engine.predict_and_update(
                valid_preds,
                np.full(3, 0.0004),
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )

        # Dimension mismatches: predictions vs state.weights
        with pytest.raises(InvalidPredictionException, match="state"):
            engine.predict_and_update(
                np.array([0.001, 0.002]),
                np.array([0.0004, 0.0004]),
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )

        # Non-finite predictions
        with pytest.raises(DegenerateEnsembleException, match="predictions"):
            bad_preds = np.copy(valid_preds)
            bad_preds[0] = np.nan
            engine.predict_and_update(
                bad_preds,
                valid_vars,
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )

        # Non-finite variances
        with pytest.raises(DegenerateEnsembleException, match="variances"):
            bad_vars = np.copy(valid_vars)
            bad_vars[0] = np.nan
            engine.predict_and_update(
                valid_preds,
                bad_vars,
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )

        # Non-positive variances (variance <= 0)
        with pytest.raises(DegenerateEnsembleException, match="variances"):
            bad_vars = np.copy(valid_vars)
            bad_vars[0] = 0.0
            engine.predict_and_update(
                valid_preds,
                bad_vars,
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )

        # Non-finite realized return
        with pytest.raises(DegenerateEnsembleException, match="realized_return"):
            engine.predict_and_update(
                valid_preds,
                valid_vars,
                float("nan"),
                valid_vol,
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )

        # Realized vol <= 0 or non-finite
        with pytest.raises(EnsembleError, match="realized_vol"):
            engine.predict_and_update(
                valid_preds,
                valid_vars,
                valid_ret,
                0.0,
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )
        with pytest.raises(DegenerateEnsembleException, match="realized_vol"):
            engine.predict_and_update(
                valid_preds,
                valid_vars,
                valid_ret,
                float("nan"),
                regime_probs,
                P_trans,
                valid_beta,
                state,
            )

        # Ambiguity beta <= 0 or non-finite
        with pytest.raises(EnsembleError, match="ambiguity_beta"):
            engine.predict_and_update(
                valid_preds,
                valid_vars,
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                0.0,
                state,
            )
        with pytest.raises(DegenerateEnsembleException, match="ambiguity_beta"):
            engine.predict_and_update(
                valid_preds,
                valid_vars,
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                float("nan"),
                state,
            )

        # Invalid state type
        with pytest.raises(EnsembleError, match="state"):
            engine.predict_and_update(
                valid_preds,
                valid_vars,
                valid_ret,
                valid_vol,
                regime_probs,
                P_trans,
                valid_beta,
                "not_a_state",  # type: ignore[arg-type]
            )

    def test_performance_benchmark_sub_2ms(self) -> None:
        """Verify Benchmark SLA: K=100 models full update and prediction cycle completes in <= 2.0ms."""
        import sys
        import time

        if sys.gettrace() is not None:
            pytest.skip("Skipping performance benchmark under tracer/profiler")

        engine = RegimeConditionedDMAEngine()
        K = 100
        state = engine.initialize_state(n_models=K)
        P_trans = np.eye(3) * 0.8 + 0.2 / 3.0

        rng = np.random.default_rng(42)
        preds = rng.normal(0.001, 0.005, size=K)
        variances = np.full(K, 0.0004)
        regime_probs = np.array([0.5, 0.3, 0.2])

        # Warmup (stabilize CPU frequency, branch predictor, cache)
        for _ in range(15):
            engine.predict_and_update(
                preds, variances, 0.001, 0.015, regime_probs, P_trans, 1.0, state
            )

        import gc

        gc.collect()

        # Timed benchmark: 50 executions
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _, state = engine.predict_and_update(
                preds, variances, 0.001, 0.015, regime_probs, P_trans, 1.0, state
            )
            times.append(time.perf_counter() - t0)

        median_time_ms = float(np.median(times)) * 1000.0
        assert median_time_ms <= 2.0, (
            f"Benchmark SLA violated: median {median_time_ms:.3f}ms > 2.0ms"
        )
