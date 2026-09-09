"""Unit tests for Multi-Asset Cross-Impact Propagator Engine.

Purpose:
    Validates mathematical invariants, Huberman-Stanzl symmetry, 3/2-power Square-Root Law
    scaling, exact analytical gradient accuracy, zero gradient at rest, state-space depletion,
    and panic asymmetry for src/quant/analytics/market_impact.py.

Invariants Verified:
    1. Configuration invariants and boundary checks (ERR-GAME-IMPACT-PARAM).
    2. Generalized Pseudo-Huber 3/2-power potential: Psi(0) = 0, Psi'(0) = 0, Psi''(u) > 0.
    3. Huberman-Stanzl No-Arbitrage: Lambda_cross is strictly symmetric positive-definite.
    4. Exact zero gradient at rest: nabla C(0) = 0.
    5. Gradient verification: Analytical gradient matches finite difference within 1e-4.
    6. Asymmetric panic liquidation: selling in panic costs strictly more than buying.
    7. State-space depletion bounds: B_t is strictly contained in [0, B_max].
    8. Dimensionality error handling (ERR-GAME-IMPACT-DIM).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from quant.analytics.market_impact import (
    ERR_IMPACT_DIM,
    ERR_IMPACT_PARAM,
    GeneralizedPseudoHuber,
    HubermanStanzlCrossImpact,
    MarketImpactConfig,
    MultiAssetMarketImpactEngine,
)


class TestMarketImpactConfig:
    """Test suite for MarketImpactConfig parameter invariants and validation."""

    def test_default_config_valid(self) -> None:
        """Verify default configuration instantiates properly."""
        config = MarketImpactConfig()
        assert config.fee_floor == 0.0005
        assert config.huber_delta == 1e-4
        assert config.resilience_rate == 0.50
        assert config.depletion_cap == 0.20
        assert config.portfolio_value == 1_000_000.0

    def test_invalid_fee_floor(self) -> None:
        """Verify non-positive fee_floor raises ValueError."""
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(fee_floor=0.0)
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(fee_floor=-0.001)

    def test_invalid_huber_delta(self) -> None:
        """Verify non-positive huber_delta raises ValueError."""
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(huber_delta=0.0)

    def test_invalid_resilience_rate(self) -> None:
        """Verify resilience_rate outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(resilience_rate=0.0)
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(resilience_rate=1.0)

    def test_invalid_depletion_cap(self) -> None:
        """Verify non-positive depletion_cap raises ValueError."""
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(depletion_cap=0.0)

    def test_invalid_portfolio_value(self) -> None:
        """Verify non-positive portfolio_value raises ValueError."""
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(portfolio_value=0.0)

    def test_invalid_bar_time_fraction(self) -> None:
        """Verify bar_time_fraction outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(bar_time_fraction=0.0)
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(bar_time_fraction=1.5)

    def test_invalid_crowding_max_mult(self) -> None:
        """Verify crowding_max_mult < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=ERR_IMPACT_PARAM):
            MarketImpactConfig(crowding_max_mult=0.8)


class TestGeneralizedPseudoHuber:
    """Test suite for 3/2-power Generalized Pseudo-Huber Potential."""

    def test_zero_rest_properties(self) -> None:
        """Verify potential and gradient are identically zero at origin."""
        huber = GeneralizedPseudoHuber(delta=1e-4)
        u_zero = np.array([0.0])

        val = huber.evaluate(u_zero)
        grad = huber.gradient(u_zero)

        assert pytest.approx(val[0], abs=1e-12) == 0.0
        assert pytest.approx(grad[0], abs=1e-12) == 0.0

    def test_strict_convexity_everywhere(self) -> None:
        """Verify second derivative is strictly positive across negative and positive domain."""
        huber = GeneralizedPseudoHuber(delta=1e-4)
        u_grid = np.linspace(-5.0, 5.0, 100)

        hess = huber.hessian_diagonal(u_grid)
        assert np.all(hess > 0.0)

    def test_square_root_asymptotic_marginal_scaling(self) -> None:
        """Verify first derivative scales with sqrt(|u|) for large participation."""
        huber = GeneralizedPseudoHuber(delta=1e-4)
        u_large = np.array([4.0, 9.0, 16.0])

        grad = huber.gradient(u_large)
        # Expected scaling: 1.5 * sqrt(u)
        expected = 1.5 * np.sqrt(u_large)
        np.testing.assert_allclose(grad, expected, rtol=1e-3)

    def test_gradient_finite_difference_consistency(self) -> None:
        """Verify analytical gradient matches numerical finite difference."""
        huber = GeneralizedPseudoHuber(delta=1e-3)
        u_points = np.array([-2.5, -0.1, 0.0, 0.05, 1.8])
        eps = 1e-6

        val_plus = huber.evaluate(u_points + eps)
        val_minus = huber.evaluate(u_points - eps)
        num_grad = (val_plus - val_minus) / (2.0 * eps)

        ana_grad = huber.gradient(u_points)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-5, atol=1e-6)


class TestHubermanStanzlCrossImpact:
    """Test suite for Huberman-Stanzl Arbitrage-Free Cross-Impact Tensor."""

    def test_symmetry_and_positive_definiteness(self) -> None:
        """Verify cross-impact matrix is strictly symmetric and positive definite."""
        constructor = HubermanStanzlCrossImpact(fee_floor=0.0005, cross_impact_scale=0.15)
        rng = np.random.RandomState(42)

        # Generate random 4-asset positive-definite covariance
        raw = rng.randn(4, 4) * 0.02
        cov = np.dot(raw, raw.T) + np.eye(4) * 1e-4
        adv_dol = np.array([1e7, 5e6, 8e6, 2e7])
        z_t = np.array([0.9, 0.3, 0.8, 0.5])

        lambda_matrix = constructor.build_matrix(cov, adv_dol, z_t)

        # Symmetry assertion
        np.testing.assert_allclose(lambda_matrix, lambda_matrix.T, atol=1e-12)

        # Positive-definiteness assertion (all eigenvalues >= fee_floor)
        eigenvalues = scipy.linalg.eigvalsh(lambda_matrix)
        assert np.all(eigenvalues >= 0.0005 - 1e-9)

    def test_pairs_trade_hedging_cancellation(self) -> None:
        """Verify opposite trades in correlated assets incur lower cross-impact than co-directional."""
        constructor = HubermanStanzlCrossImpact(fee_floor=0.0005, cross_impact_scale=0.20)

        # Highly correlated 2-asset universe (correlation ~ 0.90)
        cov = np.array([[0.04, 0.036], [0.036, 0.04]])
        adv_dol = np.array([1e7, 1e7])

        lambda_mat = constructor.build_matrix(cov, adv_dol)

        # Co-directional trade: buy both
        co_dir = np.array([0.05, 0.05])
        cost_co = np.dot(co_dir, np.dot(lambda_mat, co_dir))

        # Hedged pairs trade: buy one, sell the other
        hedged = np.array([0.05, -0.05])
        cost_hedged = np.dot(hedged, np.dot(lambda_mat, hedged))

        # Hedged cross-impact must be strictly lower due to correlation cancellation
        assert cost_hedged < cost_co


class TestMultiAssetMarketImpactEngine:
    """Test suite for master MultiAssetMarketImpactEngine orchestration."""

    def test_zero_trade_zero_gradient(self) -> None:
        """Verify zero rebalance produces identically zero friction and zero gradient."""
        engine = MultiAssetMarketImpactEngine()
        cov = np.eye(3) * 0.0004
        vols = np.array([1e7, 1e7, 1e7])
        sigmas = np.array([0.015, 0.012, 0.020])

        delta_zero = np.zeros(3)
        res = engine.evaluate_impact(
            delta_allocations=delta_zero,
            covariance=cov,
            daily_dollar_volumes=vols,
            asset_volatilities=sigmas,
            panic_probability=0.5,
        )

        assert pytest.approx(res.total_cost, abs=1e-12) == 0.0
        assert pytest.approx(res.permanent_cost, abs=1e-12) == 0.0
        assert pytest.approx(res.transient_cost, abs=1e-12) == 0.0
        np.testing.assert_allclose(res.gradient, np.zeros(3), atol=1e-12)

    def test_analytical_gradient_matches_finite_difference(self) -> None:
        """Verify analytical gradient vector matches two-sided numerical finite difference."""
        config = MarketImpactConfig(
            fee_floor=0.0005,
            huber_delta=1e-3,
            cross_impact_scale=0.10,
            transient_scale=0.40,
            panic_penalty_scale=1.2,
            portfolio_value=500_000.0,
        )
        engine = MultiAssetMarketImpactEngine(config=config)

        rng = np.random.RandomState(99)
        raw = rng.randn(3, 3) * 0.01
        cov = np.dot(raw, raw.T) + np.eye(3) * 1e-4
        vols = np.array([5e6, 8e6, 1.2e7])
        sigmas = np.array([0.018, 0.022, 0.015])
        delta_test = np.array([0.04, -0.03, 0.02])

        # Evaluate analytical result
        res = engine.evaluate_impact(
            delta_allocations=delta_test,
            covariance=cov,
            daily_dollar_volumes=vols,
            asset_volatilities=sigmas,
            panic_probability=0.6,
        )

        # Finite difference check
        eps = 1e-6
        num_grad = np.zeros(3)
        for i in range(3):
            delta_p = delta_test.copy()
            delta_m = delta_test.copy()
            delta_p[i] += eps
            delta_m[i] -= eps

            c_p = engine.evaluate_impact(
                delta_p, cov, vols, sigmas, panic_probability=0.6
            ).total_cost
            c_m = engine.evaluate_impact(
                delta_m, cov, vols, sigmas, panic_probability=0.6
            ).total_cost

            num_grad[i] = (c_p - c_m) / (2.0 * eps)

        np.testing.assert_allclose(res.gradient, num_grad, rtol=1e-4, atol=1e-5)

    def test_panic_asymmetry_liquidation_penalty(self) -> None:
        """Verify selling into panic incurs strictly higher friction than buying into panic."""
        engine = MultiAssetMarketImpactEngine()
        cov = np.eye(2) * 0.0004
        vols = np.array([1e7, 1e7])
        sigmas = np.array([0.02, 0.02])

        # Sell order under full panic (pi_panic = 1.0)
        res_sell = engine.evaluate_impact(
            delta_allocations=np.array([-0.05, -0.05]),
            covariance=cov,
            daily_dollar_volumes=vols,
            asset_volatilities=sigmas,
            panic_probability=1.0,
        )

        # Buy order of identical size under full panic (pi_panic = 1.0)
        res_buy = engine.evaluate_impact(
            delta_allocations=np.array([0.05, 0.05]),
            covariance=cov,
            daily_dollar_volumes=vols,
            asset_volatilities=sigmas,
            panic_probability=1.0,
        )

        # Sells must pay substantially more transient cost due to panic asymmetry
        assert res_sell.transient_cost > res_buy.transient_cost * 1.5

    def test_state_space_depletion_bounding(self) -> None:
        """Verify consecutive trades compound depletion but remain bounded within depletion_cap."""
        config = MarketImpactConfig(depletion_cap=0.20, resilience_rate=0.70)
        engine = MultiAssetMarketImpactEngine(config=config)

        cov = np.eye(2) * 0.0004
        vols = np.array([1e6, 1e6])  # Small volume to produce large participation
        sigmas = np.array([0.02, 0.02])

        # Execute 25 consecutive massive trades in the same direction
        for _ in range(25):
            res = engine.evaluate_impact(
                delta_allocations=np.array([0.10, 0.10]),
                covariance=cov,
                daily_dollar_volumes=vols,
                asset_volatilities=sigmas,
                update_state=True,
            )

        # Verify depletion vector never exceeds depletion_cap
        assert np.all(res.depletion_state <= config.depletion_cap + 1e-12)
        assert np.all(res.depletion_state > 0.0)

    def test_input_dimension_validation(self) -> None:
        """Verify mismatched dimensions trigger deterministic diagnostic errors."""
        engine = MultiAssetMarketImpactEngine()
        cov = np.eye(3) * 0.0004

        # 3 assets in covariance, but 2 in delta_allocations
        with pytest.raises(ValueError, match=ERR_IMPACT_DIM):
            engine.evaluate_impact(
                delta_allocations=np.array([0.05, -0.05]),
                covariance=cov,
                daily_dollar_volumes=np.array([1e7, 1e7]),
                asset_volatilities=np.array([0.01, 0.01]),
            )
