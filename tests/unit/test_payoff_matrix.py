"""Unit test suite for Step 3: Stackelberg Leader-Follower Trajectory and Payoff Tensor.

Purpose:
    Verifies mathematical and algorithmic correctness of:
    1. DiscreteHyperbolicPropagator (partition of unity, monotonic decay, TWAP limit, friction reduction).
    2. StackelbergPayoffEngine (analytical gradient vs finite difference, negative-definite Hessian, predatory shading).
    3. InstitutionalBenchmarkUniverse (Equal Weight, Risk Parity, Inverse Volatility, Cash, friction parity).
    4. StackelbergPayoffTensorConstructor (payoff tensor and non-negative regret matrix).
    5. Invariant enforcement and error diagnostics (ERR-GAME-STACK-*).
"""

import numpy as np
import pytest

from quant.analytics.market_impact import MarketImpactConfig, MultiAssetMarketImpactEngine
from quant.analytics.payoff_matrix import (
    ERR_STACK_BENCHMARK,
    ERR_STACK_DIM,
    ERR_STACK_PARAM,
    ERR_STACK_PROPAGATOR,
    DiscreteHyperbolicPropagator,
    InstitutionalBenchmarkUniverse,
    StackelbergConfig,
    StackelbergPayoffEngine,
    StackelbergPayoffTensorConstructor,
)


class TestStackelbergConfig:
    """Test suite for StackelbergConfig validation."""

    def test_default_config_valid(self) -> None:
        """Verify default configuration instantiates successfully."""
        config = StackelbergConfig()
        assert config.execution_horizon == 5
        assert config.hyperbolic_decay_rate == 0.5
        assert config.risk_aversion == 1.0
        assert config.predatory_shading_intensity == 0.25
        assert config.risk_free_rate == 0.0
        assert config.max_weight_per_asset == 0.30
        assert config.gross_leverage_limit == 1.0

    def test_invalid_execution_horizon(self) -> None:
        """Verify horizon < 1 raises ERR_STACK_PARAM."""
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(execution_horizon=0)

    def test_invalid_decay_rate(self) -> None:
        """Verify negative or non-finite decay rate raises ERR_STACK_PARAM."""
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(hyperbolic_decay_rate=-0.1)
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(hyperbolic_decay_rate=float("inf"))

    def test_invalid_risk_aversion(self) -> None:
        """Verify non-positive risk aversion raises ERR_STACK_PARAM."""
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(risk_aversion=0.0)
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(risk_aversion=-1.0)

    def test_invalid_predatory_shading(self) -> None:
        """Verify negative predatory shading raises ERR_STACK_PARAM."""
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(predatory_shading_intensity=-0.5)

    def test_invalid_risk_free_rate(self) -> None:
        """Verify negative risk free rate raises ERR_STACK_PARAM."""
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(risk_free_rate=-0.01)

    def test_invalid_asset_weights(self) -> None:
        """Verify invalid weight bounds or leverage inconsistencies raise ERR_STACK_PARAM."""
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(max_weight_per_asset=0.0)
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(max_weight_per_asset=1.5)
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(gross_leverage_limit=0.0)
        with pytest.raises(ValueError, match=ERR_STACK_PARAM):
            StackelbergConfig(max_weight_per_asset=0.8, gross_leverage_limit=0.5)


class TestDiscreteHyperbolicPropagator:
    """Test suite for DiscreteHyperbolicPropagator."""

    @pytest.mark.parametrize("horizon", [1, 2, 3, 5, 10, 20])
    @pytest.mark.parametrize("kappa", [0.0, 0.01, 0.2, 0.5, 1.0, 2.5, 5.0])
    def test_partition_of_unity(self, horizon: int, kappa: float) -> None:
        """Verify execution weights sum identically to 1.0 across all horizons and kappas."""
        propagator = DiscreteHyperbolicPropagator()
        weights = propagator.compute_schedule_weights(horizon=horizon, decay_rate=kappa)

        assert len(weights) == horizon
        assert np.all(weights > 0.0)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-12)

    def test_monotonic_decay(self) -> None:
        """Verify strictly monotonic front-loading alpha_1 > alpha_2 > ... > alpha_H for kappa > 0."""
        propagator = DiscreteHyperbolicPropagator()
        weights = propagator.compute_schedule_weights(horizon=5, decay_rate=0.8)

        for k in range(len(weights) - 1):
            assert weights[k] > weights[k + 1]

    def test_asymptotic_twap_convergence(self) -> None:
        """Verify weights converge to flat 1/H (TWAP) as kappa -> 0."""
        propagator = DiscreteHyperbolicPropagator()
        h = 5
        weights_zero = propagator.compute_schedule_weights(horizon=h, decay_rate=0.0)
        weights_small = propagator.compute_schedule_weights(horizon=h, decay_rate=1e-6)

        expected = np.full(h, 0.2)
        assert np.allclose(weights_zero, expected, atol=1e-12)
        assert np.allclose(weights_small, expected, atol=1e-5)

    def test_single_bar_horizon(self) -> None:
        """Verify single-bar horizon produces exact [1.0]."""
        propagator = DiscreteHyperbolicPropagator()
        weights = propagator.compute_schedule_weights(horizon=1)
        assert len(weights) == 1
        assert weights[0] == 1.0

    def test_trajectory_and_slices(self) -> None:
        """Verify compute_trajectory and compute_execution_slices match endpoint constraints."""
        propagator = DiscreteHyperbolicPropagator(StackelbergConfig(execution_horizon=4))
        a_0 = np.array([0.1, 0.2, 0.0])
        a_target = np.array([0.25, 0.05, 0.30])

        trajectory = propagator.compute_trajectory(a_0, a_target)
        slices = propagator.compute_execution_slices(a_0, a_target)

        assert trajectory.shape == (5, 3)
        assert np.allclose(trajectory[0], a_0)
        assert np.allclose(trajectory[-1], a_target)

        assert slices.shape == (4, 3)
        assert np.allclose(np.sum(slices, axis=0), a_target - a_0)

    def test_multi_period_friction_reduction(self) -> None:
        """Verify multi-bar slicing reduces peak impact compared to single-bar execution."""
        config_h5 = StackelbergConfig(execution_horizon=5, hyperbolic_decay_rate=0.5)
        propagator_h5 = DiscreteHyperbolicPropagator(config_h5)

        impact_config = MarketImpactConfig(portfolio_value=1_000_000.0)
        impact_engine_h1 = MultiAssetMarketImpactEngine(impact_config)
        impact_engine_h5 = MultiAssetMarketImpactEngine(impact_config)

        # 3 assets with moderate daily volumes
        a_0 = np.zeros(3)
        a_target = np.array([0.25, 0.25, 0.25])
        cov = np.eye(3) * 0.04
        vols = np.array([500_000.0, 500_000.0, 500_000.0])
        sigmas = np.array([0.02, 0.02, 0.02])

        # Single bar execution (H=1)
        res_h1 = impact_engine_h1.evaluate_impact(
            delta_allocations=a_target - a_0,
            covariance=cov,
            daily_dollar_volumes=vols,
            asset_volatilities=sigmas,
            update_state=True,
        )

        # Multi-bar execution (H=5)
        friction_h5, _ = propagator_h5.evaluate_trajectory_friction(
            current_weights=a_0,
            target_weights=a_target,
            impact_engine=impact_engine_h5,
            covariance=cov,
            daily_dollar_volumes=vols,
            asset_volatilities=sigmas,
        )

        # Multi-bar execution must have lower peak friction due to sub-linear 3/2 power and decay
        assert friction_h5 < res_h1.total_cost

    def test_propagator_error_handling(self) -> None:
        """Verify invalid inputs raise appropriate diagnostic codes."""
        propagator = DiscreteHyperbolicPropagator()
        with pytest.raises(ValueError, match=ERR_STACK_PROPAGATOR):
            propagator.compute_schedule_weights(horizon=0)
        with pytest.raises(ValueError, match=ERR_STACK_PROPAGATOR):
            propagator.compute_schedule_weights(decay_rate=-0.5)
        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            propagator.compute_trajectory(np.zeros(2), np.zeros(3))
        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            propagator.compute_execution_slices(np.zeros(2), np.zeros(3))


class TestStackelbergPayoffEngine:
    """Test suite for StackelbergPayoffEngine."""

    @pytest.fixture
    def setup_data(self) -> dict[str, np.ndarray]:
        """Create standard multi-asset testing environment."""
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.03, 0.008], [0.005, 0.008, 0.05]])
        mu = np.array([0.08, 0.06, 0.09])
        vols = np.array([1_000_000.0, 800_000.0, 1_200_000.0])
        sigmas = np.array([0.02, 0.018, 0.022])
        return {"cov": cov, "mu": mu, "vols": vols, "sigmas": sigmas}

    def test_zero_rebalance_utility(self, setup_data: dict[str, np.ndarray]) -> None:
        """Verify zero rebalance (a == a_0) incurs zero friction cost."""
        engine = StackelbergPayoffEngine()
        a = np.array([0.2, 0.2, 0.2])

        u = engine.evaluate_utility(
            allocation=a,
            initial_allocation=a,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
        )

        # Theoretical utility with C(0) = 0
        gross = float(np.dot(a, setup_data["mu"]))
        cash = (1.0 - np.sum(a)) * engine.config.risk_free_rate
        quad = float(np.dot(a, np.dot(setup_data["cov"], a)))
        var_pen = 0.5 * engine.config.risk_aversion * quad
        pred_pen = engine.config.predatory_shading_intensity * quad
        expected_u = gross + cash - var_pen - pred_pen

        assert np.isclose(u, expected_u, atol=1e-10)

    def test_cash_yield_contribution(self, setup_data: dict[str, np.ndarray]) -> None:
        """Verify unallocated capital earns risk-free interest."""
        rf = 0.0002  # per-bar risk-free rate
        engine_rf = StackelbergPayoffEngine(StackelbergConfig(risk_free_rate=rf))
        engine_zero_rf = StackelbergPayoffEngine(StackelbergConfig(risk_free_rate=0.0))

        a = np.array([0.1, 0.1, 0.1])  # 70% cash
        u_rf = engine_rf.evaluate_utility(
            allocation=a,
            initial_allocation=a,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
        )
        u_zero = engine_zero_rf.evaluate_utility(
            allocation=a,
            initial_allocation=a,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
        )

        expected_cash_diff = 0.7 * rf
        assert np.isclose(u_rf - u_zero, expected_cash_diff, atol=1e-10)

    def test_predatory_shading_drag(self, setup_data: dict[str, np.ndarray]) -> None:
        """Verify increasing theta_pred penalizes risky allocations without affecting cash."""
        engine_low = StackelbergPayoffEngine(StackelbergConfig(predatory_shading_intensity=0.0))
        engine_high = StackelbergPayoffEngine(StackelbergConfig(predatory_shading_intensity=0.5))

        # Risky allocation
        a_risky = np.array([0.3, 0.3, 0.3])
        u_low_risky = engine_low.evaluate_utility(
            allocation=a_risky,
            initial_allocation=a_risky,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
        )
        u_high_risky = engine_high.evaluate_utility(
            allocation=a_risky,
            initial_allocation=a_risky,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
        )
        assert u_high_risky < u_low_risky

        # Pure cash allocation
        a_cash = np.zeros(3)
        u_low_cash = engine_low.evaluate_utility(
            allocation=a_cash,
            initial_allocation=a_cash,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
        )
        u_high_cash = engine_high.evaluate_utility(
            allocation=a_cash,
            initial_allocation=a_cash,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
        )
        assert np.isclose(u_low_cash, u_high_cash, atol=1e-12)

    def test_analytical_gradient_vs_finite_difference(
        self, setup_data: dict[str, np.ndarray]
    ) -> None:
        """Verify analytical gradient nabla_a U matches numerical finite differences."""
        engine = StackelbergPayoffEngine(
            StackelbergConfig(
                risk_aversion=1.2, predatory_shading_intensity=0.3, risk_free_rate=0.001
            )
        )
        a = np.array([0.15, 0.20, 0.10])
        a_0 = np.array([0.05, 0.10, 0.25])

        analytical_grad = engine.evaluate_gradient(
            allocation=a,
            initial_allocation=a_0,
            expected_returns=setup_data["mu"],
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
            panic_probability=0.2,
        )

        # Central finite difference approximation: (U(a + eps) - U(a - eps)) / (2 * eps)
        eps = 1e-6
        numerical_grad = np.empty_like(analytical_grad)

        for i in range(len(a)):
            a_plus = a.copy()
            a_plus[i] += eps
            u_plus = engine.evaluate_utility(
                allocation=a_plus,
                initial_allocation=a_0,
                expected_returns=setup_data["mu"],
                covariance=setup_data["cov"],
                daily_dollar_volumes=setup_data["vols"],
                asset_volatilities=setup_data["sigmas"],
                panic_probability=0.2,
            )

            a_minus = a.copy()
            a_minus[i] -= eps
            u_minus = engine.evaluate_utility(
                allocation=a_minus,
                initial_allocation=a_0,
                expected_returns=setup_data["mu"],
                covariance=setup_data["cov"],
                daily_dollar_volumes=setup_data["vols"],
                asset_volatilities=setup_data["sigmas"],
                panic_probability=0.2,
            )

            numerical_grad[i] = (u_plus - u_minus) / (2.0 * eps)

        # Verify close agreement
        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-5)

    def test_analytical_hessian_negative_definiteness(
        self, setup_data: dict[str, np.ndarray]
    ) -> None:
        """Verify analytical Hessian H_a U is strictly negative definite (all eigenvalues < 0)."""
        engine = StackelbergPayoffEngine(
            StackelbergConfig(risk_aversion=1.0, predatory_shading_intensity=0.25)
        )
        a = np.array([0.2, 0.15, 0.25])
        a_0 = np.array([0.1, 0.1, 0.1])

        hessian = engine.evaluate_hessian(
            allocation=a,
            initial_allocation=a_0,
            covariance=setup_data["cov"],
            daily_dollar_volumes=setup_data["vols"],
            asset_volatilities=setup_data["sigmas"],
            panic_probability=0.3,
        )

        # 1. Symmetry
        assert np.allclose(hessian, hessian.T, atol=1e-12)

        # 2. Eigenvalues must all be strictly negative (strict concavity)
        eigenvalues = np.linalg.eigvalsh(hessian)
        assert np.all(eigenvalues < 0.0)
        max_eigenval = float(np.max(eigenvalues))
        assert max_eigenval < -1e-4

    def test_payoff_engine_error_handling(self, setup_data: dict[str, np.ndarray]) -> None:
        """Verify dimension mismatch raises ERR_STACK_DIM."""
        engine = StackelbergPayoffEngine()
        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            engine.evaluate_utility(
                allocation=np.zeros(2),  # 2 assets instead of 3
                initial_allocation=np.zeros(3),
                expected_returns=setup_data["mu"],
                covariance=setup_data["cov"],
                daily_dollar_volumes=setup_data["vols"],
                asset_volatilities=setup_data["sigmas"],
            )

        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            engine.evaluate_gradient(
                allocation=np.zeros(3),
                initial_allocation=np.zeros(3),
                expected_returns=np.zeros(2),  # mismatch
                covariance=setup_data["cov"],
                daily_dollar_volumes=setup_data["vols"],
                asset_volatilities=setup_data["sigmas"],
            )

        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            engine.evaluate_hessian(
                allocation=np.zeros(3),
                initial_allocation=np.zeros(3),
                covariance=np.zeros((2, 2)),  # mismatch
                daily_dollar_volumes=setup_data["vols"],
                asset_volatilities=setup_data["sigmas"],
            )


class TestInstitutionalBenchmarkUniverse:
    """Test suite for InstitutionalBenchmarkUniverse."""

    @pytest.fixture
    def setup_env(self) -> dict[str, np.ndarray]:
        """Standard testing assets."""
        cov = np.diag([0.01, 0.04, 0.09, 0.16])
        volatilities = np.sqrt(np.diag(cov))
        adv = np.full(4, 1_000_000.0)
        return {"cov": cov, "sigmas": volatilities, "adv": adv}

    def test_equal_weight_generation(self) -> None:
        """Verify Equal Weight allocation respects concentration caps."""
        config = StackelbergConfig(max_weight_per_asset=0.20)
        bench_gen = InstitutionalBenchmarkUniverse(config)

        # 4 assets -> 1/4 = 0.25, but capped at 0.20
        ew = bench_gen.generate_equal_weight(4)
        assert len(ew) == 4
        assert np.all(ew == 0.20)
        assert np.sum(ew) == 0.80

    def test_risk_parity_generation(self, setup_env: dict[str, np.ndarray]) -> None:
        """Verify Risk Parity assigns higher weight to lower variance assets."""
        config = StackelbergConfig(max_weight_per_asset=0.40)
        bench_gen = InstitutionalBenchmarkUniverse(config)

        rp = bench_gen.generate_risk_parity(setup_env["cov"])
        assert len(rp) == 4
        assert np.all(rp <= 0.40)
        assert np.sum(rp) <= 1.0 + 1e-12

        # Asset 0 has lowest variance (0.01), asset 3 has highest (0.16)
        assert rp[0] > rp[1] > rp[2] > rp[3]

    def test_inverse_volatility_generation(self, setup_env: dict[str, np.ndarray]) -> None:
        """Verify Inverse Volatility assigns higher weight to lower volatility assets."""
        config = StackelbergConfig(max_weight_per_asset=0.50)
        bench_gen = InstitutionalBenchmarkUniverse(config)

        iv = bench_gen.generate_inverse_volatility(setup_env["sigmas"])
        assert len(iv) == 4
        assert np.all(iv <= 0.50)
        assert np.isclose(np.sum(iv), 1.0, atol=1e-12)
        assert iv[0] > iv[1] > iv[2] > iv[3]

    def test_cash_only_generation(self) -> None:
        """Verify cash only allocation is all zeros."""
        bench_gen = InstitutionalBenchmarkUniverse()
        cash = bench_gen.generate_cash_only(4)
        assert np.all(cash == 0.0)

    def test_friction_parity_in_benchmark_evaluation(
        self, setup_env: dict[str, np.ndarray]
    ) -> None:
        """Verify benchmarks incur transaction costs when starting from non-zero a_0."""
        bench_gen = InstitutionalBenchmarkUniverse()
        mu = np.array([0.05, 0.05, 0.05, 0.05])

        # Initial allocation already equal weight
        a_0_ew = np.full(4, 0.25)
        res_from_ew = bench_gen.evaluate_benchmark_universe(
            initial_allocation=a_0_ew,
            expected_returns=mu,
            covariance=setup_env["cov"],
            daily_dollar_volumes=setup_env["adv"],
            asset_volatilities=setup_env["sigmas"],
        )

        # Initial allocation far from equal weight (all in asset 3)
        a_0_far = np.array([0.0, 0.0, 0.0, 1.0])
        res_from_far = bench_gen.evaluate_benchmark_universe(
            initial_allocation=a_0_far,
            expected_returns=mu,
            covariance=setup_env["cov"],
            daily_dollar_volumes=setup_env["adv"],
            asset_volatilities=setup_env["sigmas"],
        )

        # EqualWeight utility must be strictly higher when starting from EW (no friction)
        assert (
            res_from_ew.all_benchmark_utilities["EqualWeight"]
            > res_from_far.all_benchmark_utilities["EqualWeight"]
        )

    def test_cash_wins_in_severe_panic(self, setup_env: dict[str, np.ndarray]) -> None:
        """Verify CashOnly is selected as the ceiling benchmark during severe market panic."""
        bench_gen = InstitutionalBenchmarkUniverse()
        # Severe negative returns in crash
        mu_crash = np.array([-0.15, -0.20, -0.18, -0.25])

        res = bench_gen.evaluate_benchmark_universe(
            initial_allocation=np.zeros(4),
            expected_returns=mu_crash,
            covariance=setup_env["cov"],
            daily_dollar_volumes=setup_env["adv"],
            asset_volatilities=setup_env["sigmas"],
            panic_probability=0.9,
        )

        assert res.best_benchmark_name == "CashOnly"
        assert res.best_benchmark_utility >= 0.0

    def test_benchmark_error_handling(self) -> None:
        """Verify invalid inputs raise ERR_STACK_BENCHMARK."""
        bench_gen = InstitutionalBenchmarkUniverse()
        with pytest.raises(ValueError, match=ERR_STACK_BENCHMARK):
            bench_gen.generate_equal_weight(0)
        with pytest.raises(ValueError, match=ERR_STACK_BENCHMARK):
            bench_gen.generate_risk_parity(np.zeros((3, 2)))  # non-square
        with pytest.raises(ValueError, match=ERR_STACK_BENCHMARK):
            bench_gen.generate_inverse_volatility(np.array([]))
        with pytest.raises(ValueError, match=ERR_STACK_BENCHMARK):
            bench_gen.generate_cash_only(0)


class TestStackelbergPayoffTensorConstructor:
    """Test suite for StackelbergPayoffTensorConstructor."""

    @pytest.fixture
    def setup_regimes(self) -> dict[str, object]:
        """Create multi-regime simulation environment."""
        n_assets = 3
        daily_vols = np.full(n_assets, 1_000_000.0)
        asset_sigmas = np.array([0.02, 0.025, 0.018])

        scenarios = [
            {
                "name": "Absorption",
                "expected_returns": np.array([0.03, 0.02, 0.025]),
                "covariance": np.eye(3) * 0.02,
                "panic_probability": 0.05,
            },
            {
                "name": "Momentum",
                "expected_returns": np.array([0.08, 0.07, 0.06]),
                "covariance": np.eye(3) * 0.03,
                "panic_probability": 0.0,
            },
            {
                "name": "Panic",
                "expected_returns": np.array([-0.10, -0.12, -0.08]),
                "covariance": np.eye(3) * 0.08,
                "panic_probability": 0.85,
            },
        ]

        candidates = np.array(
            [
                [0.2, 0.2, 0.2],  # Diversified
                [0.3, 0.0, 0.0],  # Single asset heavy
                [0.0, 0.0, 0.0],  # 100% Cash
                [0.1, 0.1, 0.1],  # Conservative
            ]
        )

        return {
            "n_assets": n_assets,
            "vols": daily_vols,
            "sigmas": asset_sigmas,
            "scenarios": scenarios,
            "candidates": candidates,
        }

    def test_tensor_construction_and_regret_properties(
        self, setup_regimes: dict[str, object]
    ) -> None:
        """Verify payoff tensor and non-negative regret matrix construction."""
        constructor = StackelbergPayoffTensorConstructor()
        candidates = setup_regimes["candidates"]
        assert isinstance(candidates, np.ndarray)
        scenarios = setup_regimes["scenarios"]
        assert isinstance(scenarios, list)
        vols = setup_regimes["vols"]
        assert isinstance(vols, np.ndarray)
        sigmas = setup_regimes["sigmas"]
        assert isinstance(sigmas, np.ndarray)

        result = constructor.construct_tensor(
            candidate_allocations=candidates,
            initial_allocation=np.zeros(3),
            regime_scenarios=scenarios,
            daily_dollar_volumes=vols,
            asset_volatilities=sigmas,
        )

        p = len(candidates)
        m = len(scenarios)

        # 1. Shape validations
        assert result.payoff_tensor.shape == (p, m)
        assert result.regret_matrix.shape == (p, m)
        assert len(result.benchmark_utilities) == m
        assert len(result.regime_names) == m

        # 2. Strict non-negativity of regret
        assert np.all(result.regret_matrix >= 0.0)

        # 3. Cash candidate regret in panic regime must be zero (Cash is optimal in Panic)
        # Cash candidate is index 2
        panic_regime_idx = 2
        assert np.isclose(result.regret_matrix[2, panic_regime_idx], 0.0, atol=1e-10)

    def test_tensor_error_handling(self, setup_regimes: dict[str, object]) -> None:
        """Verify invalid candidate dimensions raise ERR_STACK_DIM."""
        constructor = StackelbergPayoffTensorConstructor()
        vols = setup_regimes["vols"]
        assert isinstance(vols, np.ndarray)
        sigmas = setup_regimes["sigmas"]
        assert isinstance(sigmas, np.ndarray)
        scenarios = setup_regimes["scenarios"]
        assert isinstance(scenarios, list)

        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            constructor.construct_tensor(
                candidate_allocations=np.zeros((0, 3)),  # 0 candidates
                initial_allocation=np.zeros(3),
                regime_scenarios=scenarios,
                daily_dollar_volumes=vols,
                asset_volatilities=sigmas,
            )

        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            constructor.construct_tensor(
                candidate_allocations=np.zeros(3),  # 1D instead of 2D
                initial_allocation=np.zeros(3),
                regime_scenarios=scenarios,
                daily_dollar_volumes=vols,
                asset_volatilities=sigmas,
            )

        with pytest.raises(ValueError, match=ERR_STACK_DIM):
            constructor.construct_tensor(
                candidate_allocations=np.zeros((2, 3)),
                initial_allocation=np.zeros(3),
                regime_scenarios=[],  # 0 scenarios
                daily_dollar_volumes=vols,
                asset_volatilities=sigmas,
            )
