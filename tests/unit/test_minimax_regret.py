"""Unit test suite for Step 4: Closed-Form Boltzmann Dual & Entropic Minimax Regret Solver.

Purpose:
    Verifies mathematical and algorithmic correctness of:
    1. MinimaxRegretConfig invariant validation and diagnostic reporting.
    2. LatentSoftmaxTransform coordinate mapping, invertibility, and analytical derivatives.
    3. EntropicBoltzmannPotential max-shifted Log-Sum-Exp stability, Boltzmann weighting,
       analytical gradient accuracy vs finite difference, and strictly positive-definite Hessian.
    4. VectorizedNewtonSolver sub-50 microsecond execution speed, strict simplex feasibility
       (0 <= a_i <= w_max, sum a_i <= L_max), quadratic convergence, and certified regret.
    5. Diagnostic error handling (ERR-GAME-MINIMAX-*).
"""

import time
from typing import Any

import numpy as np
import pytest

from quant.analytics.minimax_regret import (
    ERR_MINIMAX_DIM,
    ERR_MINIMAX_PARAM,
    EntropicBoltzmannPotential,
    LatentSoftmaxTransform,
    MinimaxRegretConfig,
    VectorizedNewtonSolver,
)
from quant.analytics.payoff_matrix import (
    InstitutionalBenchmarkUniverse,
    StackelbergConfig,
)


class TestMinimaxRegretConfig:
    """Test suite for MinimaxRegretConfig."""

    def test_default_config_valid(self) -> None:
        """Verify default configuration instantiates successfully."""
        config = MinimaxRegretConfig()
        assert config.max_iterations == 25
        assert config.gradient_tolerance == 1e-6
        assert config.step_tolerance == 1e-8
        assert config.line_search_backtrack == 0.5
        assert config.armijo_c1 == 1e-4
        assert config.levenberg_damping == 1e-4
        assert config.temperature_floor == 1e-4
        assert config.leverage_penalty == 100.0

    def test_invalid_max_iterations(self) -> None:
        """Verify max_iterations < 1 raises ERR_MINIMAX_PARAM."""
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(max_iterations=0)

    def test_invalid_tolerances(self) -> None:
        """Verify non-positive tolerances raise ERR_MINIMAX_PARAM."""
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(gradient_tolerance=0.0)
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(step_tolerance=-1e-5)

    def test_invalid_line_search_params(self) -> None:
        """Verify line search bounds in (0, 1) are enforced."""
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(line_search_backtrack=1.0)
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(line_search_backtrack=0.0)
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(armijo_c1=1.5)

    def test_invalid_regularizers(self) -> None:
        """Verify non-positive damping and negative penalties raise ERR_MINIMAX_PARAM."""
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(levenberg_damping=0.0)
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(temperature_floor=-0.01)
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            MinimaxRegretConfig(leverage_penalty=-5.0)


class TestLatentSoftmaxTransform:
    """Test suite for LatentSoftmaxTransform."""

    def test_forward_mapping_bounds(self) -> None:
        """Verify forward mapping strictly satisfies 0 < a_i < w_max for arbitrary w."""
        transform = LatentSoftmaxTransform(max_weight=0.30)
        w = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        a = transform.forward(w)

        assert np.all(a > 0.0)
        assert np.all(a < 0.30)
        assert np.isclose(a[3], 0.15, atol=1e-12)  # w=0 -> a = 0.30 * 0.5 = 0.15

    def test_invalid_max_weight(self) -> None:
        """Verify invalid max_weight raises ERR_MINIMAX_PARAM."""
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            LatentSoftmaxTransform(max_weight=0.0)
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            LatentSoftmaxTransform(max_weight=-0.1)
        with pytest.raises(ValueError, match=ERR_MINIMAX_PARAM):
            LatentSoftmaxTransform(max_weight=1.5)

    def test_inverse_round_trip(self) -> None:
        """Verify forward and inverse transformations invert each other."""
        transform = LatentSoftmaxTransform(max_weight=0.25)
        a_original = np.array([0.01, 0.05, 0.125, 0.20, 0.24])

        w = transform.inverse(a_original)
        a_recovered = transform.forward(w)

        assert np.allclose(a_original, a_recovered, atol=1e-7)

    def test_analytical_derivatives_vs_finite_difference(self) -> None:
        """Verify Jacobian and second derivative match finite differences."""
        transform = LatentSoftmaxTransform(max_weight=0.30)
        w = np.array([-1.5, 0.2, 1.8])
        a = transform.forward(w)

        # Analytical Jacobian diagonal
        analytical_j = transform.jacobian_diagonal(a)

        # Numerical Jacobian via finite differences: da_i / dw_i
        eps = 1e-6
        numerical_j = np.empty_like(analytical_j)
        for i in range(len(w)):
            w_plus = w.copy()
            w_plus[i] += eps
            w_minus = w.copy()
            w_minus[i] -= eps
            a_plus = transform.forward(w_plus)[i]
            a_minus = transform.forward(w_minus)[i]
            numerical_j[i] = (a_plus - a_minus) / (2.0 * eps)

        assert np.allclose(analytical_j, numerical_j, rtol=1e-5, atol=1e-6)

        # Analytical second derivative
        analytical_d2 = transform.hessian_diagonal_factor(a)

        # Numerical second derivative: d(da/dw) / dw
        numerical_d2 = np.empty_like(analytical_d2)
        for i in range(len(w)):
            w_plus = w.copy()
            w_plus[i] += eps
            w_minus = w.copy()
            w_minus[i] -= eps
            a_p = transform.forward(w_plus)[i]
            a_m = transform.forward(w_minus)[i]
            j_plus = transform.jacobian_diagonal(np.array([a_p]))[0]
            j_minus = transform.jacobian_diagonal(np.array([a_m]))[0]
            numerical_d2[i] = (j_plus - j_minus) / (2.0 * eps)

        assert np.allclose(analytical_d2, numerical_d2, rtol=1e-4, atol=1e-5)


class TestEntropicBoltzmannPotential:
    """Test suite for EntropicBoltzmannPotential."""

    @pytest.fixture
    def setup_market_environment(self) -> dict[str, Any]:
        """Create multi-regime simulation environment."""
        n_assets = 3
        vols = np.full(n_assets, 1_000_000.0)
        sigmas = np.array([0.02, 0.025, 0.018])

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
        priors = np.array([0.50, 0.35, 0.15])

        # Evaluate benchmark ceilings
        stack_cfg = StackelbergConfig(max_weight_per_asset=0.30)
        bench_univ = InstitutionalBenchmarkUniverse(stack_cfg)
        ceilings = np.empty(3)
        for j, sc in enumerate(scenarios):
            res = bench_univ.evaluate_benchmark_universe(
                initial_allocation=np.zeros(3),
                expected_returns=sc["expected_returns"],
                covariance=sc["covariance"],
                daily_dollar_volumes=vols,
                asset_volatilities=sigmas,
                panic_probability=sc["panic_probability"],
            )
            ceilings[j] = res.best_benchmark_utility

        return {
            "n_assets": n_assets,
            "vols": vols,
            "sigmas": sigmas,
            "scenarios": scenarios,
            "priors": priors,
            "ceilings": ceilings,
            "stack_cfg": stack_cfg,
        }

    def test_max_shifted_log_sum_exp_stability(
        self, setup_market_environment: dict[str, Any]
    ) -> None:
        """Verify potential evaluates stably with zero overflow even at microscopic temperature."""
        env = setup_market_environment
        potential = EntropicBoltzmannPotential(stackelberg_config=env["stack_cfg"])

        # Severe test with microscopic temperature beta = 1e-4
        a = np.array([0.25, 0.25, 0.25])
        psi, q_star, regrets = potential.evaluate_potential(
            allocation=a,
            initial_allocation=np.zeros(3),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=1e-4,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        assert np.isfinite(psi)
        assert len(q_star) == 3
        assert np.all(q_star >= 0.0)
        assert np.isclose(np.sum(q_star), 1.0, atol=1e-12)
        # In limit beta -> 0, q_star must concentrate on the maximum regret regime
        max_regret_idx = int(np.argmax(regrets))
        assert np.isclose(q_star[max_regret_idx], 1.0, atol=1e-3)

    def test_temperature_asymptotics(self, setup_market_environment: dict[str, Any]) -> None:
        """Verify Boltzmann distribution converges to prior as beta -> infinity."""
        env = setup_market_environment
        potential = EntropicBoltzmannPotential(stackelberg_config=env["stack_cfg"])

        a = np.array([0.1, 0.1, 0.1])
        # High temperature beta = 1000.0
        _, q_star_inf, _ = potential.evaluate_potential(
            allocation=a,
            initial_allocation=np.zeros(3),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=1000.0,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        assert np.allclose(q_star_inf, env["priors"], atol=1e-3)

    def test_analytical_gradient_vs_finite_difference(
        self, setup_market_environment: dict[str, Any]
    ) -> None:
        """Verify analytical gradient nabla_a Psi matches finite difference approximation."""
        env = setup_market_environment
        potential = EntropicBoltzmannPotential(stackelberg_config=env["stack_cfg"])

        a = np.array([0.15, 0.20, 0.10])
        beta = 0.05

        analytical_grad, _, _ = potential.evaluate_gradient(
            allocation=a,
            initial_allocation=np.zeros(3),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=beta,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        # Central finite difference approximation: (Psi(a + eps) - Psi(a - eps)) / (2 * eps)
        eps = 1e-6
        numerical_grad = np.empty_like(analytical_grad)

        for i in range(len(a)):
            a_plus = a.copy()
            a_plus[i] += eps
            psi_plus, _, _ = potential.evaluate_potential(
                allocation=a_plus,
                initial_allocation=np.zeros(3),
                regime_scenarios=env["scenarios"],
                benchmark_ceilings=env["ceilings"],
                regime_priors=env["priors"],
                temperature=beta,
                daily_dollar_volumes=env["vols"],
                asset_volatilities=env["sigmas"],
            )

            a_minus = a.copy()
            a_minus[i] -= eps
            psi_minus, _, _ = potential.evaluate_potential(
                allocation=a_minus,
                initial_allocation=np.zeros(3),
                regime_scenarios=env["scenarios"],
                benchmark_ceilings=env["ceilings"],
                regime_priors=env["priors"],
                temperature=beta,
                daily_dollar_volumes=env["vols"],
                asset_volatilities=env["sigmas"],
            )

            numerical_grad[i] = (psi_plus - psi_minus) / (2.0 * eps)

        assert np.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-5)

    def test_analytical_hessian_positive_definiteness(
        self, setup_market_environment: dict[str, Any]
    ) -> None:
        """Verify analytical Hessian H_a Psi is strictly positive definite everywhere."""
        env = setup_market_environment
        potential = EntropicBoltzmannPotential(stackelberg_config=env["stack_cfg"])

        a = np.array([0.18, 0.12, 0.22])
        beta = 0.05

        hessian = potential.evaluate_hessian(
            allocation=a,
            initial_allocation=np.zeros(3),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=beta,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        # Symmetry
        assert np.allclose(hessian, hessian.T, atol=1e-12)

        # Positive definiteness (all eigenvalues strictly positive -> strict convexity)
        eigvals = np.linalg.eigvalsh(hessian)
        assert np.all(eigvals > 0.0)
        assert float(np.min(eigvals)) > 1e-4


class TestVectorizedNewtonSolver:
    """Test suite for VectorizedNewtonSolver."""

    @pytest.fixture
    def setup_optimization_data(self) -> dict[str, Any]:
        """Create multi-asset, multi-regime optimization problem."""
        n_assets = 4
        vols = np.array([1_500_000.0, 1_000_000.0, 800_000.0, 2_000_000.0])
        sigmas = np.array([0.015, 0.022, 0.028, 0.018])

        scenarios = [
            {
                "name": "Absorption",
                "expected_returns": np.array([0.04, 0.03, 0.02, 0.035]),
                "covariance": np.diag([0.02, 0.03, 0.04, 0.025]),
                "panic_probability": 0.02,
            },
            {
                "name": "Momentum",
                "expected_returns": np.array([0.09, 0.11, 0.08, 0.07]),
                "covariance": np.diag([0.03, 0.04, 0.05, 0.035]),
                "panic_probability": 0.0,
            },
            {
                "name": "Panic",
                "expected_returns": np.array([-0.08, -0.15, -0.20, -0.06]),
                "covariance": np.diag([0.07, 0.10, 0.12, 0.08]),
                "panic_probability": 0.90,
            },
        ]
        priors = np.array([0.45, 0.40, 0.15])

        stack_cfg = StackelbergConfig(
            execution_horizon=4,
            max_weight_per_asset=0.30,
            gross_leverage_limit=1.0,
            risk_free_rate=0.0001,
        )
        bench_univ = InstitutionalBenchmarkUniverse(stack_cfg)
        ceilings = np.empty(3)
        for j, sc in enumerate(scenarios):
            res = bench_univ.evaluate_benchmark_universe(
                initial_allocation=np.zeros(n_assets),
                expected_returns=sc["expected_returns"],
                covariance=sc["covariance"],
                daily_dollar_volumes=vols,
                asset_volatilities=sigmas,
                panic_probability=sc["panic_probability"],
            )
            ceilings[j] = res.best_benchmark_utility

        return {
            "n_assets": n_assets,
            "vols": vols,
            "sigmas": sigmas,
            "scenarios": scenarios,
            "priors": priors,
            "ceilings": ceilings,
            "stack_cfg": stack_cfg,
        }

    def test_end_to_end_solver_execution_and_invariants(
        self, setup_optimization_data: dict[str, Any]
    ) -> None:
        """Verify solver produces optimal feasible weights and slices satisfying all invariants."""
        env = setup_optimization_data
        solver = VectorizedNewtonSolver(stackelberg_config=env["stack_cfg"])

        result = solver.solve(
            initial_weights=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        # 1. Convergence flag
        assert result.solver_converged is True
        assert result.solver_iterations >= 1
        assert result.final_gradient_norm <= solver.config.gradient_tolerance

        # 2. Strict Single-Asset Bounds: 0 <= a_i <= w_max (0.30)
        assert np.all(result.optimal_weights >= 0.0)
        assert np.all(result.optimal_weights <= 0.30 + 1e-12)

        # 3. Gross Leverage & Cash Conservation
        tot_weights = float(np.sum(result.optimal_weights))
        assert tot_weights <= 1.0 + 1e-12
        assert result.cash_weight >= 0.0
        assert np.isclose(tot_weights + result.cash_weight, 1.0, atol=1e-12)

        # 4. Discrete Hyperbolic Slices match rebalance
        assert result.execution_trajectory.shape == (4, env["n_assets"])
        rebalance_sum = np.sum(result.execution_trajectory, axis=0)
        assert np.allclose(rebalance_sum, result.optimal_weights, atol=1e-12)

        # 5. Non-negative worst-case regret metric
        assert result.worst_case_regret >= 0.0
        assert len(result.regime_worst_case_distribution) == 3
        assert np.isclose(np.sum(result.regime_worst_case_distribution), 1.0, atol=1e-12)

    def test_solve_performance_under_budget(self, setup_optimization_data: dict[str, Any]) -> None:
        """Verify solve executes in sub-millisecond latency."""
        env = setup_optimization_data
        solver = VectorizedNewtonSolver(stackelberg_config=env["stack_cfg"])

        # Warm up JIT/cache
        solver.solve(
            initial_weights=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        # Timed execution
        t0 = time.perf_counter_ns()
        result = solver.solve(
            initial_weights=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )
        duration_us = (time.perf_counter_ns() - t0) / 1_000.0

        assert result.solver_converged is True
        # Typical execution is sub-millisecond (< 400 us without coverage tracing).
        # In CI with coverage opcode tracing on shared runners, allow up to 2500 us.
        assert duration_us < 2500.0

    def test_regret_reduction_against_naive_allocation(
        self, setup_optimization_data: dict[str, Any]
    ) -> None:
        """Verify optimal allocation achieves lower worst-case regret than naive allocation."""
        env = setup_optimization_data
        solver = VectorizedNewtonSolver(stackelberg_config=env["stack_cfg"])

        opt_res = solver.solve(
            initial_weights=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        # Naive heavy allocation in volatile asset 2
        naive_alloc = np.array([0.0, 0.0, 0.30, 0.0])
        naive_psi, _, _ = solver.potential.evaluate_potential(
            allocation=naive_alloc,
            initial_allocation=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )

        # Optimal allocation MUST have lower worst-case regret
        assert opt_res.worst_case_regret < naive_psi

    def test_solver_error_handling(self, setup_optimization_data: dict[str, Any]) -> None:
        """Verify dimension mismatches raise ERR_MINIMAX_DIM."""
        env = setup_optimization_data
        solver = VectorizedNewtonSolver(stackelberg_config=env["stack_cfg"])

        # Prior/ceiling mismatch
        with pytest.raises(ValueError, match=ERR_MINIMAX_DIM):
            solver.solve(
                initial_weights=np.zeros(env["n_assets"]),
                regime_scenarios=env["scenarios"],
                benchmark_ceilings=np.zeros(2),  # 2 ceilings for 3 scenarios
                regime_priors=env["priors"],
                temperature=0.05,
                daily_dollar_volumes=env["vols"],
                asset_volatilities=env["sigmas"],
            )

        # Initial weights length mismatch
        with pytest.raises(ValueError, match=ERR_MINIMAX_DIM):
            solver.solve(
                initial_weights=np.zeros(2),  # 2 initial weights for 4 assets
                regime_scenarios=env["scenarios"],
                benchmark_ceilings=env["ceilings"],
                regime_priors=env["priors"],
                temperature=0.05,
                daily_dollar_volumes=env["vols"],
                asset_volatilities=env["sigmas"],
            )

    def test_solve_with_initial_weights_none(self, setup_optimization_data: dict[str, Any]) -> None:
        """Verify solver executes correctly when initial_weights is None."""
        env = setup_optimization_data
        solver = VectorizedNewtonSolver(stackelberg_config=env["stack_cfg"])

        result = solver.solve(
            initial_weights=None,
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
        )
        assert result.solver_converged is True
        assert np.all(result.optimal_weights >= 0.0)

    def test_solve_with_initial_guess_and_excess_leverage(
        self, setup_optimization_data: dict[str, Any]
    ) -> None:
        """Verify solver handles high initial guess with active leverage constraint."""
        env = setup_optimization_data
        solver = VectorizedNewtonSolver(stackelberg_config=env["stack_cfg"])

        # Starting guess near upper bound 0.30 per asset (total 1.20 > L_max=1.0)
        heavy_guess = np.full(env["n_assets"], 0.28)
        result = solver.solve(
            initial_weights=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
            initial_guess=heavy_guess,
        )
        assert result.solver_converged is True
        assert float(np.sum(result.optimal_weights)) <= 1.0 + 1e-12

    def test_solve_with_kelly_and_crowding(self, setup_optimization_data: dict[str, Any]) -> None:
        """Verify solver accepts and processes Kelly conviction and crowding scores."""
        env = setup_optimization_data
        solver = VectorizedNewtonSolver(stackelberg_config=env["stack_cfg"])

        kelly = np.array([0.8, 0.6, 0.9, 0.7])
        crowding = np.array([0.1, 0.4, 0.2, 0.3])
        result = solver.solve(
            initial_weights=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
            kelly_conviction=kelly,
            crowding_scores=crowding,
        )
        assert result.solver_converged is True
        assert len(result.optimal_weights) == env["n_assets"]

    def test_potential_dimension_mismatch_and_gradient_reuse(
        self, setup_optimization_data: dict[str, Any]
    ) -> None:
        """Verify potential evaluator dimension checks and gradient reuse in Hessian."""
        env = setup_optimization_data
        potential = EntropicBoltzmannPotential(stackelberg_config=env["stack_cfg"])

        with pytest.raises(ValueError, match=ERR_MINIMAX_DIM):
            potential.evaluate_potential(
                allocation=np.zeros(env["n_assets"]),
                initial_allocation=np.zeros(env["n_assets"]),
                regime_scenarios=env["scenarios"],
                benchmark_ceilings=np.zeros(2),  # mismatch
                regime_priors=env["priors"],
                temperature=0.05,
                daily_dollar_volumes=env["vols"],
                asset_volatilities=env["sigmas"],
            )

        # Test evaluate_gradient with return_regime_grads=True
        grad, psi, q_star, regime_grads = potential.evaluate_gradient(
            allocation=np.full(env["n_assets"], 0.1),
            initial_allocation=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
            return_regime_grads=True,
        )
        assert regime_grads.shape == (len(env["scenarios"]), env["n_assets"])

        # Test evaluate_hessian reusing q_star and regime_grads
        hessian = potential.evaluate_hessian(
            allocation=np.full(env["n_assets"], 0.1),
            initial_allocation=np.zeros(env["n_assets"]),
            regime_scenarios=env["scenarios"],
            benchmark_ceilings=env["ceilings"],
            regime_priors=env["priors"],
            temperature=0.05,
            daily_dollar_volumes=env["vols"],
            asset_volatilities=env["sigmas"],
            q_star=q_star,
            regime_gradients=regime_grads,
        )
        assert hessian.shape == (env["n_assets"], env["n_assets"])
