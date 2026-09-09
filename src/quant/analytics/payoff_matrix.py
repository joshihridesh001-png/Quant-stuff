"""Stackelberg Leader-Follower Trajectory and Payoff Tensor Engine.

Purpose:
    Implements Step 3 of Phase 3 (Entropic Distributionally Robust Stackelberg Engine).
    Provides multi-bar discrete hyperbolic execution scheduling, continuous quadratic-bilinear
    Stackelberg payoff evaluation with predatory quote shading, friction-consistent institutional
    benchmark generation, and non-negative regret tensor computation across Bayesian market regimes.

Dependencies:
    - numpy: Vectorized linear algebra, hyperbolic functions, and matrix transformations.
    - quant.analytics.market_impact: Multi-asset cross-impact engine with 3/2-power potential.

Structural Relationship:
    - Consumes regime covariance Sigma_j and expectations mu_j from Step 1 (regimes.py).
    - Consumes market friction C(Delta a) and gradient nabla C from Step 2 (market_impact.py).
    - Feeds regret matrix R_{i, j} and payoff derivatives to Step 4 (minimax_regret.py).

Invariants:
    - Hyperbolic execution weights sum identically to 1.0 (sum_{k=1}^H alpha_k == 1.0).
    - Execution weights are strictly positive (alpha_k > 0 for all k).
    - Payoff Hessian is strictly negative definite (H_a U < 0), guaranteeing global concavity.
    - Regret entries are strictly non-negative (R_{i, j} >= 0.0 everywhere).
    - All benchmarks satisfy single-asset caps (b_i <= w_max) and leverage limits (sum b_i <= 1.0).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from quant.analytics.market_impact import (
    MarketImpactConfig,
    MarketImpactResult,
    MultiAssetMarketImpactEngine,
)

# Diagnostic Error Codes
ERR_STACK_PARAM = "ERR-GAME-STACK-001"
ERR_STACK_PROPAGATOR = "ERR-GAME-STACK-002"
ERR_STACK_DIM = "ERR-GAME-STACK-003"
ERR_STACK_BENCHMARK = "ERR-GAME-STACK-004"


@dataclass(frozen=True)
class StackelbergConfig:
    """Immutable configuration for Stackelberg trajectory scheduling and payoff evaluation.

    Attributes:
        execution_horizon: Number of bars H >= 1 over which portfolio rebalancing is sliced.
        hyperbolic_decay_rate: Curvature parameter kappa >= 0.0 governing front-loading.
        risk_aversion: Portfolio variance risk aversion coefficient gamma > 0.0.
        predatory_shading_intensity: Market maker quote-shading coefficient theta_pred >= 0.0.
        risk_free_rate: Per-bar risk-free rate of return r_f >= 0.0 on unallocated cash.
        max_weight_per_asset: Maximum single-asset concentration cap w_max in (0.0, 1.0].
        gross_leverage_limit: Maximum sum of asset allocations L_max in (0.0, 1.0].
    """

    execution_horizon: int = 5
    hyperbolic_decay_rate: float = 0.5
    risk_aversion: float = 1.0
    predatory_shading_intensity: float = 0.25
    risk_free_rate: float = 0.0
    max_weight_per_asset: float = 0.30
    gross_leverage_limit: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration invariants.

        Raises:
            ValueError: If any parameter violates physical or economic boundaries.
        """
        if self.execution_horizon < 1:
            raise ValueError(
                f"{ERR_STACK_PARAM}: execution_horizon must be an integer >= 1, "
                f"got {self.execution_horizon}"
            )
        if not np.isfinite(self.hyperbolic_decay_rate) or self.hyperbolic_decay_rate < 0.0:
            raise ValueError(
                f"{ERR_STACK_PARAM}: hyperbolic_decay_rate must be non-negative and finite, "
                f"got {self.hyperbolic_decay_rate}"
            )
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0.0:
            raise ValueError(
                f"{ERR_STACK_PARAM}: risk_aversion must be strictly positive, "
                f"got {self.risk_aversion}"
            )
        if (
            not np.isfinite(self.predatory_shading_intensity)
            or self.predatory_shading_intensity < 0.0
        ):
            raise ValueError(
                f"{ERR_STACK_PARAM}: predatory_shading_intensity must be non-negative, "
                f"got {self.predatory_shading_intensity}"
            )
        if not np.isfinite(self.risk_free_rate) or self.risk_free_rate < 0.0:
            raise ValueError(
                f"{ERR_STACK_PARAM}: risk_free_rate must be non-negative, got {self.risk_free_rate}"
            )
        if not (0.0 < self.max_weight_per_asset <= 1.0):
            raise ValueError(
                f"{ERR_STACK_PARAM}: max_weight_per_asset must be in (0.0, 1.0], "
                f"got {self.max_weight_per_asset}"
            )
        if not (0.0 < self.gross_leverage_limit <= 1.0):
            raise ValueError(
                f"{ERR_STACK_PARAM}: gross_leverage_limit must be in (0.0, 1.0], "
                f"got {self.gross_leverage_limit}"
            )
        if self.max_weight_per_asset > self.gross_leverage_limit:
            raise ValueError(
                f"{ERR_STACK_PARAM}: max_weight_per_asset ({self.max_weight_per_asset}) "
                f"cannot exceed gross_leverage_limit ({self.gross_leverage_limit})"
            )


@dataclass(frozen=True)
class BenchmarkEvaluationResult:
    """Immutable result from evaluating institutional benchmark universe under regime conditions.

    Attributes:
        best_benchmark_name: Name of the optimal benchmark providing the utility ceiling.
        best_benchmark_weights: Asset allocation vector of the ceiling benchmark.
        best_benchmark_utility: Maximum net utility value U_bench^*(s_j).
        all_benchmark_utilities: Map from benchmark name to its evaluated net utility.
        all_benchmark_weights: Map from benchmark name to its asset allocation vector.
    """

    best_benchmark_name: str
    best_benchmark_weights: np.ndarray
    best_benchmark_utility: float
    all_benchmark_utilities: dict[str, float]
    all_benchmark_weights: dict[str, np.ndarray]


@dataclass(frozen=True)
class PayoffTensorResult:
    """Immutable result containing candidate payoff tensor and non-negative regret matrix.

    Attributes:
        payoff_tensor: 2D array of shape (P, M) containing net utilities U(a_i, s_j).
        benchmark_utilities: 1D array of shape (M,) containing ceiling utilities U_bench^*(s_j).
        regret_matrix: 2D array of shape (P, M) containing non-negative regret R_{i, j} >= 0.
        regime_names: List of M regime identifier labels.
        candidate_allocations: 2D array of shape (P, N) containing evaluated actions.
    """

    payoff_tensor: np.ndarray
    benchmark_utilities: np.ndarray
    regret_matrix: np.ndarray
    regime_names: list[str]
    candidate_allocations: np.ndarray


class DiscreteHyperbolicPropagator:
    """Discrete Monotone Hyperbolic Execution Trajectory Propagator.

    Slices a portfolio transition Delta a = a - a_0 across H discrete bars using a
    telescoping hyperbolic schedule:
        alpha_k = [sinh(kappa * (H - k + 1)) - sinh(kappa * (H - k))] / sinh(kappa * H)

    Properties:
        - Exact Partition of Unity: sum_{k=1}^H alpha_k == 1.0 (telescoping sum identity).
        - Strict Positivity: alpha_k > 0 for all k in {1, ..., H}.
        - Monotonic Decay: alpha_1 > alpha_2 > ... > alpha_H > 0 for kappa > 0.
        - Asymptotic TWAP Convergence: lim_{kappa -> 0} alpha_k = 1 / H.
    """

    def __init__(self, config: StackelbergConfig | None = None) -> None:
        """Initialize propagator with optional configuration.

        Args:
            config: Stackelberg configuration dataclass. Uses defaults if omitted.
        """
        self.config = config or StackelbergConfig()

    def compute_schedule_weights(
        self,
        horizon: int | None = None,
        decay_rate: float | None = None,
    ) -> np.ndarray:
        """Compute normalized execution slice weights alpha_k for k = 1, ..., H.

        Args:
            horizon: Number of bars H. Defaults to config.execution_horizon.
            decay_rate: Curvature kappa. Defaults to config.hyperbolic_decay_rate.

        Returns:
            1D array of length H containing normalized weights summing to 1.0.

        Raises:
            ValueError: If horizon < 1 or decay_rate < 0.
        """
        h = horizon if horizon is not None else self.config.execution_horizon
        kappa = decay_rate if decay_rate is not None else self.config.hyperbolic_decay_rate

        if h < 1:
            raise ValueError(f"{ERR_STACK_PROPAGATOR}: Execution horizon must be >= 1, got {h}")
        if not np.isfinite(kappa) or kappa < 0.0:
            raise ValueError(
                f"{ERR_STACK_PROPAGATOR}: Decay rate kappa must be non-negative and finite, "
                f"got {kappa}"
            )

        # Single bar degenerate case: immediate complete execution
        if h == 1:
            return np.ones(1, dtype=np.float64)

        # Flat TWAP fallback when kappa is near zero to prevent 0 / 0 floating point division
        if kappa < 1e-5:
            return np.full(h, 1.0 / float(h), dtype=np.float64)

        # Exact, overflow-free exponential formulation of the telescoping hyperbolic schedule:
        # alpha_k = [sinh(kappa * (H - k + 1)) - sinh(kappa * (H - k))] / sinh(kappa * H)
        #         = (1 - e^{-kappa}) * [e^{-kappa * (k - 1)} + e^{-kappa * (2H - k)}] / (1 - e^{-2 * kappa * H})
        # Every exponent is <= 0, completely eliminating floating-point overflow for all kappa and H.
        k_indices = np.arange(1, h + 1, dtype=np.float64)
        term1 = np.exp(-kappa * (k_indices - 1.0))
        term2 = np.exp(-kappa * (2.0 * float(h) - k_indices))

        num = (1.0 - np.exp(-kappa)) * (term1 + term2)
        den = 1.0 - np.exp(-2.0 * kappa * float(h))

        raw_weights = num / den

        # Mathematical cleanup: ensure strict positivity and exact unity sum
        clean_weights = np.maximum(raw_weights, 1e-30)
        weight_sum = np.sum(clean_weights)
        clean_weights /= weight_sum

        return np.asarray(clean_weights, dtype=np.float64)

    def compute_trajectory(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Compute sequential portfolio states across execution horizon.

        Args:
            current_weights: 1D array of current asset weights a_0 (length N).
            target_weights: 1D array of target asset weights a (length N).
            horizon: Number of bars H. Defaults to config.execution_horizon.

        Returns:
            2D array of shape (H + 1, N) where row 0 is a_0 and row H is a.

        Raises:
            ValueError: If input dimensions do not match.
        """
        a_0 = np.asarray(current_weights, dtype=np.float64).flatten()
        a_target = np.asarray(target_weights, dtype=np.float64).flatten()

        if len(a_0) != len(a_target):
            raise ValueError(
                f"{ERR_STACK_DIM}: current_weights length ({len(a_0)}) does not match "
                f"target_weights length ({len(a_target)})"
            )

        alpha = self.compute_schedule_weights(horizon=horizon)
        cum_alpha = np.cumsum(alpha)
        # Force final entry to exactly 1.0
        cum_alpha[-1] = 1.0

        delta_total = a_target - a_0
        n_assets = len(a_0)
        h = len(alpha)

        trajectory = np.empty((h + 1, n_assets), dtype=np.float64)
        trajectory[0] = a_0

        for step in range(h):
            trajectory[step + 1] = a_0 + cum_alpha[step] * delta_total

        return trajectory

    def compute_execution_slices(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Compute per-bar rebalancing trade slices Delta a_k = alpha_k * (a - a_0).

        Args:
            current_weights: 1D array of current asset weights a_0 (length N).
            target_weights: 1D array of target asset weights a (length N).
            horizon: Number of bars H. Defaults to config.execution_horizon.

        Returns:
            2D array of shape (H, N) containing bar-by-bar execution slices.

        Raises:
            ValueError: If input dimensions do not match.
        """
        a_0 = np.asarray(current_weights, dtype=np.float64).flatten()
        a_target = np.asarray(target_weights, dtype=np.float64).flatten()

        if len(a_0) != len(a_target):
            raise ValueError(
                f"{ERR_STACK_DIM}: current_weights length ({len(a_0)}) does not match "
                f"target_weights length ({len(a_target)})"
            )

        alpha = self.compute_schedule_weights(horizon=horizon)
        delta_total = a_target - a_0
        return np.outer(alpha, delta_total)

    def evaluate_trajectory_friction(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        impact_engine: MultiAssetMarketImpactEngine,
        covariance: np.ndarray,
        daily_dollar_volumes: np.ndarray,
        asset_volatilities: np.ndarray,
        panic_probability: float = 0.0,
        kelly_conviction: np.ndarray | None = None,
        crowding_scores: np.ndarray | None = None,
    ) -> tuple[float, list[MarketImpactResult]]:
        """Evaluate cumulative multi-period market impact friction along the trajectory.

        Simulates multi-bar order book depletion dynamics across k = 1, ..., H.
        At each bar k, the depletion buffer B_{t-1} decays by resilience rho before
        absorbing trade slice Delta a_k.

        Args:
            current_weights: 1D array of current asset weights a_0.
            target_weights: 1D array of target asset weights a.
            impact_engine: Configured MultiAssetMarketImpactEngine instance.
            covariance: Asset return covariance matrix Sigma (N x N).
            daily_dollar_volumes: Daily dollar volume vector ADV (length N).
            asset_volatilities: Asset volatility vector sigma (length N).
            panic_probability: Probability of panic regime pi_panic in [0, 1].
            kelly_conviction: Optional Kelly confidence scores.
            crowding_scores: Optional crowding indicators.

        Returns:
            Tuple of:
                - cumulative_friction_cost: Total multi-period friction sum_{k=1}^H C_k.
                - bar_results: List of MarketImpactResult payloads for each bar k.
        """
        slices = self.compute_execution_slices(current_weights, target_weights)
        h = len(slices)

        total_friction = 0.0
        bar_results: list[MarketImpactResult] = []

        for k in range(h):
            slice_k = slices[k]
            res = impact_engine.evaluate_impact(
                delta_allocations=slice_k,
                covariance=covariance,
                daily_dollar_volumes=daily_dollar_volumes,
                asset_volatilities=asset_volatilities,
                panic_probability=panic_probability,
                kelly_conviction=kelly_conviction,
                crowding_scores=crowding_scores,
                update_state=True,
            )
            total_friction += res.total_cost
            bar_results.append(res)

        return total_friction, bar_results


class StackelbergPayoffEngine:
    """Continuous Quadratic-Bilinear Stackelberg Payoff Functional Engine.

    Evaluates institutional portfolio utility against predatory algorithmic flow:
        U(a, s_j) = a^T mu_j + a_cash * r_f - (gamma / 2) * a^T Sigma_j a
                    - a^T M_pred a - C(a - a_0, s_j)

    Where:
        - a_cash = max(0, 1 - sum(a_i)) is the unallocated risk-free cash allocation.
        - M_pred = theta_pred * Sigma_j represents market maker quote shading against leader flow.
        - C(Delta a, s_j) is the multi-asset cross-impact and 3/2-power transient friction cost.

    Properties:
        - Exact Analytical Gradient: nabla_a U = mu_j - r_f * 1 - (gamma + 2 * theta_pred) Sigma_j a - nabla C.
        - Negative Definite Hessian: H_a U = -(gamma + 2 * theta_pred) Sigma_j - H C < 0,
          guaranteeing global concavity and unique optimal allocations.
    """

    def __init__(
        self,
        config: StackelbergConfig | None = None,
        impact_engine: MultiAssetMarketImpactEngine | None = None,
    ) -> None:
        """Initialize payoff engine.

        Args:
            config: Stackelberg configuration dataclass. Uses defaults if omitted.
            impact_engine: MultiAssetMarketImpactEngine instance. Instantiates default if omitted.
        """
        self.config = config or StackelbergConfig()
        self.impact_engine = impact_engine or MultiAssetMarketImpactEngine(MarketImpactConfig())
        self.propagator = DiscreteHyperbolicPropagator(self.config)

    def evaluate_utility(
        self,
        allocation: np.ndarray,
        initial_allocation: np.ndarray | None,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        daily_dollar_volumes: np.ndarray,
        asset_volatilities: np.ndarray,
        panic_probability: float = 0.0,
        kelly_conviction: np.ndarray | None = None,
        crowding_scores: np.ndarray | None = None,
    ) -> float:
        """Evaluate net continuous Stackelberg utility U(a, s_j).

        Args:
            allocation: 1D array of target asset weights a (length N).
            initial_allocation: 1D array of starting asset weights a_0 (length N). If None, defaults to 0.
            expected_returns: 1D array of expected returns mu_j (length N).
            covariance: 2D array of asset covariance Sigma_j (N x N).
            daily_dollar_volumes: 1D array of daily dollar volumes (length N).
            asset_volatilities: 1D array of asset volatilities (length N).
            panic_probability: Probability of panic regime pi_panic in [0, 1].
            kelly_conviction: Optional Kelly confidence scores.
            crowding_scores: Optional crowding indicators.

        Returns:
            Scalar net utility value in portfolio return space.
        """
        a = np.asarray(allocation, dtype=np.float64).flatten()
        n = len(a)

        a_0 = (
            np.zeros(n, dtype=np.float64)
            if initial_allocation is None
            else np.asarray(initial_allocation, dtype=np.float64).flatten()
        )
        mu = np.asarray(expected_returns, dtype=np.float64).flatten()
        sigma_mat = np.asarray(covariance, dtype=np.float64)

        if len(a_0) != n or len(mu) != n or sigma_mat.shape != (n, n):
            raise ValueError(
                f"{ERR_STACK_DIM}: Dimensional mismatch: a={n}, a_0={len(a_0)}, "
                f"mu={len(mu)}, sigma={sigma_mat.shape}"
            )

        # 1. Gross Return with Risk-Free Cash Component
        gross_asset_return = float(np.dot(a, mu))
        cash_weight = max(0.0, 1.0 - float(np.sum(a)))
        cash_return = cash_weight * self.config.risk_free_rate
        total_return = gross_asset_return + cash_return

        # 2. Portfolio Variance & Follower Predatory Quote Shading
        # quad_term = a^T Sigma a
        quad_term = float(np.dot(a, np.dot(sigma_mat, a)))
        variance_penalty = 0.5 * self.config.risk_aversion * quad_term
        predatory_penalty = self.config.predatory_shading_intensity * quad_term

        # 3. Market Friction Cost C(a - a_0)
        delta_a = a - a_0
        impact_result = self.impact_engine.evaluate_impact(
            delta_allocations=delta_a,
            covariance=sigma_mat,
            daily_dollar_volumes=daily_dollar_volumes,
            asset_volatilities=asset_volatilities,
            panic_probability=panic_probability,
            kelly_conviction=kelly_conviction,
            crowding_scores=crowding_scores,
            update_state=False,
        )
        friction_cost = impact_result.total_cost

        # Net Utility: U(a, s_j)
        return total_return - variance_penalty - predatory_penalty - friction_cost

    def evaluate_gradient(
        self,
        allocation: np.ndarray,
        initial_allocation: np.ndarray | None,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        daily_dollar_volumes: np.ndarray,
        asset_volatilities: np.ndarray,
        panic_probability: float = 0.0,
        kelly_conviction: np.ndarray | None = None,
        crowding_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate exact analytical gradient nabla_a U(a, s_j).

        nabla_a U = mu_j - r_f * 1 - (gamma + 2 * theta_pred) * Sigma_j a - nabla C(a - a_0)

        Args:
            Same as evaluate_utility.

        Returns:
            1D array of length N containing analytical utility gradient.
        """
        a = np.asarray(allocation, dtype=np.float64).flatten()
        n = len(a)

        a_0 = (
            np.zeros(n, dtype=np.float64)
            if initial_allocation is None
            else np.asarray(initial_allocation, dtype=np.float64).flatten()
        )
        mu = np.asarray(expected_returns, dtype=np.float64).flatten()
        sigma_mat = np.asarray(covariance, dtype=np.float64)

        if len(a_0) != n or len(mu) != n or sigma_mat.shape != (n, n):
            raise ValueError(
                f"{ERR_STACK_DIM}: Dimensional mismatch: a={n}, a_0={len(a_0)}, "
                f"mu={len(mu)}, sigma={sigma_mat.shape}"
            )

        # 1. Gross Return Gradient: d(a^T mu + (1 - sum a_i) * r_f) / d a = mu - r_f * 1
        grad_return = mu - self.config.risk_free_rate * np.ones(n, dtype=np.float64)

        # 2. Combined Variance & Follower Predatory Shading Gradient
        # d / da [ (gamma / 2 + theta_pred) a^T Sigma a ] = (gamma + 2 * theta_pred) Sigma a
        effective_penalty_coef = (
            self.config.risk_aversion + 2.0 * self.config.predatory_shading_intensity
        )
        grad_variance = effective_penalty_coef * np.dot(sigma_mat, a)

        # 3. Market Friction Gradient: dC(Delta a) / da
        delta_a = a - a_0
        impact_result = self.impact_engine.evaluate_impact(
            delta_allocations=delta_a,
            covariance=sigma_mat,
            daily_dollar_volumes=daily_dollar_volumes,
            asset_volatilities=asset_volatilities,
            panic_probability=panic_probability,
            kelly_conviction=kelly_conviction,
            crowding_scores=crowding_scores,
            update_state=False,
        )
        grad_friction = impact_result.gradient

        return np.asarray(grad_return - grad_variance - grad_friction, dtype=np.float64)

    def evaluate_hessian(
        self,
        allocation: np.ndarray,
        initial_allocation: np.ndarray | None,
        covariance: np.ndarray,
        daily_dollar_volumes: np.ndarray,
        asset_volatilities: np.ndarray,
        panic_probability: float = 0.0,
        crowding_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate exact analytical Hessian matrix H_a U(a, s_j).

        H_a U = - (gamma + 2 * theta_pred) * Sigma_j - H_friction
        Where H_friction = Lambda_cross + diag(H_transient).

        Since Sigma_j > 0, Lambda_cross > 0, and H_transient >= 0,
        H_a U < 0 is strictly negative definite everywhere.

        Args:
            allocation: 1D array of target asset weights a (length N).
            initial_allocation: 1D array of starting asset weights a_0 (length N).
            covariance: 2D array of asset covariance Sigma_j (N x N).
            daily_dollar_volumes: 1D array of daily dollar volumes (length N).
            asset_volatilities: 1D array of asset volatilities (length N).
            panic_probability: Probability of panic regime pi_panic in [0, 1].
            crowding_scores: Optional crowding indicators.

        Returns:
            2D symmetric negative-definite array of shape (N, N).
        """
        a = np.asarray(allocation, dtype=np.float64).flatten()
        n = len(a)

        a_0 = (
            np.zeros(n, dtype=np.float64)
            if initial_allocation is None
            else np.asarray(initial_allocation, dtype=np.float64).flatten()
        )
        sigma_mat = np.asarray(covariance, dtype=np.float64)
        vols = np.asarray(daily_dollar_volumes, dtype=np.float64).flatten()
        sigmas = np.asarray(asset_volatilities, dtype=np.float64).flatten()

        if len(a_0) != n or sigma_mat.shape != (n, n) or len(vols) != n or len(sigmas) != n:
            raise ValueError(f"{ERR_STACK_DIM}: Dimensional mismatch in Hessian evaluation")

        # 1. Variance and Follower Hessian Component: -(gamma + 2 * theta_pred) * Sigma
        effective_penalty_coef = (
            self.config.risk_aversion + 2.0 * self.config.predatory_shading_intensity
        )
        hessian_variance = effective_penalty_coef * sigma_mat

        # 2. Permanent Cross-Impact Hessian: Lambda_cross
        lambda_cross = self.impact_engine.cross_constructor.build_matrix(
            covariance=sigma_mat,
            daily_dollar_volumes=vols,
        )

        # 3. 3/2-Power Transient Hessian Diagonal
        delta_a = a - a_0
        bar_dollar_volumes = vols * self.impact_engine.config.bar_time_fraction
        safe_bar_vols = np.maximum(bar_dollar_volumes, 1.0)
        participation_rates = (self.impact_engine.config.portfolio_value * delta_a) / safe_bar_vols

        depletion_b = self.impact_engine._get_depletion_vector(n)
        effective_u = (
            participation_rates
            + self.impact_engine.config.depletion_weight
            * np.sign(participation_rates)
            * depletion_b
        )

        psi_hess_diag = self.impact_engine.huber.hessian_diagonal(effective_u)
        psi_vals = self.impact_engine.huber.evaluate(effective_u)
        psi_grads = self.impact_engine.huber.gradient(effective_u)

        # Smooth panic multiplier derivatives
        zeta = self.impact_engine.config.sigmoidal_slope
        clipped_exp_arg = np.clip(zeta * participation_rates, -50.0, 50.0)
        sigma_asym = 1.0 / (1.0 + np.exp(clipped_exp_arg))
        d_sigma_asym = -zeta * sigma_asym * (1.0 - sigma_asym)
        d2_sigma_asym = (zeta**2) * sigma_asym * (1.0 - sigma_asym) * (2.0 * sigma_asym - 1.0)

        pi_panic_clamped = float(np.clip(panic_probability, 0.0, 1.0))
        panic_mult = (
            1.0 + self.impact_engine.config.panic_penalty_scale * pi_panic_clamped * sigma_asym
        )
        d_panic_mult = (
            self.impact_engine.config.panic_penalty_scale * pi_panic_clamped * d_sigma_asym
        )
        d2_panic_mult = (
            self.impact_engine.config.panic_penalty_scale * pi_panic_clamped * d2_sigma_asym
        )

        # Crowding multiplier
        if crowding_scores is None:
            crowding_mult = np.ones(n, dtype=np.float64)
        else:
            c_scores = np.asarray(crowding_scores, dtype=np.float64).flatten()
            c_exp = np.clip(self.impact_engine.config.crowding_steepness * c_scores, -30.0, 30.0)
            crowding_mult = 1.0 + (self.impact_engine.config.crowding_max_mult - 1.0) / (
                1.0 + np.exp(-c_exp)
            )

        # Second derivative of transient term: d^2 C_trans / d nu_i^2
        # d^2 (psi * panic_mult) / d nu^2 = psi'' * panic + 2 * psi' * panic' + psi * panic''
        term_d2 = (
            psi_hess_diag * panic_mult + 2.0 * psi_grads * d_panic_mult + psi_vals * d2_panic_mult
        )
        eta = self.impact_engine.config.transient_scale
        d2_c_d_nu2 = eta * sigmas * crowding_mult * term_d2

        # Chain rule: d^2 C / d (Delta a_i)^2 = (W_0 / Bar_Dol_i)^2 * d2_c_d_nu2
        d_nu_d_delta_a = self.impact_engine.config.portfolio_value / safe_bar_vols
        hessian_transient_diag = np.maximum(d2_c_d_nu2 * (d_nu_d_delta_a**2), 0.0)

        # Total Hessian: - (Variance_Hessian + Lambda_cross + diag(Hessian_transient))
        hessian_friction = lambda_cross + np.diag(hessian_transient_diag)
        total_hessian = -(hessian_variance + hessian_friction)

        # Symmetrize to eliminate machine precision asymmetries
        sym_hessian = 0.5 * (total_hessian + total_hessian.T)
        return np.asarray(sym_hessian, dtype=np.float64)


class InstitutionalBenchmarkUniverse:
    """Institutional Benchmark Portfolio Universe Generator with Friction Parity.

    Generates realistic, institutionally constrained systematic benchmark allocations:
        1. Equal Weight (EW): b_i = min(w_max, 1 / N).
        2. Risk Parity (RP): b_i proportional to 1 / sqrt(Sigma_ii), capped at w_max.
        3. Inverse Volatility (IV): b_i proportional to 1 / sigma_i, capped at w_max.
        4. Pure Cash (Cash): b = 0, b_cash = 1.0.

    Guarantees Friction Parity:
        Evaluates each benchmark's net utility U(b, s_j) under the identical transaction
        and market impact friction incurred when rebalancing from initial portfolio a_0:
            U_bench^*(s_j) = max_{b in B} U(b, s_j)

        This prevents artificial regret inflation caused by unrealistic frictionless benchmarks.
    """

    def __init__(
        self,
        config: StackelbergConfig | None = None,
        payoff_engine: StackelbergPayoffEngine | None = None,
    ) -> None:
        """Initialize benchmark universe generator.

        Args:
            config: Stackelberg configuration dataclass.
            payoff_engine: StackelbergPayoffEngine instance.
        """
        self.config = config or StackelbergConfig()
        self.payoff_engine = payoff_engine or StackelbergPayoffEngine(self.config)

    def generate_equal_weight(self, n_assets: int) -> np.ndarray:
        """Generate institutional Equal Weight allocation b_EW.

        Args:
            n_assets: Number of assets N in the investment universe.

        Returns:
            1D array of length N with equal weights clamped to max_weight_per_asset.
        """
        if n_assets < 1:
            raise ValueError(f"{ERR_STACK_BENCHMARK}: n_assets must be >= 1, got {n_assets}")
        raw_w = 1.0 / float(n_assets)
        w = min(raw_w, self.config.max_weight_per_asset)
        return np.full(n_assets, w, dtype=np.float64)

    def generate_risk_parity(self, covariance: np.ndarray) -> np.ndarray:
        """Generate institutional Risk Parity allocation b_RP.

        Weights are inversely proportional to asset asset return standard deviation:
            w_i proportional to 1 / sqrt(Sigma_ii)

        Excess weights above max_weight_per_asset are iteratively redistributed
        among uncapped assets to respect the gross leverage limit.

        Args:
            covariance: 2D asset return covariance matrix Sigma (N x N).

        Returns:
            1D array of length N satisfying 0 <= b_i <= w_max and sum b_i <= 1.0.
        """
        sigma_mat = np.asarray(covariance, dtype=np.float64)
        if sigma_mat.ndim != 2 or sigma_mat.shape[0] != sigma_mat.shape[1]:
            raise ValueError(f"{ERR_STACK_BENCHMARK}: Covariance matrix must be square (N x N)")

        diag_var = np.diag(sigma_mat)
        safe_std = np.sqrt(np.maximum(diag_var, 1e-12))
        raw_weights = 1.0 / safe_std
        return self._cap_and_redistribute(raw_weights)

    def generate_inverse_volatility(self, asset_volatilities: np.ndarray) -> np.ndarray:
        """Generate institutional Inverse Volatility allocation b_IV.

        Weights are inversely proportional to daily realized asset volatilities:
            w_i proportional to 1 / sigma_i

        Args:
            asset_volatilities: 1D array of asset volatilities (length N).

        Returns:
            1D array of length N satisfying 0 <= b_i <= w_max and sum b_i <= 1.0.
        """
        sigmas = np.asarray(asset_volatilities, dtype=np.float64).flatten()
        if len(sigmas) < 1:
            raise ValueError(f"{ERR_STACK_BENCHMARK}: asset_volatilities must not be empty")

        safe_sigmas = np.maximum(sigmas, 1e-12)
        raw_weights = 1.0 / safe_sigmas
        return self._cap_and_redistribute(raw_weights)

    def generate_cash_only(self, n_assets: int) -> np.ndarray:
        """Generate pure risk-off Cash allocation (b = 0).

        Args:
            n_assets: Number of assets N.

        Returns:
            1D array of zeros of length N.
        """
        if n_assets < 1:
            raise ValueError(f"{ERR_STACK_BENCHMARK}: n_assets must be >= 1, got {n_assets}")
        return np.zeros(n_assets, dtype=np.float64)

    def _cap_and_redistribute(self, raw_weights: np.ndarray) -> np.ndarray:
        """Iteratively redistribute weights to enforce single-asset and gross leverage caps.

        Args:
            raw_weights: 1D array of positive raw unnormalized weights.

        Returns:
            Feasible allocation array satisfying 0 <= w_i <= w_max and sum w_i <= L_max.
        """
        n = len(raw_weights)
        w_max = self.config.max_weight_per_asset
        l_max = self.config.gross_leverage_limit

        total_raw = np.sum(raw_weights)
        if total_raw <= 0.0:
            return np.zeros(n, dtype=np.float64)

        weights = (raw_weights / total_raw) * l_max
        capped = np.zeros(n, dtype=bool)

        for _ in range(n):
            excess = 0.0
            over_mask = (~capped) & (weights > w_max)
            if not np.any(over_mask):
                break

            capped[over_mask] = True
            excess = float(np.sum(weights[over_mask] - w_max))
            weights[over_mask] = w_max

            uncapped_count = int(np.sum(~capped))
            if uncapped_count > 0:
                weights[~capped] += excess / float(uncapped_count)
            else:
                break

        # Final safety clamp
        weights = np.clip(weights, 0.0, w_max)
        tot = float(np.sum(weights))
        if tot > l_max:
            weights = (weights / tot) * l_max

        return np.asarray(weights, dtype=np.float64)

    def evaluate_benchmark_universe(
        self,
        initial_allocation: np.ndarray | None,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        daily_dollar_volumes: np.ndarray,
        asset_volatilities: np.ndarray,
        panic_probability: float = 0.0,
        kelly_conviction: np.ndarray | None = None,
        crowding_scores: np.ndarray | None = None,
    ) -> BenchmarkEvaluationResult:
        """Evaluate all institutional benchmarks under regime conditions with friction parity.

        Finds the utility ceiling U_bench^*(s_j) across Equal Weight, Risk Parity,
        Inverse Volatility, and Pure Cash.

        Args:
            initial_allocation: 1D array of current portfolio weights a_0 (length N).
            expected_returns: 1D array of expected returns mu_j (length N).
            covariance: 2D asset return covariance matrix Sigma_j (N x N).
            daily_dollar_volumes: 1D array of daily dollar volumes (length N).
            asset_volatilities: 1D array of asset volatilities (length N).
            panic_probability: Probability of panic regime pi_panic.
            kelly_conviction: Optional Kelly confidence scores.
            crowding_scores: Optional crowding indicators.

        Returns:
            BenchmarkEvaluationResult containing the optimal ceiling benchmark and details.
        """
        n_assets = len(expected_returns)
        benchmarks: dict[str, np.ndarray] = {
            "EqualWeight": self.generate_equal_weight(n_assets),
            "RiskParity": self.generate_risk_parity(covariance),
            "InverseVolatility": self.generate_inverse_volatility(asset_volatilities),
            "CashOnly": self.generate_cash_only(n_assets),
        }

        utilities: dict[str, float] = {}
        best_name = "CashOnly"
        best_u = -float("inf")
        best_w = benchmarks["CashOnly"]

        for name, weights in benchmarks.items():
            u = self.payoff_engine.evaluate_utility(
                allocation=weights,
                initial_allocation=initial_allocation,
                expected_returns=expected_returns,
                covariance=covariance,
                daily_dollar_volumes=daily_dollar_volumes,
                asset_volatilities=asset_volatilities,
                panic_probability=panic_probability,
                kelly_conviction=kelly_conviction,
                crowding_scores=crowding_scores,
            )
            utilities[name] = u
            if u > best_u:
                best_u = u
                best_name = name
                best_w = weights

        return BenchmarkEvaluationResult(
            best_benchmark_name=best_name,
            best_benchmark_weights=best_w,
            best_benchmark_utility=best_u,
            all_benchmark_utilities=utilities,
            all_benchmark_weights=benchmarks,
        )


class StackelbergPayoffTensorConstructor:
    """Payoff Tensor and Non-Negative Regret Matrix Constructor.

    Evaluates a population of candidate actions A = {a_1, ..., a_P} across market regimes
    S = {s_1, ..., s_M}, producing:
        - Payoff Tensor U in R^{P x M}: U_{i, j} = U(a_i, s_j).
        - Benchmark Ceiling Vector U_bench^* in R^M: U_bench^*(s_j) = max_{b in B} U(b, s_j).
        - Non-Negative Regret Matrix R in R^{P x M}_{>= 0}:
            R_{i, j} = max(0.0, U_bench^*(s_j) - U_{i, j})

    Feeds directly into Step 4's closed-form Boltzmann dual and Newton minimax solver.
    """

    def __init__(
        self,
        config: StackelbergConfig | None = None,
        payoff_engine: StackelbergPayoffEngine | None = None,
        benchmark_universe: InstitutionalBenchmarkUniverse | None = None,
    ) -> None:
        """Initialize tensor constructor.

        Args:
            config: Stackelberg configuration dataclass.
            payoff_engine: StackelbergPayoffEngine instance.
            benchmark_universe: InstitutionalBenchmarkUniverse instance.
        """
        self.config = config or StackelbergConfig()
        self.payoff_engine = payoff_engine or StackelbergPayoffEngine(self.config)
        self.benchmark_universe = benchmark_universe or InstitutionalBenchmarkUniverse(
            self.config, self.payoff_engine
        )

    def construct_tensor(
        self,
        candidate_allocations: np.ndarray,
        initial_allocation: np.ndarray | None,
        regime_scenarios: list[dict[str, Any]],
        daily_dollar_volumes: np.ndarray,
        asset_volatilities: np.ndarray,
        kelly_conviction: np.ndarray | None = None,
        crowding_scores: np.ndarray | None = None,
    ) -> PayoffTensorResult:
        """Construct the complete payoff tensor and non-negative regret matrix.

        Args:
            candidate_allocations: 2D array of shape (P, N) containing P candidate allocations.
            initial_allocation: 1D array of current portfolio weights a_0 (length N).
            regime_scenarios: List of M dictionaries, each containing:
                - "name": str
                - "expected_returns": 1D array (length N)
                - "covariance": 2D array (N x N)
                - "panic_probability": float in [0, 1]
            daily_dollar_volumes: 1D array of daily dollar volumes (length N).
            asset_volatilities: 1D array of asset volatilities (length N).
            kelly_conviction: Optional Kelly confidence scores.
            crowding_scores: Optional crowding indicators.

        Returns:
            PayoffTensorResult containing payoff tensor, regret matrix, and metadata.

        Raises:
            ValueError: If candidates or scenarios are empty or dimensionally inconsistent.
        """
        candidates = np.asarray(candidate_allocations, dtype=np.float64)
        if candidates.ndim != 2:
            raise ValueError(
                f"{ERR_STACK_DIM}: candidate_allocations must be 2D array of shape (P, N)"
            )

        p_candidates, n_assets = candidates.shape
        m_regimes = len(regime_scenarios)

        if p_candidates < 1:
            raise ValueError(f"{ERR_STACK_DIM}: Must provide at least 1 candidate allocation")
        if m_regimes < 1:
            raise ValueError(f"{ERR_STACK_DIM}: Must provide at least 1 regime scenario")

        payoff_tensor = np.empty((p_candidates, m_regimes), dtype=np.float64)
        benchmark_ceilings = np.empty(m_regimes, dtype=np.float64)
        regime_names: list[str] = []

        # 1. Evaluate benchmark ceiling for each regime s_j
        for j, scenario in enumerate(regime_scenarios):
            r_name = str(scenario.get("name", f"Regime_{j}"))
            regime_names.append(r_name)

            mu_j = np.asarray(scenario["expected_returns"], dtype=np.float64).flatten()
            sigma_j = np.asarray(scenario["covariance"], dtype=np.float64)
            pi_panic_j = float(scenario.get("panic_probability", 0.0))

            bench_res = self.benchmark_universe.evaluate_benchmark_universe(
                initial_allocation=initial_allocation,
                expected_returns=mu_j,
                covariance=sigma_j,
                daily_dollar_volumes=daily_dollar_volumes,
                asset_volatilities=asset_volatilities,
                panic_probability=pi_panic_j,
                kelly_conviction=kelly_conviction,
                crowding_scores=crowding_scores,
            )
            benchmark_ceilings[j] = bench_res.best_benchmark_utility

            # 2. Evaluate candidate payoffs U(a_i, s_j)
            for i in range(p_candidates):
                a_i = candidates[i]
                u_ij = self.payoff_engine.evaluate_utility(
                    allocation=a_i,
                    initial_allocation=initial_allocation,
                    expected_returns=mu_j,
                    covariance=sigma_j,
                    daily_dollar_volumes=daily_dollar_volumes,
                    asset_volatilities=asset_volatilities,
                    panic_probability=pi_panic_j,
                    kelly_conviction=kelly_conviction,
                    crowding_scores=crowding_scores,
                )
                payoff_tensor[i, j] = u_ij

        # 3. Construct Non-Negative Regret Matrix: R_{i, j} = max(0, U_bench^*(s_j) - U_{i, j})
        raw_regret = benchmark_ceilings[np.newaxis, :] - payoff_tensor
        regret_matrix = np.maximum(raw_regret, 0.0)

        return PayoffTensorResult(
            payoff_tensor=payoff_tensor,
            benchmark_utilities=benchmark_ceilings,
            regret_matrix=regret_matrix,
            regime_names=regime_names,
            candidate_allocations=candidates,
        )
