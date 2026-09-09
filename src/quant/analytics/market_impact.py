"""Multi-Asset Cross-Impact Propagator Engine (Bouchaud-Huberman-Stanzl Architecture).

Purpose:
    Provides an institutional-grade, mathematically rigorous market impact and transaction
    friction engine modeling:
    1. Huberman-Stanzl Arbitrage-Free Cross-Impact: Symmetric positive-definite tensor
       coupling price impact across correlated assets via OAS covariance eigen-modes.
    2. 3/2-Power Generalized Pseudo-Huber Potential: Models the exact universal Square-Root
       Law of price impact (Psi'_{3/2}(u) ~ sqrt(u)) while maintaining C^infinity smoothness
       and strict convexity everywhere.
    3. Smooth Bayesian Panic Asymmetry Gate: Continuous sigmoidal gate driven by Step 1's
       Bayesian panic probability with identically zero gradient at rest (nabla C(0) = 0).
    4. State-Space Bounded Order Book Depletion Buffer: Tracks transient liquidity depletion
       across consecutive trading bars with saturating hyperbolic tangent bounding.
    5. Exact Analytical Gradient Vector: Closed-form nabla_{Delta a} C enabling sub-10 microsecond
       quadratic convergence in downstream Newton-Raphson solvers.

Dependencies:
    - numpy: Vectorized linear algebra and array transforms.
    - scipy.linalg: Spectral matrix operations and linear solvers.

Structural Relationship:
    - Ingests: Step 1 RegimeEstimationResult (covariance Sigma, panic prob pi_panic, vol sigma_t)
      and Phase 2 Step 5 continuous Kelly meta-label conviction vector z_t.
    - Emits: Total friction cost C(Delta a), analytical gradient nabla C, and cross-impact matrix.
    - Consumed by: payoff_matrix.py (Step 3) and minimax_regret.py (Step 4).

Invariants:
    1. No Price Manipulation Arbitrage: Cross-impact matrix Lambda_cross is strictly symmetric
       and positive definite (lambda_min >= fee_floor > 0).
    2. Zero Gradient at Rest: Both friction and its gradient are identically zero at Delta a = 0.
    3. Strict Global Convexity: Second derivative Psi''_{3/2}(u) > 0 for all u in R.
    4. Bounded Depletion Memory: B_t is strictly contained within [0, depletion_cap].
    5. Dimensional Consistency: Conversion from portfolio weights to dollar participation
       is dimensionally homogeneous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.linalg

logger = logging.getLogger(__name__)

# Error code constants (Rule 2: Deterministic Diagnostics)
ERR_IMPACT_DIM = (
    "ERR-GAME-IMPACT-DIM: Dimensionality mismatch between weight vector and market parameters"
)
ERR_IMPACT_PARAM = (
    "ERR-GAME-IMPACT-PARAM: Invalid configuration parameter passed to MarketImpactConfig"
)
ERR_IMPACT_SINGULAR = "ERR-GAME-IMPACT-SINGULAR: Non-positive definite covariance or volume preventing cross-impact construction"


@dataclass(frozen=True)
class MarketImpactConfig:
    """Configuration container for multi-asset cross-impact and friction parameters.

    Attributes:
        fee_floor: Minimum exchange fee and half-spread floor lambda_fee (default 5 bps).
        huber_delta: Smoothing threshold delta eliminating gradient singularities at zero.
        cross_impact_scale: Scaling intensity theta for permanent cross-impact.
        transient_scale: Scaling intensity eta for 3/2-power transient order book impact.
        panic_penalty_scale: Multiplier kappa_0 for selling into panic regimes.
        sigmoidal_slope: Steepness parameter zeta for the smooth panic asymmetry gate.
        resilience_rate: Order book replenishment rate rho in (0, 1) for depletion decay.
        depletion_cap: Maximum order book depletion ceiling B_max as a fraction of bar depth.
        depletion_weight: Coupling weight omega of historical depletion into transient impact.
        portfolio_value: Total portfolio capital W_0 in nominal dollars (AUM).
        bar_time_fraction: Bar duration as fraction of a full trading day tau_bar (e.g. 1/390).
        crowding_max_mult: Maximum ceiling Omega_max for consensus crowding multiplier.
        crowding_steepness: Steepness parameter gamma_crowd for crowding saturation.
    """

    fee_floor: float = 0.0005
    huber_delta: float = 1e-4
    cross_impact_scale: float = 0.10
    transient_scale: float = 0.50
    panic_penalty_scale: float = 1.50
    sigmoidal_slope: float = 10.0
    resilience_rate: float = 0.50
    depletion_cap: float = 0.20
    depletion_weight: float = 0.30
    portfolio_value: float = 1_000_000.0
    bar_time_fraction: float = 1.0 / 390.0
    crowding_max_mult: float = 2.50
    crowding_steepness: float = 2.0

    def __post_init__(self) -> None:
        """Validate invariant boundaries on configuration parameters."""
        if self.fee_floor <= 0.0:
            raise ValueError(f"{ERR_IMPACT_PARAM}: fee_floor must be > 0, got {self.fee_floor}")
        if self.huber_delta <= 0.0:
            raise ValueError(f"{ERR_IMPACT_PARAM}: huber_delta must be > 0, got {self.huber_delta}")
        if self.cross_impact_scale < 0.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: cross_impact_scale must be >= 0, got {self.cross_impact_scale}"
            )
        if self.transient_scale < 0.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: transient_scale must be >= 0, got {self.transient_scale}"
            )
        if self.panic_penalty_scale < 0.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: panic_penalty_scale must be >= 0, got {self.panic_penalty_scale}"
            )
        if self.sigmoidal_slope <= 0.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: sigmoidal_slope must be > 0, got {self.sigmoidal_slope}"
            )
        if not (0.0 < self.resilience_rate < 1.0):
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: resilience_rate must be in (0, 1), got {self.resilience_rate}"
            )
        if self.depletion_cap <= 0.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: depletion_cap must be > 0, got {self.depletion_cap}"
            )
        if self.depletion_weight < 0.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: depletion_weight must be >= 0, got {self.depletion_weight}"
            )
        if self.portfolio_value <= 0.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: portfolio_value must be > 0, got {self.portfolio_value}"
            )
        if not (0.0 < self.bar_time_fraction <= 1.0):
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: bar_time_fraction must be in (0, 1], got {self.bar_time_fraction}"
            )
        if self.crowding_max_mult < 1.0:
            raise ValueError(
                f"{ERR_IMPACT_PARAM}: crowding_max_mult must be >= 1.0, got {self.crowding_max_mult}"
            )


@dataclass
class DepletionState:
    """State-space memory tracking transient order book exhaustion across bars.

    Attributes:
        depletion_vector: Vector of current order book depletion B_t for each asset.
        last_update_bar: Bar index of last recorded trade update.
    """

    depletion_vector: np.ndarray
    last_update_bar: int = 0


@dataclass(frozen=True)
class MarketImpactResult:
    """Immutable result payload containing evaluated friction metrics and analytical derivatives.

    Attributes:
        total_cost: Total execution friction penalty C(Delta a) in portfolio return space.
        permanent_cost: Quadratic permanent cross-impact component.
        transient_cost: 3/2-power transient order book consumption component.
        gradient: Exact analytical gradient vector nabla_{Delta a} C (shape: n_assets).
        cross_impact_matrix: Symmetric positive-definite Huberman-Stanzl matrix Lambda_cross.
        participation_rates: Dimensionless bar participation rates nu_i = Delta Dol_i / Bar_Dol_i.
        depletion_state: Updated order book depletion vector B_t.
    """

    total_cost: float
    permanent_cost: float
    transient_cost: float
    gradient: np.ndarray
    cross_impact_matrix: np.ndarray
    participation_rates: np.ndarray
    depletion_state: np.ndarray


class GeneralizedPseudoHuber:
    """3/2-Power Generalized Pseudo-Huber Potential for the Square-Root Law of Market Impact.

    Computes:
        Psi_{3/2}(u) = (u^2 + delta^2)^{3/4} - delta^{1.5}

    Properties:
        - At u = 0: Psi(0) = 0, Psi'(0) = 0.
        - For large |u|: Psi'(u) ~ 1.5 * sign(u) * sqrt(|u|) (Exact Universal Square-Root Law).
        - Second derivative Psi''(u) > 0 everywhere (Strict Global Convexity).
    """

    def __init__(self, delta: float = 1e-4) -> None:
        """Initialize potential with smoothing parameter delta."""
        self.delta = delta
        self._delta_sq = delta**2
        self._delta_15 = delta**1.5

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Evaluate the 3/2-power friction potential.

        Args:
            u: Array of dimensionless order book participation rates.

        Returns:
            Array of potential values with identical shape to u.
        """
        inner = u**2 + self._delta_sq
        return np.asarray(inner**0.75 - self._delta_15, dtype=np.float64)

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Evaluate exact analytical first derivative Psi'_{3/2}(u).

        Psi'_{3/2}(u) = 1.5 * u / (u^2 + delta^2)^{1/4}
        """
        inner = u**2 + self._delta_sq
        denom = inner**0.25
        return np.asarray(1.5 * u / denom, dtype=np.float64)

    def hessian_diagonal(self, u: np.ndarray) -> np.ndarray:
        """Evaluate exact analytical second derivative Psi''_{3/2}(u).

        Psi''_{3/2}(u) = 1.5 * (0.5 * u^2 + delta^2) / (u^2 + delta^2)^{5/4}
        """
        inner = u**2 + self._delta_sq
        num = 0.5 * (u**2) + self._delta_sq
        denom = inner**1.25
        return np.asarray(1.5 * num / denom, dtype=np.float64)


class HubermanStanzlCrossImpact:
    """Huberman-Stanzl Arbitrage-Free Cross-Impact Tensor Constructor.

    Constructs a strictly symmetric, positive-definite cross-impact matrix:
        Lambda_cross = lambda_fee * I + theta * Z^{1/2} * (D_vol^{-1/2} * Sigma^{1/2} * D_vol^{-1/2}) * Z^{1/2}

    Guarantees:
        1. Lambda_cross = Lambda_cross^T (Symmetric Sandwich).
        2. lambda_min(Lambda_cross) >= lambda_fee > 0 (Strict Positive Definiteness).
        3. Zero Price Manipulation Arbitrage.
    """

    def __init__(self, fee_floor: float = 0.0005, cross_impact_scale: float = 0.10) -> None:
        """Initialize Huberman-Stanzl constructor parameters."""
        self.fee_floor = fee_floor
        self.cross_impact_scale = cross_impact_scale

    def build_matrix(
        self,
        covariance: np.ndarray,
        daily_dollar_volumes: np.ndarray,
        kelly_conviction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct the symmetric positive-definite cross-impact tensor.

        Args:
            covariance: Positive-definite asset covariance matrix Sigma of shape (N, N).
            daily_dollar_volumes: 1D array of daily dollar volumes ADV_Dol of length N.
            kelly_conviction: Optional 1D array of Kelly confidence scores z_t in [0, 1].

        Returns:
            Symmetric positive-definite cross-impact matrix of shape (N, N).
        """
        cov = np.asarray(covariance, dtype=np.float64)
        vols = np.asarray(daily_dollar_volumes, dtype=np.float64).flatten()
        n_assets = len(vols)

        if cov.shape != (n_assets, n_assets):
            raise ValueError(
                f"{ERR_IMPACT_DIM}: Covariance shape {cov.shape} does not match {n_assets} assets"
            )
        if np.any(vols <= 0.0):
            raise ValueError(f"{ERR_IMPACT_PARAM}: Daily dollar volumes must be strictly positive")

        if kelly_conviction is None:
            z_vec = np.ones(n_assets, dtype=np.float64)
        else:
            z_vec = np.asarray(kelly_conviction, dtype=np.float64).flatten()
            if len(z_vec) != n_assets:
                raise ValueError(
                    f"{ERR_IMPACT_DIM}: kelly_conviction length {len(z_vec)} does not match {n_assets}"
                )
            z_vec = np.clip(z_vec, 0.0, 1.0)

        # Compute spectral matrix square root Sigma^{1/2} = V * diag(sqrt(lambda)) * V^T
        eigenvalues, eigenvectors = scipy.linalg.eigh(cov)
        clamped_eigs = np.maximum(eigenvalues, 1e-8)
        sqrt_cov = np.dot(eigenvectors, np.dot(np.diag(np.sqrt(clamped_eigs)), eigenvectors.T))
        sqrt_cov = (sqrt_cov + sqrt_cov.T) / 2.0

        # Scale by inverse square root of volume D_vol^{-1/2}
        inv_sqrt_vols = 1.0 / np.sqrt(vols)
        d_inv = np.diag(inv_sqrt_vols)
        normalized_kernel = np.dot(d_inv, np.dot(sqrt_cov, d_inv))

        # Apply symmetric sandwich scaling with Kelly conviction Z^{1/2}
        z_diag_sqrt = np.diag(np.sqrt(z_vec + 1e-6))
        scaled_cross = np.dot(z_diag_sqrt, np.dot(normalized_kernel, z_diag_sqrt))

        # Combine with fee floor identity matrix
        lambda_matrix = self.fee_floor * np.eye(n_assets) + self.cross_impact_scale * scaled_cross
        symmetric_lambda = (lambda_matrix + lambda_matrix.T) / 2.0

        return np.asarray(symmetric_lambda, dtype=np.float64)


class MultiAssetMarketImpactEngine:
    """Master Multi-Asset Cross-Impact Propagator Engine.

    Synthesizes:
    - Huberman-Stanzl permanent cross-impact.
    - 3/2-power generalized pseudo-Huber transient friction.
    - Zero-gradient smooth Bayesian panic asymmetry gate.
    - Bounded order book depletion state-space memory.
    """

    def __init__(self, config: MarketImpactConfig | None = None) -> None:
        """Initialize market impact engine with configuration."""
        self.config = config or MarketImpactConfig()
        self.huber = GeneralizedPseudoHuber(delta=self.config.huber_delta)
        self.cross_constructor = HubermanStanzlCrossImpact(
            fee_floor=self.config.fee_floor,
            cross_impact_scale=self.config.cross_impact_scale,
        )
        self._depletion_state: DepletionState | None = None

    def reset(self) -> None:
        """Reset internal order book depletion state."""
        self._depletion_state = None

    def _get_depletion_vector(self, n_assets: int) -> np.ndarray:
        """Retrieve current order book depletion vector or initialize zeros."""
        if self._depletion_state is None:
            return np.zeros(n_assets, dtype=np.float64)
        if len(self._depletion_state.depletion_vector) != n_assets:
            return np.zeros(n_assets, dtype=np.float64)
        return self._depletion_state.depletion_vector.copy()

    def update_depletion(
        self,
        participation_rates: np.ndarray,
    ) -> np.ndarray:
        """Update state-space order book depletion with hyperbolic tangent saturation.

        B_t = rho * B_{t-1} + (1 - rho) * B_max * tanh(|nu_t| / B_max)
        """
        n_assets = len(participation_rates)
        current_b = self._get_depletion_vector(n_assets)
        nu_abs = np.abs(participation_rates)

        b_max = self.config.depletion_cap
        rho = self.config.resilience_rate

        injection = (1.0 - rho) * b_max * np.tanh(nu_abs / max(b_max, 1e-8))
        new_b = rho * current_b + injection
        clamped_b = np.clip(new_b, 0.0, b_max)

        if self._depletion_state is None:
            self._depletion_state = DepletionState(depletion_vector=clamped_b)
        else:
            self._depletion_state.depletion_vector = clamped_b
            self._depletion_state.last_update_bar += 1

        return clamped_b.copy()

    def evaluate_impact(
        self,
        delta_allocations: np.ndarray,
        covariance: np.ndarray,
        daily_dollar_volumes: np.ndarray,
        asset_volatilities: np.ndarray,
        panic_probability: float = 0.0,
        kelly_conviction: np.ndarray | None = None,
        crowding_scores: np.ndarray | None = None,
        update_state: bool = False,
        cross_impact_matrix: np.ndarray | None = None,
    ) -> MarketImpactResult:
        """Evaluate total execution friction and exact analytical gradient.

        Args:
            delta_allocations: 1D array of intended portfolio weight changes Delta a (length N).
            covariance: Positive-definite asset covariance matrix Sigma of shape (N, N).
            daily_dollar_volumes: 1D array of daily dollar volumes ADV_Dol of length N.
            asset_volatilities: 1D array of asset realized volatilities sigma_t of length N.
            panic_probability: Current Bayesian panic probability pi_panic in [0, 1] from Step 1.
            kelly_conviction: Optional 1D array of Kelly confidence scores z_t in [0, 1].
            crowding_scores: Optional 1D array of trade crowding indicators.
            update_state: If True, updates the internal order book depletion state B_t.
            cross_impact_matrix: Optional precomputed symmetric Huberman-Stanzl matrix Lambda_cross.

        Returns:
            MarketImpactResult containing total cost, permanent/transient split, and gradient.
        """
        delta_a = np.asarray(delta_allocations, dtype=np.float64).flatten()
        n_assets = len(delta_a)

        vols = np.asarray(daily_dollar_volumes, dtype=np.float64).flatten()
        sigmas = np.asarray(asset_volatilities, dtype=np.float64).flatten()

        if len(vols) != n_assets or len(sigmas) != n_assets:
            raise ValueError(f"{ERR_IMPACT_DIM}: Array lengths do not match {n_assets} assets")

        # 1. Compute dimensional participation rates nu_i
        bar_dollar_volumes = vols * self.config.bar_time_fraction
        safe_bar_vols = np.maximum(bar_dollar_volumes, 1.0)
        participation_rates = (self.config.portfolio_value * delta_a) / safe_bar_vols

        # 2. Permanent Cross-Impact Layer: 0.5 * Delta a^T * Lambda_cross * Delta a
        if cross_impact_matrix is not None:
            lambda_cross = cross_impact_matrix
        else:
            lambda_cross = self.cross_constructor.build_matrix(
                covariance=covariance,
                daily_dollar_volumes=vols,
                kelly_conviction=kelly_conviction,
            )
        permanent_cost = 0.5 * float(np.dot(delta_a, np.dot(lambda_cross, delta_a)))
        grad_permanent = np.dot(lambda_cross, delta_a)

        # 3. Retrieve order book depletion state B_{t-1}
        depletion_b = self._get_depletion_vector(n_assets)
        effective_u = (
            participation_rates
            + self.config.depletion_weight * np.sign(participation_rates) * depletion_b
        )

        # 4. 3/2-Power Transient Potential Psi_{3/2}(u)
        psi_vals = self.huber.evaluate(effective_u)
        psi_grads = self.huber.gradient(effective_u)

        # 5. Smooth Sigmoidal Panic Asymmetry Gate: sigma_asym(nu) = 1 / (1 + exp(zeta * nu))
        zeta = self.config.sigmoidal_slope
        # Clipped exponent to prevent numerical overflow in exp
        clipped_exp_arg = np.clip(zeta * participation_rates, -50.0, 50.0)
        sigma_asym = 1.0 / (1.0 + np.exp(clipped_exp_arg))
        d_sigma_asym = -zeta * sigma_asym * (1.0 - sigma_asym)

        # Panic multiplier: 1.0 + kappa_0 * pi_panic * sigma_asym(nu)
        pi_panic_clamped = float(np.clip(panic_probability, 0.0, 1.0))
        panic_mult = 1.0 + self.config.panic_penalty_scale * pi_panic_clamped * sigma_asym

        # 6. Saturating Crowding Multiplier
        if crowding_scores is None:
            crowding_mult = np.ones(n_assets, dtype=np.float64)
        else:
            c_scores = np.asarray(crowding_scores, dtype=np.float64).flatten()
            c_exp = np.clip(self.config.crowding_steepness * c_scores, -30.0, 30.0)
            crowding_mult = 1.0 + (self.config.crowding_max_mult - 1.0) / (1.0 + np.exp(-c_exp))

        # 7. Transient Cost Summation
        # Base transient cost: eta * sigma_t * Psi_{3/2}(u) * Multipliers
        eta = self.config.transient_scale
        base_transient_terms = eta * sigmas * psi_vals * panic_mult * crowding_mult
        transient_cost = float(np.sum(base_transient_terms))

        # 8. Exact Analytical Gradient Computation
        # Chain rule: dC_transient / d(Delta a_i) = dC / d(nu_i) * d(nu_i) / d(Delta a_i)
        # d(nu_i) / d(Delta a_i) = W_0 / Bar_Dol_i
        d_nu_d_delta_a = self.config.portfolio_value / safe_bar_vols

        # dC / d(nu_i) = eta * sigma_i * crowding_i * [ Psi' * panic_mult + Psi * (kappa_0 * pi_panic * d_sigma_asym) ]
        term1 = psi_grads * panic_mult
        term2 = psi_vals * (self.config.panic_penalty_scale * pi_panic_clamped) * d_sigma_asym
        d_c_d_nu = eta * sigmas * crowding_mult * (term1 + term2)

        grad_transient = d_c_d_nu * d_nu_d_delta_a
        total_gradient = grad_permanent + grad_transient

        # 9. Total Execution Friction Cost
        total_cost = permanent_cost + transient_cost

        # 10. Update state-space depletion if requested
        new_depletion = self.update_depletion(participation_rates) if update_state else depletion_b

        return MarketImpactResult(
            total_cost=total_cost,
            permanent_cost=permanent_cost,
            transient_cost=transient_cost,
            gradient=total_gradient,
            cross_impact_matrix=lambda_cross,
            participation_rates=participation_rates,
            depletion_state=new_depletion,
        )
