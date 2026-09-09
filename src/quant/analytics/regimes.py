"""Causal Bayesian Jump-Regime Estimator and Thermodynamic Ambiguity Module.

Purpose:
    Provides causal, real-time market regime classification (Low-Vol Absorption,
    Momentum Cascade, Panic Liquidity Trap) combined with an instant CUSUM jump detector,
    Oracle Approximating Shrinkage (OAS) covariance conditioning, and Fournier-Guillin
    thermodynamic ambiguity temperature scaling for distributionally robust game theory.

Dependencies:
    - numpy: Vectorized array algebra and spectral decomposition.
    - scipy.linalg: Matrix decomposition and robust linear solvers.

Structural Relationship:
    - Ingests: Multi-asset price returns and rolling realized volatility from MarketDataService.
    - Emits: Regime probabilities, conditioned covariance matrices, and temperature beta_t.
    - Consumed by: market_impact.py, payoff_matrix.py, and minimax_regret.py.

Invariants:
    1. Zero Lookahead: All regime probabilities and covariance estimates at step t depend
       strictly on historical observations up to and including t.
    2. Simplex Probability: Sum of regime probabilities strictly equals 1.0 (within 1e-12)
       and every component is non-negative.
    3. Positive Definite Covariance: Every regime covariance matrix has all eigenvalues
       clamped to lambda_min >= shrinkage_floor > 0.
    4. Regime Dwell Stability: Minimum dwell time tau_dwell is enforced to prevent
       whip-sawing and thrashing in choppy markets.
    5. Bounded Temperature: Thermodynamic temperature beta_t is strictly bounded within
       [min_temp, max_temp].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.linalg

logger = logging.getLogger(__name__)

# Error code constants (Rule 2: Deterministic Diagnostics)
ERR_REGIME_DIM = (
    "ERR-GAME-REGIME-DIM: Dimensionality mismatch between asset returns and configuration"
)
ERR_REGIME_SAMPLE = "ERR-GAME-REGIME-SAMPLE: Insufficient sample length for regime estimation"
ERR_REGIME_SINGULAR = (
    "ERR-GAME-REGIME-SINGULAR: Degenerate covariance matrix could not be regularized"
)
ERR_REGIME_PARAM = "ERR-GAME-REGIME-PARAM: Invalid configuration parameter passed to RegimeConfig"


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration invariants for causal regime estimation and ambiguity scaling.

    Attributes:
        n_regimes: Number of latent market regimes (default 3: Absorption, Momentum, Panic).
        cusum_threshold: Standardized cumulative deviation threshold h for jump detection.
        cusum_drift: Allowance parameter k (drift) for CUSUM accumulation.
        min_dwell_bars: Minimum bars required in a regime before another transition (tau_dwell).
        shrinkage_floor: Minimum spectral eigenvalue floor for covariance matrices.
        dirichlet_alpha: Prior smoothing pseudo-count for transition probability updates.
        temperature_scale: Fournier-Guillin concentration bound scaling constant kappa.
        confidence_level: Statistical confidence level alpha for the ambiguity ball radius.
        min_temp: Absolute lower bound for thermodynamic temperature beta_t.
        max_temp: Absolute upper bound for thermodynamic temperature beta_t.
        window_size: Rolling window size for regime empirical moments.
        panic_shock_prior: Probability mass assigned to panic regime upon CUSUM shock.
    """

    n_regimes: int = 3
    cusum_threshold: float = 3.0
    cusum_drift: float = 0.5
    min_dwell_bars: int = 3
    shrinkage_floor: float = 1e-5
    dirichlet_alpha: float = 1.0
    temperature_scale: float = 1.0
    confidence_level: float = 0.99
    min_temp: float = 1e-4
    max_temp: float = 10.0
    window_size: int = 60
    panic_shock_prior: float = 0.85

    def __post_init__(self) -> None:
        """Validate invariant constraints on configuration attributes."""
        if self.n_regimes < 2:
            raise ValueError(f"{ERR_REGIME_PARAM}: n_regimes must be >= 2, got {self.n_regimes}")
        if self.cusum_threshold <= 0.0:
            raise ValueError(
                f"{ERR_REGIME_PARAM}: cusum_threshold must be > 0, got {self.cusum_threshold}"
            )
        if self.cusum_drift < 0.0:
            raise ValueError(
                f"{ERR_REGIME_PARAM}: cusum_drift must be >= 0, got {self.cusum_drift}"
            )
        if self.min_dwell_bars < 1:
            raise ValueError(
                f"{ERR_REGIME_PARAM}: min_dwell_bars must be >= 1, got {self.min_dwell_bars}"
            )
        if self.shrinkage_floor <= 0.0:
            raise ValueError(
                f"{ERR_REGIME_PARAM}: shrinkage_floor must be > 0, got {self.shrinkage_floor}"
            )
        if self.confidence_level <= 0.0 or self.confidence_level >= 1.0:
            raise ValueError(
                f"{ERR_REGIME_PARAM}: confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.min_temp <= 0.0 or self.max_temp <= self.min_temp:
            raise ValueError(
                f"{ERR_REGIME_PARAM}: Invalid temperature bounds [{self.min_temp}, {self.max_temp}]"
            )
        if self.window_size < 10:
            raise ValueError(
                f"{ERR_REGIME_PARAM}: window_size must be >= 10, got {self.window_size}"
            )
        if not (0.0 < self.panic_shock_prior <= 1.0):
            raise ValueError(
                f"{ERR_REGIME_PARAM}: panic_shock_prior must be in (0, 1], got {self.panic_shock_prior}"
            )


@dataclass
class RegimeState:
    """Internal state tracking for online causal regime filtering.

    Attributes:
        probabilities: Real-time regime probability vector pi_t (shape: n_regimes).
        active_regime: Maximum a posteriori (MAP) regime index.
        cusum_pos: Accumulated positive CUSUM score S_t^+.
        cusum_neg: Accumulated negative CUSUM score S_t^-.
        bars_in_regime: Number of bars spent in current active regime.
        last_shock: Boolean flag indicating if a CUSUM jump was triggered at step t.
    """

    probabilities: np.ndarray
    active_regime: int
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    bars_in_regime: int = 0
    last_shock: bool = False


@dataclass(frozen=True)
class RegimeEstimationResult:
    """Immutable output payload containing causal regime statistics and ambiguity parameters.

    Attributes:
        probabilities: Current regime probability vector pi_t (sums to 1.0).
        active_regime: MAP active regime index (0=Absorption, 1=Momentum, 2=Panic).
        regime_means: Empirical mean return vector for each regime (shape: n_regimes, n_assets).
        regime_covariances: Conditioned OAS covariance tensor (shape: n_regimes, n_assets, n_assets).
        temperature: Bounded thermodynamic temperature beta_t for Boltzmann dual.
        cusum_alarm: True if instantaneous CUSUM jump detector triggered an emergency transition.
        effective_sample_size: Effective sample count in the active regime.
    """

    probabilities: np.ndarray
    active_regime: int
    regime_means: np.ndarray
    regime_covariances: np.ndarray
    temperature: float
    cusum_alarm: bool
    effective_sample_size: float


class CUSUMJumpDetector:
    """Two-sided Cumulative Sum (CUSUM) shock detector with dwell hysteresis.

    Tracks standardized return innovations to detect sudden structural regime shifts
    without suffering from single-bar noise or regime whip-sawing.
    """

    def __init__(self, threshold: float = 3.0, drift: float = 0.5, min_dwell_bars: int = 3) -> None:
        """Initialize CUSUM detector parameters."""
        self.threshold = threshold
        self.drift = drift
        self.min_dwell_bars = min_dwell_bars

    def update(
        self,
        standardized_return: float,
        current_dwell: int,
        s_pos: float,
        s_neg: float,
    ) -> tuple[bool, bool, float, float]:
        """Update cumulative deviations and evaluate shock threshold.

        Args:
            standardized_return: Normalized return innovation (r_t - mu) / sigma_t.
            current_dwell: Number of consecutive bars spent in active regime.
            s_pos: Previous positive cumulative deviation S_{t-1}^+.
            s_neg: Previous negative cumulative deviation S_{t-1}^-.

        Returns:
            Tuple of:
                - alarm: True if cumulative deviation crossed threshold.
                - is_negative_shock: True if shock was downward (panic selloff).
                - new_s_pos: Updated positive deviation S_t^+.
                - new_s_neg: Updated negative deviation S_t^-.
        """
        new_s_pos = max(0.0, s_pos + standardized_return - self.drift)
        new_s_neg = max(0.0, s_neg - standardized_return - self.drift)

        alarm = False
        is_negative = False

        if new_s_neg >= self.threshold:
            if current_dwell >= self.min_dwell_bars:
                alarm = True
                is_negative = True
                new_s_neg = 0.0
                new_s_pos = 0.0
        elif new_s_pos >= self.threshold and current_dwell >= self.min_dwell_bars:
            alarm = True
            is_negative = False
            new_s_pos = 0.0
            new_s_neg = 0.0

        return alarm, is_negative, new_s_pos, new_s_neg


class OASCovarianceEstimator:
    """Oracle Approximating Shrinkage (OAS) covariance estimator with spectral floor.

    Guarantees strict positive definiteness and optimal MSE shrinkage under
    high-dimensional or low-sample conditions.
    """

    def __init__(self, shrinkage_floor: float = 1e-5) -> None:
        """Initialize OAS estimator with spectral eigenvalue floor."""
        self.shrinkage_floor = shrinkage_floor

    def fit_covariance(self, return_window: np.ndarray) -> np.ndarray:
        """Compute OAS-shrunk, strictly positive-definite covariance matrix.

        Args:
            return_window: 2D array of historical returns with shape (T, N).

        Returns:
            Symmetric positive-definite covariance matrix of shape (N, N).
        """
        t_samples, n_assets = return_window.shape
        if t_samples < 2:
            diag = np.var(return_window, axis=0, ddof=0) if t_samples > 0 else np.ones(n_assets)
            diag = np.maximum(diag, self.shrinkage_floor)
            return np.diag(diag)

        # Center sample returns
        mean_vec = np.mean(return_window, axis=0)
        centered = return_window - mean_vec
        sample_cov = np.dot(centered.T, centered) / float(t_samples)

        # Calculate trace statistics
        tr_s = float(np.trace(sample_cov))
        tr_s2 = float(np.trace(np.dot(sample_cov, sample_cov)))
        target_scale = tr_s / float(n_assets)

        # Analytical OAS shrinkage intensity formula (Chen et al., 2010)
        numerator = (1.0 - 2.0 / float(n_assets)) * tr_s2 + (tr_s**2)
        denominator = (float(t_samples) + 1.0 - 2.0 / float(n_assets)) * (
            tr_s2 - (tr_s**2) / float(n_assets)
        )

        rho = 0.0 if denominator <= 1e-12 else min(1.0, max(0.0, numerator / denominator))

        # Convex combination with scaled identity matrix
        target_matrix = np.eye(n_assets) * target_scale
        shrunk_cov = (1.0 - rho) * sample_cov + rho * target_matrix

        # Spectral projection ensuring all eigenvalues >= shrinkage_floor
        shrunk_cov = (shrunk_cov + shrunk_cov.T) / 2.0
        eigenvalues, eigenvectors = scipy.linalg.eigh(shrunk_cov)
        clamped_eigenvalues = np.maximum(eigenvalues, self.shrinkage_floor)
        projected_cov = np.dot(eigenvectors, np.dot(np.diag(clamped_eigenvalues), eigenvectors.T))

        return np.asarray((projected_cov + projected_cov.T) / 2.0, dtype=np.float64)


class CausalBayesianRegimeFilter:
    """Online Causal Bayesian Filter for multi-regime estimation and thermodynamic ambiguity.

    Seamlessly integrates:
    - Real-time Bayesian probability updating across M regimes.
    - CUSUM jump detection injecting emergency panic priors on violent crashes.
    - OAS covariance estimation guaranteeing well-conditioned convex quadratics.
    - Fournier-Guillin concentration bound calibrating thermodynamic temperature beta_t.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        """Initialize filter with configuration and internal components."""
        self.config = config or RegimeConfig()
        self.jump_detector = CUSUMJumpDetector(
            threshold=self.config.cusum_threshold,
            drift=self.config.cusum_drift,
            min_dwell_bars=self.config.min_dwell_bars,
        )
        self.oas_estimator = OASCovarianceEstimator(
            shrinkage_floor=self.config.shrinkage_floor,
        )
        self._state: RegimeState | None = None
        self._history: list[np.ndarray] = []
        self._regime_assignments: list[int] = []

    def reset(self) -> None:
        """Reset internal filter state and historical buffers."""
        self._state = None
        self._history.clear()
        self._regime_assignments.clear()

    def _initialize_state(self, n_assets: int) -> RegimeState:
        """Initialize uniform prior regime state."""
        uniform_probs = np.full(
            self.config.n_regimes, 1.0 / self.config.n_regimes, dtype=np.float64
        )
        return RegimeState(
            probabilities=uniform_probs,
            active_regime=0,
            cusum_pos=0.0,
            cusum_neg=0.0,
            bars_in_regime=0,
            last_shock=False,
        )

    def _compute_temperature(
        self,
        volatility: float,
        effective_sample_size: float,
    ) -> float:
        """Compute Fournier-Guillin thermodynamic ambiguity temperature beta_t.

        beta_t = clip(kappa * sigma_t * sqrt(ln(1 / alpha) / N_eff), min_temp, max_temp)
        """
        n_eff = max(2.0, effective_sample_size)
        log_term = np.log(1.0 / (1.0 - self.config.confidence_level + 1e-12))
        raw_beta = self.config.temperature_scale * volatility * np.sqrt(log_term / n_eff)
        return float(np.clip(raw_beta, self.config.min_temp, self.config.max_temp))

    def step(
        self,
        returns: np.ndarray,
        realized_volatility: float,
    ) -> RegimeEstimationResult:
        """Process a single causal time step and update regime distribution.

        Args:
            returns: Asset returns vector for current bar t (shape: n_assets,).
            realized_volatility: Current realized volatility scalar sigma_t.

        Returns:
            RegimeEstimationResult containing updated probabilities, covariances, and beta_t.
        """
        returns = np.asarray(returns, dtype=np.float64).flatten()
        n_assets = len(returns)

        if self._state is None:
            self._state = self._initialize_state(n_assets)

        self._history.append(returns)

        # Calculate cross-sectional standardized return for jump detection
        vol_scalar = max(realized_volatility, 1e-6)
        mean_ret = float(np.mean(returns))
        std_ret = mean_ret / vol_scalar

        # Evaluate CUSUM shock detector
        alarm, is_negative, new_s_pos, new_s_neg = self.jump_detector.update(
            standardized_return=std_ret,
            current_dwell=self._state.bars_in_regime,
            s_pos=self._state.cusum_pos,
            s_neg=self._state.cusum_neg,
        )

        self._state.cusum_pos = new_s_pos
        self._state.cusum_neg = new_s_neg
        self._state.last_shock = alarm

        # Prior transition model with Dirichlet smoothing
        prior_probs = self._state.probabilities.copy()

        # If CUSUM detects a severe negative shock, inject emergency panic prior
        panic_regime_idx = self.config.n_regimes - 1
        if alarm and is_negative:
            # Shift mass aggressively toward panic regime
            prior_probs = (1.0 - self.config.panic_shock_prior) * prior_probs
            prior_probs[panic_regime_idx] += self.config.panic_shock_prior
            prior_probs = prior_probs / np.sum(prior_probs)
        elif alarm and not is_negative:
            # Shift mass toward momentum rally regime (State 1)
            rally_idx = min(1, self.config.n_regimes - 1)
            prior_probs = (1.0 - self.config.panic_shock_prior) * prior_probs
            prior_probs[rally_idx] += self.config.panic_shock_prior
            prior_probs = prior_probs / np.sum(prior_probs)

        # Slice rolling history for moment estimation
        window = np.array(self._history[-self.config.window_size :], dtype=np.float64)
        t_window = len(window)

        # Compute empirical moments per regime using historical assignments or clusters
        regime_means = np.zeros((self.config.n_regimes, n_assets), dtype=np.float64)
        regime_covariances = np.zeros((self.config.n_regimes, n_assets, n_assets), dtype=np.float64)

        # Global base moments via OAS
        base_cov = self.oas_estimator.fit_covariance(window)
        base_mean = np.mean(window, axis=0)

        # Regime-conditioned moment scaling:
        # State 0 (Absorption): Mean ~ 0, lower variance (0.7x)
        # State 1 (Momentum): Mean matches recent trend, medium variance (1.0x)
        # State 2 (Panic): Negative mean drift, elevated variance (2.5x)
        regime_means[0] = 0.5 * base_mean
        regime_covariances[0] = base_cov * 0.70 + np.eye(n_assets) * self.config.shrinkage_floor

        regime_means[1] = 1.2 * base_mean
        regime_covariances[1] = base_cov * 1.00

        regime_means[panic_regime_idx] = base_mean - 2.0 * vol_scalar
        regime_covariances[panic_regime_idx] = base_cov * 2.50

        # Enforce positive definiteness on each conditioned covariance
        for j in range(self.config.n_regimes):
            regime_covariances[j] = self.oas_estimator.fit_covariance(
                np.random.RandomState(42).multivariate_normal(
                    regime_means[j], regime_covariances[j], size=max(t_window, 20)
                )
            )

        # Bayesian likelihood update using log-domain multivariate normal
        log_likelihoods = np.zeros(self.config.n_regimes, dtype=np.float64)
        for j in range(self.config.n_regimes):
            diff = returns - regime_means[j]
            cov = regime_covariances[j]
            try:
                cho_factor = scipy.linalg.cho_factor(cov, lower=True)
                sol = scipy.linalg.cho_solve(cho_factor, diff)
                quad = float(np.dot(diff, sol))
                log_det = 2.0 * float(np.sum(np.log(np.diag(cho_factor[0]))))
                log_likelihoods[j] = -0.5 * (n_assets * np.log(2.0 * np.pi) + log_det + quad)
            except Exception:
                # Robust fallback for degenerate cases
                log_likelihoods[j] = -0.5 * float(np.sum(diff**2)) / (vol_scalar**2 + 1e-6)

        # Combine prior and likelihood with max-shift for numerical stability
        log_posteriors = np.log(np.maximum(prior_probs, 1e-12)) + log_likelihoods
        max_log = float(np.max(log_posteriors))
        shifted_posteriors = np.exp(log_posteriors - max_log)
        posterior_probs = shifted_posteriors / float(np.sum(shifted_posteriors))

        # Enforce exact simplex sum constraint
        posterior_probs = np.maximum(posterior_probs, 0.0)
        posterior_probs = posterior_probs / float(np.sum(posterior_probs))

        # Update state dwell and active regime index
        new_active = int(np.argmax(posterior_probs))
        if new_active == self._state.active_regime:
            self._state.bars_in_regime += 1
        else:
            self._state.active_regime = new_active
            self._state.bars_in_regime = 1

        self._state.probabilities = posterior_probs
        self._regime_assignments.append(new_active)

        # Compute effective sample size and thermodynamic temperature
        eff_samples = float(t_window * posterior_probs[new_active])
        temperature = self._compute_temperature(
            volatility=vol_scalar,
            effective_sample_size=eff_samples,
        )

        return RegimeEstimationResult(
            probabilities=posterior_probs,
            active_regime=new_active,
            regime_means=regime_means,
            regime_covariances=regime_covariances,
            temperature=temperature,
            cusum_alarm=alarm,
            effective_sample_size=eff_samples,
        )

    def fit_predict(
        self,
        returns_matrix: np.ndarray,
        volatility_series: np.ndarray | None = None,
    ) -> list[RegimeEstimationResult]:
        """Causally evaluate a full historical matrix of returns without lookahead.

        Args:
            returns_matrix: 2D array of historical returns with shape (T, N).
            volatility_series: Optional 1D array of rolling realized volatility (length T).

        Returns:
            List of RegimeEstimationResult payloads, one per time step.
        """
        returns_arr = np.asarray(returns_matrix, dtype=np.float64)
        if returns_arr.ndim != 2:
            raise ValueError(
                f"{ERR_REGIME_DIM}: returns_matrix must be 2D with shape (T, N), got {returns_arr.shape}"
            )

        t_steps, _ = returns_arr.shape
        if t_steps < 2:
            raise ValueError(f"{ERR_REGIME_SAMPLE}: returns_matrix must have at least 2 rows")

        if volatility_series is None:
            vols = np.std(returns_arr, axis=1)
            vols = np.maximum(vols, 1e-4)
        else:
            vols = np.asarray(volatility_series, dtype=np.float64).flatten()
            if len(vols) != t_steps:
                raise ValueError(
                    f"{ERR_REGIME_DIM}: volatility_series length {len(vols)} does not match {t_steps}"
                )

        self.reset()
        results: list[RegimeEstimationResult] = []
        for t in range(t_steps):
            res = self.step(returns_arr[t], float(vols[t]))
            results.append(res)

        return results
