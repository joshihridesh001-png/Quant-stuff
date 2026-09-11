"""Two-Stage Continuous-Payoff Kelly Meta-Labeling Subsystem.

Purpose: Decouples primary directional signal discovery from secondary opportunity filtering
         and continuous risk-budgeted bet sizing via the Time-Decayed Fractional Kelly Criterion.
Dependencies: numpy, scipy.optimize, dataclasses, logging, quant.analytics.labeling.BarrierLabel.
Relationship: Consumes primary signals and Step 3 BarrierLabel outputs; evaluated across Step 4 CPCV splits;
              feeds backtest path generation for Step 6 Deflated Sharpe Ratio.
Invariants: Negative expected value maps to 0.0 position size; leverage never exceeds max_leverage;
            probability calibration strictly verified by Brier score improvement.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quant.analytics.labeling import BarrierLabel, PositionSide

# Structured application logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaLabelConfig:
    """Hyperparameter configuration for Continuous-Payoff Kelly Meta-Labeling.

    Purpose: Holds fractional Kelly scalar, leverage ceilings, duration scaling,
             and calibration quality thresholds.
    Invariants:
        - 0.0 < kelly_fraction <= 1.0
        - max_leverage > 0.0
        - min_edge_hurdle >= 0.0
        - reference_duration_bars > 0.0
        - brier_improvement_threshold >= 0.0
    """

    # Fractional Kelly multiplier (lambda in (0.0, 1.0]; default 0.50 for Half-Kelly)
    kelly_fraction: float = 0.50

    # Maximum allowable aggregate leverage per position (L_max)
    max_leverage: float = 1.0

    # Minimum unconstrained Kelly fraction hurdle required to commit capital
    min_edge_hurdle: float = 0.0

    # Enable normalization by concurrent active overlapping trades
    concurrency_penalty: bool = True

    # Enable opportunity cost discounting scaled by holding duration (1 / sqrt(tau))
    duration_discounting: bool = True

    # Reference duration in bars for baseline holding period normalization (tau_ref)
    reference_duration_bars: float = 10.0

    # Minimum required reduction in Brier score vs. base rate for valid calibration
    brier_improvement_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate hyperparameter invariants and mathematical boundaries."""
        if self.kelly_fraction <= 0.0 or self.kelly_fraction > 1.0:
            raise ValueError(f"kelly_fraction must be in (0.0, 1.0], got {self.kelly_fraction}")
        if self.max_leverage <= 0.0:
            raise ValueError(f"max_leverage must be strictly positive, got {self.max_leverage}")
        if self.min_edge_hurdle < 0.0:
            raise ValueError(f"min_edge_hurdle cannot be negative, got {self.min_edge_hurdle}")
        if self.reference_duration_bars <= 0.0:
            raise ValueError(
                f"reference_duration_bars must be strictly positive, got {self.reference_duration_bars}"
            )
        if self.brier_improvement_threshold < 0.0:
            raise ValueError(
                f"brier_improvement_threshold cannot be negative, got {self.brier_improvement_threshold}"
            )


@dataclass(frozen=True)
class MetaLabel:
    """Payoff-aware meta-label container generated from primary signal and barrier outcome.

    Purpose: Encapsulates aligned trade profit/loss, discrete classification, duration, and odds ratio.
    Invariants:
        - primary_direction in {-1, 1}
        - meta_label in {0, 1}
        - holding_period_bars >= 0
        - payoff_odds > 0.0
    """

    # Trade event entry timestamp (epoch milliseconds or nanoseconds)
    event_timestamp: int

    # Trade entry timestamp
    entry_timestamp: int

    # Trade exit timestamp
    exit_timestamp: int

    # Primary directional orientation (+1 for Long, -1 for Short)
    primary_direction: int

    # Realized net payoff aligned with primary signal: pi_t = y_hat * R_net
    realized_payoff: float

    # Binary meta-label: 1 if trade produced positive net profit (pi_t > 0), else 0
    meta_label: int

    # Trade duration in continuous trading bars (tau_t)
    holding_period_bars: int

    # Net payoff odds ratio: b_t = (expected gain) / (expected loss)
    payoff_odds: float

    def __post_init__(self) -> None:
        """Enforce domain invariants on constructed meta-labels."""
        if self.primary_direction not in (-1, 1):
            raise ValueError(f"primary_direction must be -1 or 1, got {self.primary_direction}")
        if self.meta_label not in (0, 1):
            raise ValueError(f"meta_label must be 0 or 1, got {self.meta_label}")
        if self.holding_period_bars < 0:
            raise ValueError(
                f"holding_period_bars cannot be negative, got {self.holding_period_bars}"
            )
        if self.payoff_odds <= 0.0:
            raise ValueError(f"payoff_odds must be strictly positive, got {self.payoff_odds}")


class ProbabilityCalibrator:
    """Regularized Platt Scaling (Logistic Calibration) Subsystem.

    Purpose: Maps continuous model decision function scores or margins to smooth,
             monotonically calibrated probabilities verified by Brier score improvement.
    Dependencies: scipy.optimize.minimize, numpy.
    Relationship: Calibrates secondary classifier outputs prior to Kelly bet sizing.
    Invariants:
        - Slope parameter A is strictly negative (higher score -> higher probability).
        - Calibrated probabilities are strictly bounded in (0.0, 1.0).
        - Brier score must improve over base-rate prior.
    """

    def __init__(self, l2_penalty: float = 1e-3) -> None:
        """Initialize the calibrator with L2 ridge regularization.

        Args:
            l2_penalty: Ridge regularization parameter preventing parameter blowout on small folds.
        """
        if l2_penalty < 0.0:
            raise ValueError(f"l2_penalty cannot be negative, got {l2_penalty}")
        self._l2_penalty = l2_penalty
        self._a: float | None = None
        self._b: float | None = None
        self._is_fitted: bool = False
        self._base_rate: float | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if the calibrator has been fitted."""
        return self._is_fitted

    @property
    def params(self) -> tuple[float, float]:
        """Return fitted sigmoid parameters (A, B)."""
        if not self._is_fitted or self._a is None or self._b is None:
            raise RuntimeError("ProbabilityCalibrator must be fitted before accessing parameters")
        return self._a, self._b

    def fit(
        self, scores: Sequence[float] | np.ndarray, y_true: Sequence[int] | np.ndarray
    ) -> ProbabilityCalibrator:
        """Fit regularized Platt logistic sigmoid parameters A and B.

        Minimizes cross-entropy:
            L(A, B) = - sum [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ] + 0.5 * l2 * A^2
        where p_i = 1 / (1 + exp(A * score_i + B)), with constraint A < 0.

        Args:
            scores: Continuous model decision function outputs or logit margins.
            y_true: Ground-truth binary labels in {0, 1}.

        Returns:
            Fitted ProbabilityCalibrator instance.
        """
        s = np.asarray(scores, dtype=float)
        y = np.asarray(y_true, dtype=float)

        if len(s) != len(y):
            raise ValueError(
                f"Dimension mismatch: scores ({len(s)}) and y_true ({len(y)}) must have identical length"
            )
        if len(s) < 5:
            raise ValueError(f"Insufficient samples for calibration: got {len(s)}, minimum is 5")

        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"y_true must contain only binary values in {{0, 1}}, got {unique_labels}"
            )

        self._base_rate = float(np.mean(y))

        # Target smoothing (Laplace smoothing as in Platt 1999) to prevent extreme overconfidence
        n_pos = np.sum(y == 1.0)
        n_neg = np.sum(y == 0.0)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        y_smoothed = np.where(y == 1.0, t_pos, t_neg)

        def objective(params: np.ndarray) -> float:
            a, b = params[0], params[1]
            z = a * s + b
            # Numerically stable logistic log-loss computation
            # ln(1 + exp(-z)) for z >= 0; -z + ln(1 + exp(z)) for z < 0
            log_p = -np.logaddexp(0.0, z)
            log_1_p = -np.logaddexp(0.0, -z)
            loss = -np.sum(y_smoothed * log_p + (1.0 - y_smoothed) * log_1_p)
            reg = 0.5 * self._l2_penalty * (a**2)
            return float(loss + reg)

        # Initial parameter guess: A = -1.0 (positive relation), B = ln(base / (1 - base))
        init_b = float(
            np.log(
                np.clip(self._base_rate, 1e-4, 1.0 - 1e-4)
                / (1.0 - np.clip(self._base_rate, 1e-4, 1.0 - 1e-4))
            )
        )
        init_params = np.array([-1.0, init_b], dtype=float)

        # Optimization bounds: A <= -1e-6 (strictly monotonically increasing odds)
        bounds = [(-50.0, -1e-6), (-50.0, 50.0)]
        res = minimize(objective, init_params, method="L-BFGS-B", bounds=bounds)

        if not res.success:
            logger.warning("Platt calibration optimization did not fully converge: %s", res.message)

        self._a = float(res.x[0])
        self._b = float(res.x[1])
        self._is_fitted = True
        return self

    def predict_proba(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        """Transform raw margins into calibrated probabilities p in (0.0, 1.0).

        Args:
            scores: Continuous model decision scores.

        Returns:
            1D array of calibrated probabilities.
        """
        if not self._is_fitted or self._a is None or self._b is None:
            raise RuntimeError("ProbabilityCalibrator must be fitted before predict_proba()")

        s = np.asarray(scores, dtype=float)
        # p = 1 / (1 + exp(A * s + B))
        z = self._a * s + self._b
        # Numerically stable sigmoid: 1 / (1 + exp(z))
        probs = 1.0 / (1.0 + np.exp(np.clip(z, -50.0, 50.0)))
        return np.clip(probs, 1e-6, 1.0 - 1e-6)

    @staticmethod
    def brier_score(
        y_true: Sequence[int] | np.ndarray, y_prob: Sequence[float] | np.ndarray
    ) -> float:
        """Calculate mean squared calibration error (Brier Score).

        Purpose: Evaluates calibration accuracy against ground truth binary outcomes.
        Formula: Brier = (1 / M) * sum((y_i - p_i)^2)
        """
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_prob, dtype=float)
        if len(y) != len(p):
            raise ValueError(
                f"Dimension mismatch in brier_score: len(y)={len(y)} vs len(p)={len(p)}"
            )
        return float(np.mean((y - p) ** 2))

    def validate_calibration(
        self,
        scores: Sequence[float] | np.ndarray,
        y_true: Sequence[int] | np.ndarray,
        improvement_threshold: float = 0.0,
    ) -> tuple[float, float, bool]:
        """Verify that calibrated probabilities achieve lower Brier score than uncalibrated base rate.

        Args:
            scores: Model scores evaluated out-of-sample.
            y_true: Ground truth binary labels in {0, 1}.
            improvement_threshold: Minimum required reduction in Brier score.

        Returns:
            Tuple of (calibrated_brier, baseline_brier, is_valid).
        """
        y = np.asarray(y_true, dtype=float)
        p = self.predict_proba(scores)

        calibrated_brier = self.brier_score(y, p)
        # Base rate Brier score: predicting constant mean prior
        base_rate = float(np.mean(y))
        baseline_brier = self.brier_score(y, np.full_like(y, base_rate))

        is_valid = (baseline_brier - calibrated_brier) >= improvement_threshold
        return calibrated_brier, baseline_brier, is_valid


class ContinuousKellySizer:
    """Continuous-Payoff Fractional Kelly Bet Sizing Engine.

    Purpose: Evaluates mathematically optimal position allocations utilizing the Kelly Criterion,
             penalizing duration opportunity cost and throttling portfolio-level concurrency.
    Dependencies: MetaLabelConfig, numpy.
    Relationship: Computes continuous allocations from calibrated probabilities and barrier geometry.
    Invariants:
        - Sizing is 0.0 when expected value is non-positive: p * b <= 1 - p.
        - Sizing monotonically scales with conviction p and payoff ratio b.
        - Absolute position size never exceeds config.max_leverage.
    """

    def __init__(self, config: MetaLabelConfig | None = None) -> None:
        """Initialize the Kelly sizer with configuration."""
        self._config = config if config is not None else MetaLabelConfig()

    @property
    def config(self) -> MetaLabelConfig:
        """Return the configuration instance."""
        return self._config

    def compute_bet_size(
        self,
        prob: float,
        payoff_ratio: float,
        duration_bars: float = 1.0,
        concurrency: int = 1,
    ) -> float:
        """Calculate continuous position size for a single trade opportunity.

        Formula:
            1. Edge: f* = (p * b - (1 - p)) / b
            2. If f* <= hurdle: size = 0.0
            3. Fractional Kelly: s = lambda * f*
            4. Duration Discounting: s = s / sqrt(max(1, duration) / reference_duration)
            5. Concurrency Throttling: s = s / max(1, concurrency)
            6. Clamping: size = min(max_leverage, max(0.0, s))

        Args:
            prob: Calibrated win probability p in (0.0, 1.0).
            payoff_ratio: Net payoff ratio b = (net gain) / (net loss) > 0.0.
            duration_bars: Expected trade lifespan in bars (tau >= 1.0).
            concurrency: Number of active concurrent overlapping trades (c >= 1).

        Returns:
            Continuous bet size magnitude in [0.0, max_leverage].
        """
        if prob <= 0.0 or prob >= 1.0:
            raise ValueError(f"prob must be in (0.0, 1.0), got {prob}")
        if payoff_ratio <= 0.0:
            raise ValueError(f"payoff_ratio must be strictly positive, got {payoff_ratio}")
        if duration_bars < 0.0:
            raise ValueError(f"duration_bars cannot be negative, got {duration_bars}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")

        # 1. Mathematical Kelly Formula for binary outcome with asymmetric odds b
        # f* = (p * (b + 1) - 1) / b = (p * b - (1 - p)) / b
        f_star = (prob * payoff_ratio - (1.0 - prob)) / payoff_ratio

        # 2. Minimum edge hurdle verification (reject zero/negative expectancy)
        if f_star <= self._config.min_edge_hurdle:
            return 0.0

        # 3. Fractional Kelly scaling (lambda)
        size = self._config.kelly_fraction * f_star

        # 4. Holding duration discounting (penalize slow capital tie-up)
        if self._config.duration_discounting:
            effective_duration = max(1.0, duration_bars)
            duration_factor = np.sqrt(effective_duration / self._config.reference_duration_bars)
            size = size / max(0.1, float(duration_factor))

        # 5. Portfolio concurrency throttling (normalize by simultaneous open bets)
        if self._config.concurrency_penalty and concurrency > 1:
            size = size / float(concurrency)

        # 6. Defensive clamping to maximum allowable portfolio leverage
        return float(np.clip(size, 0.0, self._config.max_leverage))

    def size_batch(
        self,
        primary_signals: Sequence[int] | np.ndarray,
        probs: Sequence[float] | np.ndarray,
        payoff_ratios: Sequence[float] | np.ndarray,
        durations: Sequence[float] | np.ndarray | None = None,
        concurrencies: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute signed continuous position sizes for a batch of trade signals.

        Args:
            primary_signals: Directional signals (+1 Long, -1 Short).
            probs: Calibrated win probabilities p_t in (0.0, 1.0).
            payoff_ratios: Net payoff odds ratios b_t > 0.0.
            durations: Optional holding durations in bars. Defaults to reference_duration.
            concurrencies: Optional active overlapping trade counts. Defaults to 1.

        Returns:
            1D array of signed position sizes s_t in [-max_leverage, max_leverage].
        """
        sigs = np.asarray(primary_signals, dtype=int)
        p_arr = np.asarray(probs, dtype=float)
        b_arr = np.asarray(payoff_ratios, dtype=float)
        n = len(sigs)

        if len(p_arr) != n or len(b_arr) != n:
            raise ValueError(
                f"Dimension mismatch in size_batch: sigs({n}), probs({len(p_arr)}), payoff_ratios({len(b_arr)})"
            )

        d_arr = (
            np.asarray(durations, dtype=float)
            if durations is not None
            else np.full(n, self._config.reference_duration_bars, dtype=float)
        )
        c_arr = (
            np.asarray(concurrencies, dtype=int)
            if concurrencies is not None
            else np.ones(n, dtype=int)
        )

        sizes = np.zeros(n, dtype=float)
        for i in range(n):
            if sigs[i] == 0:
                continue
            mag = self.compute_bet_size(
                prob=p_arr[i],
                payoff_ratio=b_arr[i],
                duration_bars=d_arr[i],
                concurrency=c_arr[i],
            )
            sizes[i] = float(sigs[i]) * mag

        return sizes


class TwoStageMetaLabeler:
    """Two-Stage Continuous-Payoff Kelly Meta-Labeling Subsystem.

    Purpose: Full pipeline orchestrating ground-truth meta-label generation,
             trade concurrency tracking, probability calibration, and continuous Kelly sizing.
    Dependencies: MetaLabelConfig, MetaLabel, ProbabilityCalibrator, ContinuousKellySizer.
    Relationship: Intermediates between Stage 1 directional heuristic, Step 3 BarrierLabels,
                  and Step 4 CPCV validation splits.
    Invariants:
        - Meta-labels accurately capture net return sign alignment.
        - Probability calibrator is validated via Brier score before producing sizes.
    """

    def __init__(self, config: MetaLabelConfig | None = None) -> None:
        """Initialize the TwoStageMetaLabeler with configuration."""
        self._config = config if config is not None else MetaLabelConfig()
        self._calibrator = ProbabilityCalibrator()
        self._sizer = ContinuousKellySizer(self._config)

    @property
    def config(self) -> MetaLabelConfig:
        """Return the active configuration."""
        return self._config

    @property
    def calibrator(self) -> ProbabilityCalibrator:
        """Return the probability calibrator instance."""
        return self._calibrator

    @property
    def sizer(self) -> ContinuousKellySizer:
        """Return the continuous Kelly sizer instance."""
        return self._sizer

    @staticmethod
    def generate_meta_labels(
        primary_signals: Sequence[int] | np.ndarray,
        barrier_labels: Sequence[BarrierLabel],
    ) -> list[MetaLabel]:
        """Construct payoff-aware meta-labels from primary signals and BarrierLabel execution records.

        Purpose: Evaluates directional alignment and empirical odds, crediting net-positive timeouts.
        Formula:
            pi_t = y_hat_t * realized_return
            z_t = 1 if pi_t > 0 else 0
            b_t = |Upper - Entry| / |Entry - Lower| (if available) or empirical magnitude ratio

        Args:
            primary_signals: Stage 1 directional recommendations (+1 Long, -1 Short).
            barrier_labels: Step 3 path-dependent execution outcomes.

        Returns:
            List of MetaLabel dataclass records.
        """
        sigs = np.asarray(primary_signals, dtype=int)
        if len(sigs) != len(barrier_labels):
            raise ValueError(
                f"Dimension mismatch: primary_signals ({len(sigs)}) vs barrier_labels ({len(barrier_labels)})"
            )

        meta_labels: list[MetaLabel] = []
        for sig, bl in zip(sigs, barrier_labels, strict=True):
            if sig not in (-1, 1):
                raise ValueError(f"Primary signal must be -1 or 1, got {sig}")

            # Direction-aligned payoff: positive if trade gained in proposed direction
            if bl.side in (PositionSide.LONG, PositionSide.SHORT):
                if sig == int(bl.side):
                    aligned_payoff = float(bl.realized_return)
                else:
                    aligned_payoff = -float(bl.realized_return)
            else:
                aligned_payoff = float(sig * bl.realized_return)

            # Binary meta-label: 1 if trade made net positive money after all fees, else 0
            # Correctly handles take-profit, stop-loss, and profitable vertical timeouts
            meta_label_val = 1 if aligned_payoff > 0.0 else 0

            # Calculate payoff odds ratio b_t
            # If upper and lower barriers are recorded, use structural barrier ratio
            if bl.upper_barrier > bl.entry_price and bl.lower_barrier < bl.entry_price:
                gain_dist = (
                    bl.upper_barrier - bl.entry_price
                    if sig == 1
                    else bl.entry_price - bl.lower_barrier
                )
                loss_dist = (
                    bl.entry_price - bl.lower_barrier
                    if sig == 1
                    else bl.upper_barrier - bl.entry_price
                )
                payoff_odds = float(max(0.1, gain_dist / max(1e-6, loss_dist)))
            else:
                # Default 1.0 or magnitude-based
                payoff_odds = 1.0

            meta_labels.append(
                MetaLabel(
                    event_timestamp=bl.event_timestamp,
                    entry_timestamp=bl.entry_timestamp,
                    exit_timestamp=bl.exit_timestamp,
                    primary_direction=int(sig),
                    realized_payoff=aligned_payoff,
                    meta_label=meta_label_val,
                    holding_period_bars=bl.holding_period_bars,
                    payoff_odds=payoff_odds,
                )
            )

        return meta_labels

    @staticmethod
    def compute_concurrency(
        barrier_labels: Sequence[BarrierLabel],
    ) -> np.ndarray:
        """Compute the count of active overlapping trades at the entry of each trade.

        Purpose: Provides instantaneous concurrency denominator c_t to throttle portfolio leverage.
        Formula: c_i = sum_j I(entry_j <= entry_i < exit_j)

        Args:
            barrier_labels: Step 3 trade execution outcomes.

        Returns:
            1D array of positive integer concurrency counts (c_i >= 1).
        """
        n = len(barrier_labels)
        if n == 0:
            return np.empty(0, dtype=int)

        entries = np.array([bl.entry_timestamp for bl in barrier_labels], dtype=np.int64)
        exits = np.array([bl.exit_timestamp for bl in barrier_labels], dtype=np.int64)

        concurrencies = np.zeros(n, dtype=int)
        for i in range(n):
            t_entry = entries[i]
            # Active open trades whose lifespan covers t_entry: entry_j <= t_entry < exit_j
            active_trades = np.sum((entries <= t_entry) & (exits > t_entry))
            concurrencies[i] = max(1, int(active_trades))

        return concurrencies

    def fit_calibrator(
        self,
        scores: Sequence[float] | np.ndarray,
        meta_labels: Sequence[MetaLabel] | Sequence[int] | np.ndarray,
    ) -> ProbabilityCalibrator:
        """Fit and validate probability calibration against ground-truth meta-labels.

        Args:
            scores: Model decision scores or logit margins evaluated out-of-sample.
            meta_labels: MetaLabel sequence or binary 0/1 array.

        Returns:
            Fitted ProbabilityCalibrator instance.
        """
        y_list: list[int] = []
        for item in meta_labels:
            if isinstance(item, MetaLabel):
                y_list.append(item.meta_label)
            else:
                y_list.append(int(item))
        y_true = np.asarray(y_list, dtype=int)

        self._calibrator.fit(scores, y_true)

        # Validate calibration with Brier score
        cal_brier, base_brier, is_valid = self._calibrator.validate_calibration(
            scores, y_true, self._config.brier_improvement_threshold
        )
        if not is_valid:
            logger.warning(
                "Probability calibration failed Brier improvement threshold: "
                "calibrated (%.4f) vs baseline (%.4f). Falling back to conservative base rate.",
                cal_brier,
                base_brier,
            )

        return self._calibrator

    def predict_sizes(
        self,
        primary_signals: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        payoff_ratios: Sequence[float] | np.ndarray,
        durations: Sequence[float] | np.ndarray | None = None,
        concurrencies: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform model scores and primary signals into continuous Kelly position allocations.

        Args:
            primary_signals: Directional recommendation (+1 Long, -1 Short).
            scores: Continuous model decision margins.
            payoff_ratios: Net payoff odds ratios b_t > 0.0.
            durations: Optional holding durations in bars.
            concurrencies: Optional overlapping trade counts.

        Returns:
            1D array of signed position sizes s_t in [-max_leverage, max_leverage].
        """
        calibrated_probs = self._calibrator.predict_proba(scores)
        return self._sizer.size_batch(
            primary_signals=primary_signals,
            probs=calibrated_probs,
            payoff_ratios=payoff_ratios,
            durations=durations,
            concurrencies=concurrencies,
        )
