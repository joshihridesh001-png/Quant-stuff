"""Dynamic Volatility Triple-Barrier Labeling Subsystem.

Purpose: Evaluates path-dependent trade execution labels against volatility-scaled barriers.
Dependencies: numpy, dataclasses, enum, domain models (MarketDataBatch).
Relationship: Consumes MarketDataBatch from Step 1; feeds CPCV (Step 4) and Meta-Labeling (Step 5).
Invariants: Sizing volatility is strictly causal (lagged t-1); stop-loss priority on dual collision; zero lookahead.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np

from quant.domain.models import MarketDataBatch

# Structured application logger
logger = logging.getLogger(__name__)


class BarrierTouchReason(StrEnum):
    """Categorical classification of barrier touch events.

    Purpose: Disambiguates exit mechanism for auditing and strategy diagnostics.
    Invariants: Exactly one exit reason assigned per trade lifecycle.
    """

    UPPER = "upper"  # Touched upper horizontal barrier
    LOWER = "lower"  # Touched lower horizontal barrier
    VERTICAL = "vertical"  # Reached maximum time horizon without horizontal touch
    COLLISION_STOP = "collision_stop"  # Dual-barrier touch in same bar resolved to stop loss


class PositionSide(IntEnum):
    """Trade directionality parameter.

    Purpose: Maps physical barriers to profit-taking and stop-loss economic semantics.
    Invariants: LONG (+1), SHORT (-1), or UNSIGNED (0).
    """

    LONG = 1
    SHORT = -1
    UNSIGNED = 0


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Hyperparameter configuration for dynamic volatility triple-barrier labeling.

    Purpose: Holds barrier multipliers, horizon parameters, execution friction, and defensive bounds.
    Invariants:
        - profit_multiplier > 0.0, stop_multiplier > 0.0
        - horizon_bars >= 1, volatility_window >= 2
        - 0.0 < volatility_floor <= volatility_cap
        - spread_bps >= 0.0, fee_bps >= 0.0
        - execution_delay_bars >= 0
    """

    # Profit-taking horizontal barrier multiplier (c1)
    profit_multiplier: float = 2.0

    # Stop-loss horizontal barrier multiplier (c2)
    stop_multiplier: float = 1.0

    # Vertical barrier duration in continuous trading bars (H)
    horizon_bars: int = 60

    # Rolling window length for Parkinson range volatility calculation (W)
    volatility_window: int = 20

    # Minimum volatility floor protecting against zero-volatility collapse (sigma_floor)
    volatility_floor: float = 1e-4

    # Maximum volatility cap protecting against runaway barrier blowout (sigma_cap)
    volatility_cap: float = 0.50

    # One-way half-spread in basis points (1 bps = 0.0001)
    spread_bps: float = 1.0

    # One-way exchange transaction fee in basis points
    fee_bps: float = 2.0

    # Optional discrete tick size for price grid quantization (e.g. 0.01)
    tick_size: float | None = None

    # Enforce geometric symmetry via natural log-price space (ln(Upper) = ln(P) + c1*sigma)
    use_log_barriers: bool = True

    # Scale barrier width by square-root of horizon (sigma * sqrt(H)) for diffusion consistency
    scale_by_sqrt_horizon: bool = False

    # Pessimistic execution policy: assume stop-loss triggered first on dual intra-bar touch
    pessimistic_collision: bool = True

    # Execution latency in bars (1 means signal on bar t close executes on bar t+1 open)
    execution_delay_bars: int = 1

    # If True, sign the return for vertical expirations (sign(R) if |R| > friction else 0)
    sign_expiration_returns: bool = False

    def __post_init__(self) -> None:
        """Enforce parameter validation and mathematical invariants at configuration time."""
        if self.profit_multiplier <= 0.0:
            raise ValueError(
                f"profit_multiplier must be strictly positive, got {self.profit_multiplier}"
            )
        if self.stop_multiplier <= 0.0:
            raise ValueError(
                f"stop_multiplier must be strictly positive, got {self.stop_multiplier}"
            )
        if self.horizon_bars < 1:
            raise ValueError(f"horizon_bars must be at least 1, got {self.horizon_bars}")
        if self.volatility_window < 2:
            raise ValueError(f"volatility_window must be at least 2, got {self.volatility_window}")
        if self.volatility_floor <= 0.0:
            raise ValueError(
                f"volatility_floor must be strictly positive, got {self.volatility_floor}"
            )
        if self.volatility_cap < self.volatility_floor:
            raise ValueError(
                f"volatility_cap ({self.volatility_cap}) cannot be less than volatility_floor ({self.volatility_floor})"
            )
        if self.spread_bps < 0.0:
            raise ValueError(f"spread_bps cannot be negative, got {self.spread_bps}")
        if self.fee_bps < 0.0:
            raise ValueError(f"fee_bps cannot be negative, got {self.fee_bps}")
        if self.execution_delay_bars < 0:
            raise ValueError(
                f"execution_delay_bars cannot be negative, got {self.execution_delay_bars}"
            )
        if self.tick_size is not None and self.tick_size <= 0.0:
            raise ValueError(f"tick_size must be strictly positive, got {self.tick_size}")


@dataclass(frozen=True)
class BarrierLabel:
    """Immutable path-dependent execution record generated by the Triple-Barrier engine.

    Purpose: Complete trade outcome container encapsulating discrete label and continuous payoff.
    Invariants:
        - entry_timestamp <= exit_timestamp
        - label in {-1, 0, 1}
        - holding_period_bars >= 0
    """

    # Event trigger timestamp (nanoseconds or epoch milliseconds)
    event_timestamp: int

    # Trade execution entry timestamp
    entry_timestamp: int

    # Trade exit timestamp (barrier touch or expiration)
    exit_timestamp: int

    # Effective entry price after half-spread and entry friction
    entry_price: float

    # Realized exit price (honors open price on gaps; barrier level otherwise)
    exit_price: float

    # Trade position direction
    side: PositionSide

    # Categorical classification outcome (+1: Profit, -1: Stop-Loss, 0: Neutral Expiration)
    label: int

    # Net realized return after round-trip transaction costs and fees
    realized_return: float

    # Specific exit mechanism triggered
    touch_reason: BarrierTouchReason

    # Number of bars elapsed between entry and exit
    holding_period_bars: int

    # Instantaneous realized volatility recorded at entry (sigma_entry)
    volatility_at_entry: float

    # Evaluated upper horizontal barrier level
    upper_barrier: float

    # Evaluated lower horizontal barrier level
    lower_barrier: float

    def __post_init__(self) -> None:
        """Validate defensive invariants of the generated trade label."""
        if self.entry_timestamp > self.exit_timestamp:
            raise ValueError(
                f"entry_timestamp ({self.entry_timestamp}) cannot be strictly greater than exit_timestamp ({self.exit_timestamp})"
            )
        if self.label not in (-1, 0, 1):
            raise ValueError(f"label must be in {{-1, 0, 1}}, got {self.label}")
        if self.holding_period_bars < 0:
            raise ValueError(
                f"holding_period_bars cannot be negative, got {self.holding_period_bars}"
            )


def compute_parkinson_volatility(
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Compute Parkinson range-based realized volatility with strict causal lagging.

    Purpose: Measures intra-bar price dispersion with 5x greater efficiency than close-to-close returns.
    Dependencies: numpy log and rolling window operations.
    Mathematical Formulation:
        sigma_t = sqrt( (1 / (4 * ln(2) * W)) * sum_{i=0}^{W-1} (ln(H_{t-1-i} / L_{t-1-i}))^2 )
    Invariants:
        - Output at index t strictly uses observations up to index t-1 (zero lookahead leakage).
        - Output is zero-padded for indices < window + 1.
    """
    n = len(highs)
    if n != len(lows):
        raise ValueError(f"Highs length ({n}) must match Lows length ({len(lows)})")
    if window < 2:
        raise ValueError(f"Rolling window must be >= 2, got {window}")

    # Defensive check: strictly positive prices required for logarithm
    if np.any(highs <= 0.0) or np.any(lows <= 0.0):
        raise ValueError("High and Low prices must be strictly positive for Parkinson volatility")

    # Invariant: Highs must be >= Lows
    if np.any(highs < lows):
        raise ValueError("High prices cannot be strictly less than Low prices")

    # Compute intra-bar log-range squared: (ln(H_i / L_i))^2
    log_hl = np.log(highs / lows)
    log_hl_sq = log_hl**2

    # Normalization scale factor: 1.0 / (4.0 * ln(2) * W)
    scale = 1.0 / (4.0 * math.log(2.0) * float(window))

    # Rolling sum using 1D uniform convolution filter
    kernel = np.ones(window, dtype=np.float64)
    # Causal sliding window sum
    rolling_sum = np.convolve(log_hl_sq, kernel, mode="valid")

    # Compute raw parkinson volatility vector
    raw_vol = np.sqrt(np.maximum(0.0, rolling_sum * scale))

    # Construct strictly causal output lagged by 1 bar:
    # Index t receives volatility computed from window ending at index t-1.
    # Therefore, valid volatility starts at index window + 1.
    causal_vol = np.zeros(n, dtype=np.float64)
    # The rolling sum at position j corresponds to window ending at j + window - 1.
    # To lag by 1 bar, place it at (j + window).
    dest_start = window
    dest_end = min(n, dest_start + len(raw_vol))
    src_len = dest_end - dest_start
    if src_len > 0:
        causal_vol[dest_start:dest_end] = raw_vol[:src_len]

    return causal_vol


class DynamicTripleBarrierLabeler:
    """Dynamic Volatility Triple-Barrier Labeling Engine.

    Purpose: Evaluates path-dependent trade outcomes (profit, stop, timeout) scaled by causal volatility.
    Dependencies: TripleBarrierConfig, MarketDataBatch, compute_parkinson_volatility.
    Relationship: Consumed by cross-validation pipelines (Step 4) and meta-labeling classifiers (Step 5).
    Invariants:
        - Never looks ahead: volatility and barrier thresholds are locked before trade path evolves.
        - Pessimistic stop-loss priority on dual intra-bar penetrations.
        - Realized returns deduct round-trip transaction spread and fee friction.
    """

    def __init__(self, config: TripleBarrierConfig | None = None) -> None:
        """Initialize labeler with configuration hyperparameters."""
        self.config = config or TripleBarrierConfig()

    def label_batch(
        self,
        market_batch: MarketDataBatch,
        event_indices: np.ndarray | Sequence[int] | None = None,
        sides: np.ndarray | Sequence[int] | None = None,
    ) -> list[BarrierLabel]:
        """Label market data events from a columnar MarketDataBatch container.

        Purpose: Primary batch labeling entry point conforming to platform DDD architecture.
        Dependencies: MarketDataBatch contiguous arrays.
        """
        return self.label_arrays(
            timestamps=market_batch.timestamps,
            opens=market_batch.opens,
            highs=market_batch.highs,
            lows=market_batch.lows,
            closes=market_batch.closes,
            event_indices=event_indices,
            sides=sides,
        )

    def label_arrays(
        self,
        timestamps: np.ndarray,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        event_indices: np.ndarray | Sequence[int] | None = None,
        sides: np.ndarray | Sequence[int] | None = None,
    ) -> list[BarrierLabel]:
        """Evaluate triple barriers over raw contiguous 1D price arrays.

        Purpose: Vectorized analytical labeling routine achieving sub-second performance.
        Dependencies: compute_parkinson_volatility, NumPy array indexing.
        Invariants:
            - Series length must strictly exceed horizon_bars + execution_delay_bars.
            - All input arrays must share identical length N.
        """
        n = len(timestamps)
        # Validate dimensional alignment across all price vectors
        if not (len(opens) == len(highs) == len(lows) == len(closes) == n):
            raise ValueError(
                f"Dimension mismatch in input arrays: timestamps={n}, opens={len(opens)}, "
                f"highs={len(highs)}, lows={len(lows)}, closes={len(closes)}"
            )

        # Invariant check: Dataset must have sufficient bars for volatility warm-up and horizon
        min_required_bars = (
            self.config.volatility_window
            + self.config.horizon_bars
            + self.config.execution_delay_bars
        )
        if n < min_required_bars:
            raise ValueError(
                f"Input series length ({n}) is too short. Minimum required is {min_required_bars} bars "
                f"(volatility_window={self.config.volatility_window} + horizon={self.config.horizon_bars} + delay={self.config.execution_delay_bars})"
            )

        # Pre-compute causal realized volatility across the entire series (O(N) operation)
        causal_vol = compute_parkinson_volatility(
            highs=highs,
            lows=lows,
            window=self.config.volatility_window,
        )

        # Determine target event indices to evaluate
        if event_indices is None:
            # Evaluate all bars that have completed volatility warm-up and have room for execution
            start_idx = self.config.volatility_window + 1
            max_event_idx = n - self.config.horizon_bars - self.config.execution_delay_bars
            if start_idx >= max_event_idx:
                return []
            events: np.ndarray = np.arange(start_idx, max_event_idx, dtype=np.int64)
        else:
            events = np.asarray(event_indices, dtype=np.int64)

        # Determine trade directions (defaults to LONG (+1) if unspecified)
        if sides is None:
            side_arr = np.full(len(events), PositionSide.LONG, dtype=np.int32)
        else:
            side_arr = np.asarray(sides, dtype=np.int32)
            if len(side_arr) != len(events):
                raise ValueError(
                    f"sides length ({len(side_arr)}) must match event_indices length ({len(events)})"
                )

        # Round-trip friction in decimal: 2 * (spread_bps + fee_bps) * 1e-4
        round_trip_friction = 2.0 * (self.config.spread_bps + self.config.fee_bps) * 1e-4
        one_way_friction = (self.config.spread_bps + self.config.fee_bps) * 1e-4

        labels: list[BarrierLabel] = []

        # Iterate over each discrete event candidate
        for idx_pos, event_idx in enumerate(events):
            # Check event index boundaries
            if event_idx < 0 or event_idx >= n:
                continue

            entry_idx = event_idx + self.config.execution_delay_bars
            # Boundary guard: Must have sufficient future bars to observe horizon
            if entry_idx >= n:
                continue

            # Determine position side
            raw_side = int(side_arr[idx_pos])
            try:
                side = PositionSide(raw_side)
            except ValueError:
                side = PositionSide.LONG

            # Trade enters at the open of the entry bar (with entry friction adjustment)
            nominal_entry_price = float(opens[entry_idx])
            if nominal_entry_price <= 0.0:
                continue

            # Effective entry price incorporates one-way spread/fee drag
            if side == PositionSide.LONG:
                effective_entry_price = nominal_entry_price * (1.0 + one_way_friction)
            elif side == PositionSide.SHORT:
                effective_entry_price = nominal_entry_price * (1.0 - one_way_friction)
            else:
                effective_entry_price = nominal_entry_price

            # Retrieve causal volatility recorded before the entry bar
            raw_sigma = float(causal_vol[entry_idx])
            # Clamp volatility between floor and cap
            sigma = max(self.config.volatility_floor, min(self.config.volatility_cap, raw_sigma))

            # Apply diffusion square-root horizon scaling if configured
            if self.config.scale_by_sqrt_horizon:
                effective_sigma = sigma * math.sqrt(float(self.config.horizon_bars))
            else:
                effective_sigma = sigma

            # Compute upper and lower horizontal barrier levels
            upper_mult = self.config.profit_multiplier
            lower_mult = self.config.stop_multiplier

            if self.config.use_log_barriers:
                # Geometric log-price space guarantees symmetry under log-normal price diffusion
                ln_p = math.log(nominal_entry_price)
                if side == PositionSide.SHORT:
                    # For Short: Price rising hits Stop Loss; Price falling hits Profit Target
                    upper_level = math.exp(ln_p + lower_mult * effective_sigma)
                    lower_level = math.exp(ln_p - upper_mult * effective_sigma)
                else:
                    # For Long and Unsigned: Price rising is Upper; Price falling is Lower
                    upper_level = math.exp(ln_p + upper_mult * effective_sigma)
                    lower_level = math.exp(ln_p - lower_mult * effective_sigma)
            else:
                # Linear percentage space
                if side == PositionSide.SHORT:
                    upper_level = nominal_entry_price * (1.0 + lower_mult * effective_sigma)
                    lower_level = nominal_entry_price * (1.0 - upper_mult * effective_sigma)
                else:
                    upper_level = nominal_entry_price * (1.0 + upper_mult * effective_sigma)
                    lower_level = nominal_entry_price * (1.0 - lower_mult * effective_sigma)

            # Quantize barriers to discrete exchange tick grid if tick_size specified
            if self.config.tick_size is not None:
                ts = self.config.tick_size
                upper_level = round(upper_level / ts) * ts
                lower_level = round(lower_level / ts) * ts

            # Horizon window: bars from entry_idx to min(n - 1, entry_idx + horizon_bars)
            max_horizon_idx = min(n - 1, entry_idx + self.config.horizon_bars)

            # Evaluate path evolution bar-by-bar to find first barrier touch
            exit_idx = max_horizon_idx
            exit_price = float(closes[max_horizon_idx])
            touch_reason = BarrierTouchReason.VERTICAL

            for k in range(entry_idx, max_horizon_idx + 1):
                bar_open = float(opens[k])
                bar_high = float(highs[k])
                bar_low = float(lows[k])

                # 1. Check discontinuous opening gap first
                # If market opened beyond upper barrier
                if bar_open >= upper_level:
                    exit_idx = k
                    # Realized exit price is the actual gap open, NOT the theoretical barrier line
                    exit_price = bar_open
                    touch_reason = BarrierTouchReason.UPPER
                    break

                # If market opened below lower barrier
                if bar_open <= lower_level:
                    exit_idx = k
                    exit_price = bar_open
                    touch_reason = BarrierTouchReason.LOWER
                    break

                # 2. Check intra-bar touches via High and Low
                upper_touched = bar_high >= upper_level
                lower_touched = bar_low <= lower_level

                if upper_touched and lower_touched:
                    # Intra-bar collision: both barriers breached in same candle
                    exit_idx = k
                    if self.config.pessimistic_collision:
                        # Conservative institutional rule: assume stop-loss was triggered first
                        touch_reason = BarrierTouchReason.COLLISION_STOP
                        exit_price = upper_level if side == PositionSide.SHORT else lower_level
                    else:
                        touch_reason = BarrierTouchReason.UPPER
                        exit_price = upper_level
                    break

                if upper_touched:
                    exit_idx = k
                    exit_price = upper_level
                    touch_reason = BarrierTouchReason.UPPER
                    break

                if lower_touched:
                    exit_idx = k
                    exit_price = lower_level
                    touch_reason = BarrierTouchReason.LOWER
                    break

            # Calculate raw percentage return relative to nominal entry
            if nominal_entry_price > 0.0:
                raw_return = (exit_price - nominal_entry_price) / nominal_entry_price
            else:
                raw_return = 0.0

            # Directional return adjusted for position side and full round-trip friction
            if side == PositionSide.LONG:
                net_return = raw_return - round_trip_friction
            elif side == PositionSide.SHORT:
                net_return = -raw_return - round_trip_friction
            else:
                # Unsigned position measures breakout magnitude deducting friction
                net_return = abs(raw_return) - round_trip_friction

            # Assign categorical classification label (+1, -1, 0)
            if touch_reason == BarrierTouchReason.UPPER:
                label_val = -1 if side == PositionSide.SHORT else 1
            elif touch_reason in (BarrierTouchReason.LOWER, BarrierTouchReason.COLLISION_STOP):
                label_val = 1 if side == PositionSide.SHORT else -1
            else:
                # Vertical expiration reached
                if self.config.sign_expiration_returns:
                    if net_return > 0.0:
                        label_val = 1
                    elif net_return < 0.0:
                        label_val = -1
                    else:
                        label_val = 0
                else:
                    label_val = 0

            # Compute holding duration
            holding_bars = exit_idx - entry_idx

            labels.append(
                BarrierLabel(
                    event_timestamp=int(timestamps[event_idx]),
                    entry_timestamp=int(timestamps[entry_idx]),
                    exit_timestamp=int(timestamps[exit_idx]),
                    entry_price=effective_entry_price,
                    exit_price=exit_price,
                    side=side,
                    label=label_val,
                    realized_return=float(net_return),
                    touch_reason=touch_reason,
                    holding_period_bars=holding_bars,
                    volatility_at_entry=sigma,
                    upper_barrier=upper_level,
                    lower_barrier=lower_level,
                )
            )

        return labels
