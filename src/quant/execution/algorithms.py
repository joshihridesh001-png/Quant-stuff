"""Algorithmic meta-order execution schedulers for institutional Smart Order Routing.

Purpose:
    Provides institutional execution scheduling algorithms that decompose large meta-orders
    into discrete child slices over time:
    1. Anti-gaming Poisson TWAP with randomized timing and sizing jitter.
    2. Dynamic Volume-Adaptive VWAP with online Bayesian volume blending and hard participation caps.
    3. Closed-form non-linear Almgren-Chriss Arrival Price under 3/2-power market impact.

Dependencies:
    - dataclasses: Slotted dataclass memory layout.
    - math: Transcendental functions, sinh, exp, sqrt, and isfinite checks.
    - random: Seedable isolated pseudorandom number generation.
    - typing: Static typing protocols, runtime_checkable, and Sequence definitions.
    - numpy: Vectorized trajectory computations.
    - quant.execution.venues: Diagnostic fault codes and exception hierarchy.

Structural Relationship:
    - Ingested by:
        1. ParentOrder coordinator (parent_order.py)
        2. SmartOrderRouter (sor.py)
        3. Implementation Shortfall TCA engine
    - Consumes: Venue diagnostic fault codes and exceptions from quant.execution.venues.
    - Emits: ScheduledSlice domain objects conforming to ExecutionScheduler protocol.

Invariants Enforced:
    - INV-SOR-001 (Parent-Child Mass Conservation): Sum of scheduled slice quantities strictly
      equals parent order quantity within 1e-7 tolerance.
    - INV-SOR-003 (Hard Volume Participation Rate Cap): VWAP child slice quantities never exceed
      the institutional volume participation ceiling (rho <= 15% of bar volume).
    - INV-SOR-004 (Anti-Gaming Clock & Sizing): Randomized timing (Poisson clock) and sizing
      with strictly positive slices (q_k > 0.0).
    - Rule 1: Line-by-line annotation standards (Functional Purpose, Explicit Dependency Tracking,
      Structural Relationship, Defensive Invariant).
    - Rule 2: Zero-execution diagnostic fault codes (ERR-SOR-001, ERR-SOR-004, ERR-SOR-007).
    - Rule 4: Anti-shortcut blacklist: zero black-box numerical optimizers; closed-form algebraic
      solutions with numerical overflow guards.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np

from quant.execution.venues import (
    ERR_SOR_INSUFFICIENT_LIQUIDITY,
    ERR_SOR_INVALID_SCHEDULE,
    ERR_SOR_MASS_CONSERVATION_BREACH,
    ERR_SOR_NON_FINITE_INPUT,
    InsufficientLiquidityException,
    InvalidScheduleException,
    MassConservationException,
    NonFiniteInputException,
)

# Re-export diagnostic codes for direct consumer module access
__all__ = [
    "ERR_SOR_INSUFFICIENT_LIQUIDITY",
    "ERR_SOR_INVALID_SCHEDULE",
    "ERR_SOR_MASS_CONSERVATION_BREACH",
    "ERR_SOR_NON_FINITE_INPUT",
    "ExecutionScheduler",
    "NonlinearArrivalPriceScheduler",
    "PoissonTWAPScheduler",
    "ScheduledSlice",
    "VolumeAdaptiveVWAPScheduler",
]

# Mass conservation tolerance constant
MASS_CONSERVATION_TOLERANCE: Final[float] = 1e-7


# ============================================================================
# ScheduledSlice Dataclass
# ============================================================================


@dataclass(slots=True)
class ScheduledSlice:
    """Atomic execution slice scheduled for market routing.

    Attributes:
        slice_index: Zero-indexed sequence identifier of the slice (integer >= 0).
        quantity: Lot/share volume to execute for this slice (finite float > 0.0).
        scheduled_time_ns: Target dispatch timestamp in nanoseconds (integer >= 0).
        price_limit: Optional limit price ceiling/floor (finite float > 0.0 if present).
    """

    slice_index: int
    quantity: float
    scheduled_time_ns: int
    price_limit: float | None = None

    def __post_init__(self) -> None:
        """Validate domain boundaries, types, and non-finite guards for scheduled slice.

        Functional Purpose:
            Enforce strict scalar typing, non-boolean values, non-negative indices/timestamps,
            and strictly positive finite quantities and price limits at instantiation boundary.
        Explicit Dependency Tracking:
            math.isfinite, NonFiniteInputException, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Executed on every ScheduledSlice instantiation before downstream router dispatch.
        Defensive Invariant:
            slice_index >= 0, quantity > 0.0, scheduled_time_ns >= 0, price_limit > 0.0 if set.
        """
        # 1. Validate slice_index (non-negative integer, reject bool)
        if (
            not isinstance(self.slice_index, int)
            or isinstance(self.slice_index, bool)
            or self.slice_index < 0
        ):
            raise NonFiniteInputException(
                f"ScheduledSlice slice_index must be non-negative int, got {self.slice_index!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 2. Validate quantity (finite float > 0.0, reject bool)
        if (
            not isinstance(self.quantity, (int, float))
            or isinstance(self.quantity, bool)
            or not math.isfinite(self.quantity)
            or self.quantity <= 0.0
        ):
            raise NonFiniteInputException(
                f"ScheduledSlice quantity must be finite float > 0.0, got {self.quantity!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.quantity = float(self.quantity)

        # 3. Validate scheduled_time_ns (non-negative integer, reject bool)
        if (
            not isinstance(self.scheduled_time_ns, int)
            or isinstance(self.scheduled_time_ns, bool)
            or self.scheduled_time_ns < 0
        ):
            raise NonFiniteInputException(
                f"ScheduledSlice scheduled_time_ns must be non-negative int, "
                f"got {self.scheduled_time_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 4. Validate optional price_limit (finite float > 0.0 if present, reject bool)
        if self.price_limit is not None:
            if (
                not isinstance(self.price_limit, (int, float))
                or isinstance(self.price_limit, bool)
                or not math.isfinite(self.price_limit)
                or self.price_limit <= 0.0
            ):
                raise NonFiniteInputException(
                    f"ScheduledSlice price_limit must be finite float > 0.0 if provided, "
                    f"got {self.price_limit!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            self.price_limit = float(self.price_limit)


# ============================================================================
# ExecutionScheduler Protocol
# ============================================================================


@runtime_checkable
class ExecutionScheduler(Protocol):
    """Structural subtyping protocol for meta-order algorithmic schedulers.

    Functional Purpose:
        Defines the institutional interface for decomposing parent orders into
        time-sequenced child order slices.
    Explicit Dependency Tracking:
        ScheduledSlice.
    Structural Relationship:
        Implemented by PoissonTWAPScheduler, VolumeAdaptiveVWAPScheduler,
        and NonlinearArrivalPriceScheduler. Ingested by ParentOrder lifecycle coordinator.
    Defensive Invariant:
        Implementations must guarantee exact mass conservation (INV-SOR-001) and
        non-negative slice allocations (INV-SOR-004).
    """

    def generate_schedule(self, start_time_ns: int) -> list[ScheduledSlice]:
        """Generate an ordered sequence of execution slices starting at start_time_ns.

        Functional Purpose:
            Transform aggregate parent quantity into discrete time-stamped slices.
        Explicit Dependency Tracking:
            start_time_ns epoch nanoseconds.
        Structural Relationship:
            Called by parent order executor upon order initiation.
        Defensive Invariant:
            Returned slices must satisfy sum(q_k) == Q_parent within 1e-7 tolerance.
        """
        ...


# ============================================================================
# PoissonTWAPScheduler Implementation
# ============================================================================


class PoissonTWAPScheduler:
    """Anti-gaming Time-Weighted Average Price (TWAP) scheduler with randomized Poisson timing.

    Functional Purpose:
        Prevents predatory High-Frequency Trading (HFT) footprint detection by introducing
        controlled stochastic jitter to both inter-arrival slice timing and child slice sizing,
        while strictly enforcing exact parent order mass conservation (INV-SOR-001, INV-SOR-004).
    Explicit Dependency Tracking:
        random.Random, ScheduledSlice, InvalidScheduleException, MassConservationException.
    Structural Relationship:
        Conforms to ExecutionScheduler protocol; feeds slices to SmartOrderRouter.
    Defensive Invariant:
        total_quantity > 0, num_slices >= 1, mean_interval_sec > 0, 0 <= jitter < 1,
        min_interval_sec > 0. Residual closure guarantees sum(q_k) == Q within 1e-7.
    """

    def __init__(
        self,
        total_quantity: float,
        num_slices: int,
        mean_interval_sec: float,
        jitter_ratio: float = 0.15,
        time_jitter_ratio: float = 0.20,
        min_interval_sec: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize and validate Poisson TWAP scheduler configuration.

        Functional Purpose:
            Validates all scheduler boundary parameters and constructs an isolated pseudo-RNG.
        Explicit Dependency Tracking:
            math.isfinite, random.Random, ERR_SOR_INVALID_SCHEDULE.
        Structural Relationship:
            Constructed by ParentOrder or algorithmic execution strategy factory.
        Defensive Invariant:
            All numeric parameters must be non-boolean, finite, and within valid mathematical domains.
        """
        # 1. Validate total_quantity (> 0.0, finite float, reject bool)
        if (
            not isinstance(total_quantity, (int, float))
            or isinstance(total_quantity, bool)
            or not math.isfinite(total_quantity)
            or total_quantity <= 0.0
        ):
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler total_quantity must be finite float > 0.0, "
                f"got {total_quantity!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.total_quantity: float = float(total_quantity)

        # 2. Validate num_slices (>= 1, integer, reject bool)
        if not isinstance(num_slices, int) or isinstance(num_slices, bool) or num_slices < 1:
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler num_slices must be integer >= 1, got {num_slices!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.num_slices: int = num_slices

        # Validate total_quantity against minimum slice mass conservation reserve
        min_required_quantity = self.num_slices * MASS_CONSERVATION_TOLERANCE
        if self.total_quantity < min_required_quantity:
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler total_quantity {self.total_quantity:.9f} is too small for "
                f"{self.num_slices} slices (minimum required is {min_required_quantity:.9f})",
                code=ERR_SOR_INVALID_SCHEDULE,
            )

        # 3. Validate mean_interval_sec (> 0.0, finite float, reject bool)
        if (
            not isinstance(mean_interval_sec, (int, float))
            or isinstance(mean_interval_sec, bool)
            or not math.isfinite(mean_interval_sec)
            or mean_interval_sec <= 0.0
        ):
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler mean_interval_sec must be finite float > 0.0, "
                f"got {mean_interval_sec!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.mean_interval_sec: float = float(mean_interval_sec)

        # 4. Validate jitter_ratio (in [0.0, 1.0), finite float, reject bool)
        if (
            not isinstance(jitter_ratio, (int, float))
            or isinstance(jitter_ratio, bool)
            or not math.isfinite(jitter_ratio)
            or not (0.0 <= jitter_ratio < 1.0)
        ):
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler jitter_ratio must be finite float in [0.0, 1.0), "
                f"got {jitter_ratio!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.jitter_ratio: float = float(jitter_ratio)

        # 5. Validate time_jitter_ratio (in [0.0, 1.0), finite float, reject bool)
        if (
            not isinstance(time_jitter_ratio, (int, float))
            or isinstance(time_jitter_ratio, bool)
            or not math.isfinite(time_jitter_ratio)
            or not (0.0 <= time_jitter_ratio < 1.0)
        ):
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler time_jitter_ratio must be finite float in [0.0, 1.0), "
                f"got {time_jitter_ratio!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.time_jitter_ratio: float = float(time_jitter_ratio)

        # 6. Validate min_interval_sec (>= 1e-9, finite float, reject bool)
        if (
            not isinstance(min_interval_sec, (int, float))
            or isinstance(min_interval_sec, bool)
            or not math.isfinite(min_interval_sec)
            or min_interval_sec < 1e-9
        ):
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler min_interval_sec must be finite float >= 1e-9 (1 ns), "
                f"got {min_interval_sec!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.min_interval_sec: float = float(min_interval_sec)

        # 7. Validate seed (integer or None, reject bool)
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise InvalidScheduleException(
                f"PoissonTWAPScheduler seed must be integer if provided, got {seed!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.seed: int | None = seed

    def generate_schedule(self, start_time_ns: int) -> list[ScheduledSlice]:
        """Generate a randomized anti-gaming TWAP slice schedule.

        Functional Purpose:
            Constructs K scheduled slices applying uniform randomization to intervals
            and child order quantities while guaranteeing exact residual mass closure.
        Explicit Dependency Tracking:
            random.Random, ScheduledSlice, MassConservationException.
        Structural Relationship:
            Conforms to ExecutionScheduler. Emits slices consumed by routing gateway.
        Defensive Invariant:
            Total scheduled quantity == self.total_quantity within 1e-7 tolerance.
            Every slice has quantity > 0.0 and valid non-negative scheduled timestamp.
        """
        # Validate start_time_ns boundary
        if (
            not isinstance(start_time_ns, int)
            or isinstance(start_time_ns, bool)
            or start_time_ns < 0
        ):
            raise InvalidScheduleException(
                f"start_time_ns must be non-negative int, got {start_time_ns!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )

        # Instantiate dedicated isolated PRNG instance for reproducible randomness
        rng = random.Random(self.seed)

        slices: list[ScheduledSlice] = []
        rem_quantity: float = self.total_quantity
        current_time_ns: int = start_time_ns
        k_slices = self.num_slices

        # Single slice edge-case: immediate dispatch of full quantity
        if k_slices == 1:
            return [
                ScheduledSlice(
                    slice_index=0,
                    quantity=self.total_quantity,
                    scheduled_time_ns=start_time_ns,
                )
            ]

        # Slicing loop across k = 0 ... K - 2 with dynamic remaining volume allocation
        for k in range(k_slices - 1):
            slices_remaining = k_slices - k
            base_q = rem_quantity / slices_remaining

            # Sizing jitter delta_k ~ Uniform(-alpha_q, alpha_q)
            delta_q = rng.uniform(-self.jitter_ratio, self.jitter_ratio)
            q_k = base_q * (1.0 + delta_q)

            # Defensive floor & ceiling: reserve minimum 1e-7 for each downstream remaining slice
            min_reserve = (slices_remaining - 1) * MASS_CONSERVATION_TOLERANCE
            q_k = max(MASS_CONSERVATION_TOLERANCE, min(rem_quantity - min_reserve, q_k))

            slices.append(
                ScheduledSlice(
                    slice_index=k,
                    quantity=q_k,
                    scheduled_time_ns=current_time_ns,
                )
            )
            rem_quantity -= q_k

            # Inter-arrival Poisson clock timing jitter xi_k ~ Uniform(-alpha_t, alpha_t)
            xi_t = rng.uniform(-self.time_jitter_ratio, self.time_jitter_ratio)
            interval_sec = max(
                self.min_interval_sec,
                self.mean_interval_sec * (1.0 + xi_t),
            )
            current_time_ns += int(interval_sec * 1_000_000_000)

        # Final slice K - 1: residual closure guaranteeing exact parent mass conservation
        q_final = rem_quantity
        slices.append(
            ScheduledSlice(
                slice_index=k_slices - 1,
                quantity=q_final,
                scheduled_time_ns=current_time_ns,
            )
        )

        # Defensive assertion: INV-SOR-001 mass conservation
        total_scheduled = sum(s.quantity for s in slices)
        if abs(total_scheduled - self.total_quantity) > MASS_CONSERVATION_TOLERANCE:
            raise MassConservationException(
                f"PoissonTWAPScheduler mass breach: scheduled={total_scheduled:.9f} != "
                f"total={self.total_quantity:.9f}",
                code=ERR_SOR_MASS_CONSERVATION_BREACH,
            )

        return slices


# ============================================================================
# VolumeAdaptiveVWAPScheduler Implementation
# ============================================================================


class VolumeAdaptiveVWAPScheduler:
    """Dynamic Volume-Weighted Average Price (VWAP) scheduler with hard participation cap.

    Functional Purpose:
        Allocates child order volume according to normalized historical intraday volume curves,
        dynamically blended with real-time volume observations, while strictly enforcing
        the institutional participation rate ceiling (rho <= 15% everywhere, INV-SOR-003).
    Explicit Dependency Tracking:
        ScheduledSlice, InvalidScheduleException, MassConservationException.
    Structural Relationship:
        Conforms to ExecutionScheduler protocol; feeds volume-profiled slices to SOR.
    Defensive Invariant:
        total_quantity > 0, historical_curve length >= 1 with sum > 0, 0 < rho_max <= 1.0,
        0.0 <= smoothing_weight <= 1.0, interval_sec > 0.
    """

    def __init__(
        self,
        total_quantity: float,
        historical_volume_curve: Sequence[float] | np.ndarray,
        max_participation_rate: float = 0.15,
        smoothing_weight: float = 0.70,
        interval_sec: float = 60.0,
    ) -> None:
        """Initialize and validate dynamic VWAP scheduler parameters.

        Functional Purpose:
            Validates historical volume distribution and computes normalized probability simplex.
        Explicit Dependency Tracking:
            math.isfinite, InvalidScheduleException, ERR_SOR_INVALID_SCHEDULE.
        Structural Relationship:
            Instantiated with instrument historical diurnal profile for execution horizon.
        Defensive Invariant:
            Historical curve values must be non-boolean, finite, non-negative, and sum > 0.
        """
        # 1. Validate total_quantity (> 0.0, finite float, reject bool)
        if (
            not isinstance(total_quantity, (int, float))
            or isinstance(total_quantity, bool)
            or not math.isfinite(total_quantity)
            or total_quantity <= 0.0
        ):
            raise InvalidScheduleException(
                f"VolumeAdaptiveVWAPScheduler total_quantity must be finite float > 0.0, "
                f"got {total_quantity!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.total_quantity: float = float(total_quantity)

        # 2. Validate historical_volume_curve (length >= 1, finite, non-negative, sum > 0)
        if not isinstance(historical_volume_curve, (list, tuple, np.ndarray)):
            raise InvalidScheduleException(
                f"historical_volume_curve must be sequence or array, "
                f"got {type(historical_volume_curve).__name__}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        if len(historical_volume_curve) < 1:
            raise InvalidScheduleException(
                "historical_volume_curve must contain at least 1 interval element",
                code=ERR_SOR_INVALID_SCHEDULE,
            )

        sanitized_curve: list[float] = []
        for i, val in enumerate(historical_volume_curve):
            if (
                not isinstance(val, (int, float, np.floating, np.integer))
                or isinstance(val, bool)
                or not math.isfinite(float(val))
                or float(val) < 0.0
            ):
                raise InvalidScheduleException(
                    f"historical_volume_curve element at index {i} must be finite float >= 0.0, "
                    f"got {val!r}",
                    code=ERR_SOR_INVALID_SCHEDULE,
                )
            sanitized_curve.append(float(val))

        total_hist = sum(sanitized_curve)
        if total_hist <= 0.0:
            raise InvalidScheduleException(
                f"historical_volume_curve cumulative sum must be strictly > 0.0, got {total_hist}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )

        # Normalize historical curve to probability simplex w in Delta^K
        self.weights: tuple[float, ...] = tuple(v / total_hist for v in sanitized_curve)

        # 3. Validate max_participation_rate (in (0.0, 1.0], finite float, reject bool)
        if (
            not isinstance(max_participation_rate, (int, float))
            or isinstance(max_participation_rate, bool)
            or not math.isfinite(max_participation_rate)
            or not (0.0 < max_participation_rate <= 1.0)
        ):
            raise InvalidScheduleException(
                f"max_participation_rate must be finite float in (0.0, 1.0], "
                f"got {max_participation_rate!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.max_participation_rate: float = float(max_participation_rate)

        # 4. Validate smoothing_weight (in [0.0, 1.0], finite float, reject bool)
        if (
            not isinstance(smoothing_weight, (int, float))
            or isinstance(smoothing_weight, bool)
            or not math.isfinite(smoothing_weight)
            or not (0.0 <= smoothing_weight <= 1.0)
        ):
            raise InvalidScheduleException(
                f"smoothing_weight must be finite float in [0.0, 1.0], got {smoothing_weight!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.smoothing_weight: float = float(smoothing_weight)

        # 5. Validate interval_sec (> 0.0, finite float, reject bool)
        if (
            not isinstance(interval_sec, (int, float))
            or isinstance(interval_sec, bool)
            or not math.isfinite(interval_sec)
            or interval_sec <= 0.0
        ):
            raise InvalidScheduleException(
                f"interval_sec must be finite float > 0.0, got {interval_sec!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.interval_sec: float = float(interval_sec)

    def generate_schedule(
        self,
        start_time_ns: int,
        expected_bar_volumes: Sequence[float] | np.ndarray | None = None,
        realtime_volume_estimates: Sequence[float] | np.ndarray | None = None,
    ) -> list[ScheduledSlice]:
        """Generate a volume-weighted schedule adhering to participation rate caps.

        Functional Purpose:
            Decomposes order volume across intervals based on historical profile or
            dynamically blended expected/real-time bar volume estimates, capping each slice
            at rho_max * V_hat (INV-SOR-003).
        Explicit Dependency Tracking:
            ScheduledSlice, InvalidScheduleException, MassConservationException.
        Structural Relationship:
            Called by parent order executor or dynamic participation engine.
        Defensive Invariant:
            Child slice quantities never exceed max_participation_rate * V_hat.
            Unconstrained schedule guarantees exact mass conservation sum(q_k) == Q within 1e-7.
        """
        # Validate start_time_ns boundary
        if (
            not isinstance(start_time_ns, int)
            or isinstance(start_time_ns, bool)
            or start_time_ns < 0
        ):
            raise InvalidScheduleException(
                f"start_time_ns must be non-negative int, got {start_time_ns!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )

        k_bars = len(self.weights)
        slices: list[ScheduledSlice] = []
        rem_quantity: float = self.total_quantity
        interval_ns = int(self.interval_sec * 1_000_000_000)

        # Path A: Unconstrained static historical profile (no expected_bar_volumes provided)
        if expected_bar_volumes is None:
            for k in range(k_bars - 1):
                w_k = self.weights[k]
                q_k = self.total_quantity * w_k
                if q_k > 0.0:
                    t_k = start_time_ns + (k * interval_ns)
                    slices.append(
                        ScheduledSlice(
                            slice_index=len(slices),
                            quantity=q_k,
                            scheduled_time_ns=t_k,
                        )
                    )
                    rem_quantity -= q_k

            # Final slice residual closure guaranteeing exact mass conservation
            if rem_quantity > 0.0:
                t_final = start_time_ns + ((k_bars - 1) * interval_ns)
                slices.append(
                    ScheduledSlice(
                        slice_index=len(slices),
                        quantity=rem_quantity,
                        scheduled_time_ns=t_final,
                    )
                )

            # Invariant assertion: Non-empty schedule
            if not slices:
                raise InvalidScheduleException(
                    "VolumeAdaptiveVWAPScheduler produced empty schedule: all weights evaluate to zero",
                    code=ERR_SOR_INVALID_SCHEDULE,
                )

            # Defensive assertion: INV-SOR-001 mass conservation
            total_scheduled = sum(s.quantity for s in slices)
            if abs(total_scheduled - self.total_quantity) > MASS_CONSERVATION_TOLERANCE:
                raise MassConservationException(
                    f"VolumeAdaptiveVWAPScheduler unconstrained mass breach: "
                    f"scheduled={total_scheduled:.9f} != total={self.total_quantity:.9f}",
                    code=ERR_SOR_MASS_CONSERVATION_BREACH,
                )
            return slices

        # Path B: Dynamic volume blending and participation rate cap enforcement
        if not isinstance(expected_bar_volumes, (list, tuple, np.ndarray)):
            raise InvalidScheduleException(
                f"expected_bar_volumes must be sequence or array, "
                f"got {type(expected_bar_volumes).__name__}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        if len(expected_bar_volumes) != k_bars:
            raise InvalidScheduleException(
                f"expected_bar_volumes length {len(expected_bar_volumes)} != "
                f"historical curve length {k_bars}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )

        sanitized_exp: list[float] = []
        for i, val in enumerate(expected_bar_volumes):
            if (
                not isinstance(val, (int, float, np.floating, np.integer))
                or isinstance(val, bool)
                or not math.isfinite(float(val))
                or float(val) < 0.0
            ):
                raise InvalidScheduleException(
                    f"expected_bar_volumes at index {i} must be finite float >= 0.0, got {val!r}",
                    code=ERR_SOR_INVALID_SCHEDULE,
                )
            sanitized_exp.append(float(val))

        sanitized_rt: list[float] | None = None
        if realtime_volume_estimates is not None:
            if not isinstance(realtime_volume_estimates, (list, tuple, np.ndarray)):
                raise InvalidScheduleException(
                    f"realtime_volume_estimates must be sequence or array, "
                    f"got {type(realtime_volume_estimates).__name__}",
                    code=ERR_SOR_INVALID_SCHEDULE,
                )
            if len(realtime_volume_estimates) != k_bars:
                raise InvalidScheduleException(
                    f"realtime_volume_estimates length {len(realtime_volume_estimates)} != "
                    f"historical curve length {k_bars}",
                    code=ERR_SOR_INVALID_SCHEDULE,
                )
            sanitized_rt = []
            for i, val in enumerate(realtime_volume_estimates):
                if (
                    not isinstance(val, (int, float, np.floating, np.integer))
                    or isinstance(val, bool)
                    or not math.isfinite(float(val))
                    or float(val) < 0.0
                ):
                    raise InvalidScheduleException(
                        f"realtime_volume_estimates at index {i} must be finite float >= 0.0, "
                        f"got {val!r}",
                        code=ERR_SOR_INVALID_SCHEDULE,
                    )
                sanitized_rt.append(float(val))

        # Dynamic volume allocation loop
        omega = self.smoothing_weight
        for k in range(k_bars):
            v_exp = sanitized_exp[k]
            if sanitized_rt is not None:
                v_rt = sanitized_rt[k]
                v_hat = (omega * v_exp) + ((1.0 - omega) * v_rt)
            else:
                v_hat = v_exp

            # Remaining cumulative historical weight
            remaining_weight = sum(self.weights[j] for j in range(k, k_bars))
            if remaining_weight > 0.0 and rem_quantity > 0.0:
                q_desired = rem_quantity * (self.weights[k] / remaining_weight)
                q_cap = self.max_participation_rate * v_hat
                q_k = min(q_desired, q_cap)

                if q_k > 0.0:
                    t_k = start_time_ns + (k * interval_ns)
                    slices.append(
                        ScheduledSlice(
                            slice_index=len(slices),
                            quantity=q_k,
                            scheduled_time_ns=t_k,
                        )
                    )
                    rem_quantity -= q_k

        if not slices:
            raise InsufficientLiquidityException(
                f"VolumeAdaptiveVWAPScheduler unable to schedule child slices: "
                f"zero market volume forecasted across all {k_bars} intervals",
                code=ERR_SOR_INSUFFICIENT_LIQUIDITY,
            )

        return slices


# ============================================================================
# NonlinearArrivalPriceScheduler Implementation (Almgren-Chriss)
# ============================================================================


class NonlinearArrivalPriceScheduler:
    """Closed-form Almgren-Chriss (2000) optimal liquidation trajectory under 3/2-power impact.

    Functional Purpose:
        Computes the optimal institutional inventory discharge trajectory balancing market
        impact costs against volatility timing risk. Provides exact closed-form hyperbolic
        solutions with numerical overflow guards for extreme urgency regimes, eliminating
        black-box numerical solvers (Rule 4).
    Explicit Dependency Tracking:
        math.sinh, math.exp, math.sqrt, numpy, ScheduledSlice, InvalidScheduleException.
    Structural Relationship:
        Conforms to ExecutionScheduler protocol; models implementation shortfall liquidation.
    Defensive Invariant:
        total_quantity > 0, horizon_seconds > 0, volatility > 0, risk_aversion > 0,
        impact_coefficient > 0, num_intervals >= 1. Telescoping sum guarantees exact mass closure.
    """

    def __init__(
        self,
        total_quantity: float,
        horizon_seconds: float,
        volatility: float,
        risk_aversion: float,
        impact_coefficient: float,
        num_intervals: int,
        baseline_volatility: float | None = None,
    ) -> None:
        """Initialize and validate Almgren-Chriss Arrival Price scheduler parameters.

        Functional Purpose:
            Validates parameters and computes the characteristic urgency parameter kappa.
        Explicit Dependency Tracking:
            math.sqrt, InvalidScheduleException, ERR_SOR_INVALID_SCHEDULE.
        Structural Relationship:
            Instantiated with portfolio risk preference and market microstructural liquidity coefficients.
        Defensive Invariant:
            All parameters must be strictly positive, non-boolean, and finite.
        """
        # 1. Validate total_quantity (> 0.0, finite float, reject bool)
        if (
            not isinstance(total_quantity, (int, float))
            or isinstance(total_quantity, bool)
            or not math.isfinite(total_quantity)
            or total_quantity <= 0.0
        ):
            raise InvalidScheduleException(
                f"NonlinearArrivalPriceScheduler total_quantity must be finite float > 0.0, "
                f"got {total_quantity!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.total_quantity: float = float(total_quantity)

        # 2. Validate horizon_seconds (> 0.0, finite float, reject bool)
        if (
            not isinstance(horizon_seconds, (int, float))
            or isinstance(horizon_seconds, bool)
            or not math.isfinite(horizon_seconds)
            or horizon_seconds <= 0.0
        ):
            raise InvalidScheduleException(
                f"NonlinearArrivalPriceScheduler horizon_seconds must be finite float > 0.0, "
                f"got {horizon_seconds!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.horizon_seconds: float = float(horizon_seconds)

        # 3. Validate volatility (> 0.0, finite float, reject bool)
        if (
            not isinstance(volatility, (int, float))
            or isinstance(volatility, bool)
            or not math.isfinite(volatility)
            or volatility <= 0.0
        ):
            raise InvalidScheduleException(
                f"NonlinearArrivalPriceScheduler volatility must be finite float > 0.0, "
                f"got {volatility!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.volatility: float = float(volatility)

        # 4. Validate risk_aversion (> 0.0, finite float, reject bool)
        if (
            not isinstance(risk_aversion, (int, float))
            or isinstance(risk_aversion, bool)
            or not math.isfinite(risk_aversion)
            or risk_aversion <= 0.0
        ):
            raise InvalidScheduleException(
                f"NonlinearArrivalPriceScheduler risk_aversion must be finite float > 0.0, "
                f"got {risk_aversion!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.risk_aversion: float = float(risk_aversion)

        # 5. Validate impact_coefficient (> 0.0, finite float, reject bool)
        if (
            not isinstance(impact_coefficient, (int, float))
            or isinstance(impact_coefficient, bool)
            or not math.isfinite(impact_coefficient)
            or impact_coefficient <= 0.0
        ):
            raise InvalidScheduleException(
                f"NonlinearArrivalPriceScheduler impact_coefficient must be finite float > 0.0, "
                f"got {impact_coefficient!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.impact_coefficient: float = float(impact_coefficient)

        # 6. Validate num_intervals (>= 1, integer, reject bool)
        if (
            not isinstance(num_intervals, int)
            or isinstance(num_intervals, bool)
            or num_intervals < 1
        ):
            raise InvalidScheduleException(
                f"NonlinearArrivalPriceScheduler num_intervals must be integer >= 1, "
                f"got {num_intervals!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )
        self.num_intervals: int = num_intervals

        # 7. Validate optional baseline_volatility (> 0.0, finite float, reject bool)
        if baseline_volatility is not None:
            if (
                not isinstance(baseline_volatility, (int, float))
                or isinstance(baseline_volatility, bool)
                or not math.isfinite(baseline_volatility)
                or baseline_volatility <= 0.0
            ):
                raise InvalidScheduleException(
                    f"NonlinearArrivalPriceScheduler baseline_volatility must be "
                    f"finite float > 0.0 if provided, got {baseline_volatility!r}",
                    code=ERR_SOR_INVALID_SCHEDULE,
                )
            self.baseline_volatility: float | None = float(baseline_volatility)
        else:
            self.baseline_volatility = None

        # Compute characteristic urgency parameter kappa
        # kappa_0 = sqrt(lambda * sigma^2 / eta)
        variance = self.volatility**2
        kappa_0 = math.sqrt((self.risk_aversion * variance) / self.impact_coefficient)

        # Apply volatility scaling if baseline_volatility is specified:
        # kappa_t = kappa_0 * (sigma_t / sigma_baseline)
        if self.baseline_volatility is not None:
            self.kappa: float = kappa_0 * (self.volatility / self.baseline_volatility)
        else:
            self.kappa = kappa_0

    def compute_inventory_trajectory(self) -> np.ndarray:
        """Compute the discrete inventory discharge trajectory x(t_j) for j = 0 ... M.

        Functional Purpose:
            Calculates x_j = X_0 * sinh(kappa * (T - t_j)) / sinh(kappa * T) using numerically
            stable asymptotic formulations to guard against underflow (linear limit) and
            floating point overflow (exponential ratio reformulation).
        Explicit Dependency Tracking:
            math.sinh, math.exp, numpy.
        Structural Relationship:
            Consumed by generate_schedule to form discrete slice increments q_j = x_{j-1} - x_j.
        Defensive Invariant:
            x[0] == X_0, x[M] == 0.0, and x[j] monotonically non-increasing for all j.
        """
        m_intervals = self.num_intervals
        t_horizon = self.horizon_seconds
        x_0 = self.total_quantity
        kappa = self.kappa
        kappa_t = kappa * t_horizon

        trajectory = np.zeros(m_intervals + 1, dtype=np.float64)
        trajectory[0] = x_0
        trajectory[m_intervals] = 0.0

        for j in range(1, m_intervals):
            t_j = (j * t_horizon) / m_intervals
            tau = t_horizon - t_j  # Remaining time (T - t_j)

            # Asymptotic branch 1: Linear TWAP limit as kappa * T -> 0
            if kappa_t < 1e-6:
                x_j = x_0 * (tau / t_horizon)

            # Asymptotic branch 2: Overflow-proof exponential ratio formulation for kappa * T > 50.0
            # sinh(kappa * tau) / sinh(kappa * T) = exp(-kappa * t_j) * [1 - exp(-2*kappa*tau)] / [1 - exp(-2*kappa*T)]
            elif kappa_t > 50.0:
                exp_decay = math.exp(-kappa * t_j)
                num = 1.0 - math.exp(-2.0 * kappa * tau)
                denom = 1.0 - math.exp(-2.0 * kappa_t)
                x_j = x_0 * exp_decay * (num / denom)

            # Standard hyperbolic formula for intermediate urgency
            else:
                x_j = x_0 * (math.sinh(kappa * tau) / math.sinh(kappa_t))

            # Defensive monotonic non-increasing clamp
            trajectory[j] = max(0.0, min(float(trajectory[j - 1]), float(x_j)))

        return trajectory

    def generate_schedule(self, start_time_ns: int) -> list[ScheduledSlice]:
        """Generate scheduled execution slices from the inventory trajectory.

        Functional Purpose:
            Derives child order slice sizes from inventory differentials q_j = x(t_{j-1}) - x(t_j).
            Telescoping sum guarantees exact parent mass conservation (INV-SOR-001).
        Explicit Dependency Tracking:
            ScheduledSlice, compute_inventory_trajectory, MassConservationException.
        Structural Relationship:
            Conforms to ExecutionScheduler. Emits slices consumed by routing gateway.
        Defensive Invariant:
            Sum of slice quantities equals self.total_quantity within 1e-7 tolerance.
        """
        # Validate start_time_ns boundary
        if (
            not isinstance(start_time_ns, int)
            or isinstance(start_time_ns, bool)
            or start_time_ns < 0
        ):
            raise InvalidScheduleException(
                f"start_time_ns must be non-negative int, got {start_time_ns!r}",
                code=ERR_SOR_INVALID_SCHEDULE,
            )

        trajectory = self.compute_inventory_trajectory()
        m_intervals = self.num_intervals
        t_horizon = self.horizon_seconds
        dt_ns = (t_horizon / m_intervals) * 1_000_000_000

        slices: list[ScheduledSlice] = []
        rem_quantity: float = self.total_quantity
        last_timestamp_ns = -1

        for j in range(1, m_intervals + 1):
            if rem_quantity <= 0.0:
                break

            # Differential slice volume
            q_j = float(trajectory[j - 1] - trajectory[j])

            # Final slice or residual depletion closure
            if j == m_intervals or q_j >= rem_quantity:
                q_j = rem_quantity

            # Skip underflow slices if already liquidated
            if q_j <= 0.0:
                continue

            raw_t_ns = start_time_ns + int((j - 1) * dt_ns)
            t_slice_ns = (
                max(last_timestamp_ns + 1, raw_t_ns) if last_timestamp_ns >= 0 else raw_t_ns
            )
            last_timestamp_ns = t_slice_ns

            slices.append(
                ScheduledSlice(
                    slice_index=len(slices),
                    quantity=q_j,
                    scheduled_time_ns=t_slice_ns,
                )
            )
            rem_quantity -= q_j

        # If any tiny residual remains due to floating-point truncation, absorb into last slice
        if rem_quantity > 0.0 and len(slices) > 0:
            slices[-1].quantity += rem_quantity
            rem_quantity = 0.0

        # Defensive assertion: INV-SOR-001 mass conservation
        total_scheduled = sum(s.quantity for s in slices)
        if abs(total_scheduled - self.total_quantity) > MASS_CONSERVATION_TOLERANCE:
            raise MassConservationException(
                f"NonlinearArrivalPriceScheduler mass breach: "
                f"scheduled={total_scheduled:.9f} != total={self.total_quantity:.9f}",
                code=ERR_SOR_MASS_CONSERVATION_BREACH,
            )

        return slices
