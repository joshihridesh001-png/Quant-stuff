"""Execution venue profiles, consolidated quotes, and NBBO invariants for Smart Order Routing.

Purpose:
    Establishes core domain value objects and microstructural entities representing
    execution venues (lit exchanges, dark pools), consolidated market quotes with
    hard NBBO invariants, and diagnostic exception taxonomies for Phase 6 Step 2.

Dependencies:
    - dataclasses: High-performance memory-efficient slotted dataclasses.
    - enum: Python 3.11+ StrEnum for zero-overhead string-compatible enumerations.
    - math: Strict non-finite verification (math.isfinite).
    - typing: Static typing annotations and Final constants.

Structural Relationship:
    - Ingested by:
        1. PoissonTWAPScheduler, VolumeAdaptiveVWAPScheduler, NonlinearArrivalPriceScheduler (algorithms.py)
        2. SmartOrderRouter (sor.py)
        3. ParentOrder and Implementation Shortfall TCA (parent_order.py)
    - Emits: VenueType, VenueProfile, ConsolidatedQuote, and SORError diagnostic hierarchy.

Invariants Enforced:
    - INV-SOR-001 (Mass Conservation): Slices sum to parent target within 1e-7 tolerance.
    - INV-SOR-002 (Non-Worse-Than-NBBO & Dark Midpoint Price Improvement): Uncrossed NBBO required;
      locked or crossed markets (bid >= ask) strictly rejected at instantiation boundary.
    - INV-SOR-005 (Strict Non-Finite Input Protection): Complete rejection of NaN, Inf, bool,
      negative prices, negative quantities, or corrupted depth mappings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_SOR_INVALID_SCHEDULE: Final[str] = "ERR-SOR-001"
ERR_SOR_INSUFFICIENT_LIQUIDITY: Final[str] = "ERR-SOR-002"
ERR_SOR_NBBO_VIOLATION: Final[str] = "ERR-SOR-003"
ERR_SOR_MASS_CONSERVATION_BREACH: Final[str] = "ERR-SOR-004"
ERR_SOR_CHILD_ORDER_FAILED: Final[str] = "ERR-SOR-005"
ERR_SOR_ALGORITHM_TIMEOUT: Final[str] = "ERR-SOR-006"
ERR_SOR_NON_FINITE_INPUT: Final[str] = "ERR-SOR-007"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Exception Protocol)
# ============================================================================


class SORError(Exception):
    """Base exception for all Smart Order Router and microstructural execution errors."""

    def __init__(self, message: str, code: str = "ERR-SOR-000") -> None:
        # Functional Purpose: Initialize base SOR exception with structured diagnostic fault code.
        # Explicit Dependency Tracking: Python Exception root.
        # Structural Relationship: Root of the Smart Order Router exception taxonomy.
        # Defensive Invariant: Diagnostic code must be a non-empty string identifier.
        super().__init__(message)
        self.message: str = message
        self.code: str = code


class NBBOViolationException(SORError):
    """Raised when an NBBO is crossed or locked, or routing price crosses NBBO (INV-SOR-002)."""

    def __init__(self, message: str, code: str = ERR_SOR_NBBO_VIOLATION) -> None:
        # Functional Purpose: Signal crossed/locked market or aggressive execution price breaching NBBO.
        # Explicit Dependency Tracking: ERR_SOR_NBBO_VIOLATION diagnostic fault code.
        # Structural Relationship: Emitted by ConsolidatedQuote.__post_init__ and SmartOrderRouter.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-003.
        super().__init__(message=message, code=code)


class InsufficientLiquidityException(SORError):
    """Raised when available venue liquidity is zero or inadequate to fill slice (INV-SOR-002)."""

    def __init__(self, message: str, code: str = ERR_SOR_INSUFFICIENT_LIQUIDITY) -> None:
        # Functional Purpose: Signal total book depletion or zero depth across configured routing venues.
        # Explicit Dependency Tracking: ERR_SOR_INSUFFICIENT_LIQUIDITY diagnostic fault code.
        # Structural Relationship: Emitted by SmartOrderRouter waterfilling engine during liquidity exhaustion.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-002.
        super().__init__(message=message, code=code)


class InvalidScheduleException(SORError):
    """Raised when an execution schedule configuration is invalid (INV-SOR-004)."""

    def __init__(self, message: str, code: str = ERR_SOR_INVALID_SCHEDULE) -> None:
        # Functional Purpose: Signal non-positive intervals, invalid horizon duration, or malformed scheduler args.
        # Explicit Dependency Tracking: ERR_SOR_INVALID_SCHEDULE diagnostic fault code.
        # Structural Relationship: Emitted by algorithmic schedulers during parameter validation.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-001.
        super().__init__(message=message, code=code)


class MassConservationException(SORError):
    """Raised when child slices deviate from parent order quantity by > 1e-7 (INV-SOR-001)."""

    def __init__(self, message: str, code: str = ERR_SOR_MASS_CONSERVATION_BREACH) -> None:
        # Functional Purpose: Signal leak or creation of order mass during slicing or routing.
        # Explicit Dependency Tracking: ERR_SOR_MASS_CONSERVATION_BREACH diagnostic fault code.
        # Structural Relationship: Emitted by ParentOrder and meta-order schedulers upon mass discrepancy.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-004.
        super().__init__(message=message, code=code)


class ChildOrderFailedException(SORError):
    """Raised when an execution gateway rejects a child order slice (INV-SOR-001)."""

    def __init__(self, message: str, code: str = ERR_SOR_CHILD_ORDER_FAILED) -> None:
        # Functional Purpose: Signal child order rejection by downstream gateway or broker transport.
        # Explicit Dependency Tracking: ERR_SOR_CHILD_ORDER_FAILED diagnostic fault code.
        # Structural Relationship: Emitted by SmartOrderRouter when child slice dispatch fails.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-005.
        super().__init__(message=message, code=code)


class AlgorithmTimeoutException(SORError):
    """Raised when parent order execution horizon expires with unfilled leaves (INV-SOR-005)."""

    def __init__(self, message: str, code: str = ERR_SOR_ALGORITHM_TIMEOUT) -> None:
        # Functional Purpose: Signal expiration of scheduled execution horizon before full completion.
        # Explicit Dependency Tracking: ERR_SOR_ALGORITHM_TIMEOUT diagnostic fault code.
        # Structural Relationship: Emitted by ParentOrder lifecycle coordinator upon horizon expiry.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-006.
        super().__init__(message=message, code=code)


class InvalidSORInputException(SORError):
    """Raised when SOR domain parameters violate domain boundaries, types, or finiteness (INV-SOR-005)."""

    def __init__(self, message: str, code: str = ERR_SOR_NON_FINITE_INPUT) -> None:
        # Functional Purpose: Signal boolean, NaN, infinite, negative, or degenerate numeric inputs.
        # Explicit Dependency Tracking: ERR_SOR_NON_FINITE_INPUT diagnostic fault code.
        # Structural Relationship: Emitted by VenueProfile and ConsolidatedQuote validation routines.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-007.
        super().__init__(message=message, code=code)


class NonFiniteInputException(InvalidSORInputException):
    """Raised specifically when numeric inputs are non-finite, NaN, or boolean."""

    def __init__(self, message: str, code: str = ERR_SOR_NON_FINITE_INPUT) -> None:
        # Functional Purpose: Specialized subtype for non-finite scalar violations.
        # Explicit Dependency Tracking: InvalidSORInputException base.
        # Structural Relationship: Emitted on math.isfinite failure or bool contamination.
        # Defensive Invariant: Diagnostic code defaults to ERR-SOR-007.
        super().__init__(message=message, code=code)


# ============================================================================
# Domain Enums (StrEnum)
# ============================================================================


class VenueType(StrEnum):
    """Execution venue operational classification."""

    LIT_EXCHANGE = "LIT_EXCHANGE"
    DARK_POOL = "DARK_POOL"


# ============================================================================
# Domain Entities and Value Objects (Rule 1 & Rule 4: Best-of-the-Best Contracts)
# ============================================================================


@dataclass(slots=True)
class VenueProfile:
    """Institutional execution venue profile defining fee structure, routing constraints, and latency.

    Attributes:
        venue_id: Unique string identifier for the execution venue (e.g. 'NASDAQ', 'BATS', 'DARK_ATS').
        venue_type: Operational category (LIT_EXCHANGE or DARK_POOL).
        maker_fee_bps: Maker fee in basis points (negative values represent liquidity rebates).
        taker_fee_bps: Taker fee in basis points.
        min_order_size: Minimum permissible child order size in lots (finite, > 0.0).
        lot_size: Round-lot increment for order slicing (finite, > 0.0).
        avg_latency_ms: Expected one-way network/matching latency in milliseconds (finite, >= 0.0).
        dark_fill_probability: Prior baseline probability of midpoint fill in dark pool [0.0, 1.0].
    """

    venue_id: str
    venue_type: VenueType
    maker_fee_bps: float
    taker_fee_bps: float
    min_order_size: float = 1.0
    lot_size: float = 1.0
    avg_latency_ms: float = 1.0
    dark_fill_probability: float = 0.35

    def __post_init__(self) -> None:
        """Validate domain bounds, types, non-finite guards, and fee invariants upon instantiation."""
        # Functional Purpose: Enforce strict scalar typing, non-boolean values, and physical boundaries.
        # Explicit Dependency Tracking: math.isfinite, InvalidSORInputException, ERR_SOR_NON_FINITE_INPUT.
        # Structural Relationship: Executed automatically on every VenueProfile instantiation.
        # Defensive Invariant: Enforce venue_id non-empty, venue_type valid enum, and finite numeric constraints.

        # 1. Validate venue_id identifier
        if not isinstance(self.venue_id, str) or not self.venue_id:
            raise InvalidSORInputException(
                f"VenueProfile venue_id must be a non-empty string, got {self.venue_id!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 2. Validate venue_type enum
        if not isinstance(self.venue_type, VenueType):
            raise InvalidSORInputException(
                f"VenueProfile venue_type must be VenueType enum instance, got {self.venue_type!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 3. Validate maker_fee_bps (can be negative for rebate, must be finite float)
        if (
            not isinstance(self.maker_fee_bps, (int, float))
            or isinstance(self.maker_fee_bps, bool)
            or not math.isfinite(self.maker_fee_bps)
        ):
            raise InvalidSORInputException(
                f"VenueProfile maker_fee_bps must be finite float, got {self.maker_fee_bps!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.maker_fee_bps = float(self.maker_fee_bps)

        # 4. Validate taker_fee_bps (must be finite float)
        if (
            not isinstance(self.taker_fee_bps, (int, float))
            or isinstance(self.taker_fee_bps, bool)
            or not math.isfinite(self.taker_fee_bps)
        ):
            raise InvalidSORInputException(
                f"VenueProfile taker_fee_bps must be finite float, got {self.taker_fee_bps!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.taker_fee_bps = float(self.taker_fee_bps)

        # 5. Validate min_order_size (> 0.0, finite, non-boolean)
        if (
            not isinstance(self.min_order_size, (int, float))
            or isinstance(self.min_order_size, bool)
            or not math.isfinite(self.min_order_size)
            or self.min_order_size <= 0.0
        ):
            raise InvalidSORInputException(
                f"VenueProfile min_order_size must be finite float > 0.0, got {self.min_order_size!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.min_order_size = float(self.min_order_size)

        # 6. Validate lot_size (> 0.0, finite, non-boolean)
        if (
            not isinstance(self.lot_size, (int, float))
            or isinstance(self.lot_size, bool)
            or not math.isfinite(self.lot_size)
            or self.lot_size <= 0.0
        ):
            raise InvalidSORInputException(
                f"VenueProfile lot_size must be finite float > 0.0, got {self.lot_size!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.lot_size = float(self.lot_size)

        # 7. Validate avg_latency_ms (>= 0.0, finite, non-boolean)
        if (
            not isinstance(self.avg_latency_ms, (int, float))
            or isinstance(self.avg_latency_ms, bool)
            or not math.isfinite(self.avg_latency_ms)
            or self.avg_latency_ms < 0.0
        ):
            raise InvalidSORInputException(
                f"VenueProfile avg_latency_ms must be finite float >= 0.0, got {self.avg_latency_ms!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.avg_latency_ms = float(self.avg_latency_ms)

        # 8. Validate dark_fill_probability (in [0.0, 1.0], finite, non-boolean)
        if (
            not isinstance(self.dark_fill_probability, (int, float))
            or isinstance(self.dark_fill_probability, bool)
            or not math.isfinite(self.dark_fill_probability)
            or not (0.0 <= self.dark_fill_probability <= 1.0)
        ):
            raise InvalidSORInputException(
                f"VenueProfile dark_fill_probability must be finite float in [0.0, 1.0], "
                f"got {self.dark_fill_probability!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.dark_fill_probability = float(self.dark_fill_probability)


@dataclass(slots=True)
class ConsolidatedQuote:
    """Consolidated top-of-book NBBO quote with multi-venue depth aggregation.

    Attributes:
        symbol: Market ticker or instrument identifier.
        bid_price: National Best Bid (NBB) price (must be finite, > 0.0).
        bid_quantity: Cumulative or displayed quantity at NBB (must be finite, >= 0.0).
        ask_price: National Best Offer (NBO) price (must be finite, > 0.0).
        ask_quantity: Cumulative or displayed quantity at NBO (must be finite, >= 0.0).
        timestamp_ns: Ingestion epoch timestamp in nanoseconds (>= 0).
        venue_depths: Mapping of venue_id to (bid_depth, ask_depth) available liquidity tuples.
    """

    symbol: str
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    timestamp_ns: int
    venue_depths: dict[str, tuple[float, float]]

    def __post_init__(self) -> None:
        """Validate NBBO invariants (bid < ask), scalar types, non-finite guards, and depth mappings."""
        # Functional Purpose: Enforce INV-SOR-002 (uncrossed market required) and non-finite guards.
        # Explicit Dependency Tracking: math.isfinite, NBBOViolationException, InvalidSORInputException.
        # Structural Relationship: Executed on every quote construction before routing decisions.
        # Defensive Invariant: bid_price < ask_price, strictly positive prices, non-negative quantities/depths.

        # 1. Validate symbol
        if not isinstance(self.symbol, str) or not self.symbol:
            raise InvalidSORInputException(
                f"ConsolidatedQuote symbol must be a non-empty string, got {self.symbol!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 2. Validate timestamp_ns (integer >= 0, reject boolean)
        if (
            not isinstance(self.timestamp_ns, int)
            or isinstance(self.timestamp_ns, bool)
            or self.timestamp_ns < 0
        ):
            raise InvalidSORInputException(
                f"ConsolidatedQuote timestamp_ns must be non-negative integer, got {self.timestamp_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 3. Validate bid_price (> 0.0, finite, non-boolean)
        if (
            not isinstance(self.bid_price, (int, float))
            or isinstance(self.bid_price, bool)
            or not math.isfinite(self.bid_price)
            or self.bid_price <= 0.0
        ):
            raise InvalidSORInputException(
                f"ConsolidatedQuote bid_price must be finite float > 0.0, got {self.bid_price!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.bid_price = float(self.bid_price)

        # 4. Validate ask_price (> 0.0, finite, non-boolean)
        if (
            not isinstance(self.ask_price, (int, float))
            or isinstance(self.ask_price, bool)
            or not math.isfinite(self.ask_price)
            or self.ask_price <= 0.0
        ):
            raise InvalidSORInputException(
                f"ConsolidatedQuote ask_price must be finite float > 0.0, got {self.ask_price!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.ask_price = float(self.ask_price)

        # 5. Validate bid_quantity (>= 0.0, finite, non-boolean)
        if (
            not isinstance(self.bid_quantity, (int, float))
            or isinstance(self.bid_quantity, bool)
            or not math.isfinite(self.bid_quantity)
            or self.bid_quantity < 0.0
        ):
            raise InvalidSORInputException(
                f"ConsolidatedQuote bid_quantity must be finite float >= 0.0, got {self.bid_quantity!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.bid_quantity = float(self.bid_quantity)

        # 6. Validate ask_quantity (>= 0.0, finite, non-boolean)
        if (
            not isinstance(self.ask_quantity, (int, float))
            or isinstance(self.ask_quantity, bool)
            or not math.isfinite(self.ask_quantity)
            or self.ask_quantity < 0.0
        ):
            raise InvalidSORInputException(
                f"ConsolidatedQuote ask_quantity must be finite float >= 0.0, got {self.ask_quantity!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.ask_quantity = float(self.ask_quantity)

        # 7. Validate venue_depths mapping and depth pairs
        if not isinstance(self.venue_depths, dict):
            raise InvalidSORInputException(
                f"ConsolidatedQuote venue_depths must be dict, got {type(self.venue_depths).__name__}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        sanitized_depths: dict[str, tuple[float, float]] = {}
        for venue_id, depths in self.venue_depths.items():
            if not isinstance(venue_id, str) or not venue_id:
                raise InvalidSORInputException(
                    f"Venue ID in venue_depths must be non-empty string, got {venue_id!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            if not isinstance(depths, (tuple, list)) or len(depths) != 2:
                raise InvalidSORInputException(
                    f"Depth entry for venue {venue_id!r} must be 2-tuple (bid_depth, ask_depth), got {depths!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            bid_depth, ask_depth = depths[0], depths[1]
            if (
                not isinstance(bid_depth, (int, float))
                or isinstance(bid_depth, bool)
                or not math.isfinite(bid_depth)
                or bid_depth < 0.0
            ):
                raise InvalidSORInputException(
                    f"Bid depth for venue {venue_id!r} must be finite float >= 0.0, got {bid_depth!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            if (
                not isinstance(ask_depth, (int, float))
                or isinstance(ask_depth, bool)
                or not math.isfinite(ask_depth)
                or ask_depth < 0.0
            ):
                raise InvalidSORInputException(
                    f"Ask depth for venue {venue_id!r} must be finite float >= 0.0, got {ask_depth!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            sanitized_depths[venue_id] = (float(bid_depth), float(ask_depth))
        self.venue_depths = sanitized_depths

        # 8. Invariant INV-SOR-002: Hard NBBO uncrossed market check (bid < ask)
        if self.bid_price >= self.ask_price:
            raise NBBOViolationException(
                f"Locked or crossed market detected for {self.symbol}: "
                f"bid={self.bid_price:.6f} >= ask={self.ask_price:.6f}",
                code=ERR_SOR_NBBO_VIOLATION,
            )

    @property
    def midpoint(self) -> float:
        """Calculate the unweighted midpoint price between NBB and NBO.

        Functional Purpose:
            Provides the fair-value benchmark reference price for midpoint dark pool matching
            and implementation shortfall benchmarks.
        Explicit Dependency Tracking:
            Derived from self.bid_price and self.ask_price.
        Structural Relationship:
            Used by SmartOrderRouter for dark pool pegging and markout computation.
        Defensive Invariant:
            Guaranteed strictly positive because bid_price > 0 and ask_price > 0.
        """
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        """Calculate the absolute bid-ask spread.

        Functional Purpose:
            Quantifies the execution cost penalty incurred when crossing the spread.
        Explicit Dependency Tracking:
            Derived from self.ask_price and self.bid_price.
        Structural Relationship:
            Consumed by KKT waterfilling algorithm to evaluate lit venue transaction costs.
        Defensive Invariant:
            Guaranteed strictly positive (> 0.0) by NBBO uncrossed invariant.
        """
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Calculate the relative spread in basis points normalized by the midpoint.

        Functional Purpose:
            Normalizes spread cost across instruments of disparate nominal share prices.
        Explicit Dependency Tracking:
            Derived from self.spread and self.midpoint.
        Structural Relationship:
            Consumed by TCA reporting and routing cost optimization.
        Defensive Invariant:
            Guaranteed strictly positive (> 0.0) since spread > 0.0 and midpoint > 0.0.
        """
        return (self.spread / self.midpoint) * 10_000.0

    @property
    def order_book_imbalance(self) -> float:
        """Calculate the normalized order book volume imbalance between [-1.0, 1.0].

        Functional Purpose:
            Measures instantaneous directional queue pressure:
            imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty).
        Explicit Dependency Tracking:
            Derived from self.bid_quantity and self.ask_quantity.
        Structural Relationship:
            Fed into dynamic microstructural scheduling models to adjust urgency.
        Defensive Invariant:
            Bounded in [-1.0, 1.0]; returns 0.0 if total quantity is zero.
        """
        total_qty = self.bid_quantity + self.ask_quantity
        if total_qty > 0.0:
            return (self.bid_quantity - self.ask_quantity) / total_qty
        return 0.0

    @property
    def total_bid_depth(self) -> float:
        """Sum cumulative available bid liquidity across all venues.

        Functional Purpose:
            Aggregates total displayed and available bid liquidity across all venues.
        Explicit Dependency Tracking:
            Derived from self.venue_depths values.
        Structural Relationship:
            Consumed by waterfilling solver to establish upper capacity bounds.
        Defensive Invariant:
            Guaranteed finite and >= 0.0.
        """
        return sum(depth[0] for depth in self.venue_depths.values())

    @property
    def total_ask_depth(self) -> float:
        """Sum cumulative available ask liquidity across all venues.

        Functional Purpose:
            Aggregates total displayed and available ask liquidity across all venues.
        Explicit Dependency Tracking:
            Derived from self.venue_depths values.
        Structural Relationship:
            Consumed by waterfilling solver to establish upper capacity bounds.
        Defensive Invariant:
            Guaranteed finite and >= 0.0.
        """
        return sum(depth[1] for depth in self.venue_depths.values())
