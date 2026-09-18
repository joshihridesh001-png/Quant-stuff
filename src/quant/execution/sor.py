"""Smart Order Router (SOR) Engine: Dark Probing, Markout Watchdog & Closed-Form KKT Waterfilling.

Purpose:
    Provides institutional Smart Order Routing (SOR) that decomposes parent or algorithmic
    slices across fragmented execution venues:
    1. Sequential Dark Pool Midpoint Probing: Route to non-displayed alternative trading systems
       (ATS) at the exact NBBO midpoint to capture half-spread price improvement with zero
       market footprint.
    2. Toxic Markout Watchdog: Tracks post-trade adverse selection in basis points across venues
       and automatically quarantines venues exhibiting persistent toxic price decay.
    3. Closed-Form Algebraic KKT Lit Waterfilling: Allocates residual slices across lit exchanges
       in O(M log M) closed-form complexity, minimizing total transaction and spread costs without
       iterative numerical optimization, strictly adhering to the sub-0.05ms latency SLA.

Dependencies:
    - dataclasses: High-performance memory-efficient slotted dataclasses.
    - math: Finite scalar arithmetic and non-finite tripwires (math.isfinite).
    - uuid: Unique client order identifier generation.
    - typing: Static typing annotations and Final constants.
    - quant.execution.gateway: ExecutionGateway protocol.
    - quant.execution.models: Order, ExecutionReport, OrderSide, OrderState, OrderType, TimeInForce.
    - quant.execution.venues: ConsolidatedQuote, VenueProfile, VenueType, and SORError hierarchy.

Structural Relationship:
    - Ingested by:
        1. ParentOrder coordinator (parent_order.py)
        2. Execution Strategy Orchestrator
    - Consumes: VenueProfile and ConsolidatedQuote from quant.execution.venues.
    - Dispatches to: ExecutionGateway implementations (e.g. PaperExecutionGateway).
    - Emits: RoutedVenueOrder domain specifications and ExecutionReport collections.

Invariants Enforced:
    - INV-SOR-001 (Mass Conservation): Slices sum to target quantity within 1e-7 tolerance.
    - INV-SOR-002 (NBBO & Dark Midpoint Price Improvement): Dark probing executes at exact
      (P_bid + P_ask) / 2; lit orders never route worse than NBBO.
    - INV-SOR-005 (Strict Non-Finite Input Protection): Complete rejection of NaN, Inf, bool,
      and non-positive lot sizes or prices.
    - INV-SOR-006 (Hot-Path Latency SLA): Multi-venue routing allocation executes in < 0.05ms (50us).
    - Rule 1: Line-by-line annotation standards (Functional Purpose, Explicit Dependency Tracking,
      Structural Relationship, Defensive Invariant).
    - Rule 2: Zero-execution diagnostic fault codes (ERR-SOR-002, ERR-SOR-003, ERR-SOR-005, ERR-SOR-007).
    - Rule 4: Mandatory adversarial red-teaming and zero numerical iterative solvers.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Final

from quant.execution.gateway import ExecutionGateway
from quant.execution.models import (
    ExecutionReport,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)
from quant.execution.venues import (
    ERR_SOR_CHILD_ORDER_FAILED,
    ERR_SOR_INSUFFICIENT_LIQUIDITY,
    ERR_SOR_NBBO_VIOLATION,
    ERR_SOR_NON_FINITE_INPUT,
    ChildOrderFailedException,
    ConsolidatedQuote,
    InsufficientLiquidityException,
    InvalidSORInputException,
    NBBOViolationException,
    VenueProfile,
    VenueType,
)

# Residual tolerance for floating-point mass conservation and zero-depth checks
ROUTING_EPSILON: Final[float] = 1e-7


# ============================================================================
# Domain Value Objects: RoutedVenueOrder
# ============================================================================


@dataclass(slots=True)
class RoutedVenueOrder:
    """Atomic routing instruction destined for an individual execution venue.

    Attributes:
        venue_id: Target execution venue identifier (e.g. 'NASDAQ', 'DARK_ATS').
        venue_type: Operational classification of target venue (LIT_EXCHANGE or DARK_POOL).
        symbol: Market ticker or instrument identifier.
        side: Order trading direction (BUY or SELL).
        quantity: Share or contract volume allocated to this venue (finite, > 0.0).
        price: Execution limit price ceiling/floor (finite, > 0.0).
        time_in_force: Execution lifespan policy (default: TimeInForce.IOC).
        order_type: Order matching instruction (default: OrderType.LIMIT).
        is_dark: Boolean flag denoting whether this order is routed to non-displayed liquidity.
    """

    venue_id: str
    venue_type: VenueType
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    time_in_force: TimeInForce = TimeInForce.IOC
    order_type: OrderType = OrderType.LIMIT
    is_dark: bool = False

    def __post_init__(self) -> None:
        """Validate boundary bounds, scalar types, and non-finite tripwires upon instantiation.

        Functional Purpose:
            Enforce strict scalar typing, reject boolean numeric masks, verify positive
            quantities and prices, and prevent corrupt routing instructions from dispatching.
        Explicit Dependency Tracking:
            math.isfinite, InvalidSORInputException, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Executed immediately on every RoutedVenueOrder construction within the router.
        Defensive Invariant:
            venue_id and symbol non-empty; quantity > 0.0 finite float; price > 0.0 finite float;
            valid enum types.
        """
        # 1. Validate venue_id
        if not isinstance(self.venue_id, str) or not self.venue_id.strip():
            raise InvalidSORInputException(
                f"RoutedVenueOrder venue_id must be non-empty str, got {self.venue_id!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 2. Validate venue_type
        if not isinstance(self.venue_type, VenueType):
            raise InvalidSORInputException(
                f"RoutedVenueOrder venue_type must be VenueType enum, got {self.venue_type!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 3. Validate symbol
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise InvalidSORInputException(
                f"RoutedVenueOrder symbol must be non-empty str, got {self.symbol!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 4. Validate side
        if not isinstance(self.side, OrderSide):
            raise InvalidSORInputException(
                f"RoutedVenueOrder side must be OrderSide enum, got {self.side!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 5. Validate quantity (> 0.0, finite float, reject bool)
        if (
            not isinstance(self.quantity, (int, float))
            or isinstance(self.quantity, bool)
            or not math.isfinite(self.quantity)
            or self.quantity <= 0.0
        ):
            raise InvalidSORInputException(
                f"RoutedVenueOrder quantity must be finite float > 0.0, got {self.quantity!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.quantity = float(self.quantity)

        # 6. Validate price (> 0.0, finite float, reject bool)
        if (
            not isinstance(self.price, (int, float))
            or isinstance(self.price, bool)
            or not math.isfinite(self.price)
            or self.price <= 0.0
        ):
            raise InvalidSORInputException(
                f"RoutedVenueOrder price must be finite float > 0.0, got {self.price!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.price = float(self.price)

        # 7. Validate time_in_force
        if not isinstance(self.time_in_force, TimeInForce):
            raise InvalidSORInputException(
                f"RoutedVenueOrder time_in_force must be TimeInForce enum, got {self.time_in_force!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 8. Validate order_type
        if not isinstance(self.order_type, OrderType):
            raise InvalidSORInputException(
                f"RoutedVenueOrder order_type must be OrderType enum, got {self.order_type!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 9. Validate is_dark
        if not isinstance(self.is_dark, bool):
            raise InvalidSORInputException(
                f"RoutedVenueOrder is_dark must be bool, got {type(self.is_dark).__name__}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )


# ============================================================================
# Toxic Markout Watchdog: VenueHealth
# ============================================================================


@dataclass(slots=True)
class VenueHealth:
    """Microstructural venue quality and adverse selection monitoring state.

    Attributes:
        venue_id: Identifier of the monitored execution venue.
        is_quarantined: Whether the venue is actively quarantined due to adverse selection.
        total_fills: Cumulative number of execution fills recorded on this venue.
        cumulative_markout_bps: Cumulative post-fill markout sum in basis points.
        consecutive_toxic_fills: Current streak of consecutive fills below toxic threshold.
        quarantine_until_ns: Epoch timestamp in nanoseconds until which quarantine remains active.
    """

    venue_id: str
    is_quarantined: bool = False
    total_fills: int = 0
    cumulative_markout_bps: float = 0.0
    consecutive_toxic_fills: int = 0
    quarantine_until_ns: int = 0

    def __post_init__(self) -> None:
        """Validate venue health initialization invariants.

        Functional Purpose:
            Verify non-empty identifier and valid non-negative metrics on initialization.
        Explicit Dependency Tracking:
            math.isfinite, InvalidSORInputException, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Maintained per-venue inside SmartOrderRouter.
        Defensive Invariant:
            venue_id non-empty str; counts and timestamps non-negative non-boolean integers.
        """
        if not isinstance(self.venue_id, str) or not self.venue_id.strip():
            raise InvalidSORInputException(
                f"VenueHealth venue_id must be non-empty str, got {self.venue_id!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(self.is_quarantined, bool):
            raise InvalidSORInputException(
                f"VenueHealth is_quarantined must be bool, got {self.is_quarantined!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(self.total_fills, int)
            or isinstance(self.total_fills, bool)
            or self.total_fills < 0
        ):
            raise InvalidSORInputException(
                f"VenueHealth total_fills must be non-negative int, got {self.total_fills!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(self.cumulative_markout_bps, (int, float))
            or isinstance(self.cumulative_markout_bps, bool)
            or not math.isfinite(self.cumulative_markout_bps)
        ):
            raise InvalidSORInputException(
                f"VenueHealth cumulative_markout_bps must be finite float, got {self.cumulative_markout_bps!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self.cumulative_markout_bps = float(self.cumulative_markout_bps)

        if (
            not isinstance(self.consecutive_toxic_fills, int)
            or isinstance(self.consecutive_toxic_fills, bool)
            or self.consecutive_toxic_fills < 0
        ):
            raise InvalidSORInputException(
                f"VenueHealth consecutive_toxic_fills must be non-negative int, got {self.consecutive_toxic_fills!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(self.quarantine_until_ns, int)
            or isinstance(self.quarantine_until_ns, bool)
            or self.quarantine_until_ns < 0
        ):
            raise InvalidSORInputException(
                f"VenueHealth quarantine_until_ns must be non-negative int, got {self.quarantine_until_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

    @property
    def average_markout_bps(self) -> float:
        """Calculate historical volume/fill-weighted average markout in basis points.

        Functional Purpose:
            Provide overall metric of venue adverse selection or price improvement quality.
        Explicit Dependency Tracking:
            self.cumulative_markout_bps, self.total_fills.
        Structural Relationship:
            Exposed for execution analytics and routing venue ranking.
        Defensive Invariant:
            Returns 0.0 when total_fills == 0; always finite float.
        """
        if self.total_fills == 0:
            return 0.0
        return self.cumulative_markout_bps / self.total_fills

    def is_active(self, now_ns: int = 0) -> bool:
        """Check whether venue is active for routing, handling automatic quarantine expiration.

        Args:
            now_ns: Current epoch timestamp in nanoseconds (default: 0).

        Returns:
            True if venue is unquarantined or quarantine has expired; False if actively quarantined.

        Functional Purpose:
            Enforce dynamic quarantine timeout and auto-restoration of quarantined venues
            after the quarantine duration horizon elapses without manual intervention.
        Explicit Dependency Tracking:
            self.is_quarantined, self.quarantine_until_ns.
        Structural Relationship:
            Queried by SmartOrderRouter before allocating slices to venues.
        Defensive Invariant:
            Auto-restores is_quarantined to False and resets consecutive toxic fills when expired.
        """
        if self.is_quarantined:
            if self.quarantine_until_ns > 0 and now_ns >= self.quarantine_until_ns:
                # Quarantine horizon elapsed: auto-restore venue to active rotation
                self.is_quarantined = False
                self.consecutive_toxic_fills = 0
                return True
            return False
        return True

    def record_fill(
        self,
        markout_bps: float,
        now_ns: int,
        toxic_threshold_bps: float = -2.0,
        toxic_limit: int = 3,
        quarantine_duration_ns: int = 60_000_000_000,
    ) -> None:
        """Update venue health metrics with newly realized fill markout and manage quarantine status.

        Args:
            markout_bps: Realized post-trade markout in basis points (negative indicates adverse selection).
            now_ns: Current epoch timestamp in nanoseconds.
            toxic_threshold_bps: Markout floor below which a fill is classified as toxic (e.g. -2.0 bps).
            toxic_limit: Number of consecutive toxic fills required to trigger quarantine.
            quarantine_duration_ns: Duration in nanoseconds to quarantine the venue (e.g. 60s).

        Functional Purpose:
            Detect adverse selection on dark pools and lit venues, incrementing consecutive
            toxic streak or resetting upon healthy fills, and imposing quarantine upon threshold breach.
        Explicit Dependency Tracking:
            self.total_fills, self.cumulative_markout_bps, self.consecutive_toxic_fills.
        Structural Relationship:
            Invoked by SmartOrderRouter.record_fill_markout upon receiving execution reports.
        Defensive Invariant:
            Non-finite or boolean values rejected; toxic fills increment counter; non-toxic fills reset counter.
        """
        if (
            not isinstance(markout_bps, (int, float))
            or isinstance(markout_bps, bool)
            or not math.isfinite(markout_bps)
        ):
            raise InvalidSORInputException(
                f"VenueHealth record_fill markout_bps must be finite float, got {markout_bps!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(now_ns, int) or isinstance(now_ns, bool) or now_ns < 0:
            raise InvalidSORInputException(
                f"VenueHealth record_fill now_ns must be non-negative int, got {now_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(toxic_threshold_bps, (int, float))
            or isinstance(toxic_threshold_bps, bool)
            or not math.isfinite(toxic_threshold_bps)
        ):
            raise InvalidSORInputException(
                f"VenueHealth record_fill toxic_threshold_bps must be finite float, got {toxic_threshold_bps!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(toxic_limit, int) or isinstance(toxic_limit, bool) or toxic_limit < 1:
            raise InvalidSORInputException(
                f"VenueHealth record_fill toxic_limit must be int >= 1, got {toxic_limit!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(quarantine_duration_ns, int)
            or isinstance(quarantine_duration_ns, bool)
            or quarantine_duration_ns < 0
        ):
            raise InvalidSORInputException(
                f"VenueHealth record_fill quarantine_duration_ns must be non-negative int, got {quarantine_duration_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # Check for auto-recovery if already quarantined
        if (
            self.is_quarantined
            and self.quarantine_until_ns > 0
            and now_ns >= self.quarantine_until_ns
        ):
            self.is_quarantined = False
            self.consecutive_toxic_fills = 0

        self.total_fills += 1
        self.cumulative_markout_bps += float(markout_bps)

        if markout_bps < toxic_threshold_bps:
            self.consecutive_toxic_fills += 1
            if self.consecutive_toxic_fills >= toxic_limit:
                self.is_quarantined = True
                self.quarantine_until_ns = max(
                    self.quarantine_until_ns, now_ns + quarantine_duration_ns
                )
        else:
            # Healthy non-toxic fill resets the consecutive streak
            self.consecutive_toxic_fills = 0


# ============================================================================
# Core Engine: SmartOrderRouter
# ============================================================================


class SmartOrderRouter:
    """Microstructural Smart Order Router with sequential dark probing and KKT lit waterfilling.

    Attributes:
        venues: Mapping of venue identifier to VenueProfile configurations.
        toxic_markout_threshold_bps: Adverse selection markout threshold in bps (default: -2.0).
        toxic_fill_limit: Consecutive toxic fill threshold to quarantine a venue (default: 3).
        quarantine_duration_sec: Horizon in seconds for which toxic venue is quarantined (default: 60.0).
    """

    def __init__(
        self,
        venues: dict[str, VenueProfile],
        toxic_markout_threshold_bps: float = -2.0,
        toxic_fill_limit: int = 3,
        quarantine_duration_sec: float = 60.0,
    ) -> None:
        """Initialize the Smart Order Router with configured venue profiles and watchdog parameters.

        Args:
            venues: Non-empty mapping of venue identifier to VenueProfile entities.
            toxic_markout_threshold_bps: Negative bps threshold triggering toxic fill warning.
            toxic_fill_limit: Consecutive toxic fills triggering venue quarantine.
            quarantine_duration_sec: Time in seconds for quarantine to remain active.

        Functional Purpose:
            Validate all venue profiles, establish venue health tracking containers, and configure
            adverse selection parameters. Pre-caches dark and lit venue subsets for ultra-low latency routing.
        Explicit Dependency Tracking:
            VenueProfile, VenueHealth, InvalidSORInputException, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Instantiated at execution engine startup; coordinates order routing and markout tracking.
        Defensive Invariant:
            venues must be non-empty dict of valid VenueProfile instances; parameters strictly finite.
        """
        # 1. Validate venues dict
        if not isinstance(venues, dict) or not venues:
            raise InvalidSORInputException(
                "SmartOrderRouter venues must be a non-empty dictionary",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        sanitized_venues: dict[str, VenueProfile] = {}
        for v_id, profile in venues.items():
            if not isinstance(v_id, str) or not v_id.strip():
                raise InvalidSORInputException(
                    f"Venue key must be non-empty str, got {v_id!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            if not isinstance(profile, VenueProfile):
                raise InvalidSORInputException(
                    f"Venue value for {v_id!r} must be VenueProfile instance, got {profile!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            sanitized_venues[v_id] = profile

        # 2. Validate toxic_markout_threshold_bps
        if (
            not isinstance(toxic_markout_threshold_bps, (int, float))
            or isinstance(toxic_markout_threshold_bps, bool)
            or not math.isfinite(toxic_markout_threshold_bps)
        ):
            raise InvalidSORInputException(
                f"toxic_markout_threshold_bps must be finite float, got {toxic_markout_threshold_bps!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 3. Validate toxic_fill_limit
        if (
            not isinstance(toxic_fill_limit, int)
            or isinstance(toxic_fill_limit, bool)
            or toxic_fill_limit < 1
        ):
            raise InvalidSORInputException(
                f"toxic_fill_limit must be int >= 1, got {toxic_fill_limit!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 4. Validate quarantine_duration_sec
        if (
            not isinstance(quarantine_duration_sec, (int, float))
            or isinstance(quarantine_duration_sec, bool)
            or not math.isfinite(quarantine_duration_sec)
            or quarantine_duration_sec < 0.0
        ):
            raise InvalidSORInputException(
                f"quarantine_duration_sec must be finite float >= 0.0, got {quarantine_duration_sec!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        self.venues: dict[str, VenueProfile] = sanitized_venues
        self._toxic_markout_threshold_bps: float = float(toxic_markout_threshold_bps)
        self._toxic_fill_limit: int = toxic_fill_limit
        self._quarantine_duration_sec: float = float(quarantine_duration_sec)

        # Pre-cache dark and lit collections for hot-path sub-0.05ms execution
        self._dark_venues: tuple[VenueProfile, ...] = tuple(
            v for v in sanitized_venues.values() if v.venue_type == VenueType.DARK_POOL
        )
        self._lit_venues: tuple[VenueProfile, ...] = tuple(
            v for v in sanitized_venues.values() if v.venue_type == VenueType.LIT_EXCHANGE
        )

        # Initialize health tracker for every registered venue
        self._venue_health: dict[str, VenueHealth] = {
            vid: VenueHealth(venue_id=vid) for vid in sanitized_venues
        }

    def compute_routing_allocation(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        quote: ConsolidatedQuote,
        enable_dark: bool = True,
        allow_partial: bool = False,
        now_ns: int = 0,
    ) -> dict[str, RoutedVenueOrder]:
        """Compute optimal venue allocation using sequential dark probing and closed-form KKT lit waterfilling.

        Args:
            symbol: Target market instrument ticker.
            side: Trading direction (BUY or SELL).
            quantity: Total slice volume to route (finite float > 0.0).
            quote: Consolidated top-of-book NBBO quote with multi-venue depth mappings.
            enable_dark: Whether to probe eligible dark pools at the midpoint before lit routing.
            allow_partial: Whether to return partial allocation if total lit depth is insufficient.
            now_ns: Current epoch timestamp in nanoseconds for quarantine evaluation.

        Returns:
            Dictionary mapping venue_id -> RoutedVenueOrder.

        Raises:
            NBBOViolationException: If quote market is locked or crossed (bid >= ask) (ERR-SOR-003).
            InsufficientLiquidityException: If lit depth < quantity and allow_partial=False (ERR-SOR-002).
            InvalidSORInputException: If inputs violate finiteness, types, or bounds (ERR-SOR-007).

        Functional Purpose:
            Executes two-phase routing:
            Phase 1: Probes dark pool with IOC limit order at quote.midpoint to capture half-spread.
            Phase 2: Closed-form algebraic KKT waterfilling sorting lit venues by taker fee,
            filling up to available top-of-book depth, and spilling over to next cheapest venue.
        Explicit Dependency Tracking:
            self.venues, self._venue_health, ConsolidatedQuote, RoutedVenueOrder.
        Structural Relationship:
            Core allocation engine called by route_slice, ParentOrder, and execution algorithms.
        Defensive Invariant:
            INV-SOR-002: Routed lit prices equal NBBO; dark probing equals exact midpoint.
            INV-SOR-006: Pure closed-form algebraic sorting in O(M log M) executing in < 0.05ms.
        """
        # 1. Input parameter validation
        if not isinstance(symbol, str) or not symbol.strip():
            raise InvalidSORInputException(
                f"symbol must be non-empty str, got {symbol!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(quote, ConsolidatedQuote):
            raise InvalidSORInputException(
                f"quote must be ConsolidatedQuote instance, got {quote!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if symbol.strip() != quote.symbol:
            raise InvalidSORInputException(
                f"symbol {symbol!r} does not match quote.symbol {quote.symbol!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(side, OrderSide):
            raise InvalidSORInputException(
                f"side must be OrderSide enum instance, got {side!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(quantity, (int, float))
            or isinstance(quantity, bool)
            or not math.isfinite(quantity)
            or quantity <= 0.0
        ):
            raise InvalidSORInputException(
                f"quantity must be finite float > 0.0, got {quantity!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        flt_quantity = float(quantity)

        if not isinstance(enable_dark, bool):
            raise InvalidSORInputException(
                f"enable_dark must be bool, got {enable_dark!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(allow_partial, bool):
            raise InvalidSORInputException(
                f"allow_partial must be bool, got {allow_partial!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(now_ns, int) or isinstance(now_ns, bool) or now_ns < 0:
            raise InvalidSORInputException(
                f"now_ns must be non-negative int, got {now_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 2. Invariant INV-SOR-002: Hard NBBO uncrossed market check (bid < ask)
        if quote.bid_price >= quote.ask_price:
            raise NBBOViolationException(
                f"Locked or crossed market detected for {quote.symbol}: bid={quote.bid_price} >= ask={quote.ask_price}",
                code=ERR_SOR_NBBO_VIOLATION,
            )

        # ====================================================================
        # Phase 1: Sequential Dark Pool Midpoint Probing
        # ====================================================================
        if enable_dark and self._dark_venues:
            eligible_dark_venues: list[VenueProfile] = []
            for v in self._dark_venues:
                health = self._venue_health[v.venue_id]
                if (
                    not health.is_quarantined or health.is_active(now_ns)
                ) and v.min_order_size <= flt_quantity:
                    eligible_dark_venues.append(v)

            if eligible_dark_venues:
                # Rank dark pools: lowest taker fee first, highest fill probability second, lowest latency third
                eligible_dark_venues.sort(
                    key=lambda v: (
                        v.taker_fee_bps,
                        -v.dark_fill_probability,
                        v.avg_latency_ms,
                        v.venue_id,
                    )
                )
                best_dark = eligible_dark_venues[0]
                midpoint_price = quote.midpoint
                dark_order = RoutedVenueOrder(
                    venue_id=best_dark.venue_id,
                    venue_type=VenueType.DARK_POOL,
                    symbol=symbol,
                    side=side,
                    quantity=flt_quantity,
                    price=midpoint_price,
                    time_in_force=TimeInForce.IOC,
                    order_type=OrderType.LIMIT,
                    is_dark=True,
                )
                return {best_dark.venue_id: dark_order}

        # ====================================================================
        # Phase 2: Closed-Form Algebraic KKT Lit Waterfilling
        # ====================================================================
        if not self._lit_venues:
            if not allow_partial:
                raise InsufficientLiquidityException(
                    f"No active lit exchanges available for {symbol}",
                    code=ERR_SOR_INSUFFICIENT_LIQUIDITY,
                )
            return {}

        target_price = quote.ask_price if side == OrderSide.BUY else quote.bid_price
        side_idx = 1 if side == OrderSide.BUY else 0
        depths_map = quote.venue_depths

        # Collect available top-of-book depth
        lit_candidates: list[tuple[VenueProfile, float]] = []
        for v in self._lit_venues:
            health = self._venue_health[v.venue_id]
            if health.is_quarantined and not health.is_active(now_ns):
                continue
            depth_pair = depths_map.get(v.venue_id)
            if depth_pair is not None:
                avail_depth = depth_pair[side_idx]
                if avail_depth > ROUTING_EPSILON:
                    lit_candidates.append((v, avail_depth))

        if not lit_candidates:
            if not allow_partial:
                raise InsufficientLiquidityException(
                    f"Zero available lit depth for {symbol} on {side.value} side",
                    code=ERR_SOR_INSUFFICIENT_LIQUIDITY,
                )
            return {}

        # Sort lit venues by marginal cost:
        # Since all lit venues execute at the identical NBBO top-of-book price,
        # marginal cost ordering reduces strictly to taker_fee_bps ascending.
        # Secondary sorting prioritizes lowest latency and deterministic venue_id.
        lit_candidates.sort(
            key=lambda item: (item[0].taker_fee_bps, item[0].avg_latency_ms, item[0].venue_id)
        )

        allocations: dict[str, RoutedVenueOrder] = {}
        q_rem = flt_quantity

        for v, avail_depth in lit_candidates:
            if q_rem <= ROUTING_EPSILON:
                break
            fill_qty = min(q_rem, avail_depth)
            if fill_qty <= ROUTING_EPSILON:
                continue

            allocations[v.venue_id] = RoutedVenueOrder(
                venue_id=v.venue_id,
                venue_type=VenueType.LIT_EXCHANGE,
                symbol=symbol,
                side=side,
                quantity=fill_qty,
                price=target_price,
                time_in_force=TimeInForce.IOC,
                order_type=OrderType.LIMIT,
                is_dark=False,
            )
            q_rem -= fill_qty

        # Invariant check: liquidity exhaustion
        if q_rem > ROUTING_EPSILON and not allow_partial:
            raise InsufficientLiquidityException(
                f"Insufficient lit liquidity to fill order for {symbol}: requested={flt_quantity}, unfilled={q_rem:.6f}",
                code=ERR_SOR_INSUFFICIENT_LIQUIDITY,
            )

        return allocations

    def record_fill_markout(
        self,
        venue_id: str,
        side: OrderSide,
        fill_price: float,
        post_fill_midpoint: float,
        timestamp_ns: int,
    ) -> None:
        """Compute post-trade markout and record fill outcome for toxic adverse selection tracking.

        Args:
            venue_id: Execution venue identifier where the trade occurred.
            side: Trading direction of the executed order (BUY or SELL).
            fill_price: Exact price at which order was executed (finite, > 0.0).
            post_fill_midpoint: Market NBBO midpoint at horizon tau after execution (finite, > 0.0).
            timestamp_ns: Epoch timestamp in nanoseconds when markout was recorded.

        Functional Purpose:
            Computes adverse selection markout in basis points:
            BUY:  (P_post - P_fill) / P_fill * 1e4
            SELL: (P_fill - P_post) / P_fill * 1e4
            Adverse selection occurs when price drops after BUY or rises after SELL (negative markout).
            Feeds VenueHealth and automatically triggers quarantine if toxic limit is breached.
        Explicit Dependency Tracking:
            self._venue_health, VenueHealth.record_fill, InvalidSORInputException.
        Structural Relationship:
            Called by execution gateway event callbacks or parent order TCA reconcilers.
        Defensive Invariant:
            fill_price > 0.0 and post_fill_midpoint > 0.0 finite floats; venue_id must exist in health map.
        """
        if not isinstance(venue_id, str) or not venue_id.strip():
            raise InvalidSORInputException(
                f"venue_id must be non-empty str, got {venue_id!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if venue_id not in self._venue_health:
            raise InvalidSORInputException(
                f"Unrecognized venue_id {venue_id!r} not in configured venues",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(side, OrderSide):
            raise InvalidSORInputException(
                f"side must be OrderSide enum instance, got {side!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(fill_price, (int, float))
            or isinstance(fill_price, bool)
            or not math.isfinite(fill_price)
            or fill_price <= 0.0
        ):
            raise InvalidSORInputException(
                f"fill_price must be finite float > 0.0, got {fill_price!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if (
            not isinstance(post_fill_midpoint, (int, float))
            or isinstance(post_fill_midpoint, bool)
            or not math.isfinite(post_fill_midpoint)
            or post_fill_midpoint <= 0.0
        ):
            raise InvalidSORInputException(
                f"post_fill_midpoint must be finite float > 0.0, got {post_fill_midpoint!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool) or timestamp_ns < 0:
            raise InvalidSORInputException(
                f"timestamp_ns must be non-negative int, got {timestamp_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # Markout in bps:
        # For BUY: positive if market rose after buy, negative if price fell (adverse selection)
        # For SELL: positive if market fell after sell, negative if price rallied (adverse selection)
        if side == OrderSide.BUY:
            markout_bps = ((post_fill_midpoint - fill_price) / fill_price) * 10_000.0
        else:
            markout_bps = ((fill_price - post_fill_midpoint) / fill_price) * 10_000.0

        health = self._venue_health[venue_id]
        health.record_fill(
            markout_bps=markout_bps,
            now_ns=timestamp_ns,
            toxic_threshold_bps=self._toxic_markout_threshold_bps,
            toxic_limit=self._toxic_fill_limit,
            quarantine_duration_ns=int(self._quarantine_duration_sec * 1_000_000_000),
        )

    def get_venue_health(self, venue_id: str, now_ns: int = 0) -> VenueHealth:
        """Retrieve current health metrics for a venue, evaluating quarantine expiration against now_ns.

        Args:
            venue_id: Identifier of the target venue.
            now_ns: Current epoch timestamp in nanoseconds.

        Returns:
            VenueHealth instance reflecting active/quarantined status and markout statistics.

        Functional Purpose:
            Inspect venue status, triggering auto-recovery if quarantine duration has elapsed.
        Explicit Dependency Tracking:
            self._venue_health, VenueHealth.is_active.
        Structural Relationship:
            Queried during routing decisions and telemetry reporting.
        Defensive Invariant:
            venue_id must exist in configured health map; returns valid VenueHealth entity.
        """
        if not isinstance(venue_id, str) or venue_id not in self._venue_health:
            raise InvalidSORInputException(
                f"Venue {venue_id!r} not found in router health registry",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        health = self._venue_health[venue_id]
        # Trigger auto-recovery check
        health.is_active(now_ns=now_ns)
        return health

    async def route_slice(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        quote: ConsolidatedQuote,
        gateway: ExecutionGateway,
        enable_dark: bool = True,
        now_ns: int = 0,
    ) -> list[ExecutionReport]:
        """Execute complete asynchronous routing orchestration: dark probing -> lit waterfilling.

        Args:
            symbol: Target market ticker.
            side: Trading direction (BUY or SELL).
            quantity: Slice volume to execute.
            quote: Consolidated NBBO quote with available venue depths.
            gateway: Asynchronous execution gateway implementing ExecutionGateway protocol.
            enable_dark: Whether to probe dark pools before lit routing.
            now_ns: Current timestamp in nanoseconds.

        Returns:
            List of ExecutionReport instances resulting from the routed child orders.

        Raises:
            ChildOrderFailedException: If gateway rejects child order or connection is down (ERR-SOR-005).
            InsufficientLiquidityException: If liquidity is inadequate (ERR-SOR-002).
            NBBOViolationException: If quote market is locked or crossed (ERR-SOR-003).

        Functional Purpose:
            Orchestrates sequential execution lifecycle against execution gateway:
            1. Probes eligible dark pool with midpoint limit IOC order.
            2. If fully filled, completes slice execution immediately.
            3. If unfilled or partially filled, cancels resting IOC leaves and runs closed-form
               KKT lit waterfilling across active lit venues for the residual quantity.
        Explicit Dependency Tracking:
            gateway.submit_order, gateway.cancel_order, compute_routing_allocation, Order.
        Structural Relationship:
            Primary asynchronous interface used by ParentOrder coordinator to route each scheduled slice.
        Defensive Invariant:
            INV-SOR-001: Sum of child fills and leaves strictly conserved; gateway errors wrapped.
        """
        # 1. Connection check
        if not gateway.is_connected:
            raise ChildOrderFailedException(
                "Execution gateway is not connected; cannot route slice",
                code=ERR_SOR_CHILD_ORDER_FAILED,
            )

        reports: list[ExecutionReport] = []
        unfilled_quantity = float(quantity)

        # 2. Phase 1: Sequential Dark Probing
        if enable_dark:
            dark_alloc = self.compute_routing_allocation(
                symbol=symbol,
                side=side,
                quantity=unfilled_quantity,
                quote=quote,
                enable_dark=True,
                allow_partial=True,
                now_ns=now_ns,
            )

            # Check if a dark pool was allocated
            if dark_alloc:
                dark_vid, dark_order = next(iter(dark_alloc.items()))
                if dark_order.is_dark:
                    cl_ord_id = f"sor-dark-{dark_vid}-{uuid.uuid4().hex[:8]}"
                    order = Order(
                        cl_ord_id=cl_ord_id,
                        symbol=symbol,
                        side=side,
                        order_type=dark_order.order_type,
                        quantity=dark_order.quantity,
                        price=dark_order.price,
                        time_in_force=dark_order.time_in_force,
                        created_at_ns=now_ns,
                        updated_at_ns=now_ns,
                    )
                    try:
                        rep = await gateway.submit_order(order)
                    except Exception as exc:
                        raise ChildOrderFailedException(
                            f"Dark child order submission failed on {dark_vid}: {exc}",
                            code=ERR_SOR_CHILD_ORDER_FAILED,
                        ) from exc

                    if rep.exec_type == OrderState.REJECTED:
                        raise ChildOrderFailedException(
                            f"Dark child order rejected on {dark_vid}: {rep.text}",
                            code=ERR_SOR_CHILD_ORDER_FAILED,
                        )

                    # Check fill status
                    if rep.exec_type == OrderState.FILLED:
                        # Full match at midpoint in dark pool!
                        reports.append(rep)
                        return reports

                    if rep.cum_quantity > ROUTING_EPSILON:
                        # Partial match in dark pool
                        reports.append(rep)
                        unfilled_quantity -= rep.cum_quantity
                        if rep.exec_type in (OrderState.NEW, OrderState.PARTIALLY_FILLED):
                            # Cancel unfulfilled resting IOC remainder
                            await gateway.cancel_order(cl_ord_id)
                    else:
                        # Zero match in dark pool: cancel if resting
                        if rep.exec_type in (OrderState.NEW, OrderState.PENDING_NEW):
                            await gateway.cancel_order(cl_ord_id)

        # 3. Phase 2: Closed-Form KKT Lit Waterfilling for Residual Quantity
        if unfilled_quantity > ROUTING_EPSILON:
            lit_alloc = self.compute_routing_allocation(
                symbol=symbol,
                side=side,
                quantity=unfilled_quantity,
                quote=quote,
                enable_dark=False,
                allow_partial=False,
                now_ns=now_ns,
            )

            for lit_vid, lit_routed_order in lit_alloc.items():
                cl_ord_id = f"sor-lit-{lit_vid}-{uuid.uuid4().hex[:8]}"
                child_order = Order(
                    cl_ord_id=cl_ord_id,
                    symbol=symbol,
                    side=side,
                    order_type=lit_routed_order.order_type,
                    quantity=lit_routed_order.quantity,
                    price=lit_routed_order.price,
                    time_in_force=lit_routed_order.time_in_force,
                    created_at_ns=now_ns,
                    updated_at_ns=now_ns,
                )
                try:
                    rep = await gateway.submit_order(child_order)
                except Exception as exc:
                    raise ChildOrderFailedException(
                        f"Lit child order failed on {lit_vid}: {exc}",
                        code=ERR_SOR_CHILD_ORDER_FAILED,
                    ) from exc

                if rep.exec_type == OrderState.REJECTED:
                    raise ChildOrderFailedException(
                        f"Lit child order rejected on {lit_vid}: {rep.text}",
                        code=ERR_SOR_CHILD_ORDER_FAILED,
                    )
                reports.append(rep)

        return reports
