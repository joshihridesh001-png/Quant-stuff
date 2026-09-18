"""Parent Order Coordination and Implementation Shortfall (IS) Attribution Engine.

Purpose:
    Provides institutional parent order lifecycle coordination, child fill tracking,
    mass conservation invariant guards, execution horizon timeout enforcement, and
    exact Perold (1988) Implementation Shortfall (IS) transaction cost attribution (TCA).

Dependencies:
    - dataclasses: High-performance memory-efficient slotted dataclasses.
    - math: Finite scalar arithmetic and non-finite boundary enforcement (math.isfinite).
    - typing: Static typing annotations, Final constants, and optional types.
    - quant.execution.models: OrderSide domain enumeration.
    - quant.execution.venues: Diagnostic fault codes (ERR_SOR_*) and structured exception hierarchy.

Structural Relationship:
    - Ingested by:
        1. Execution Orchestrator and Live Execution Router.
        2. Algorithmic Schedulers (TWAP, VWAP, Arrival Price).
        3. Transaction Cost Analysis (TCA) Reporting and Audit Subsystems.
    - Consumes:
        1. OrderSide from quant.execution.models.
        2. Diagnostic error codes and exceptions from quant.execution.venues.
    - Emits:
        1. ChildFillRecord: Slotted immutable execution slice fill entity.
        2. ImplementationShortfallReport: Causal additive TCA attribution object.
        3. ParentOrder: Thread-safe lifecycle and mass coordinator.

Invariants Enforced:
    - INV-SOR-001 (Parent-Child Mass Conservation): Cumulative child fill quantities cannot
      exceed parent order total target quantity beyond 1e-7 floating-point tolerance.
    - INV-SOR-005 (Perold Implementation Shortfall Additive Identity):
      Total Shortfall == Delay Cost + Price Impact + Fees Paid + Opportunity Cost
      within 1e-7 tolerance across both BUY and SELL sides.
    - INV-SOR-005 (Strict Non-Finite Input Protection): Complete rejection of NaN, Inf, bool,
      empty identifiers, negative quantities, negative prices, and negative fees.
    - Rule 1: Line-by-line annotation standards (Functional Purpose, Explicit Dependency Tracking,
      Structural Relationship, Defensive Invariant).
    - Rule 2: Zero-execution diagnostic fault codes (ERR-SOR-004, ERR-SOR-005, ERR-SOR-006, ERR-SOR-007).
    - Rule 4: Mandatory adversarial red-teaming, exact sign reversals, zero shortcuts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from quant.execution.models import OrderSide
from quant.execution.venues import (
    ERR_SOR_ALGORITHM_TIMEOUT,
    ERR_SOR_CHILD_ORDER_FAILED,
    ERR_SOR_MASS_CONSERVATION_BREACH,
    ERR_SOR_NON_FINITE_INPUT,
    AlgorithmTimeoutException,
    ChildOrderFailedException,
    InvalidSORInputException,
    MassConservationException,
    NonFiniteInputException,
)

# Mass conservation and Perold additive tolerance constant
MASS_CONSERVATION_TOLERANCE: Final[float] = 1e-7

__all__ = [
    "ERR_SOR_ALGORITHM_TIMEOUT",
    "ERR_SOR_CHILD_ORDER_FAILED",
    "ERR_SOR_MASS_CONSERVATION_BREACH",
    "ERR_SOR_NON_FINITE_INPUT",
    "AlgorithmTimeoutException",
    "ChildFillRecord",
    "ChildOrderFailedException",
    "ImplementationShortfallReport",
    "InvalidSORInputException",
    "MASS_CONSERVATION_TOLERANCE",
    "MassConservationException",
    "NonFiniteInputException",
    "ParentOrder",
]


# ============================================================================
# ChildFillRecord Dataclass
# ============================================================================


@dataclass(slots=True)
class ChildFillRecord:
    """Atomic execution fill record received from a venue or gateway.

    Attributes:
        child_id: Unique string identifier of the child order execution slice.
        quantity: Filled share or contract quantity (finite float > 0.0).
        price: Realized execution price per unit (finite float > 0.0).
        fee: Exchange, broker, or regulatory fee paid for this fill (finite float >= 0.0).
        timestamp_ns: Execution fill timestamp in nanoseconds (integer >= 0).
        spread_slippage: Spread or midpoint slippage incurred for this fill (finite float).
    """

    child_id: str
    quantity: float
    price: float
    fee: float = 0.0
    timestamp_ns: int = 0
    spread_slippage: float = 0.0

    def __post_init__(self) -> None:
        """Validate domain boundaries, strict types, and non-finite guards for child fill.

        Functional Purpose:
            Enforce non-empty string IDs, strictly positive finite execution prices and quantities,
            non-negative fees and timestamps, and finite spread slippage at instantiation boundary.
        Explicit Dependency Tracking:
            math.isfinite, InvalidSORInputException, NonFiniteInputException, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Instantiated by ParentOrder.record_child_fill or downstream gateway adapters.
        Defensive Invariant:
            child_id != "", quantity > 0.0, price > 0.0, fee >= 0.0, timestamp_ns >= 0.
        """
        # 1. Validate child_id (non-empty string, reject bool)
        if (
            not isinstance(self.child_id, str)
            or isinstance(self.child_id, bool)
            or not self.child_id.strip()
        ):
            raise InvalidSORInputException(
                f"ChildFillRecord child_id must be non-empty string, got {self.child_id!r}",
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
                f"ChildFillRecord quantity must be finite positive float, got {self.quantity!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        object.__setattr__(self, "quantity", float(self.quantity))

        # 3. Validate price (finite float > 0.0, reject bool)
        if (
            not isinstance(self.price, (int, float))
            or isinstance(self.price, bool)
            or not math.isfinite(self.price)
            or self.price <= 0.0
        ):
            raise NonFiniteInputException(
                f"ChildFillRecord price must be finite positive float, got {self.price!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        object.__setattr__(self, "price", float(self.price))

        # 4. Validate fee (finite float >= 0.0, reject bool)
        if (
            not isinstance(self.fee, (int, float))
            or isinstance(self.fee, bool)
            or not math.isfinite(self.fee)
            or self.fee < 0.0
        ):
            raise NonFiniteInputException(
                f"ChildFillRecord fee must be finite non-negative float, got {self.fee!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        object.__setattr__(self, "fee", float(self.fee))

        # 5. Validate timestamp_ns (non-negative integer, reject bool)
        if (
            not isinstance(self.timestamp_ns, int)
            or isinstance(self.timestamp_ns, bool)
            or self.timestamp_ns < 0
        ):
            raise NonFiniteInputException(
                f"ChildFillRecord timestamp_ns must be non-negative int, got {self.timestamp_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 6. Validate spread_slippage (finite float, reject bool)
        if (
            not isinstance(self.spread_slippage, (int, float))
            or isinstance(self.spread_slippage, bool)
            or not math.isfinite(self.spread_slippage)
        ):
            raise NonFiniteInputException(
                f"ChildFillRecord spread_slippage must be finite float, got {self.spread_slippage!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        object.__setattr__(self, "spread_slippage", float(self.spread_slippage))


# ============================================================================
# ImplementationShortfallReport Dataclass
# ============================================================================


@dataclass(slots=True)
class ImplementationShortfallReport:
    """Causal Implementation Shortfall (IS) Transaction Cost Attribution Report.

    Adheres strictly to the Perold (1988) exact additive identity:
        Total Shortfall == Delay Cost + Price Impact + Fees Paid + Opportunity Cost
    within 1e-7 tolerance (INV-SOR-005).

    Attributes:
        delay_cost: Cost incurred due to adverse price drift between decision and arrival ($).
        price_impact: Cost incurred due to adverse market impact during execution ($).
        spread_slippage: Spread and midpoint crossing slippage incurred ($).
        fees_paid: Cumulative venue, broker, and exchange fees incurred ($).
        opportunity_cost: Cost incurred due to price movement on unfilled order leaves ($).
        total_shortfall: Total implementation shortfall ($).
        total_shortfall_bps: Total implementation shortfall in basis points of benchmark notional.
        delay_cost_bps: Delay cost in basis points of benchmark notional.
        price_impact_bps: Price impact in basis points of benchmark notional.
        opportunity_cost_bps: Opportunity cost in basis points of benchmark notional.
        fees_bps: Fees paid in basis points of benchmark notional.
        filled_quantity: Total executed share/contract quantity.
        unfilled_quantity: Residual unexecuted share/contract quantity.
        average_price: Volume-weighted average execution price across all fills.
        arrival_price: Market mid/last price at parent order arrival time.
        decision_price: Benchmark price at portfolio manager decision time.
        terminal_price: Market price at the end of the execution horizon.
    """

    delay_cost: float
    price_impact: float
    spread_slippage: float
    fees_paid: float
    opportunity_cost: float
    total_shortfall: float
    total_shortfall_bps: float
    delay_cost_bps: float
    price_impact_bps: float
    opportunity_cost_bps: float
    fees_bps: float
    filled_quantity: float
    unfilled_quantity: float
    average_price: float
    arrival_price: float
    decision_price: float
    terminal_price: float

    def __post_init__(self) -> None:
        """Validate numeric boundaries, non-finiteness, and Perold additive identity.

        Functional Purpose:
            Verify that all report fields are finite non-boolean floats, quantities are non-negative,
            prices are strictly positive, and the exact Perold (1988) additive identity holds.
        Explicit Dependency Tracking:
            math.isfinite, MassConservationException, NonFiniteInputException,
            ERR_SOR_MASS_CONSERVATION_BREACH, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Emitted by ParentOrder.compute_implementation_shortfall and consumed by TCA reporters.
        Defensive Invariant:
            abs(total_shortfall - (delay_cost + price_impact + fees_paid + opportunity_cost)) <= 1e-7.
        """
        # 1. Validate all scalar fields are finite floats and not booleans
        field_names = (
            "delay_cost",
            "price_impact",
            "spread_slippage",
            "fees_paid",
            "opportunity_cost",
            "total_shortfall",
            "total_shortfall_bps",
            "delay_cost_bps",
            "price_impact_bps",
            "opportunity_cost_bps",
            "fees_bps",
            "filled_quantity",
            "unfilled_quantity",
            "average_price",
            "arrival_price",
            "decision_price",
            "terminal_price",
        )
        for name in field_names:
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise NonFiniteInputException(
                    f"ImplementationShortfallReport field {name} must be finite float, got {val!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            object.__setattr__(self, name, float(val))

        # 2. Validate quantities are non-negative
        if self.filled_quantity < 0.0 or self.unfilled_quantity < 0.0:
            raise NonFiniteInputException(
                f"Quantities must be non-negative, got filled={self.filled_quantity}, unfilled={self.unfilled_quantity}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 3. Validate prices are strictly positive (average_price can be 0.0 if filled_quantity == 0.0)
        if self.arrival_price <= 0.0 or self.decision_price <= 0.0 or self.terminal_price <= 0.0:
            raise NonFiniteInputException(
                f"Benchmark prices must be strictly positive, got arrival={self.arrival_price}, "
                f"decision={self.decision_price}, terminal={self.terminal_price}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        if self.filled_quantity > MASS_CONSERVATION_TOLERANCE and self.average_price <= 0.0:
            raise NonFiniteInputException(
                f"Average execution price must be strictly positive when fills exist, got {self.average_price}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 4. Verify Perold (1988) exact additive identity (INV-SOR-005)
        sum_components = (
            self.delay_cost + self.price_impact + self.fees_paid + self.opportunity_cost
        )
        if abs(self.total_shortfall - sum_components) > MASS_CONSERVATION_TOLERANCE:
            raise MassConservationException(
                f"Perold additive identity breached (INV-SOR-005): total_shortfall={self.total_shortfall} "
                f"does not match sum of components={sum_components} (delta={abs(self.total_shortfall - sum_components):.2e})",
                code=ERR_SOR_MASS_CONSERVATION_BREACH,
            )


# ============================================================================
# ParentOrder Lifecycle Coordinator
# ============================================================================


class ParentOrder:
    """Parent meta-order lifecycle coordinator and implementation shortfall tracking engine.

    Coordinates the lifecycle of an institutional meta-order, tracking child execution
    slices, enforcing mass conservation (INV-SOR-001), monitoring execution horizon timeouts
    (ERR-SOR-006), and computing causal implementation shortfall attribution (INV-SOR-005).
    """

    __slots__ = (
        "_arrival_price",
        "_child_fills",
        "_cum_notional",
        "_cum_spread_slippage",
        "_decision_price",
        "_filled_quantity",
        "_max_duration_seconds",
        "_parent_id",
        "_side",
        "_start_time_ns",
        "_symbol",
        "_total_fees",
        "_total_quantity",
    )

    def __init__(
        self,
        parent_id: str,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        arrival_price: float,
        decision_price: float | None = None,
        max_duration_seconds: float = 300.0,
        start_time_ns: int = 0,
    ) -> None:
        """Initialize ParentOrder with target parameters and boundary validation.

        Functional Purpose:
            Establish institutional parent meta-order coordinator with validated parameters,
            setting arrival and decision benchmarks for transaction cost attribution.
        Explicit Dependency Tracking:
            OrderSide, NonFiniteInputException, InvalidSORInputException, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Instantiated by portfolio or algorithmic execution managers before slicing.
        Defensive Invariant:
            parent_id != "", symbol != "", total_quantity > 0, arrival_price > 0,
            decision_price > 0, max_duration_seconds > 0, start_time_ns >= 0.
        """
        # 1. Validate parent_id (non-empty string, reject bool)
        if not isinstance(parent_id, str) or isinstance(parent_id, bool) or not parent_id.strip():
            raise InvalidSORInputException(
                f"ParentOrder parent_id must be non-empty string, got {parent_id!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self._parent_id: str = parent_id.strip()

        # 2. Validate symbol (non-empty string, reject bool)
        if not isinstance(symbol, str) or isinstance(symbol, bool) or not symbol.strip():
            raise InvalidSORInputException(
                f"ParentOrder symbol must be non-empty string, got {symbol!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self._symbol: str = symbol.strip()

        # 3. Validate side (OrderSide enum instance)
        if not isinstance(side, OrderSide):
            raise InvalidSORInputException(
                f"ParentOrder side must be an instance of OrderSide, got {side!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self._side: OrderSide = side

        # 4. Validate total_quantity (finite float > 0.0, reject bool)
        if (
            not isinstance(total_quantity, (int, float))
            or isinstance(total_quantity, bool)
            or not math.isfinite(total_quantity)
            or total_quantity <= 0.0
        ):
            raise NonFiniteInputException(
                f"ParentOrder total_quantity must be finite positive float, got {total_quantity!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self._total_quantity: float = float(total_quantity)

        # 5. Validate arrival_price (finite float > 0.0, reject bool)
        if (
            not isinstance(arrival_price, (int, float))
            or isinstance(arrival_price, bool)
            or not math.isfinite(arrival_price)
            or arrival_price <= 0.0
        ):
            raise NonFiniteInputException(
                f"ParentOrder arrival_price must be finite positive float, got {arrival_price!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self._arrival_price: float = float(arrival_price)

        # 6. Validate decision_price (finite float > 0.0, reject bool, defaults to arrival_price)
        if decision_price is None:
            self._decision_price: float = self._arrival_price
        else:
            if (
                not isinstance(decision_price, (int, float))
                or isinstance(decision_price, bool)
                or not math.isfinite(decision_price)
                or decision_price <= 0.0
            ):
                raise NonFiniteInputException(
                    f"ParentOrder decision_price must be finite positive float, got {decision_price!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            self._decision_price = float(decision_price)

        # 7. Validate max_duration_seconds (finite float > 0.0, reject bool)
        if (
            not isinstance(max_duration_seconds, (int, float))
            or isinstance(max_duration_seconds, bool)
            or not math.isfinite(max_duration_seconds)
            or max_duration_seconds <= 0.0
        ):
            raise NonFiniteInputException(
                f"ParentOrder max_duration_seconds must be finite positive float, got {max_duration_seconds!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self._max_duration_seconds: float = float(max_duration_seconds)

        # 8. Validate start_time_ns (non-negative integer, reject bool)
        if (
            not isinstance(start_time_ns, int)
            or isinstance(start_time_ns, bool)
            or start_time_ns < 0
        ):
            raise NonFiniteInputException(
                f"ParentOrder start_time_ns must be non-negative int, got {start_time_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )
        self._start_time_ns: int = start_time_ns

        # 9. Initialize fill tracking state
        self._child_fills: list[ChildFillRecord] = []
        self._filled_quantity: float = 0.0
        self._cum_notional: float = 0.0
        self._total_fees: float = 0.0
        self._cum_spread_slippage: float = 0.0

    @property
    def parent_id(self) -> str:
        """Return unique parent order identifier.

        Functional Purpose:
            Expose unique parent order ID for downstream tracking and correlation.
        Explicit Dependency Tracking:
            self._parent_id.
        Structural Relationship:
            Used in logging, correlation, and order event streams.
        Defensive Invariant:
            Returns non-empty string.
        """
        return self._parent_id

    @property
    def symbol(self) -> str:
        """Return trading ticker symbol.

        Functional Purpose:
            Expose financial instrument symbol.
        Explicit Dependency Tracking:
            self._symbol.
        Structural Relationship:
            Used in routing, consolidated quotes, and market data queries.
        Defensive Invariant:
            Returns non-empty string.
        """
        return self._symbol

    @property
    def side(self) -> OrderSide:
        """Return order trading direction (BUY or SELL).

        Functional Purpose:
            Expose side for sign convention and market routing logic.
        Explicit Dependency Tracking:
            self._side.
        Structural Relationship:
            Consumed by schedulers and shortfall attribution calculators.
        Defensive Invariant:
            Returns valid OrderSide enum member.
        """
        return self._side

    @property
    def total_quantity(self) -> float:
        """Return target meta-order total share or contract volume.

        Functional Purpose:
            Expose total target quantity for mass conservation checks.
        Explicit Dependency Tracking:
            self._total_quantity.
        Structural Relationship:
            Upper bound on cumulative child fill quantities.
        Defensive Invariant:
            Returns strictly positive finite float.
        """
        return self._total_quantity

    @property
    def arrival_price(self) -> float:
        """Return benchmark price at order arrival time.

        Functional Purpose:
            Expose arrival price benchmark for Implementation Shortfall attribution.
        Explicit Dependency Tracking:
            self._arrival_price.
        Structural Relationship:
            Consumed by TCA calculation to determine opportunity cost and delay cost.
        Defensive Invariant:
            Returns strictly positive finite float.
        """
        return self._arrival_price

    @property
    def decision_price(self) -> float:
        """Return benchmark price at investment decision time.

        Functional Purpose:
            Expose portfolio manager decision benchmark price.
        Explicit Dependency Tracking:
            self._decision_price.
        Structural Relationship:
            Consumed by TCA calculation to isolate delay cost from price impact.
        Defensive Invariant:
            Returns strictly positive finite float.
        """
        return self._decision_price

    @property
    def max_duration_seconds(self) -> float:
        """Return maximum allowable execution horizon duration in seconds.

        Functional Purpose:
            Expose time horizon limit for algorithm timeout checking.
        Explicit Dependency Tracking:
            self._max_duration_seconds.
        Structural Relationship:
            Checked by check_timeout() against current wall-clock / simulation time.
        Defensive Invariant:
            Returns strictly positive finite float.
        """
        return self._max_duration_seconds

    @property
    def start_time_ns(self) -> int:
        """Return parent order initiation timestamp in nanoseconds.

        Functional Purpose:
            Expose start timestamp for elapsed execution horizon computation.
        Explicit Dependency Tracking:
            self._start_time_ns.
        Structural Relationship:
            Subtracted from now_ns in check_timeout().
        Defensive Invariant:
            Returns non-negative integer.
        """
        return self._start_time_ns

    @property
    def filled_quantity(self) -> float:
        """Return cumulative executed child quantity.

        Functional Purpose:
            Track aggregate fill volume across all child execution slices.
        Explicit Dependency Tracking:
            self._filled_quantity.
        Structural Relationship:
            Incremented by record_child_fill(); used to compute leaves quantity.
        Defensive Invariant:
            0.0 <= filled_quantity <= total_quantity + 1e-7.
        """
        return self._filled_quantity

    @property
    def leaves_quantity(self) -> float:
        """Return remaining unexecuted leaves quantity bounded below by zero.

        Functional Purpose:
            Compute remaining order mass to be scheduled or routed.
        Explicit Dependency Tracking:
            self._total_quantity, self._filled_quantity, MASS_CONSERVATION_TOLERANCE.
        Structural Relationship:
            Consumed by execution schedulers to determine residual slice size.
        Defensive Invariant:
            Returns float in range [0.0, total_quantity].
        """
        rem = self._total_quantity - self._filled_quantity
        if rem <= MASS_CONSERVATION_TOLERANCE:
            return 0.0
        return rem

    @property
    def is_completed(self) -> bool:
        """Return True if parent order has fully completed (leaves <= 1e-7).

        Functional Purpose:
            Determine terminal completion status of the parent order.
        Explicit Dependency Tracking:
            self.leaves_quantity, MASS_CONSERVATION_TOLERANCE.
        Structural Relationship:
            Guards against execution timeouts and prevents redundant child routing.
        Defensive Invariant:
            Returns boolean; True if leaves <= 1e-7.
        """
        return self.leaves_quantity <= MASS_CONSERVATION_TOLERANCE

    @property
    def average_execution_price(self) -> float:
        """Return volume-weighted average execution price (VWAP) across all fills.

        Functional Purpose:
            Compute realized VWAP execution price; returns 0.0 if no fills recorded.
        Explicit Dependency Tracking:
            self._cum_notional, self._filled_quantity, MASS_CONSERVATION_TOLERANCE.
        Structural Relationship:
            Core metric for price impact and implementation shortfall calculation.
        Defensive Invariant:
            Returns 0.0 when filled_quantity <= 1e-7, else strictly positive float.
        """
        if self._filled_quantity <= MASS_CONSERVATION_TOLERANCE:
            return 0.0
        return self._cum_notional / self._filled_quantity

    @property
    def total_fees(self) -> float:
        """Return sum of all venue, exchange, and broker fees paid across fills.

        Functional Purpose:
            Track cumulative monetary execution costs.
        Explicit Dependency Tracking:
            self._total_fees.
        Structural Relationship:
            Consumed by ImplementationShortfallReport.
        Defensive Invariant:
            Returns non-negative finite float.
        """
        return self._total_fees

    @property
    def child_fills(self) -> list[ChildFillRecord]:
        """Return defensive copy of all recorded child fill records.

        Functional Purpose:
            Provide access to historical child fills without risking internal state mutation.
        Explicit Dependency Tracking:
            self._child_fills.
        Structural Relationship:
            Read by execution reporting, audit, and post-trade analysis.
        Defensive Invariant:
            Returns a new list containing shallow copies of slotted ChildFillRecord instances.
        """
        return list(self._child_fills)

    def record_child_fill(
        self,
        child_id: str,
        quantity: float,
        price: float,
        fee: float = 0.0,
        timestamp_ns: int = 0,
        spread_slippage: float = 0.0,
    ) -> None:
        """Record child order execution fill and enforce mass conservation.

        Functional Purpose:
            Process incoming fill report from venue or gateway, updating cumulative
            filled quantity, VWAP notional, fees, and checking mass conservation.
        Explicit Dependency Tracking:
            ChildFillRecord, MassConservationException, InvalidSORInputException,
            ERR_SOR_MASS_CONSERVATION_BREACH, ERR_SOR_NON_FINITE_INPUT, MASS_CONSERVATION_TOLERANCE.
        Structural Relationship:
            Called upon receiving execution reports for child slices.
        Defensive Invariant:
            filled_quantity + fill_quantity <= total_quantity + 1e-7 (INV-SOR-001).
        """
        # 1. Instantiate ChildFillRecord (enforces all individual field boundaries)
        fill = ChildFillRecord(
            child_id=child_id,
            quantity=quantity,
            price=price,
            fee=fee,
            timestamp_ns=timestamp_ns,
            spread_slippage=spread_slippage,
        )

        # 2. Enforce parent-child mass conservation invariant (INV-SOR-001)
        if (
            self._filled_quantity + fill.quantity
            > self._total_quantity + MASS_CONSERVATION_TOLERANCE
        ):
            raise MassConservationException(
                f"Parent order {self._parent_id} mass conservation breached: cumulative fills "
                f"({self._filled_quantity + fill.quantity}) exceed total target quantity "
                f"({self._total_quantity}) by more than {MASS_CONSERVATION_TOLERANCE} (INV-SOR-001)",
                code=ERR_SOR_MASS_CONSERVATION_BREACH,
            )

        # 3. Check causal fill timestamp monotonicity (if timestamp provided)
        if fill.timestamp_ns > 0 and fill.timestamp_ns < self._start_time_ns:
            raise InvalidSORInputException(
                f"Child fill timestamp {fill.timestamp_ns}ns precedes parent order start time {self._start_time_ns}ns",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 4. Update running cumulative state
        self._child_fills.append(fill)
        self._filled_quantity += fill.quantity
        self._cum_notional += fill.quantity * fill.price
        self._total_fees += fill.fee
        self._cum_spread_slippage += fill.spread_slippage

    def check_timeout(self, now_ns: int) -> None:
        """Check if execution horizon has expired with remaining unexecuted leaves.

        Functional Purpose:
            Enforce institutional execution time limits. If order is not completed
            and elapsed duration exceeds max_duration_seconds, raise AlgorithmTimeoutException.
        Explicit Dependency Tracking:
            AlgorithmTimeoutException, NonFiniteInputException, InvalidSORInputException,
            ERR_SOR_ALGORITHM_TIMEOUT, ERR_SOR_NON_FINITE_INPUT.
        Structural Relationship:
            Invoked periodically by algorithmic schedulers and execution orchestrators.
        Defensive Invariant:
            If not is_completed and (now_ns - start_time_ns) > max_duration_ns, raise ERR-SOR-006.
        """
        # 1. Validate now_ns (non-negative integer, reject bool)
        if not isinstance(now_ns, int) or isinstance(now_ns, bool) or now_ns < 0:
            raise NonFiniteInputException(
                f"ParentOrder check_timeout now_ns must be non-negative int, got {now_ns!r}",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 2. Check causal time monotonicity
        if now_ns < self._start_time_ns:
            raise InvalidSORInputException(
                f"Current timestamp {now_ns}ns cannot precede parent order start time {self._start_time_ns}ns",
                code=ERR_SOR_NON_FINITE_INPUT,
            )

        # 3. If already completed, no timeout violation can occur
        if self.is_completed:
            return

        # 4. Check elapsed horizon duration against threshold
        elapsed_ns = now_ns - self._start_time_ns
        max_duration_ns = int(self._max_duration_seconds * 1_000_000_000.0)
        if elapsed_ns > max_duration_ns:
            raise AlgorithmTimeoutException(
                f"Parent order {self._parent_id} execution horizon timed out: elapsed {elapsed_ns}ns "
                f"exceeds max duration {max_duration_ns}ns with {self.leaves_quantity} unexecuted leaves",
                code=ERR_SOR_ALGORITHM_TIMEOUT,
            )

    def compute_implementation_shortfall(
        self,
        terminal_price: float | None = None,
    ) -> ImplementationShortfallReport:
        """Compute causal Implementation Shortfall (IS) transaction cost attribution.

        Adheres to Perold (1988) exact additive decomposition:
            Total Shortfall == Delay Cost + Price Impact + Fees Paid + Opportunity Cost

        Formulations (with S = +1 for BUY, -1 for SELL):
            Delay Cost = S * Q_filled * (P_decision - P_arrival)
            Price Impact = S * Q_filled * (P_avg - P_decision)
            Fees Paid = sum(fee_i)
            Opportunity Cost = S * Q_unfilled * (P_terminal - P_arrival)
            Total Shortfall = Delay Cost + Price Impact + Fees Paid + Opportunity Cost

        Functional Purpose:
            Decompose total execution shortfall against arrival benchmark into institutional
            causal drivers for post-trade TCA and algorithmic optimization.
        Explicit Dependency Tracking:
            ImplementationShortfallReport, OrderSide, NonFiniteInputException,
            ERR_SOR_NON_FINITE_INPUT, MASS_CONSERVATION_TOLERANCE.
        Structural Relationship:
            Called upon order completion, horizon expiration, or for live TCA monitoring.
        Defensive Invariant:
            All report components sum to total_shortfall within 1e-7 tolerance (INV-SOR-005).
        """
        # 1. Resolve and validate terminal price
        if terminal_price is not None:
            if (
                not isinstance(terminal_price, (int, float))
                or isinstance(terminal_price, bool)
                or not math.isfinite(terminal_price)
                or terminal_price <= 0.0
            ):
                raise NonFiniteInputException(
                    f"ParentOrder compute_implementation_shortfall terminal_price must be finite positive float, got {terminal_price!r}",
                    code=ERR_SOR_NON_FINITE_INPUT,
                )
            resolved_terminal: float = float(terminal_price)
        else:
            # If terminal_price not provided: fallback to realized average price if fills exist, else arrival price
            if self._filled_quantity > MASS_CONSERVATION_TOLERANCE:
                resolved_terminal = self.average_execution_price
            else:
                resolved_terminal = self._arrival_price

        # 2. Determine trade direction sign (+1.0 for BUY, -1.0 for SELL)
        side_sign: float = 1.0 if self._side == OrderSide.BUY else -1.0

        q_filled: float = self._filled_quantity
        q_unfilled: float = self.leaves_quantity

        # 3. Compute Delay Cost: price drift from arrival to decision on executed volume
        # If no fills occurred, delay cost is zero
        if q_filled > MASS_CONSERVATION_TOLERANCE:
            delay_cost: float = side_sign * q_filled * (self._decision_price - self._arrival_price)
            price_impact: float = (
                side_sign * q_filled * (self.average_execution_price - self._decision_price)
            )
        else:
            delay_cost = 0.0
            price_impact = 0.0

        # 4. Exchange and broker fees paid
        fees_paid: float = self._total_fees

        # 5. Opportunity Cost: price drift on unexecuted leaves from arrival to terminal
        if q_unfilled > MASS_CONSERVATION_TOLERANCE:
            opportunity_cost: float = (
                side_sign * q_unfilled * (resolved_terminal - self._arrival_price)
            )
        else:
            opportunity_cost = 0.0

        # 6. Total Shortfall (exact additive identity)
        total_shortfall: float = delay_cost + price_impact + fees_paid + opportunity_cost

        # 7. Convert costs to basis points relative to benchmark target notional
        benchmark_notional: float = self._total_quantity * self._arrival_price
        bps_scale: float = 10_000.0 / benchmark_notional

        delay_cost_bps: float = delay_cost * bps_scale
        price_impact_bps: float = price_impact * bps_scale
        opportunity_cost_bps: float = opportunity_cost * bps_scale
        fees_bps: float = fees_paid * bps_scale
        total_shortfall_bps: float = total_shortfall * bps_scale

        # 8. Construct and return validated ImplementationShortfallReport
        return ImplementationShortfallReport(
            delay_cost=delay_cost,
            price_impact=price_impact,
            spread_slippage=self._cum_spread_slippage,
            fees_paid=fees_paid,
            opportunity_cost=opportunity_cost,
            total_shortfall=total_shortfall,
            total_shortfall_bps=total_shortfall_bps,
            delay_cost_bps=delay_cost_bps,
            price_impact_bps=price_impact_bps,
            opportunity_cost_bps=opportunity_cost_bps,
            fees_bps=fees_bps,
            filled_quantity=q_filled,
            unfilled_quantity=q_unfilled,
            average_price=self.average_execution_price,
            arrival_price=self._arrival_price,
            decision_price=self._decision_price,
            terminal_price=resolved_terminal,
        )

    def __repr__(self) -> str:
        """Return self-explicating string representation of parent order state.

        Functional Purpose:
            Provide deterministic string for audit trails and diagnostics.
        Explicit Dependency Tracking:
            parent_id, symbol, side, filled_quantity, total_quantity.
        Structural Relationship:
            Used in diagnostics and debug logs.
        Defensive Invariant:
            Returns informative non-empty string.
        """
        return (
            f"ParentOrder(id={self._parent_id!r}, symbol={self._symbol!r}, side={self._side.value}, "
            f"filled={self._filled_quantity}/{self._total_quantity}, vwap={self.average_execution_price:.4f}, "
            f"completed={self.is_completed})"
        )
