"""Unit test suite verifying SmartOrderRouter, VenueHealth, and RoutedVenueOrder invariants.

Purpose:
    Exhaustively tests microstructural smart order routing, sequential dark probing,
    midpoint price improvement, toxic markout watchdog and quarantine auto-recovery,
    closed-form algebraic KKT lit waterfilling, sub-0.05ms latency SLA, and asynchronous
    gateway orchestration.

Dependencies:
    - pytest: Test framework and async test execution.
    - time: Monotonic clock for sub-millisecond benchmarking.
    - quant.execution.gateway: PaperExecutionGateway.
    - quant.execution.models: OrderSide, OrderState, OrderType, TimeInForce.
    - quant.execution.sor: SmartOrderRouter, RoutedVenueOrder, VenueHealth.
    - quant.execution.venues: ConsolidatedQuote, VenueProfile, VenueType, SORError,
      InsufficientLiquidityException, NBBOViolationException, ChildOrderFailedException,
      InvalidSORInputException, ERR_SOR_INSUFFICIENT_LIQUIDITY, ERR_SOR_NBBO_VIOLATION,
      ERR_SOR_CHILD_ORDER_FAILED, ERR_SOR_NON_FINITE_INPUT.

Invariants Enforced:
    - INV-SOR-001: Execution Mass Conservation across routed child orders.
    - INV-SOR-002: Non-Worse-Than-NBBO and exact midpoint dark price improvement.
    - INV-SOR-005: Strict non-finite input protection and boolean rejection.
    - INV-SOR-006: Hot-path latency SLA (< 0.05ms per routing allocation).
"""

from __future__ import annotations

import time

import pytest

from quant.execution.gateway import PaperExecutionGateway
from quant.execution.models import (
    ExecutionReport,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)
from quant.execution.sor import (
    RoutedVenueOrder,
    SmartOrderRouter,
    VenueHealth,
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

# ============================================================================
# Dataclass Invariant Tests
# ============================================================================


def test_routed_venue_order_validation() -> None:
    """Verify RoutedVenueOrder enforces boundary types and positive finite numbers."""
    order = RoutedVenueOrder(
        venue_id="NASDAQ",
        venue_type=VenueType.LIT_EXCHANGE,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        price=150.0,
    )
    assert order.venue_id == "NASDAQ"
    assert order.quantity == 100.0
    assert order.price == 150.0
    assert order.time_in_force == TimeInForce.IOC
    assert order.order_type == OrderType.LIMIT
    assert not order.is_dark

    # Rejection of bool quantity
    with pytest.raises(InvalidSORInputException) as exc_info:
        RoutedVenueOrder(
            venue_id="NASDAQ",
            venue_type=VenueType.LIT_EXCHANGE,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=True,  # type: ignore[arg-type]
            price=150.0,
        )
    assert exc_info.value.code == ERR_SOR_NON_FINITE_INPUT

    # Rejection of non-positive quantity
    with pytest.raises(InvalidSORInputException) as exc_info:
        RoutedVenueOrder(
            venue_id="NASDAQ",
            venue_type=VenueType.LIT_EXCHANGE,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0.0,
            price=150.0,
        )
    assert exc_info.value.code == ERR_SOR_NON_FINITE_INPUT

    # Rejection of NaN price
    with pytest.raises(InvalidSORInputException) as exc_info:
        RoutedVenueOrder(
            venue_id="NASDAQ",
            venue_type=VenueType.LIT_EXCHANGE,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            price=float("nan"),
        )
    assert exc_info.value.code == ERR_SOR_NON_FINITE_INPUT


def test_routed_venue_order_all_rejections() -> None:
    """Verify complete domain rejection suite on RoutedVenueOrder."""
    # Empty venue_id
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder("", VenueType.LIT_EXCHANGE, "AAPL", OrderSide.BUY, 10.0, 100.0)

    # Invalid venue_type
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder("V1", "INVALID", "AAPL", OrderSide.BUY, 10.0, 100.0)  # type: ignore[arg-type]

    # Empty symbol
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder("V1", VenueType.LIT_EXCHANGE, "  ", OrderSide.BUY, 10.0, 100.0)

    # Invalid side
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder("V1", VenueType.LIT_EXCHANGE, "AAPL", "INVALID", 10.0, 100.0)  # type: ignore[arg-type]

    # Invalid price (bool, negative, inf)
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder(
            "V1",
            VenueType.LIT_EXCHANGE,
            "AAPL",
            OrderSide.BUY,
            10.0,
            False,  # type: ignore[arg-type]
        )
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder("V1", VenueType.LIT_EXCHANGE, "AAPL", OrderSide.BUY, 10.0, -10.0)
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder("V1", VenueType.LIT_EXCHANGE, "AAPL", OrderSide.BUY, 10.0, float("inf"))

    # Invalid time_in_force
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder(
            "V1",
            VenueType.LIT_EXCHANGE,
            "AAPL",
            OrderSide.BUY,
            10.0,
            100.0,
            time_in_force="INVALID",  # type: ignore[arg-type]
        )

    # Invalid order_type
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder(
            "V1",
            VenueType.LIT_EXCHANGE,
            "AAPL",
            OrderSide.BUY,
            10.0,
            100.0,
            order_type="INVALID",  # type: ignore[arg-type]
        )

    # Invalid is_dark
    with pytest.raises(InvalidSORInputException):
        RoutedVenueOrder(
            "V1",
            VenueType.LIT_EXCHANGE,
            "AAPL",
            OrderSide.BUY,
            10.0,
            100.0,
            is_dark="NOT_A_BOOL",  # type: ignore[arg-type]
        )


def test_venue_health_properties_and_recovery() -> None:
    """Verify VenueHealth tracking, average markout, and quarantine lifecycle."""
    health = VenueHealth(venue_id="DARK_ATS")
    assert health.is_active(now_ns=1_000_000)
    assert health.average_markout_bps == 0.0

    # Record first fill with favorable markout
    health.record_fill(markout_bps=5.0, now_ns=1_000_000, toxic_threshold_bps=-2.0, toxic_limit=2)
    assert health.total_fills == 1
    assert health.cumulative_markout_bps == 5.0
    assert health.average_markout_bps == 5.0
    assert health.consecutive_toxic_fills == 0
    assert not health.is_quarantined

    # Record 1st toxic fill
    health.record_fill(markout_bps=-3.5, now_ns=2_000_000, toxic_threshold_bps=-2.0, toxic_limit=2)
    assert health.total_fills == 2
    assert health.consecutive_toxic_fills == 1
    assert not health.is_quarantined

    # Record non-toxic fill: resets consecutive toxic counter
    health.record_fill(markout_bps=1.0, now_ns=3_000_000, toxic_threshold_bps=-2.0, toxic_limit=2)
    assert health.consecutive_toxic_fills == 0
    assert not health.is_quarantined

    # Record 2 consecutive toxic fills -> triggers quarantine
    health.record_fill(
        markout_bps=-4.0,
        now_ns=4_000_000,
        toxic_threshold_bps=-2.0,
        toxic_limit=2,
        quarantine_duration_ns=10_000_000,
    )
    assert health.consecutive_toxic_fills == 1
    assert not health.is_quarantined

    health.record_fill(
        markout_bps=-5.0,
        now_ns=5_000_000,
        toxic_threshold_bps=-2.0,
        toxic_limit=2,
        quarantine_duration_ns=10_000_000,
    )
    assert health.consecutive_toxic_fills == 2
    assert health.is_quarantined
    assert health.quarantine_until_ns == 15_000_000
    assert not health.is_active(now_ns=10_000_000)

    # Auto-recovery after quarantine horizon expires
    assert health.is_active(now_ns=15_000_000)
    assert not health.is_quarantined
    assert health.consecutive_toxic_fills == 0


def test_venue_health_all_rejections() -> None:
    """Verify initialization and record_fill rejections on VenueHealth."""
    # VenueHealth init rejections
    with pytest.raises(InvalidSORInputException):
        VenueHealth("")
    with pytest.raises(InvalidSORInputException):
        VenueHealth("V1", is_quarantined="NOT_BOOL")  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        VenueHealth("V1", total_fills=-1)
    with pytest.raises(InvalidSORInputException):
        VenueHealth("V1", total_fills=True)  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        VenueHealth("V1", cumulative_markout_bps=float("nan"))
    with pytest.raises(InvalidSORInputException):
        VenueHealth("V1", cumulative_markout_bps=True)  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        VenueHealth("V1", consecutive_toxic_fills=-1)
    with pytest.raises(InvalidSORInputException):
        VenueHealth("V1", quarantine_until_ns=-1)

    # record_fill parameter rejections
    health = VenueHealth("V1")
    with pytest.raises(InvalidSORInputException):
        health.record_fill(float("nan"), 1_000_000)
    with pytest.raises(InvalidSORInputException):
        health.record_fill(True, 1_000_000)  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        health.record_fill(1.0, -1)
    with pytest.raises(InvalidSORInputException):
        health.record_fill(1.0, True)  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        health.record_fill(1.0, 1_000, toxic_threshold_bps=float("inf"))
    with pytest.raises(InvalidSORInputException):
        health.record_fill(1.0, 1_000, toxic_limit=0)
    with pytest.raises(InvalidSORInputException):
        health.record_fill(1.0, 1_000, quarantine_duration_ns=-1)


# ============================================================================
# SmartOrderRouter Allocation & Waterfilling Tests
# ============================================================================


def test_smart_order_router_init_rejections() -> None:
    """Verify SmartOrderRouter constructor input validation."""
    with pytest.raises(InvalidSORInputException):
        SmartOrderRouter({})  # Empty venues
    with pytest.raises(InvalidSORInputException):
        SmartOrderRouter("NOT_A_DICT")  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        SmartOrderRouter({"": VenueProfile("V", VenueType.LIT_EXCHANGE, 0.0, 1.0)})
    with pytest.raises(InvalidSORInputException):
        SmartOrderRouter({"V": "NOT_PROFILE"})  # type: ignore[arg-type]
    venues = {"LIT_1": VenueProfile("LIT_1", VenueType.LIT_EXCHANGE, 0.0, 1.0)}
    with pytest.raises(InvalidSORInputException):
        SmartOrderRouter(venues, toxic_markout_threshold_bps=float("nan"))
    with pytest.raises(InvalidSORInputException):
        SmartOrderRouter(venues, toxic_fill_limit=0)
    with pytest.raises(InvalidSORInputException):
        SmartOrderRouter(venues, quarantine_duration_sec=-1.0)


def test_dark_first_probing_and_midpoint_pricing() -> None:
    """Verify dark pool is probed first with exact midpoint price."""
    venues = {
        "DARK_1": VenueProfile("DARK_1", VenueType.DARK_POOL, maker_fee_bps=0.0, taker_fee_bps=0.5),
        "LIT_1": VenueProfile(
            "LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=-0.2, taker_fee_bps=1.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )
    allocations = router.compute_routing_allocation(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        enable_dark=True,
    )
    # Must probe dark pool first with midpoint limit price
    assert "DARK_1" in allocations
    assert len(allocations) == 1
    assert allocations["DARK_1"].price == 150.05
    assert allocations["DARK_1"].quantity == 100.0
    assert allocations["DARK_1"].is_dark
    assert allocations["DARK_1"].time_in_force == TimeInForce.IOC


def test_closed_form_kkt_lit_waterfilling() -> None:
    """Verify closed-form KKT waterfilling fills lowest fee first and spills over."""
    venues = {
        "CHEAP_EXCHANGE": VenueProfile(
            "CHEAP_EXCHANGE", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=0.5
        ),
        "PRICEY_EXCHANGE": VenueProfile(
            "PRICEY_EXCHANGE", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=2.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={
            "CHEAP_EXCHANGE": (500.0, 50.0),  # Only 50 available on cheap venue
            "PRICEY_EXCHANGE": (500.0, 500.0),  # Ample depth on pricey venue
        },
    )
    allocations = router.compute_routing_allocation(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        enable_dark=False,
    )
    # Should fill 50 on cheap exchange, remaining 50 on pricey exchange
    assert allocations["CHEAP_EXCHANGE"].quantity == 50.0
    assert allocations["PRICEY_EXCHANGE"].quantity == 50.0
    assert allocations["CHEAP_EXCHANGE"].price == 150.10
    assert allocations["PRICEY_EXCHANGE"].price == 150.10


def test_negative_fee_rebates_prioritizing_rebate_venues() -> None:
    """Verify inverted lit venues with negative taker fees (rebates) are prioritized."""
    venues = {
        "STANDARD_LIT": VenueProfile(
            "STANDARD_LIT", VenueType.LIT_EXCHANGE, maker_fee_bps=-0.1, taker_fee_bps=0.8
        ),
        "INVERTED_REBATE_LIT": VenueProfile(
            "INVERTED_REBATE_LIT", VenueType.LIT_EXCHANGE, maker_fee_bps=0.5, taker_fee_bps=-0.3
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="MSFT",
        bid_price=300.00,
        bid_quantity=1000.0,
        ask_price=300.05,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={
            "STANDARD_LIT": (200.0, 200.0),
            "INVERTED_REBATE_LIT": (100.0, 100.0),
        },
    )
    # Slicing 80 shares should be routed entirely to inverted rebate venue
    allocations = router.compute_routing_allocation(
        symbol="MSFT",
        side=OrderSide.BUY,
        quantity=80.0,
        quote=quote,
        enable_dark=False,
    )
    assert len(allocations) == 1
    assert "INVERTED_REBATE_LIT" in allocations
    assert allocations["INVERTED_REBATE_LIT"].quantity == 80.0


def test_insufficient_liquidity_rejection() -> None:
    """Verify InsufficientLiquidityException is raised when depth < quantity and allow_partial=False."""
    venues = {
        "LIT_1": VenueProfile(
            "LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=30.0,
        ask_price=150.10,
        ask_quantity=30.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (30.0, 30.0)},
    )
    with pytest.raises(InsufficientLiquidityException) as exc_info:
        router.compute_routing_allocation(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            quote=quote,
            enable_dark=False,
            allow_partial=False,
        )
    assert exc_info.value.code == ERR_SOR_INSUFFICIENT_LIQUIDITY

    # With allow_partial=True, should fill available 30
    partial = router.compute_routing_allocation(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        enable_dark=False,
        allow_partial=True,
    )
    assert partial["LIT_1"].quantity == 30.0


def test_sell_order_waterfilling() -> None:
    """Verify SELL order waterfilling routes at bid price across lit venues."""
    venues = {
        "VENUE_A": VenueProfile(
            "VENUE_A", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=0.2
        ),
        "VENUE_B": VenueProfile(
            "VENUE_B", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=0.6
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="TSLA",
        bid_price=200.00,
        bid_quantity=500.0,
        ask_price=200.05,
        ask_quantity=500.0,
        timestamp_ns=1_000_000,
        venue_depths={
            "VENUE_A": (40.0, 100.0),
            "VENUE_B": (100.0, 100.0),
        },
    )
    allocations = router.compute_routing_allocation(
        symbol="TSLA",
        side=OrderSide.SELL,
        quantity=60.0,
        quote=quote,
        enable_dark=False,
    )
    assert allocations["VENUE_A"].quantity == 40.0
    assert allocations["VENUE_A"].price == 200.00
    assert allocations["VENUE_B"].quantity == 20.0
    assert allocations["VENUE_B"].price == 200.00


def test_compute_routing_allocation_validations() -> None:
    """Verify boundary checks on compute_routing_allocation inputs."""
    venues = {
        "LIT_1": VenueProfile(
            "LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=100.0,
        ask_price=150.10,
        ask_quantity=100.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (100.0, 100.0)},
    )
    # Empty symbol
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("", OrderSide.BUY, 10.0, quote)

    # Quote type invalid
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", OrderSide.BUY, 10.0, "NOT_A_QUOTE")  # type: ignore[arg-type]

    # Side type invalid
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", "BUY", 10.0, quote)  # type: ignore[arg-type]

    # Quantity non-positive, bool, nan
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", OrderSide.BUY, 0.0, quote)
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", OrderSide.BUY, True, quote)  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", OrderSide.BUY, float("nan"), quote)

    # Boolean flags invalid
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", OrderSide.BUY, 10.0, quote, enable_dark="NO")  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", OrderSide.BUY, 10.0, quote, allow_partial="NO")  # type: ignore[arg-type]

    # now_ns invalid
    with pytest.raises(InvalidSORInputException):
        router.compute_routing_allocation("AAPL", OrderSide.BUY, 10.0, quote, now_ns=-1)


def test_zero_lit_depth_and_no_lit_venues_edge_cases() -> None:
    """Verify edge cases when no lit venues or zero depth exists."""
    # Router with only dark pool
    dark_venues = {
        "DARK_1": VenueProfile("DARK_1", VenueType.DARK_POOL, 0.0, 0.5),
    }
    dark_router = SmartOrderRouter(venues=dark_venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=100.0,
        ask_price=150.10,
        ask_quantity=100.0,
        timestamp_ns=1_000_000,
        venue_depths={},
    )
    # allow_partial=True when no lit venues exist
    assert (
        dark_router.compute_routing_allocation(
            "AAPL", OrderSide.BUY, 10.0, quote, enable_dark=False, allow_partial=True
        )
        == {}
    )
    # allow_partial=False when no lit venues exist
    with pytest.raises(InsufficientLiquidityException):
        dark_router.compute_routing_allocation(
            "AAPL", OrderSide.BUY, 10.0, quote, enable_dark=False, allow_partial=False
        )

    # Router with lit venue reporting 0 depth
    lit_venues = {
        "LIT_1": VenueProfile("LIT_1", VenueType.LIT_EXCHANGE, 0.0, 1.0),
    }
    lit_router = SmartOrderRouter(venues=lit_venues)
    zero_depth_quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=0.0,
        ask_price=150.10,
        ask_quantity=0.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (0.0, 0.0)},
    )
    assert (
        lit_router.compute_routing_allocation(
            "AAPL", OrderSide.BUY, 10.0, zero_depth_quote, enable_dark=False, allow_partial=True
        )
        == {}
    )
    with pytest.raises(InsufficientLiquidityException):
        lit_router.compute_routing_allocation(
            "AAPL", OrderSide.BUY, 10.0, zero_depth_quote, enable_dark=False, allow_partial=False
        )


def test_locked_or_crossed_nbbo_rejection() -> None:
    """Verify locked or crossed NBBO triggers NBBOViolationException."""
    venues = {
        "LIT_1": VenueProfile("LIT_1", VenueType.LIT_EXCHANGE, 0.0, 1.0),
    }
    router = SmartOrderRouter(venues=venues)
    with pytest.raises(NBBOViolationException) as exc_info:
        ConsolidatedQuote(
            symbol="AAPL",
            bid_price=150.10,
            bid_quantity=100.0,
            ask_price=150.00,
            ask_quantity=100.0,
            timestamp_ns=1_000_000,
            venue_depths={"LIT_1": (100.0, 100.0)},
        )
    assert exc_info.value.code == ERR_SOR_NBBO_VIOLATION

    # Also test router's defense against crossed quote
    valid_quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=100.0,
        ask_price=150.10,
        ask_quantity=100.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (100.0, 100.0)},
    )
    object.__setattr__(valid_quote, "bid_price", 150.20)
    with pytest.raises(NBBOViolationException) as exc_info2:
        router.compute_routing_allocation("AAPL", OrderSide.BUY, 10.0, valid_quote)
    assert exc_info2.value.code == ERR_SOR_NBBO_VIOLATION


def test_symbol_mismatch_rejection() -> None:
    """Verify symbol mismatch between order and quote raises InvalidSORInputException."""
    venues = {
        "LIT_1": VenueProfile(
            "LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=100.0,
        ask_price=150.10,
        ask_quantity=100.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (100.0, 100.0)},
    )
    with pytest.raises(InvalidSORInputException) as exc_info:
        router.compute_routing_allocation(
            symbol="GOOG",
            side=OrderSide.BUY,
            quantity=50.0,
            quote=quote,
            enable_dark=False,
        )
    assert exc_info.value.code == ERR_SOR_NON_FINITE_INPUT


def test_dark_pool_min_order_size_bypass() -> None:
    """Verify dark pool with min_order_size > quantity is bypassed in favor of lit routing."""
    venues = {
        "BLOCK_DARK": VenueProfile(
            "BLOCK_DARK",
            VenueType.DARK_POOL,
            maker_fee_bps=0.0,
            taker_fee_bps=0.1,
            min_order_size=500.0,  # Block crossing ATS
        ),
        "LIT_1": VenueProfile(
            "LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )
    # Slicing 100 shares < min_order_size of 500
    allocations = router.compute_routing_allocation(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        enable_dark=True,
    )
    # Should bypass BLOCK_DARK and route directly to LIT_1
    assert "BLOCK_DARK" not in allocations
    assert "LIT_1" in allocations
    assert allocations["LIT_1"].quantity == 100.0


# ============================================================================
# Toxic Markout Watchdog & Quarantine Tests
# ============================================================================


def test_toxic_markout_watchdog_quarantine_and_recovery() -> None:
    """Verify adverse selection triggers quarantine and bypasses dark pool until recovery."""
    venues = {
        "TOXIC_DARK": VenueProfile(
            "TOXIC_DARK", VenueType.DARK_POOL, maker_fee_bps=0.0, taker_fee_bps=0.2
        ),
        "LIT_EXCHANGE": VenueProfile(
            "LIT_EXCHANGE", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.0
        ),
    }
    # Set toxic threshold to -2.0 bps, limit to 2 consecutive fills, quarantine duration to 10s
    router = SmartOrderRouter(
        venues=venues,
        toxic_markout_threshold_bps=-2.0,
        toxic_fill_limit=2,
        quarantine_duration_sec=10.0,
    )
    quote = ConsolidatedQuote(
        symbol="NVDA",
        bid_price=100.00,
        bid_quantity=1000.0,
        ask_price=100.20,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_EXCHANGE": (1000.0, 1000.0)},
    )

    # Initial routing targets TOXIC_DARK
    alloc = router.compute_routing_allocation("NVDA", OrderSide.BUY, 50.0, quote, enable_dark=True)
    assert "TOXIC_DARK" in alloc

    # 1st toxic fill: bought at 100.10, post midpoint dropped to 100.05
    # markout = (100.05 - 100.10) / 100.10 * 10000 = -4.995 bps < -2.0 bps
    router.record_fill_markout(
        venue_id="TOXIC_DARK",
        side=OrderSide.BUY,
        fill_price=100.10,
        post_fill_midpoint=100.05,
        timestamp_ns=10_000_000_000,
    )
    health = router.get_venue_health("TOXIC_DARK", now_ns=10_000_000_000)
    assert not health.is_quarantined
    assert health.consecutive_toxic_fills == 1

    # 2nd toxic fill: triggers quarantine!
    router.record_fill_markout(
        venue_id="TOXIC_DARK",
        side=OrderSide.BUY,
        fill_price=100.10,
        post_fill_midpoint=100.05,
        timestamp_ns=11_000_000_000,
    )
    health = router.get_venue_health("TOXIC_DARK", now_ns=11_000_000_000)
    assert health.is_quarantined
    assert health.quarantine_until_ns == 21_000_000_000

    # Next allocation should bypass quarantined dark pool and route to LIT_EXCHANGE!
    alloc_quarantined = router.compute_routing_allocation(
        "NVDA", OrderSide.BUY, 50.0, quote, enable_dark=True, now_ns=12_000_000_000
    )
    assert "TOXIC_DARK" not in alloc_quarantined
    assert "LIT_EXCHANGE" in alloc_quarantined
    assert alloc_quarantined["LIT_EXCHANGE"].quantity == 50.0

    # Advance time past quarantine expiration: auto-recovery!
    alloc_recovered = router.compute_routing_allocation(
        "NVDA", OrderSide.BUY, 50.0, quote, enable_dark=True, now_ns=22_000_000_000
    )
    assert "TOXIC_DARK" in alloc_recovered


def test_sell_markout_calculation() -> None:
    """Verify SELL order markout calculation formula: (P_fill - P_post) / P_fill * 1e4."""
    venues = {
        "DARK_1": VenueProfile("DARK_1", VenueType.DARK_POOL, maker_fee_bps=0.0, taker_fee_bps=0.5),
    }
    router = SmartOrderRouter(venues=venues)

    # SELL at 100.0, post midpoint rose to 100.10 (adverse selection)
    # markout = (100.0 - 100.10) / 100.0 * 10000 = -10.0 bps
    router.record_fill_markout(
        venue_id="DARK_1",
        side=OrderSide.SELL,
        fill_price=100.0,
        post_fill_midpoint=100.10,
        timestamp_ns=1_000_000,
    )
    health = router.get_venue_health("DARK_1")
    assert round(health.cumulative_markout_bps, 2) == -10.0


def test_record_fill_markout_and_get_health_rejections() -> None:
    """Verify input validation rejections in record_fill_markout and get_venue_health."""
    venues = {
        "DARK_1": VenueProfile("DARK_1", VenueType.DARK_POOL, 0.0, 0.5),
    }
    router = SmartOrderRouter(venues=venues)

    with pytest.raises(InvalidSORInputException):
        router.record_fill_markout("", OrderSide.BUY, 100.0, 100.0, 1_000)
    with pytest.raises(InvalidSORInputException):
        router.record_fill_markout("UNKNOWN_VENUE", OrderSide.BUY, 100.0, 100.0, 1_000)
    with pytest.raises(InvalidSORInputException):
        router.record_fill_markout("DARK_1", "BUY", 100.0, 100.0, 1_000)  # type: ignore[arg-type]
    with pytest.raises(InvalidSORInputException):
        router.record_fill_markout("DARK_1", OrderSide.BUY, -1.0, 100.0, 1_000)
    with pytest.raises(InvalidSORInputException):
        router.record_fill_markout("DARK_1", OrderSide.BUY, 100.0, 0.0, 1_000)
    with pytest.raises(InvalidSORInputException):
        router.record_fill_markout("DARK_1", OrderSide.BUY, 100.0, 100.0, -1)

    with pytest.raises(InvalidSORInputException):
        router.get_venue_health("UNKNOWN_VENUE")


# ============================================================================
# Latency SLA Benchmark Test (< 0.05ms)
# ============================================================================


def test_latency_sla_ten_venues() -> None:
    """Verify routing allocation across 10 venues completes in < 0.05ms (50us)."""
    venues = {
        f"LIT_{i}": VenueProfile(
            f"LIT_{i}",
            VenueType.LIT_EXCHANGE,
            maker_fee_bps=-0.1,
            taker_fee_bps=0.1 * i,
        )
        for i in range(1, 11)
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="SPY",
        bid_price=450.00,
        bid_quantity=10000.0,
        ask_price=450.02,
        ask_quantity=10000.0,
        timestamp_ns=1_000_000,
        venue_depths={f"LIT_{i}": (500.0, 500.0) for i in range(1, 11)},
    )

    # Warm-up JIT and bytecode cache
    for _ in range(50):
        router.compute_routing_allocation(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=2500.0,
            quote=quote,
            enable_dark=False,
        )

    # Measure 100 allocation runs
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        router.compute_routing_allocation(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=2500.0,
            quote=quote,
            enable_dark=False,
        )
    elapsed_sec = time.perf_counter() - start
    avg_latency_ms = (elapsed_sec / n_runs) * 1000.0

    # Rule 4 & INV-SOR-006: Hot-path latency SLA < 0.05ms
    assert avg_latency_ms < 0.05, f"Expected latency < 0.05ms, got {avg_latency_ms:.4f}ms"


# ============================================================================
# Async route_slice Integration with PaperExecutionGateway Tests
# ============================================================================


@pytest.mark.asyncio
async def test_async_route_slice_dark_fill() -> None:
    """Verify async route_slice fills completely in dark pool when marketable."""
    venues = {
        "DARK_ATS": VenueProfile(
            "DARK_ATS", VenueType.DARK_POOL, maker_fee_bps=0.0, taker_fee_bps=0.2
        ),
        "LIT_1": VenueProfile(
            "LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    gateway = PaperExecutionGateway(initial_balance=100_000.0, latency_ms=0.0)
    await gateway.connect()

    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )
    # Set gateway market price to midpoint so dark limit order is immediately marketable
    gateway.set_market_price("AAPL", 150.05)

    reports = await router.route_slice(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        gateway=gateway,
        enable_dark=True,
    )
    assert len(reports) == 1
    assert reports[0].exec_type == OrderState.FILLED
    assert reports[0].cum_quantity == 100.0
    assert reports[0].last_price == 150.05

    await gateway.disconnect()


@pytest.mark.asyncio
async def test_async_route_slice_dark_miss_spills_over_to_lit() -> None:
    """Verify async route_slice probes dark, cancels unfulfilled IOC, and sweeps lit venues."""
    venues = {
        "DARK_ATS": VenueProfile(
            "DARK_ATS", VenueType.DARK_POOL, maker_fee_bps=0.0, taker_fee_bps=0.2
        ),
        "CHEAP_LIT": VenueProfile(
            "CHEAP_LIT", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=0.5
        ),
        "PRICEY_LIT": VenueProfile(
            "PRICEY_LIT", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.5
        ),
    }
    router = SmartOrderRouter(venues=venues)
    gateway = PaperExecutionGateway(initial_balance=100_000.0, latency_ms=0.0)
    await gateway.connect()

    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={
            "CHEAP_LIT": (500.0, 40.0),
            "PRICEY_LIT": (500.0, 200.0),
        },
    )
    # Market price is at ask (150.10), so dark midpoint order at 150.05 will not match immediately
    gateway.set_market_price("AAPL", 150.10)

    reports = await router.route_slice(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        gateway=gateway,
        enable_dark=True,
    )
    # Dark order rested and was cancelled; lit sweep executed 40 on CHEAP_LIT, 60 on PRICEY_LIT
    assert len(reports) == 2
    assert reports[0].cum_quantity == 40.0
    assert reports[0].exec_type == OrderState.FILLED
    assert reports[1].cum_quantity == 60.0
    assert reports[1].exec_type == OrderState.FILLED

    # Check that positions accurately reflect 100 units bought
    positions = await gateway.get_positions()
    assert positions["AAPL"] == 100.0

    await gateway.disconnect()


@pytest.mark.asyncio
async def test_async_route_slice_gateway_disconnected() -> None:
    """Verify route_slice raises ChildOrderFailedException when gateway is disconnected."""
    venues = {
        "LIT_1": VenueProfile(
            "LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=1.0
        ),
    }
    router = SmartOrderRouter(venues=venues)
    gateway = PaperExecutionGateway(initial_balance=100_000.0)
    # Do not connect gateway!

    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )
    with pytest.raises(ChildOrderFailedException) as exc_info:
        await router.route_slice(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100.0,
            quote=quote,
            gateway=gateway,
            enable_dark=False,
        )
    assert exc_info.value.code == ERR_SOR_CHILD_ORDER_FAILED


class FailingGateway:
    """Mock gateway that simulates rejections or transport exceptions."""

    def __init__(self, reject_orders: bool = True, raise_on_submit: bool = False) -> None:
        self.is_connected = True
        self.reject_orders = reject_orders
        self.raise_on_submit = raise_on_submit

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def cancel_order(self, cl_ord_id: str) -> ExecutionReport:
        return ExecutionReport(
            report_id="rep-cancel",
            cl_ord_id=cl_ord_id,
            exchange_order_id="ex-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            exec_type=OrderState.CANCELLED,
            last_quantity=0.0,
            last_price=0.0,
            cum_quantity=0.0,
            leaves_quantity=0.0,
            cum_quote_amount=0.0,
            average_price=0.0,
            fee=0.0,
            timestamp_ns=1_000,
        )

    async def submit_order(self, order: Order) -> ExecutionReport:
        if self.raise_on_submit:
            raise ConnectionResetError("Broker socket dropped")
        if self.reject_orders:
            return ExecutionReport(
                report_id="rep-rej",
                cl_ord_id=order.cl_ord_id,
                exchange_order_id=None,
                symbol=order.symbol,
                side=order.side,
                exec_type=OrderState.REJECTED,
                last_quantity=0.0,
                last_price=0.0,
                cum_quantity=0.0,
                leaves_quantity=0.0,
                cum_quote_amount=0.0,
                average_price=0.0,
                fee=0.0,
                timestamp_ns=1_000,
                text="Order rejected by exchange risk filter",
            )
        return ExecutionReport(
            report_id="rep-fil",
            cl_ord_id=order.cl_ord_id,
            exchange_order_id="ex-1",
            symbol=order.symbol,
            side=order.side,
            exec_type=OrderState.FILLED,
            last_quantity=order.quantity,
            last_price=order.price or 100.0,
            cum_quantity=order.quantity,
            leaves_quantity=0.0,
            cum_quote_amount=(order.price or 100.0) * order.quantity,
            average_price=order.price or 100.0,
            fee=0.0,
            timestamp_ns=1_000,
        )


class PartialDarkGateway:
    """Mock gateway that fills half of dark order and fills lit orders."""

    def __init__(self) -> None:
        self.is_connected = True

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def cancel_order(self, cl_ord_id: str) -> ExecutionReport:
        return ExecutionReport(
            report_id="rep-c",
            cl_ord_id=cl_ord_id,
            exchange_order_id="ex-c",
            symbol="AAPL",
            side=OrderSide.BUY,
            exec_type=OrderState.CANCELLED,
            last_quantity=0.0,
            last_price=0.0,
            cum_quantity=50.0,
            leaves_quantity=0.0,
            cum_quote_amount=7502.5,
            average_price=150.05,
            fee=0.0,
            timestamp_ns=1_000,
        )

    async def submit_order(self, order: Order) -> ExecutionReport:
        if "dark" in order.cl_ord_id:
            # Partial fill
            return ExecutionReport(
                report_id="rep-part",
                cl_ord_id=order.cl_ord_id,
                exchange_order_id="ex-d",
                symbol=order.symbol,
                side=order.side,
                exec_type=OrderState.PARTIALLY_FILLED,
                last_quantity=50.0,
                last_price=150.05,
                cum_quantity=50.0,
                leaves_quantity=50.0,
                cum_quote_amount=7502.5,
                average_price=150.05,
                fee=0.0,
                timestamp_ns=1_000,
            )
        return ExecutionReport(
            report_id="rep-lit",
            cl_ord_id=order.cl_ord_id,
            exchange_order_id="ex-l",
            symbol=order.symbol,
            side=order.side,
            exec_type=OrderState.FILLED,
            last_quantity=order.quantity,
            last_price=150.10,
            cum_quantity=order.quantity,
            leaves_quantity=0.0,
            cum_quote_amount=150.10 * order.quantity,
            average_price=150.10,
            fee=0.0,
            timestamp_ns=1_000,
        )


@pytest.mark.asyncio
async def test_async_route_slice_gateway_failure_and_partial_fill() -> None:
    """Verify route_slice error handling on gateway exceptions, rejections, and partial fills."""
    venues = {
        "DARK_1": VenueProfile("DARK_1", VenueType.DARK_POOL, 0.0, 0.5),
        "LIT_1": VenueProfile("LIT_1", VenueType.LIT_EXCHANGE, 0.0, 1.0),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )

    # 1. Dark child order rejected
    reject_gw = FailingGateway(reject_orders=True)
    with pytest.raises(ChildOrderFailedException) as exc:
        await router.route_slice(
            "AAPL",
            OrderSide.BUY,
            100.0,
            quote,
            reject_gw,
            enable_dark=True,  # type: ignore[arg-type]
        )
    assert exc.value.code == ERR_SOR_CHILD_ORDER_FAILED

    # 2. Dark child order exception
    raise_gw = FailingGateway(raise_on_submit=True)
    with pytest.raises(ChildOrderFailedException) as exc:
        await router.route_slice(
            "AAPL",
            OrderSide.BUY,
            100.0,
            quote,
            raise_gw,
            enable_dark=True,  # type: ignore[arg-type]
        )
    assert exc.value.code == ERR_SOR_CHILD_ORDER_FAILED

    # 3. Lit child order rejected
    with pytest.raises(ChildOrderFailedException) as exc:
        await router.route_slice(
            "AAPL",
            OrderSide.BUY,
            100.0,
            quote,
            reject_gw,
            enable_dark=False,  # type: ignore[arg-type]
        )
    assert exc.value.code == ERR_SOR_CHILD_ORDER_FAILED

    # 4. Lit child order exception
    with pytest.raises(ChildOrderFailedException) as exc:
        await router.route_slice(
            "AAPL",
            OrderSide.BUY,
            100.0,
            quote,
            raise_gw,
            enable_dark=False,  # type: ignore[arg-type]
        )
    assert exc.value.code == ERR_SOR_CHILD_ORDER_FAILED

    # 5. Partial fill on dark pool spills over to lit
    partial_gw = PartialDarkGateway()
    reports = await router.route_slice(
        "AAPL",
        OrderSide.BUY,
        100.0,
        quote,
        partial_gw,
        enable_dark=True,  # type: ignore[arg-type]
    )
    assert len(reports) == 2
    assert reports[0].cum_quantity == 50.0
    assert reports[1].cum_quantity == 50.0
