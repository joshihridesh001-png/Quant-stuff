"""Unit test suite verifying VenueProfile, ConsolidatedQuote, and SOR exception invariants.

Purpose:
    Exhaustively stress-tests microstructural venue configurations, uncrossed NBBO constraints,
    derived quote analytics, non-finite scalar tripwires, and diagnostic fault codes.

Dependencies:
    - pytest: Test execution and parametric fixture framework.
    - math: Non-finite verification.
    - quant.execution.venues: VenueProfile, ConsolidatedQuote, VenueType, and SORError hierarchy.

Invariants Enforced:
    - INV-SOR-002: Non-Worse-Than-NBBO (uncrossed and unlocked market quotes required).
    - Rule 1: Self-explicating test coverage with boundary validation.
    - Rule 2: Zero-execution diagnostic fault code verification.
    - Rule 4: Mandatory adversarial red-teaming (NaN, Inf, booleans, crossed quotes).
"""

from __future__ import annotations

import pytest

from quant.execution.venues import (
    ERR_SOR_ALGORITHM_TIMEOUT,
    ERR_SOR_CHILD_ORDER_FAILED,
    ERR_SOR_INSUFFICIENT_LIQUIDITY,
    ERR_SOR_INVALID_SCHEDULE,
    ERR_SOR_MASS_CONSERVATION_BREACH,
    ERR_SOR_NBBO_VIOLATION,
    ERR_SOR_NON_FINITE_INPUT,
    AlgorithmTimeoutException,
    ChildOrderFailedException,
    ConsolidatedQuote,
    InsufficientLiquidityException,
    InvalidScheduleException,
    InvalidSORInputException,
    MassConservationException,
    NBBOViolationException,
    NonFiniteInputException,
    SORError,
    VenueProfile,
    VenueType,
)

# ============================================================================
# Exception Hierarchy and Fault Code Tests
# ============================================================================


def test_sor_exception_hierarchy() -> None:
    """Verify all SOR exceptions inherit from SORError and provide diagnostic codes."""
    # Functional Purpose: Confirm uniform diagnostic failure code accessibility.
    # Explicit Dependency Tracking: SORError taxonomy.
    # Structural Relationship: Rule 2 diagnostic catalog verification.
    # Defensive Invariant: Subclasses preserve error codes and messages.
    base = SORError("Base error", code="ERR-SOR-000")
    assert isinstance(base, Exception)
    assert base.code == "ERR-SOR-000"
    assert str(base) == "Base error"

    nbbo_exc = NBBOViolationException("Crossed NBBO")
    assert isinstance(nbbo_exc, SORError)
    assert nbbo_exc.code == ERR_SOR_NBBO_VIOLATION
    assert nbbo_exc.code == "ERR-SOR-003"

    liq_exc = InsufficientLiquidityException("No depth")
    assert isinstance(liq_exc, SORError)
    assert liq_exc.code == ERR_SOR_INSUFFICIENT_LIQUIDITY
    assert liq_exc.code == "ERR-SOR-002"

    sched_exc = InvalidScheduleException("Invalid K")
    assert isinstance(sched_exc, SORError)
    assert sched_exc.code == ERR_SOR_INVALID_SCHEDULE
    assert sched_exc.code == "ERR-SOR-001"

    mass_exc = MassConservationException("Mass leak")
    assert isinstance(mass_exc, SORError)
    assert mass_exc.code == ERR_SOR_MASS_CONSERVATION_BREACH
    assert mass_exc.code == "ERR-SOR-004"

    child_exc = ChildOrderFailedException("Child rejected")
    assert isinstance(child_exc, SORError)
    assert child_exc.code == ERR_SOR_CHILD_ORDER_FAILED
    assert child_exc.code == "ERR-SOR-005"

    timeout_exc = AlgorithmTimeoutException("Timeout")
    assert isinstance(timeout_exc, SORError)
    assert timeout_exc.code == ERR_SOR_ALGORITHM_TIMEOUT
    assert timeout_exc.code == "ERR-SOR-006"

    input_exc = InvalidSORInputException("Invalid float")
    assert isinstance(input_exc, SORError)
    assert input_exc.code == ERR_SOR_NON_FINITE_INPUT
    assert input_exc.code == "ERR-SOR-007"

    non_finite_exc = NonFiniteInputException("NaN detected")
    assert isinstance(non_finite_exc, InvalidSORInputException)
    assert isinstance(non_finite_exc, SORError)
    assert non_finite_exc.code == ERR_SOR_NON_FINITE_INPUT


# ============================================================================
# VenueType Enum Tests
# ============================================================================


def test_venue_type_values() -> None:
    """Verify VenueType string enum members and string equality."""
    assert VenueType.LIT_EXCHANGE == "LIT_EXCHANGE"
    assert VenueType.DARK_POOL == "DARK_POOL"
    assert str(VenueType.LIT_EXCHANGE) == "LIT_EXCHANGE"
    assert str(VenueType.DARK_POOL) == "DARK_POOL"


# ============================================================================
# VenueProfile Dataclass Tests
# ============================================================================


def test_venue_profile_creation_and_slots() -> None:
    """Verify nominal creation of VenueProfile with default values and slot enforcement."""
    profile = VenueProfile(
        venue_id="NASDAQ",
        venue_type=VenueType.LIT_EXCHANGE,
        maker_fee_bps=-0.5,
        taker_fee_bps=1.5,
        min_order_size=1.0,
        lot_size=1.0,
        avg_latency_ms=0.75,
        dark_fill_probability=0.35,
    )
    assert profile.venue_id == "NASDAQ"
    assert profile.venue_type == VenueType.LIT_EXCHANGE
    assert profile.maker_fee_bps == -0.5  # Maker rebate
    assert profile.taker_fee_bps == 1.5
    assert profile.min_order_size == 1.0
    assert profile.lot_size == 1.0
    assert profile.avg_latency_ms == 0.75
    assert profile.dark_fill_probability == 0.35
    # Slotted dataclass has no __dict__
    assert not hasattr(profile, "__dict__")


def test_venue_profile_defaults() -> None:
    """Verify default values on VenueProfile."""
    profile = VenueProfile(
        venue_id="DARK_ATS",
        venue_type=VenueType.DARK_POOL,
        maker_fee_bps=0.0,
        taker_fee_bps=0.8,
    )
    assert profile.min_order_size == 1.0
    assert profile.lot_size == 1.0
    assert profile.avg_latency_ms == 1.0
    assert profile.dark_fill_probability == 0.35


@pytest.mark.parametrize(
    "field,value",
    [
        ("venue_id", ""),
        ("venue_id", 123),
        ("venue_type", "NOT_A_VENUE_TYPE"),
        ("maker_fee_bps", True),
        ("maker_fee_bps", float("nan")),
        ("maker_fee_bps", float("inf")),
        ("taker_fee_bps", False),
        ("taker_fee_bps", float("-inf")),
        ("min_order_size", 0.0),
        ("min_order_size", -1.0),
        ("min_order_size", True),
        ("min_order_size", float("nan")),
        ("lot_size", 0.0),
        ("lot_size", -5.0),
        ("lot_size", False),
        ("lot_size", float("inf")),
        ("avg_latency_ms", -0.01),
        ("avg_latency_ms", True),
        ("avg_latency_ms", float("nan")),
        ("dark_fill_probability", -0.01),
        ("dark_fill_probability", 1.01),
        ("dark_fill_probability", True),
        ("dark_fill_probability", float("nan")),
    ],
)
def test_venue_profile_rejections(field: str, value: object) -> None:
    """Verify invalid parameters for VenueProfile trigger InvalidSORInputException."""
    kwargs: dict[str, object] = {
        "venue_id": "IEX",
        "venue_type": VenueType.LIT_EXCHANGE,
        "maker_fee_bps": -0.2,
        "taker_fee_bps": 0.9,
        "min_order_size": 1.0,
        "lot_size": 1.0,
        "avg_latency_ms": 1.2,
        "dark_fill_probability": 0.4,
    }
    kwargs[field] = value

    with pytest.raises(InvalidSORInputException) as exc_info:
        VenueProfile(**kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_SOR_NON_FINITE_INPUT


# ============================================================================
# ConsolidatedQuote Dataclass & NBBO Tests
# ============================================================================


def test_consolidated_quote_nominal_and_properties() -> None:
    """Verify nominal quote construction, derived properties, and slot enforcement."""
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=500.0,
        ask_price=150.05,
        ask_quantity=300.0,
        timestamp_ns=1_700_000_000_000_000,
        venue_depths={
            "NASDAQ": (300.0, 200.0),
            "NYSE": (200.0, 100.0),
        },
    )
    assert not hasattr(quote, "__dict__")
    assert quote.symbol == "AAPL"
    assert quote.bid_price == 150.00
    assert quote.ask_price == 150.05
    assert quote.bid_quantity == 500.0
    assert quote.ask_quantity == 300.0
    assert quote.timestamp_ns == 1_700_000_000_000_000
    assert quote.venue_depths["NASDAQ"] == (300.0, 200.0)

    # Derived properties
    assert quote.midpoint == pytest.approx(150.025)
    assert quote.spread == pytest.approx(0.05)
    expected_spread_bps = (0.05 / 150.025) * 10_000.0
    assert quote.spread_bps == pytest.approx(expected_spread_bps)
    expected_imbalance = (500.0 - 300.0) / 800.0
    assert quote.order_book_imbalance == pytest.approx(expected_imbalance)
    assert quote.total_bid_depth == pytest.approx(500.0)
    assert quote.total_ask_depth == pytest.approx(300.0)


def test_consolidated_quote_zero_quantity_imbalance() -> None:
    """Verify order_book_imbalance returns 0.0 when both bid and ask quantities are zero."""
    quote = ConsolidatedQuote(
        symbol="MSFT",
        bid_price=200.0,
        bid_quantity=0.0,
        ask_price=200.10,
        ask_quantity=0.0,
        timestamp_ns=1_000,
        venue_depths={},
    )
    assert quote.order_book_imbalance == 0.0
    assert quote.total_bid_depth == 0.0
    assert quote.total_ask_depth == 0.0


def test_consolidated_quote_locked_market_rejected() -> None:
    """Locked market (bid_price == ask_price) must be rejected with ERR_SOR_NBBO_VIOLATION."""
    with pytest.raises(NBBOViolationException) as exc_info:
        ConsolidatedQuote(
            symbol="NVDA",
            bid_price=100.00,
            bid_quantity=100.0,
            ask_price=100.00,  # Locked!
            ask_quantity=100.0,
            timestamp_ns=1_000,
            venue_depths={"NASDAQ": (100.0, 100.0)},
        )
    assert exc_info.value.code == ERR_SOR_NBBO_VIOLATION
    assert "Locked or crossed market" in str(exc_info.value)


def test_consolidated_quote_crossed_market_rejected() -> None:
    """Crossed market (bid_price > ask_price) must be rejected with ERR_SOR_NBBO_VIOLATION."""
    with pytest.raises(NBBOViolationException) as exc_info:
        ConsolidatedQuote(
            symbol="NVDA",
            bid_price=100.05,
            bid_quantity=100.0,
            ask_price=100.00,  # Crossed!
            ask_quantity=100.0,
            timestamp_ns=1_000,
            venue_depths={"NASDAQ": (100.0, 100.0)},
        )
    assert exc_info.value.code == ERR_SOR_NBBO_VIOLATION
    assert "Locked or crossed market" in str(exc_info.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", ""),
        ("symbol", None),
        ("symbol", 123),
        ("timestamp_ns", -1),
        ("timestamp_ns", True),
        ("timestamp_ns", 1.5),
        ("bid_price", 0.0),
        ("bid_price", -10.0),
        ("bid_price", True),
        ("bid_price", float("nan")),
        ("bid_price", float("inf")),
        ("ask_price", 0.0),
        ("ask_price", -10.0),
        ("ask_price", False),
        ("ask_price", float("nan")),
        ("ask_price", float("inf")),
        ("bid_quantity", -1.0),
        ("bid_quantity", True),
        ("bid_quantity", float("nan")),
        ("bid_quantity", float("inf")),
        ("ask_quantity", -0.01),
        ("ask_quantity", False),
        ("ask_quantity", float("nan")),
        ("ask_quantity", float("inf")),
        ("venue_depths", "not_a_dict"),
        ("venue_depths", {123: (10.0, 10.0)}),
        ("venue_depths", {"NASDAQ": (True, 10.0)}),
        ("venue_depths", {"NASDAQ": (10.0, -1.0)}),
        ("venue_depths", {"NASDAQ": (float("nan"), 10.0)}),
        ("venue_depths", {"NASDAQ": (10.0, float("inf"))}),
        ("venue_depths", {"NASDAQ": (10.0,)}),  # Invalid tuple length
    ],
)
def test_consolidated_quote_invalid_inputs(field: str, value: object) -> None:
    """Verify invalid scalar, bool, or non-finite inputs in ConsolidatedQuote raise InvalidSORInputException."""
    base_kwargs: dict[str, object] = {
        "symbol": "GOOGL",
        "bid_price": 100.0,
        "bid_quantity": 50.0,
        "ask_price": 100.1,
        "ask_quantity": 60.0,
        "timestamp_ns": 1_000_000,
        "venue_depths": {"NASDAQ": (50.0, 60.0)},
    }
    base_kwargs[field] = value

    with pytest.raises(InvalidSORInputException) as exc_info:
        ConsolidatedQuote(**base_kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_SOR_NON_FINITE_INPUT
