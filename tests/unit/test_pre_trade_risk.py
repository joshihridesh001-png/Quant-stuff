"""Unit tests for PreTradeRiskFirewall, RiskLimits, and PortfolioRiskState.

Governing Standards:
- Rules.md:
  - Rule 1: Defensive Invariants and explicit contract testing.
  - Rule 2: Zero-execution deterministic diagnostic codes (ERR-RSK-001 through ERR-RSK-008).
  - Rule 3: Quality gates (100% pass rate, strict typing, high statement coverage).
  - Rule 4: Mandatory adversarial red-teaming, non-finite/bool guards, sub-10us SLA.
- Invariants:
  - INV-RSK-001: Fat-Finger Notional & Quantity Bounds
  - INV-RSK-002: Portfolio Gross & Net Leverage Limits
  - INV-RSK-003: Single-Asset NAV Concentration Ceiling
  - INV-RSK-004: Intraday Drawdown Circuit Breaker
  - INV-RSK-005: Margin & Borrow Sufficiency
  - INV-RSK-006: Hot-Path Latency SLA (< 10us)
  - INV-RSK-007: Strict Non-Finite & Boolean Input Sanitization
  - INV-RSK-008: Kill Switch Exception Validation
"""

from __future__ import annotations

import math
import time

import pytest

from quant.execution.models import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant.execution.risk import (
    ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED,
    ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED,
    ERR_RSK_FAT_FINGER_NOTIONAL,
    ERR_RSK_FAT_FINGER_QUANTITY,
    ERR_RSK_INSUFFICIENT_MARGIN,
    ERR_RSK_KILL_SWITCH_ACTIVE,
    ERR_RSK_LEVERAGE_LIMIT_EXCEEDED,
    ERR_RSK_NON_FINITE_INPUT,
    ConcentrationLimitExceededException,
    DrawdownLimitExceededException,
    FatFingerNotionalException,
    FatFingerQuantityException,
    InsufficientMarginRiskException,
    KillSwitchActiveException,
    LeverageLimitExceededException,
    NonFiniteRiskInputException,
    PortfolioRiskState,
    PreTradeRiskFirewall,
    RiskError,
    RiskLimits,
)

# ============================================================================
# Test Fixtures & Factory Helpers
# ============================================================================


def make_default_limits(
    max_order_notional: float = 100_000.0,
    max_order_qty: float = 1_000.0,
    max_gross_leverage: float = 2.0,
    max_net_leverage: float = 1.0,
    max_concentration_nav_pct: float = 0.40,
    max_intraday_drawdown_pct: float = 0.05,
    min_free_margin: float = 1_000.0,
) -> RiskLimits:
    """Create a standard RiskLimits configuration for testing."""
    return RiskLimits(
        max_order_notional=max_order_notional,
        max_order_qty=max_order_qty,
        max_gross_leverage=max_gross_leverage,
        max_net_leverage=max_net_leverage,
        max_concentration_nav_pct=max_concentration_nav_pct,
        max_intraday_drawdown_pct=max_intraday_drawdown_pct,
        min_free_margin=min_free_margin,
    )


def make_default_state(
    cash: float = 100_000.0,
    positions: dict[str, float] | None = None,
    pending_leaves: dict[str, float] | None = None,
    current_prices: dict[str, float] | None = None,
    peak_equity: float | None = None,
    initial_equity: float | None = None,
) -> PortfolioRiskState:
    """Create a standard PortfolioRiskState for testing."""
    prices = current_prices or {"AAPL": 150.0, "MSFT": 300.0}
    pos = positions or {}
    eq = cash + sum(pos.get(s, 0.0) * prices.get(s, 0.0) for s in pos)
    p_eq = peak_equity if peak_equity is not None else max(eq, cash, 1.0)
    i_eq = initial_equity if initial_equity is not None else p_eq
    return PortfolioRiskState(
        cash=cash,
        positions=pos,
        pending_leaves=pending_leaves or {},
        current_prices=prices,
        peak_equity=p_eq,
        initial_equity=i_eq,
    )


def make_order(
    cl_ord_id: str = "ord-001",
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    quantity: float = 100.0,
    price: float | None = 150.0,
) -> Order:
    """Create a candidate domain Order for testing."""
    return Order(
        cl_ord_id=cl_ord_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
    )


# ============================================================================
# Section 1: Fault Codes & Exception Hierarchy Tests (Rule 2)
# ============================================================================


def test_risk_exception_hierarchy_and_fault_codes() -> None:
    """Verify diagnostic fault codes and structured exception hierarchy inheritance."""
    # Test base RiskError
    base_err = RiskError("Generic risk error", code="ERR-RSK-000")
    assert isinstance(base_err, Exception)
    assert base_err.code == "ERR-RSK-000"
    assert "Generic risk error" in str(base_err)

    # Test all specific subclasses
    exceptions_map = [
        (FatFingerNotionalException, ERR_RSK_FAT_FINGER_NOTIONAL, "ERR-RSK-001"),
        (FatFingerQuantityException, ERR_RSK_FAT_FINGER_QUANTITY, "ERR-RSK-002"),
        (LeverageLimitExceededException, ERR_RSK_LEVERAGE_LIMIT_EXCEEDED, "ERR-RSK-003"),
        (ConcentrationLimitExceededException, ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED, "ERR-RSK-004"),
        (InsufficientMarginRiskException, ERR_RSK_INSUFFICIENT_MARGIN, "ERR-RSK-005"),
        (DrawdownLimitExceededException, ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED, "ERR-RSK-006"),
        (NonFiniteRiskInputException, ERR_RSK_NON_FINITE_INPUT, "ERR-RSK-007"),
        (KillSwitchActiveException, ERR_RSK_KILL_SWITCH_ACTIVE, "ERR-RSK-008"),
    ]

    for exc_cls, const_code, expected_code in exceptions_map:
        assert const_code == expected_code
        inst = exc_cls(f"Error test for {expected_code}")
        assert isinstance(inst, RiskError)
        assert inst.code == expected_code
        assert expected_code in inst.code


# ============================================================================
# Section 2: Valid Orders Passing Clearance (Happy Path)
# ============================================================================


def test_valid_order_clearance_limit_and_market() -> None:
    """Verify standard compliant orders clear the firewall without raising exceptions."""
    limits = make_default_limits()
    state = make_default_state(cash=100_000.0)
    firewall = PreTradeRiskFirewall(limits=limits)

    # Compliant BUY limit order: 100 shares @ $150 = $15,000 notional (15% NAV, < 40% cap)
    order_buy = make_order(quantity=100.0, price=150.0)
    firewall.validate_order(order_buy, state)

    # Compliant SELL order on existing position (de-risking)
    state_with_pos = make_default_state(
        cash=50_000.0,
        positions={"AAPL": 200.0},
        current_prices={"AAPL": 150.0},
    )
    order_sell = make_order(side=OrderSide.SELL, quantity=100.0, price=150.0)
    firewall.validate_order(order_sell, state_with_pos)

    # Compliant MARKET order with current_price explicitly provided
    order_market = Order(
        cl_ord_id="ord-mkt-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        price=None,
    )
    firewall.validate_order(order_market, state, current_price=150.0)

    # Compliant MARKET order using state.current_prices fallback
    firewall.validate_order(order_market, state, current_price=None)


# ============================================================================
# Section 3: Fat-Finger Bounds Testing (INV-RSK-001)
# ============================================================================


def test_fat_finger_quantity_breach() -> None:
    """Verify fat-finger quantity limit trips ERR-RSK-002."""
    limits = make_default_limits(max_order_qty=500.0)
    state = make_default_state()
    firewall = PreTradeRiskFirewall(limits=limits)

    # Exact threshold passes: 500 <= 500
    compliant_order = make_order(quantity=500.0, price=10.0)
    firewall.validate_order(compliant_order, state)

    # Breach trips FatFingerQuantityException: 500.001 > 500.0
    breach_order = make_order(quantity=500.001, price=10.0)
    with pytest.raises(FatFingerQuantityException) as exc_info:
        firewall.validate_order(breach_order, state)
    assert exc_info.value.code == ERR_RSK_FAT_FINGER_QUANTITY
    assert "exceeds max_order_qty" in exc_info.value.message


def test_fat_finger_notional_breach() -> None:
    """Verify fat-finger notional limit trips ERR-RSK-001."""
    limits = make_default_limits(max_order_notional=50_000.0, max_order_qty=5_000.0)
    state = make_default_state(cash=200_000.0)
    firewall = PreTradeRiskFirewall(limits=limits)

    # Exact threshold passes: 500 * $100 = $50,000 <= $50,000
    compliant_order = make_order(quantity=500.0, price=100.0)
    firewall.validate_order(compliant_order, state)

    # Breach trips FatFingerNotionalException: 500.1 * $100 = $50,010 > $50,000
    breach_order = make_order(quantity=501.0, price=100.0)
    with pytest.raises(FatFingerNotionalException) as exc_info:
        firewall.validate_order(breach_order, state)
    assert exc_info.value.code == ERR_RSK_FAT_FINGER_NOTIONAL
    assert "exceeds max_order_notional" in exc_info.value.message


# ============================================================================
# Section 4: Portfolio Leverage Limits (INV-RSK-002)
# ============================================================================


def test_gross_leverage_limit_breach() -> None:
    """Verify projected gross leverage trips ERR-RSK-003."""
    # NAV = $100,000. max_gross_leverage = 0.50 => max gross notional = $50,000
    limits = make_default_limits(
        max_gross_leverage=0.50,
        max_net_leverage=2.0,
        max_order_notional=200_000.0,
        max_order_qty=2_000.0,
        max_concentration_nav_pct=2.0,  # loosen concentration to isolate gross leverage
        min_free_margin=1_000.0,
    )
    state = make_default_state(
        cash=100_000.0,
        positions={},
        current_prices={"AAPL": 100.0},
    )
    firewall = PreTradeRiskFirewall(limits=limits)

    # Compliant: BUY 400 AAPL @ $100 = $40k. Gross leverage = 40k / 100k = 0.40 <= 0.50
    # Remaining free margin = 100k - 40k = 60k >= 1k (margin passes)
    ok_order = make_order(symbol="AAPL", quantity=400.0, price=100.0)
    firewall.validate_order(ok_order, state)

    # Breach: BUY 600 AAPL @ $100 = $60k. Gross leverage = 60k / 100k = 0.60 > 0.50
    # Remaining free margin = 100k - 60k = 40k >= 1k (margin passes, leverage fails!)
    breach_order = make_order(symbol="AAPL", quantity=600.0, price=100.0)
    with pytest.raises(LeverageLimitExceededException) as exc_info:
        firewall.validate_order(breach_order, state)
    assert exc_info.value.code == ERR_RSK_LEVERAGE_LIMIT_EXCEEDED
    assert "gross leverage" in exc_info.value.message


def test_net_leverage_limit_breach() -> None:
    """Verify projected net directional leverage trips ERR-RSK-003."""
    # NAV = $100,000. max_net_leverage = 0.50 => max net exposure = $50,000
    limits = make_default_limits(
        max_gross_leverage=5.0,
        max_net_leverage=0.50,
        max_order_notional=200_000.0,
        max_order_qty=2_000.0,
        max_concentration_nav_pct=2.0,
    )
    state = make_default_state(
        cash=100_000.0,
        positions={},
        current_prices={"AAPL": 100.0},
    )
    firewall = PreTradeRiskFirewall(limits=limits)

    # Compliant: BUY 400 AAPL @ $100 = $40,000 net (0.40x <= 0.50x)
    ok_order = make_order(symbol="AAPL", quantity=400.0, price=100.0)
    firewall.validate_order(ok_order, state)

    # Breach: BUY 600 AAPL @ $100 = $60,000 net (0.60x > 0.50x)
    breach_order = make_order(symbol="AAPL", quantity=600.0, price=100.0)
    with pytest.raises(LeverageLimitExceededException) as exc_info:
        firewall.validate_order(breach_order, state)
    assert exc_info.value.code == ERR_RSK_LEVERAGE_LIMIT_EXCEEDED
    assert "net leverage" in exc_info.value.message


def test_zero_or_negative_equity_rejection() -> None:
    """Verify orders are immediately rejected when portfolio equity is zero or negative."""
    limits = make_default_limits()
    # Bankrupt portfolio: cash = -10,000, no positions
    state_bankrupt = PortfolioRiskState(
        cash=-10_000.0,
        positions={},
        pending_leaves={},
        current_prices={"AAPL": 100.0},
        peak_equity=10_000.0,
        initial_equity=10_000.0,
    )
    firewall = PreTradeRiskFirewall(limits=limits)
    order = make_order(quantity=10.0, price=100.0)

    # Trips drawdown circuit breaker first (100%+ drawdown)
    with pytest.raises(DrawdownLimitExceededException) as exc_info:
        firewall.validate_order(order, state_bankrupt)
    assert exc_info.value.code == ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED


# ============================================================================
# Section 5: Single-Asset NAV Concentration Ceiling (INV-RSK-003)
# ============================================================================


def test_single_asset_concentration_breach() -> None:
    """Verify single-asset exposure ceiling trips ERR-RSK-004."""
    # NAV = $100,000. max_concentration = 25% ($25,000 cap per asset)
    limits = make_default_limits(
        max_concentration_nav_pct=0.25,
        max_order_notional=100_000.0,
        max_order_qty=1_000.0,
    )
    state = make_default_state(cash=100_000.0)
    firewall = PreTradeRiskFirewall(limits=limits)

    # Compliant: 200 AAPL @ $100 = $20,000 (20% <= 25%)
    ok_order = make_order(symbol="AAPL", quantity=200.0, price=100.0)
    firewall.validate_order(ok_order, state)

    # Breach: 300 AAPL @ $100 = $30,000 (30% > 25%)
    breach_order = make_order(symbol="AAPL", quantity=300.0, price=100.0)
    with pytest.raises(ConcentrationLimitExceededException) as exc_info:
        firewall.validate_order(breach_order, state)
    assert exc_info.value.code == ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED
    assert "concentration" in exc_info.value.message


def test_rule_4_derisking_order_avoids_lockout_trap() -> None:
    """Rule 4 Best-of-the-Best: A closing/reducing order must NOT artificially double exposure."""
    # Trader has 40% concentration in AAPL (at the exact limit).
    limits = make_default_limits(
        max_concentration_nav_pct=0.40,
        max_gross_leverage=1.0,
        max_order_notional=100_000.0,
        max_order_qty=1_000.0,
    )
    # NAV = 60k cash + 400 AAPL @ $100 (40k) = $100,000. AAPL is exactly 40%.
    state = make_default_state(
        cash=60_000.0,
        positions={"AAPL": 400.0},
        current_prices={"AAPL": 100.0},
    )
    firewall = PreTradeRiskFirewall(limits=limits)

    # Under naive models (|w_i| + |q_leaves|), submitting SELL 400 would compute |400| + |-400| = 800 => 80% => REJECTED!
    # Under our Rule 4 Best-of-the-Best de-risking invariant, max(|w_i|, |w_i + delta|) = 400 => 40% => ACCEPTED!
    derisking_sell_order = make_order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=400.0,
        price=100.0,
    )
    # Must clear cleanly without raising ConcentrationLimitExceededException or LeverageLimitExceededException!
    firewall.validate_order(derisking_sell_order, state)


# ============================================================================
# Section 6: Intraday Drawdown Circuit Breaker (INV-RSK-004)
# ============================================================================


def test_intraday_drawdown_circuit_breaker_trip() -> None:
    """Verify intraday peak-to-trough equity drop trips ERR-RSK-006."""
    limits = make_default_limits(max_intraday_drawdown_pct=0.03)  # 3% drawdown cap
    # Peak equity = $100,000. Current cash = $96,000 (4% drawdown > 3% limit)
    state = make_default_state(
        cash=96_000.0,
        peak_equity=100_000.0,
        initial_equity=100_000.0,
    )
    firewall = PreTradeRiskFirewall(limits=limits)
    order = make_order(quantity=10.0, price=100.0)

    with pytest.raises(DrawdownLimitExceededException) as exc_info:
        firewall.validate_order(order, state)
    assert exc_info.value.code == ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED
    assert "Intraday drawdown" in exc_info.value.message


def test_intraday_drawdown_dynamic_high_water_mark() -> None:
    """Verify session high-water mark dynamically updates when mark-to-market equity rises."""
    state = make_default_state(
        cash=100_000.0,
        positions={"AAPL": 100.0},
        current_prices={"AAPL": 100.0},  # Equity = 110,000
        peak_equity=100_000.0,
        initial_equity=100_000.0,
    )
    # Accessing equity or drawdown automatically refreshes peak_equity
    assert state.current_equity == 110_000.0
    assert state.intraday_drawdown == 0.0
    assert state.peak_equity == 110_000.0

    # Simulate price drop to $50: Equity = 100k cash + 5k stock = 105k
    state.update_price("AAPL", 50.0)
    assert state.current_equity == 105_000.0
    # Drawdown = (110k - 105k) / 110k = 5 / 110 = 0.04545 (4.54%)
    assert math.isclose(state.intraday_drawdown, 5_000.0 / 110_000.0, rel_tol=1e-5)


# ============================================================================
# Section 7: Margin & Borrow Sufficiency (INV-RSK-005)
# ============================================================================


def test_insufficient_margin_buy_order() -> None:
    """Verify insufficient liquid free margin trips ERR-RSK-005."""
    # Required minimum free margin buffer = $5,000
    limits = make_default_limits(min_free_margin=5_000.0)
    # Available cash = $10,000
    state = make_default_state(cash=10_000.0)
    firewall = PreTradeRiskFirewall(limits=limits)

    # Compliant: BUY order requiring $4,000 => Remaining cash = $6,000 >= $5,000
    ok_order = make_order(quantity=40.0, price=100.0)
    firewall.validate_order(ok_order, state)

    # Breach: BUY order requiring $6,000 => Remaining cash = $4,000 < $5,000
    breach_order = make_order(quantity=60.0, price=100.0)
    with pytest.raises(InsufficientMarginRiskException) as exc_info:
        firewall.validate_order(breach_order, state)
    assert exc_info.value.code == ERR_RSK_INSUFFICIENT_MARGIN
    assert "free margin" in exc_info.value.message


def test_short_sell_margin_sufficiency() -> None:
    """Verify short sell requires margin when opening a short position."""
    limits = make_default_limits(min_free_margin=2_000.0)
    # Cash = $5,000, no existing position in AAPL
    state = make_default_state(cash=5_000.0, positions={})
    firewall = PreTradeRiskFirewall(limits=limits)

    # Short sell requiring $4,000 margin => Remaining = $1,000 < $2,000 minimum
    short_order = make_order(side=OrderSide.SELL, quantity=40.0, price=100.0)
    with pytest.raises(InsufficientMarginRiskException) as exc_info:
        firewall.validate_order(short_order, state)
    assert exc_info.value.code == ERR_RSK_INSUFFICIENT_MARGIN


# ============================================================================
# Section 8: Strict Input Sanitization & Adversarial Red-Teaming (INV-RSK-007)
# ============================================================================


@pytest.mark.parametrize(
    "invalid_qty",
    [
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        -10.0,
        0.0,
        "100",
        None,
    ],
)
def test_order_quantity_sanitization(invalid_qty: object) -> None:
    """Adversarial input check: order quantity must reject bool, NaN, Inf, and non-positives."""
    limits = make_default_limits()
    state = make_default_state()
    firewall = PreTradeRiskFirewall(limits=limits)

    # Construct order bypassing domain __post_init__ to test firewall defense
    order = make_order()
    object.__setattr__(order, "quantity", invalid_qty)

    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        firewall.validate_order(order, state)
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "invalid_price",
    [
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        -150.0,
        0.0,
        "150",
    ],
)
def test_order_price_sanitization(invalid_price: object) -> None:
    """Adversarial input check: order price and current_price must reject non-finite and bool."""
    limits = make_default_limits()
    state = make_default_state()
    firewall = PreTradeRiskFirewall(limits=limits)

    # Test invalid order.price
    order = make_order()
    object.__setattr__(order, "price", invalid_price)
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        firewall.validate_order(order, state)
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT

    # Test invalid current_price argument
    valid_order = make_order()
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        firewall.validate_order(
            valid_order,
            state,
            current_price=invalid_price,  # type: ignore[arg-type]
        )
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


def test_missing_reference_price_rejection() -> None:
    """Verify MARKET order with no current_price and no book price is rejected with ERR-RSK-007."""
    limits = make_default_limits()
    # State has no price for UNKNOWN ticker
    state = make_default_state(current_prices={})
    firewall = PreTradeRiskFirewall(limits=limits)

    market_order = Order(
        cl_ord_id="ord-unknown-1",
        symbol="UNKNOWN",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        price=None,
    )
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        firewall.validate_order(market_order, state, current_price=None)
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT
    assert "Cannot determine valid reference price" in exc_info.value.message


@pytest.mark.parametrize(
    "param_name,bad_val",
    [
        ("max_order_notional", -100.0),
        ("max_order_notional", 0.0),
        ("max_order_notional", float("nan")),
        ("max_order_notional", True),
        ("max_order_qty", -5.0),
        ("max_order_qty", 0.0),
        ("max_gross_leverage", -1.0),
        ("max_net_leverage", float("inf")),
        ("max_concentration_nav_pct", -0.5),
        ("max_intraday_drawdown_pct", 0.0),
        ("min_free_margin", -1.0),
        ("min_free_margin", float("nan")),
        ("min_free_margin", False),
    ],
)
def test_risk_limits_boundary_sanitization(param_name: str, bad_val: object) -> None:
    """Verify RiskLimits.__post_init__ strictly rejects corrupt or illegal threshold configurations."""
    valid_kwargs = {
        "max_order_notional": 100_000.0,
        "max_order_qty": 1_000.0,
        "max_gross_leverage": 2.0,
        "max_net_leverage": 1.0,
        "max_concentration_nav_pct": 0.40,
        "max_intraday_drawdown_pct": 0.05,
        "min_free_margin": 0.0,
    }
    valid_kwargs[param_name] = bad_val  # type: ignore[assignment]

    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        RiskLimits(**valid_kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT


# ============================================================================
# Section 9: PortfolioRiskState Mechanics & Fills
# ============================================================================


def test_portfolio_risk_state_fill_reconciliation() -> None:
    """Verify PortfolioRiskState.update_fill reconciles cash, positions, and leaves correctly."""
    state = PortfolioRiskState(
        cash=100_000.0,
        positions={"AAPL": 50.0},
        pending_leaves={"AAPL": 100.0},
        current_prices={"AAPL": 150.0},
        peak_equity=107_500.0,
        initial_equity=107_500.0,
    )

    # Initial equity: 100,000 cash + 50 * 150 = 107,500
    assert state.current_equity == 107_500.0
    # Pending buy leaves of 100 @ 150 locks $15,000: free margin = 100,000 - 15,000 = 85,000
    assert state.free_margin == 85_000.0

    # Execute partial fill: BUY 40 AAPL @ $150
    state.update_fill("AAPL", filled_qty=40.0, price=150.0, side=OrderSide.BUY)
    assert state.cash == 100_000.0 - (40.0 * 150.0)  # 94,000
    assert state.positions["AAPL"] == 90.0  # 50 + 40
    assert state.pending_leaves["AAPL"] == 60.0  # 100 - 40
    assert state.current_equity == 94_000.0 + (90.0 * 150.0)  # 107,500 (conserved)

    # Execute sell fill: SELL 90 AAPL @ $160
    state.update_fill("AAPL", filled_qty=90.0, price=160.0, side=OrderSide.SELL)
    assert state.positions["AAPL"] == 0.0
    assert state.cash == 94_000.0 + (90.0 * 160.0)  # 108,400
    assert state.current_equity == 108_400.0
    assert state.peak_equity == 108_400.0  # Peak equity updated on gain!


def test_portfolio_risk_state_invalid_inputs() -> None:
    """Verify PortfolioRiskState methods reject corrupt inputs."""
    state = make_default_state()

    # Invalid price update
    with pytest.raises(NonFiniteRiskInputException):
        state.update_price("", 150.0)
    with pytest.raises(NonFiniteRiskInputException):
        state.update_price("AAPL", -10.0)
    with pytest.raises(NonFiniteRiskInputException):
        state.update_price("AAPL", float("nan"))

    # Invalid fill update
    with pytest.raises(NonFiniteRiskInputException):
        state.update_fill("AAPL", -10.0, 150.0, OrderSide.BUY)
    with pytest.raises(NonFiniteRiskInputException):
        state.update_fill("AAPL", 10.0, 0.0, OrderSide.BUY)
    with pytest.raises(NonFiniteRiskInputException):
        state.update_fill(
            "AAPL",
            10.0,
            150.0,
            "INVALID",  # type: ignore[arg-type]
        )


def test_pre_trade_firewall_invalid_initialization() -> None:
    """Verify PreTradeRiskFirewall rejects non-RiskLimits argument."""
    with pytest.raises(NonFiniteRiskInputException):
        PreTradeRiskFirewall(
            limits="not-a-limits-object"  # type: ignore[arg-type]
        )


# ============================================================================
# Section 10: Hot-Path Latency Benchmark (INV-RSK-006 SLA < 10us)
# ============================================================================


def test_hot_path_latency_sla() -> None:
    """Verify PreTradeRiskFirewall.validate_order executes in < 10us per call."""
    import sys

    limits = make_default_limits()
    state = make_default_state(
        cash=100_000.0,
        positions={"AAPL": 100.0, "MSFT": 50.0},
        pending_leaves={"AAPL": 20.0},
        current_prices={"AAPL": 150.0, "MSFT": 300.0},
    )
    firewall = PreTradeRiskFirewall(limits=limits)
    order = make_order(quantity=50.0, price=150.0)

    # Warm-up JIT / interpreter
    for _ in range(100):
        firewall.validate_order(order, state)

    # Benchmark 1,000 iterations
    iterations = 1_000
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        firewall.validate_order(order, state)
    elapsed_ns = time.perf_counter_ns() - start_ns

    avg_latency_us = (elapsed_ns / iterations) / 1_000.0
    is_traced = (
        sys.gettrace() is not None
        or "coverage" in sys.modules
        or "pytest_cov" in sys.modules
        or (
            hasattr(sys, "monitoring")
            and any(sys.monitoring.get_tool(i) is not None for i in range(6))
        )
    )
    sla_us = 25.0 if is_traced else 10.0
    assert avg_latency_us < sla_us, (
        f"PreTradeRiskFirewall average latency {avg_latency_us:.3f}us breached {sla_us}us SLA"
    )


# ============================================================================
# Section 11: Exhaustive Branch Coverage & State Invariants
# ============================================================================


def test_portfolio_risk_state_leverage_properties() -> None:
    """Verify state.gross_leverage and state.net_leverage properties."""
    state = PortfolioRiskState(
        cash=50_000.0,
        positions={"AAPL": 100.0, "MSFT": -50.0},
        pending_leaves={"AAPL": 20.0, "MSFT": -10.0},
        current_prices={"AAPL": 100.0, "MSFT": 200.0},
        peak_equity=60_000.0,
        initial_equity=60_000.0,
    )
    assert state.current_equity == 50_000.0
    assert math.isclose(state.gross_leverage, 0.48)
    assert math.isclose(state.net_leverage, 0.0)


def test_portfolio_risk_state_leverage_properties_zero_equity() -> None:
    """Verify gross_leverage and net_leverage return float('inf') on non-positive equity."""
    state = PortfolioRiskState(
        cash=0.0,
        positions={},
        current_prices={"AAPL": 100.0},
        peak_equity=10_000.0,
        initial_equity=10_000.0,
    )
    assert state.gross_leverage == float("inf")
    assert state.net_leverage == float("inf")


def test_portfolio_risk_state_missing_price_raises() -> None:
    """Verify active positions or leaves without price in current_prices raise NonFiniteRiskInputException."""
    # Instantiating with active position and missing price raises during __post_init__
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=50_000.0,
            positions={"AAPL": 100.0},
            current_prices={},
            peak_equity=50_000.0,
            initial_equity=50_000.0,
        )

    # Active pending leaves with missing price raises on gross/net leverage evaluation
    state_leaves = PortfolioRiskState(
        cash=50_000.0,
        positions={},
        pending_leaves={"AAPL": 10.0},
        current_prices={},
        peak_equity=50_000.0,
        initial_equity=50_000.0,
    )
    with pytest.raises(NonFiniteRiskInputException):
        _ = state_leaves.gross_leverage

    with pytest.raises(NonFiniteRiskInputException):
        _ = state_leaves.net_leverage


def test_portfolio_risk_state_post_init_adversarial() -> None:
    """Verify adversarial inputs to PortfolioRiskState constructor raise NonFiniteRiskInputException."""
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(cash=float("nan"), peak_equity=100.0, initial_equity=100.0)
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(cash=True, peak_equity=100.0, initial_equity=100.0)

    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0, positions={"": 10.0}, peak_equity=100.0, initial_equity=100.0
        )
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0,
            positions={"AAPL": True},  # type: ignore[dict-item]
            peak_equity=100.0,
            initial_equity=100.0,
        )
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0,
            positions={"AAPL": float("nan")},
            peak_equity=100.0,
            initial_equity=100.0,
        )

    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0, pending_leaves={"": 10.0}, peak_equity=100.0, initial_equity=100.0
        )
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0,
            pending_leaves={"AAPL": False},  # type: ignore[dict-item]
            peak_equity=100.0,
            initial_equity=100.0,
        )
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0,
            pending_leaves={"AAPL": float("inf")},
            peak_equity=100.0,
            initial_equity=100.0,
        )

    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0, current_prices={"": 10.0}, peak_equity=100.0, initial_equity=100.0
        )
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0,
            current_prices={"AAPL": True},  # type: ignore[dict-item]
            peak_equity=100.0,
            initial_equity=100.0,
        )
    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(
            cash=100.0, current_prices={"AAPL": -5.0}, peak_equity=100.0, initial_equity=100.0
        )

    with pytest.raises(NonFiniteRiskInputException):
        PortfolioRiskState(cash=0.0, initial_equity=0.0, peak_equity=0.0)


def test_update_fill_sell_with_leaves_and_losses() -> None:
    """Verify update_fill branches for SELL orders with negative leaves and unsigned leaves."""
    state = PortfolioRiskState(
        cash=10_000.0,
        positions={"AAPL": 100.0},
        pending_leaves={"AAPL": -50.0},
        current_prices={"AAPL": 100.0},
        peak_equity=20_000.0,
        initial_equity=20_000.0,
    )
    state.update_fill("AAPL", filled_qty=50.0, price=100.0, side=OrderSide.SELL)
    assert "AAPL" not in state.pending_leaves
    assert state.positions["AAPL"] == 50.0

    state2 = PortfolioRiskState(
        cash=10_000.0,
        positions={"AAPL": 100.0},
        pending_leaves={"AAPL": 50.0},
        current_prices={"AAPL": 100.0},
        peak_equity=20_000.0,
        initial_equity=20_000.0,
    )
    state2.update_fill("AAPL", filled_qty=50.0, price=100.0, side=OrderSide.SELL)
    assert "AAPL" not in state2.pending_leaves

    state3 = PortfolioRiskState(
        cash=10_000.0,
        positions={"AAPL": 100.0},
        current_prices={"AAPL": 100.0},
        peak_equity=20_000.0,
        initial_equity=20_000.0,
    )
    state3.update_fill("AAPL", filled_qty=100.0, price=50.0, side=OrderSide.SELL)
    assert state3.current_equity == 15_000.0
    assert state3.peak_equity == 20_000.0
    assert math.isclose(state3.intraday_drawdown, 0.25)


def test_drawdown_peak_equity_zero_boundary() -> None:
    """Verify intraday_drawdown returns 1.0 or 0.0 if peak_equity is zero."""
    state = PortfolioRiskState(
        cash=0.0,
        peak_equity=1.0,
        initial_equity=1.0,
    )
    object.__setattr__(state, "peak_equity", 0.0)
    assert state.intraday_drawdown == 1.0


def test_validate_order_projected_equity_zero() -> None:
    """Verify validate_order raises LeverageLimitExceededException when projected equity <= 0."""
    limits = make_default_limits(max_intraday_drawdown_pct=2.0)
    state = PortfolioRiskState(
        cash=10_000.0,
        positions={"AAPL": -100.0},
        current_prices={"AAPL": 100.0},
        peak_equity=10_000.0,
        initial_equity=10_000.0,
    )
    firewall = PreTradeRiskFirewall(limits=limits)
    order = make_order(quantity=10.0, price=100.0)
    with pytest.raises(LeverageLimitExceededException) as exc_info:
        firewall.validate_order(order, state)
    assert exc_info.value.code == ERR_RSK_LEVERAGE_LIMIT_EXCEEDED
    assert "non-positive" in exc_info.value.message


# ============================================================================
# Section 9: Phase 6 Step 3 Task 1 Adversarial Remediation Tests
# ============================================================================


def test_selling_long_positions_with_negative_cash_passes_margin_check() -> None:
    """Remediation 1: Verify selling long inventory when cash is negative passes Check 6 unconditionally."""
    # Leveraged account: cash = -$10,000, 200 shares AAPL @ $150 => equity = $20,000.
    # free_margin = -$10,000. min_free_margin = 1,000.
    limits = make_default_limits(
        max_order_notional=100_000.0,
        max_order_qty=1_000.0,
        max_gross_leverage=3.5,
        max_net_leverage=2.0,
        max_concentration_nav_pct=2.0,
        max_intraday_drawdown_pct=0.50,
        min_free_margin=1_000.0,
    )
    state = PortfolioRiskState(
        cash=-10_000.0,
        positions={"AAPL": 200.0},
        current_prices={"AAPL": 150.0},
        peak_equity=20_000.0,
        initial_equity=20_000.0,
    )
    firewall = PreTradeRiskFirewall(limits=limits)

    # Selling 100 shares long AAPL: required_margin is 0.0.
    # Must pass Check 6 unconditionally without raising InsufficientMarginRiskException!
    sell_order = Order(
        cl_ord_id="close-long-1",
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        price=150.0,
        time_in_force=TimeInForce.GTC,
    )
    firewall.validate_order(sell_order, state)

    # Also test buying to close short when cash is negative:
    state_short = PortfolioRiskState(
        cash=-5_000.0,
        positions={"AAPL": -100.0, "MSFT": 200.0},
        current_prices={"AAPL": 150.0, "MSFT": 200.0},
        peak_equity=20_000.0,
        initial_equity=20_000.0,
    )
    # Buying 100 AAPL closes the short position, required_margin = 0.0, must pass Check 6 unconditionally!
    buy_cover_order = Order(
        cl_ord_id="close-short-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        price=150.0,
        time_in_force=TimeInForce.GTC,
    )
    firewall.validate_order(buy_cover_order, state_short)


def test_missing_or_corrupt_price_for_leaves_or_positions_raises() -> None:
    """Remediation 2: Verify missing or non-positive price for non-order symbol raises ERR_RSK_NON_FINITE_INPUT."""
    limits = make_default_limits(
        max_order_notional=100_000.0,
        max_order_qty=1_000.0,
        max_gross_leverage=1.0,
        max_net_leverage=1.0,
        min_free_margin=0.0,
    )

    # Subcase A: resting leaves in unpriced symbol "GHOST"
    state_unpriced_leaves = PortfolioRiskState(
        cash=10_000.0,
        positions={},
        pending_leaves={"GHOST": 1_000_000.0},
        current_prices={"AAPL": 100.0},
        peak_equity=10_000.0,
        initial_equity=10_000.0,
    )
    firewall = PreTradeRiskFirewall(limits=limits)
    candidate_order = Order(
        cl_ord_id="ord-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=100.0,
        time_in_force=TimeInForce.GTC,
    )
    with pytest.raises(NonFiniteRiskInputException) as exc_info:
        firewall.validate_order(candidate_order, state_unpriced_leaves)
    assert exc_info.value.code == ERR_RSK_NON_FINITE_INPUT
    assert (
        "Missing current market price for symbol 'GHOST' in portfolio state"
        in exc_info.value.message
    )

    # Subcase B: free_margin property directly on unpriced leaves raises
    with pytest.raises(NonFiniteRiskInputException) as exc_fm:
        _ = state_unpriced_leaves.free_margin
    assert exc_fm.value.code == ERR_RSK_NON_FINITE_INPUT
    assert (
        "Missing current market price for symbol 'GHOST' in portfolio state" in exc_fm.value.message
    )

    # Subcase C: non-order symbol with corrupt/negative price raises
    state_corrupt_price = PortfolioRiskState(
        cash=10_000.0,
        positions={"MSFT": 10.0},
        current_prices={"AAPL": 100.0, "MSFT": 200.0},
        peak_equity=12_000.0,
        initial_equity=12_000.0,
    )
    # Bypass post-init to corrupt price
    state_corrupt_price.current_prices["MSFT"] = -50.0
    with pytest.raises(NonFiniteRiskInputException) as exc_corrupt:
        firewall.validate_order(candidate_order, state_corrupt_price)
    assert exc_corrupt.value.code == ERR_RSK_NON_FINITE_INPUT
    assert "strictly positive finite scalar" in exc_corrupt.value.message


def test_resting_short_leaves_encumber_free_margin() -> None:
    """Remediation 3: Verify resting short sell leaves properly encumber margin in free_margin."""
    limits = make_default_limits(min_free_margin=5_000.0)

    # Cash = $10,000, resting short order of 40 shares @ $100 => locks $4,000
    state = PortfolioRiskState(
        cash=10_000.0,
        positions={},
        pending_leaves={"AAPL": -40.0},
        current_prices={"AAPL": 100.0},
        peak_equity=10_000.0,
        initial_equity=10_000.0,
    )
    # free_margin should be exactly 10,000 - 4,000 = 6,000
    assert math.isclose(state.free_margin, 6_000.0)

    # Submitting another short order of 30 shares requires $3,000 margin:
    # free_margin_after = 6,000 - 3,000 = 3,000 < min_free_margin (5,000) => REJECTED!
    firewall = PreTradeRiskFirewall(limits=limits)
    second_short = Order(
        cl_ord_id="short-2",
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=30.0,
        price=100.0,
        time_in_force=TimeInForce.GTC,
    )
    with pytest.raises(InsufficientMarginRiskException) as exc_info:
        firewall.validate_order(second_short, state)
    assert exc_info.value.code == ERR_RSK_INSUFFICIENT_MARGIN
    assert "3000.00 breaches minimum buffer 5000.00" in exc_info.value.message

    # Test covered sell leaves: long 50 shares, sell leaves = -70 shares => uncovered = 20 shares
    state_covered = PortfolioRiskState(
        cash=10_000.0,
        positions={"AAPL": 50.0},
        pending_leaves={"AAPL": -70.0},
        current_prices={"AAPL": 100.0},
        peak_equity=15_000.0,
        initial_equity=15_000.0,
    )
    # Uncovered short leaves = max(0, 70 - 50) = 20 => locked = $2,000 => free_margin = $8,000
    assert math.isclose(state_covered.free_margin, 8_000.0)


def test_universal_directional_netting_across_all_symbols() -> None:
    """Remediation 4: Resting exit leaves on other symbols do not artificially double gross leverage."""
    # NAV = $60k cash + 400 MSFT @ $100 ($40k) = $100,000 equity.
    # Max gross leverage = 1.0 (max allowable gross notional = $100,000).
    limits = make_default_limits(
        max_gross_leverage=1.0,
        max_order_notional=100_000.0,
        max_order_qty=1_000.0,
        max_concentration_nav_pct=1.0,
    )
    # Portfolio holds 400 MSFT and has resting limit order to sell 400 MSFT (leaves = -400)
    state = PortfolioRiskState(
        cash=60_000.0,
        positions={"MSFT": 400.0},
        pending_leaves={"MSFT": -400.0},
        current_prices={"AAPL": 100.0, "MSFT": 100.0},
        peak_equity=100_000.0,
        initial_equity=100_000.0,
    )
    # 1. Directly check gross_leverage property:
    # MSFT exposure is max(|400|, |400 - 400|) * 100 = $40,000 (not 80,000!)
    # Gross leverage = $40,000 / $100,000 = 0.40
    assert math.isclose(state.gross_leverage, 0.40)

    # 2. In validate_order: submit BUY 300 AAPL @ $100 = $30,000 notional.
    # Under naive addition: MSFT was 80k + AAPL 30k = 110k (1.1x leverage => REJECTED).
    # Under universal directional netting: MSFT is 40k + AAPL 30k = 70k (0.7x leverage <= 1.0x => ACCEPTED).
    firewall = PreTradeRiskFirewall(limits=limits)
    buy_aapl = Order(
        cl_ord_id="ord-aapl-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=300.0,
        price=100.0,
        time_in_force=TimeInForce.GTC,
    )
    # Must clear cleanly without raising LeverageLimitExceededException
    firewall.validate_order(buy_aapl, state)
