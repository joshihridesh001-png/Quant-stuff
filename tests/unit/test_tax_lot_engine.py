"""Unit tests for Deterministic Tax-Lot Engine and Capital Gains Allocation.

Functional Purpose:
    Verifies FIFO / LIFO order fill allocation against tax lots, realized PnL accuracy,
    holding period calculations, short sales & covers, position flips, and inventory conservation.

Explicit Dependency Tracking:
    - pytest, datetime.
    - quant.execution.tax_lot: TaxLotEngine, TaxLotMethod, LotSide, LotStatus,
      NegativeInventoryError, OrphanedFillError, NonFiniteLedgerInputError.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from quant.execution.tax_lot import (
    LotSide,
    LotStatus,
    NegativeInventoryError,
    NonFiniteLedgerInputError,
    OrphanedFillError,
    TaxLotEngine,
    TaxLotMethod,
)


def test_fifo_allocation_lifecycle() -> None:
    """Verify standard FIFO matching: oldest lots depleted first."""
    engine = TaxLotEngine(portfolio_id="p1", default_method=TaxLotMethod.FIFO)
    t0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Buy 100 shares @ $100
    res1 = engine.allocate_fill("AAPL", "BUY", 100.0, 100.0, timestamp=t0)
    assert len(res1.new_open_lots) == 1
    assert engine.get_position("AAPL") == 100.0

    # Buy 50 shares @ $110 at t0 + 1 hour
    t1 = t0 + timedelta(hours=1)
    res2 = engine.allocate_fill("AAPL", "BUY", 50.0, 110.0, timestamp=t1)
    assert len(res2.new_open_lots) == 1
    assert engine.get_position("AAPL") == 150.0

    # Sell 120 shares @ $120 at t0 + 24 hours
    t2 = t0 + timedelta(days=1)
    res3 = engine.allocate_fill("AAPL", "SELL", 120.0, 120.0, timestamp=t2)

    # 100 shares matched from lot 1 ($100 basis) -> PnL: 100 * (120 - 100) = $2,000
    # 20 shares matched from lot 2 ($110 basis) -> PnL: 20 * (120 - 110) = $200
    # Total PnL: $2,200
    assert len(res3.allocated_trades) == 2
    assert math.isclose(res3.total_realized_pnl, 2200.0)

    trade1 = res3.allocated_trades[0]
    assert trade1.quantity == 100.0
    assert trade1.cost_basis == 10000.0
    assert trade1.realized_pnl == 2000.0
    assert trade1.holding_period_seconds == 86400.0

    trade2 = res3.allocated_trades[1]
    assert trade2.quantity == 20.0
    assert trade2.cost_basis == 2200.0
    assert trade2.realized_pnl == 200.0

    # Remaining position: 30 shares
    assert engine.get_position("AAPL") == 30.0
    open_lots = engine.get_open_lots("AAPL")
    assert len(open_lots) == 1
    assert open_lots[0].remaining_quantity == 30.0
    assert open_lots[0].status == LotStatus.PARTIAL


def test_lifo_allocation_lifecycle() -> None:
    """Verify LIFO matching: newest lots depleted first."""
    engine = TaxLotEngine(portfolio_id="p1", default_method=TaxLotMethod.LIFO)
    t0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Buy 100 shares @ $100
    engine.allocate_fill("NVDA", "BUY", 100.0, 100.0, timestamp=t0)
    # Buy 50 shares @ $110 (newer)
    t1 = t0 + timedelta(hours=2)
    engine.allocate_fill("NVDA", "BUY", 50.0, 110.0, timestamp=t1)

    # Sell 40 shares @ $120 under LIFO
    t2 = t0 + timedelta(hours=4)
    res = engine.allocate_fill("NVDA", "SELL", 40.0, 120.0, timestamp=t2)

    # In LIFO, the 40 shares are taken strictly from the 50-share lot @ $110
    # PnL: 40 * (120 - 110) = $400
    assert len(res.allocated_trades) == 1
    assert math.isclose(res.total_realized_pnl, 400.0)

    open_lots = engine.get_open_lots("NVDA")
    assert len(open_lots) == 2
    # Oldest lot is completely untouched (100 shares)
    assert open_lots[0].remaining_quantity == 100.0
    assert open_lots[0].status == LotStatus.OPEN
    # Newest lot has 10 shares remaining
    assert open_lots[1].remaining_quantity == 10.0
    assert open_lots[1].status == LotStatus.PARTIAL
    assert engine.get_position("NVDA") == 110.0


def test_short_sale_and_cover() -> None:
    """Verify short selling creates SHORT lots and cover orders realize gains correctly."""
    engine = TaxLotEngine(portfolio_id="p1", allow_shorting=True)
    t0 = datetime(2024, 2, 1, 10, 0, 0, tzinfo=UTC)

    # Short sell 50 shares of TSLA @ $200
    res_short = engine.allocate_fill("TSLA", "SELL", 50.0, 200.0, timestamp=t0)
    assert len(res_short.new_open_lots) == 1
    assert res_short.new_open_lots[0].side == LotSide.SHORT
    assert engine.get_position("TSLA") == -50.0

    # Cover 30 shares @ $170 (buying to cover)
    t1 = t0 + timedelta(days=3)
    res_cover = engine.allocate_fill("TSLA", "BUY", 30.0, 170.0, timestamp=t1)

    # Realized gain on short: 30 * (200 - 170) = +$900
    assert len(res_cover.allocated_trades) == 1
    assert res_cover.allocated_trades[0].side == "COVER"
    assert math.isclose(res_cover.total_realized_pnl, 900.0)
    assert engine.get_position("TSLA") == -20.0


def test_position_flip_long_to_short() -> None:
    """Verify selling more shares than currently long cleanly flips the position to short."""
    engine = TaxLotEngine(portfolio_id="p1", allow_shorting=True)
    t0 = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)

    # Long 50 MSFT @ $300
    engine.allocate_fill("MSFT", "BUY", 50.0, 300.0, timestamp=t0)
    assert engine.get_position("MSFT") == 50.0

    # Sell 80 MSFT @ $320: 50 closed @ profit, 30 new SHORT lot
    t1 = t0 + timedelta(days=2)
    res = engine.allocate_fill("MSFT", "SELL", 80.0, 320.0, timestamp=t1)

    # 50 shares realized pnl: 50 * (320 - 300) = $1,000
    assert math.isclose(res.total_realized_pnl, 1000.0)
    # Remaining position is -30 (Short 30)
    assert engine.get_position("MSFT") == -30.0
    open_lots = engine.get_open_lots("MSFT")
    assert len(open_lots) == 1
    assert open_lots[0].side == LotSide.SHORT
    assert open_lots[0].remaining_quantity == 30.0
    assert open_lots[0].cost_basis_per_share == 320.0


def test_unrealized_pnl_calculation() -> None:
    """Verify mark-to-market unrealized PnL computation."""
    engine = TaxLotEngine(portfolio_id="p1")
    t0 = datetime(2024, 1, 1, tzinfo=UTC)

    # Long 100 shares @ $150
    engine.allocate_fill("SPY", "BUY", 100.0, 150.0, timestamp=t0)
    # Long 50 shares @ $160
    engine.allocate_fill("SPY", "BUY", 50.0, 160.0, timestamp=t0)

    # Current price = $170
    # Lot 1 unrealized: 100 * (170 - 150) = $2,000
    # Lot 2 unrealized: 50 * (170 - 160) = $500
    # Total unrealized = $2,500
    unrealized = engine.get_unrealized_pnl("SPY", 170.0)
    assert math.isclose(unrealized, 2500.0)


def test_shorting_disabled_guard() -> None:
    """Verify selling without inventory when allow_shorting=False raises OrphanedFillError."""
    engine = TaxLotEngine(portfolio_id="p1", allow_shorting=False)
    with pytest.raises(OrphanedFillError):
        engine.allocate_fill("QQQ", "SELL", 50.0, 400.0)


def test_non_finite_inputs_rejected() -> None:
    """Verify non-finite inputs, booleans, and negative numbers are rejected."""
    engine = TaxLotEngine(portfolio_id="p1")

    with pytest.raises(NonFiniteLedgerInputError):
        engine.allocate_fill("AAPL", "BUY", float("nan"), 100.0)

    with pytest.raises(NonFiniteLedgerInputError):
        engine.allocate_fill("AAPL", "BUY", 100.0, -50.0)

    with pytest.raises(NonFiniteLedgerInputError):
        engine.allocate_fill("AAPL", "BUY", True, 100.0)  # bool rejected

    with pytest.raises(NonFiniteLedgerInputError):
        engine.allocate_fill("AAPL", "INVALID_SIDE", 100.0, 100.0)


def test_negative_inventory_guard() -> None:
    """Verify inventory conservation error triggers NegativeInventoryError if corrupted."""
    engine = TaxLotEngine(portfolio_id="p1")
    engine.allocate_fill("AAPL", "BUY", 100.0, 150.0)
    # Corrupt remaining quantity manually to simulate inventory drift
    engine._lots["AAPL"][0].remaining_quantity = -5.0
    with pytest.raises(NegativeInventoryError):
        engine.verify_inventory_conservation("AAPL")
