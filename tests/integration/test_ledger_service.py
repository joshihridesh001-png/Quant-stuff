"""Integration tests for LedgerService, Tax Lot Persistence, and System Rehydration.

Functional Purpose:
    Verifies end-to-end relational ledger operations:
    1. Initializing and rehydrating portfolio risk state from database.
    2. Atomically recording execution fills (buy, partial sell, realized gains).
    3. Simulating complete system restart and verifying zero state divergence in rehydrated state.
    4. End-of-day daily financing accrual, journal entry logging, and PnL snapshots.

Explicit Dependency Tracking:
    - pytest, sqlalchemy.
    - quant.services.ledger_service: LedgerService, RehydrationError.
    - quant.execution.tax_lot: LotSide, LotStatus.
    - quant.infrastructure.database.ledger_models: DBLedgerEntry, DBPortfolioLedger,
      DBRealizedTrade, DBTaxLot, DBDailyPnLSnapshot.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from quant.infrastructure.database.ledger_models import (
    DBDailyPnLSnapshot,
    DBLedgerEntry,
    DBPortfolioLedger,
    DBTaxLot,
)
from quant.services.ledger_service import LedgerService


@pytest.mark.asyncio
async def test_ledger_service_rehydration_lifecycle(db_session: AsyncSession) -> None:
    """Verify state initialization, order fill recording, and restart rehydration."""
    port_id = "test_port_rehydrate_001"
    service1 = LedgerService(portfolio_id=port_id)

    # 1. Initial rehydration (fresh portfolio setup)
    state1 = await service1.initialize_or_rehydrate(session=db_session)
    assert state1.cash == 100000.0
    assert len(state1.positions) == 0

    t0 = datetime(2024, 4, 1, 10, 0, 0, tzinfo=UTC)

    # 2. Record BUY 100 AAPL @ $150 ($15,000 cash spent)
    res_buy1 = await service1.record_fill(
        session=db_session,
        symbol="AAPL",
        side="BUY",
        quantity=100.0,
        execution_price=150.0,
        timestamp=t0,
    )
    assert len(res_buy1.new_open_lots) == 1
    assert service1.tax_lot_engine.get_position("AAPL") == 100.0

    # 3. Record BUY 50 AAPL @ $160 ($8,000 cash spent)
    t1 = datetime(2024, 4, 2, 10, 0, 0, tzinfo=UTC)
    await service1.record_fill(
        session=db_session,
        symbol="AAPL",
        side="BUY",
        quantity=50.0,
        execution_price=160.0,
        timestamp=t1,
    )
    assert service1.tax_lot_engine.get_position("AAPL") == 150.0

    # 4. Record SELL 70 AAPL @ $170 (partial sale of lot 1 under FIFO)
    t2 = datetime(2024, 4, 3, 10, 0, 0, tzinfo=UTC)
    res_sell = await service1.record_fill(
        session=db_session,
        symbol="AAPL",
        side="SELL",
        quantity=70.0,
        execution_price=170.0,
        timestamp=t2,
    )
    # Realized gain on 70 shares @ ($170 - $150) = $1,400
    assert math.isclose(res_sell.total_realized_pnl, 1400.0)
    assert service1.tax_lot_engine.get_position("AAPL") == 80.0

    # Verify database state after trades
    lot_stmt = select(DBTaxLot).where(DBTaxLot.portfolio_id == port_id)
    lots_res = await db_session.execute(lot_stmt)
    persisted_lots = lots_res.scalars().all()
    assert len(persisted_lots) == 2

    # Lot 1 should have 30 remaining shares (100 - 70) and status PARTIAL
    lot1 = next(lot for lot in persisted_lots if lot.cost_basis_per_share == 150.0)
    assert lot1.remaining_quantity == 30.0
    assert lot1.status == "PARTIAL"

    # Lot 2 should have 50 remaining shares and status OPEN
    lot2 = next(lot for lot in persisted_lots if lot.cost_basis_per_share == 160.0)
    assert lot2.remaining_quantity == 50.0
    assert lot2.status == "OPEN"

    # Verify cash balance in DBPortfolioLedger:
    # 100,000 - 15,000 - 8,000 + (70 * 170 = 11,900) = 88,900
    ledger_stmt = select(DBPortfolioLedger).where(DBPortfolioLedger.portfolio_id == port_id)
    l_res = await db_session.execute(ledger_stmt)
    db_ledger = l_res.scalar_one()
    assert math.isclose(db_ledger.cash_balance, 88900.0)

    # 5. SIMULATE ENGINE RESTART
    # Instantiate completely new LedgerService instance with blank memory
    service2 = LedgerService(portfolio_id=port_id)
    assert len(service2.tax_lot_engine._lots) == 0

    # Rehydrate state from database
    state2 = await service2.initialize_or_rehydrate(
        session=db_session,
        current_prices={"AAPL": 175.0},
    )

    # Verify 100% exact state restoration without discrepancies
    assert math.isclose(state2.cash, 88900.0)
    assert state2.positions.get("AAPL") == 80.0
    assert service2.tax_lot_engine.get_position("AAPL") == 80.0

    rehydrated_lots = service2.tax_lot_engine.get_open_lots("AAPL")
    assert len(rehydrated_lots) == 2
    assert sum(lot.remaining_quantity for lot in rehydrated_lots) == 80.0


@pytest.mark.asyncio
async def test_ledger_service_daily_eod_accrual(db_session: AsyncSession) -> None:
    """Verify daily end-of-day financing calculation and snapshot creation."""
    port_id = "test_port_eod_002"
    service = LedgerService(portfolio_id=port_id)
    await service.initialize_or_rehydrate(session=db_session)

    # Simulate short position: SELL 100 TSLA @ $200
    t0 = datetime(2024, 5, 1, 10, 0, 0, tzinfo=UTC)
    await service.record_fill(
        session=db_session,
        symbol="TSLA",
        side="SELL",
        quantity=100.0,
        execution_price=200.0,
        timestamp=t0,
    )
    assert service.tax_lot_engine.get_position("TSLA") == -100.0

    # Execute EOD accrual
    prices = {"TSLA": 205.0}
    accrual = await service.accrue_daily_eod(
        session=db_session,
        date_str="2024-05-01",
        current_prices=prices,
        days=1.0,
    )

    assert accrual.borrow_fees_accrued > 0.0
    assert accrual.total_financing_drag > 0.0

    # Verify snapshot persisted in database
    snap_stmt = select(DBDailyPnLSnapshot).where(
        DBDailyPnLSnapshot.portfolio_id == port_id,
        DBDailyPnLSnapshot.date_str == "2024-05-01",
    )
    snap_res = await db_session.execute(snap_stmt)
    snapshot = snap_res.scalar_one_or_none()
    assert snapshot is not None
    assert snapshot.borrow_fees_accrued == accrual.borrow_fees_accrued
    assert snapshot.gross_notional == 20500.0  # 100 * 205.0


@pytest.mark.asyncio
async def test_ledger_service_compound_journal_entries_gaap(db_session: AsyncSession) -> None:
    """Verify that selling inventory posts balanced compound journal entries relieving cost basis and booking realized gain."""
    port_id = "test_port_gaap_003"
    service = LedgerService(portfolio_id=port_id)
    await service.initialize_or_rehydrate(session=db_session)

    t0 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
    # Buy 10 shares of MSFT @ $100 ($1,000 cost basis)
    await service.record_fill(
        session=db_session,
        symbol="MSFT",
        side="BUY",
        quantity=10.0,
        execution_price=100.0,
        timestamp=t0,
    )

    t1 = datetime(2024, 6, 2, 10, 0, 0, tzinfo=UTC)
    # Sell 10 shares of MSFT @ $150 ($1,500 proceeds, $500 realized gain)
    await service.record_fill(
        session=db_session,
        symbol="MSFT",
        side="SELL",
        quantity=10.0,
        execution_price=150.0,
        timestamp=t1,
    )

    # Fetch ledger entries for port_id
    stmt = (
        select(DBLedgerEntry)
        .where(DBLedgerEntry.portfolio_id == port_id)
        .order_by(DBLedgerEntry.timestamp.asc())
    )
    res = await db_session.execute(stmt)
    entries = res.scalars().all()

    # Total debits must equal total credits
    total_debits = sum(e.amount for e in entries if e.entry_type == "DEBIT")
    total_credits = sum(e.amount for e in entries if e.entry_type == "CREDIT")
    assert math.isclose(total_debits, total_credits, abs_tol=1e-3)

    # Check inventory entries:
    # 1. Debit ASSET_INVENTORY_MSFT 1000.0
    # 2. Credit ASSET_INVENTORY_MSFT 1000.0 (relief of cost basis!)
    inv_debits = sum(
        e.amount
        for e in entries
        if e.account_name == "ASSET_INVENTORY_MSFT" and e.entry_type == "DEBIT"
    )
    inv_credits = sum(
        e.amount
        for e in entries
        if e.account_name == "ASSET_INVENTORY_MSFT" and e.entry_type == "CREDIT"
    )
    assert inv_debits == 1000.0
    assert inv_credits == 1000.0  # NOT 1500.0! Inventory balance is exactly 0.0

    # Check realized gain
    gain_credits = sum(
        e.amount
        for e in entries
        if e.account_name == "REVENUE_REALIZED_GAIN" and e.entry_type == "CREDIT"
    )
    assert gain_credits == 500.0
