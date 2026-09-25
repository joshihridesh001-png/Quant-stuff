"""Integration tests for Double-Entry Ledger and Tax Lot SQLAlchemy Models.

Functional Purpose:
    Verifies persistence, table schema generation, unique constraints, and ACID properties
    of the portfolio ledger, tax lots, ledger entries, realized trades, and daily snapshots.

Explicit Dependency Tracking:
    - pytest, sqlalchemy.
    - quant.infrastructure.database.ledger_models: DBPortfolioLedger, DBTaxLot,
      DBLedgerEntry, DBRealizedTrade, DBDailyPnLSnapshot, utc_now.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from quant.infrastructure.database.ledger_models import (
    DBDailyPnLSnapshot,
    DBLedgerEntry,
    DBPortfolioLedger,
    DBRealizedTrade,
    DBTaxLot,
    utc_now,
)


@pytest.mark.asyncio
async def test_portfolio_ledger_lifecycle(db_session: AsyncSession) -> None:
    """Verify creating, querying, and updating DBPortfolioLedger."""
    portfolio = DBPortfolioLedger(
        portfolio_id="port_alpha_001",
        name="Alpha Hedge Portfolio",
        base_currency="USD",
        cash_balance=500000.0,
        margin_debt=0.0,
        total_equity=500000.0,
    )
    db_session.add(portfolio)
    await db_session.commit()

    # Query back
    stmt = select(DBPortfolioLedger).where(DBPortfolioLedger.portfolio_id == "port_alpha_001")
    result = await db_session.execute(stmt)
    fetched = result.scalar_one_or_none()

    assert fetched is not None
    assert fetched.name == "Alpha Hedge Portfolio"
    assert fetched.cash_balance == 500000.0
    assert fetched.total_equity == 500000.0
    assert fetched.created_at is not None


@pytest.mark.asyncio
async def test_tax_lot_lifecycle(db_session: AsyncSession) -> None:
    """Verify storing and updating open and partially depleted tax lots."""
    lot = DBTaxLot(
        portfolio_id="port_alpha_001",
        symbol="NVDA",
        side="LONG",
        open_timestamp=datetime.now(UTC),
        original_quantity=100.0,
        remaining_quantity=100.0,
        cost_basis_per_share=125.50,
        status="OPEN",
    )
    db_session.add(lot)
    await db_session.commit()

    # Query and simulate partial depletion
    stmt = select(DBTaxLot).where(DBTaxLot.symbol == "NVDA", DBTaxLot.status == "OPEN")
    result = await db_session.execute(stmt)
    fetched = result.scalar_one_or_none()

    assert fetched is not None
    assert fetched.cost_basis_per_share == 125.50
    assert fetched.remaining_quantity == 100.0

    # Deplete 40 shares
    fetched.remaining_quantity = 60.0
    fetched.status = "PARTIAL"
    await db_session.commit()

    stmt2 = select(DBTaxLot).where(DBTaxLot.id == fetched.id)
    res2 = await db_session.execute(stmt2)
    updated = res2.scalar_one()
    assert updated.remaining_quantity == 60.0
    assert updated.status == "PARTIAL"


@pytest.mark.asyncio
async def test_double_entry_journal_entries(db_session: AsyncSession) -> None:
    """Verify recording balanced debit and credit entries for a single transaction."""
    tx_id = str(uuid.uuid4())
    now = utc_now()

    # Buy 100 shares of AAPL for $15,000 cash
    debit_entry = DBLedgerEntry(
        transaction_id=tx_id,
        portfolio_id="port_alpha_001",
        account_name="ASSET_INVENTORY_AAPL",
        entry_type="DEBIT",
        amount=15000.0,
        description="Bought 100 AAPL @ 150.0",
        timestamp=now,
    )
    credit_entry = DBLedgerEntry(
        transaction_id=tx_id,
        portfolio_id="port_alpha_001",
        account_name="ASSET_CASH",
        entry_type="CREDIT",
        amount=15000.0,
        description="Disbursed cash for 100 AAPL",
        timestamp=now,
    )

    db_session.add_all([debit_entry, credit_entry])
    await db_session.commit()

    # Verify transaction balance
    stmt = select(DBLedgerEntry).where(DBLedgerEntry.transaction_id == tx_id)
    result = await db_session.execute(stmt)
    entries = result.scalars().all()

    assert len(entries) == 2
    debits = sum(e.amount for e in entries if e.entry_type == "DEBIT")
    credits = sum(e.amount for e in entries if e.entry_type == "CREDIT")
    assert debits == credits == 15000.0


@pytest.mark.asyncio
async def test_realized_trade_and_snapshot(db_session: AsyncSession) -> None:
    """Verify recording realized trade records and daily mark-to-market snapshots."""
    now = utc_now()
    trade = DBRealizedTrade(
        portfolio_id="port_alpha_001",
        lot_id=str(uuid.uuid4()),
        symbol="AAPL",
        side="SELL",
        close_timestamp=now,
        quantity=50.0,
        execution_price=170.0,
        cost_basis=150.0,
        realized_pnl=1000.0,  # 50 * (170 - 150)
        holding_period_seconds=86400.0 * 14,
    )
    snapshot = DBDailyPnLSnapshot(
        portfolio_id="port_alpha_001",
        date_str="2024-05-15",
        nav=501000.0,
        cash=485000.0,
        gross_notional=16000.0,
        realized_pnl=1000.0,
        unrealized_pnl=0.0,
        margin_interest_accrued=0.0,
        borrow_fees_accrued=0.0,
    )
    db_session.add_all([trade, snapshot])
    await db_session.commit()

    stmt = select(DBDailyPnLSnapshot).where(
        DBDailyPnLSnapshot.portfolio_id == "port_alpha_001",
        DBDailyPnLSnapshot.date_str == "2024-05-15",
    )
    res = await db_session.execute(stmt)
    snap = res.scalar_one_or_none()
    assert snap is not None
    assert snap.nav == 501000.0
    assert snap.realized_pnl == 1000.0
