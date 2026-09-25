"""Relational Double-Entry Ledger and Tax-Lot Accounting SQLAlchemy Models.

Functional Purpose:
    Defines persistent database schemas for institutional multi-asset double-entry
    bookkeeping, FIFO/LIFO tax-lot depletion tracking, financing drag accruals,
    and daily mark-to-market PnL snapshots.

Explicit Dependency Tracking:
    - SQLAlchemy 2.0: Declarative ORM mappings, mapped_column, Mapped, String, Float, DateTime.
    - datetime: UTC timestamp generation and tracking.
    - uuid: Unique entity primary key generation.
    - quant.infrastructure.database.session: Base declarative foundation.

Structural Relationship:
    - Database entity layer for Phase 16.
    - Consumed by TaxLotEngine (16.2), FinancingService (16.3), and LedgerService (16.4).
    - Persisted to SQLite (WAL mode) or PostgreSQL.

Defensive Invariants:
    - INV-LDG-001: Double-Entry Conservation: For every transaction_id, sum(Debits) == sum(Credits).
    - INV-LDG-002: Inventory Conservation: remaining_quantity in [0.0, original_quantity].
    - INV-LDG-003: Cost Basis Non-Negativity: cost_basis_per_share >= 0.0.
    - Rule 1: Four-tier docstrings on every entity and method.
    - Rule 2: Catalog diagnostic codes (ERR-LDG-001 to ERR-LDG-005).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Final

from sqlalchemy import DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from quant.infrastructure.database.session import Base

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_LDG_NEGATIVE_INVENTORY: Final[str] = "ERR-LDG-001"
ERR_LDG_ORPHANED_FILL: Final[str] = "ERR-LDG-002"
ERR_LDG_UNBALANCED_TRANSACTION: Final[str] = "ERR-LDG-003"
ERR_LDG_LOT_NOT_FOUND: Final[str] = "ERR-LDG-004"
ERR_LDG_NON_FINITE_INPUT: Final[str] = "ERR-LDG-005"


def utc_now() -> datetime:
    """Return timezone-aware current UTC timestamp.

    Functional Purpose:
        Generates standard UTC datetime instances for database created/updated columns.
    Explicit Dependency Tracking:
        datetime.now(UTC).
    Defensive Invariant:
        Guarantees UTC timezone-aware datetime.
    """
    return datetime.now(UTC)


class DBPortfolioLedger(Base):
    """Relational table tracking core portfolio account balances and capital state.

    Functional Purpose:
        Persists authoritative unencumbered cash balance, margin debt, and total equity.
    Explicit Dependency Tracking:
        SQLAlchemy Base, String, Float, DateTime.
    Structural Relationship:
        Parent entity for tax lots, ledger entries, and daily PnL snapshots.
    Defensive Invariant:
        portfolio_id must be unique across all portfolios.
    """

    __tablename__ = "portfolio_ledgers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False, default="Default Portfolio")
    base_currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USD")
    cash_balance: Mapped[float] = mapped_column(Float, nullable=False, default=100000.0)
    margin_debt: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_equity: Mapped[float] = mapped_column(Float, nullable=False, default=100000.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class DBTaxLot(Base):
    """Relational table tracking individual open or partially depleted tax lots.

    Functional Purpose:
        Maintains granular purchase lots to compute exact capital gains upon disposition.
    Explicit Dependency Tracking:
        SQLAlchemy Base, String, Float, DateTime.
    Structural Relationship:
        Referenced by TaxLotEngine to match sells/covers against specific lots.
    Defensive Invariant:
        INV-LDG-002: 0.0 <= remaining_quantity <= original_quantity.
    """

    __tablename__ = "tax_lots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # "LONG" or "SHORT"
    open_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    original_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    remaining_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    cost_basis_per_share: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="OPEN", index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        Index("ix_tax_lots_portfolio_symbol_status", "portfolio_id", "symbol", "status"),
        Index("ix_tax_lots_symbol_open_timestamp", "symbol", "open_timestamp"),
    )


class DBLedgerEntry(Base):
    """Relational table recording atomic double-entry bookkeeping journal entries.

    Functional Purpose:
        Records immutable debits and credits adhering to GAAP double-entry rules.
    Explicit Dependency Tracking:
        SQLAlchemy Base, String, Float, DateTime.
    Structural Relationship:
        Every financial mutation produces paired DBLedgerEntry rows sharing transaction_id.
    Defensive Invariant:
        INV-LDG-001: Balanced transaction sum(debit) == sum(credit).
    """

    __tablename__ = "ledger_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    portfolio_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    account_name: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    entry_type: Mapped[str] = mapped_column(String(10), nullable=False)  # "DEBIT" or "CREDIT"
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)

    __table_args__ = (Index("ix_ledger_entries_portfolio_timestamp", "portfolio_id", "timestamp"),)


class DBRealizedTrade(Base):
    """Relational table capturing closed trades, holding duration, and realized PnL.

    Functional Purpose:
        Stores finalized trade dispositions after matching order fills against tax lots.
    Explicit Dependency Tracking:
        SQLAlchemy Base, String, Float, DateTime.
    Structural Relationship:
        Generated by TaxLotEngine upon partial or full closing of a tax lot.
    Defensive Invariant:
        quantity > 0.0, holding_period_seconds >= 0.0.
    """

    __tablename__ = "realized_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    lot_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # "SELL" or "COVER"
    close_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    execution_price: Mapped[float] = mapped_column(Float, nullable=False)
    cost_basis: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    holding_period_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class DBDailyPnLSnapshot(Base):
    """Relational table recording daily end-of-day mark-to-market performance snapshots.

    Functional Purpose:
        Maintains historical daily ledger state for performance attribution and audit.
    Explicit Dependency Tracking:
        SQLAlchemy Base, String, Float, DateTime.
    Structural Relationship:
        Aggregated by reporting and audit modules.
    Defensive Invariant:
        nav == cash + marked holdings value.
    """

    __tablename__ = "daily_pnl_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    date_str: Mapped[str] = mapped_column(String(10), index=True, nullable=False)  # "YYYY-MM-DD"
    nav: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    gross_notional: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    margin_interest_accrued: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    borrow_fees_accrued: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (
        Index("ix_daily_pnl_portfolio_date", "portfolio_id", "date_str", unique=True),
    )
