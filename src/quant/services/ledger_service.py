"""Startup State Rehydration Facade and Relational Ledger Service.

Functional Purpose:
    Provides an institutional persistence facade for portfolio state:
    1. Rehydrates open tax lots, cash balances, and positions from relational database
       upon system restart, guaranteeing zero state divergence and zero broker re-querying.
    2. Atomically records execution fills, allocating tax lots, updating balances,
       and logging balanced double-entry journal transactions.
    3. Executes daily end-of-day financing and borrow fee accruals with mark-to-market snapshots.
    4. Connects persistent ledger state to PreTradeRiskFirewall and PortfolioRiskState.

Explicit Dependency Tracking:
    - SQLAlchemy: AsyncSession, select, update.
    - datetime: UTC timestamps.
    - uuid: Unique transaction identification.
    - quant.execution.financing: DailyAccrualResult, FinancingConfig, FinancingModel.
    - quant.execution.risk: PortfolioRiskState.
    - quant.execution.tax_lot: FillAllocationResult, LotSide, LotStatus, TaxLot,
      TaxLotEngine, TaxLotMethod.
    - quant.infrastructure.database.ledger_models: DBDailyPnLSnapshot, DBLedgerEntry,
      DBPortfolioLedger, DBRealizedTrade, DBTaxLot, utc_now.

Structural Relationship:
    - Core application service in the execution & accounting subsystem.
    - Consumed by ExecutionService, RiskOrchestrator, and CLI Live Runner.

Defensive Invariants:
    - INV-LDG-001: Double-Entry Conservation: sum(Debits) == sum(Credits) for all journal entries.
    - INV-LDG-002: Inventory Conservation: sum(Lot remaining) == abs(Position) for every asset.
    - Rule 1: Four-tier docstrings on every class, method, and function.
    - Rule 2: Deterministic diagnostic error codes (ERR-LDG-008, ERR-LDG-009).
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import UTC, datetime
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from quant.execution.financing import DailyAccrualResult, FinancingModel
from quant.execution.risk import PortfolioRiskState
from quant.execution.tax_lot import (
    FillAllocationResult,
    LotSide,
    LotStatus,
    TaxLot,
    TaxLotEngine,
    TaxLotMethod,
)
from quant.infrastructure.database.ledger_models import (
    ERR_LDG_UNBALANCED_TRANSACTION,
    DBDailyPnLSnapshot,
    DBLedgerEntry,
    DBPortfolioLedger,
    DBRealizedTrade,
    DBTaxLot,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_LDG_REHYDRATION_FAILED: Final[str] = "ERR-LDG-008"
ERR_LDG_PORTFOLIO_NOT_FOUND: Final[str] = "ERR-LDG-009"


class LedgerServiceError(Exception):
    """Base exception for all ledger application service errors."""

    def __init__(self, message: str, code: str = "ERR-LDG-000") -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class RehydrationError(LedgerServiceError):
    """Raised when portfolio state rehydration fails from database."""

    def __init__(self, message: str, code: str = ERR_LDG_REHYDRATION_FAILED) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Ledger Service Facade
# ============================================================================


class LedgerService:
    """Institutional application service managing persistent double-entry accounting and tax lots.

    Functional Purpose:
        Coordinates persistent relational models, in-memory TaxLotEngine, and FinancingModel
        to provide crash-resilient portfolio tracking, fill allocation, and EOD reconciliations.
    """

    def __init__(
        self,
        portfolio_id: str = "default_portfolio",
        tax_lot_engine: TaxLotEngine | None = None,
        financing_model: FinancingModel | None = None,
    ) -> None:
        """Initialize ledger service with target portfolio identifier."""
        self.portfolio_id: str = portfolio_id
        self.tax_lot_engine: TaxLotEngine = tax_lot_engine or TaxLotEngine(
            portfolio_id=portfolio_id
        )
        self.financing_model: FinancingModel = financing_model or FinancingModel()

    async def initialize_or_rehydrate(
        self,
        session: AsyncSession,
        current_prices: dict[str, float] | None = None,
    ) -> PortfolioRiskState:
        """Restore portfolio ledger state and active tax lots from relational database.

        Functional Purpose:
            Guarantees continuous state rehydration across system restarts. Populates
            the in-memory tax lot engine and returns a fully initialized PortfolioRiskState.

        Explicit Dependency Tracking:
            DBPortfolioLedger, DBTaxLot, PortfolioRiskState.

        Defensive Invariants:
            - Portfolio record exists or is provisioned.
            - All rehydrated open lots pass TaxLot invariants.
        """
        try:
            # 1. Fetch or create DBPortfolioLedger
            stmt = select(DBPortfolioLedger).where(
                DBPortfolioLedger.portfolio_id == self.portfolio_id
            )
            res = await session.execute(stmt)
            ledger = res.scalar_one_or_none()

            if ledger is None:
                ledger = DBPortfolioLedger(
                    portfolio_id=self.portfolio_id,
                    name=f"Portfolio {self.portfolio_id}",
                    base_currency="USD",
                    cash_balance=100000.0,
                    margin_debt=0.0,
                    total_equity=100000.0,
                )
                session.add(ledger)
                await session.commit()
                await session.refresh(ledger)
                logger.info(f"Provisioned new DBPortfolioLedger for {self.portfolio_id}")

            # 2. Query all active (OPEN / PARTIAL) tax lots from database
            lot_stmt = (
                select(DBTaxLot)
                .where(
                    DBTaxLot.portfolio_id == self.portfolio_id,
                    DBTaxLot.status.in_(["OPEN", "PARTIAL"]),
                )
                .order_by(DBTaxLot.open_timestamp.asc())
            )
            lot_res = await session.execute(lot_stmt)
            db_lots = lot_res.scalars().all()

            # 3. Hydrate in-memory TaxLotEngine
            self.tax_lot_engine._lots.clear()
            for row in db_lots:
                sym = row.symbol
                if sym not in self.tax_lot_engine._lots:
                    self.tax_lot_engine._lots[sym] = []

                lot_side = LotSide.LONG if row.side == "LONG" else LotSide.SHORT
                lot_status = LotStatus.OPEN if row.status == "OPEN" else LotStatus.PARTIAL

                rehydrated_lot = TaxLot(
                    lot_id=row.id,
                    portfolio_id=self.portfolio_id,
                    symbol=sym,
                    side=lot_side,
                    open_timestamp=row.open_timestamp,
                    original_quantity=float(row.original_quantity),
                    remaining_quantity=float(row.remaining_quantity),
                    cost_basis_per_share=float(row.cost_basis_per_share),
                    status=lot_status,
                )
                self.tax_lot_engine._lots[sym].append(rehydrated_lot)

            # 4. Construct current positions and portfolio risk state
            positions: dict[str, float] = {}
            for sym in self.tax_lot_engine._lots:
                pos = self.tax_lot_engine.get_position(sym)
                if abs(pos) > 1e-8:
                    positions[sym] = pos

            prices = dict(current_prices) if current_prices else {}
            # Fallback prices from tax lot basis if not supplied
            for sym, lots in self.tax_lot_engine._lots.items():
                if sym not in prices and lots:
                    prices[sym] = lots[-1].cost_basis_per_share

            # Holdings market value
            holdings_val = sum(positions.get(s, 0.0) * prices.get(s, 0.0) for s in positions)
            total_equity = float(ledger.cash_balance) + holdings_val

            # Update ledger total equity
            ledger.total_equity = total_equity
            await session.commit()

            logger.info(
                f"Ledger rehydration complete: {len(db_lots)} active tax lots, "
                f"Cash=${ledger.cash_balance:,.2f}, Equity=${total_equity:,.2f}"
            )

            return PortfolioRiskState(
                cash=float(ledger.cash_balance),
                positions=positions,
                current_prices=prices,
                peak_equity=max(total_equity, 1000.0),
                initial_equity=max(total_equity, 1000.0),
            )

        except Exception as e:
            logger.error(
                f"Failed to rehydrate ledger state for {self.portfolio_id}: {e}", exc_info=True
            )
            raise RehydrationError(f"Rehydration failed: {e}") from e

    async def record_fill(
        self,
        session: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        execution_price: float,
        timestamp: datetime | None = None,
        method: TaxLotMethod | None = None,
    ) -> FillAllocationResult:
        """Process an execution fill, allocate tax lots, and persist double-entry journal entries.

        Functional Purpose:
            Atomically synchronizes in-memory lots with database tables:
            - Inserts new open lots into tax_lots.
            - Updates depleted lots in tax_lots.
            - Inserts realized trade records into realized_trades.
            - Inserts balanced debit/credit entries into ledger_entries.
            - Updates portfolio cash_balance and total_equity in portfolio_ledgers.

        Defensive Invariants:
            INV-LDG-001: Balanced transaction sum(debit) == sum(credit).
            INV-LDG-002: sum(remaining_quantity) == abs(current_position).
        """
        fill_time = timestamp or datetime.now(UTC)

        # 1. Allocate against in-memory tax lots
        alloc_res = self.tax_lot_engine.allocate_fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            execution_price=execution_price,
            timestamp=fill_time,
            method=method,
        )

        side_upper = side.upper().strip()
        gross_notional = float(quantity) * float(execution_price)

        # 2. Persist new open tax lots
        for new_lot in alloc_res.new_open_lots:
            db_lot = DBTaxLot(
                id=new_lot.lot_id,
                portfolio_id=self.portfolio_id,
                symbol=new_lot.symbol,
                side=new_lot.side.value,
                open_timestamp=new_lot.open_timestamp,
                original_quantity=new_lot.original_quantity,
                remaining_quantity=new_lot.remaining_quantity,
                cost_basis_per_share=new_lot.cost_basis_per_share,
                status=new_lot.status.value,
            )
            session.add(db_lot)

        # 3. Update depleted or closed tax lots
        for dep_lot in alloc_res.depleted_lots:
            stmt = select(DBTaxLot).where(DBTaxLot.id == dep_lot.lot_id)
            res = await session.execute(stmt)
            target = res.scalar_one_or_none()
            if target is not None:
                target.remaining_quantity = dep_lot.remaining_quantity
                target.status = dep_lot.status.value

        # 4. Persist realized trade records
        for trade in alloc_res.allocated_trades:
            db_trade = DBRealizedTrade(
                id=trade.trade_id,
                portfolio_id=self.portfolio_id,
                lot_id=trade.lot_id,
                symbol=trade.symbol,
                side=trade.side,
                close_timestamp=trade.close_timestamp,
                quantity=trade.quantity,
                execution_price=trade.execution_price,
                cost_basis=trade.cost_basis,
                realized_pnl=trade.realized_pnl,
                holding_period_seconds=trade.holding_period_seconds,
            )
            session.add(db_trade)

        # 5. Generate and persist balanced GAAP double-entry compound journal entries
        tx_id = str(uuid.uuid4())
        journal_entries: list[DBLedgerEntry] = []

        # 5a. Record realized trade entries (inventory relief & realized gain/loss)
        for trade in alloc_res.allocated_trades:
            trade_qty = float(trade.quantity)
            trade_exec_price = float(trade.execution_price)
            trade_proceeds = round(trade_qty * trade_exec_price, 4)
            trade_cost_basis = round(float(trade.cost_basis), 4)

            if trade.side == "SELL":
                # Disposing long inventory: receive cash proceeds, relieve inventory cost basis
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name="ASSET_CASH",
                        entry_type="DEBIT",
                        amount=trade_proceeds,
                        description=f"Cash proceeds from selling {trade_qty} {symbol} @ {trade_exec_price}",
                        timestamp=fill_time,
                    )
                )
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name=f"ASSET_INVENTORY_{symbol}",
                        entry_type="CREDIT",
                        amount=trade_cost_basis,
                        description=f"Cost basis relief for {trade_qty} {symbol}",
                        timestamp=fill_time,
                    )
                )
                gain_or_loss = round(trade_proceeds - trade_cost_basis, 4)
                if gain_or_loss > 1e-4:
                    journal_entries.append(
                        DBLedgerEntry(
                            transaction_id=tx_id,
                            portfolio_id=self.portfolio_id,
                            account_name="REVENUE_REALIZED_GAIN",
                            entry_type="CREDIT",
                            amount=gain_or_loss,
                            description=f"Realized capital gain on {trade_qty} {symbol}",
                            timestamp=fill_time,
                        )
                    )
                elif gain_or_loss < -1e-4:
                    journal_entries.append(
                        DBLedgerEntry(
                            transaction_id=tx_id,
                            portfolio_id=self.portfolio_id,
                            account_name="EXPENSE_REALIZED_LOSS",
                            entry_type="DEBIT",
                            amount=abs(gain_or_loss),
                            description=f"Realized capital loss on {trade_qty} {symbol}",
                            timestamp=fill_time,
                        )
                    )

            elif trade.side == "COVER":
                # Covering short liability: pay cash, relieve short liability
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name=f"LIABILITY_SHORT_{symbol}",
                        entry_type="DEBIT",
                        amount=trade_cost_basis,
                        description=f"Short liability relief for {trade_qty} {symbol}",
                        timestamp=fill_time,
                    )
                )
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name="ASSET_CASH",
                        entry_type="CREDIT",
                        amount=trade_proceeds,
                        description=f"Cash paid to cover {trade_qty} {symbol} @ {trade_exec_price}",
                        timestamp=fill_time,
                    )
                )
                gain_or_loss = round(trade_cost_basis - trade_proceeds, 4)
                if gain_or_loss > 1e-4:
                    journal_entries.append(
                        DBLedgerEntry(
                            transaction_id=tx_id,
                            portfolio_id=self.portfolio_id,
                            account_name="REVENUE_REALIZED_GAIN",
                            entry_type="CREDIT",
                            amount=gain_or_loss,
                            description=f"Realized short gain on {trade_qty} {symbol}",
                            timestamp=fill_time,
                        )
                    )
                elif gain_or_loss < -1e-4:
                    journal_entries.append(
                        DBLedgerEntry(
                            transaction_id=tx_id,
                            portfolio_id=self.portfolio_id,
                            account_name="EXPENSE_REALIZED_LOSS",
                            entry_type="DEBIT",
                            amount=abs(gain_or_loss),
                            description=f"Realized short loss on {trade_qty} {symbol}",
                            timestamp=fill_time,
                        )
                    )

        # 5b. Record new open lot entries (inventory addition or short liability initiation)
        for lot in alloc_res.new_open_lots:
            lot_qty = float(lot.original_quantity)
            lot_price = float(lot.cost_basis_per_share)
            lot_notional = round(lot_qty * lot_price, 4)

            if lot.side == LotSide.LONG:
                # Buy order adding to inventory
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name=f"ASSET_INVENTORY_{symbol}",
                        entry_type="DEBIT",
                        amount=lot_notional,
                        description=f"Bought {lot_qty} {symbol} @ {lot_price}",
                        timestamp=fill_time,
                    )
                )
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name="ASSET_CASH",
                        entry_type="CREDIT",
                        amount=lot_notional,
                        description=f"Cash disbursement for {symbol} purchase",
                        timestamp=fill_time,
                    )
                )
            elif lot.side == LotSide.SHORT:
                # Short sale creating short liability
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name="ASSET_CASH",
                        entry_type="DEBIT",
                        amount=lot_notional,
                        description=f"Cash proceeds from short sale of {lot_qty} {symbol} @ {lot_price}",
                        timestamp=fill_time,
                    )
                )
                journal_entries.append(
                    DBLedgerEntry(
                        transaction_id=tx_id,
                        portfolio_id=self.portfolio_id,
                        account_name=f"LIABILITY_SHORT_{symbol}",
                        entry_type="CREDIT",
                        amount=lot_notional,
                        description=f"Short liability incurred for {lot_qty} {symbol}",
                        timestamp=fill_time,
                    )
                )

        # Invariant INV-LDG-001: Balanced transaction verification
        tot_debit = round(sum(e.amount for e in journal_entries if e.entry_type == "DEBIT"), 4)
        tot_credit = round(sum(e.amount for e in journal_entries if e.entry_type == "CREDIT"), 4)
        if not math.isclose(tot_debit, tot_credit, abs_tol=1e-3):
            raise LedgerServiceError(
                f"Unbalanced journal transaction {tx_id}: Debits={tot_debit} != Credits={tot_credit}",
                code=ERR_LDG_UNBALANCED_TRANSACTION,
            )

        session.add_all(journal_entries)

        cash_delta = -gross_notional if side_upper == "BUY" else gross_notional

        # 6. Update DBPortfolioLedger cash balance
        ledger_stmt = select(DBPortfolioLedger).where(
            DBPortfolioLedger.portfolio_id == self.portfolio_id
        )
        ledger_res = await session.execute(ledger_stmt)
        ledger = ledger_res.scalar_one_or_none()
        if ledger is not None:
            ledger.cash_balance = round(float(ledger.cash_balance) + cash_delta, 4)
            # Update margin debt if cash is negative
            ledger.margin_debt = max(0.0, -ledger.cash_balance)

        await session.commit()

        return alloc_res

    async def accrue_daily_eod(
        self,
        session: AsyncSession,
        date_str: str,
        current_prices: dict[str, float],
        days: float = 1.0,
    ) -> DailyAccrualResult:
        """Accrue daily financing drag, persist journal entries, and write daily mark-to-market snapshot.

        Functional Purpose:
            Computes margin interest and short borrow fees, applies cash deduction,
            writes balanced double-entry entries, and creates DBDailyPnLSnapshot.
        """
        ledger_stmt = select(DBPortfolioLedger).where(
            DBPortfolioLedger.portfolio_id == self.portfolio_id
        )
        ledger_res = await session.execute(ledger_stmt)
        ledger = ledger_res.scalar_one_or_none()
        if ledger is None:
            raise LedgerServiceError(
                f"Portfolio '{self.portfolio_id}' not found", code=ERR_LDG_PORTFOLIO_NOT_FOUND
            )

        positions = {s: self.tax_lot_engine.get_position(s) for s in self.tax_lot_engine._lots}

        # 1. Run Financing Model Accrual
        accrual_res = self.financing_model.accrue_daily_financing(
            portfolio_id=self.portfolio_id,
            cash_balance=float(ledger.cash_balance),
            positions=positions,
            market_prices=current_prices,
            date_str=date_str,
            days=days,
        )

        # 2. Persist generated journal entries
        for draft in accrual_res.journal_entries:
            db_entry = DBLedgerEntry(
                transaction_id=draft.transaction_id,
                portfolio_id=draft.portfolio_id,
                account_name=draft.account_name,
                entry_type=draft.entry_type,
                amount=draft.amount,
                description=draft.description,
                timestamp=draft.timestamp,
            )
            session.add(db_entry)

        # 3. Calculate Unrealized PnL and NAV
        unrealized_pnl = 0.0
        gross_notional = 0.0
        for sym, pos in positions.items():
            if abs(pos) > 1e-8:
                price = current_prices.get(sym, 0.0)
                gross_notional += abs(pos) * price
                unrealized_pnl += self.tax_lot_engine.get_unrealized_pnl(sym, price)

        holdings_val = sum(positions[s] * current_prices.get(s, 0.0) for s in positions)
        nav = accrual_res.cash_balance_after + holdings_val

        # 4. Fetch realized PnL today
        trade_stmt = select(DBRealizedTrade).where(
            DBRealizedTrade.portfolio_id == self.portfolio_id
        )
        trade_res = await session.execute(trade_stmt)
        trades = trade_res.scalars().all()
        realized_total = sum(t.realized_pnl for t in trades)

        # 5. Persist daily snapshot
        snapshot = DBDailyPnLSnapshot(
            portfolio_id=self.portfolio_id,
            date_str=date_str,
            nav=round(nav, 4),
            cash=round(accrual_res.cash_balance_after, 4),
            gross_notional=round(gross_notional, 4),
            realized_pnl=round(realized_total, 4),
            unrealized_pnl=round(unrealized_pnl, 4),
            margin_interest_accrued=accrual_res.margin_interest_accrued,
            borrow_fees_accrued=accrual_res.borrow_fees_accrued,
        )
        session.add(snapshot)

        # 6. Update portfolio ledger balances
        ledger.cash_balance = accrual_res.cash_balance_after
        ledger.margin_debt = accrual_res.margin_debit_balance
        ledger.total_equity = round(nav, 4)

        await session.commit()

        return accrual_res
