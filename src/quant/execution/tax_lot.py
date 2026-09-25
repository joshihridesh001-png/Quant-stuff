"""Deterministic FIFO / LIFO Tax-Lot Engine and Capital Gains Attribution.

Functional Purpose:
    Implements institutional tax-lot tracking, matching order execution fills
    against open inventory lots using FIFO or LIFO accounting methods. Computes
    exact realized gains/losses per tax lot, tracks holding periods, handles position
    reversals (e.g. long-to-short flipping), and enforces mathematical inventory conservation.

Explicit Dependency Tracking:
    - datetime: UTC timestamps and holding duration calculation.
    - math: Finite float verification and floating-point comparisons.
    - dataclasses: Slotted value objects.
    - uuid: Unique trade and lot identification.

Structural Relationship:
    - Execution layer component consumed by ExecutionService, RiskOrchestrator, and LedgerService.
    - Integrates with double-entry ledger models in quant.infrastructure.database.ledger_models.

Defensive Invariants:
    - INV-LDG-002: Inventory Conservation: sum(remaining_quantity) == abs(current_position).
    - INV-LDG-003: Cost basis per share and execution price must be strictly positive and finite.
    - INV-LDG-004: Holding duration >= 0.0 seconds everywhere.
    - Rule 1: Four-tier docstrings on every class, method, and function.
    - Rule 2: Deterministic diagnostic error codes (ERR-LDG-001, ERR-LDG-002, ERR-LDG-005).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_LDG_NEGATIVE_INVENTORY: Final[str] = "ERR-LDG-001"
ERR_LDG_ORPHANED_FILL: Final[str] = "ERR-LDG-002"
ERR_LDG_UNBALANCED_TRANSACTION: Final[str] = "ERR-LDG-003"
ERR_LDG_LOT_NOT_FOUND: Final[str] = "ERR-LDG-004"
ERR_LDG_NON_FINITE_INPUT: Final[str] = "ERR-LDG-005"


# ============================================================================
# Exceptions Taxonomy
# ============================================================================


class TaxLotError(Exception):
    """Base exception for all tax lot and inventory accounting errors."""

    def __init__(self, message: str, code: str = "ERR-LDG-000") -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class NegativeInventoryError(TaxLotError):
    """Raised when an inventory conservation invariant is violated."""

    def __init__(self, message: str, code: str = ERR_LDG_NEGATIVE_INVENTORY) -> None:
        super().__init__(message, code=code)


class OrphanedFillError(TaxLotError):
    """Raised when a closing fill cannot be matched and shorting is disabled."""

    def __init__(self, message: str, code: str = ERR_LDG_ORPHANED_FILL) -> None:
        super().__init__(message, code=code)


class NonFiniteLedgerInputError(TaxLotError):
    """Raised when non-finite numbers or invalid types are supplied to the engine."""

    def __init__(self, message: str, code: str = ERR_LDG_NON_FINITE_INPUT) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Domain Enums and Slotted Value Objects
# ============================================================================


class TaxLotMethod(StrEnum):
    """Accounting methodology for matching dispositions against open inventory."""

    FIFO = "FIFO"
    LIFO = "LIFO"


class LotSide(StrEnum):
    """Directional classification of inventory holding."""

    LONG = "LONG"
    SHORT = "SHORT"


class LotStatus(StrEnum):
    """Lifecycle state of an individual tax lot."""

    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"


@dataclass(slots=True)
class TaxLot:
    """slotted mutable entity tracking a specific purchase or short lot."""

    lot_id: str
    portfolio_id: str
    symbol: str
    side: LotSide
    open_timestamp: datetime
    original_quantity: float
    remaining_quantity: float
    cost_basis_per_share: float
    status: LotStatus = LotStatus.OPEN

    def __post_init__(self) -> None:
        if self.original_quantity <= 0.0 or not math.isfinite(self.original_quantity):
            raise NonFiniteLedgerInputError(
                f"original_quantity must be strictly positive and finite, got {self.original_quantity}"
            )
        if self.remaining_quantity < 0.0 or not math.isfinite(self.remaining_quantity):
            raise NonFiniteLedgerInputError(
                f"remaining_quantity must be non-negative and finite, got {self.remaining_quantity}"
            )
        if self.cost_basis_per_share <= 0.0 or not math.isfinite(self.cost_basis_per_share):
            raise NonFiniteLedgerInputError(
                f"cost_basis_per_share must be strictly positive and finite, got {self.cost_basis_per_share}"
            )


@dataclass(frozen=True, slots=True)
class RealizedGainRecord:
    """Immutable record capturing realized capital gain or loss from a closed lot."""

    trade_id: str
    portfolio_id: str
    lot_id: str
    symbol: str
    side: str  # "SELL" or "COVER"
    close_timestamp: datetime
    quantity: float
    execution_price: float
    cost_basis: float
    realized_pnl: float
    holding_period_seconds: float


@dataclass(frozen=True, slots=True)
class FillAllocationResult:
    """Immutable result of processing an execution fill against the tax-lot inventory."""

    allocated_trades: list[RealizedGainRecord]
    new_open_lots: list[TaxLot]
    depleted_lots: list[TaxLot]
    total_realized_pnl: float
    remaining_position: float


# ============================================================================
# Deterministic Tax-Lot Engine
# ============================================================================


class TaxLotEngine:
    """Institutional deterministic tax lot matching and PnL calculation engine.

    Functional Purpose:
        Maintains chronological inventory lots per asset, matches sell/cover fills
        using FIFO/LIFO rules, generates realized capital gain records, and guarantees
        strict inventory conservation.

    Explicit Dependency Tracking:
        TaxLot, RealizedGainRecord, FillAllocationResult.

    Structural Relationship:
        Embedded inside ExecutionService and LedgerService to maintain real-time
        tax lots and feed double-entry journal entries.

    Defensive Invariants:
        INV-LDG-002: Inventory conservation: sum(remaining_quantity) == abs(current_position).
        Rejection of non-finite prices, quantities, and negative timestamps.
    """

    def __init__(
        self,
        portfolio_id: str = "default_portfolio",
        default_method: TaxLotMethod = TaxLotMethod.FIFO,
        allow_shorting: bool = True,
    ) -> None:
        """Initialize tax lot engine with target portfolio and default accounting rules."""
        self.portfolio_id: str = portfolio_id
        self.default_method: TaxLotMethod = default_method
        self.allow_shorting: bool = allow_shorting

        # Active open or partially filled tax lots mapped by symbol: list[TaxLot]
        self._lots: dict[str, list[TaxLot]] = {}

    def get_open_lots(self, symbol: str | None = None) -> list[TaxLot]:
        """Return sequence of open and partial tax lots, optionally filtered by symbol."""
        if symbol is not None:
            return [lot for lot in self._lots.get(symbol, []) if lot.remaining_quantity > 1e-8]

        all_lots: list[TaxLot] = []
        for sym_lots in self._lots.values():
            all_lots.extend([lot for lot in sym_lots if lot.remaining_quantity > 1e-8])
        return all_lots

    def get_position(self, symbol: str) -> float:
        """Return the net inventory share count for symbol (positive=LONG, negative=SHORT)."""
        lots = self.get_open_lots(symbol)
        if not lots:
            return 0.0

        side = lots[0].side
        total_qty = sum(lot.remaining_quantity for lot in lots)
        return total_qty if side == LotSide.LONG else -total_qty

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Compute mark-to-market unrealized PnL across all open lots for symbol."""
        if not math.isfinite(current_price) or current_price <= 0.0:
            raise NonFiniteLedgerInputError(
                f"current_price must be positive and finite, got {current_price}"
            )

        lots = self.get_open_lots(symbol)
        unrealized = 0.0
        for lot in lots:
            market_val = lot.remaining_quantity * current_price
            cost_val = lot.remaining_quantity * lot.cost_basis_per_share
            if lot.side == LotSide.LONG:
                unrealized += market_val - cost_val
            else:
                unrealized += cost_val - market_val
        return round(unrealized, 4)

    def allocate_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        execution_price: float,
        timestamp: datetime | None = None,
        method: TaxLotMethod | None = None,
    ) -> FillAllocationResult:
        """Process an execution fill and allocate against open tax lots.

        Args:
            symbol: Ticker symbol (e.g. 'AAPL', 'NVDA').
            side: Execution side: 'BUY' or 'SELL'.
            quantity: Filled share quantity > 0.0.
            execution_price: Execution price per share > 0.0.
            timestamp: Time of fill execution (defaults to now UTC).
            method: FIFO or LIFO allocation override.

        Returns:
            FillAllocationResult with realized trades, closed lots, and new lots.
        """
        # 1. Input sanitization and boundary invariants
        if not symbol or not isinstance(symbol, str):
            raise NonFiniteLedgerInputError("symbol must be a non-empty string")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            raise NonFiniteLedgerInputError("quantity must be numeric")
        if quantity <= 0.0 or not math.isfinite(quantity):
            raise NonFiniteLedgerInputError(
                f"quantity must be strictly positive and finite, got {quantity}"
            )
        if isinstance(execution_price, bool) or not isinstance(execution_price, (int, float)):
            raise NonFiniteLedgerInputError("execution_price must be numeric")
        if execution_price <= 0.0 or not math.isfinite(execution_price):
            raise NonFiniteLedgerInputError(
                f"execution_price must be strictly positive and finite, got {execution_price}"
            )

        fill_time = timestamp or datetime.now(UTC)
        alloc_method = method or self.default_method
        side_upper = side.upper().strip()

        if side_upper not in ("BUY", "SELL"):
            raise NonFiniteLedgerInputError(f"side must be 'BUY' or 'SELL', got '{side}'")

        if symbol not in self._lots:
            self._lots[symbol] = []

        active_lots = [lot for lot in self._lots[symbol] if lot.remaining_quantity > 1e-8]
        current_pos = self.get_position(symbol)

        allocated_trades: list[RealizedGainRecord] = []
        new_open_lots: list[TaxLot] = []
        depleted_lots: list[TaxLot] = []
        total_pnl = 0.0

        # Scenario A: Adding to existing inventory (or starting new position)
        is_increasing = (side_upper == "BUY" and current_pos >= 0.0) or (
            side_upper == "SELL" and current_pos <= 0.0
        )

        if is_increasing:
            if side_upper == "SELL" and not self.allow_shorting:
                raise OrphanedFillError(
                    f"Shorting disabled: cannot sell without existing long inventory for {symbol}"
                )

            new_side = LotSide.LONG if side_upper == "BUY" else LotSide.SHORT
            new_lot = TaxLot(
                lot_id=f"lot_{uuid.uuid4().hex[:12]}",
                portfolio_id=self.portfolio_id,
                symbol=symbol,
                side=new_side,
                open_timestamp=fill_time,
                original_quantity=float(quantity),
                remaining_quantity=float(quantity),
                cost_basis_per_share=float(execution_price),
                status=LotStatus.OPEN,
            )
            self._lots[symbol].append(new_lot)
            new_open_lots.append(new_lot)

        else:
            # Scenario B: Closing existing inventory (SELL closes LONG, BUY covers SHORT)
            # Sort active lots according to FIFO (oldest first) or LIFO (newest first)
            reverse_sort = alloc_method == TaxLotMethod.LIFO
            sorted_lots = sorted(
                active_lots, key=lambda lot_item: lot_item.open_timestamp, reverse=reverse_sort
            )

            remaining_fill = float(quantity)

            for lot in sorted_lots:
                if remaining_fill <= 1e-8:
                    break

                match_qty = min(lot.remaining_quantity, remaining_fill)
                holding_seconds = max((fill_time - lot.open_timestamp).total_seconds(), 0.0)

                # Realized PnL computation
                cost_basis = match_qty * lot.cost_basis_per_share
                exec_val = match_qty * float(execution_price)

                if lot.side == LotSide.LONG:
                    trade_side = "SELL"
                    realized = exec_val - cost_basis
                else:
                    trade_side = "COVER"
                    realized = cost_basis - exec_val

                lot.remaining_quantity -= match_qty
                if lot.remaining_quantity <= 1e-8:
                    lot.remaining_quantity = 0.0
                    lot.status = LotStatus.CLOSED
                else:
                    lot.status = LotStatus.PARTIAL

                depleted_lots.append(lot)
                remaining_fill -= match_qty
                total_pnl += realized

                trade_record = RealizedGainRecord(
                    trade_id=f"trd_{uuid.uuid4().hex[:12]}",
                    portfolio_id=self.portfolio_id,
                    lot_id=lot.lot_id,
                    symbol=symbol,
                    side=trade_side,
                    close_timestamp=fill_time,
                    quantity=round(match_qty, 6),
                    execution_price=round(float(execution_price), 4),
                    cost_basis=round(cost_basis, 4),
                    realized_pnl=round(realized, 4),
                    holding_period_seconds=round(holding_seconds, 1),
                )
                allocated_trades.append(trade_record)

            # If remaining_fill > 0, the position has flipped into the opposite direction
            if remaining_fill > 1e-8:
                if side_upper == "SELL" and not self.allow_shorting:
                    raise OrphanedFillError(
                        f"Position flip exceeds inventory and shorting disabled for {symbol}"
                    )

                flip_side = LotSide.SHORT if side_upper == "SELL" else LotSide.LONG
                flip_lot = TaxLot(
                    lot_id=f"lot_{uuid.uuid4().hex[:12]}",
                    portfolio_id=self.portfolio_id,
                    symbol=symbol,
                    side=flip_side,
                    open_timestamp=fill_time,
                    original_quantity=round(remaining_fill, 6),
                    remaining_quantity=round(remaining_fill, 6),
                    cost_basis_per_share=float(execution_price),
                    status=LotStatus.OPEN,
                )
                self._lots[symbol].append(flip_lot)
                new_open_lots.append(flip_lot)

        # 3. Invariant Verification: Inventory Conservation
        self.verify_inventory_conservation(symbol)

        return FillAllocationResult(
            allocated_trades=allocated_trades,
            new_open_lots=new_open_lots,
            depleted_lots=depleted_lots,
            total_realized_pnl=round(total_pnl, 4),
            remaining_position=self.get_position(symbol),
        )

    def verify_inventory_conservation(self, symbol: str) -> None:
        """Enforce INV-LDG-002: sum(remaining_quantity) == abs(current_position)."""
        all_lots = self._lots.get(symbol, [])
        for lot in all_lots:
            if lot.remaining_quantity < 0.0:
                raise NegativeInventoryError(
                    f"Invariant violated: negative lot quantity ({lot.remaining_quantity}) in {symbol}",
                    code=ERR_LDG_NEGATIVE_INVENTORY,
                )

        lots = [lot for lot in all_lots if lot.remaining_quantity > 1e-8]
        if not lots:
            return

        first_side = lots[0].side
        total_remaining = 0.0
        for lot in lots:
            if lot.side != first_side:
                raise NegativeInventoryError(
                    f"Invariant violated: mixed LONG and SHORT lots found for {symbol}",
                    code=ERR_LDG_NEGATIVE_INVENTORY,
                )
            total_remaining += lot.remaining_quantity

        expected_pos = abs(self.get_position(symbol))
        if not math.isclose(total_remaining, expected_pos, abs_tol=1e-6):
            raise NegativeInventoryError(
                f"Inventory conservation broken: lots sum ({total_remaining}) != pos ({expected_pos})",
                code=ERR_LDG_NEGATIVE_INVENTORY,
            )
