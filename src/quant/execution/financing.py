"""Institutional Financing Costs & Short Borrow Fee Accounting Model.

Functional Purpose:
    Calculates daily institutional financing drag:
    1. Margin debit interest: Accrued when cash balance is negative under leverage.
    2. Short borrow fees: Accrued daily on short equity positions against general
       collateral or hard-to-borrow (HTB) fee schedules.
    Emits balanced GAAP double-entry journal transactions for the persistent ledger.

Explicit Dependency Tracking:
    - dataclasses: Slotted value objects and configuration.
    - datetime: Date and timestamping.
    - math: Finite float verification and money market math.
    - uuid: Unique transaction identification.
    - quant.execution.tax_lot: NonFiniteLedgerInputError, ERR_LDG_NON_FINITE_INPUT.

Structural Relationship:
    - Execution layer component called during daily end-of-day reconciliation.
    - Supplies financing drag records to LedgerService, DailyPnLSnapshot, and PreTradeRiskFirewall.

Defensive Invariants:
    - INV-LDG-001: Balanced double-entry transactions: sum(Debits) == sum(Credits).
    - Financing rates r_margin >= 0.0 and r_borrow >= 0.0 everywhere.
    - Accrual periods days > 0.0.
    - Rule 1: Four-tier docstrings on every class, method, and function.
    - Rule 2: Catalog diagnostic codes (ERR-LDG-006, ERR-LDG-007, ERR-LDG-005).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from quant.execution.tax_lot import NonFiniteLedgerInputError

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_LDG_FINANCING_CALCULATION: Final[str] = "ERR-LDG-006"
ERR_LDG_INVALID_RATE: Final[str] = "ERR-LDG-007"
ERR_LDG_NON_FINITE_INPUT: Final[str] = "ERR-LDG-005"


# ============================================================================
# Exception Taxonomy
# ============================================================================


class FinancingError(Exception):
    """Base exception for financing cost and borrow fee calculation failures."""

    def __init__(self, message: str, code: str = ERR_LDG_FINANCING_CALCULATION) -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class InvalidRateError(FinancingError):
    """Raised when an interest or borrow rate is negative or out of bounds."""

    def __init__(self, message: str, code: str = ERR_LDG_INVALID_RATE) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Value Objects and Configurations
# ============================================================================


@dataclass(slots=True)
class FinancingConfig:
    """slotted configuration parameters for financing rates and conventions."""

    annual_margin_rate: float = 0.065  # 6.5% broker margin rate
    default_borrow_rate: float = 0.015  # 1.5% general collateral borrow rate
    hard_to_borrow_rates: dict[str, float] = field(default_factory=dict)
    day_count_convention: int = 360  # Money market standard: ACT/360

    def __post_init__(self) -> None:
        if isinstance(self.annual_margin_rate, bool) or not isinstance(
            self.annual_margin_rate, (int, float)
        ):
            raise NonFiniteLedgerInputError(
                "annual_margin_rate must be numeric", code=ERR_LDG_NON_FINITE_INPUT
            )
        if self.annual_margin_rate < 0.0 or not math.isfinite(self.annual_margin_rate):
            raise InvalidRateError(
                f"annual_margin_rate must be non-negative, got {self.annual_margin_rate}"
            )

        if isinstance(self.default_borrow_rate, bool) or not isinstance(
            self.default_borrow_rate, (int, float)
        ):
            raise NonFiniteLedgerInputError(
                "default_borrow_rate must be numeric", code=ERR_LDG_NON_FINITE_INPUT
            )
        if self.default_borrow_rate < 0.0 or not math.isfinite(self.default_borrow_rate):
            raise InvalidRateError(
                f"default_borrow_rate must be non-negative, got {self.default_borrow_rate}"
            )

        if self.day_count_convention not in (360, 365):
            raise FinancingError(
                f"day_count_convention must be 360 or 365, got {self.day_count_convention}"
            )


@dataclass(frozen=True, slots=True)
class JournalEntryDraft:
    """Draft representation of a balanced double-entry journal transaction."""

    transaction_id: str
    portfolio_id: str
    account_name: str
    entry_type: str  # "DEBIT" or "CREDIT"
    amount: float
    description: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class DailyAccrualResult:
    """Immutable result of end-of-day financing and borrow fee calculations."""

    portfolio_id: str
    date_str: str
    cash_balance_before: float
    margin_debit_balance: float
    margin_interest_accrued: float
    short_borrow_notional: float
    borrow_fees_accrued: float
    total_financing_drag: float
    cash_balance_after: float
    journal_entries: tuple[JournalEntryDraft, ...]


# ============================================================================
# Financing and Borrow Fee Model
# ============================================================================


class FinancingModel:
    """Institutional financing drag calculator and double-entry transaction generator.

    Functional Purpose:
        Computes accurate margin interest on debit balances and short borrow fees
        on short equity inventory, emitting balanced GAAP journal entries.

    Explicit Dependency Tracking:
        FinancingConfig, DailyAccrualResult, JournalEntryDraft.

    Structural Relationship:
        Executed by LedgerService during daily accounting snapshots.

    Defensive Invariants:
        INV-LDG-001: sum(Debits) == sum(Credits) for all generated journal entries.
        Rejection of non-finite market prices, cash balances, or positions.
    """

    def __init__(self, config: FinancingConfig | None = None) -> None:
        """Initialize financing model with configuration."""
        self.config: FinancingConfig = config or FinancingConfig()

    def calculate_margin_interest(self, cash_balance: float, days: float = 1.0) -> float:
        """Calculate margin debit interest for a given cash balance and day count.

        Functional Purpose:
            Computes interest charged on borrowed funds when cash is negative.
        Formula:
            Interest = max(0, -cash) * (annual_margin_rate / day_count_convention) * days
        """
        if isinstance(cash_balance, bool) or not isinstance(cash_balance, (int, float)):
            raise NonFiniteLedgerInputError(
                "cash_balance must be numeric", code=ERR_LDG_NON_FINITE_INPUT
            )
        if not math.isfinite(cash_balance):
            raise NonFiniteLedgerInputError(
                "cash_balance must be finite", code=ERR_LDG_NON_FINITE_INPUT
            )
        if isinstance(days, bool) or not isinstance(days, (int, float)):
            raise NonFiniteLedgerInputError("days must be numeric", code=ERR_LDG_NON_FINITE_INPUT)
        if days < 0.0 or not math.isfinite(days):
            raise NonFiniteLedgerInputError(f"days must be non-negative and finite, got {days}")

        if cash_balance >= 0.0 or days == 0.0:
            return 0.0

        debit = abs(float(cash_balance))
        daily_rate = self.config.annual_margin_rate / float(self.config.day_count_convention)
        interest = debit * daily_rate * float(days)
        return round(interest, 4)

    def calculate_borrow_fees(
        self,
        positions: dict[str, float],
        market_prices: dict[str, float],
        days: float = 1.0,
    ) -> tuple[float, float, dict[str, float]]:
        """Calculate daily short borrow fees across all short positions.

        Args:
            positions: Mapping of ticker symbol to net share position.
            market_prices: Mapping of ticker symbol to current market price.
            days: Day count fraction (default 1.0).

        Returns:
            Tuple of (total_borrow_fees, total_short_notional, per_symbol_fees).
        """
        if isinstance(days, bool) or not isinstance(days, (int, float)):
            raise NonFiniteLedgerInputError("days must be numeric", code=ERR_LDG_NON_FINITE_INPUT)
        if days < 0.0 or not math.isfinite(days):
            raise NonFiniteLedgerInputError(f"days must be non-negative, got {days}")

        total_fees = 0.0
        total_short_notional = 0.0
        per_symbol_fees: dict[str, float] = {}

        for sym, qty in positions.items():
            if isinstance(qty, bool) or not isinstance(qty, (int, float)):
                raise NonFiniteLedgerInputError(f"Position for {sym} must be numeric")
            if not math.isfinite(qty):
                raise NonFiniteLedgerInputError(f"Position for {sym} must be finite")

            # Only short positions incur borrow fees (qty < 0)
            if qty >= 0.0:
                continue

            price = market_prices.get(sym)
            if price is None or isinstance(price, bool) or not isinstance(price, (int, float)):
                raise NonFiniteLedgerInputError(
                    f"Missing or non-numeric price for short symbol {sym}"
                )
            if price <= 0.0 or not math.isfinite(price):
                raise NonFiniteLedgerInputError(
                    f"Price for short symbol {sym} must be positive and finite"
                )

            short_notional = abs(float(qty)) * float(price)
            total_short_notional += short_notional

            # Use Hard-to-Borrow rate if defined, else general collateral default rate
            borrow_rate = self.config.hard_to_borrow_rates.get(sym, self.config.default_borrow_rate)
            daily_borrow_rate = borrow_rate / float(self.config.day_count_convention)
            fee = short_notional * daily_borrow_rate * float(days)

            fee_rounded = round(fee, 4)
            per_symbol_fees[sym] = fee_rounded
            total_fees += fee_rounded

        return round(total_fees, 4), round(total_short_notional, 2), per_symbol_fees

    def accrue_daily_financing(
        self,
        portfolio_id: str,
        cash_balance: float,
        positions: dict[str, float],
        market_prices: dict[str, float],
        date_str: str,
        days: float = 1.0,
    ) -> DailyAccrualResult:
        """Execute complete daily financing accrual and generate balanced journal entries.

        Functional Purpose:
            Calculates margin interest and short borrow fees, applies cash deduction,
            and produces verifiable GAAP double-entry journal transactions.
        """
        margin_interest = self.calculate_margin_interest(cash_balance=cash_balance, days=days)
        borrow_fees, short_notional, _ = self.calculate_borrow_fees(
            positions=positions, market_prices=market_prices, days=days
        )

        total_drag = round(margin_interest + borrow_fees, 4)
        margin_debit = max(0.0, -float(cash_balance))
        now = datetime.now(UTC)

        entries: list[JournalEntryDraft] = []

        # 1. Margin interest journal entry (if > 0)
        if margin_interest > 1e-6:
            tx_id_margin = str(uuid.uuid4())
            debit_entry = JournalEntryDraft(
                transaction_id=tx_id_margin,
                portfolio_id=portfolio_id,
                account_name="EXPENSE_MARGIN_INTEREST",
                entry_type="DEBIT",
                amount=margin_interest,
                description=f"Margin interest debit for {date_str} ({days:.1f} days @ {self.config.annual_margin_rate * 100:.2f}%)",
                timestamp=now,
            )
            credit_entry = JournalEntryDraft(
                transaction_id=tx_id_margin,
                portfolio_id=portfolio_id,
                account_name="ASSET_CASH",
                entry_type="CREDIT",
                amount=margin_interest,
                description=f"Cash deduction for margin interest on {date_str}",
                timestamp=now,
            )
            entries.extend([debit_entry, credit_entry])

        # 2. Short borrow fee journal entry (if > 0)
        if borrow_fees > 1e-6:
            tx_id_borrow = str(uuid.uuid4())
            debit_entry = JournalEntryDraft(
                transaction_id=tx_id_borrow,
                portfolio_id=portfolio_id,
                account_name="EXPENSE_SHORT_BORROW",
                entry_type="DEBIT",
                amount=borrow_fees,
                description=f"Short borrow fees for {date_str} ({days:.1f} days on ${short_notional:,.2f} short notional)",
                timestamp=now,
            )
            credit_entry = JournalEntryDraft(
                transaction_id=tx_id_borrow,
                portfolio_id=portfolio_id,
                account_name="ASSET_CASH",
                entry_type="CREDIT",
                amount=borrow_fees,
                description=f"Cash deduction for short borrow fees on {date_str}",
                timestamp=now,
            )
            entries.extend([debit_entry, credit_entry])

        # Verify INV-LDG-001: Double-Entry Balance
        debit_sum = sum(e.amount for e in entries if e.entry_type == "DEBIT")
        credit_sum = sum(e.amount for e in entries if e.entry_type == "CREDIT")
        if not math.isclose(debit_sum, credit_sum, abs_tol=1e-6):
            raise FinancingError(
                f"Double-entry balance broken: debits ({debit_sum}) != credits ({credit_sum})"
            )

        cash_after = round(float(cash_balance) - total_drag, 4)

        return DailyAccrualResult(
            portfolio_id=portfolio_id,
            date_str=date_str,
            cash_balance_before=float(cash_balance),
            margin_debit_balance=margin_debit,
            margin_interest_accrued=margin_interest,
            short_borrow_notional=short_notional,
            borrow_fees_accrued=borrow_fees,
            total_financing_drag=total_drag,
            cash_balance_after=cash_after,
            journal_entries=tuple(entries),
        )
