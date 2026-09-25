"""Unit tests for Financing Costs and Short Borrow Fee Model.

Functional Purpose:
    Verifies margin interest calculations, short borrow fee schedules (GC and HTB),
    end-of-day financing accruals, balanced double-entry journal creation, and input sanitization.

Explicit Dependency Tracking:
    - pytest, math.
    - quant.execution.financing: FinancingModel, FinancingConfig, InvalidRateError,
      FinancingError.
    - quant.execution.tax_lot: NonFiniteLedgerInputError.
"""

from __future__ import annotations

import math

import pytest

from quant.execution.financing import (
    FinancingConfig,
    FinancingError,
    FinancingModel,
    InvalidRateError,
)
from quant.execution.tax_lot import NonFiniteLedgerInputError


def test_margin_interest_calculation() -> None:
    """Verify margin interest calculation on positive vs negative cash balances."""
    model = FinancingModel(FinancingConfig(annual_margin_rate=0.065, day_count_convention=360))

    # Positive cash -> zero margin interest
    assert model.calculate_margin_interest(cash_balance=100000.0, days=1.0) == 0.0
    assert model.calculate_margin_interest(cash_balance=0.0, days=1.0) == 0.0

    # Negative cash (-$100,000 debit balance)
    # Expected 1-day interest: 100,000 * (0.065 / 360) * 1 = 18.0556
    interest_1d = model.calculate_margin_interest(cash_balance=-100000.0, days=1.0)
    assert math.isclose(interest_1d, 18.0556, abs_tol=1e-4)

    # 3-day weekend interest: 100,000 * (0.065 / 360) * 3 = 54.1667
    interest_3d = model.calculate_margin_interest(cash_balance=-100000.0, days=3.0)
    assert math.isclose(interest_3d, 54.1667, abs_tol=1e-4)


def test_short_borrow_fee_general_collateral_and_htb() -> None:
    """Verify borrow fee calculation across general collateral and hard-to-borrow assets."""
    config = FinancingConfig(
        default_borrow_rate=0.015,  # 1.5% general collateral
        hard_to_borrow_rates={"TSLA": 0.08, "GME": 0.25},  # 8% and 25% HTB rates
        day_count_convention=360,
    )
    model = FinancingModel(config)

    positions = {
        "AAPL": 100.0,  # Long 100 shares -> 0 borrow fee
        "SPY": -100.0,  # Short 100 shares GC @ $400 ($40,000 notional)
        "TSLA": -50.0,  # Short 50 shares HTB @ $200 ($10,000 notional)
    }
    market_prices = {
        "AAPL": 150.0,
        "SPY": 400.0,
        "TSLA": 200.0,
    }

    # Expected fees for 1 day:
    # SPY: 40,000 * (0.015 / 360) * 1 = 1.6667
    # TSLA: 10,000 * (0.08 / 360) * 1 = 2.2222
    # Total = 3.8889
    total_fees, short_notional, per_sym = model.calculate_borrow_fees(
        positions=positions, market_prices=market_prices, days=1.0
    )

    assert math.isclose(short_notional, 50000.0)
    assert math.isclose(total_fees, 3.8889, abs_tol=1e-4)
    assert "AAPL" not in per_sym
    assert math.isclose(per_sym["SPY"], 1.6667, abs_tol=1e-4)
    assert math.isclose(per_sym["TSLA"], 2.2222, abs_tol=1e-4)


def test_accrue_daily_financing_balanced_journal() -> None:
    """Verify full daily accrual generates balanced double-entry journal transactions."""
    config = FinancingConfig(annual_margin_rate=0.065, default_borrow_rate=0.02)
    model = FinancingModel(config)

    cash_start = -50000.0  # Debit balance
    positions = {"NVDA": -40.0}  # Short 40 NVDA @ $120 ($4,800 notional)
    prices = {"NVDA": 120.0}

    result = model.accrue_daily_financing(
        portfolio_id="test_port_001",
        cash_balance=cash_start,
        positions=positions,
        market_prices=prices,
        date_str="2024-06-15",
        days=1.0,
    )

    assert result.margin_interest_accrued > 0.0
    assert result.borrow_fees_accrued > 0.0
    assert result.total_financing_drag > 0.0
    assert math.isclose(
        result.cash_balance_after, cash_start - result.total_financing_drag, abs_tol=1e-4
    )

    # Verify journal entries adhere to INV-LDG-001
    assert len(result.journal_entries) == 4  # 2 for margin, 2 for borrow
    debit_total = sum(e.amount for e in result.journal_entries if e.entry_type == "DEBIT")
    credit_total = sum(e.amount for e in result.journal_entries if e.entry_type == "CREDIT")
    assert math.isclose(debit_total, credit_total, abs_tol=1e-6)
    assert math.isclose(debit_total, result.total_financing_drag, abs_tol=1e-6)


def test_invalid_rate_and_non_finite_rejection() -> None:
    """Verify input validation rejects invalid rates, day counts, and non-finite values."""
    with pytest.raises(InvalidRateError):
        FinancingConfig(annual_margin_rate=-0.05)

    with pytest.raises(InvalidRateError):
        FinancingConfig(default_borrow_rate=-0.01)

    with pytest.raises(FinancingError):
        FinancingConfig(day_count_convention=364)  # Invalid convention

    model = FinancingModel()

    with pytest.raises(NonFiniteLedgerInputError):
        model.calculate_margin_interest(cash_balance=float("nan"))

    with pytest.raises(NonFiniteLedgerInputError):
        model.calculate_margin_interest(cash_balance=-10000.0, days=-1.0)

    with pytest.raises(NonFiniteLedgerInputError):
        model.calculate_margin_interest(cash_balance=True)  # Bool rejection
