"""Unit tests for historical market data domain entities, value objects, and invariants.

Functional Purpose:
    Adversarially tests HistoricalPriceBar, CorporateActionRecord, HistoricalBarBatch,
    and HistoricalDataQuery. Validates all defensive boundary invariants (INV-DATA-004 through
    INV-DATA-007), strict non-finite & boolean rejection, price geometry feasibility, and
    deterministic diagnostic error codes (ERR-DATA-005 through ERR-DATA-010).

Explicit Dependency Tracking:
    - pytest: Test execution framework.
    - math: NaN and Inf generation.
    - quant.domain.historical: Value objects, exception taxonomy, and fault codes.
    - quant.domain.models: Resolution and PriceBar.

Structural Relationship:
    Primary test module for Phase 13 Step 13.1.
"""

import math

import pytest

from quant.domain.historical import (
    ERR_DATA_EMPTY_BATCH,
    ERR_DATA_INVALID_BAR_GEOMETRY,
    ERR_DATA_INVALID_QUERY_RANGE,
    ERR_DATA_INVALID_SPLIT_FACTOR,
    ERR_DATA_NON_FINITE_INPUT,
    ERR_DATA_NON_MONOTONIC_TIMESTAMP,
    CorporateActionRecord,
    CorporateActionType,
    EmptyHistoricalBatchError,
    HistoricalBarBatch,
    HistoricalDataQuery,
    HistoricalPriceBar,
    InvalidBarGeometryError,
    InvalidCorporateActionError,
    InvalidHistoricalQueryError,
    NonFiniteHistoricalInputError,
    NonMonotonicTimestampError,
    PriceAdjustmentType,
)
from quant.domain.models import Resolution

# ============================================================================
# CorporateActionRecord Tests
# ============================================================================


def test_corporate_action_valid_construction() -> None:
    """Validate valid creation of CorporateActionRecord."""
    rec = CorporateActionRecord(
        asset_id="AAPL",
        timestamp=1600000000000000000,
        action_type=CorporateActionType.SPLIT,
        split_ratio=4.0,
        cash_dividend=0.0,
    )
    assert rec.asset_id == "AAPL"
    assert rec.split_ratio == 4.0
    assert rec.cash_dividend == 0.0
    assert rec.action_type == CorporateActionType.SPLIT


def test_corporate_action_rejects_empty_symbol_and_invalid_timestamp() -> None:
    """Ensure CorporateActionRecord rejects empty symbol or non-positive timestamp."""
    with pytest.raises(NonFiniteHistoricalInputError) as exc_info:
        CorporateActionRecord(
            asset_id="",
            timestamp=1000,
            action_type=CorporateActionType.SPLIT,
        )
    assert exc_info.value.code == ERR_DATA_NON_FINITE_INPUT

    with pytest.raises(NonFiniteHistoricalInputError) as exc_info:
        CorporateActionRecord(
            asset_id="AAPL",
            timestamp=-1,
            action_type=CorporateActionType.SPLIT,
        )
    assert exc_info.value.code == ERR_DATA_NON_FINITE_INPUT


def test_corporate_action_rejects_booleans() -> None:
    """Adversarial check: ensure CorporateActionRecord strictly rejects boolean inputs (INV-DATA-007)."""
    with pytest.raises(NonFiniteHistoricalInputError):
        CorporateActionRecord(
            asset_id="AAPL",
            timestamp=True,  # type: ignore[arg-type]
            action_type=CorporateActionType.SPLIT,
        )

    with pytest.raises(NonFiniteHistoricalInputError):
        CorporateActionRecord(
            asset_id="AAPL",
            timestamp=1000000,
            action_type=CorporateActionType.SPLIT,
            split_ratio=True,  # type: ignore[arg-type]
        )

    with pytest.raises(NonFiniteHistoricalInputError):
        CorporateActionRecord(
            asset_id="AAPL",
            timestamp=1000000,
            action_type=CorporateActionType.CASH_DIVIDEND,
            cash_dividend=False,  # type: ignore[arg-type]
        )


def test_corporate_action_rejects_non_finite_and_invalid_ratios() -> None:
    """Reject NaN, Inf, zero, or negative split ratios and negative dividends."""
    with pytest.raises(InvalidCorporateActionError) as exc_info:
        CorporateActionRecord(
            asset_id="AAPL",
            timestamp=1000000,
            action_type=CorporateActionType.SPLIT,
            split_ratio=0.0,
        )
    assert exc_info.value.code == ERR_DATA_INVALID_SPLIT_FACTOR

    with pytest.raises(NonFiniteHistoricalInputError):
        CorporateActionRecord(
            asset_id="AAPL",
            timestamp=1000000,
            action_type=CorporateActionType.SPLIT,
            split_ratio=float("nan"),
        )

    with pytest.raises(InvalidCorporateActionError) as exc_info:
        CorporateActionRecord(
            asset_id="AAPL",
            timestamp=1000000,
            action_type=CorporateActionType.CASH_DIVIDEND,
            cash_dividend=-0.50,
        )
    assert exc_info.value.code == ERR_DATA_INVALID_SPLIT_FACTOR


# ============================================================================
# HistoricalPriceBar Tests
# ============================================================================


def test_historical_price_bar_valid_construction() -> None:
    """Validate construction of valid HistoricalPriceBar."""
    bar = HistoricalPriceBar(
        asset_id="SPY",
        timestamp=1600000000000000000,
        open=400.0,
        high=405.0,
        low=398.0,
        close=402.5,
        volume=1000000.0,
        vwap=401.8,
        resolution=Resolution.ONE_DAY,
        adj_close=402.0,
        split_factor=1.0,
        dividend_amount=1.25,
    )
    assert bar.asset_id == "SPY"
    assert bar.high == 405.0
    assert bar.adj_close == 402.0

    # Test conversion to domain PriceBar
    price_bar = bar.to_price_bar()
    assert price_bar.asset_id == "SPY"
    assert price_bar.close == 402.5
    assert price_bar.volume == 1000000.0


def test_historical_price_bar_split_adjustment() -> None:
    """Validate that to_adjusted_bar scales prices and volume consistently."""
    bar = HistoricalPriceBar(
        asset_id="NVDA",
        timestamp=1600000000000000000,
        open=1000.0,
        high=1020.0,
        low=990.0,
        close=1010.0,
        volume=50000.0,
        vwap=1005.0,
        split_factor=0.10,  # 10-for-1 forward split adjustment
        dividend_amount=0.50,
    )
    adj_bar = bar.to_adjusted_bar()
    assert math.isclose(adj_bar.open, 100.0)
    assert math.isclose(adj_bar.high, 102.0)
    assert math.isclose(adj_bar.low, 99.0)
    assert math.isclose(adj_bar.close, 101.0)
    assert math.isclose(adj_bar.volume, 500000.0)
    assert math.isclose(adj_bar.vwap, 100.5)
    assert math.isclose(adj_bar.split_factor, 1.0)
    assert math.isclose(adj_bar.dividend_amount, 0.05)


def test_historical_price_bar_rejects_geometry_violations() -> None:
    """Enforce physical envelope invariants: High >= max(O,C) and Low <= min(O,C)."""
    # High < Open
    with pytest.raises(InvalidBarGeometryError) as exc_info:
        HistoricalPriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=150.0,
            high=149.0,
            low=140.0,
            close=145.0,
            volume=100.0,
            vwap=145.0,
        )
    assert exc_info.value.code == ERR_DATA_INVALID_BAR_GEOMETRY

    # Low > Close
    with pytest.raises(InvalidBarGeometryError) as exc_info:
        HistoricalPriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=150.0,
            high=155.0,
            low=148.0,
            close=145.0,
            volume=100.0,
            vwap=150.0,
        )
    assert exc_info.value.code == ERR_DATA_INVALID_BAR_GEOMETRY


def test_historical_price_bar_rejects_negative_and_non_finite() -> None:
    """Enforce complete rejection of non-positive prices, negative volumes, and NaNs."""
    # Negative close
    with pytest.raises(InvalidBarGeometryError):
        HistoricalPriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=100.0,
            high=105.0,
            low=95.0,
            close=-10.0,
            volume=100.0,
            vwap=100.0,
        )

    # Negative volume
    with pytest.raises(InvalidBarGeometryError):
        HistoricalPriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=-1.0,
            vwap=100.0,
        )

    # NaN in open
    with pytest.raises(NonFiniteHistoricalInputError) as exc_info:
        HistoricalPriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=float("nan"),
            high=105.0,
            low=95.0,
            close=100.0,
            volume=100.0,
            vwap=100.0,
        )
    assert exc_info.value.code == ERR_DATA_NON_FINITE_INPUT

    # Boolean in volume
    with pytest.raises(NonFiniteHistoricalInputError):
        HistoricalPriceBar(
            asset_id="AAPL",
            timestamp=1000,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=True,  # type: ignore[arg-type]
            vwap=100.0,
        )


# ============================================================================
# HistoricalBarBatch Tests
# ============================================================================


def test_historical_bar_batch_valid_monotonic() -> None:
    """Ensure HistoricalBarBatch correctly handles strictly monotonic bars."""
    bars = (
        HistoricalPriceBar(
            asset_id="MSFT",
            timestamp=1000,
            open=200.0,
            high=205.0,
            low=199.0,
            close=202.0,
            volume=1000.0,
            vwap=202.0,
            adj_close=200.0,
        ),
        HistoricalPriceBar(
            asset_id="MSFT",
            timestamp=2000,
            open=202.0,
            high=206.0,
            low=201.0,
            close=205.0,
            volume=1200.0,
            vwap=204.0,
            adj_close=203.0,
        ),
        HistoricalPriceBar(
            asset_id="MSFT",
            timestamp=3000,
            open=205.0,
            high=208.0,
            low=204.0,
            close=207.0,
            volume=1500.0,
            vwap=206.5,
            adj_close=205.0,
        ),
    )
    batch = HistoricalBarBatch(symbol="MSFT", resolution=Resolution.ONE_DAY, bars=bars)

    assert len(batch) == 3
    assert batch.start_timestamp == 1000
    assert batch.end_timestamp == 3000
    assert batch.timestamps() == (1000, 2000, 3000)
    assert batch.closes() == (202.0, 205.0, 207.0)
    assert batch.adj_closes() == (200.0, 203.0, 205.0)

    # Returns: (203.0 / 200.0 - 1.0, 205.0 / 203.0 - 1.0)
    returns = batch.to_returns(use_adj_close=True)
    assert len(returns) == 2
    assert math.isclose(returns[0], 203.0 / 200.0 - 1.0)
    assert math.isclose(returns[1], 205.0 / 203.0 - 1.0)


def test_historical_bar_batch_rejects_empty() -> None:
    """Ensure HistoricalBarBatch rejects empty bars sequence."""
    with pytest.raises(EmptyHistoricalBatchError) as exc_info:
        HistoricalBarBatch(symbol="MSFT", resolution=Resolution.ONE_DAY, bars=())
    assert exc_info.value.code == ERR_DATA_EMPTY_BATCH


def test_historical_bar_batch_rejects_non_monotonic() -> None:
    """Adversarial check: reject out-of-order or duplicate timestamps (INV-DATA-004)."""
    # Duplicate timestamps
    bars_dup = (
        HistoricalPriceBar(
            asset_id="MSFT",
            timestamp=1000,
            open=200.0,
            high=205.0,
            low=199.0,
            close=202.0,
            volume=1000.0,
            vwap=202.0,
        ),
        HistoricalPriceBar(
            asset_id="MSFT",
            timestamp=1000,  # Duplicate
            open=202.0,
            high=206.0,
            low=201.0,
            close=205.0,
            volume=1200.0,
            vwap=204.0,
        ),
    )
    with pytest.raises(NonMonotonicTimestampError) as exc_info:
        HistoricalBarBatch(symbol="MSFT", resolution=Resolution.ONE_DAY, bars=bars_dup)
    assert exc_info.value.code == ERR_DATA_NON_MONOTONIC_TIMESTAMP

    # Backward clock jump
    bars_backward = (
        HistoricalPriceBar(
            asset_id="MSFT",
            timestamp=2000,
            open=200.0,
            high=205.0,
            low=199.0,
            close=202.0,
            volume=1000.0,
            vwap=202.0,
        ),
        HistoricalPriceBar(
            asset_id="MSFT",
            timestamp=1500,  # Backward jump
            open=202.0,
            high=206.0,
            low=201.0,
            close=205.0,
            volume=1200.0,
            vwap=204.0,
        ),
    )
    with pytest.raises(NonMonotonicTimestampError) as exc_info:
        HistoricalBarBatch(symbol="MSFT", resolution=Resolution.ONE_DAY, bars=bars_backward)
    assert exc_info.value.code == ERR_DATA_NON_MONOTONIC_TIMESTAMP


# ============================================================================
# HistoricalDataQuery Tests
# ============================================================================


def test_historical_data_query_valid_construction() -> None:
    """Validate creation of valid HistoricalDataQuery."""
    q = HistoricalDataQuery(
        symbols=("SPY", "QQQ"),
        start_timestamp=1600000000000000000,
        end_timestamp=1700000000000000000,
        resolution=Resolution.ONE_DAY,
        adjustment=PriceAdjustmentType.SPLIT_ADJUSTED,
    )
    assert q.symbols == ("SPY", "QQQ")
    assert q.start_timestamp < q.end_timestamp
    assert q.adjustment == PriceAdjustmentType.SPLIT_ADJUSTED


def test_historical_data_query_rejects_invalid_ranges() -> None:
    """Ensure query rejects end_timestamp <= start_timestamp and empty symbol tuples."""
    with pytest.raises(InvalidHistoricalQueryError) as exc_info:
        HistoricalDataQuery(
            symbols=(),
            start_timestamp=1000,
            end_timestamp=2000,
        )
    assert exc_info.value.code == ERR_DATA_INVALID_QUERY_RANGE

    with pytest.raises(InvalidHistoricalQueryError) as exc_info:
        HistoricalDataQuery(
            symbols=("SPY",),
            start_timestamp=2000,
            end_timestamp=1000,  # End < Start
        )
    assert exc_info.value.code == ERR_DATA_INVALID_QUERY_RANGE

    with pytest.raises(InvalidHistoricalQueryError) as exc_info:
        HistoricalDataQuery(
            symbols=("SPY",),
            start_timestamp=1000,
            end_timestamp=1000,  # Equal
        )
    assert exc_info.value.code == ERR_DATA_INVALID_QUERY_RANGE
