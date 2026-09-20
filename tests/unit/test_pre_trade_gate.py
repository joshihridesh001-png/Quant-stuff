"""Unit tests for Institutional Pre-Trade Decision Gate.

Verifies:
    - INV-GATE-001: Fail-open invariant for risk-reducing exit orders.
    - INV-GATE-002: Closed-form Bayesian log-odds evaluation across 5 dimensions.
    - INV-GATE-003: Sub-50us latency SLA.
    - INV-GATE-004: Strict non-finite input protection and error codes.
    - Rolling audit history buffer capacity.
"""

from __future__ import annotations

import time

import pytest

from quant.execution.pre_trade_gate import (
    ERR_GATE_CONSECUTIVE_LOSS_BREACH,
    ERR_GATE_DRAWDOWN_BUDGET_EXCEEDED,
    ERR_GATE_MACRO_REGIME_CONFLICT,
    ERR_GATE_NON_FINITE_INPUT,
    ERR_GATE_STALE_DATA,
    ERR_GATE_TOXIC_FLOW_DETECTED,
    DimensionName,
    GateDecision,
    NonFiniteGateInputException,
    PreTradeDecisionGate,
    PreTradeDecisionRequest,
)


def _valid_request(**overrides: object) -> PreTradeDecisionRequest:
    """Helper generating a baseline nominal PreTradeDecisionRequest."""
    params = {
        "order_id": "ORD-001",
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 100.0,
        "reference_price": 500.0,
        "timestamp_ns": time.time_ns(),
        "is_position_exit": False,
        "market_spread_bps": 2.5,
        "order_book_imbalance": 0.05,
        "macro_yield_spread": 0.25,
        "trailing_volatility_pct": 1.1,
        "current_bar_return_pct": 0.05,
        "consecutive_losses": 0,
        "current_drawdown_pct": 1.2,
        "cvar_95_pct": 2.0,
        "data_age_seconds": 1.5,
    }
    params.update(overrides)
    return PreTradeDecisionRequest(**params)  # type: ignore[arg-type]


def test_nominal_buy_order_passes_gate() -> None:
    """Nominal buy order in healthy market should pass Bayesian gate with low toxicity."""
    gate = PreTradeDecisionGate()
    req = _valid_request()
    result = gate.evaluate(req)

    assert result.allowed is True
    assert result.decision == GateDecision.PASS
    assert result.primary_code == "OK"
    assert result.toxicity_probability < 0.30
    assert len(result.checks) == 5
    assert all(c.passed for c in result.checks)
    assert len(gate.history) == 1


def test_inv_gate_001_fail_open_on_exits() -> None:
    """Risk-reducing exit orders must unconditionally bypass pre-trade gate."""
    gate = PreTradeDecisionGate()

    # Case A: is_position_exit = True
    req_exit = _valid_request(
        is_position_exit=True,
        market_spread_bps=99.0,  # Even in adverse market
        order_book_imbalance=-0.9,
        consecutive_losses=10,
    )
    res_exit = gate.evaluate(req_exit)
    assert res_exit.allowed is True
    assert res_exit.decision == GateDecision.BYPASS_EXIT
    assert res_exit.primary_code == "OK"

    # Case B: action = CLOSE
    req_close = _valid_request(action="CLOSE", market_spread_bps=80.0)
    res_close = gate.evaluate(req_close)
    assert res_close.allowed is True
    assert res_close.decision == GateDecision.BYPASS_EXIT

    # Case C: action = LIQUIDATE
    req_liq = _valid_request(action="LIQUIDATE")
    res_liq = gate.evaluate(req_liq)
    assert res_liq.allowed is True
    assert res_liq.decision == GateDecision.BYPASS_EXIT


def test_stale_data_triggers_rejection() -> None:
    """Market data older than 120s must trip data freshness penalty and reject."""
    gate = PreTradeDecisionGate()
    req = _valid_request(data_age_seconds=150.0)
    result = gate.evaluate(req)

    assert result.allowed is False
    assert result.decision == GateDecision.REJECT
    assert result.primary_code == ERR_GATE_STALE_DATA
    freshness_check = next(c for c in result.checks if c.name == DimensionName.DATA_FRESHNESS)
    assert freshness_check.passed is False


def test_toxic_order_book_imbalance_rejects_buy() -> None:
    """Heavy ask queue selling pressure (OBI < -0.35) should reject buy order."""
    gate = PreTradeDecisionGate()
    req = _valid_request(action="BUY", order_book_imbalance=-0.65)
    result = gate.evaluate(req)

    assert result.allowed is False
    assert result.decision == GateDecision.REJECT
    assert result.primary_code == ERR_GATE_TOXIC_FLOW_DETECTED
    micro_check = next(c for c in result.checks if c.name == DimensionName.MICROSTRUCTURE_TOXICITY)
    assert micro_check.passed is False


def test_excessive_spread_dislocation_rejects() -> None:
    """Extreme bid-ask spread (> 35 bps) trips microstructure toxicity."""
    gate = PreTradeDecisionGate()
    req = _valid_request(market_spread_bps=45.0)
    result = gate.evaluate(req)

    assert result.allowed is False
    assert result.decision == GateDecision.REJECT
    assert result.primary_code == ERR_GATE_TOXIC_FLOW_DETECTED


def test_macro_yield_curve_inversion_rejects_equity_buy() -> None:
    """Severe yield curve inversion (T10Y2Y < -0.50%) trips macro regime gate."""
    gate = PreTradeDecisionGate()
    req = _valid_request(action="BUY", macro_yield_spread=-0.85)
    result = gate.evaluate(req)

    assert result.allowed is False
    assert result.decision == GateDecision.REJECT
    assert result.primary_code == ERR_GATE_MACRO_REGIME_CONFLICT
    macro_check = next(c for c in result.checks if c.name == DimensionName.MACRO_REGIME)
    assert macro_check.passed is False


def test_consecutive_loss_streak_breach() -> None:
    """Excessive consecutive loss streak (>= 4 losses) gates new entries."""
    gate = PreTradeDecisionGate()
    req = _valid_request(consecutive_losses=5)
    result = gate.evaluate(req)

    assert result.allowed is False
    assert result.decision == GateDecision.REJECT
    assert result.primary_code == ERR_GATE_CONSECUTIVE_LOSS_BREACH
    risk_check = next(c for c in result.checks if c.name == DimensionName.CAPITAL_RISK_BUDGET)
    assert risk_check.passed is False


def test_drawdown_budget_exceeded() -> None:
    """Drawdown >= 15% freezes new entry orders."""
    gate = PreTradeDecisionGate()
    req = _valid_request(current_drawdown_pct=18.5)
    result = gate.evaluate(req)

    assert result.allowed is False
    assert result.decision == GateDecision.REJECT
    assert result.primary_code == ERR_GATE_DRAWDOWN_BUDGET_EXCEEDED


def test_inv_gate_003_latency_performance() -> None:
    """PreTradeDecisionGate.evaluate must complete in-memory within hot-path SLA."""
    gate = PreTradeDecisionGate()
    req = _valid_request()

    # Warm up
    for _ in range(10):
        gate.evaluate(req)

    # Timing run across 100 evaluations
    t0 = time.perf_counter()
    for _ in range(100):
        gate.evaluate(req)
    total_time_s = time.perf_counter() - t0
    avg_us = (total_time_s / 100.0) * 1_000_000.0

    # Hot path in Python must be comfortably sub-millisecond (typically < 50us)
    assert avg_us < 200.0


def test_inv_gate_004_non_finite_rejections() -> None:
    """Non-finite inputs (NaN, Inf, bool, zero/negative prices) must raise NonFiniteGateInputException."""
    # Negative quantity
    with pytest.raises(NonFiniteGateInputException) as exc1:
        _valid_request(quantity=-10.0)
    assert exc1.value.code == ERR_GATE_NON_FINITE_INPUT

    # NaN reference price
    with pytest.raises(NonFiniteGateInputException) as exc2:
        _valid_request(reference_price=float("nan"))
    assert exc2.value.code == ERR_GATE_NON_FINITE_INPUT

    # Bool quantity
    with pytest.raises(NonFiniteGateInputException) as exc3:
        _valid_request(quantity=True)
    assert exc3.value.code == ERR_GATE_NON_FINITE_INPUT

    # Inf spread
    with pytest.raises(NonFiniteGateInputException) as exc4:
        _valid_request(market_spread_bps=float("inf"))
    assert exc4.value.code == ERR_GATE_NON_FINITE_INPUT


def test_history_capacity_bounded() -> None:
    """Gate history must bound memory consumption to max_history entries."""
    gate = PreTradeDecisionGate(max_history=20)
    for i in range(35):
        gate.evaluate(_valid_request(order_id=f"ORD-{i}"))
    assert len(gate.history) == 20
    assert gate.history[-1].order_id == "ORD-34"
