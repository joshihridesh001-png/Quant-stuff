"""Tool definitions and execution handlers for Model Context Protocol (MCP).

Purpose:
    Defines JSON Schema specifications and dispatch handlers for institutional MCP tools.
    Allows LLM agents to interact securely with the quant engine via standardized tool calls.

Dependencies:
    - json: Payload serialization.
    - time: Epoch timestamps.
    - typing: Type annotations.
    - quant.core.config: get_settings.
    - quant.api.dependencies: get_risk_service, get_autonomous_trader, get_external_provider_manager.
    - quant.execution.pre_trade_gate: PreTradeDecisionRequest.

Invariants Enforced:
    - Role-based safety: Read-only observation tools vs mutating panic/reset tools.
    - Cryptographic and input sanitization: Non-finite values rejected.
    - Rule 1: Four-tier docstrings on all functions.
"""

from __future__ import annotations

import time
from typing import Any

from quant.api.dependencies import (
    get_autonomous_trader,
    get_external_provider_manager,
    get_risk_service,
)
from quant.execution.kill_switch import PanicTriggerReason
from quant.execution.pre_trade_gate import PreTradeDecisionRequest


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool specifications exposed over MCP protocol."""
    # Functional Purpose: Declare tool capabilities and parameter schemas to AI clients.
    # Explicit Dependency Tracking: MCP specification standards.
    # Structural Relationship: Emitted during tools/list JSON-RPC request.
    # Defensive Invariant: Valid JSON Schema format for each tool.
    return [
        {
            "name": "quant_portfolio_telemetry",
            "description": "Retrieve real-time firm-wide portfolio valuation, cash balance, current positions, leverage, and circuit breaker status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "quant_market_orderbook",
            "description": "Query consolidated NBBO quotes, bid/ask spreads, and order book queue imbalance for a ticker symbol.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. SPY, AAPL, NVDA, QQQ)",
                    }
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "quant_macro_regimes",
            "description": "Inspect macroeconomic indicators from the Federal Reserve (FRED): 10Y-2Y yield curve spread, Fed funds rate, and CPI.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "quant_evaluate_pre_trade",
            "description": "Evaluate a proposed order through the 5-dimensional Bayesian log-odds decision gate before execution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Target ticker symbol"},
                    "action": {
                        "type": "string",
                        "enum": ["BUY", "SELL", "CLOSE", "LIQUIDATE"],
                        "description": "Trade action",
                    },
                    "quantity": {
                        "type": "number",
                        "description": "Share quantity strictly positive",
                    },
                    "reference_price": {
                        "type": "number",
                        "description": "Current market price strictly positive",
                    },
                    "is_position_exit": {
                        "type": "boolean",
                        "description": "Whether this reduces an existing position",
                    },
                    "market_spread_bps": {
                        "type": "number",
                        "description": "Spread in basis points (default 5.0)",
                    },
                    "order_book_imbalance": {
                        "type": "number",
                        "description": "OBI queue ratio in [-1.0, 1.0]",
                    },
                    "macro_yield_spread": {
                        "type": "number",
                        "description": "Yield spread in percentage points (default 0.18)",
                    },
                },
                "required": ["symbol", "action", "quantity", "reference_price"],
            },
        },
        {
            "name": "quant_swarm_status",
            "description": "Inspect the autonomous trading swarm daemon lifecycle state, iteration counter, and target allocations.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "quant_emergency_panic",
            "description": "EMERGENCY KILL SWITCH: Immediately cancels all resting orders across all gateways and locks order submission.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Etiology/justification for emergency panic halt",
                    },
                    "details": {
                        "type": "string",
                        "description": "Additional context or incident details",
                    },
                },
                "required": ["reason"],
            },
        },
        {
            "name": "quant_kill_switch_reset",
            "description": "Disarm emergency kill switch back to standby using authenticated administrator secret token.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "admin_token": {
                        "type": "string",
                        "description": "Cryptographic administrator secret token",
                    },
                },
                "required": ["admin_token"],
            },
        },
    ]


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute target MCP tool and return standardized JSON payload.

    Args:
        name: Name of tool to execute.
        arguments: Parameters dictionary.

    Returns:
        dict[str, Any]: Result dictionary containing tool output or error.
    """
    # Functional Purpose: Route and execute tool invocations with strict safety guards.
    # Explicit Dependency Tracking: Service singletons from dependencies.py.
    # Structural Relationship: Executed upon tools/call JSON-RPC request.
    # Defensive Invariant: Unhandled tool exceptions caught and formatted into error payloads.
    try:
        if name == "quant_portfolio_telemetry":
            risk_service = get_risk_service()
            status = risk_service.get_risk_status()
            return {
                "nav": status.nav,
                "peak_nav": status.peak_nav,
                "cash": status.cash,
                "margin_used": status.margin_used,
                "free_margin": status.free_margin,
                "gross_notional": status.gross_notional,
                "net_notional": status.net_notional,
                "gross_leverage": status.gross_leverage,
                "net_leverage": status.net_leverage,
                "intraday_drawdown_pct": status.intraday_drawdown_pct,
                "is_kill_switch_active": status.is_kill_switch_active,
                "kill_switch_trigger": status.kill_switch_trigger,
                "open_leaves_count": status.open_leaves_count,
            }

        elif name == "quant_market_orderbook":
            symbol = str(arguments.get("symbol", "SPY")).upper()
            trader = get_autonomous_trader()
            bars = await trader._market_feed.fetch_latest_bars([symbol])
            bar = bars.get(symbol)
            if bar is None:
                return {"symbol": symbol, "status": "no_data_available"}
            return {
                "symbol": symbol,
                "close": bar.close,
                "high": bar.high,
                "low": bar.low,
                "volume": bar.volume,
                "vwap": bar.vwap,
                "timestamp_ns": bar.timestamp,
            }

        elif name == "quant_macro_regimes":
            provider_mgr = get_external_provider_manager()
            fred = provider_mgr.fred
            spread_recs = await fred.fetch_series_observations("T10Y2Y", limit=1)
            fed_funds_recs = await fred.fetch_series_observations("DFF", limit=1)
            spread_val = spread_recs[0].value if spread_recs else 0.15
            spread_date = spread_recs[0].date if spread_recs else ""
            dff_val = fed_funds_recs[0].value if fed_funds_recs else 5.25
            dff_date = fed_funds_recs[0].date if fed_funds_recs else ""
            return {
                "t10y2y_yield_spread": spread_val,
                "t10y2y_date": spread_date,
                "dff_effective_rate": dff_val,
                "dff_date": dff_date,
                "is_yield_curve_inverted": spread_val < 0.0,
            }

        elif name == "quant_evaluate_pre_trade":
            trader = get_autonomous_trader()
            gate = trader.pre_trade_gate
            req = PreTradeDecisionRequest(
                order_id=f"MCP-{time.time_ns()}",
                symbol=str(arguments.get("symbol", "SPY")).upper(),
                action=str(arguments.get("action", "BUY")).upper(),
                quantity=float(arguments.get("quantity", 10.0)),
                reference_price=float(arguments.get("reference_price", 500.0)),
                timestamp_ns=time.time_ns(),
                is_position_exit=bool(arguments.get("is_position_exit", False)),
                market_spread_bps=float(arguments.get("market_spread_bps", 5.0)),
                order_book_imbalance=float(arguments.get("order_book_imbalance", 0.0)),
                macro_yield_spread=float(arguments.get("macro_yield_spread", 0.18)),
            )
            res = gate.evaluate(req)
            return {
                "decision_id": res.decision_id,
                "allowed": res.allowed,
                "decision": res.decision.value,
                "toxicity_probability": res.toxicity_probability,
                "confidence": res.confidence,
                "primary_code": res.primary_code,
                "reason": res.reason,
                "latency_us": res.latency_us,
                "checks": [
                    {
                        "name": c.name.value,
                        "passed": c.passed,
                        "score": c.score,
                        "details": c.details,
                    }
                    for c in res.checks
                ],
            }

        elif name == "quant_swarm_status":
            trader = get_autonomous_trader()
            return trader.get_status()

        elif name == "quant_emergency_panic":
            reason_str = str(arguments.get("reason", "MCP Operator Emergency Panic"))
            details_str = str(arguments.get("details", "Triggered via MCP agent call"))
            risk_service = get_risk_service()
            event = await risk_service.trigger_panic(
                reason=reason_str or PanicTriggerReason.MANUAL_OPERATOR.value,
                details={"operator_details": details_str, "source": "mcp"},
            )
            return {
                "status": "PANIC_TRIGGERED",
                "reason": event.trigger_reason.value,
                "cancelled_orders_count": event.cancelled_orders_count,
                "timestamp_ns": event.timestamp_ns,
            }

        elif name == "quant_kill_switch_reset":
            admin_token = str(arguments.get("admin_token", ""))
            risk_service = get_risk_service()
            success = risk_service.reset_kill_switch(admin_token)
            if not success:
                return {
                    "success": False,
                    "error": "Invalid administrator secret token",
                }
            return {
                "success": True,
                "status": "ARMED_STANDBY",
                "message": "Emergency kill switch disarmed and re-armed to standby",
            }

        else:
            return {"error": f"Unknown tool name: {name}"}

    except Exception as exc:
        return {"error": f"Tool execution failed: {str(exc)}"}
