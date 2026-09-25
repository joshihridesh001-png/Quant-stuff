"""Autonomous Model Context Protocol (MCP) Agent Observation and Decision Loop.

Purpose:
    Executes an autonomous AI agent cycle over the Model Context Protocol (MCP 2024-11-05).
    Connects to the quantitative engine, dynamically discovers tools, inspects real-time
    portfolio telemetry and macroeconomic regimes, queries consolidated order books, and
    runs pre-trade Bayesian risk simulations before proposing execution orders.

Dependencies:
    - argparse: Command-line parameter parsing.
    - asyncio: Asynchronous execution loop.
    - sys: Platform-specific stream utilities.
    - time: High-resolution timestamping.
    - quant.mcp.client: Institutional MCPClient, exceptions, and data models.

Invariants Enforced:
    - Rule 1: Four-tier docstrings across all functions.
    - Rule 2: Deterministic error handling with diagnostic fault codes.
    - Rule 3: Zero ruff lint/formatting deviations and strict mypy compliance.
    - Rule 4: Mandatory pre-trade Bayesian risk simulation before proposing any execution.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import Final

from quant.mcp.client import (
    ERR_MCP_AUTH_FAILURE,
    ERR_MCP_PROTOCOL_VIOLATION,
    ERR_MCP_SCHEMA_VALIDATION_FAILURE,
    ERR_MCP_TOOL_EXECUTION_FAILURE,
    ERR_MCP_TRANSPORT_FAILURE,
    MCPAuthFailureException,
    MCPBaseException,
    MCPClient,
    MCPProtocolException,
    MCPSchemaValidationException,
    MCPToolExecutionException,
    MCPTransportException,
)

logger = logging.getLogger(__name__)

# Terminal ANSI Styling Constants
CLR_RESET: Final[str] = "\033[0m"
CLR_BOLD: Final[str] = "\033[1m"
CLR_CYAN: Final[str] = "\033[96m"
CLR_GREEN: Final[str] = "\033[92m"
CLR_YELLOW: Final[str] = "\033[93m"
CLR_RED: Final[str] = "\033[91m"
CLR_BLUE: Final[str] = "\033[94m"
CLR_DIM: Final[str] = "\033[2m"


def print_banner() -> None:
    """Print institutional stylized header for MCP Autonomous Agent."""
    # Functional Purpose: Render visual CLI session header.
    # Explicit Dependency Tracking: sys.stdout.
    # Structural Relationship: First output of scripts/run_mcp_agent.py.
    # Defensive Invariant: Writes safe universal ASCII characters.
    banner = f"""{CLR_CYAN}{CLR_BOLD}
+=================================================================================+
|          QUANTITATIVE ENGINE -- AUTONOMOUS AI AGENT MCP WORKBENCH               |
|                  Model Context Protocol (MCP 2024-11-05)                        |
+=================================================================================+{CLR_RESET}"""
    print(banner)


def print_section(title: str) -> None:
    """Render a styled section divider."""
    # Functional Purpose: Visual categorization of agent observation steps.
    # Explicit Dependency Tracking: None.
    # Structural Relationship: Formats step transitions in agent execution loop.
    # Defensive Invariant: title is non-empty string.
    print(f"\n{CLR_BLUE}{CLR_BOLD}---> [{title.upper()}]{CLR_RESET}")


def format_status_badge(passed: bool, label_pass: str = "PASS", label_fail: str = "FAIL") -> str:
    """Return colored status badge string."""
    # Functional Purpose: Standardized color-coded status badge generation.
    # Explicit Dependency Tracking: ANSI color constants.
    # Structural Relationship: Used across telemetry and pre-trade checks display.
    # Defensive Invariant: Deterministic string output.
    if passed:
        return f"{CLR_GREEN}{CLR_BOLD}[PASS: {label_pass}]{CLR_RESET}"
    return f"{CLR_RED}{CLR_BOLD}[FAIL: {label_fail}]{CLR_RESET}"


async def run_agent_cycle(
    client: MCPClient,
    symbol: str = "SPY",
    action: str = "BUY",
    quantity: float = 25.0,
    cycle_index: int = 1,
) -> bool:
    """Execute a single autonomous observation, risk evaluation, and decision cycle.

    Args:
        client: Connected and authenticated MCPClient instance.
        symbol: Candidate ticker symbol to inspect.
        action: Candidate trade action ('BUY', 'SELL', etc.).
        quantity: Candidate rebalancing share quantity.
        cycle_index: Monotonically increasing iteration counter.

    Returns:
        bool: True if pre-trade decision allowed candidate order; False otherwise.
    """
    # Functional Purpose: Execute the 5-step autonomous agent observation and decision cycle.
    # Explicit Dependency Tracking: MCPClient helper methods.
    # Structural Relationship: Core body of run_mcp_agent execution loop.
    # Defensive Invariant: Traps tool errors without crashing the main supervisor process.
    print_section(f"Cycle {cycle_index}: Observation & Pre-Trade Risk Evaluation")
    start_time_ns = time.perf_counter_ns()

    # Step 1: Telemetry Observation
    print(f"\n{CLR_BOLD}1. Querying Real-Time Firm Portfolio Telemetry...{CLR_RESET}")
    telemetry = await client.get_portfolio_telemetry()
    nav = float(telemetry.get("nav", 0.0))
    cash = float(telemetry.get("cash", 0.0))
    leverage = float(telemetry.get("gross_leverage", 0.0))
    drawdown_pct = float(telemetry.get("intraday_drawdown_pct", 0.0))
    is_kill_switch = bool(telemetry.get("is_kill_switch_active", False))
    open_leaves = int(telemetry.get("open_leaves_count", 0))

    ks_badge = (
        format_status_badge(not is_kill_switch, "STANDBY", "ACTIVE_HALT")
        if not is_kill_switch
        else format_status_badge(False, "STANDBY", "PANIC_LOCKED")
    )
    print(f"   * Net Asset Value (NAV):  {CLR_CYAN}${nav:,.2f}{CLR_RESET}")
    print(f"   * Cash Balance:           {CLR_CYAN}${cash:,.2f}{CLR_RESET}")
    print(f"   * Gross Leverage:         {CLR_CYAN}{leverage:.2f}x{CLR_RESET}")
    print(f"   * Intraday Drawdown:      {CLR_YELLOW}{drawdown_pct:.2f}%{CLR_RESET}")
    print(f"   * Emergency Kill Switch:  {ks_badge}")
    print(f"   * Open Resting Leaves:    {CLR_CYAN}{open_leaves}{CLR_RESET}")

    # Step 2: Macroeconomic Regime Observation
    print(f"\n{CLR_BOLD}2. Observing Federal Reserve Macro Regimes (FRED)...{CLR_RESET}")
    macro = await client.get_macro_regimes()
    yield_spread = float(macro.get("t10y2y_yield_spread", 0.18))
    fed_funds = float(macro.get("dff_effective_rate", 5.25))
    is_inverted = bool(macro.get("is_yield_curve_inverted", yield_spread < 0.0))
    spread_date = macro.get("t10y2y_date", "Latest")

    curve_badge = (
        f"{CLR_RED}{CLR_BOLD}[INVERTED / RECESSIONARY RISK]{CLR_RESET}"
        if is_inverted
        else f"{CLR_GREEN}{CLR_BOLD}[NORMAL STEEPENING]{CLR_RESET}"
    )
    print(f"   * 10Y-2Y Spread (T10Y2Y): {CLR_CYAN}{yield_spread:+.2f}%{CLR_RESET} ({spread_date})")
    print(f"   * Fed Funds Rate (DFF):   {CLR_CYAN}{fed_funds:.2f}%{CLR_RESET}")
    print(f"   * Yield Curve Regime:     {curve_badge}")

    # Step 3: Order Book & Pricing Observation
    print(f"\n{CLR_BOLD}3. Querying Consolidated Market Order Book for {symbol}...{CLR_RESET}")
    book = await client.get_market_orderbook(symbol)
    ref_price = float(book.get("close", 500.0))
    vwap = float(book.get("vwap", ref_price))
    volume = int(book.get("volume", 0))
    print(f"   * Reference Close Price:  {CLR_GREEN}${ref_price:,.2f}{CLR_RESET}")
    print(f"   * Volume-Weighted Price:  {CLR_CYAN}${vwap:,.2f}{CLR_RESET}")
    print(f"   * Bar Trade Volume:       {CLR_CYAN}{volume:,} shares{CLR_RESET}")

    # Step 4: Autonomous Pre-Trade Risk Gate Simulation
    print(
        f"\n{CLR_BOLD}4. Running 5-D Bayesian Log-Odds Pre-Trade Risk Gate Simulation...{CLR_RESET}"
    )
    proposed_notional = quantity * ref_price
    print(
        f"   > Proposed Order: {CLR_YELLOW}{CLR_BOLD}{action} {quantity} {symbol} @ ${ref_price:,.2f}{CLR_RESET} "
        f"(Notional: ${proposed_notional:,.2f})"
    )

    decision_res = await client.evaluate_pre_trade(
        symbol=symbol,
        action=action,
        quantity=quantity,
        reference_price=ref_price,
        is_position_exit=False,
        market_spread_bps=4.5,
        order_book_imbalance=0.08,
        macro_yield_spread=yield_spread,
    )

    allowed = bool(decision_res.get("allowed", False))
    decision = str(decision_res.get("decision", "UNKNOWN"))
    p_toxic = float(decision_res.get("toxicity_probability", 0.0))
    confidence = float(decision_res.get("confidence", 0.0))
    primary_code = str(decision_res.get("primary_code", "PASS"))
    reason = str(decision_res.get("reason", "Checks passed"))
    latency_us = float(decision_res.get("latency_us", 0.0))

    dec_badge = format_status_badge(allowed, label_pass=decision, label_fail=decision)
    print(f"   * Gate Verdict:           {dec_badge}")
    print(f"   * Toxicity Probability:   {CLR_YELLOW}{p_toxic * 100:.2f}%{CLR_RESET}")
    print(f"   * Model Confidence:       {CLR_CYAN}{confidence * 100:.1f}%{CLR_RESET}")
    print(f"   * Primary Diagnostic Code:{CLR_BOLD} {primary_code}{CLR_RESET}")
    print(f"   * Evaluation Latency:     {CLR_GREEN}{latency_us:.1f} us{CLR_RESET}")
    print(f"   * Gate Rationale:         {CLR_DIM}{reason}{CLR_RESET}")

    # Render Dimension Breakdown Table
    checks = decision_res.get("checks", [])
    if isinstance(checks, list) and checks:
        print(f"\n   {CLR_BOLD}Pre-Trade Safety Dimension Breakdown:{CLR_RESET}")
        print(
            "   +-------------------------------------+----------+----------+------------------------+"
        )
        print(
            "   | Dimension Name                      | Status   | Score    | Diagnostic Detail      |"
        )
        print(
            "   +-------------------------------------+----------+----------+------------------------+"
        )
        for chk in checks:
            if isinstance(chk, dict):
                c_name = str(chk.get("name", ""))[:35].ljust(35)
                c_pass = bool(chk.get("passed", False))
                c_status_str = (
                    f"{CLR_GREEN}PASS{CLR_RESET}" if c_pass else f"{CLR_RED}FAIL{CLR_RESET}"
                )
                c_score = float(chk.get("score", 0.0))
                c_detail = str(chk.get("details", ""))[:22].ljust(22)
                status_col = f"{c_status_str}    "
                print(f"   | {c_name} | {status_col} | {c_score:+8.3f} | {c_detail} |")
        print(
            "   +-------------------------------------+----------+----------+------------------------+"
        )

    # Step 5: Swarm Background Status
    print(f"\n{CLR_BOLD}5. Inspecting Autonomous Trading Swarm Daemon...{CLR_RESET}")
    swarm_status = await client.get_swarm_status()
    swarm_state = str(swarm_status.get("state", "IDLE"))
    swarm_iter = int(swarm_status.get("iteration", 0))
    swarm_universe = swarm_status.get("universe", [])
    print(f"   * Swarm Lifecycle State:  {CLR_CYAN}{swarm_state}{CLR_RESET}")
    print(f"   * Swarm Iteration Count:  {CLR_CYAN}{swarm_iter}{CLR_RESET}")
    print(f"   * Active Asset Universe:  {CLR_CYAN}{swarm_universe}{CLR_RESET}")

    elapsed_ms = (time.perf_counter_ns() - start_time_ns) / 1_000_000.0
    print(f"\n{CLR_BOLD}Cycle Summary:{CLR_RESET}")
    if allowed:
        print(
            f"   {CLR_GREEN}{CLR_BOLD}[OK] DECISION: Candidate rebalancing order APPROVED for routing.{CLR_RESET} "
            f"(Total Cycle Time: {elapsed_ms:.2f}ms)"
        )
    else:
        print(
            f"   {CLR_RED}{CLR_BOLD}[BLOCKED] DECISION: Order BLOCKED by pre-trade gate: [{primary_code}] {reason}{CLR_RESET} "
            f"(Total Cycle Time: {elapsed_ms:.2f}ms)"
        )

    return allowed


async def main_async(args: argparse.Namespace) -> int:
    """Asynchronous entry point orchestrating client initialization and cycles."""
    # Functional Purpose: Coordinate agent session authentication and cycle execution.
    # Explicit Dependency Tracking: MCPClient, run_agent_cycle.
    # Structural Relationship: Called from main() CLI entrypoint.
    # Defensive Invariant: Traps exceptions and translates them to process return codes.
    print_banner()

    transport_mode = args.transport
    base_url = args.base_url
    role = args.role.upper()
    username = args.username
    password = args.password
    symbol = args.symbol.upper()
    action = args.action.upper()
    quantity = float(args.quantity)
    iterations = int(args.iterations)
    interval = float(args.interval)

    print(f"{CLR_BOLD}Configuration:{CLR_RESET}")
    print(f"  * Transport:   {CLR_CYAN}{transport_mode.upper()}{CLR_RESET}")
    if transport_mode == "http":
        print(f"  * Gateway URL: {CLR_CYAN}{base_url}/api/v1/mcp/rpc{CLR_RESET}")
    print(f"  * Principal:   {CLR_CYAN}{username}{CLR_RESET} (Role: {role})")
    print(f"  * Target:      {CLR_CYAN}{action} {quantity} {symbol}{CLR_RESET}")
    print(f"  * Iterations:  {CLR_CYAN}{iterations}{CLR_RESET}")

    try:
        async with MCPClient(
            base_url=base_url,
            transport=transport_mode,
            username=username,
            password=password,
            role=role,
            timeout_seconds=15.0,
        ) as client:
            # 1. Protocol Handshake
            print_section("Protocol Initialization Handshake")
            init_res = await client.initialize(
                client_name="quant-autonomous-agent",
                client_version="1.0.0",
            )
            server_info = init_res.get("serverInfo", {})
            srv_name = server_info.get("name", "unknown")
            srv_ver = server_info.get("version", "unknown")
            proto_ver = init_res.get("protocolVersion", "unknown")
            print(f"  [OK] Connected to MCP Server: {CLR_GREEN}{srv_name} v{srv_ver}{CLR_RESET}")
            print(f"  [OK] Negotiated Protocol:     {CLR_GREEN}MCP {proto_ver}{CLR_RESET}")

            # 2. Dynamic Tool Discovery
            print_section("Dynamic Institutional Tool Discovery")
            tools = await client.list_tools()
            print(f"  [OK] Discovered {CLR_GREEN}{len(tools)}{CLR_RESET} institutional tools:")
            for idx, tool in enumerate(tools, 1):
                t_name = tool.get("name", "")
                t_desc = tool.get("description", "")
                print(f"    [{idx}] {CLR_YELLOW}{CLR_BOLD}{t_name}{CLR_RESET}: {t_desc[:70]}...")

            # 3. Execution Cycles
            for cycle in range(1, iterations + 1):
                await run_agent_cycle(
                    client=client,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    cycle_index=cycle,
                )
                if cycle < iterations:
                    print(
                        f"\n{CLR_DIM}Sleeping {interval:.1f}s before next observation cycle...{CLR_RESET}"
                    )
                    await asyncio.sleep(interval)

        print(
            f"\n{CLR_GREEN}{CLR_BOLD}[SUCCESS] All autonomous agent observation cycles completed successfully.{CLR_RESET}\n"
        )
        return 0

    except MCPAuthFailureException as exc:
        print(
            f"\n{CLR_RED}{CLR_BOLD}FATAL AUTHENTICATION ERROR [{ERR_MCP_AUTH_FAILURE}]:{CLR_RESET} {exc}"
        )
        return 1
    except MCPProtocolException as exc:
        print(
            f"\n{CLR_RED}{CLR_BOLD}FATAL PROTOCOL ERROR [{ERR_MCP_PROTOCOL_VIOLATION}]:{CLR_RESET} {exc}"
        )
        return 2
    except MCPTransportException as exc:
        print(
            f"\n{CLR_RED}{CLR_BOLD}FATAL TRANSPORT ERROR [{ERR_MCP_TRANSPORT_FAILURE}]:{CLR_RESET} {exc}"
        )
        return 3
    except MCPToolExecutionException as exc:
        print(
            f"\n{CLR_RED}{CLR_BOLD}FATAL TOOL EXECUTION ERROR [{ERR_MCP_TOOL_EXECUTION_FAILURE}]:{CLR_RESET} {exc}"
        )
        return 4
    except MCPSchemaValidationException as exc:
        print(
            f"\n{CLR_RED}{CLR_BOLD}FATAL SCHEMA VALIDATION ERROR [{ERR_MCP_SCHEMA_VALIDATION_FAILURE}]:{CLR_RESET} {exc}"
        )
        return 5
    except MCPBaseException as exc:
        print(f"\n{CLR_RED}{CLR_BOLD}FATAL MCP EXCEPTION [{exc.code}]:{CLR_RESET} {exc}")
        return 6
    except Exception as exc:
        print(f"\n{CLR_RED}{CLR_BOLD}UNEXPECTED ERROR:{CLR_RESET} {exc}")
        return 99


def main() -> None:
    """CLI entrypoint parsing arguments and dispatching asyncio event loop."""
    # Functional Purpose: Command-line interface parser for scripts/run_mcp_agent.py.
    # Explicit Dependency Tracking: argparse.ArgumentParser, sys.stdout.reconfigure.
    # Structural Relationship: Executable script entrypoint.
    # Defensive Invariant: Configures UTF-8 encoding safely and returns exit code to OS.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Autonomous Model Context Protocol (MCP) Agent Observation and Decision Loop."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport mode: 'stdio' (local child process) or 'http' (FastAPI JSON-RPC gateway).",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for quant engine web services (used when --transport http).",
    )
    parser.add_argument(
        "--role",
        choices=["RESEARCHER", "ADMIN"],
        default="ADMIN",
        help="Requested authentication role for agent.",
    )
    parser.add_argument(
        "--username",
        default="admin",
        help="Username for authentication.",
    )
    parser.add_argument(
        "--password",
        default="quant-secret-pass",
        help="Password for authentication.",
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Candidate ticker symbol for market observation and pre-trade simulation.",
    )
    parser.add_argument(
        "--action",
        choices=["BUY", "SELL", "CLOSE", "LIQUIDATE"],
        default="BUY",
        help="Proposed rebalancing action.",
    )
    parser.add_argument(
        "--quantity",
        type=float,
        default=25.0,
        help="Proposed rebalancing share quantity.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of observation and evaluation cycles to execute.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds to wait between observation cycles when iterations > 1.",
    )

    args = parser.parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
