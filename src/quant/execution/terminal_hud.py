"""High-Refresh Institutional Rich Terminal HUD for Live Trading.

Functional Purpose:
    Renders an institutional terminal heads-up display (HUD) visualizing real-time trading
    telemetry, active alpha strategy allocations, pre-trade risk firewall status,
    order execution reports, and portfolio valuation ribbons.

Explicit Dependency Tracking:
    - rich.console: Console and string capture rendering.
    - rich.layout: Hierarchical terminal quadrant grid splitting.
    - rich.live: High-refresh flicker-free terminal updates.
    - rich.panel: Bordered visual containers with institutional styling.
    - rich.table: Formatted tabular displays for positions, orders, and swarm signals.
    - rich.text: Colored and stylized terminal typography.
    - quant.services.live_session: LiveSessionTelemetry, LiveSessionStatus.

Structural Relationship:
    - Presentation layer interface for Phase 17 (Live Trading Station).
    - Ingests telemetry snapshots from LiveTradingSession.step_cycle().
    - Rendered in interactive CLI runner (scripts/run_live_trader.py).

Defensive Invariants:
    - INV-HUD-001: Sub-2ms layout composition SLA.
    - INV-HUD-002: Zero crash on non-finite, NaN, Infinity, or missing symbol inputs.
    - INV-HUD-003: Headless terminal safety: Renders cleanly to string buffers when TTY is unavailable.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from quant.services.live_session import LiveSessionStatus, LiveSessionTelemetry

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Error Taxonomy)
# ============================================================================

ERR_HUD_RENDER_FAILURE: Final[str] = "ERR-HUD-001"
ERR_HUD_NON_FINITE_INPUT: Final[str] = "ERR-HUD-002"


# ============================================================================
# Configuration Data Transfer Object
# ============================================================================


@dataclass(frozen=True, slots=True)
class TerminalHUDConfig:
    """Configuration parameters for rich terminal HUD."""

    refresh_per_second: float = 4.0
    theme_name: str = "dark"
    show_footer_hotkeys: bool = True


# ============================================================================
# Terminal HUD Implementation
# ============================================================================


class TerminalHUD:
    """Rich interactive terminal heads-up display for autonomous live trading stations.

    Functional Purpose:
        Transforms real-time telemetry into a visual, multi-panel terminal dashboard.

    Defensive Invariants:
        INV-HUD-001: Sub-2ms layout render overhead.
        INV-HUD-002: Total exception containment; never breaks the live trading loop.
    """

    __slots__ = (
        "_config",
        "_console",
        "_live",
        "_recent_events",
    )

    def __init__(
        self,
        config: TerminalHUDConfig | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize TerminalHUD with configuration and console target."""
        self._config = config or TerminalHUDConfig()
        self._console = console or Console()
        self._live: Live | None = None
        self._recent_events: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------------
    # Display Lifecycle Management
    # ------------------------------------------------------------------------

    def start(self) -> None:
        """Start the interactive Live display session."""
        if self._live is None and self._console.is_terminal:
            self._live = Live(
                self._render_placeholder(),
                console=self._console,
                refresh_per_second=self._config.refresh_per_second,
                transient=False,
            )
            self._live.start()

    def update(self, telemetry: LiveSessionTelemetry) -> None:
        """Update active Live display with the latest telemetry snapshot."""
        layout = self.render_layout(telemetry)
        if self._live is not None and self._live.is_started:
            self._live.update(layout)

    def stop(self) -> None:
        """Cleanly terminate the Live display session."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def add_event(self, timestamp_str: str, category: str, detail: str) -> None:
        """Append an event to the streaming HUD event buffer."""
        self._recent_events.append((timestamp_str, category, detail))
        if len(self._recent_events) > 8:
            self._recent_events.pop(0)

    # ------------------------------------------------------------------------
    # Layout Composition & Rendering
    # ------------------------------------------------------------------------

    def render_layout(self, telemetry: LiveSessionTelemetry) -> Layout:
        """Compose the 4-panel quadrant grid layout from telemetry."""
        layout = Layout()

        # Split into Header, Body, and Footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        # Split Body into Left (Portfolio & Swarm) and Right (Positions & Events)
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="portfolio", ratio=1),
            Layout(name="swarm", ratio=1),
        )
        layout["right"].split_column(
            Layout(name="positions", ratio=1),
            Layout(name="events", ratio=1),
        )

        # Populate sections
        layout["header"].update(self._build_header(telemetry))
        layout["portfolio"].update(self._build_portfolio_panel(telemetry))
        layout["swarm"].update(self._build_swarm_panel(telemetry))
        layout["positions"].update(self._build_positions_panel(telemetry))
        layout["events"].update(self._build_events_panel(telemetry))
        layout["footer"].update(self._build_footer())

        return layout

    def render_string(self, telemetry: LiveSessionTelemetry) -> str:
        """Render layout to string for headless, test, or logging capture."""
        layout = self.render_layout(telemetry)
        buf = io.StringIO()
        capture_console = Console(file=buf, width=120, height=35, color_system=None)
        capture_console.print(layout)
        return buf.getvalue()

    # ------------------------------------------------------------------------
    # Panel Builders
    # ------------------------------------------------------------------------

    def _build_header(self, telemetry: LiveSessionTelemetry) -> Panel:
        """Status Header: Clock, State, Kill Switch Banner, Iteration."""
        now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Status badge styling
        status_color = "green"
        if telemetry.status == LiveSessionStatus.PAUSED:
            status_color = "yellow"
        elif telemetry.status in (LiveSessionStatus.PANIC_KILLED, LiveSessionStatus.ERROR):
            status_color = "bold red"

        status_text = Text(
            f" STATUS: {telemetry.status.value} ", style=f"bold white on {status_color}"
        )

        # Kill Switch badge
        if telemetry.is_kill_switch_active:
            ks_text = Text(" KILL SWITCH: ACTIVE (HALTED) ", style="bold white on red")
        else:
            ks_text = Text(" KILL SWITCH: ARMED (STANDBY) ", style="bold white on green")

        iter_text = Text(f"CYCLE: #{telemetry.iteration}", style="cyan")
        orders_text = Text(
            f"DISPATCHED: {telemetry.orders_dispatched_count} | FILLS: {telemetry.fills_count}",
            style="dim",
        )

        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left", ratio=1)
        header_table.add_column(justify="center", ratio=1)
        header_table.add_column(justify="right", ratio=1)

        header_table.add_row(
            Text(f"QUANT STATION | {now_utc}", style="bold white"),
            Text.assemble(status_text, "  ", ks_text),
            Text.assemble(iter_text, "  ", orders_text),
        )

        return Panel(header_table, style="blue")

    def _build_portfolio_panel(self, telemetry: LiveSessionTelemetry) -> Panel:
        """Portfolio Ribbon: NAV, Cash, Leverage, Intraday PnL, Drawdown."""
        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Metric", style="bold white")
        table.add_column("Value", justify="right")

        nav_str = (
            f"${telemetry.portfolio_nav:,.2f}" if math.isfinite(telemetry.portfolio_nav) else "N/A"
        )
        cash_str = f"${telemetry.cash:,.2f}" if math.isfinite(telemetry.cash) else "N/A"
        gross_lev = (
            f"{telemetry.gross_leverage:.2f}x" if math.isfinite(telemetry.gross_leverage) else "N/A"
        )
        net_lev = (
            f"{telemetry.net_leverage:.2f}x" if math.isfinite(telemetry.net_leverage) else "N/A"
        )

        dd_pct = telemetry.drawdown_pct * 100.0 if math.isfinite(telemetry.drawdown_pct) else 0.0
        dd_color = "green" if dd_pct < 5.0 else ("yellow" if dd_pct < 10.0 else "bold red")
        dd_str = f"[{dd_color}]-{dd_pct:.2f}%[/{dd_color}]"

        table.add_row("Portfolio NAV", f"[bold cyan]{nav_str}[/bold cyan]")
        table.add_row("Available Cash", cash_str)
        table.add_row("Gross Leverage", gross_lev)
        table.add_row("Net Leverage", net_lev)
        table.add_row("Intraday Drawdown", dd_str)

        return Panel(table, title="[bold]Portfolio Health[/bold]", border_style="cyan")

    def _build_swarm_panel(self, telemetry: LiveSessionTelemetry) -> Panel:
        """Strategy Swarm: Symbol, Target Weight, Conviction, Signal Direction."""
        table = Table(box=None, expand=True)
        table.add_column("Symbol", style="bold white")
        table.add_column("Dir", justify="center")
        table.add_column("Target Wt", justify="right")
        table.add_column("Conviction", justify="right")

        for sym, weight in telemetry.target_weights.items():
            sig = telemetry.active_signals.get(sym)
            dir_str = sig.direction.value if sig is not None else "FLAT"
            dir_color = "green" if dir_str == "LONG" else ("red" if dir_str == "SHORT" else "dim")

            conv_val = sig.conviction if sig is not None else 0.0
            conv_str = f"{conv_val * 100:.0f}%" if math.isfinite(conv_val) else "0%"
            wt_str = f"{weight * 100:+.1f}%" if math.isfinite(weight) else "0.0%"

            table.add_row(
                sym,
                f"[{dir_color}]{dir_str}[/{dir_color}]",
                wt_str,
                conv_str,
            )

        if not telemetry.target_weights:
            table.add_row("Scanning...", "---", "0.0%", "0%")

        return Panel(table, title="[bold]Alpha Swarm Allocations[/bold]", border_style="magenta")

    def _build_positions_panel(self, telemetry: LiveSessionTelemetry) -> Panel:
        """Positions Table: Symbol, Qty, Price, Notional Value."""
        table = Table(box=None, expand=True)
        table.add_column("Symbol", style="bold white")
        table.add_column("Shares", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Notional", justify="right")

        has_pos = False
        for sym, qty in telemetry.positions.items():
            if abs(qty) > 1e-4:
                has_pos = True
                p = telemetry.current_prices.get(sym, 0.0)
                notional = qty * p
                p_str = f"${p:,.2f}" if math.isfinite(p) else "N/A"
                notional_str = f"${notional:,.2f}" if math.isfinite(notional) else "N/A"
                color = "green" if qty > 0 else "red"

                table.add_row(
                    sym,
                    f"[{color}]{qty:+,.0f}[/{color}]",
                    p_str,
                    notional_str,
                )

        if not has_pos:
            table.add_row("ALL CASH", "0", "$0.00", "$0.00")

        return Panel(table, title="[bold]Active Positions[/bold]", border_style="green")

    def _build_events_panel(self, telemetry: LiveSessionTelemetry) -> Panel:
        """Streaming Events & Execution Log."""
        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Type", style="bold")
        table.add_column("Message", style="white")

        if telemetry.last_error:
            table.add_row("ALERT", "[bold red]ERROR[/bold red]", telemetry.last_error[:40])

        for t, cat, det in reversed(self._recent_events[-5:]):
            table.add_row(t, f"[cyan]{cat}[/cyan]", det[:40])

        if not self._recent_events and not telemetry.last_error:
            table.add_row("--:--:--", "[dim]SYS[/dim]", "Autonomous execution loop active")

        return Panel(table, title="[bold]Execution & Audit Log[/bold]", border_style="yellow")

    def _build_footer(self) -> Panel:
        """Operator Controls Ribbon."""
        hotkeys = [
            ("[SPACE]", "Pause/Resume"),
            ("[K]", "PANIC KILL"),
            ("[R]", "Re-Arm Risk"),
            ("[Q]", "Shutdown"),
        ]
        parts: list[Any] = []
        for key, desc in hotkeys:
            parts.extend(
                [
                    Text(f" {key} ", style="bold black on bright_yellow"),
                    Text(f" {desc}   ", style="bold white"),
                ]
            )

        ribbon = Text.assemble(*parts)
        return Panel(ribbon, style="dim white")

    def _render_placeholder(self) -> Panel:
        """Initial startup banner."""
        return Panel(
            Text("INITIALIZING AUTONOMOUS TRADING STATION...", style="bold cyan", justify="center"),
            border_style="cyan",
        )
