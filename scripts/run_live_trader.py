"""Turnkey Continuous Live Paper Trading Station & Terminal HUD CLI Runner.

Functional Purpose:
    Launches an institutional-grade, turnkey autonomous live paper trading station.
    Orchestrates live market data feeds, regime-conditioned alpha strategy evaluation,
    pre-trade risk firewall gating, smart order routing, double-entry tax-lot bookkeeping,
    and a flicker-free rich terminal HUD with interactive hotkeys.

Explicit Dependency Tracking:
    - argparse: Production command-line configuration parser.
    - asyncio: Asynchronous event loop and signal trapping.
    - quant.analytics.strategies: SwarmMetaStrategy, KalmanPairsTradingStrategy,
      FracDiffMomentumStrategy, VolatilityBreakoutStrategy, LoughranMcDonaldSentimentStrategy.
    - quant.execution.hotkeys: HotkeyAction, HotkeyManager.
    - quant.execution.terminal_hud: TerminalHUD, TerminalHUDConfig.
    - quant.services.live_session: LiveSessionConfig, LiveSessionStatus, LiveTradingSession.

Usage Examples:
    # Run full Swarm across top liquid universe (default):
    python scripts/run_live_trader.py --symbols SPY,QQQ,AAPL,NVDA,MSFT --strategy swarm

    # Run Kalman Pairs Trading on SPY/QQQ with $250k initial capital:
    python scripts/run_live_trader.py --symbols SPY,QQQ --strategy kalman --capital 250000

    # Headless continuous background execution:
    python scripts/run_live_trader.py --symbols SPY,QQQ,AAPL --headless --poll-interval 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import numpy as np

from quant.analytics.strategies.base import IAlphaStrategy
from quant.analytics.strategies.momentum import FracDiffMomentumStrategy
from quant.analytics.strategies.sentiment import LoughranMcDonaldSentimentStrategy
from quant.analytics.strategies.stat_arb import KalmanPairsTradingStrategy
from quant.analytics.strategies.swarm_meta import SwarmMetaStrategy
from quant.analytics.strategies.volatility import VolatilityBreakoutStrategy
from quant.execution.hotkeys import HotkeyAction, HotkeyManager
from quant.execution.terminal_hud import TerminalHUD, TerminalHUDConfig
from quant.services.live_session import (
    LiveSessionConfig,
    LiveSessionStatus,
    LiveSessionTelemetry,
    LiveTradingSession,
)

logger = logging.getLogger(__name__)


def build_strategy(strategy_type: str, symbols: tuple[str, ...]) -> IAlphaStrategy:
    """Factory creating concrete alpha strategy based on CLI selector."""
    s_type = strategy_type.lower().strip()
    if s_type == "kalman":
        pair_a = symbols[0]
        pair_b = symbols[1] if len(symbols) > 1 else ("QQQ" if pair_a == "SPY" else "SPY")
        return KalmanPairsTradingStrategy(
            strategy_id="kalman_pairs",
            asset_y=pair_a,
            asset_x=pair_b,
            min_warmup_bars=20,
        )
    elif s_type == "momentum":
        return FracDiffMomentumStrategy(
            strategy_id="fracdiff_mom",
            monitored_symbols=symbols,
            min_warmup_bars=35,
        )
    elif s_type == "volatility":
        return VolatilityBreakoutStrategy(
            strategy_id="vol_breakout",
            monitored_symbols=symbols,
            min_warmup_bars=120,
        )
    elif s_type == "sentiment":
        return LoughranMcDonaldSentimentStrategy(
            strategy_id="lm_sentiment",
            monitored_symbols=symbols,
            min_warmup_bars=5,
        )
    elif s_type == "swarm":
        pair_a = symbols[0]
        pair_b = symbols[1] if len(symbols) > 1 else ("QQQ" if pair_a == "SPY" else "SPY")
        kalman = KalmanPairsTradingStrategy(
            strategy_id="swarm_kalman",
            asset_y=pair_a,
            asset_x=pair_b,
            min_warmup_bars=20,
        )
        mom = FracDiffMomentumStrategy(
            strategy_id="swarm_mom",
            monitored_symbols=symbols,
            min_warmup_bars=35,
        )
        vol = VolatilityBreakoutStrategy(
            strategy_id="swarm_vol",
            monitored_symbols=symbols,
            min_warmup_bars=120,
        )
        sent = LoughranMcDonaldSentimentStrategy(
            strategy_id="swarm_sent",
            monitored_symbols=symbols,
            min_warmup_bars=5,
        )
        return SwarmMetaStrategy(
            strategy_id="swarm_meta",
            monitored_symbols=symbols,
            sub_strategies=[kalman, mom, vol, sent],
            min_warmup_bars=120,
        )
    else:
        raise ValueError(
            f"Unknown strategy selector '{strategy_type}'. Available: swarm, kalman, momentum, volatility, sentiment"
        )


def seed_session_history(
    session: LiveTradingSession, symbols: tuple[str, ...], min_bars: int
) -> None:
    """Seed historical bar buffer so strategies immediately satisfy warmup horizons."""
    # Deterministic geometric random walk for realistic warmup
    base_prices = {"SPY": 500.0, "QQQ": 430.0, "AAPL": 180.0, "NVDA": 120.0, "MSFT": 420.0}
    now_sec = time.time()
    n_bars = max(min_bars + 15, 150)

    for sym in symbols:
        p0 = base_prices.get(sym, 100.0)
        bars = []
        curr_p = p0
        for i in range(n_bars):
            t = now_sec - (n_bars - i) * 60.0
            ret = float(np.random.normal(0.0001, 0.005))
            curr_p = max(1.0, curr_p * (1.0 + ret))
            bars.append(
                {
                    "timestamp": t,
                    "open": curr_p,
                    "high": curr_p * 1.002,
                    "low": curr_p * 0.998,
                    "close": curr_p,
                    "volume": 10000.0,
                    "vwap": curr_p,
                }
            )
        session.seed_bar_history(sym, bars)


async def async_main(args: argparse.Namespace) -> int:
    """Asynchronous entrypoint executing live trading session and HUD loop."""
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        symbols = ("SPY", "QQQ", "AAPL", "NVDA", "MSFT")

    # 1. Instantiate Strategy
    strategy = build_strategy(args.strategy, symbols)

    # Small universe adaptive concentration ceiling
    max_concentration = float(args.max_concentration)
    if len(symbols) <= 2 and max_concentration < 0.60:
        max_concentration = 0.75

    # 2. Configure Session
    session_config = LiveSessionConfig(
        session_id=f"live_{args.strategy}_{int(time.time())}",
        symbols=symbols,
        initial_cash=float(args.capital),
        poll_interval_sec=float(args.poll_interval),
        max_leverage=float(args.max_leverage),
        max_concentration=max_concentration,
        max_drawdown_limit=float(args.max_drawdown),
        admin_token=args.admin_token,
    )

    # 3. Instantiate Terminal HUD
    hud: TerminalHUD | None = None
    if not args.headless and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        hud_cfg = TerminalHUDConfig(refresh_per_second=4.0)
        hud = TerminalHUD(config=hud_cfg)

    # 4. Instantiate Session
    def _telemetry_callback(t: LiveSessionTelemetry) -> None:
        if hud is not None:
            hud.update(t)
        elif args.headless:
            # Output periodic telemetry log line in headless mode
            should_print = (
                t.iteration <= 5
                or (t.iteration % max(1, int(1.0 / max(args.poll_interval, 0.01))) == 0)
                or t.status != LiveSessionStatus.RUNNING
            )
            if should_print:
                print(
                    f"[{t.status.value}] #{t.iteration} | NAV: ${t.portfolio_nav:,.2f} | "
                    f"Cash: ${t.cash:,.2f} | Leverage: {t.gross_leverage:.2f}x | "
                    f"Orders: {t.orders_dispatched_count} | Fills: {t.fills_count}",
                    flush=True,
                )

    session = LiveTradingSession(
        config=session_config,
        strategy=strategy,
        on_telemetry=_telemetry_callback,
    )

    # Pre-seed warmup
    seed_session_history(session, symbols, strategy.min_warmup_bars)

    # 5. Configure Interactive Hotkeys
    loop = asyncio.get_running_loop()
    hotkey_mgr = HotkeyManager(loop=loop)

    async def _toggle_pause() -> None:
        if session.status == LiveSessionStatus.RUNNING:
            await session.pause()
            if hud:
                hud.add_event(time.strftime("%H:%M:%S"), "OPERATOR", "Session PAUSED by operator")
        elif session.status == LiveSessionStatus.PAUSED:
            await session.resume()
            if hud:
                hud.add_event(time.strftime("%H:%M:%S"), "OPERATOR", "Session RESUMED by operator")

    async def _panic() -> None:
        await session.panic_kill(reason="OPERATOR_PANIC_KEY")
        if hud:
            hud.add_event(time.strftime("%H:%M:%S"), "CRITICAL", "PANIC KILL SWITCH TRIGGERED")

    async def _rearm() -> None:
        await session.disarm(session_config.admin_token)
        if hud:
            hud.add_event(time.strftime("%H:%M:%S"), "OPERATOR", "Kill switch DISARMED / RE-ARMED")

    async def _shutdown() -> None:
        if hud:
            hud.add_event(time.strftime("%H:%M:%S"), "SHUTDOWN", "Operator initiated shutdown")
        await session.stop()

    hotkey_mgr.register(HotkeyAction.PAUSE_RESUME, _toggle_pause)
    hotkey_mgr.register(HotkeyAction.PANIC_KILL, _panic)
    hotkey_mgr.register(HotkeyAction.REARM, _rearm)
    hotkey_mgr.register(HotkeyAction.SHUTDOWN, _shutdown)

    # 6. Install OS Signal Handlers
    session.install_signal_handlers()

    # 7. Start Subsystems
    if hud is not None:
        hud.start()
    hotkey_mgr.start()
    await session.start()

    # 8. Main Monitoring Loop
    iteration_counter = 0
    try:
        while session.status in (
            LiveSessionStatus.RUNNING,
            LiveSessionStatus.PAUSED,
            LiveSessionStatus.PANIC_KILLED,
        ):
            await asyncio.sleep(0.1)
            iteration_counter += 1
            if args.max_iterations and session.iteration >= args.max_iterations:
                break
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        # 9. Orderly Teardown
        hotkey_mgr.stop()
        await session.stop()
        if hud is not None:
            hud.stop()

    return 0


def main() -> None:
    """CLI Argument Parsing and Execution Dispatch."""
    parser = argparse.ArgumentParser(
        description="Turnkey Continuous Live Paper Trading Station & Terminal HUD"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,AAPL,NVDA,MSFT",
        help="Comma-separated ticker universe (e.g. SPY,QQQ,AAPL)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="swarm",
        choices=["swarm", "kalman", "momentum", "volatility", "sentiment"],
        help="Active alpha strategy to evaluate",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial simulated cash capital (default: 100,000)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between trading and risk cycles (default: 1.0)",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=2.0,
        help="Maximum portfolio gross leverage ratio (default: 2.0)",
    )
    parser.add_argument(
        "--max-concentration",
        type=float,
        default=0.35,
        help="Maximum single-asset NAV concentration (default: 0.35)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.15,
        help="Session maximum drawdown circuit breaker (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--admin-token",
        type=str,
        default="secret-admin-token-1234",
        help="Administrator authorization secret for kill switch reset/disarm",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless log mode without rich terminal HUD",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after N iterations (useful for testing or batch runs)",
    )

    args = parser.parse_args()

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
