# Turnkey Continuous Live Paper Trading Station: Operator Manual

## 1. Executive Overview

The **Turnkey Continuous Live Paper Trading Station** provides an institutional-grade, real-time autonomous trading environment. Built upon the quantitative foundation of the five institutional pillars (DuckDB Columnar Data Lake, Modular Alpha Library, CFA-Grade Backtesting Studio, Double-Entry Relational Tax-Lot Ledger, and Bayesian Pre-Trade Risk Firewall), the station executes sub-second trading cycles with continuous real-time risk surveillance, terminal telemetry visualization, and non-blocking operator hotkeys.

```
+-----------------------------------------------------------------------------------------+
|                                LIVE TRADING EVENT LOOP                                  |
|                                                                                         |
|  [Market Data Poller] ---> [Historical Buffer] ---> [Bayesian Regime Filter]            |
|                                                              |                          |
|                                                              v                          |
|  [Tax-Lot Ledger] <--- [Execution Gateway] <--- [Pre-Trade Risk] <--- [Alpha Swarm]    |
|          |                                             |                                |
|          +---------------------------------------------+                                |
|                                  |                                                      |
|                                  v                                                      |
|                 [Terminal HUD & Hotkey Controller]                                      |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Command-Line Interface (CLI)

The live trading station is invoked via the turnkey script [`scripts/run_live_trader.py`](file:///scripts/run_live_trader.py).

### Quickstart Commands

```bash
# 1. Run Composite Evolutionary Swarm on top 5 liquid assets ($100k capital, 1s tick)
python scripts/run_live_trader.py --symbols SPY,QQQ,AAPL,NVDA,MSFT --strategy swarm

# 2. Run Kalman Filter Statistical Arbitrage on SPY/QQQ with $250k initial capital
python scripts/run_live_trader.py --symbols SPY,QQQ --strategy kalman --capital 250000

# 3. Run Memory-Preserving FracDiff Momentum with custom risk thresholds
python scripts/run_live_trader.py --symbols SPY,QQQ,AAPL --strategy momentum --max-leverage 1.5 --max-concentration 0.40

# 4. Headless background daemon execution (ideal for tmux / systemd)
python scripts/run_live_trader.py --symbols SPY,QQQ,AAPL,NVDA,MSFT --strategy swarm --headless --poll-interval 2.0
```

### CLI Arguments Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--symbols` | `str` | `SPY,QQQ,AAPL,NVDA,MSFT` | Comma-separated universe of ticker symbols to monitor and trade. |
| `--strategy` | `str` | `swarm` | Alpha strategy selector: `swarm`, `kalman`, `momentum`, `volatility`, `sentiment`. |
| `--capital` | `float` | `100000.0` | Initial starting cash balance allocated to the paper trading session. |
| `--poll-interval` | `float` | `1.0` | Cycle tick interval in seconds between bar ingestion and signal calculation. |
| `--max-leverage` | `float` | `2.0` | Portfolio gross leverage upper bound enforced by the pre-trade risk firewall. |
| `--max-concentration` | `float` | `0.35` | Single-asset NAV concentration ceiling (dynamically adjusted for universes $\le 2$). |
| `--max-drawdown` | `float` | `0.15` | Peak-to-trough intraday drawdown circuit breaker ($15\%$) triggering automatic panic kill. |
| `--admin-token` | `str` | `secret-admin-token-1234`| Constant-time token secret required to disarm and re-arm the emergency kill switch. |
| `--headless` | `flag` | `False` | Run in streaming text log mode without the fullscreen interactive terminal HUD. |
| `--max-iterations` | `int` | `None` | Optional iteration budget before graceful automatic session shutdown. |

---

## 3. Interactive Operator Hotkeys

When running in interactive terminal mode (non-headless), the background thread captures asynchronous single-key commands without requiring the Enter key or blocking the event loop:

| Hotkey | Action | Effect |
|---|---|---|
| **`[SPACE]`** | **Pause / Resume** | Toggles the session state between `RUNNING` and `PAUSED`. When paused, market data buffering continues, but alpha signal execution and order generation are suspended. |
| **`[K]`** | **Emergency Panic Kill** | Instantly arms the `EmergencyKillSwitch` (`INV-RSK-008`), atomically triggers concurrent cancellation sweeps across all open orders, and locks the pre-trade firewall against new orders. |
| **`[R]`** | **Re-Arm / Disarm** | Re-arms the trading session by verifying the administrator secret token via constant-time authentication (`INV-RSK-008`). Restores normal trading readiness. |
| **`[Q]`** | **Orderly Shutdown** | Traps execution, flushes telemetry, cancels resting orders, logs reconciliation summaries, and exits with return code `0`. |

---

## 4. Rich Terminal HUD Architecture

The terminal HUD utilizes `rich` live display buffers running at 4 FPS to present high-density institutional telemetry:

1. **Header Ribbon:**
   - Active strategy, operational status badge (`RUNNING`, `PAUSED`, `PANIC_KILLED`), cycle counter, and current market timestamp.
2. **Portfolio Telemetry Bar:**
   - Current Net Asset Value (NAV), Free Cash balance, Gross Leverage ($x$), Net Exposure ($\Delta$), and Session Intraday Drawdown ($\%$).
3. **Strategy Swarm Allocation Panel:**
   - Relative simplex weights allocated across active sub-strategies (Kalman Stat-Arb, FracDiff Momentum, Volatility Breakout, LM Sentiment) driven by Entropic Mirror Descent.
4. **Active Orders & Fills Table:**
   - Real-time rolling ledger of the 10 most recent orders: order ID, symbol, side (`BUY`/`SELL`), order type, price, quantity, execution status (`FILLED`, `REJECTED`, `NEW`), and gateway latency.

---

## 5. Risk Safeguards & Diagnostic Error Codes

Every event, state transition, and failure condition produces deterministic diagnostic codes:

| Fault Code | System Component | Description |
|---|---|---|
| `ERR-LIVE-001` | Live Trading Session | Non-finite or boolean parameter supplied to session configuration. |
| `ERR-LIVE-002` | Live Trading Session | Invalid or illegal state transition attempted (e.g., resuming from terminated state). |
| `ERR-LIVE-003` | Live Trading Session | Warmup horizon failure (insufficient bar history available for alpha calculation). |
| `ERR-LIVE-004` | Live Trading Session | Peak equity drawdown breach exceeding configured circuit breaker. |
| `ERR-LIVE-005` | Live Trading Session | Invalid or unauthorized admin token supplied for kill switch disarm. |
| `ERR-HUD-001` | Terminal HUD | Non-finite numeric value detected in telemetry update payload. |
| `ERR-HUD-002` | Terminal HUD | Layout or render failure encountered in terminal buffer formatting. |
| `ERR-KEY-001` | Hotkey Controller | Hotkey manager background polling thread failure or termination error. |
| `ERR-KEY-002` | Hotkey Controller | Invalid key registration or invalid callback callable. |
| `ERR-RSK-008` | Emergency Kill Switch | Order rejected because emergency kill switch is currently armed/active. |

---

## 6. Operational Runbook

### Starting a Live Session
1. Ensure the historical market data lake is populated (`data/market_data.duckdb`).
2. Run the desired strategy command line with appropriate leverage limits.
3. Observe initial warmup status in the header ribbon until `RUNNING` turns green.

### Responding to a Risk Breach
If the intraday drawdown circuit breaker triggers:
1. The status badge will flash red with `PANIC_KILLED` and all pending orders will be mass-cancelled.
2. Review the incident log event table in the HUD.
3. Once market conditions stabilize, press `[R]` to re-arm the engine with the admin token.

### Shutting Down
1. Press `[Q]` in the terminal window (or send `SIGINT`/`SIGTERM` via `Ctrl+C`).
2. The orchestrator will safely cancel all pending orders, sync open tax lots to the relational database, print the final session NAV summary, and exit cleanly.
