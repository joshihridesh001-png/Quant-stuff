# Quantitative Research Workbench: Frontend Architecture Specification

- **Document Identifier:** SPEC-QUANT-FRONTEND-2026-V1.0
- **Creation Date:** 2026-09-20
- **Status:** Approved / Design Baseline
- **Tech Stack:** React 19, TypeScript, Vite, Tailwind CSS, Recharts, Lucide Icons

---

## 1. Executive Summary & Architectural Scope

This specification establishes the architectural blueprint for the **Quantitative Research Workbench**, a modern, modular, single-page application built with React 19, TypeScript, Vite, and Recharts.

The workbench replaces the legacy monolithic single-file HTML prototype (`src/quant/templates/trading_terminal.html`). Rather than mimicking a retail broker buying app with synthetic order books and sine-wave charts, the workbench is designed as an institutional research and autonomous trading operations console centered strictly on the **5 foundational quantitative pillars** established in `PRD.md` and `SRS.md`:

1. **Pillar 1: Causal News & Temporal Decay Engine**
2. **Pillar 2: Econometric Stationarity & Feature Rig**
3. **Pillar 3: Bayesian Game Theory & Jump Regimes**
4. **Pillar 4: Evolutionary Strategy Swarm (NSGA-II & Hypergamy)**
5. **Pillar 5: Live Replay Simulator, Risk Studio & Swarm Telemetry**

---

## 2. Directory Structure & Modular Layout

The frontend application is contained within the `web/` directory at the repository root:

```
quant/
├── src/quant/                        # Python 3.13 Backend
│   ├── api/v1/                       # REST routes & WebSockets
│   ├── main.py                       # FastAPI application & static mount
│   └── ...
├── web/                              # React 19 SPA Workspace
│   ├── index.html                    # Single-page HTML entry
│   ├── package.json                  # Dependencies & build scripts
│   ├── tsconfig.json                 # Strict TypeScript configuration
│   ├── vite.config.ts                # Vite proxy & build setup
│   └── src/
│       ├── main.tsx                  # React 19 root bootstrap
│       ├── App.tsx                   # Main layout shell & view router
│       ├── index.css                 # Dark institutional styling & Tailwind
│       ├── api/
│       │   ├── client.ts             # Strongly typed REST client & JWT negotiator
│       │   └── useWebSocket.ts       # Full-duplex /ws/risk & /ws/executions hooks
│       ├── types/
│       │   └── quant.ts              # TypeScript schemas matching Pydantic DTOs
│       ├── components/
│       │   ├── HUD.tsx               # Persistent top telemetry bar & metric cards
│       │   ├── Navigation.tsx        # 5-Pillar navigation tab switcher
│       │   ├── DisarmModal.tsx       # HMAC emergency kill switch reset dialog
│       │   └── MetricCard.tsx        # Financial KPI card with sparklines/badges
│       └── views/
│           ├── NewsDecayView.tsx     # Pillar 1: News Harvester & S_news(t)
│           ├── EconometricsView.tsx  # Pillar 2: Fractional Diff & Triple Barrier
│           ├── RegimesView.tsx       # Pillar 3: Bayesian Game Theory & Jump Regimes
│           ├── StrategySwarmView.tsx # Pillar 4: NSGA-II Pareto & Chromosomes
│           └── SimulationRiskView.tsx# Pillar 5: Replay Studio, Blotter & Positions
└── docs/superpowers/specs/
    └── 2026-09-20-quantitative-research-workbench-design.md
```

---

## 3. Communication & State Management Layer

### 3.1 Authentication & REST Client (`web/src/api/client.ts`)
- On application mount, `client.ts` automatically authenticates with `POST /api/v1/auth/token` using default institutional credentials (`username: "admin"`, `role: "ADMIN"`).
- The acquired JWT access token is held in memory and injected into the `Authorization: Bearer <token>` header for all outbound HTTP requests.
- Automatic 401 interceptor re-negotiates tokens if expiration occurs.

### 3.2 Real-Time Full-Duplex WebSockets (`web/src/api/useWebSocket.ts`)
Two persistent, auto-reconnecting WebSocket connections are maintained:

1. **`/api/v1/ws/risk` (Risk Telemetry Feed)**:
   - Handshake: Receives initial `SNAPSHOT` containing complete `RiskStatusResponse` metrics.
   - Heartbeat: Receives periodic 1.0s `RISK_UPDATE` telemetry packets.
   - Alert: Receives immediate `KILL_SWITCH_EVENT` packets on panic trigger or reset.
   - Disconnection handling: Exponential backoff reconnection loop (1s, 2s, 4s, up to 10s max).

2. **`/api/v1/ws/executions` (Order Lifecycle Feed)**:
   - Handshake: Receives `SNAPSHOT` of recent parent orders.
   - Updates: Receives `ORDER_UPDATE` and child slice fills in real time.

---

## 4. Persistent Top Telemetry HUD & Global Controls (`HUD.tsx`)

Pinned across the entire application:

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Q QUANT ALPHA WORKBENCH   WS Risk: 🟢 LIVE  WS Exec: 🟢 LIVE   SWARM: [ RUNNING ] Iter #18   [▶] [⏸] [⏭] [⏹]    [🚨 PANIC LOCK] |
+------------------+-------------------+-------------------+-------------------+-------------------+------------------------+
| 1. PORTFOLIO NAV | 2. CASH & MARGIN  | 3. UNREALIZED P&L | 4. REALIZED P&L   | 5. FIRM LEVERAGE  | 6. DRAWDOWN & BREAKER  |
|    $10,248.50    |    $9,850.00      |    +$42.50 (MTM)  |    +$206.00       |    0.42x Gross    |    0.00% Intraday      |
|    +2.48% Return |    $9,850.00 Free |    2 Positions    |    -$4.20 Fees    |    +0.18x Net     |    🛡️ ARMED (< 5.0%)   |
+------------------+-------------------+-------------------+-------------------+-------------------+------------------------+
```

### 4.1 Global Operational Capabilities
- **Swarm Daemon Lifecycle**:
  - State indicator: `IDLE`, `RUNNING`, `PAUSED`, `STOPPED`.
  - Iteration counter: Displays live rebalance iteration.
  - Controls: Start (`POST /api/v1/autonomous/start`), Pause (`POST /api/v1/autonomous/pause`), Step Once (`POST /api/v1/autonomous/step`), Stop (`POST /api/v1/autonomous/stop`).
- **Emergency Circuit Breaker**:
  - `🚨 PANIC LOCKOUT` button executing `POST /api/v1/risk/panic`. Cancels all open orders across venues, locks out gateway submissions, and halts the swarm.
  - `DisarmModal.tsx`: Enter HMAC admin secret (`QUANT_SECRET_2026_PROD_RECOVERY_KEY`) to restore operations via `POST /api/v1/risk/kill-switch/reset`.
- **6 Key Financial KPI Cards**:
  - NAV (Baseline: $10,000.00), Free Cash & Margin, Unrealized MTM PnL, Realized Settled PnL, Gross & Net Leverage, and Intraday Drawdown.

---

## 5. Specification of the 5 Core Pillar Views

### 5.1 Pillar 1: Causal News & Temporal Decay Engine (`NewsDecayView.tsx`)
- **Backend API**: `GET /api/v1/news/latest`, `POST /api/v1/news/harvest`, `POST /api/v1/news/predict`, `GET /api/v1/events/state/{ticker}`.
- **Components**:
  - Live Ingested News Wire: Filterable by ticker (NVDA, AAPL, MSFT, SPY, QQQ).
  - Loughran-McDonald Sentiment Breakdown: Polarity score $s \in [-1, 1]$, Uncertainty $u \in [0, 1]$, and Shannon entropy $\mathcal{H}$.
  - Bi-Exponential Temporal Decay Curve (Recharts Area Chart):
    $$S_{\text{news}}(t) = S_0 \cdot \left(w_f e^{-t / \tau_{\text{fast}}} + (1 - w_f) e^{-t / \tau_{\text{slow}}}\right)$$
    Visualizes fast half-life ($\tau_{\text{fast}} = 1\text{h}$) and slow macro regime half-life ($\tau_{\text{slow}} = 24\text{h}$).
  - News State Vector Gauge: Real-time norm $\|S_{\text{news}}\|$.
  - Ad-hoc Shock Predictor: Input any financial headline to preview causal price reaction and Triple-Barrier alignment.

### 5.2 Pillar 2: Econometric Stationarity & Feature Rig (`EconometricsView.tsx`)
- **Backend API**: `GET /api/v1/market-data/bars/latest`, `GET /api/v1/providers/macro/sync`.
- **Components**:
  - Fractional Differentiation Optimizer (FFD $d^*$ Search Chart):
    Line chart plotting Augmented Dickey-Fuller (ADF) test $p$-value (stationarity boundary at $p \le 0.01$) against Pearson correlation $\rho$ with raw price as differencing order $d \in [0, 1]$ varies. Highlights the optimal $d^*$ that preserves maximum multi-period memory while guaranteeing stationarity.
  - Dynamic Volatility Triple-Barrier Preview:
    Interactive visualization of Upper Take-Profit barrier, Lower Stop-Loss barrier, and Vertical Holding Period barrier over Parkinson range volatility.
  - Realized Volatility Inspector:
    Rolling 20-bar realized volatility ($\sigma_t$) vs. historical baseline.

### 5.3 Pillar 3: Bayesian Game Theory & Jump Regimes (`RegimesView.tsx`)
- **Backend API**: `GET /api/v1/pre-trade/status`, `GET /api/v1/providers/status`.
- **Components**:
  - 3-Simplex Regime Probabilities (Recharts Bar/Pie Chart):
    Displays instantaneous Bayesian regime probabilities:
    - $P(\text{Absorption})$ (Mean-Reversion regime)
    - $P(\text{Momentum})$ (Trend-Following regime)
    - $P(\text{Panic})$ (Liquidity Cascade / High-volatility regime)
  - Two-Sided CUSUM Shock Detector:
    Live chart tracking positive $S_t^+$ and negative $S_t^-$ drift against the panic threshold.
  - Minimax Regret Payoff Matrix:
    Payoff tensor against Nature and Counterparty strategies (Immediate Reversal, Momentum Cascade, Liquidity Squeeze).

### 5.4 Pillar 4: Evolutionary Strategy Swarm (`StrategySwarmView.tsx`)
- **Backend API**: `GET /api/v1/genotypes/pareto`, `POST /api/v1/genotypes/seed`.
- **Components**:
  - 2D NSGA-II Non-Dominated Pareto Frontier (Recharts Scatter Plot):
    - X-Axis: Max Drawdown (%)
    - Y-Axis: Deflated Sharpe Ratio (DSR)
    - Distinguishes Rank 1 Non-Dominated Frontier (Gold stars) from Aspirants (Cyan dots).
    - Tooltip displays Genotype ID, Cohort, Regret, and Novelty scores.
  - Interactive 20-Gene Chromosome Radar Chart (Recharts Radar Chart):
    Clicking any genotype renders its radial chromosome profile across:
    - $\mathbf{g}_{\text{repr}}$: Decay kernel parameters ($\tau_{\text{fast}}, \tau_{\text{slow}}$, fractional $d$).
    - $\mathbf{g}_{\text{game}}$: Ambiguity temperature, risk aversion $\lambda$, belief priors.
    - $\mathbf{g}_{\text{infer}}$: Execution horizon, profit-take/stop-loss multipliers, meta-label thresholds.
    - $\mathbf{g}_{\text{risk}}$: Volatility target, max weight, turnover budget.
  - Genotype Leaderboard Table with Seed Population trigger.

### 5.5 Pillar 5: Simulation Replay & Risk Studio (`SimulationRiskView.tsx`)
- **Backend API**: `POST /api/v1/simulation/run`, `GET /api/v1/orders`, `POST /api/v1/orders`.
- **Components**:
  - Replay Backtest Studio Form:
    Select Asset, Bar Count, Initial Capital ($10,000 baseline), Fee Schedule (bps), and Slippage.
  - Cumulative Equity Curve (Recharts Area Chart):
    High-refresh mark-to-market equity curve with drawdowns and friction.
  - Quantitative Tear Sheet:
    Total Return, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, 95% CVaR, Friction Paid.
    Deflated Sharpe Ratio (DSR) Certification ($p < 0.05$).
  - Active Positions Table:
    Live mark-to-market inventory, notional value, unrealized PnL, and one-click position liquidation buttons.
  - Live Order Blotter:
    Parent order execution status and child slicing progress streaming via `/ws/executions`.

---

## 6. Build, Deployment & Quality Gates

1. **Vite Development Server**:
   `vite.config.ts` configured with proxy:
   - `/api` -> `http://127.0.0.1:8000/api`
   - `/ws` -> `ws://127.0.0.1:8000/api/v1/ws` (with WebSocket upgrade)
2. **Production Build**:
   - `npm run build` in `web/` outputs to `web/dist/`.
   - FastAPI in `src/quant/main.py` mounts `web/dist/` as static files at `/dashboard` and root `/`.
   - Legacy template `src/quant/templates/trading_terminal.html` is removed.
3. **Quality Verification**:
   - `npm run build` passes with zero TypeScript compiler errors.
   - `python -m pytest` passes 100% (2,088+ tests).
   - `python -m mypy src --strict` passes with 0 errors across all 90 source files.
   - `python -m ruff check .` passes with 0 deviations.
