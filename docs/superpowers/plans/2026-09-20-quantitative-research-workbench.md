# Quantitative Research Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade, modular Single Page Application in React 19 + TypeScript + Tailwind CSS + Recharts strictly implementing the 5 foundational quantitative pillars of the platform, replacing the legacy single-file HTML prototype.

**Architecture:** A dedicated modern frontend workspace in `web/` using Vite, connecting to the FastAPI backend via auto-authenticating REST clients and full-duplex WebSockets (`/ws/risk`, `/ws/executions`). In production, compiled static assets in `web/dist/` are served directly by FastAPI.

**Tech Stack:** React 19, TypeScript 5.7+, Vite 6, Tailwind CSS 3.4, Recharts 2.15+, Lucide React, FastAPI, Python 3.13.

**Spec:** [`docs/superpowers/specs/2026-09-20-quantitative-research-workbench-design.md`](file:///c:/Users/jishu/OneDrive/Documents/quant/docs/superpowers/specs/2026-09-20-quantitative-research-workbench-design.md)

## Global Constraints

- Python 3.13 strict typing (`mypy src --strict`) with zero errors across all source files.
- Zero linting or formatting deviations (`ruff check .`, `ruff format --check .`).
- 100% test pass rate across the entire pytest suite.
- Frontend must build cleanly with zero TypeScript errors under strict mode (`npm run build`).
- Zero mock interval timers in the frontend: all telemetry must stream from live backend APIs and WebSockets.
- $10,000.00 standard retail baseline portfolio capital.

---

### Task 1: Scaffolding `web/` Workspace & TypeScript Domain Schemas

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.node.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/index.css`
- Create: `web/src/types/quant.ts`

**Interfaces:**
- Produces: TypeScript domain interfaces matching backend Pydantic models (`RiskMetrics`, `AutonomousStatus`, `ParentOrder`, `Genotype`, `SimulationReport`, `NewsItem`, `NewsDecayState`).

- [ ] **Step 1: Create `web/package.json` with React 19, Vite, Tailwind, Recharts, and Lucide**
- [ ] **Step 2: Create `web/tsconfig.json` with strict typing (`strict: true`, `noImplicitAny: true`)**
- [ ] **Step 3: Create `web/vite.config.ts` configuring development proxy for `/api` and `/ws` to `http://127.0.0.1:8000`**
- [ ] **Step 4: Create `web/src/types/quant.ts` with complete domain interfaces matching Pydantic DTOs**
- [ ] **Step 5: Run `npm install` in `web/` to generate lockfile and install dependencies**
- [ ] **Step 6: Commit: `feat(web): initialize React 19 Vite workspace with TypeScript domain types`**

---

### Task 2: Authenticated REST Client & Resilient WebSocket Hooks

**Files:**
- Create: `web/src/api/client.ts`
- Create: `web/src/api/useWebSocket.ts`
- Create: `web/src/api/context.tsx`

**Interfaces:**
- Consumes: `web/src/types/quant.ts`
- Produces:
  - `apiClient`: Typed fetch wrapper with automatic JWT negotiation and 401 recovery.
  - `useRiskWebSocket()`: Reactive hook streaming `RiskMetrics` and gateway status from `/ws/risk`.
  - `useExecutionsWebSocket()`: Reactive hook streaming live parent orders and child fills from `/ws/executions`.

- [ ] **Step 1: Implement `web/src/api/client.ts` with auto-negotiated `/api/v1/auth/token` authentication**
- [ ] **Step 2: Implement `web/src/api/useWebSocket.ts` with exponential backoff and message parsing**
- [ ] **Step 3: Implement `web/src/api/context.tsx` providing global state across all components**
- [ ] **Step 4: Verify TypeScript compilation passes without errors**
- [ ] **Step 5: Commit: `feat(web): implement authenticated REST client and WebSocket hooks`**

---

### Task 3: Pinned HUD, Telemetry & Emergency Panic Controls

**Files:**
- Create: `web/src/components/HUD.tsx`
- Create: `web/src/components/MetricCard.tsx`
- Create: `web/src/components/DisarmModal.tsx`
- Create: `web/src/components/Navigation.tsx`

**Interfaces:**
- Consumes: `useRiskWebSocket()`, `apiClient`
- Produces: Top-level HUD header, 6 KPI cards, Swarm daemon lifecycle buttons, and Emergency Panic Modal.

- [ ] **Step 1: Implement `MetricCard.tsx` with formatted values, deltas, and glowing status borders**
- [ ] **Step 2: Implement `DisarmModal.tsx` for entering HMAC secret to disarm kill switch**
- [ ] **Step 3: Implement `HUD.tsx` rendering WebSocket indicators, Swarm controls, and the 6 KPI cards**
- [ ] **Step 4: Implement `Navigation.tsx` rendering the 5-Pillar tab switcher**
- [ ] **Step 5: Commit: `feat(web): build persistent HUD, metric cards, and emergency kill switch controls`**

---

### Task 4: Pillar 1 & 2 Views (Causal News & Econometric Rig)

**Files:**
- Create: `web/src/views/NewsDecayView.tsx`
- Create: `web/src/views/EconometricsView.tsx`

**Interfaces:**
- Consumes: `apiClient` (`/api/v1/news/*`, `/api/v1/events/state/*`, `/api/v1/market-data/*`)
- Produces:
  - `NewsDecayView`: Ingested headlines, Loughran-McDonald sentiment breakdown, Recharts bi-exponential decay curve $S_{\text{news}}(t)$, and scenario shock simulator.
  - `EconometricsView`: Fractional Differentiation $d^*$ optimizer chart (ADF $p$-value vs. correlation $\rho$), dynamic volatility Triple-Barrier preview, and DuckDB realized volatility.

- [ ] **Step 1: Implement `NewsDecayView.tsx` with Recharts decay curve and live news harvester trigger**
- [ ] **Step 2: Implement `EconometricsView.tsx` with FFD stationarity search curve and barrier visualizer**
- [ ] **Step 3: Verify TypeScript builds cleanly (`npm run build`)**
- [ ] **Step 4: Commit: `feat(web): implement Pillar 1 (News Decay) and Pillar 2 (Econometric Rig) views`**

---

### Task 5: Pillar 3 & 4 Views (Bayesian Regimes & Evolutionary Swarm)

**Files:**
- Create: `web/src/views/RegimesView.tsx`
- Create: `web/src/views/StrategySwarmView.tsx`

**Interfaces:**
- Consumes: `apiClient` (`/api/v1/pre-trade/*`, `/api/v1/genotypes/*`)
- Produces:
  - `RegimesView`: 3-Simplex regime probability chart ($P(\text{Absorption}), P(\text{Momentum}), P(\text{Panic})$), two-sided CUSUM shock detector, and Minimax Regret payoff matrix.
  - `StrategySwarmView`: 2D NSGA-II Non-Dominated Pareto Frontier scatter plot (MaxDD vs. DSR), interactive 20-gene chromosome radar profile, and population seeding controls.

- [ ] **Step 1: Implement `RegimesView.tsx` with regime simplex distribution and CUSUM shock chart**
- [ ] **Step 2: Implement `StrategySwarmView.tsx` with Recharts Pareto scatter plot and chromosome radar chart**
- [ ] **Step 3: Verify TypeScript builds cleanly**
- [ ] **Step 4: Commit: `feat(web): implement Pillar 3 (Game Theory & Regimes) and Pillar 4 (Strategy Swarm) views`**

---

### Task 6: Pillar 5 View (Simulation Replay & Risk Studio)

**Files:**
- Create: `web/src/views/SimulationRiskView.tsx`
- Create: `web/src/App.tsx`
- Create: `web/src/main.tsx`

**Interfaces:**
- Consumes: `apiClient` (`/api/v1/simulation/run`, `/api/v1/orders`), `useExecutionsWebSocket()`, `useRiskWebSocket()`
- Produces:
  - ReplayEngine backtest runner form and interactive cumulative equity curve.
  - Quantitative tear sheet metrics with DSR statistical significance certification ($p < 0.05$).
  - Active positions table with one-click liquidation buttons.
  - Live execution order blotter with child slicing progress streaming.

- [ ] **Step 1: Implement `SimulationRiskView.tsx` with backtest runner, equity chart, blotter, and positions**
- [ ] **Step 2: Implement `App.tsx` integrating HUD, Navigation, and view routing**
- [ ] **Step 3: Implement `main.tsx` bootstrapping React 19**
- [ ] **Step 4: Run `npm run build` in `web/` to confirm complete, clean production bundle generation**
- [ ] **Step 5: Commit: `feat(web): implement Pillar 5 (Simulation & Risk Studio) and complete App shell`**

---

### Task 7: Backend FastAPI Static Integration & Legacy Template Deprecation

**Files:**
- Modify: `src/quant/main.py`
- Modify: `tests/api/test_health_and_auth_api.py`
- Delete: `src/quant/templates/trading_terminal.html`

**Interfaces:**
- Consumes: `web/dist/`
- Produces: FastAPI static files mounted at `/dashboard` and root `/`.

- [ ] **Step 1: Update `src/quant/main.py` to mount `web/dist` when present, serving `index.html` at `/dashboard`**
- [ ] **Step 2: Remove legacy single-file prototype `src/quant/templates/trading_terminal.html`**
- [ ] **Step 3: Update `tests/api/test_health_and_auth_api.py` to assert dashboard response matches React bundle**
- [ ] **Step 4: Run `python -m pytest` across all test suites to ensure 100% green pass rate**
- [ ] **Step 5: Run `python -m mypy src --strict` and `python -m ruff check .`**
- [ ] **Step 6: Commit: `feat(api): mount compiled React Quantitative Research Workbench and deprecate legacy template`**

---

### Task 8: End-to-End Verification & Operational Validation

**Files:**
- Verification only

- [ ] **Step 1: Verify `npm run build` generates clean output in `web/dist/` with 0 errors**
- [ ] **Step 2: Run full Python test suite: `python -m pytest` (100% passing across all 2,088+ tests)**
- [ ] **Step 3: Run static type checks: `python -m mypy src --strict` (0 errors across 90 files)**
- [ ] **Step 4: Run linter and formatter: `python -m ruff check .` and `python -m ruff format --check .` (0 deviations)**
- [ ] **Step 5: Start background Uvicorn server and verify HTTP 200 on `/dashboard` and `/healthz`**
- [ ] **Step 6: Push changes to `origin main`**
