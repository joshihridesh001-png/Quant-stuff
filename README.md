# News-Driven Quantitative Prediction Engine

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLAlchemy 2.0](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00?logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![Coverage 94%](https://img.shields.io/badge/Coverage-94%25-brightgreen)](https://pytest.org/)
[![Tests 1456 Passed](https://img.shields.io/badge/Tests-1456%20Passed-brightgreen)](https://pytest.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Mypy Strict](https://img.shields.io/badge/Mypy-Strict-blue)](https://mypy-lang.org/)

An institutional-grade, multi-algorithmic market prediction and alpha generation platform that integrates unstructured news ingestion with causal temporal decay kernels, Bayesian game-theoretic scenario stress-testing, and an evolutionary strategy population powered by **algorithmic hypergamy dynamics**.

---

## Canonical Documentation Suite

The system's operational and architectural standards are codified across 6 canonical documents:

1. **[`PRD.md`](./PRD.md)**: Product Requirements Document — product vision, market inefficiencies, user personas, and feature matrices.
2. **[`Architecture.md`](./Architecture.md)**: System Architecture & Component Topology — layered domain design, hybrid storage (DuckDB + PostgreSQL), module maps, and data flow pipelines.
3. **[`Rules.md`](./Rules.md)**: The Three Fundamental Engineering Rules — micro-level code transparency, zero-execution diagnostics, and strict CI quality gates.
4. **[`Phases.md`](./Phases.md)**: Implementation Roadmap & Delivery Phases — granular phase-by-phase delivery schedule (Sprint 1 through Phase 6 Step 1 complete).
5. **[`Design.md`](./Design.md)**: Quantitative & Technical Design Specification — mathematical formulations for decay kernels, fractional diff, triple-barrier labeling, CPCV, DSR, hypergamic mating, RD-DMA, circuit breakers, EVT tail risk, and live execution gateways.
6. **[`Memory.md`](./Memory.md)**: Project Memory — Autonomous Decision Register (ADR-001 through ADR-022), Deterministic Diagnostic Failure Matrix (ERR-GW-001..006), and complete engineering audit trail.

---

## System Architecture

The codebase enforces a **Layered Domain-Driven Design (DDD)** structure to ensure quantitative domain mathematics remain completely decoupled from database frameworks and web servers:

```
quant/
├── src/quant/
│   ├── domain/               # Pure business models, value objects, and repository interfaces (ABCs)
│   │   ├── models.py         # NewsEvent, Asset, Genotype, PriceBar, MarketDataBatch
│   │   └── interfaces.py     # IAssetRepository, IEventRepository, IGenotypeRepository, IMarketDataRepository
│   ├── analytics/            # High-performance econometric, game-theoretic, evolutionary & simulation engines
│   │   ├── fractional_diff.py       # Memory-preserving fractional differentiation (FFD)
│   │   ├── labeling.py              # Dynamic volatility triple-barrier labeling
│   │   ├── cross_validation.py      # Combinatorial purged cross-validation (CPCV)
│   │   ├── meta_labeling.py         # Two-stage continuous-payoff Kelly meta-labeling
│   │   ├── deflated_sharpe.py       # Deflated Sharpe Ratio (DSR) & MinBTL
│   │   ├── regimes.py               # Causal Bayesian jump-regime filter & OAS covariance
│   │   ├── market_impact.py         # Multi-asset cross-impact propagator & Pseudo-Huber
│   │   ├── payoff_matrix.py         # Stackelberg leader-follower trajectory & payoff tensor
│   │   ├── minimax_regret.py        # Vectorized Newton entropic minimax regret solver
│   │   ├── chromosomes.py           # Scale-free 20-gene chromosome vector codec
│   │   ├── pareto_sorting.py        # Boundary-anchored RVEA & SVD subspace orthogonal sorting
│   │   ├── hypergamic_selection.py  # Hypergamic assortative mating & residual orthogonality
│   │   ├── evolutionary_lifecycle.py # Cauchy mutation, APD Rechenberg adaptation & (mu+lambda) selection
│   │   ├── ensemble.py              # Regime-conditioned DMA (RD-DMA) & Entropic Mirror Descent
│   │   ├── circuit_breakers.py      # Epistemic disagreement entropy & multi-tier circuit breakers
│   │   ├── tail_risk.py             # Semi-parametric EVT-POT GPD with closed-form PWM & CVaR
│   │   ├── execution_sizing.py      # Strictly concave convex sizer & 2D Newton dual projection
│   │   └── simulation.py            # End-to-end live replay simulator & institutional benchmarking
│   ├── execution/            # Live execution gateway, order state machine & async audit logger
│   │   ├── models.py         # Pure domain entities (Order, ExecutionReport, enums, errors)
│   │   ├── fsm.py            # OrderStateMachine with causal out-of-order fill reconciliation
│   │   ├── idempotency.py    # IdempotencyRouter with deterministic UUIDv5 & FIFO ring buffer
│   │   ├── gateway.py        # ExecutionGateway protocol & PaperExecutionGateway broker
│   │   └── audit.py          # Non-blocking async SQLite WAL order audit logger
│   ├── infrastructure/       # Concrete adapters, database ORM, and repository implementations
│   │   ├── database/         # Async engine, sessionmaker, declarative models, DuckDBManager
│   │   └── repositories/     # SqlAlchemyAssetRepository, SqlAlchemyEventRepository, DuckDBMarketDataRepository
│   ├── services/             # Application orchestration & quantitative algorithms
│   │   ├── event_service.py  # Hybrid decay kernel & active news state vector calculation
│   │   ├── genotype_service.py # Multi-objective fitness evaluation & population seeding
│   │   └── market_data_service.py # Columnar bar queries & realized volatility
│   ├── api/                  # FastAPI routers, ASGI middleware, and authentication dependencies
│   │   ├── v1/endpoints/     # /events, /genotypes, /auth, /market-data
│   │   ├── middleware.py     # Correlation ID (X-Request-ID), timing, RFC 7807 problem details
│   │   └── dependencies.py   # RBAC guards and session dependency injection
│   └── main.py               # Application factory & lifespan manager
├── migrations/               # Alembic database schema migrations
├── tests/                    # Unit, integration, and API test suites (1456 tests, 100% green)
├── pyproject.toml            # PEP 621 packaging, dependency locks, and linter settings
└── CHANGELOG_DEV.md          # Technical audit log
```

---

## Quickstart Guide

### 1. Environment Setup

Clone the repository and create an isolated virtual environment:

```bash
git clone https://github.com/joshihridesh001-png/Quant-stuff.git
cd Quant-stuff

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies and local package in editable mode
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Environment Configuration & Database Parity

Copy the sample environment variables:

```bash
cp .env.example .env
```

**Development Database Options**:
* **Option A: Containerized PostgreSQL with `pgvector` (Recommended for Dev/Prod Parity)**:
  ```bash
  docker compose up -d
  ```
  Uses the official `pgvector/pgvector:pg16` image providing PostgreSQL 16 with native vector index support.
* **Option B: Local Embedded SQLite**:
  Set `DATABASE_URL=sqlite+aiosqlite:///./quant.db` in `.env` for zero-dependency local experimentation.

### 3. Database Migrations

Apply database schema migrations via Alembic:

```bash
alembic upgrade head
```

This provisions the relational tables (`assets`, `events`, `event_centralities`, `genotypes`) and composite indices.

---

## Running the Application

Launch the local ASGI server using Uvicorn:

```bash
uvicorn quant.main:app --reload --port 8000
```

Once running:
* **Interactive OpenAPI Documentation (Swagger UI)**: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)
* **Alternative ReDoc Documentation**: [`http://127.0.0.1:8000/redoc`](http://127.0.0.1:8000/redoc)
* **System Health Check**: [`http://127.0.0.1:8000/healthz`](http://127.0.0.1:8000/healthz)

---

## Primary API Endpoints

### 1. News Ingestion & Active State
* `POST /api/v1/events/ingest`: Ingest a single unstructured news event with continuous asset centrality weights ($c_{i,k}$). (Protected by `X-API-Key`).
* `POST /api/v1/events/batch`: High-throughput atomic ingestion of up to 500 news events per request. (Protected by `X-API-Key`).
* `GET /api/v1/events/{id}`: Retrieve ingested event metadata and sentiment vectors.
* `GET /api/v1/events/state/{ticker}`: Calculate the active time-decayed news state vector $\mathbf{S}_{\text{news}}^{(k)}(t)$ using the hybrid dual-decay kernel with numerical truncation tolerance ($\epsilon \le 10^{-4}$):
  $$\kappa(\Delta t, u) = \alpha \exp\left(-\frac{\Delta t}{\tau_{\text{fast}} \cdot (1 - u)}\right) + (1 - \alpha)\left(1 + \frac{\Delta t}{\tau_{\text{slow}}}\right)^{-\beta}$$
  *Pass `use_projected_subspace=true` to project dense embeddings into an energy-balanced subspace ($d_p=16$).*

### 2. Evolutionary Strategy Population
* `POST /api/v1/genotypes`: Register candidate strategy chromosome $\mathbf{g}_k = \langle \mathbf{g}_{\text{repr}}, \mathbf{g}_{\text{game}}, \mathbf{g}_{\text{infer}}, \mathbf{g}_{\text{risk}} \rangle$.
* `POST /api/v1/genotypes/seed`: Seed generation 0 population with stratified Alpha ($20\%$) and Aspirant ($80\%$) cohorts. (Requires `ADMIN` or `RESEARCHER` role).
* `GET /api/v1/genotypes/alpha`: Query the elite Alpha cohort sorted by multi-objective fitness.
* `GET /api/v1/genotypes/pareto`: Query the population ranked by **NSGA-II Non-Dominated Sorting** and crowding distance across performance, drawdown, regret, and novelty vectors.
* `POST /api/v1/genotypes/{id}/evaluate`: Evaluate and record Deflated Sharpe Ratio (DSR), Max Drawdown, and composite multi-objective fitness:
  $$\mathcal{F}(\mathcal{I}_i) = \text{DeflatedSharpe} \cdot e^{-\psi \cdot \text{MaxDD}} + \omega_1 \cdot V_i(\text{Regret}) + \omega_2 \cdot \mathcal{H}_{\text{novelty}}$$

### 3. Authentication & RBAC
* `POST /api/v1/auth/token`: Issue signed RFC 7519 JWT access tokens with constant-time verification for researchers and administrators.

---

## Verification & Quality Assurance

The codebase enforces strict static analysis and automated testing:

```bash
# 1. Static Linting & Code Style
ruff check .

# 2. Code Formatting Verification
ruff format --check .

# 3. Static Type Checking (Strict Mode across 57 source files)
mypy src --strict

# 4. Automated Test Suite with Coverage Enforcement (> 85%)
pytest tests/unit
```

All **1456 unit tests** pass with 100% green execution in $< 8$ seconds.

---

## Implementation Roadmap

| Phase | Milestone | Focus Areas | Status |
| :--- | :--- | :--- | :--- |
| **Sprint 1** | **Foundational Architecture & Quality Rig** | `src/` layout, async SQLAlchemy ORM, Alembic migrations, FastAPI routing, RBAC, decay kernel, multi-objective fitness, 89% test coverage, GitHub Actions CI. | **Complete** |
| **Sprint 2: Step 1** | **Market Data Entity & Columnar Storage** | Immutable `PriceBar` with defensive invariants, contiguous `MarketDataBatch`, embedded DuckDB engine, PyArrow zero-copy bulk ingestion, rolling realized volatility ($\sigma_t$). | **Complete** |
| **Sprint 2: Step 2** | **Fractional Differentiation Engine** | Memory-preserving differentiation $(1-B)^d$, binomial weight series with tolerance truncation ($\epsilon \le 10^{-4}$), automated stationarity search via ADF test. | **Complete** |
| **Sprint 2: Step 3** | **Dynamic Volatility Triple-Barrier Labeling** | Path-dependent horizontal/vertical barrier detection, realized volatility dynamic threshold scaling, un-hit expiration classification. | **Complete** |
| **Sprint 2: Step 4** | **Combinatorial Purged Cross-Validation (CPCV)** | Non-IID combinatorial partition generator $\binom{N}{k}$, temporal event purging, post-test embargo windows. | **Complete** |
| **Sprint 2: Step 5** | **Two-Stage Meta-Labeling Architecture** | Primary directional model decoupling, secondary probability-calibrated betting classifier ($z_t \in \{0, 1\}$), capacity-aware sizing. | **Complete** |
| **Sprint 2: Step 6** | **Deflated Sharpe Ratio (DSR) & Statistical Testing** | Adjustment for non-normality (skewness, kurtosis), sample length $T$, trial count $K$, and variance of trials $V[\{SR\}]$. | **Complete** |
| **Sprint 3** | **Scenario Matrix & Game Theory Engine** | Causal Bayesian jump-regimes, OAS covariance shrinkage, 3/2-power Pseudo-Huber cross-impact, Stackelberg trajectory, entropic minimax regret solver. | **Complete** |
| **Sprint 4** | **Evolutionary Strategy Search & Population Management** | 20-gene chromosome codec, boundary-anchored adaptive RVEA, hypergamic assortative mating, adaptive Cauchy mutation, $(\mu + \lambda)$ lifecycle engine. | **Complete** |
| **Sprint 5** | **Ensemble Aggregator, Tail Risk & Live Replay Simulator** | Regime-conditioned DMA (RD-DMA), epistemic disagreement entropy circuit breakers, semi-parametric EVT-POT GPD with closed-form PWM, unified strictly concave convex sizer, end-to-end live replay simulator & institutional benchmarking. | **Complete** |
| **Phase 6: Step 1** | **Live Execution Gateway, State Machine & Audit Logger** | Pure domain models (`Order`, `ExecutionReport`), deterministic FSM (`OrderStateMachine`) with out-of-order reconciliation, UUIDv5 `IdempotencyRouter`, `PaperExecutionGateway` broker, and non-blocking SQLite WAL `OrderAuditLogger`. | **Complete** |
| **Phase 6: Step 2** | **Microstructural Smart Order Router (SOR) & Dark Pool Routing** | Multi-venue routing, lit/dark pool liquidity fragmentation splitting, Iceberg and pegged order execution algorithms. | *Upcoming* |
| **Phase 6: Step 3** | **Real-Time Risk Monitor, OMS Heartbeats & Emergency Kill Switch** | Pre-trade/post-trade risk limits, exchange heartbeat monitors, and automated circuit breaker kill-switch tripwires. | *Upcoming* |

