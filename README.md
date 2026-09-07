# News-Driven Quantitative Prediction Engine

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLAlchemy 2.0](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00?logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![Coverage 89%](https://img.shields.io/badge/Coverage-89.4%25-brightgreen)](https://pytest.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Mypy Strict](https://img.shields.io/badge/Mypy-Strict-blue)](https://mypy-lang.org/)

An institutional-grade, multi-algorithmic market prediction and alpha generation platform that integrates unstructured news ingestion with causal temporal decay kernels, Bayesian game-theoretic scenario stress-testing, and an evolutionary strategy population powered by **algorithmic hypergamy dynamics**.

---

## Architectural Documentation

* **[system_architecture_plan.md](./system_architecture_plan.md)**: Architectural blueprint detailing the 4 core modular layers:
  1. *News Ingestion Layer & Temporal Chaining*: Continuous vectorization, semantic parsing, and dual-decay memory kernels ($\tau_{\text{fast}}$ and $\tau_{\text{slow}}$).
  2. *Game-Theoretic Scenario Generator*: Multi-scenario adversarial simulation (Immediate Reversal, Momentum Cascade, Liquidity Squeeze) and Minimax Regret evaluation.
  3. *Evolutionary Population Manager*: Chromosomal encoding, dynamic adaptive mutation, and **hypergamic mate selection** gated by residual error orthogonality ($\text{Corr}(\mathbf{e}_{\text{Alpha}}, \mathbf{e}_{\text{Aspirant}}) < \delta_{\text{ortho}}$).
  4. *Ensemble Prediction Aggregator*: Regime-weighted Bayesian aggregation and disagreement entropy circuit breakers.
* **[institutional_quant_framework.md](./institutional_quant_framework.md)**: Foundational principles for quantitative research, feature engineering (fractional differentiation, microstructural order flow), statistical significance (Deflated Sharpe Ratio, Multiple Testing Adjustments), Combinatorial Purged Cross-Validation (CPCV), execution optimization (Square-Root Law, Almgren-Chriss), and portfolio allocation (Hierarchical Risk Parity, CVaR risk budgeting).
* **[CHANGELOG_DEV.md](./CHANGELOG_DEV.md)**: Real-time running developer audit trail recording code modifications, configuration changes, architectural adjustments, and trade-offs.

---

## System Architecture

The codebase enforces a **Layered Domain-Driven Design (DDD)** structure to ensure quantitative domain mathematics remain completely decoupled from database frameworks and web servers:

```
quant/
├── src/quant/
│   ├── domain/               # Pure business models, value objects, and repository interfaces (ABCs)
│   │   ├── models.py         # NewsEvent, Asset, Genotype, ScenarioProfile
│   │   └── interfaces.py     # IAssetRepository, IEventRepository, IGenotypeRepository
│   ├── infrastructure/       # Concrete adapters, database ORM, and repository implementations
│   │   ├── database/         # Async engine, sessionmaker, declarative models
│   │   └── repositories/     # SqlAlchemyAssetRepository, SqlAlchemyEventRepository, etc.
│   ├── services/             # Application orchestration & quantitative algorithms
│   │   ├── event_service.py  # Hybrid decay kernel & active news state vector calculation
│   │   └── genotype_service.py # Multi-objective fitness evaluation & population seeding
│   ├── api/                  # FastAPI routers, ASGI middleware, and authentication dependencies
│   │   ├── v1/endpoints/     # /events, /genotypes, /auth
│   │   ├── middleware.py     # Correlation ID (X-Request-ID), timing, RFC 7807 problem details
│   │   └── dependencies.py   # RBAC guards and session dependency injection
│   └── main.py               # Application factory & lifespan manager
├── migrations/               # Alembic database schema migrations
├── tests/                    # Unit, integration, and API test suites (in-memory SQLite)
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

# 3. Static Type Checking (Strict Mode)
mypy src

# 4. Automated Test Suite with Coverage Enforcement (> 85%)
pytest --cov=quant --cov-report=term-missing --cov-fail-under=85
```

All 20 unit, integration, and API tests execute against an isolated in-memory asynchronous SQLite engine with transactional rollback.

---

## Implementation Roadmap

| Phase | Milestone | Focus Areas | Status |
| :--- | :--- | :--- | :--- |
| **Sprint 1** | **Foundational Architecture & Quality Rig** | `src/` layout, async SQLAlchemy ORM, Alembic migrations, FastAPI routing, RBAC, decay kernel, multi-objective fitness, 89% test coverage, GitHub Actions CI. | **Complete** |
| **Sprint 2** | **Econometric Rig & Feature Engineering** | Fractional differentiation $(1-B)^d$, dynamic volatility Triple-Barrier labeling, Meta-Labeling, Combinatorial Purged Cross-Validation (CPCV), Deflated Sharpe Ratio (DSR). | *Upcoming* |
| **Sprint 3** | **Scenario Matrix & Game Theory Engine** | Adversarial market regimes ($\mathbf{s}_1, \mathbf{s}_2, \mathbf{s}_3$), square-root market impact modeling, Minimax Regret evaluator. | *Planned* |
| **Sprint 4** | **Evolutionary Population & Hypergamy** | Orthogonality mating gate ($\text{Corr}(\mathbf{e}_{\text{Alpha}}, \mathbf{e}_{\text{Aspirant}}) < \delta$), adaptive volatility mutation, Pareto elite tracking. | *Planned* |
| **Sprint 5** | **Ensemble Aggregation & Risk Overlays** | Regime-conditioned Bayesian weighting, disagreement entropy circuit breakers, CUSUM kill switches, HRP portfolio allocation. | *Planned* |
