# Phase 9: Real-World Financial News Harvester & Causal Price Reaction Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated live financial news ingestion pipeline and causal price reaction prediction engine that scrapes real-world news, extracts entities and financial sentiment, and forecasts directional price breakout probabilities and trajectories.

**Architecture:** A multi-source async HTTP RSS news harvester with SHA-256 deduplication and point-in-time causality (`INV-NEWS-001`, `INV-NEWS-002`) feeds into a Loughran-McDonald financial sentiment classifier with entity-centrality weighting (`INV-NEWS-003`). The classified events pass to a causal drift-diffusion price reaction engine that computes expected price moves, Triple-Barrier breakout probabilities, and post-announcement drift trajectories (`INV-NEWS-004`, `INV-NEWS-005`), streaming live alpha signals into the autonomous trading swarm daemon in $< 10\text{ms}$ (`INV-NEWS-006`).

**Tech Stack:** Python 3.13, FastAPI, httpx, xml.etree.ElementTree, Loughran-McDonald Financial Lexicon, NumPy, Pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-phase-9-live-news-harvester-and-price-reaction-design.md`

## Global Constraints

- Strict Python 3.13 static typing (`mypy src --strict` with zero errors across all files).
- Zero linting or formatting deviations (`ruff check .`, `ruff format --check .`).
- Rule 1: Four-tier docstrings/annotations (Functional Purpose, Explicit Dependency Tracking, Structural Relationship, Defensive Invariant) on every class and public method.
- Rule 2: Diagnostic error codes (`ERR-NEWS-001` through `ERR-NEWS-006`).
- Rule 4: No black-box uncalibrated solvers in the prediction hot path; exact closed-form algebraic formulations.
- 100% automated test pass rate with $\ge 90\%$ line coverage across all new modules.

---

### Task 1: Real-World News Harvester Engine (`news_harvester.py`)

**Files:**
- Create: `src/quant/data/news_harvester.py`
- Test: `tests/unit/test_news_harvester.py`

**Interfaces:**
- Consumes: `httpx.AsyncClient`, standard library `xml.etree.ElementTree`.
- Produces: `NewsArticle`, `NewsFeedConfig`, `NewsHarvester`, `SyntheticNewsGenerator`, and diagnostic exceptions `NewsFeedUnreachableException` (`ERR-NEWS-001`), `CorruptNewsPayloadException` (`ERR-NEWS-002`), `FutureTimestampException` (`ERR-NEWS-003`).

- [ ] **Step 1: Write failing unit tests for NewsArticle and NewsHarvester**
  - Verify RSS XML parsing for Yahoo Finance, SEC EDGAR 8-K, and CNBC wire formats.
  - Verify SHA-256 deduplication ring buffer rejecting re-syndicated articles (`INV-NEWS-002`).
  - Verify rejection of articles with future timestamps (`INV-NEWS-001`, `ERR-NEWS-003`).
  - Verify synthetic replay generator produces valid offline news streams.

- [ ] **Step 2: Run tests to confirm failure**
  - `pytest tests/unit/test_news_harvester.py` (Must fail with `ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/quant/data/news_harvester.py`**
  - Implement `NewsArticle` dataclass with `slots=True`, `frozen=True`.
  - Implement `NewsHarvester` with async HTTP/2 polling, HTML sanitization, and SHA-256 deduplication.
  - Implement `SyntheticNewsGenerator` for hermetic testing.
  - Add Rule 1 four-tier annotations and diagnostic fault codes.

- [ ] **Step 4: Run tests, mypy, and ruff verification**
  - `pytest tests/unit/test_news_harvester.py -v`
  - `mypy src --strict`
  - `ruff check . && ruff format --check .`

- [ ] **Step 5: Commit Task 1**
  - `git add src/quant/data/news_harvester.py tests/unit/test_news_harvester.py`
  - `git commit -m "feat(data): implement real-world financial news harvester and deduplication engine"`

---

### Task 2: Financial Sentiment & Event Taxonomy Classifier (`news_classifier.py`)

**Files:**
- Create: `src/quant/analytics/news_classifier.py`
- Test: `tests/unit/test_news_classifier.py`

**Interfaces:**
- Consumes: `NewsArticle` from Task 1.
- Produces: `EventType`, `ClassifiedNewsEvent`, `FinancialSentimentClassifier`, `NonFiniteSignalException` (`ERR-NEWS-005`), `EmptyUniverseException` (`ERR-NEWS-004`).

- [ ] **Step 1: Write failing unit tests for sentiment classification and entity extraction**
  - Test ticker regex and company name aliasing ("Apple" $\to$ AAPL, "Nvidia" $\to$ NVDA).
  - Test Loughran-McDonald sentiment polarity with financial negation ("loss narrowed" $\to$ positive, "beat expectations" $\to$ positive, "revenue miss" $\to$ negative).
  - Test category classification (`EARNINGS`, `MACRO_FED`, `M_AND_A`, `REGULATORY_LEGAL`, `ANALYST_ACTION`, `GENERAL_MARKET`).
  - Test urgency and novelty bounds in $[0.0, 1.0]$.
  - Test non-finite input rejection (`ERR-NEWS-005`).

- [ ] **Step 2: Run tests to confirm failure**
  - `pytest tests/unit/test_news_classifier.py` (Must fail with `ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/quant/analytics/news_classifier.py`**
  - Implement `EventType` enum and `ClassifiedNewsEvent` dataclass.
  - Implement `FinancialSentimentClassifier` with Loughran-McDonald lexicon and company alias resolver.
  - Enforce bounds: $s \in [-1, 1], u \in [0, 1], \mathcal{H} \in [0, 1]$ (`INV-NEWS-003`).

- [ ] **Step 4: Run tests, mypy, and ruff verification**
  - `pytest tests/unit/test_news_classifier.py -v`
  - `mypy src --strict`
  - `ruff check . && ruff format --check .`

- [ ] **Step 5: Commit Task 2**
  - `git add src/quant/analytics/news_classifier.py tests/unit/test_news_classifier.py`
  - `git commit -m "feat(analytics): implement Loughran-McDonald financial sentiment and event classifier"`

---

### Task 3: Causal Price Reaction Prediction Engine (`price_reaction.py`)

**Files:**
- Create: `src/quant/analytics/price_reaction.py`
- Test: `tests/unit/test_price_reaction.py`

**Interfaces:**
- Consumes: `ClassifiedNewsEvent` from Task 2, `compute_decay_kernel` from `event_service.py`.
- Produces: `PriceReactionPrediction`, `NewsPriceReactionEngine`, `ReactionPredictionTimeoutException` (`ERR-NEWS-006`).

- [ ] **Step 1: Write failing unit tests for price reaction and breakout probabilities**
  - Test expected dollar move $\Delta \hat{P}_{\text{expected}} = P_t \cdot \gamma \cdot \mathcal{S}_i \cdot \sigma_t$ (`INV-NEWS-004`).
  - Test logistic probability monotonicity $\partial \mathbb{P}(\text{UP}) / \partial \mathcal{S}_i > 0$ (`INV-NEWS-005`).
  - Test Triple-Barrier price alignment ($B_{\text{upper}}, B_{\text{lower}}$).
  - Test temporal decay trajectory over elapsed time $\Delta t$.
  - Benchmark latency SLA: $< 10\text{ms}$ per prediction (`INV-NEWS-006`).

- [ ] **Step 2: Run tests to confirm failure**
  - `pytest tests/unit/test_price_reaction.py` (Must fail with `ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/quant/analytics/price_reaction.py`**
  - Implement `PriceReactionPrediction` dataclass.
  - Implement `NewsPriceReactionEngine` with category elasticity, logistic drift-diffusion probability, and dual-kernel trajectory.
  - Add Rule 1 annotations and numerical stability guards.

- [ ] **Step 4: Run tests, mypy, and ruff verification**
  - `pytest tests/unit/test_price_reaction.py -v`
  - `mypy src --strict`
  - `ruff check . && ruff format --check .`

- [ ] **Step 5: Commit Task 3**
  - `git add src/quant/analytics/price_reaction.py tests/unit/test_price_reaction.py`
  - `git commit -m "feat(analytics): implement causal price reaction and Triple-Barrier breakout engine"`

---

### Task 4: News Prediction Service, REST Endpoints & Swarm Coupling

**Files:**
- Create: `src/quant/services/news_prediction_service.py`
- Create: `src/quant/api/v1/endpoints/news.py`
- Modify: `src/quant/services/autonomous_trader.py`
- Modify: `src/quant/api/dependencies.py`
- Modify: `src/quant/main.py`
- Modify: `src/quant/templates/trading_terminal.html`
- Test: `tests/api/test_news_api.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 3, `EventService`, `AutonomousTradingEngine`.
- Produces: `/api/v1/news` REST routes, terminal news prediction HUD widget, and autonomous swarm forward prior injection.

- [ ] **Step 1: Write failing API integration tests**
  - Test `GET /api/v1/news/latest`: retrieves recently harvested and predicted headlines.
  - Test `POST /api/v1/news/harvest`: triggers immediate scrape and returns processed events.
  - Test `POST /api/v1/news/predict`: manual prediction endpoint taking headline + price + vol.
  - Test RBAC authentication guards.

- [ ] **Step 2: Run tests to confirm failure**
  - `pytest tests/api/test_news_api.py` (Must fail).

- [ ] **Step 3: Implement `NewsPredictionService` and REST endpoints**
  - Connect harvester $\to$ classifier $\to$ reaction engine $\to$ `EventService.ingest_batch`.
  - Inject news forward alpha priors into `AutonomousTradingEngine`.
  - Mount `/api/v1/news` router in `main.py`.
  - Add live news feed and predicted price reaction widget to `trading_terminal.html`.

- [ ] **Step 4: Verify full test suite, typing, and formatting**
  - `pytest tests` (All 2,003+ tests green).
  - `mypy src --strict` (0 errors).
  - `ruff check . && ruff format --check .` (0 errors).

- [ ] **Step 5: Commit Task 4**
  - `git add .`
  - `git commit -m "feat(api): integrate news prediction service, REST endpoints, and terminal HUD widget"`
