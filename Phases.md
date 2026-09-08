# Implementation Roadmap & Delivery Phases

## 1. Multi-Sprint Phased Roadmap

```mermaid
gantt
    title Quantitative Prediction Engine Multi-Phase Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    Foundational Architecture & Quality Rig :done, p1, 2026-09-01, 2026-09-07
    section Phase 2: Econometrics
    Step 1: Market Data & Columnar Store    :done, s2_1, 2026-09-08, 1d
    Step 2: Fractional Differentiation       :active, s2_2, 2026-09-09, 2d
    Step 3: Triple-Barrier Labeling         :s2_3, after s2_2, 2d
    Step 4: Combinatorial Purged CV (CPCV)  :s2_4, after s2_3, 3d
    Step 5: Two-Stage Meta-Labeling         :s2_5, after s2_4, 2d
    Step 6: Deflated Sharpe Ratio (DSR)     :s2_6, after s2_5, 2d
    section Phase 3: Game Theory
    Scenario Matrix & Adversarial Payoffs   :p3, 2026-09-22, 5d
    section Phase 4: Evolution
    Hypergamic Selection & Population Engine :p4, 2026-09-29, 5d
    section Phase 5: Productionization
    Ensemble Aggregation & Risk Overlays    :p5, 2026-10-06, 5d
```

---

## 2. Phase Breakdown & Milestone Gates

### Phase 1: Foundational Architecture & Quality Rig [COMPLETE]
* **Status:** Complete (2026-09-07)
* **Deliverables:**
  * Clean `src/` layout preventing uninstalled package imports.
  * Async SQLAlchemy engine and Alembic schema migrations (`001_initial_schema.py`).
  * Domain entities: `Asset`, `NewsEvent`, `EventCentrality`, `Genotype`, `ScenarioProfile`.
  * Application services: Causal decay kernel $\kappa(\Delta t, u)$ and NSGA-II Pareto sorting.
  * Presentation layer: FastAPI v1 REST routes (`/events`, `/genotypes`, `/auth/token`).
  * Security: Constant-time API key verification and RFC 7519 HS256 JWT RBAC.
  * Automated testing: 30 tests passing with **87.18% coverage**; strict typing (`mypy`).

---

### Phase 2: Econometric Rig & Feature Engineering [IN PROGRESS]

To eliminate cognitive overload and merge conflicts, Phase 2 is decomposed into 6 sequential micro-steps:

#### Step 1: Market Data Entity & Columnar Storage Subsystem [COMPLETE]
* **Deliverables:** Immutable `PriceBar` value object with defensive boundary invariants; `MarketDataBatch` with zero-copy PyArrow table export; embedded DuckDB columnar storage (`DuckDBManager`, `DuckDBMarketDataRepository`); `MarketDataService` with rolling realized volatility ($\sigma_t$); `/api/v1/market-data` REST endpoints; 23 tests (53 total) passing with **88.79% coverage**.

#### Step 2: Fractional Differentiation Engine [UPCOMING]
* **Scope:**
  * Memory-preserving differencing operator $(1 - B)^d$ expanded via binomial series.
  * Bounded memory weight series truncation with tolerance parameter $|\omega_k| < \epsilon$ ($\epsilon \le 10^{-4}$).
  * Automated optimal degree $d^*$ search using Augmented Dickey-Fuller (ADF) stationarity testing while maximizing correlation with the original price series.
* **Acceptance Criteria:** Unit tests verifying $d=0$ identity, $d=1$ standard returns, bounded weight decay, and ADF stationarity convergence ($p < 0.01$).

#### Step 3: Dynamic Volatility Triple-Barrier Labeling [PLANNED]
* **Scope:**
  * Path-dependent upper horizontal (profit-taking), lower horizontal (stop-loss), and vertical (expiration) barrier evaluation.
  * Dynamic horizontal threshold scaling parameterized by instantaneous realized volatility: $pt_t = c_1 \sigma_t$, $sl_t = c_2 \sigma_t$.
  * Explicit classification of un-hit expiration windows.
* **Acceptance Criteria:** Property tests ensuring timestamps obey $t_{\text{touch}} \le t_{\text{expiration}}$; zero lookahead leakage.

#### Step 4: Combinatorial Purged Cross-Validation (CPCV) [PLANNED]
* **Scope:**
  * Chronological partitioning into $N$ blocks generating $\binom{N}{k}$ combinatorial backtest splits.
  * Temporal boundary purging to remove overlapping event information horizons.
  * Post-test embargo periods to eliminate autoregressive serial correlation leakage.
* **Acceptance Criteria:** Vectorized split generator producing empirical distributions of backtest paths; memory footprint bounded $< 2\text{GB}$.

#### Step 5: Two-Stage Meta-Labeling Architecture [PLANNED]
* **Scope:**
  * Decoupling primary directional model ($\hat{y}_t \in \{-1, 1\}$) from secondary bet-sizing classifier ($z_t \in \{0, 1\}$).
  * Secondary model trained on feature vectors to predict probability of hitting profit barrier before stop-loss.
  * Probability calibration (Brier score verification) parameterizing bet size.
* **Acceptance Criteria:** Demonstrable increase in out-of-sample Sharpe ratio relative to raw directional heuristic.

#### Step 6: Deflated Sharpe Ratio (DSR) & Statistical Significance [PLANNED]
* **Scope:**
  * Adjustment of empirical Sharpe ratio for skewness $\hat{\gamma}_3$, kurtosis $\hat{\gamma}_4$, sample length $T$, trial count $K$, and variance of trials $V[\{SR_k\}]$.
  * Calculation of Expected Maximum Sharpe Ratio.
  * Hard strategy rejection threshold ($DSR < 0.95$).
* **Acceptance Criteria:** Verified against published benchmark datasets; integration with CPCV trial outputs.

---

### Phase 3: Scenario Matrix & Game Theory Engine [PLANNED]
* **Scope:**
  * Bayesian game formulation of market price formation against Nature/Counterparties.
  * Adversarial market response regimes:
    * $\mathbf{s}_1$: Immediate Absorption / Efficient Reversal
    * $\mathbf{s}_2$: Momentum Cascade / Stop-Run Continuation
    * $\mathbf{s}_3$: Adversarial Liquidity Trap / Squeeze
  * Minimax Regret payoff matrix optimization incorporating square-root market impact and crowding penalties.

---

### Phase 4: Evolutionary Population & Hypergamy Dynamics [PLANNED]
* **Scope:**
  * Algorithmic genotype chromosome encoding ($\mathbf{g}_{\text{repr}}, \mathbf{g}_{\text{game}}, \mathbf{g}_{\text{infer}}, \mathbf{g}_{\text{risk}}$).
  * Multi-objective fitness function combining Deflated Sharpe, Drawdown penalty, Minimax Regret, and Novelty distance.
  * Population stratification into Alpha ($20\%$) and Aspirant ($80\%$) cohorts.
  * Hypergamic assortative mating gated by residual orthogonality threshold ($\text{Corr}(\mathbf{e}_{\text{Alpha}}, \mathbf{e}_{\text{Aspirant}}) < \delta_{\text{ortho}}$).
  * Entropy-governed adaptive mutation rates scaled by realized volatility.

---

### Phase 5: Ensemble Aggregator & Risk Overlays [PLANNED]
* **Scope:**
  * Regime-conditioned Bayesian Model Averaging (BMA) weighting.
  * Predictive return distribution output (mean, variance, quantiles).
  * Disagreement entropy circuit breaker automatically dialing down gross exposure during conflicting model regimes.
  * CUSUM structural break detection for emergency trading halts.
