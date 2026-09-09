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
    Step 2: Fractional Differentiation       :done, s2_2, 2026-09-09, 1d
    Step 3: Triple-Barrier Labeling         :done, s2_3, 2026-09-10, 1d
    Step 4: Combinatorial Purged CV (CPCV)  :done, s2_4, 2026-09-11, 2d
    Step 5: Two-Stage Meta-Labeling         :done, s2_5, 2026-09-12, 2d
    Step 6: Deflated Sharpe Ratio (DSR)     :done, s2_6, 2026-09-13, 2d
    section Phase 3: Game Theory
    Scenario Matrix & Adversarial Payoffs   :active, p3, 2026-09-22, 5d
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

### Phase 2: Econometric Rig & Feature Engineering [COMPLETE]

To eliminate cognitive overload and merge conflicts, Phase 2 is decomposed into 6 sequential micro-steps:

#### Step 1: Market Data Entity & Columnar Storage Subsystem [COMPLETE]
* **Deliverables:** Immutable `PriceBar` value object with defensive boundary invariants; `MarketDataBatch` with zero-copy PyArrow table export; embedded DuckDB columnar storage (`DuckDBManager`, `DuckDBMarketDataRepository`); `MarketDataService` with rolling realized volatility ($\sigma_t$); `/api/v1/market-data` REST endpoints; 23 tests (53 total) passing with **88.79% coverage**.

#### Step 2: Fractional Differentiation Engine [COMPLETE]
* **Deliverables:**
  * Fixed-Width Window Fractional Differentiation (FFD) engine via recursive binomial expansion $(1 - B)^d$.
  * Bounded memory weight series truncation ($|\omega_k| < \epsilon = 10^{-4}$) with zero-sum DC-offset correction ($\sum \omega = 0$).
  * Automated optimal degree $d^*$ bisection search minimizing ADF test evaluations while preserving historical memory.
  * $O(T \log l^*)$ 1D FFT causal convolution via `scipy.signal.fftconvolve` with $O(T)$ bounded memory.
  * Analytical recursive reconstruction operator (`inverse_transform`) for price recovery.
  * Real-time `StreamingFracDiffBuffer` with DuckDB repository pre-warming (`hydrate_from_repository`) and sub-millisecond updates.
  * 20 unit tests (73 total) passing with **89.63% coverage** and zero warnings.

#### Step 3: Dynamic Volatility Triple-Barrier Labeling [COMPLETE]
* **Deliverables:**
  * Causal Parkinson range volatility estimator ($\sigma_t$) lagged by 1 bar ($t-1$) with zero lookahead bias.
  * Geometric log-price space horizontal barrier evaluation preserving bilateral random-walk symmetry.
  * Asymmetrical Long, Short, and Unsigned barrier execution mapping.
  * Defensive bounds ($\sigma_{\text{floor}}, \sigma_{\text{cap}}$) preventing zero-volatility collapse and runaway barrier blowout.
  * Opening price gap fill honoring discontinuous auction open levels.
  * Conservative stop-loss collision policy on dual intra-bar breaches (`pessimistic_collision=True`).
  * Net return deduction of round-trip bid-ask spread and exchange fee friction.
  * 24 unit tests (97 total) passing with **90.20% coverage** and zero warnings.

#### Step 4: Combinatorial Purged Cross-Validation (CPCV) [COMPLETE]
* **Deliverables:**
  * Combinatorial partitioning into $N$ contiguous chronological blocks generating $\binom{N}{k}$ out-of-sample backtest folds.
  * Exact interval intersection purging removing training trades whose lifespan $[t_{\text{entry}}, t_{\text{exit}}]$ overlaps test intervals.
  * Post-test autoregressive embargoing ($h_{\text{embargo}}$) neutralizing serial correlation leakage.
  * Deterministic budget bounding (`max_splits`) preventing factorial compute explosion during evolutionary search.
  * Defensive starvation guard (`min_train_ratio`) rejecting over-purged folds.
  * Forward-chaining mode strictly enforcing past-to-future temporal causality.
  * Continuous backtest path reconstruction ($\phi = \binom{N-1}{k-1}$ paths) with greedy positional fold assignment.
  * Path Sharpe ratio distribution and empirical variance $V[\{SR_k\}]$ evaluation feeding directly into Step 6 (Deflated Sharpe Ratio).
  * 13 unit tests (110 total) passing with **90.98% coverage** and zero warnings.

#### Step 5: Two-Stage Meta-Labeling Architecture [COMPLETE]
* **Deliverables:**
  * Two-stage Continuous-Payoff Kelly Meta-Labeling engine decoupling directional discovery from capital allocation.
  * Payoff-aware meta-labeling ($\pi_t = \hat{y}_t \cdot R_t^{\text{net}}$) correctly crediting net-profitable vertical timeouts and penalizing fee-eroded trades.
  * Regularized Platt logistic calibration (Platt scaling) mapping decision margins to smooth, monotonic probabilities with mandatory Brier score validation gate.
  * Time-Decayed Fractional Kelly Criterion ($f^* = \lambda \cdot \frac{p \cdot b - (1-p)}{b}$) maximizing long-term compound capital growth.
  * Holding duration discounting ($\sqrt{\tau_t / \tau_{\text{ref}}}$) pricing capital opportunity costs.
  * Portfolio-level concurrency throttling ($c_t$) normalizing overlapping trade allocations to enforce an aggregate $100\%$ leverage limit ($L_{\text{max}} = 1.0$).
  * 8 unit tests (118 total) passing with **91.40% coverage** and zero warnings.

#### Step 6: Deflated Sharpe Ratio (DSR) & Statistical Significance [COMPLETE]
* **Deliverables:**
  * Robust higher-moment estimation (`compute_moments`) with two-sided winsorization and Pearson bound clamp $\hat{\gamma}_4 \ge 1 + \hat{\gamma}_3^2$.
  * Probabilistic Sharpe Ratio (`compute_probabilistic_sharpe_ratio`) adjusting for skewness and excess kurtosis.
  * Extreme Value Theory selection bias hurdle (`compute_expected_max_sharpe`) with $K=1$ probit singularity guard and Euler-Mascheroni approximation.
  * Piecewise Minimum Backtest Length (`compute_min_backtest_length`) strictly evaluating to $+\infty$ for losing strategies.
  * Spectral Frobenius trace participation ratio (`compute_effective_trials`) calculating effective independent trials $K_{\text{eff}} \le K$.
  * False Discovery Rate controls (`adjust_p_values_fdr`) with Benjamini-Hochberg and Benjamini-Yekutieli algorithms.
  * Master `DeflatedSharpeEngine` with dual institutional gate: $(\text{DSR} \ge 0.95) \land (T \ge \text{MinBTL})$.
  * Direct integration with Step 4 CPCV paths (`evaluate_cpcv_results`) and population cohort screening (`evaluate_cohort`).
  * 10 unit tests (128 total) passing with **91.75% overall coverage** (95% on `deflated_sharpe.py`) and zero warnings.

---

### Phase 3: Scenario Matrix & Game Theory Engine [ACTIVE]

Decomposed into 4 sequential micro-steps implementing the Entropic Distributionally Robust Stackelberg Engine:

#### Step 1: Causal Bayesian Jump-Regime Estimator & Ambiguity Scaling [COMPLETE]
* **Deliverables:**
  * Online causal Bayesian filter tracking regime probabilities $\boldsymbol{\pi}_t = [P(\text{Absorption}), P(\text{Momentum}), P(\text{Panic})]^T$ with simplex sum enforcement ($\sum \pi_j = 1$).
  * Two-sided Cumulative Sum (CUSUM) shock detector ($S_t^+, S_t^-$) with dwell time hysteresis ($\tau_{\text{dwell}} = 3$) preventing regime whip-sawing while instantly triggering emergency panic priors on violent sell-offs.
  * Oracle Approximating Shrinkage (OAS) covariance estimator with spectral projection enforcing positive definiteness ($\lambda_{\min} \ge 10^{-5}$) under collinear or starved sample windows.
  * Fournier-Guillin concentration bound calibrating thermodynamic ambiguity temperature $\beta_t \in [\beta_{\min}, \beta_{\max}]$.
  * 21 unit tests (149 total) passing with **92.38% overall coverage** (99% on `regimes.py`) and zero warnings.

#### Step 2: Microstructure Propagator & Kyle-Obizhaeva Impact [COMPLETE]
* **Deliverables:**
  * Huberman-Stanzl Arbitrage-Free Cross-Impact Constructor (`HubermanStanzlCrossImpact`) with symmetric sandwich tensor $\mathbf{\Lambda}_{\text{cross}} \succ 0$ linking asset adverse selection to Phase 2 Step 5's Kelly meta-label $z_t$.
  * 3/2-power Generalized Pseudo-Huber potential (`GeneralizedPseudoHuber`) modeling the exact universal Square-Root Law of price impact ($\psi'_{3/2}(u) \sim \sqrt{u}$) with guaranteed strict convexity everywhere.
  * Continuous Bayesian panic asymmetry gate with exact zero gradient at rest ($\nabla \mathcal{C}(\mathbf{0}) = \mathbf{0}$) eliminating artificial drift.
  * State-space bounded order book depletion buffer with hyperbolic tangent saturation ($\mathbf{B}_t \le B_{\max}$).
  * 19 unit tests (168 total) passing with **92.40% overall coverage** (93% on `market_impact.py`) and zero warnings.

#### Step 3: Stackelberg Leader-Follower Trajectory [UPCOMING]
* **Scope:**
  * Discrete Monotone Hyperbolic Propagator scheduling multi-period execution slices ($\sum_{k=1}^H \alpha_k = 1, \alpha_k > 0$).
  * Continuous quadratic-bilinear Stackelberg payoff functional accounting for market maker quote shading.
  * Friction-consistent institutional benchmark universe (Equal Weight, Risk Parity, Inverse Volatility, Cash).

#### Step 4: Closed-Form Boltzmann Dual & Entropic Minimax Regret Solver [UPCOMING]
* **Scope:**
  * Max-shifted Log-Sum-Exp Boltzmann dual potential.
  * Bounded Latent Softmax parameterization enforcing simplex feasibility ($0 \le a_i \le w_{\max}$, $\sum a_i \le 1$).
  * Sub-50 microsecond pure NumPy vectorized Newton-Raphson solver.
  * Certified worst-case regret metric $V_i(\text{Regret})$ feeding directly into Phase 4 Genetic Algorithm fitness.

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
