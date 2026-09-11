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

### Phase 3: Scenario Matrix & Game Theory Engine [COMPLETE]

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

#### Step 3: Stackelberg Leader-Follower Trajectory [COMPLETE]
* **Deliverables:**
  * Discrete Monotone Hyperbolic Propagator (`DiscreteHyperbolicPropagator`) scheduling multi-period execution slices with exact partition of unity ($\sum_{k=1}^H \alpha_k = 1.0$) and overflow-free exponential formulation for all $\kappa \in (0, \infty)$ and $H \ge 1$.
  * Continuous quadratic-bilinear Stackelberg payoff functional (`StackelbergPayoffEngine`) incorporating market maker predatory quote shading ($\mathbf{M}_{\text{pred}} = \theta_{\text{pred}} \mathbf{\Sigma}_j$), exact analytical gradient $\nabla_{\mathbf{a}} U$, and certified negative-definite Hessian $\mathbf{H}_{\mathbf{a}} U \prec 0$.
  * Institutional Benchmark Universe (`InstitutionalBenchmarkUniverse`) providing Equal Weight, Risk Parity, Inverse Volatility, and Cash benchmarks with single-asset concentration caps ($b_i \le w_{\max}$) and Friction Parity.
  * Payoff Tensor and Non-Negative Regret Matrix Constructor (`StackelbergPayoffTensorConstructor`) producing non-negative regret $R_{i, j} \ge 0$ across market regimes.
  * 70 unit tests (238 total) passing with **93.03% overall coverage** (99% on `payoff_matrix.py`) and zero warnings.

#### Step 4: Closed-Form Boltzmann Dual & Entropic Minimax Regret Solver [COMPLETE]
* **Deliverables:**
  * Max-shifted Log-Sum-Exp Boltzmann dual potential evaluator (`EntropicBoltzmannPotential`) with certified overflow immunity and thermally tilted worst-case distributions $\mathbf{q}^* \in \Delta^M$.
  * Bounded Latent Space Transformation Engine (`LatentSoftmaxTransform`) mapping $[0, w_{\max}]^N$ bijectively to unconstrained coordinates with $O(N)$ diagonal Jacobian.
  * Sub-millisecond Vectorized Damped Newton-Raphson Solver (`VectorizedNewtonSolver`) with Levenberg-Marquardt regularization, Armijo backtracking, positive-definite Fisher information Hessian, cross-impact caching, and Karush-Kuhn-Tucker (KKT) projected gradient termination.
  * Certified worst-case regret metric $\Psi(\mathbf{a}^*)$ and multi-bar discrete hyperbolic execution trajectories feeding directly into Phase 4 Genetic Algorithm chromosome fitness.
  * 21 unit tests (259 total) passing with **93.38% overall coverage** (96% on `minimax_regret.py`) and zero warnings.

---

### Phase 4: Evolutionary Population & Hypergamy Dynamics [ACTIVE]

#### Step 1: Chromosome Architecture & Vector Encoding Engine [COMPLETE]
* **Deliverables:**
  * Declarative Gene Registry (`GENE_REGISTRY`) specifying 20 algorithmic parameters across Representation, Game Theory, Inference, and Risk blocks with scale types (`ScaleType`).
  * Continuous Unit Hypercube Codec (`ChromosomeVectorCodec`) implementing scale-invariant hybrid logarithmic-linear normalization.
  * Invariant satisfaction by construction: timescale ordering ($\tau_{\text{fast}} < \tau_{\text{slow}}$) via ratio parameterization and regime scenario simplex ($\sum p_j \equiv 1.0, p_j > 0$) via unconstrained softmax logits with zero-mean gauge fixing.
  * Bucket midpoint centering $u(k) = (k + 0.5) / K$ eliminating floating-point round-trip drift in discrete parameter quantization.
  * Gauge-invariant phenotypic distance metric operating on decoded regime probabilities rather than raw logits.
  * Typed factory adapters (`to_stackelberg_config`, `to_regime_config`, `to_minimax_config`, `to_triple_barrier_config`, `to_meta_label_config`) and backward-compatible dictionary serialization.
  * 32 unit tests (291 total) passing with **93.71% overall coverage** (98% on `chromosomes.py`) and zero warnings.

#### Step 2: Novelty Distance & Multi-Objective Pareto Sorting [COMPLETE]
* **Deliverables:**
  * Boundary-Anchored Adaptive RVEA with SVD Subspace Orthogonality & Memmel–Ledoit–Wolf Dependent Dominance (BA-ARVEA-SO) architecture.
  * Domain entities `CandidateFitness`, `ParetoFront`, and `RankingResult` enforcing strict defensive invariants (`INV-PAR-001` to `INV-PAR-006`).
  * `SVDSubspaceOrthogonalArchive` evaluating true orthogonal novelty via thin SVD basis projection ($1 - R^2$), eliminating multi-collinear clones and novelty parasites.
  * `AdaptiveReferenceLattice` generating Das-Dennis $K=28$ reference rays ($M=3, p=6$) with immutable basis coordinate anchors $[1,0,0], [0,1,0], [0,0,1]$ (`INV-PAR-004`), dynamic interior ray migration with minimum angular separation ($\theta \ge 0.10\text{ rad}$), and generation-escalated Angle-Penalized Distance (APD).
  * `DependentNonDominatedSorter` computing the exact Memmel–Ledoit–Wolf asymptotic covariance for strategy differences under return correlation $\rho$, eliminating 300% variance inflation of the independence fallacy.
  * Deb's Feasibility Rule and Efficient Non-dominated Sorting with Sequential Search (ENS-SS) partitioning populations into non-dominated fronts.
  * Unified facade `BoundaryAnchoredRVEARanker` with auto-admission of Front-1 elites into the SVD archive.
  * 29 unit tests (320 total project tests) passing with **99% line coverage** on `pareto_sorting.py` and benchmark latency $< 15\text{ms}$.

#### Step 3: Hypergamic Assortative Selection & Residual Orthogonality Gating [COMPLETE]
* **Deliverables:**
  * Front-preserving Pareto cohort stratifier (`ParetoCohortStratifier`) partitioning candidate populations into elite Alphas $\mathcal{A}$ and exploratory Aspirants $\mathcal{X}$, preserving Front 1 without truncation and padding from Front 2 (ordered by APD) while strictly excluding infeasible individuals (`INV-HYP-004`).
  * Bidirectional absolute residual orthogonality gate (`ResidualOrthogonalityGate`) evaluating $1 - |\rho(e_A, e_B)| \ge \delta_{\text{current}}$, eliminating both direct clones and inverse clones ($\rho = -0.95$) with defensive zero-variance residual guards.
  * Bounded tournament partner matcher (`HypergamicPartnerMatcher`) with adaptive threshold relaxation ($\gamma_{\text{relax}}^k$) and guaranteed zero-deadlock fallback via maximum Euclidean phenotypic distance in unit hypercube space (`INV-HYP-005`).
  * Asymmetric Latent Unit-Hypercube Crossover (`AsymmetricLatentCrossover`) executing Simulated Binary Crossover (SBX) in scale-free continuous space $\mathbf{u} \in [0, 1]^{20}$ with gene-family role-biased inheritance ($P_\alpha = 0.75$ for risk/game; $P_{\text{asp}} = 0.65$ for repr/infer), guaranteeing ordering ($\tau_{\text{fast}} < \tau_{\text{slow}}$) and simplex ($\sum p_j \equiv 1.0$) by construction (`INV-HYP-002`).
  * Master facade `HypergamicSelectionEngine` coordinating cohort stratification, partner matching, crossover, and monotonic Front-1 elitism preservation (`INV-HYP-003`), strictly guaranteeing $N_{\text{offspring}} == N_{\text{target}}$ (`INV-HYP-001`).
  * Exported all domain entities, configuration, and facades in `src/quant/analytics/__init__.py`.
  * 20 unit tests (340 total project tests) passing with **95% line coverage** on `hypergamic_selection.py` and benchmark reproduction latency well under 25ms.

#### Step 4: Adaptive Volatility Mutation & Generational Lifecycle Engine [PLANNED]

---

### Phase 5: Ensemble Aggregator & Risk Overlays [PLANNED]
* **Scope:**
  * Regime-conditioned Bayesian Model Averaging (BMA) weighting.
  * Predictive return distribution output (mean, variance, quantiles).
  * Disagreement entropy circuit breaker automatically dialing down gross exposure during conflicting model regimes.
  * CUSUM structural break detection for emergency trading halts.
