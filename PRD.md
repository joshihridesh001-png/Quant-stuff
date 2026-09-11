# Product Requirements Document & Analysis (PRDA)

## 1. Executive Summary & Product Vision
The **News-Driven Quantitative Prediction Engine** is an institutional-grade algorithmic alpha generation and market prediction platform. It bridges unstructured real-time financial news sentiment with high-frequency microstructural price series using Bayesian game theory and evolutionary computing. 

By modeling market price formation not as passive stochastic drift, but as a non-cooperative game between competing participant classes, the engine continuously adapts to shifting macro regimes. To eliminate phenotypic cloning and premature convergence inherent in traditional genetic algorithms, strategy evolution is governed by **algorithmic hypergamy dynamics**—an asymmetric selection protocol enforcing genetic innovation and orthogonal residual errors.

---

## 2. Problem Statement & Structural Market Inefficiencies

In financial markets, the empirical signal-to-noise ratio (SNR) routinely hovers below $0.05$. Traditional machine learning applications in quantitative finance systematically fail due to three structural flaws:
1. **Destructive Differencing**: Standard integer differencing ($d=1$, simple percentage returns) achieves stationarity at the cost of obliterating multi-period memory, destroying structural support/resistance and mean-reverting equilibrium signals.
2. **Path-Independent Target Misformulation**: Fixed-horizon return labels ($R_{t, t+k}$) ignore the path-dependent reality of execution, where trades are terminated prematurely by stop-loss or take-profit thresholds.
3. **Non-IID Cross-Validation Leakage**: Standard $k$-fold cross-validation allows future information and serial correlation to leak into past folds, yielding backtests that overstate real-world performance by orders of magnitude.
4. **Genetic Homogenization**: Standard evolutionary optimization clones top performers during prolonged single regimes, resulting in catastrophic drawdowns when macro regimes shift.

---

## 3. Product Scope & Core Capabilities

```mermaid
graph TD
    A[Unstructured News Feed] -->|Continuous Text| B[News Ingestion Engine]
    C[Exchange Tick/Bar Feeds] -->|Columnar Data| D[Market Data Subsystem]
    B --> E[Causal Event Graph & Decay Engine]
    D --> F[Econometric Rig & Feature Transforms]
    E --> G[Game-Theoretic Scenario Generator]
    F --> G
    G --> H[Evolutionary Population Manager]
    H -->|Pareto Elite Cohort| I[Dynamic Ensemble Aggregator]
    I --> J[Execution Alpha Engine]
```

### 3.1 Feature Requirements Matrix

| Component | Scope & Core Deliverables | Priority |
| :--- | :--- | :--- |
| **Market Data Subsystem** | High-throughput columnar ingestion; immutable `PriceBar` with boundary invariants; embedded DuckDB/PyArrow storage; zero-copy 1D array views (`MarketDataBatch`); rolling realized volatility ($\sigma_t$). | P0 (Complete) |
| **News Ingestion Layer** | Asynchronous batch ingestion; dense transformer vectorization; tri-axial sentiment scoring (polarity, subjectivity, novelty); causal DAG; hybrid dual-decay kernel ($\tau_{\text{fast}}, \tau_{\text{slow}}$). | P0 (Complete) |
| **Econometric Rig** | Memory-preserving fractional differentiation $(1-B)^d$; dynamic volatility Triple-Barrier labeling; Two-stage Meta-Labeling architecture. | P0 (Sprint 2) |
| **Validation Framework** | Combinatorial Purged Cross-Validation (CPCV) with boundary purging and post-test embargoing; Deflated Sharpe Ratio (DSR) controlling for non-normality and selection bias. | P0 (Sprint 2) |
| **Scenario Matrix (Game Theory)** | Bayesian game formulation against Nature/Counterparties; adversarial scenarios (Immediate Reversal, Momentum Cascade, Liquidity Squeeze); Minimax Regret payoff optimization. | P1 (Sprint 3) |
| **Evolutionary Manager** | Modular genotype chromosomes ($\mathbf{g}_{\text{repr}}, \mathbf{g}_{\text{game}}, \mathbf{g}_{\text{infer}}, \mathbf{g}_{\text{risk}}$); Boundary-Anchored Adaptive RVEA Pareto sorting with SVD subspace orthogonality ($\rho_{\text{ortho}}$) and Memmel–Ledoit–Wolf dependent dominance; hypergamic assortative mating gated by residual orthogonality. | P1 (Sprint 4) |
| **Ensemble Aggregator** | Regime-conditioned Bayesian model averaging; predictive variance estimation; disagreement entropy circuit breaker. | P1 (Sprint 5) |

---

## 4. User Personas & System Actors

* **Quantitative Researcher**: Designs feature hypotheses, runs non-IID cross-validation sweeps, evaluates Deflated Sharpe metrics, and configures evolutionary mutation rates.
* **Algorithmic Execution Engine**: Consumes real-time directional distributions, calibrated sizing probabilities, and regime-shift alerts to route orders.
* **Automated Data Feed / Ingestion Pipeline**: Ingests high-frequency tick/bar feeds and unstructured news headlines via authenticated machine-to-machine REST/gRPC endpoints.

---

## 5. Non-Functional Requirements (NFRs) & Engineering Invariants

1. **Deterministic Latency & Throughput**:
   * Market bar ingestion: $> 100,000$ bars/second via zero-copy PyArrow tables into DuckDB.
   * Analytical historical bar retrieval: Sub-15ms for $1,000,000$ contiguous bars.
   * News decay active state vector computation: $< 5\text{ms}$ per asset.
2. **Zero-Execution Maintainability**:
   * 100% of potential operational failures must be cataloged with exact code coordinates, symptoms, root causes, and static verification checks (`Rules.md`, `Memory.md`).
3. **Rigorous Quality Rig**:
   * Minimum automated test coverage $\ge 85\%$ across all modules.
   * Strict static type checking (`mypy --strict`) with zero errors.
   * Zero linting or formatting deviations (`ruff`).
