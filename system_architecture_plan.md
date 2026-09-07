# News-Driven Market Prediction Engine: System Architecture Plan

## Executive Summary
This document outlines the architectural blueprint for a multi-algorithmic, news-driven market prediction engine. The system continuously ingests unstructured financial news streams, constructs time-decayed causal event graphs, stress-tests candidate prediction strategies against adversarial market scenarios using Bayesian game-theoretic payoff matrices, and dynamically evolves the strategy population using a natural-selection framework enhanced with **algorithmic hypergamy dynamics** to preserve genetic diversity and prevent regime-specific overfitting.

---

## 1. System Architecture Overview

```mermaid
graph TD
    A[Unstructured News Stream] --> B[News Ingestion Layer]
    B -->|Temporal Event Graph & Decayed Vectors| C[Game-Theoretic Scenario Generator]
    C -->|Regime-Conditioned Payoff Surfaces| D[Evolutionary Population Manager]
    D -->|Pareto-Optimal Elite Cohort| E[Ensemble Prediction Aggregator]
    E -->|Distributional Forecast & Confidence| F[Execution Engine / Risk Sizing]
    F -.->|Realized PnL & Residual Feedback| D
    F -.->|Regime Realization Feedback| C
```

The system is organized into four core modular layers:
1. **News Ingestion Layer**: Semantic parsing, continuous vectorization, and temporal decay chaining.
2. **Game-Theoretic Scenario Generator**: Multi-scenario adversarial simulation and payoff matrix computation.
3. **Evolutionary Population Manager**: Genotype encoding, adaptive mutation, and hypergamous mate selection.
4. **Ensemble Prediction Aggregator**: Regime-weighted Bayesian aggregation and disagreement entropy monitoring.

---

## 2. Component Specifications

### 2.1 Component 1: News Ingestion Layer & Temporal Chaining

#### Event Representation & Projected Subspace Balancing
Incoming unstructured news items are converted into composite feature vectors. To prevent high-dimensional dense embeddings ($d_e = 768$) from swamping low-dimensional scalar signals in downstream Euclidean or metric distances, dense embeddings are projected into an energy-normalized subspace $\mathbb{R}^{d_p}$ ($d_p = 16$) prior to fusion:

$$\mathbf{e}_i = \Big[ \mathcal{P}(\mathbf{v}_{\text{dense}}) \;\|\; \mathbf{s}_{\text{sentiment}} \;\|\; \mathbf{u}_{\text{urgency}} \;\|\; \mathbf{c}_{\text{centrality}} \Big] \in \mathbb{R}^{d_p + 5}$$

- $\mathcal{P}(\mathbf{v}_{\text{dense}}) \in \mathbb{R}^{d_p}$: Subspace-projected and $L_2$-normalized dense Transformer embedding (FinBERT).
- $\mathbf{s}_{\text{sentiment}} \in [-1, 1]^3$: Tri-axial sentiment scores (polarity, subjectivity, novelty).
- $\mathbf{u}_{\text{urgency}} \in [0, 1]$: Transmission velocity / event urgency prior.
- $\mathbf{c}_{\text{centrality}} \in [0, 1]^K$: Relevance mapping across target asset universes and sectors.

#### Temporal Chaining & Truncated Memory Kernel
Events are stored in an append-only causal Directed Acyclic Graph (DAG). The active news state for asset $k$ at time $t$ is governed by a hybrid decay kernel with explicit numerical truncation tolerance ($\epsilon \le 10^{-4}$) and maximum lookback horizon ($T_{\text{max}} = 7\text{ days}$) to guarantee $O(T)$ bounded execution:

$$\mathbf{S}_{\text{news}}^{(k)}(t) = \sum_{i \in \mathcal{V}_t, \Delta t_i \le T_{\text{max}}} c_{i,k} \cdot \mathbf{e}_i \cdot \kappa(t - t_i, \mathbf{u}_i)$$

$$\kappa(\Delta t, u) = \begin{cases} 
\alpha \exp\left(-\frac{\Delta t}{\tau_{\text{fast}} \cdot (1 - u)}\right) + (1 - \alpha)\left(1 + \frac{\Delta t}{\tau_{\text{slow}}}\right)^{-\beta}, & \text{if } \kappa \ge \epsilon \text{ and } \Delta t \le T_{\text{max}} \\
0, & \text{otherwise}
\end{cases}$$

- **Fast decay ($\tau_{\text{fast}}$)**: Models rapid sentiment absorption and high-frequency noise dissipation.
- **Slow decay ($\tau_{\text{slow}}$)**: Models long-term thematic trends and structural shifts.

#### Interface Contract (Asynchronous & Batch-Capable)
```
Interface: INewsIngestionEngine
    Methods:
        - async IngestBatch(events: List[RawDocument]) -> List[EventNode]
        - async IngestDocument(doc: RawDocument) -> EventNode
        - async GetStateVectors(assets: List[AssetID], as_of: Timestamp) -> Map[AssetID, Vector[Real]]
```

---

### 2.2 Component 2: Game-Theoretic Scenario Generator

#### Justification
Financial markets are non-cooperative, imperfect-information games. Simple historical regression overfits because market prices represent the equilibrium outcome of competing participant classes (institutional flows, fast arbitrageurs, and retail momentum).

#### Scenario Payoff Formulation
Candidate algorithms act as **Player 1** (Predictor), choosing an action $\mathbf{a}_1 \in [-1, 1]^K$ against **Player 2** (Market / Nature Scenarios $\mathbf{s}_j \in \mathcal{S}$):
1. $\mathbf{s}_1$ (**Immediate Reversal**): Immediate sentiment pricing followed by market-maker mean reversion.
2. $\mathbf{s}_2$ (**Momentum Cascade**): Trend continuation driven by stop runs and herd positioning.
3. $\mathbf{s}_3$ (**Liquidity Trap / Squeeze**): Adversarial distribution where institutional liquidity absorbs the sentiment herd.

The scenario payoff utility $U(i, \mathbf{s}_j)$ incorporates non-linear execution costs and crowding penalties:

$$U(i, \mathbf{s}_j) = \mathbf{a}_1^T \mathbb{E}[\mathbf{r} \mid \mathbf{s}_j] - \lambda \cdot \mathbf{a}_1^T \mathbf{\Sigma}(\mathbf{s}_j) \mathbf{a}_1 - \mathcal{C}(\mathbf{a}_1, \mathbf{a}_1^{\text{prior}}) - \Omega_{\text{crowding}}(\mathbf{a}_1, \mathbf{s}_j)$$

Evaluation across scenarios uses **Minimax Regret**:

$$V_i = -\max_{\mathbf{s}_j \in \mathcal{S}} \left( \max_{\mathbf{a}^*} U(\mathbf{a}^*, \mathbf{s}_j) - U(\mathbf{a}_i, \mathbf{s}_j) \right)$$

#### Interface Contract
```
Interface: IScenarioGenerator
    Methods:
        - GenerateScenarios(news_state: Vector[Real], depth: OrderBookSnapshot) -> List[ScenarioProfile]
        - EvaluatePayoffs(action: ActionVector, scenarios: List[ScenarioProfile]) -> Map[ScenarioID, Float]
```

---

### 2.3 Component 3: Evolutionary Population Manager & Hypergamy Dynamics

#### Genotype Data Structure
Each algorithmic individual is encoded as a chromosome $\mathbf{g}_k = \langle \mathbf{g}_{\text{repr}}, \mathbf{g}_{\text{game}}, \mathbf{g}_{\text{infer}}, \mathbf{g}_{\text{risk}} \rangle$:
- $\mathbf{g}_{\text{repr}}$: Parameters for temporal decay rates ($\tau_{\text{fast}}, \tau_{\text{slow}}, \alpha$) and feature projection weights.
- $\mathbf{g}_{\text{game}}$: Scenario belief priors $\mathbf{p}_{\text{belief}}$ and risk/regret aversion parameters $\lambda, \gamma$.
- $\mathbf{g}_{\text{infer}}$: Inference hyperparameters (model depths, thresholds, sensitivity constants).
- $\mathbf{g}_{\text{risk}}$: Volatility-targeting scalar, maximum drawdown thresholds, and leverage limits.

#### Hypergamic Mate Selection Logic
- **Problem**: Standard fitness-proportionate selection causes premature convergence and phenotypic cloning during dominant market regimes.
- **Mechanism**: The population is divided into the **Alpha Cohort** (top $\rho = 20\%$ by multi-objective fitness) and the **Aspirant Cohort** (remaining $80\%$).
- **Rule**: Lower-tier algorithms actively seek to mate with Alpha models. However, an Alpha individual **rejects** crossover unless the candidate's historical prediction errors demonstrate residual orthogonality:

$$\text{Corr}(\mathbf{e}_{\text{Alpha}}, \mathbf{e}_{\text{Aspirant}}) < \delta_{\text{ortho}}$$

This preserves elite alpha characteristics while mathematically guaranteeing genetic and behavioral diversity.

#### Multi-Objective Optimization & Pareto Sorting (NSGA-II)
Rather than collapsing competing objectives into an arbitrary scalar sum that collapses the trade-off frontier, the evolutionary population is evaluated across a 4-dimensional objective vector:

$$\mathbf{F}(\mathcal{I}_i) = \Big[ \text{DeflatedSharpe}(\mathcal{I}_i), \; -\text{MaxDD}(\mathcal{I}_i), \; V_i(\text{Regret}), \; \mathcal{H}_{\text{novelty}}(\mathcal{I}_i) \Big]^T$$

- Individual $\mathcal{I}_a$ **Pareto-dominates** $\mathcal{I}_b$ ($\mathcal{I}_a \succ \mathcal{I}_b$) iff $\forall j, F_j(\mathcal{I}_a) \ge F_j(\mathcal{I}_b)$ and $\exists j, F_j(\mathcal{I}_a) > F_j(\mathcal{I}_b)$.
- Non-dominated sorting partitions the population into Pareto fronts $\mathcal{F}_0, \mathcal{F}_1, \dots$, where $\mathcal{F}_0$ comprises the elite non-dominated trade-off frontier.
- **Crowding Distance** metric is maintained within fronts to prioritize boundary and niche solutions, preserving diversity across the Pareto surface.
- For backward-compatible scalar evaluation, the scalarized fitness index remains supported:
  $$\mathcal{F}_{\text{scalar}}(\mathcal{I}_i) = \text{DeflatedSharpe}(\mathcal{I}_i) \cdot \exp\left( -\psi \cdot \text{MaxDD}(\mathcal{I}_i) \right) + \omega_1 \cdot V_i(\text{Regret}) + \omega_2 \cdot \mathcal{H}_{\text{novelty}}(\mathcal{I}_i)$$

#### Novelty-Governed Aspirant Selection
When an aspirant is rejected by an Alpha individual due to insufficient orthogonality ($\text{Corr} \ge \delta_{\text{ortho}}$), the fallback mechanism utilizes **Novelty Search** rather than random drift: the aspirant that maximizes behavioral parameter distance from the elite cohort is prioritized for exploratory crossover.

#### Adaptive Mutation
Mutation probability $\mu(t)$ scales dynamically with market regime volatility and news entropy:

$$\mu(t) = \mu_0 \cdot \left(1 + \tanh\left(\beta \cdot \frac{\sigma_{\text{realized}}(t)}{\bar{\sigma}_{\text{baseline}}} - 1\right)\right)$$

---

### 2.4 Component 4: Ensemble Prediction Aggregator

#### Bayesian Model Weighting
Final directional predictions are generated by an elite ensemble weighted by regime compatibility:

$$w_i(t) = \frac{\mathbb{P}(R_t \mid \mathcal{I}_i) \cdot \mathcal{F}(\mathcal{I}_i)^\gamma}{\sum_{j \in \mathcal{T}_{\text{Alpha}}} \mathbb{P}(R_t \mid \mathcal{I}_j) \cdot \mathcal{F}(\mathcal{I}_j)^\gamma}$$

#### Disagreement Entropy Circuit Breaker
If the predictive divergence among elite models exceeds an entropy limit $\theta_{\text{entropy}}$:
- Exposure and leverage are automatically dialed down.
- A **System Regime Shift Alert** triggers accelerated exploration (elevated mutation) in the evolutionary population.

---

## 3. Evolutionary Cycle: Asynchronous Steady-State Island Model

To eliminate the stop-the-world latency spikes of synchronous generational epochs in 24/7 production, the engine implements an **Asynchronous Steady-State Island Model**:

```python
Procedure AsynchronousIslandWorker(IslandSubpopulation P_island, SharedParetoArchive Archive):
    While Running:
        # 1. Continuous Non-Dominated Sorting
        Fronts = FastNonDominatedSort(P_island.objectives)
        CrowdingDistances = CalculateCrowdingDistance(Fronts[0], P_island.objectives)
        UpdateSharedArchive(Archive, Fronts[0])

        # 2. Hypergamic Selection with Novelty-Governed Fallback
        aspirant = SelectByCrowdingDistance(Fronts, P_island)
        elite_target = SampleElite(Archive)

        If ResidualCorrelation(aspirant, elite_target) < DeltaOrtho:
            child = Crossover(aspirant.genotype, elite_target.genotype)
        Else:
            novel_aspirant = SelectAspirantByNovelty(P_island.aspirants, Archive.elites)
            child = ExploratoryCrossover(aspirant.genotype, novel_aspirant.genotype)

        # 3. Dynamic Volatility Mutation & Validation
        rate = ComputeAdaptiveMutationRate(MarketState.realized_volatility)
        mutated_child = Mutate(child, rate)

        If ValidateRiskConstraints(mutated_child):
            # Asynchronous steady-state replacement (replace worst dominated individual)
            worst_individual = SelectWorstDominated(P_island)
            P_island.Replace(worst_individual, mutated_child)
```

---

## 4. Implementation Roadmap for Quantitative Engineering Team

| Phase | Milestone | Deliverables |
| :--- | :--- | :--- |
| **Phase 1** | **Data Ingestion & Event Chaining** | Transformer entity-extraction pipeline, decay kernel modules, graph storage. |
| **Phase 2** | **Scenario Matrix & Game Theory** | Market response simulator, adversarial scenario definitions, Minimax regret evaluator. |
| **Phase 3** | **Evolutionary Engine Core** | Genotype schema, hypergamic selection gate, Deflated Sharpe & novelty fitness metrics. |
| **Phase 4** | **Ensemble Aggregator & Safeguards** | Regime-conditioned Bayesian weighting, entropy circuit breaker, paper trading pipeline. |
