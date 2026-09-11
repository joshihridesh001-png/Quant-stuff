# Design Specification: Phase 4, Step 3 — Hypergamic Assortative Selection & Residual Orthogonality Gating

## 1. Executive Overview & Problem Context
In Phase 4, Step 1, we implemented the declarative 20-gene `StrategyChromosome` with continuous unit-hypercube mapping ($\mathbf{u} \in [0, 1]^{20}$) via `ChromosomeVectorCodec`. In Step 2, we built `BoundaryAnchoredRVEARanker`, generating non-dominated Pareto fronts, SVD subspace novelty, and angle-penalized distance (APD) rankings.

In Step 3, we implement **Hypergamic Assortative Selection & Residual Orthogonality Gating** in `src/quant/analytics/hypergamic_selection.py`. This subsystem governs the reproductive cycle of the evolutionary engine, generating the offspring population for generation $t+1$ from the ranked generation $t$.

### 1.1 Why Standard Crossover & Selection Fail in Financial Alpha Evolution
1. **Premature Convergence / Homogenization**: In standard GAs (e.g. NSGA-II tournament), top performers mate preferentially with other top performers of similar phenotypes. Within a few generations, the population collapses into identical clones of the dominant market regime's strategy.
2. **The "Inverse Clone" Vulnerability**: Naive correlation gates ($\text{Corr}(e_A, e_B) < \delta$) admit anti-correlated strategies ($\text{Corr} = -0.95$) because negative numbers are smaller than $\delta$. In trading, an anti-correlated strategy is a redundant mirror image, wasting portfolio slots.
3. **Mating Deadlocks**: If strict orthogonality conditions are not met in small populations, naive rejection loops spin infinitely or population sizes collapse.
4. **Destructive Crossover**: Symmetrically blending an elite Alpha strategy with an Aspirant destroys the Alpha's carefully balanced risk budget and game-theoretic parameters.
5. **Invariant Violations**: Crossing over decoded parameters directly breaks invariants (e.g. $\tau_{\text{fast}} < \tau_{\text{slow}}$ and regime belief simplex $\mathbf{p} \in \Delta^3$).

---

## 2. Mathematical Formalisms & Algorithms

### 2.1 Front-Preserving Pareto Stratification
Instead of an arbitrary percentile cutoff (e.g., top 20%) that can arbitrarily split a non-dominated front, the population is stratified based on Pareto ranking from Step 2:
- **Alpha Cohort ($\mathcal{A}$)**: Formed by all candidates in Front 1 ($\mathcal{F}_1$). If $|\mathcal{F}_1| < N_\alpha = \lceil \rho_\alpha \cdot N \rceil$, the cohort is padded with the top APD-ranked candidates from Front 2 ($\mathcal{F}_2$). If $|\mathcal{F}_1| > N_\alpha$, all of Front 1 is retained to maintain front integrity:
  $$\mathcal{A} = \begin{cases}
    \mathcal{F}_1 \cup \text{Top}_{N_\alpha - |\mathcal{F}_1|}^{\text{APD}}(\mathcal{F}_2) & \text{if } |\mathcal{F}_1| < N_\alpha \\
    \mathcal{F}_1 & \text{if } |\mathcal{F}_1| \ge N_\alpha
  \end{cases}$$
- **Aspirant Cohort ($\mathcal{X}$)**: All remaining viable candidates:
  $$\mathcal{X} = \{ c \in \mathcal{P} \mid c \notin \mathcal{A} \land c.\text{is\_viable} \}$$
- Infeasible and unviable candidates ($c.\text{is\_feasible} = \text{False}$ or $\text{DSR} < 0.50$) are disqualified from reproduction.

### 2.2 Bidirectional Residual Orthogonality Gate
Let $e_A, e_B \in \mathbb{R}^T$ be the prediction residual vectors of candidate Alpha $A$ and Aspirant $B$.
The sample Pearson correlation is:
$$\rho(e_A, e_B) = \frac{\sum_{t=1}^T (e_{A,t} - \bar{e}_A)(e_{B,t} - \bar{e}_B)}{\sqrt{\sum_{t=1}^T (e_{A,t} - \bar{e}_A)^2} \sqrt{\sum_{t=1}^T (e_{B,t} - \bar{e}_B)^2}}$$
To prevent both direct clones ($\rho \to +1$) and inverse clones ($\rho \to -1$), the gate evaluates absolute correlation:
$$1 - |\rho(e_A, e_B)| \ge \delta_{\text{current}}$$
Where $\delta_{\text{current}}$ is the active orthogonality threshold.

### 2.3 Adaptive Assortative Mating with Deadlock Guarantee
To eliminate infinite rejection loops:
1. Candidate Alpha $A \in \mathcal{A}$ is selected via tournament (or round-robin).
2. Candidate Aspirants compete in a $k$-tournament to petition $A$.
3. If $1 - |\rho(e_A, e_B)| \ge \delta_{\text{current}}$, the pair $(A, B)$ is accepted.
4. If rejected, retry counter $k_{\text{attempt}}$ increments.
5. If $k_{\text{attempt}} \ge K_{\max}$ (default 5), the gate adaptively relaxes:
   $$\delta_{\text{current}} = \delta_{\text{ortho}} \cdot (\gamma_{\text{relax}})^{k_{\text{attempt}} - K_{\max} + 1}$$
   If still unmet after $2 \times K_{\max}$, the matcher selects the candidate in $\mathcal{X}$ with the maximum phenotypic distance in unit hypercube space:
   $$B^* = \arg\max_{B \in \mathcal{X}} \|\mathbf{u}_A - \mathbf{u}_B\|_2$$
   This strictly guarantees $O(1)$ bounded pairing time with zero deadlocks.

### 2.4 Asymmetric Latent Unit-Hypercube Crossover
Crossover operates strictly in the unit hypercube $\mathbf{u} \in [0, 1]^{20}$ via `ChromosomeVectorCodec`:
1. Encode parents: $\mathbf{u}_A = \text{codec.encode}(\text{chrom}_A)$, $\mathbf{u}_B = \text{codec.encode}(\text{chrom}_B)$.
2. For each gene dimension $d \in \{0, \dots, 19\}$:
   - **Gene-Family Role Assignment**:
     - If $d \in \text{RiskChromosome}$ or $\text{GameTheoryChromosome}$: Favored parent is Alpha $A$ with inheritance bias $P_A = 0.75$.
     - If $d \in \text{RepresentationChromosome}$ or $\text{InferenceChromosome}$: Favored parent is Aspirant $B$ with inheritance bias $P_B = 0.65$.
   - **Simulated Binary Crossover (SBX)**:
     Draw random $r \sim \mathcal{U}(0, 1)$. Compute spread factor $\beta$:
     $$\beta = \begin{cases}
       (2r)^{\frac{1}{\eta_c + 1}} & \text{if } r \le 0.5 \\
       \left(\frac{1}{2(1 - r)}\right)^{\frac{1}{\eta_c + 1}} & \text{if } r > 0.5
     \end{cases}$$
     Generate candidate gene values:
     $$u_d^{(1)} = 0.5 \cdot \left( (1 + \beta) u_{A,d} + (1 - \beta) u_{B,d} \right)$$
     $$u_d^{(2)} = 0.5 \cdot \left( (1 - \beta) u_{A,d} + (1 + \beta) u_{B,d} \right)$$
     Select the candidate gene that aligns with the favored parent's bias, and clamp to $[0.0, 1.0]$.
3. Decode offspring: $\text{child} = \text{codec.decode}(\mathbf{u}^{\text{child}})$.
   - By construction of `ChromosomeVectorCodec`:
     - $\tau_{\text{fast}} = r_\tau \tau_{\text{slow}} < \tau_{\text{slow}}$ is guaranteed.
     - $\sum p_j \equiv 1.0, p_j > 0$ on the simplex is guaranteed.
     - Discrete parameters are centered at bucket midpoints with zero drift.

### 2.5 Elitism Preservation
The top $N_{\text{elite}}$ (default 2) candidates from Front 1 (ordered by APD score) are passed directly into generation $t+1$ without crossover, guaranteeing monotonic preservation of the Pareto-optimal frontier.

---

## 3. Class Architecture & Component Contracts

```
               ┌──────────────────────────────────────────────┐
               │              HypergamicConfig                │
               └──────────────────────┬───────────────────────┘
                                      │
               ┌──────────────────────▼───────────────────────┐
               │           ParetoCohortStratifier             │
               │  - Stratifies Front-1 Alphas & Aspirants     │
               └──────────────────────┬───────────────────────┘
                                      │
               ┌──────────────────────▼───────────────────────┐
               │          ResidualOrthogonalityGate           │
               │  - Evaluates 1 - |rho(e_A, e_B)| >= delta    │
               └──────────────────────┬───────────────────────┘
                                      │
               ┌──────────────────────▼───────────────────────┐
               │          HypergamicPartnerMatcher            │
               │  - Bounded tournament with adaptive fallback │
               └──────────────────────┬───────────────────────┘
                                      │
               ┌──────────────────────▼───────────────────────┐
               │         AsymmetricLatentCrossover            │
               │  - Unit-hypercube SBX with gene-family bias  │
               └──────────────────────┬───────────────────────┘
                                      │
               ┌──────────────────────▼───────────────────────┐
               │           HypergamicSelectionEngine          │
               │  - Master Facade: reproduce() -> Offspring   │
               └──────────────────────────────────────────────┘
```

### 3.1 Dataclasses & Configuration
```python
@dataclass(frozen=True)
class HypergamicConfig:
    alpha_ratio: float = 0.25
    orthogonality_threshold: float = 0.30  # delta_ortho: min 30% unexplained variance
    max_mating_attempts: int = 5
    relaxation_factor: float = 0.80
    tournament_size: int = 3
    crossover_distribution_index: float = 15.0  # eta_c for SBX
    alpha_risk_inheritance_prob: float = 0.75
    aspirant_repr_inheritance_prob: float = 0.65
    elitism_count: int = 2


@dataclass(frozen=True)
class MatingPair:
    alpha_id: str
    aspirant_id: str
    residual_correlation: float
    is_relaxed: bool
    relaxation_level: int


@dataclass(frozen=True)
class OffspringResult:
    offspring_chromosomes: Tuple[StrategyChromosome, ...]
    mating_pairs: Tuple[MatingPair, ...]
    alpha_ids: Tuple[str, ...]
    aspirant_ids: Tuple[str, ...]
    elite_ids: Tuple[str, ...]
    rejection_count: int
    relaxation_count: int
```

### 3.2 Invariant Contracts
- `INV-HYP-001`: $N_{\text{offspring}} == N_{\text{target}}$ exactly.
- `INV-HYP-002`: All offspring chromosomes are valid instances of `StrategyChromosome` satisfying `INV-CHROM-001` through `INV-CHROM-006`.
- `INV-HYP-003`: Elitism clones are bitwise identical to the top $N_{\text{elite}}$ Front-1 candidates.
- `INV-HYP-004`: Infeasible candidates are never admitted to Alpha or Aspirant cohorts.
- `INV-HYP-005`: Maximum retry bounds guarantee termination within $O(N_{\text{target}} \cdot K_{\max})$ steps with zero deadlocks.

---

## 4. Test & Verification Plan
1. **Stratification Tests**:
   - Small Front 1 ($|\mathcal{F}_1| < N_\alpha$) pulls top APD Front 2 candidates.
   - Large Front 1 ($|\mathcal{F}_1| \ge N_\alpha$) retains entire Front 1 without truncation.
   - Infeasible candidates are excluded from cohorts.
2. **Residual Gate Tests**:
   - High positive correlation ($\rho = 0.95$) rejected.
   - High negative correlation ($\rho = -0.95$) rejected (catches inverse clones).
   - Orthogonal pair ($\rho = 0.10$) accepted.
   - Zero-variance residuals handled safely without divide-by-zero.
3. **Matcher & Fallback Tests**:
   - Nominal pairing under abundant diversity.
   - Degenerate population (all Aspirants correlated with Alpha) triggers adaptive relaxation and phenotypic distance fallback within $K_{\max}$ retries without hanging.
4. **Crossover Invariant Tests**:
   - 100 consecutive crossovers produce 100% valid chromosomes with $\tau_{\text{fast}} < \tau_{\text{slow}}$, valid simplices, and bounded values.
   - Asymmetric gene-family inheritance verified statistically across 1,000 runs (Alpha dominates risk; Aspirant dominates representation).
5. **Performance Benchmark**:
   - Generating 100 offspring from 100 candidates executes in $< 25\text{ms}$.
