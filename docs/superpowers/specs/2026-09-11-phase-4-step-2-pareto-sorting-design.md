# Phase 4, Step 2: Boundary-Anchored Adaptive RVEA with Vectorized Orthogonal Quality-Diversity (BA-ARVEA-QD)
**Design Specification Document**
* **Status:** Approved / Authoritative
* **Module Target:** `src/quant/analytics/pareto_sorting.py`
* **Parent Phase:** Phase 4 (Evolutionary Population & Hypergamy Dynamics)
* **Date:** 2026-09-11

---

## 1. Executive Summary & Problem Context

In evolutionary quantitative alpha discovery, population diversity must be preserved without succumbing to "novelty parasites" (unviable, noisy strategies that score high distance metrics while destroying capital). Standard multi-objective evolutionary algorithms (such as classic NSGA-II or static RVEA) exhibit four structural failure modes when deployed in noisy financial time series:
1. **The Novelty Parasite Vulnerability**: Rewarding raw parameter distance allows economically broken strategies to dominate non-dominated fronts.
2. **Static Simplex Misalignment**: Rigid Das-Dennis reference rays waste search budget pointing into empty, economically unattainable regions of objective space.
3. **Boundary Ray Starvation**: Naive adaptive reference ray migration causes extreme single-objective alpha champions (e.g., maximum statistical significance) to be discarded in favor of central compromise solutions.
4. **Sample Noise Overconfidence**: Deterministic Pareto dominance ignores the standard error of econometric estimators (such as Deflated Sharpe Ratio), allowing strategies to dominate based on historical sample noise rather than true signal.

This specification establishes **Boundary-Anchored Adaptive RVEA with Vectorized Orthogonal Quality-Diversity (BA-ARVEA-QD)**—a production-grade mathematical framework designed to eliminate these failure modes through statistical confidence dominance, single-pass BLAS GEMM residual orthogonality, and boundary-anchored adaptive reference lattices.

---

## 2. Mathematical Formalisms & Core Formulations

### 2.1 The Tri-Objective Evaluation Space
Each evaluated candidate individual $i \in \{1, \dots, N\}$ is characterized by objective vector $\mathbf{f}_i \in \mathbb{R}^3$, formulated under uniform minimization:

$$\mathbf{f}_i = \begin{bmatrix} f_{i, 1} \\ f_{i, 2} \\ f_{i, 3} \end{bmatrix} = \begin{bmatrix} -\widehat{\text{DSR}}_i \\ \Psi_i(\mathbf{a}^*) \\ -\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E}) \end{bmatrix}$$

*   $f_{i, 1} = -\widehat{\text{DSR}}_i$: Minimizing negated Deflated Sharpe Ratio (maximizing statistical rejection of data-snooping).
*   $f_{i, 2} = \Psi_i(\mathbf{a}^*)$: Minimizing certified worst-case regret from the Phase 3 Minimax Newton Solver.
*   $f_{i, 3} = -\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E})$: Minimizing negated orthogonal residual novelty.

### 2.2 Statistical Confidence Dominance ($\tau$-Dominance Operator)
To prevent overfitting to backtest sample variance, strategy $A$ dominates strategy $B$ ($A \succ_{\tau} B$) if and only if:
1. $A$ is statistically no worse than $B$ across all three objectives:
   $$\begin{cases}
   f_{A, 1} \le f_{B, 1} + \tau_{\text{conf}} \sqrt{\text{SE}_A^2 + \text{SE}_B^2} \\
   f_{A, 2} \le f_{B, 2} \\
   f_{A, 3} \le f_{B, 3}
   \end{cases}$$
2. $A$ is strictly superior to $B$ in at least one objective:
   $$\begin{cases}
   f_{A, 1} < f_{B, 1} - \tau_{\text{conf}} \sqrt{\text{SE}_A^2 + \text{SE}_B^2} \quad \text{OR} \\
   f_{A, 2} < f_{B, 2} - \epsilon_{\text{regret}} \quad \text{OR} \\
   f_{A, 3} < f_{B, 3} - \epsilon_{\text{novelty}}
   \end{cases}$$
Where:
*   $\tau_{\text{conf}} = 1.645$ (corresponding to a 95% one-tailed confidence boundary).
*   $\text{SE}_i = \sqrt{\frac{1 + \frac{1}{2} \hat{\gamma}_{3, i}^2}{T_i - 1}}$ (standard error of the Sharpe/DSR estimator).
*   $\epsilon_{\text{regret}} = 10^{-6}$, $\epsilon_{\text{novelty}} = 10^{-4}$.

### 2.3 Viability-Gated Vectorized Residual Orthogonality (BLAS GEMM)
To eradicate the Novelty Parasite and achieve sub-millisecond execution over hundreds of historical elite strategies:
1. **Viability Gate**: If $\widehat{\text{DSR}}_i < 0.50$ or $T_i < \text{MinBTL}$, then $\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E}) \equiv 0.0$ and the individual is flagged as infeasible.
2. **Normalized Residual Matrix**: Each valid strategy's out-of-fold prediction residual vector $\mathbf{e}_i \in \mathbb{R}^T$ is centered and $L_2$-normalized upon storage:
   $$\tilde{\mathbf{e}}_i = \frac{\mathbf{e}_i - \bar{e}_i \mathbf{1}}{\|\mathbf{e}_i - \bar{e}_i \mathbf{1}\|_2} \in \mathbb{R}^T$$
3. **GEMM Correlation Calculation**: Given candidate batch matrix $\tilde{\mathbf{E}}_{\text{cand}} \in \mathbb{R}^{N \times T}$ and archive matrix $\tilde{\mathbf{E}}_{\text{arch}} \in \mathbb{R}^{N_{\text{arch}} \times T}$, the complete cross-correlation tensor is computed via a single BLAS GEMM operation:
   $$\mathbf{C} = \tilde{\mathbf{E}}_{\text{cand}} \tilde{\mathbf{E}}_{\text{arch}}^T \in \mathbb{R}^{N \times N_{\text{arch}}}$$
4. **Orthogonal Novelty Score**:
   $$\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E}) = 1.0 - \max_{1 \le k \le N_{\text{arch}}} |C_{i, k}|$$
   *If $\mathcal{E} = \emptyset$, $\rho_{\text{ortho}} \equiv 1.0$ for all viable candidates.*

### 2.4 Boundary-Anchored Adaptive Reference Lattice (BA-ARVEA)
The reference ray set $\mathbf{V} = \mathbf{V}_{\text{anchor}} \cup \mathbf{V}_{\text{interior}}$ consists of $K = 28$ rays ($M=3, p=6$):
1. **Anchor Rays ($\mathbf{V}_{\text{anchor}}$)**: The three basis coordinate rays $\mathbf{e}_1 = [1, 0, 0]^T$, $\mathbf{e}_2 = [0, 1, 0]^T$, and $\mathbf{e}_3 = [0, 0, 1]^T$ are **strictly immutable**. They are never adapted or removed, guaranteeing that extreme specialist alphas are never extinguished.
2. **Interior Adaptive Rays ($\mathbf{V}_{\text{interior}}$)**: For the remaining 25 interior rays:
   * Maintain an idle counter $g_{\text{idle}, j}$ tracking consecutive generations with zero associated solutions.
   * Every $G_{\text{adapt}} = 5$ generations, if $g_{\text{idle}, j} \ge 3$, re-project $\hat{\mathbf{v}}_j$ toward the most densely populated cluster centroid $\bar{\mathbf{f}}_{\text{dense}}^{\text{norm}}$:
     $$\hat{\mathbf{v}}_j^{(t+1)} = \frac{(1 - \eta) \hat{\mathbf{v}}_j^{(t)} + \eta \bar{\mathbf{f}}_{\text{dense}}^{\text{norm}}}{\|(1 - \eta) \hat{\mathbf{v}}_j^{(t)} + \eta \bar{\mathbf{f}}_{\text{dense}}^{\text{norm}}\|_2}, \quad \eta = 0.30$$
   * Reset $g_{\text{idle}, j} = 0$.

### 2.5 Angle-Penalized Distance (APD)
For each normalized objective vector $\mathbf{f}_i^{\text{norm}}$ associated with nearest reference ray $\hat{\mathbf{v}}_j$ under acute angle $\theta_{i, j} = \arccos\left(\frac{\mathbf{f}_i^{\text{norm}} \cdot \hat{\mathbf{v}}_j}{\|\mathbf{f}_i^{\text{norm}}\|_2}\right)$:

$$d_{\text{APD}}(\mathbf{f}_i^{\text{norm}}, \hat{\mathbf{v}}_j) = \left( 1 + M \cdot \left( \frac{t}{t_{\max}} \right)^\alpha \cdot \frac{\theta_{i, j}}{\gamma_j} \right) \cdot \|\mathbf{f}_i^{\text{norm}}\|_2$$

*   $\alpha = 2.0$ enforces exploration in early generations and convergence along reference rays in late generations.
*   $\gamma_j = \min_{k \ne j} \arccos(\hat{\mathbf{v}}_j \cdot \hat{\mathbf{v}}_k)$ provides local angle normalization.

---

## 3. Component Architecture & Data Contracts

The module `src/quant/analytics/pareto_sorting.py` will expose four decoupled, stateless analytical classes:

### 3.1 Class Topology
1. **`ViabilityGatedOrthogonalArchive`**:
   * Manages standardized residual matrix $\tilde{\mathbf{E}}_{\text{arch}} \in \mathbb{R}^{N_{\text{arch}} \times T}$ with maximum capacity $N_{\text{max}} = 500$ using an internal FIFO circular buffer.
   * Exposes `compute_novelty(residual_batch, viability_mask) -> np.ndarray` running vectorized BLAS matrix multiplication.
   * Exposes `admit(candidate_id, normalized_residual, novelty_score, dsr)` adding certified diverse alphas to the archive.
2. **`AdaptiveReferenceLattice`**:
   * Pre-computes initial Das-Dennis simplex rays ($K=28$) with anchored coordinate bases.
   * Pre-computes nearest-neighbor ray angles $\gamma_j$ via pairwise dot products.
   * Exposes `associate_and_penalize(normalized_objectives, generation_ratio) -> Tuple[ray_indices, angles, apd_scores]`.
   * Exposes `adapt_interior_rays(normalized_objectives, associations)` executing smooth cluster-guided migration.
3. **`ProbabilisticNonDominatedSorter`**:
   * Implements Efficient Non-Dominated Sort with Sequential Search (ENS-SS) under the statistical $\tau$-confidence dominance operator.
   * Partitions candidates into Pareto fronts $\mathcal{F}_1, \dots, \mathcal{F}_F$ and separate infeasible partition $\mathcal{F}_{\text{inf}}$.
4. **`BoundaryAnchoredRVEARanker` (Facade)**:
   * Coordinates normalization, orthogonal novelty calculation, ENS-SS sorting, reference ray association, and APD ranking into a clean `RankingResult`.

### 3.2 Immutability & Contract Schema
```python
@dataclass(frozen=True)
class CandidateFitness:
    candidate_id: str
    dsr: float
    dsr_standard_error: float
    minimax_regret: float
    residual_series: np.ndarray  # float64, 1D contiguous
    backtest_length: int
    is_feasible: bool

@dataclass(frozen=True)
class ParetoFront:
    rank: int
    candidate_ids: Tuple[str, ...]
    apd_scores: Tuple[float, ...]

@dataclass(frozen=True)
class RankingResult:
    fronts: Tuple[ParetoFront, ...]
    infeasible_ids: Tuple[str, ...]
    active_reference_rays: np.ndarray
    archive_size: int
```

---

## 4. Defensive Boundary Invariants

Every operation must satisfy non-negotiable architectural invariants:

| Invariant Code | Definition | Enforcement Boundary | Violation Action |
| :--- | :--- | :--- | :--- |
| `INV-PAR-001` | Strict Finiteness: $\forall f_{i, m} \in \mathbb{R}$, no `NaN`, `Inf`, or complex numbers. | `CandidateFitness.__post_init__` | Raise `CorruptedFitnessException` |
| `INV-PAR-002` | Contiguous Residual Vector: $\mathbf{e}_i$ must be 1D contiguous `np.float64` with $T \ge 100$. | `CandidateFitness.__post_init__` | Raise `InvalidResidualException` |
| `INV-PAR-003` | Immutable Ray Norms: $\|\hat{\mathbf{v}}_j\|_2 \equiv 1.0 \pm 10^{-6}$ for all $j \in \{1, \dots, K\}$. | `AdaptiveReferenceLattice.adapt` | Raise `DegenerateLatticeException` |
| `INV-PAR-004` | Non-Empty Basis Anchors: Basis rays $[1,0,0], [0,1,0], [0,0,1]$ must remain fixed at all $t$. | `AdaptiveReferenceLattice.adapt` | Assertion failure |
| `INV-PAR-005` | Partition Conservation: $\left(\bigcup \mathcal{F}_k\right) \cup \mathcal{F}_{\text{inf}} = \mathcal{P}$, $\mathcal{F}_j \cap \mathcal{F}_k = \emptyset$. | `ProbabilisticNonDominatedSorter.sort` | Assertion failure |
| `INV-PAR-006` | Bounded Orthogonality: $\rho_{\text{ortho}} \in [0.0, 1.0]$. | `ViabilityGatedOrthogonalArchive` | Clamped to $[0.0, 1.0]$ |

---

## 5. Zero-Execution Diagnostic Failure Matrix (Rule 2 Compliance)

In accordance with Rule 2 of `Rules.md`, all potential failure modes are mapped to diagnostic fault vector codes cataloged in `Memory.md`:

```
Fault Vector: ERR-EVO-PAR-001
- Component: src/quant/analytics/pareto_sorting.py::CandidateFitness::__post_init__
- Nominal Behavior: Valid finite fitness values (DSR, Regret, SE).
- Malfunction Symptoms: CorruptedFitnessException raised during candidate evaluation.
- Root Cause Etiology: Model producing NaN regret due to singular covariance matrix in Phase 3 Minimax Newton Solver, or negative variance in DSR calculation.
- Zero-Runtime Verification: Check candidate upstream execution trace in logs for zero-variance predictions or singular covariance flags.
- Remediation: Validate upstream chromosome parameter bounds; ensure epsilon damping in Minimax solver.
- Secondary Failure: Population sorting halts; whole generation cycle fails.

Fault Vector: ERR-EVO-PAR-002
- Component: src/quant/analytics/pareto_sorting.py::ViabilityGatedOrthogonalArchive::compute_novelty
- Nominal Behavior: Standardized residuals producing Pearson correlation in [-1, 1].
- Malfunction Symptoms: InvalidResidualException or correlation values exceeding [-1.0001, 1.0001].
- Root Cause Etiology: Residual series has zero variance (constant prediction), resulting in division-by-zero during L2 normalization.
- Zero-Runtime Verification: Inspect candidate residual standard deviation; confirm std(e) > 1e-12.
- Remediation: Add epsilon floor to L2-norm calculation; classify zero-variance predictors as infeasible.
- Secondary Failure: NaN propagation through BLAS GEMM into novelty scores across the entire population.

Fault Vector: ERR-EVO-PAR-003
- Component: src/quant/analytics/pareto_sorting.py::AdaptiveReferenceLattice::adapt_interior_rays
- Nominal Behavior: Interior reference rays migrate smoothly without collapsing onto a single point.
- Malfunction Symptoms: DegenerateLatticeException raised; reference rays lose angular separation (gamma_j < 1e-6).
- Root Cause Etiology: All candidates in generation t collapsed to identical objective coordinates, causing all rays to pull toward a single degenerate point.
- Zero-Runtime Verification: Check objective variance across generation candidates.
- Remediation: Enforce minimum angular separation threshold gamma_min = 0.05 during interior ray adaptation; reject updates that violate minimum separation.
- Secondary Failure: Loss of population diversity; premature evolutionary convergence.
```

---

## 6. Verification & Quality Gates (Rule 3 Compliance)

The implementation must pass all four continuous integration gates:
1. **Static Linting**: `ruff check src/quant/analytics/pareto_sorting.py` with zero warnings.
2. **Code Formatting**: `ruff format --check src/quant/analytics/pareto_sorting.py` enforcing 100-character line width.
3. **Strict Type Checking**: `mypy src/quant/analytics/pareto_sorting.py` in strict mode (`disallow_untyped_defs = true`, `no_implicit_optional = true`) with zero errors.
4. **Comprehensive Test Suite**: Target **$\ge 95\%$ line coverage** on `pareto_sorting.py` with unit tests in `tests/unit/test_pareto_sorting.py` covering:
   * Empty archive initialization (Gen 0 fallback to $\rho_{\text{ortho}} = 1.0$).
   * Orthogonal residual calculation via BLAS GEMM against known synthetic signals.
   * Viability gating (rejection of unviable novelty parasites).
   * Probabilistic $\tau$-dominance vs deterministic dominance under simulated estimator noise.
   * Boundary ray immutability across 50 adaptation cycles.
   * Angle-Penalized Distance escalation across early, mid, and late generations.
   * Extreme edge cases (zero variance, identical candidates, $100\%$ infeasible populations).
