# Phase 4, Step 2: Boundary-Anchored Adaptive RVEA with SVD Subspace Orthogonality & Dependent-Sample Dominance (BA-ARVEA-SO)
**Design Specification Document**
* **Status:** Approved / Authoritative (Hardened Institutional Frontier)
* **Module Target:** `src/quant/analytics/pareto_sorting.py`
* **Parent Phase:** Phase 4 (Evolutionary Population & Hypergamy Dynamics)
* **Date:** 2026-09-11

---

## 1. Executive Summary & Problem Context

In evolutionary quantitative alpha discovery, population diversity must be preserved without succumbing to "novelty parasites" (unviable, noisy strategies that score high distance metrics while destroying capital). Standard multi-objective evolutionary algorithms (such as classic NSGA-II or static RVEA) exhibit four fundamental theoretical flaws when deployed in noisy financial time series:
1. **The Dependent-Sample Variance Blindspot**: When strategies $A$ and $B$ are backtested over identical historical time series, their returns are strongly correlated ($\rho_{A, B} \gg 0$). Assuming independence ($\sqrt{\text{SE}_A^2 + \text{SE}_B^2}$) overestimates standard error by up to $300\%$, falsely paralyzing statistical dominance checks.
2. **The Multi-Collinearity Subspace Illusion**: Pairwise correlation checks ($\max_k |\text{Corr}|$) only detect 1-on-1 collinearity. A candidate strategy that is a linear combination of multiple existing elite strategies ($\mathbf{e}_A = 0.5 \mathbf{e}_1 + 0.5 \mathbf{e}_2$) will appear artificially novel ($30\%$ novelty score) despite adding zero irreducible alpha.
3. **Static Simplex Misalignment**: Rigid Das-Dennis reference rays waste search budget pointing into empty, economically unattainable regions of objective space.
4. **Boundary Ray Starvation**: Naive adaptive reference ray migration causes extreme single-objective alpha champions (e.g., maximum statistical significance) to be discarded in favor of central compromise solutions.

This specification establishes **Boundary-Anchored Adaptive RVEA with SVD Subspace Orthogonality & Memmel–Ledoit–Wolf Dependent Dominance (BA-ARVEA-SO)**—an institutional-grade mathematical framework resolving all four blindspots.

---

## 2. Mathematical Formalisms & Core Formulations

### 2.1 The Tri-Objective Evaluation Space
Each evaluated candidate individual $i \in \{1, \dots, N\}$ is characterized by objective vector $\mathbf{f}_i \in \mathbb{R}^3$, formulated under uniform minimization:

$$\mathbf{f}_i = \begin{bmatrix} f_{i, 1} \\ f_{i, 2} \\ f_{i, 3} \end{bmatrix} = \begin{bmatrix} -\widehat{\text{DSR}}_i \\ \Psi_i(\mathbf{a}^*) \\ -\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E}) \end{bmatrix}$$

*   $f_{i, 1} = -\widehat{\text{DSR}}_i$: Minimizing negated Deflated Sharpe Ratio (maximizing statistical rejection of data-snooping).
*   $f_{i, 2} = \Psi_i(\mathbf{a}^*)$: Minimizing certified worst-case regret from the Phase 3 Minimax Newton Solver.
*   $f_{i, 3} = -\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E})$: Minimizing negated SVD subspace orthogonal novelty.

### 2.2 Memmel–Ledoit–Wolf Dependent Asymptotic Dominance ($\tau$-Dominance)
Because strategies $A$ and $B$ are evaluated over the identical market history $t \in \{1, \dots, T\}$, their returns $\mathbf{r}_A, \mathbf{r}_B \in \mathbb{R}^T$ exhibit non-zero empirical correlation $\rho_{A, B} = \text{Corr}(\mathbf{r}_A, \mathbf{r}_B)$. 

Following Memmel (2003) and Ledoit & Wolf (2008), the exact asymptotic variance of the Sharpe/DSR difference $\Delta \widehat{\text{DSR}}_{A, B} = \widehat{\text{DSR}}_A - \widehat{\text{DSR}}_B$ is:

$$\sigma^2(\Delta \widehat{\text{DSR}}_{A, B}) = \frac{1}{T} \left( 2(1 - \rho_{A, B}) + \frac{1}{2} \left( \widehat{\text{DSR}}_A^2 + \widehat{\text{DSR}}_B^2 - 2 \rho_{A, B}^2 \widehat{\text{DSR}}_A \widehat{\text{DSR}}_B \right) \right)$$

Strategy $A$ statistically dominates strategy $B$ ($A \succ_{\tau} B$) if and only if:
1. $A$ is statistically no worse than $B$ across all three objectives:
   $$\begin{cases}
   f_{A, 1} \le f_{B, 1} + \tau_{\text{conf}} \sigma(\Delta \widehat{\text{DSR}}_{A, B}) \\
   f_{A, 2} \le f_{B, 2} \\
   f_{A, 3} \le f_{B, 3}
   \end{cases}$$
2. $A$ is strictly superior to $B$ in at least one objective:
   $$\begin{cases}
   f_{A, 1} < f_{B, 1} - \tau_{\text{conf}} \sigma(\Delta \widehat{\text{DSR}}_{A, B}) \quad \text{OR} \\
   f_{A, 2} < f_{B, 2} - \epsilon_{\text{regret}} \quad \text{OR} \\
   f_{A, 3} < f_{B, 3} - \epsilon_{\text{novelty}}
   \end{cases}$$
Where $\tau_{\text{conf}} = 1.645$ (95% one-tailed confidence boundary), $\epsilon_{\text{regret}} = 10^{-6}$, and $\epsilon_{\text{novelty}} = 10^{-4}$.

### 2.3 Truncated SVD Subspace Orthogonality ($1 - R^2$)
To completely eliminate multi-collinear redundancy and identify true irreducible residual alpha:
1. **Viability Gate**: If $\widehat{\text{DSR}}_i < 0.50$ or $T_i < \text{MinBTL}$, then $\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E}) \equiv 0.0$ and the individual is partitioned into the Infeasible Cohort ($\mathcal{F}_{\text{inf}}$).
2. **Subspace Basis Construction via Thin SVD**: Given historical elite residual matrix $\tilde{\mathbf{E}}_{\text{arch}} \in \mathbb{R}^{N_{\text{arch}} \times T}$ (with centered, $L_2$-normalized rows):
   $$\tilde{\mathbf{E}}_{\text{arch}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
   The orthonormal basis spanning the elite alpha space is $\mathbf{V}_K \in \mathbb{R}^{T \times K}$, retaining the top $K \le 50$ singular vectors capturing $\ge 99\%$ of variance.
3. **Subspace Orthogonal Projection**: For candidate residual $\tilde{\mathbf{e}}_i \in \mathbb{R}^T$:
   * Parallel projection onto elite span: $\hat{\mathbf{e}}_i^{\parallel} = \mathbf{V}_K (\mathbf{V}_K^T \tilde{\mathbf{e}}_i) \in \mathbb{R}^T$
   * Irreducible orthogonal residual: $\mathbf{e}_i^{\perp} = \tilde{\mathbf{e}}_i - \hat{\mathbf{e}}_i^{\parallel} \in \mathbb{R}^T$
4. **Fraction of Unexplained Variance ($1 - R^2$)**:
   $$\rho_{\text{ortho}}(\mathbf{e}_i, \mathcal{E}) = \frac{\|\mathbf{e}_i^{\perp}\|_2^2}{\|\tilde{\mathbf{e}}_i\|_2^2} \in [0.0, 1.0]$$
   *If $\mathcal{E} = \emptyset$ (Generation 0), $\rho_{\text{ortho}} \equiv 1.0$ for all viable candidates.*

### 2.4 Boundary-Anchored Adaptive Reference Lattice (BA-ARVEA)
The reference ray set $\mathbf{V} = \mathbf{V}_{\text{anchor}} \cup \mathbf{V}_{\text{interior}}$ consists of $K_{\text{rays}} = 28$ rays ($M=3, p=6$):
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
1. **`SVDSubspaceOrthogonalArchive`**:
   * Maintains historical residual matrix $\tilde{\mathbf{E}}_{\text{arch}} \in \mathbb{R}^{N_{\text{arch}} \times T}$ ($N_{\text{max}} = 500$) with incremental SVD basis caching ($\mathbf{V}_K$).
   * Exposes `compute_novelty(residual_batch, viability_mask) -> np.ndarray` running vectorized subspace projection in $<1.5\,\text{ms}$.
   * Exposes `admit(candidate_id, normalized_residual, novelty_score, dsr)` inserting diverse alphas and updating $\mathbf{V}_K$.
2. **`AdaptiveReferenceLattice`**:
   * Pre-computes initial Das-Dennis simplex rays ($K=28$) with anchored coordinate bases.
   * Pre-computes nearest-neighbor ray angles $\gamma_j$ via pairwise dot products.
   * Exposes `associate_and_penalize(normalized_objectives, generation_ratio) -> Tuple[ray_indices, angles, apd_scores]`.
   * Exposes `adapt_interior_rays(normalized_objectives, associations)` executing smooth cluster-guided migration.
3. **`DependentNonDominatedSorter`**:
   * Implements Efficient Non-Dominated Sort with Sequential Search (ENS-SS) under the Memmel–Ledoit–Wolf dependent asymptotic dominance operator.
   * Partitions candidates into Pareto fronts $\mathcal{F}_1, \dots, \mathcal{F}_F$ and separate infeasible partition $\mathcal{F}_{\text{inf}}$.
4. **`BoundaryAnchoredRVEARanker` (Facade)**:
   * Coordinates normalization, subspace orthogonal novelty calculation, dependent ENS-SS sorting, reference ray association, and APD ranking into a clean `RankingResult`.

### 3.2 Immutability & Contract Schema
```python
@dataclass(frozen=True)
class CandidateFitness:
    candidate_id: str
    dsr: float
    minimax_regret: float
    return_series: np.ndarray    # float64, 1D contiguous returns for Memmel-Ledoit-Wolf covariance
    residual_series: np.ndarray  # float64, 1D contiguous out-of-fold residuals for SVD orthogonality
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
    subspace_rank: int
```

---

## 4. Defensive Boundary Invariants

| Invariant Code | Definition | Enforcement Boundary | Violation Action |
| :--- | :--- | :--- | :--- |
| `INV-PAR-001` | Strict Finiteness: $\forall f_{i, m} \in \mathbb{R}$, no `NaN`, `Inf`, or complex numbers. | `CandidateFitness.__post_init__` | Raise `CorruptedFitnessException` |
| `INV-PAR-002` | Contiguous Series Vector: Both $\mathbf{r}_i$ and $\mathbf{e}_i$ must be 1D contiguous `np.float64` of identical length $T \ge 100$. | `CandidateFitness.__post_init__` | Raise `InvalidResidualException` |
| `INV-PAR-003` | Immutable Ray Norms: $\|\hat{\mathbf{v}}_j\|_2 \equiv 1.0 \pm 10^{-6}$ for all $j \in \{1, \dots, K\}$. | `AdaptiveReferenceLattice.adapt` | Raise `DegenerateLatticeException` |
| `INV-PAR-004` | Non-Empty Basis Anchors: Basis rays $[1,0,0], [0,1,0], [0,0,1]$ must remain fixed at all $t$. | `AdaptiveReferenceLattice.adapt` | Assertion failure |
| `INV-PAR-005` | Partition Conservation: $\left(\bigcup \mathcal{F}_k\right) \cup \mathcal{F}_{\text{inf}} = \mathcal{P}$, $\mathcal{F}_j \cap \mathcal{F}_k = \emptyset$. | `DependentNonDominatedSorter.sort` | Assertion failure |
| `INV-PAR-006` | SVD Subspace Orthogonality Bounds: $\rho_{\text{ortho}} \in [0.0, 1.0]$. | `SVDSubspaceOrthogonalArchive` | Clamped to $[0.0, 1.0]$ |

---

## 5. Zero-Execution Diagnostic Failure Matrix (Rule 2 Compliance)

In accordance with Rule 2 of `Rules.md`, all potential failure modes are mapped to diagnostic fault vector codes cataloged in `Memory.md`:

```
Fault Vector: ERR-EVO-PAR-001
- Component: src/quant/analytics/pareto_sorting.py::CandidateFitness::__post_init__
- Nominal Behavior: Valid finite fitness values (DSR, Regret, Returns, Residuals).
- Malfunction Symptoms: CorruptedFitnessException raised during candidate evaluation.
- Root Cause Etiology: Model producing NaN regret due to singular covariance matrix in Phase 3 Minimax Newton Solver, or negative variance in DSR calculation.
- Zero-Runtime Verification: Check candidate upstream execution trace in logs for zero-variance predictions or singular covariance flags.
- Remediation: Validate upstream chromosome parameter bounds; ensure epsilon damping in Minimax solver.
- Secondary Failure: Population sorting halts; whole generation cycle fails.

Fault Vector: ERR-EVO-PAR-002
- Component: src/quant/analytics/pareto_sorting.py::SVDSubspaceOrthogonalArchive::compute_novelty
- Nominal Behavior: Truncated SVD subspace projection yields orthogonal residual norm in [0, norm(e)].
- Malfunction Symptoms: InvalidResidualException or rho_ortho exceeding [0.0, 1.0001].
- Root Cause Etiology: Residual series has zero variance (constant prediction), resulting in division-by-zero during L2 normalization.
- Zero-Runtime Verification: Inspect candidate residual standard deviation; confirm std(e) > 1e-12.
- Remediation: Add epsilon floor to L2-norm calculation; classify zero-variance predictors as infeasible.
- Secondary Failure: NaN propagation through SVD projection into novelty scores across the entire population.

Fault Vector: ERR-EVO-PAR-003
- Component: src/quant/analytics/pareto_sorting.py::AdaptiveReferenceLattice::adapt_interior_rays
- Nominal Behavior: Interior reference rays migrate smoothly without collapsing onto a single point.
- Malfunction Symptoms: DegenerateLatticeException raised; reference rays lose angular separation (gamma_j < 1e-6).
- Root Cause Etiology: All candidates in generation t collapsed to identical objective coordinates, causing all rays to pull toward a single degenerate point.
- Zero-Runtime Verification: Check objective variance across generation candidates.
- Remediation: Enforce minimum angular separation threshold gamma_min = 0.05 during interior ray adaptation; reject updates that violate minimum separation.
- Secondary Failure: Loss of population diversity; premature evolutionary convergence.

Fault Vector: ERR-EVO-PAR-004
- Component: src/quant/analytics/pareto_sorting.py::DependentNonDominatedSorter::_compute_dependent_se
- Nominal Behavior: Memmel-Ledoit-Wolf asymptotic variance yields positive finite standard error.
- Malfunction Symptoms: Floating-point warning or negative square-root operand in sigma(Delta DSR).
- Root Cause Etiology: Return series collinearity matrix ill-conditioned or absolute correlation rho > 1.0 + 1e-9 due to precision drift.
- Zero-Runtime Verification: Check return correlation clamp bounds in logs.
- Remediation: Enforce numerical clamping rho = np.clip(rho, -0.9999, 0.9999) before asymptotic variance evaluation.
- Secondary Failure: Dominance calculation throws ValueError; Pareto sorting aborts.
```

---

## 6. Verification & Quality Gates (Rule 3 Compliance)

The implementation must pass all four continuous integration gates:
1. **Static Linting**: `ruff check src/quant/analytics/pareto_sorting.py` with zero warnings.
2. **Code Formatting**: `ruff format --check src/quant/analytics/pareto_sorting.py` enforcing 100-character line width.
3. **Strict Type Checking**: `mypy src/quant/analytics/pareto_sorting.py` in strict mode (`disallow_untyped_defs = true`, `no_implicit_optional = true`) with zero errors.
4. **Comprehensive Test Suite**: Target **$\ge 95\%$ line coverage** on `pareto_sorting.py` with unit tests in `tests/unit/test_pareto_sorting.py` covering:
   * Empty archive initialization (Gen 0 fallback to $\rho_{\text{ortho}} = 1.0$).
   * Multi-collinear linear combination strategy detection (proving $\rho_{\text{ortho}} = 0.0$ for $e_A = 0.5 e_1 + 0.5 e_2$ where pairwise correlation failed).
   * Memmel–Ledoit–Wolf dependent standard error vs naive independent standard error under varying return correlations ($\rho \in [0.0, 0.95]$).
   * Viability gating (rejection of unviable novelty parasites).
   * Boundary ray immutability across 50 adaptation cycles.
   * Angle-Penalized Distance escalation across early, mid, and late generations.
   * Extreme edge cases (zero variance, identical candidates, $100\%$ infeasible populations).
