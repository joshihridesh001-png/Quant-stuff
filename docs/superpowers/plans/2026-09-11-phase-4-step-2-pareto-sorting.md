# Phase 4, Step 2: Boundary-Anchored Adaptive RVEA with SVD Subspace Orthogonality (BA-ARVEA-SO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the multi-objective Pareto optimization engine for evolutionary alpha search using Memmel–Ledoit–Wolf dependent asymptotic dominance, SVD subspace orthogonal novelty, and boundary-anchored adaptive reference rays.

**Architecture:** Stateless analytical module in `src/quant/analytics/pareto_sorting.py` exposing four decoupled components (`SVDSubspaceOrthogonalArchive`, `AdaptiveReferenceLattice`, `DependentNonDominatedSorter`, and `BoundaryAnchoredRVEARanker` facade) operating on immutable typed dataclasses and contiguous NumPy float64 tensors.

**Tech Stack:** Python 3.12+, NumPy, SciPy (linear algebra, thin SVD), Pytest, Mypy (strict), Ruff.

**Spec:** [`docs/superpowers/specs/2026-09-11-phase-4-step-2-pareto-sorting-design.md`](file:///c:/Users/jishu/OneDrive/Documents/quant/docs/superpowers/specs/2026-09-11-phase-4-step-2-pareto-sorting-design.md)

## Global Constraints
* Every line of source code must strictly comply with `Rules.md` (Micro-level transparency, boundary invariant annotations, zero-execution diagnostic readiness).
* Strict type annotations across all functions and classes (`disallow_untyped_defs = true`, `no_implicit_optional = true`).
* Line width formatted to 100 characters max (`ruff format`).
* Test coverage on `src/quant/analytics/pareto_sorting.py` must achieve $\ge 95\%$.
* Zero regressions on the existing 291 tests in the project test suite.

---

### Task 1: Core Domain Entities, Contracts & Exception Hierarchy

**Files:**
- Create: `src/quant/analytics/pareto_sorting.py`
- Test: `tests/unit/test_pareto_sorting.py`

**Interfaces:**
- Produces: `CandidateFitness`, `ParetoFront`, `RankingResult`, `CorruptedFitnessException`, `InvalidResidualException`, `DegenerateLatticeException`.

- [ ] **Step 1: Write the failing test for domain contracts and invariant validations**

```python
# tests/unit/test_pareto_sorting.py
import numpy as np
import pytest
from quant.analytics.pareto_sorting import (
    CandidateFitness,
    ParetoFront,
    RankingResult,
    CorruptedFitnessException,
    InvalidResidualException,
)

def test_candidate_fitness_valid_instantiation():
    returns = np.random.randn(200).astype(np.float64)
    residuals = np.random.randn(200).astype(np.float64)
    fit = CandidateFitness(
        candidate_id="cand_001",
        dsr=1.25,
        minimax_regret=0.045,
        return_series=returns,
        residual_series=residuals,
        backtest_length=200,
        is_feasible=True,
    )
    assert fit.candidate_id == "cand_001"
    assert fit.dsr == 1.25
    assert fit.minimax_regret == 0.045
    assert len(fit.return_series) == 200

def test_candidate_fitness_rejects_nan():
    returns = np.array([np.nan, 0.01, 0.02], dtype=np.float64)
    residuals = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    with pytest.raises(CorruptedFitnessException):
        CandidateFitness(
            candidate_id="cand_bad",
            dsr=1.0,
            minimax_regret=0.01,
            return_series=returns,
            residual_series=residuals,
            backtest_length=3,
            is_feasible=True,
        )

def test_candidate_fitness_rejects_mismatched_series_length():
    returns = np.ones(150, dtype=np.float64)
    residuals = np.ones(140, dtype=np.float64)
    with pytest.raises(InvalidResidualException):
        CandidateFitness(
            candidate_id="cand_bad_len",
            dsr=1.0,
            minimax_regret=0.01,
            return_series=returns,
            residual_series=residuals,
            backtest_length=150,
            is_feasible=True,
        )
```

- [ ] **Step 2: Run pytest to verify test failure**
Run: `pytest tests/unit/test_pareto_sorting.py`
Confirm failure with `ModuleNotFoundError: No module named 'quant.analytics.pareto_sorting'`.

- [ ] **Step 3: Implement domain contracts and boundary invariant checks in `src/quant/analytics/pareto_sorting.py`**
Implement `CandidateFitness`, `ParetoFront`, `RankingResult`, and custom exceptions with explicit `__post_init__` checks for finiteness, non-empty contiguous arrays, and matching lengths.

- [ ] **Step 4: Run pytest and ruff to verify clean execution**
Run: `pytest tests/unit/test_pareto_sorting.py -v`
Run: `ruff check src/quant/analytics/pareto_sorting.py tests/unit/test_pareto_sorting.py`
Run: `mypy src/quant/analytics/pareto_sorting.py`

- [ ] **Step 5: Commit Task 1**
Run: `git add src/quant/analytics/pareto_sorting.py tests/unit/test_pareto_sorting.py && git commit -m "feat(evo): add domain entities and invariant contracts for pareto sorting"`

---

### Task 2: SVD Subspace Orthogonal Archive

**Files:**
- Modify: `src/quant/analytics/pareto_sorting.py`
- Test: `tests/unit/test_pareto_sorting.py`

**Interfaces:**
- Produces: `SVDSubspaceOrthogonalArchive`
  - `compute_novelty(residual_batch: np.ndarray, viability_mask: np.ndarray) -> np.ndarray`
  - `admit(candidate_id: str, residual_series: np.ndarray, novelty_score: float, dsr: float) -> bool`
  - `archive_size -> int`, `subspace_rank -> int`

- [ ] **Step 1: Write failing tests for SVD Subspace Orthogonality**
Cover:
* Empty archive returns novelty `1.0` for viable candidates, `0.0` for unviable candidates.
* SVD basis projection captures multiple collinearity: a candidate constructed as `e_A = 0.6 * e_1 + 0.8 * e_2` yields $\rho_{\text{ortho}} \approx 0.0$ (detected as redundant).
* Truly orthogonal candidate yields $\rho_{\text{ortho}} \approx 1.0$.
* FIFO buffer eviction when capacity exceeds $N_{\text{max}} = 500$.

- [ ] **Step 2: Run pytest to confirm failure**
Run: `pytest tests/unit/test_pareto_sorting.py -k "test_svd"`

- [ ] **Step 3: Implement `SVDSubspaceOrthogonalArchive` in `src/quant/analytics/pareto_sorting.py`**
Implement thin SVD basis computation via `np.linalg.svd(..., full_matrices=False)`, projection matrix $\mathbf{V}_K \mathbf{V}_K^T$, and variance-ratio calculation ($1 - R^2$). Include defensive zero-variance handling.

- [ ] **Step 4: Run test suite and type check**
Run: `pytest tests/unit/test_pareto_sorting.py -v`
Run: `mypy src/quant/analytics/pareto_sorting.py`

- [ ] **Step 5: Commit Task 2**
Run: `git add src/quant/analytics/pareto_sorting.py tests/unit/test_pareto_sorting.py && git commit -m "feat(evo): implement SVD subspace orthogonal archive with multi-collinear detection"`

---

### Task 3: Boundary-Anchored Adaptive Reference Lattice

**Files:**
- Modify: `src/quant/analytics/pareto_sorting.py`
- Test: `tests/unit/test_pareto_sorting.py`

**Interfaces:**
- Produces: `AdaptiveReferenceLattice`
  - `generate_das_dennis_rays(m: int = 3, p: int = 6) -> np.ndarray`
  - `associate_and_penalize(normalized_objectives: np.ndarray, generation_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `adapt_interior_rays(normalized_objectives: np.ndarray, associations: np.ndarray) -> int`
  - `rays -> np.ndarray` (28 x 3 unit vectors)

- [ ] **Step 1: Write failing tests for reference lattice & APD**
Cover:
* Das-Dennis generation with $M=3, p=6$ produces exactly 28 unit-normalized rays.
* The 3 coordinate basis rays $([1,0,0], [0,1,0], [0,0,1])$ remain identical before and after adaptation (`INV-PAR-004`).
* Idle interior rays adaptively migrate toward cluster centroids when idle for $\ge 3$ generations.
* APD score calculation escalates penalty with increasing `generation_ratio` ($t / t_{\max}$).

- [ ] **Step 2: Run pytest to confirm failure**
Run: `pytest tests/unit/test_pareto_sorting.py -k "test_lattice"`

- [ ] **Step 3: Implement `AdaptiveReferenceLattice`**
Implement recursive Das-Dennis simplex generator, unit normalizations, acute angle cosine projections, localized $\gamma_j$ computation, and interior ray migration with `gamma_min = 0.05` dispersion preservation.

- [ ] **Step 4: Run test suite and static analysis**
Run: `pytest tests/unit/test_pareto_sorting.py -v`
Run: `ruff check src/quant/analytics/pareto_sorting.py`
Run: `mypy src/quant/analytics/pareto_sorting.py`

- [ ] **Step 5: Commit Task 3**
Run: `git add src/quant/analytics/pareto_sorting.py tests/unit/test_pareto_sorting.py && git commit -m "feat(evo): implement boundary-anchored adaptive reference lattice (BA-ARVEA)"`

---

### Task 4: Memmel–Ledoit–Wolf Dependent Non-Dominated Sorter

**Files:**
- Modify: `src/quant/analytics/pareto_sorting.py`
- Test: `tests/unit/test_pareto_sorting.py`

**Interfaces:**
- Produces: `DependentNonDominatedSorter`
  - `compute_dependent_dsr_variance(dsr_a: float, dsr_b: float, returns_a: np.ndarray, returns_b: np.ndarray) -> float`
  - `dominates(a: CandidateFitness, b: CandidateFitness, obj_a: np.ndarray, obj_b: np.ndarray) -> bool`
  - `sort(candidates: Sequence[CandidateFitness], objective_matrix: np.ndarray) -> Tuple[List[List[int]], List[int]]`

- [ ] **Step 1: Write failing tests for dependent dominance & ENS-SS sorting**
Cover:
* Memmel–Ledoit–Wolf formula produces strictly smaller variance when return correlation $\rho = 0.90$ vs $\rho = 0.0$.
* Strategy $A$ dominating $B$ when correlation is high, whereas independent testing fails to detect dominance.
* Infeasible candidates ($\text{DSR} < 0.50$ or $T < \text{MinBTL}$) are correctly partitioned into `infeasible_indices`.
* Partition conservation: all candidate indices partitioned into fronts or infeasible set with zero loss or duplicates.

- [ ] **Step 2: Run pytest to confirm failure**
Run: `pytest tests/unit/test_pareto_sorting.py -k "test_dependent_dominance"`

- [ ] **Step 3: Implement `DependentNonDominatedSorter`**
Implement the Memmel-Ledoit-Wolf asymptotic covariance formula with numerical correlation clamping (`np.clip(rho, -0.9999, 0.9999)`), Deb's feasibility dominance, and Efficient Non-Dominated Sort with Sequential Search (ENS-SS).

- [ ] **Step 4: Run tests and type check**
Run: `pytest tests/unit/test_pareto_sorting.py -v`
Run: `mypy src/quant/analytics/pareto_sorting.py`

- [ ] **Step 5: Commit Task 4**
Run: `git add src/quant/analytics/pareto_sorting.py tests/unit/test_pareto_sorting.py && git commit -m "feat(evo): implement Memmel-Ledoit-Wolf dependent non-dominated sorter"`

---

### Task 5: Facade Integration, Export & Performance Benchmarking

**Files:**
- Modify: `src/quant/analytics/pareto_sorting.py`
- Modify: `src/quant/analytics/__init__.py`
- Test: `tests/unit/test_pareto_sorting.py`

**Interfaces:**
- Produces: `BoundaryAnchoredRVEARanker`
  - `rank_population(candidates: Sequence[CandidateFitness], generation: int, max_generations: int, auto_admit: bool = True) -> RankingResult`
- Exported in `src/quant/analytics/__init__.py`.

- [ ] **Step 1: Write failing end-to-end integration tests**
Cover:
* Complete ranking of a population of 50 candidates across multiple simulated generations.
* Auto-admission of Front-1 individuals into the SVD archive.
* Execution latency benchmark: 100 candidates with 500-bar series evaluated and ranked in $<15\,\text{ms}$.
* Edge case: $100\%$ infeasible population gracefully handled with empty fronts and all IDs in `infeasible_ids`.

- [ ] **Step 2: Run pytest to confirm failure**
Run: `pytest tests/unit/test_pareto_sorting.py -k "test_ranker_end_to_end"`

- [ ] **Step 3: Implement `BoundaryAnchoredRVEARanker` facade and export in `__init__.py`**
Wire together normalization, archive scoring, lattice association, ENS-SS sorting, APD ranking, and auto-admission.

- [ ] **Step 4: Run the complete unit test suite and check coverage**
Run: `pytest tests/unit/test_pareto_sorting.py --cov=quant.analytics.pareto_sorting --cov-report=term-missing`
Ensure coverage is $\ge 95\%$.

- [ ] **Step 5: Commit Task 5**
Run: `git add src/quant/analytics/pareto_sorting.py src/quant/analytics/__init__.py tests/unit/test_pareto_sorting.py && git commit -m "feat(evo): implement BoundaryAnchoredRVEARanker facade and exports"`

---

### Task 6: Documentation Synchronization & CI Quality Gates (Rules 2 & 3 Compliance)

**Files:**
- Modify: `Memory.md`
- Modify: `Phases.md`
- Modify: `Architecture.md`

- [ ] **Step 1: Document ADR-023 and Fault Vector Matrix in `Memory.md`**
Log ADR-023 detailing the mathematical rationale for Memmel-Ledoit-Wolf covariance and SVD subspace orthogonality. Add Fault Vectors `ERR-EVO-PAR-001` through `ERR-EVO-PAR-004`.

- [ ] **Step 2: Update `Phases.md` and `Architecture.md`**
Update Phase 4 Step 2 status to `[COMPLETE]` and link the new module coordinates and benchmark results.

- [ ] **Step 3: Run the full project verification gates**
Run: `ruff check .`
Run: `ruff format --check .`
Run: `mypy src`
Run: `pytest --cov=quant --cov-fail-under=85`

- [ ] **Step 4: Final commit for Step 2**
Run: `git add Memory.md Phases.md Architecture.md && git commit -m "docs(evo): synchronize Phase 4 Step 2 delivery, ADR-023, and diagnostic matrix"`
