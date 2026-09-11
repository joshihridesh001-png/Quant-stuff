# Phase 4, Step 3: Hypergamic Assortative Selection & Residual Orthogonality Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the institutional-grade hypergamic assortative mating engine, featuring front-preserving Pareto stratification, bidirectional absolute residual orthogonality gating, bounded adaptive relaxation, and asymmetric latent-hypercube crossover.

**Architecture:** The subsystem consumes Step 1's `StrategyChromosome` / `ChromosomeVectorCodec` and Step 2's `RankingResult` / `CandidateFitness`. It stratifies the population into Alpha and Aspirant cohorts, matches pairs through a bidirectional residual orthogonality gate with adaptive deadlock recovery, and breeds offspring via asymmetric Simulated Binary Crossover (SBX) in the normalized unit hypercube $[0, 1]^{20}$, strictly preserving institutional risk parameters while injecting fresh alpha signals.

**Tech Stack:** Python 3.13, NumPy, SciPy, dataclasses, pytest, pytest-cov, ruff, mypy.

**Spec:** [`docs/superpowers/specs/2026-09-11-phase-4-step-3-hypergamic-selection-design.md`](file:///c:/Users/jishu/OneDrive/Documents/quant/docs/superpowers/specs/2026-09-11-phase-4-step-3-hypergamic-selection-design.md)

## Global Constraints
- Python 3.13+ compatibility.
- Zero floating-point NaN or Inf propagation.
- No external heavy dependencies outside NumPy, SciPy, and existing project packages.
- Strict static typing with zero `Any` escapes under `mypy src --strict`.
- Minimum test coverage $\ge 90\%$ on `hypergamic_selection.py` with zero regressions across the 317 existing project tests.
- Execution latency ceiling: $< 25\text{ms}$ for generating 100 offspring from 100 candidates.

---

### Task 1: Domain Entities, Configuration, and Defensive Invariants

**Files:**
- Create: `src/quant/analytics/hypergamic_selection.py`
- Test: `tests/unit/test_hypergamic_selection.py`

**Interfaces:**
- Consumes: `StrategyChromosome` from `src.quant.analytics.chromosomes`
- Produces: `HypergamicConfig`, `MatingPair`, `OffspringResult`, `HypergamicSelectionError`, `InvalidCohortException`, `MatingDeadlockException`

- [ ] **Step 1: Write the failing tests for domain entities and configuration**

```python
# tests/unit/test_hypergamic_selection.py
import pytest
from quant.analytics.hypergamic_selection import (
    HypergamicConfig,
    HypergamicSelectionError,
    InvalidCohortException,
    MatingDeadlockException,
    MatingPair,
    OffspringResult,
)


def test_hypergamic_config_defaults_and_validation() -> None:
    cfg = HypergamicConfig()
    assert cfg.alpha_ratio == 0.25
    assert cfg.orthogonality_threshold == 0.30
    assert cfg.max_mating_attempts == 5
    assert cfg.elitism_count == 2

    # Reject out-of-bounds alpha_ratio
    with pytest.raises(HypergamicSelectionError):
        HypergamicConfig(alpha_ratio=0.0)
    with pytest.raises(HypergamicSelectionError):
        HypergamicConfig(alpha_ratio=1.0)

    # Reject out-of-bounds orthogonality_threshold
    with pytest.raises(HypergamicSelectionError):
        HypergamicConfig(orthogonality_threshold=-0.1)
    with pytest.raises(HypergamicSelectionError):
        HypergamicConfig(orthogonality_threshold=1.05)


def test_mating_pair_and_offspring_result_immutability() -> None:
    pair = MatingPair(
        alpha_id="alpha_1",
        aspirant_id="asp_2",
        residual_correlation=0.12,
        is_relaxed=False,
        relaxation_level=0,
    )
    assert pair.alpha_id == "alpha_1"
    assert pair.residual_correlation == 0.12

    with pytest.raises(Exception):
        pair.alpha_id = "alpha_2"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hypergamic_selection.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'quant.analytics.hypergamic_selection'"

- [ ] **Step 3: Implement domain dataclasses and exceptions**

In `src/quant/analytics/hypergamic_selection.py`:
Implement `HypergamicConfig` with `__post_init__` invariant validation, `MatingPair`, `OffspringResult`, and custom exceptions.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_hypergamic_selection.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add src/quant/analytics/hypergamic_selection.py tests/unit/test_hypergamic_selection.py
git commit -m "feat(evo): define hypergamic selection domain entities and config (Phase 4 Step 3 Task 1)"
```

---

### Task 2: Front-Preserving Pareto Cohort Stratifier

**Files:**
- Modify: `src/quant/analytics/hypergamic_selection.py`
- Test: `tests/unit/test_hypergamic_selection.py`

**Interfaces:**
- Consumes: `CandidateFitness`, `RankingResult`, `HypergamicConfig` from `src.quant.analytics.pareto_sorting`
- Produces: `ParetoCohortStratifier.stratify(candidates, ranking) -> Tuple[List[CandidateFitness], List[CandidateFitness]]`

- [ ] **Step 1: Write failing tests for Pareto cohort stratification**

```python
# In tests/unit/test_hypergamic_selection.py
def test_stratification_preserves_front_1_when_large() -> None:
    """When Front 1 exceeds alpha_ratio * N, all of Front 1 is preserved."""
    ...


def test_stratification_pads_from_front_2_when_small() -> None:
    """When Front 1 is small, pads from top APD candidates of Front 2."""
    ...


def test_stratification_excludes_infeasible_candidates() -> None:
    """Infeasible candidates are never admitted to Alpha or Aspirant cohorts."""
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hypergamic_selection.py -k test_stratification -v`
Expected: FAIL with "ImportError: cannot import name 'ParetoCohortStratifier'"

- [ ] **Step 3: Implement ParetoCohortStratifier**

Implement `ParetoCohortStratifier`:
1. Build index map from candidate ID to `CandidateFitness`.
2. Extract viable candidate IDs from Front 1 and Front 2.
3. If $|Front_1| \ge N_\alpha$, $\mathcal{A} = Front_1$.
4. If $|Front_1| < N_\alpha$, fill up to $N_\alpha$ using Front 2 ordered by APD score.
5. $\mathcal{X} = $ all other viable candidates.
6. Verify no overlap between $\mathcal{A}$ and $\mathcal{X}$ and no infeasible candidates admitted.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_hypergamic_selection.py -k test_stratification -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add src/quant/analytics/hypergamic_selection.py tests/unit/test_hypergamic_selection.py
git commit -m "feat(evo): implement front-preserving Pareto cohort stratifier (Phase 4 Step 3 Task 2)"
```

---

### Task 3: Bidirectional Residual Orthogonality Gate

**Files:**
- Modify: `src/quant/analytics/hypergamic_selection.py`
- Test: `tests/unit/test_hypergamic_selection.py`

**Interfaces:**
- Consumes: `CandidateFitness`, `HypergamicConfig`
- Produces: `ResidualOrthogonalityGate.evaluate(alpha, aspirant, relaxation_level) -> Tuple[bool, float]`

- [ ] **Step 1: Write failing tests for residual orthogonality gate**

```python
# In tests/unit/test_hypergamic_selection.py
def test_gate_rejects_positive_collinear_clones() -> None:
    """Corr = 0.95 fails gate (unexplained variance only 5%)."""
    ...


def test_gate_rejects_negative_inverse_clones() -> None:
    """Corr = -0.95 fails gate (catches inverse clones that naive signed corr passes)."""
    ...


def test_gate_accepts_orthogonal_residuals() -> None:
    """Corr = 0.10 passes gate (90% unexplained variance)."""
    ...


def test_gate_adaptive_relaxation_lowers_threshold() -> None:
    """Relaxation level > 0 multiplies threshold by relaxation_factor."""
    ...


def test_gate_zero_variance_residuals_safe_rejection() -> None:
    """Flat zero-variance series rejects gracefully with corr = 1.0."""
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hypergamic_selection.py -k test_gate -v`
Expected: FAIL with "ImportError: cannot import name 'ResidualOrthogonalityGate'"

- [ ] **Step 3: Implement ResidualOrthogonalityGate**

Implement `ResidualOrthogonalityGate`:
1. Standardize residuals: $z_A = (e_A - \bar{e}_A) / \|e_A - \bar{e}_A\|_2$.
2. Guard against zero norm ($\|e - \bar{e}\| < 10^{-12} \implies \text{rejected}$).
3. Calculate sample Pearson correlation: $\rho = \langle z_A, z_B \rangle$.
4. Clamp $\rho \in [-1.0, 1.0]$.
5. Compute effective threshold: $\delta_{\text{eff}} = \delta_{\text{ortho}} \cdot (\gamma_{\text{relax}})^{\text{level}}$.
6. Acceptance criterion: $1.0 - |\rho| \ge \delta_{\text{eff}}$.
7. Return `(is_accepted, rho)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_hypergamic_selection.py -k test_gate -v`
Expected: PASS

- [ ] **Step 5: Commit Task 3**

```bash
git add src/quant/analytics/hypergamic_selection.py tests/unit/test_hypergamic_selection.py
git commit -m "feat(evo): implement bidirectional residual orthogonality gate (Phase 4 Step 3 Task 3)"
```

---

### Task 4: Assortative Hypergamic Partner Matcher

**Files:**
- Modify: `src/quant/analytics/hypergamic_selection.py`
- Test: `tests/unit/test_hypergamic_selection.py`

**Interfaces:**
- Consumes: `ParetoCohortStratifier`, `ResidualOrthogonalityGate`, `HypergamicConfig`
- Produces: `HypergamicPartnerMatcher.match_pairs(alphas, aspirants, target_pair_count) -> List[MatingPair]`

- [ ] **Step 1: Write failing tests for hypergamic partner matching**

```python
# In tests/unit/test_hypergamic_selection.py
def test_matcher_pairs_alphas_with_diverse_aspirants() -> None:
    """Nominal population matches target_pair_count pairs."""
    ...


def test_matcher_adaptive_relaxation_on_collinear_cohort() -> None:
    """When all aspirants are collinear, matcher relaxes gate and succeeds without deadlock."""
    ...


def test_matcher_phenotypic_distance_fallback_on_exhaustion() -> None:
    """When retries exceed 2 * max_attempts, falls back to max phenotypic distance."""
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hypergamic_selection.py -k test_matcher -v`
Expected: FAIL with "ImportError: cannot import name 'HypergamicPartnerMatcher'"

- [ ] **Step 3: Implement HypergamicPartnerMatcher**

Implement `HypergamicPartnerMatcher`:
1. Loop for $i \in \{0, \dots, N_{\text{pairs}} - 1\}$:
2. Select Alpha $A \in \mathcal{A}$ via round-robin or APD-weighted sampling.
3. Run tournament of size $k$ among Aspirants $\mathcal{X}$.
4. Submit $(A, B)$ to `ResidualOrthogonalityGate`.
5. If accepted, add `MatingPair(A, B, ...)`.
6. If rejected, retry up to $K_{\max}$ with adaptive relaxation level.
7. If retries exceed $2 \times K_{\max}$, compute phenotypic distance in unit hypercube and pick most distant candidate $B^*$.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_hypergamic_selection.py -k test_matcher -v`
Expected: PASS

- [ ] **Step 5: Commit Task 4**

```bash
git add src/quant/analytics/hypergamic_selection.py tests/unit/test_hypergamic_selection.py
git commit -m "feat(evo): implement assortative hypergamic partner matcher with deadlock fallback (Phase 4 Step 3 Task 4)"
```

---

### Task 5: Asymmetric Latent Unit-Hypercube Crossover & Master Facade

**Files:**
- Modify: `src/quant/analytics/hypergamic_selection.py`
- Modify: `src/quant/analytics/__init__.py`
- Test: `tests/unit/test_hypergamic_selection.py`

**Interfaces:**
- Consumes: `StrategyChromosome`, `ChromosomeVectorCodec`, `MatingPair`
- Produces: `AsymmetricLatentCrossover`, `HypergamicSelectionEngine.reproduce(candidates, ranking, chromosome_map, target_population_size) -> OffspringResult`

- [ ] **Step 1: Write failing tests for crossover and master facade**

```python
# In tests/unit/test_hypergamic_selection.py
def test_asymmetric_crossover_satisfies_all_invariants() -> None:
    """100 crossovers produce 100% valid chromosomes with tau_fast < tau_slow and valid simplices."""
    ...


def test_asymmetric_crossover_inherits_alpha_risk_bias() -> None:
    """Risk and game genes match Alpha parent significantly more often than Aspirant parent."""
    ...


def test_master_facade_end_to_end_reproduction() -> None:
    """HypergamicSelectionEngine reproduces target population size preserving elitism."""
    ...


def test_master_facade_performance_benchmark_sub_25ms() -> None:
    """100 candidates produce 100 offspring in < 25ms."""
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hypergamic_selection.py -k test_asymmetric -v`
Expected: FAIL with "ImportError: cannot import name 'HypergamicSelectionEngine'"

- [ ] **Step 3: Implement AsymmetricLatentCrossover and HypergamicSelectionEngine**

Implement:
1. `AsymmetricLatentCrossover`:
   - Unit hypercube encode via `ChromosomeVectorCodec`.
   - SBX spread calculation with gene-family bias.
   - Clamp to $[0, 1]^{20}$ and decode.
2. `HypergamicSelectionEngine`:
   - Sort candidates into cohorts via `ParetoCohortStratifier`.
   - Preserve top $N_{\text{elite}}$ Front-1 champions directly into offspring.
   - Match remaining $N_{\text{target}} - N_{\text{elite}}$ pairs via `HypergamicPartnerMatcher`.
   - Breed offspring via `AsymmetricLatentCrossover`.
   - Package into `OffspringResult`.
3. Export in `src/quant/analytics/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_hypergamic_selection.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit Task 5**

```bash
git add src/quant/analytics/hypergamic_selection.py src/quant/analytics/__init__.py tests/unit/test_hypergamic_selection.py
git commit -m "feat(evo): implement asymmetric latent crossover and HypergamicSelectionEngine facade (Phase 4 Step 3 Task 5)"
```

---

### Task 6: Documentation Synchronization & Full Project Verification

**Files:**
- Modify: `Memory.md` (ADR-016, Fault Vectors `ERR-EVO-HYP-001` through `ERR-EVO-HYP-004`, Phase 24 Audit Trail)
- Modify: `Phases.md` (Mark Phase 4 Step 3 as `[COMPLETE]`)
- Modify: `Architecture.md` (Add `hypergamic_selection.py` component and Section 4.14 pipeline)

- [ ] **Step 1: Document ADR-016 and Fault Vectors in Memory.md**
- [ ] **Step 2: Update Phases.md and Architecture.md**
- [ ] **Step 3: Run CI Quality Gates**

```bash
ruff check .
ruff format --check .
mypy src --strict
python -m coverage run -m pytest
python -m coverage report
```

- [ ] **Step 4: Commit Task 6**

```bash
git add Memory.md Phases.md Architecture.md
git commit -m "docs: synchronize ADR-016, fault vectors, and Phase 4 Step 3 completion status"
```
