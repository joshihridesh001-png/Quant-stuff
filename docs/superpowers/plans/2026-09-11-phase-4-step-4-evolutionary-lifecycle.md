# Implementation Plan: Phase 4, Step 4 — $(\mu + \lambda)$ APD-Adaptive Evolutionary Lifecycle Engine

**Goal:** Implement the closed-loop generational evolution engine of the Quantitative Prediction Engine, featuring $(\mu + \lambda)$ environmental Pareto selection, self-adaptive truncated Cauchy mutation in unit hypercube space, continuous APD-progress Rechenberg volatility adaptation, gene-family sensitivity scaling, dual-space stagnation monitoring, and cataclysmic re-diversification.

---

## Architecture & Interfaces

- **File**: `src/quant/analytics/evolutionary_lifecycle.py`
- **Consumes**:
  - `StrategyChromosome`, `ChromosomeVectorCodec` (`chromosomes.py`)
  - `BoundaryAnchoredRVEARanker`, `CandidateFitness`, `RankingResult`, `ParetoFront` (`pareto_sorting.py`)
  - `HypergamicSelectionEngine`, `HypergamicConfig`, `ParetoCohortStratifier`, `HypergamicPartnerMatcher`, `AsymmetricLatentCrossover` (`hypergamic_selection.py`)
- **Produces**:
  - `MutationConfig`, `GenerationalState`, `LifecycleStepResult`
  - `AdaptiveVolatilityMutator`
  - `StagnationDetector`
  - `GenerationalLifecycleEngine`
  - Exceptions: `LifecycleError`, `InvalidGenerationalStateException`, `StagnationException`

---

## Step-by-Step Implementation Tasks

### Task 1: Domain Entities, Config, and Defensive Invariants
- **Files:**
  - Create `src/quant/analytics/evolutionary_lifecycle.py`
  - Create `tests/unit/test_evolutionary_lifecycle.py`
- **Steps:**
  1. Write failing tests in `TestDomainEntitiesAndInvariants`:
     - `test_mutation_config_defaults_and_validation`: domain boundary checks on `initial_step_size`, `min_step_size`, `max_step_size`, `mutation_probability`, etc.
     - `test_generational_state_immutability`: frozen dataclass mutation protection.
     - `test_lifecycle_step_result_structure`: verified attributes.
  2. Implement `MutationConfig`, `GenerationalState`, `LifecycleStepResult`, `LifecycleError`, `InvalidGenerationalStateException`, `StagnationException`.
  3. Verify tests pass.
  4. Commit Task 1.

### Task 2: Truncated Cauchy Mutator with Gene-Family Scaling & Mirror Reflection
- **Files:**
  - Modify `src/quant/analytics/evolutionary_lifecycle.py`
  - Modify `tests/unit/test_evolutionary_lifecycle.py`
- **Steps:**
  1. Write failing tests in `TestAdaptiveVolatilityMutator`:
     - `test_mutation_satisfies_all_chromosome_invariants`: 100 mutations yield 100% valid `StrategyChromosome` ($\tau_{\text{fast}} < \tau_{\text{slow}}$, valid simplex $\mathbf{p} \in \Delta^3$, bounded parameters).
     - `test_mirror_boundary_reflection_eliminates_edge_clamping`: perturbations crossing $0.0$ or $1.0$ reflect back smoothly without edge clumping.
     - `test_gene_family_differential_scaling`: verify variance of risk genes is strictly smaller than search genes: $Var(\Delta u_{\text{risk}}) < Var(\Delta u_{\text{search}})$.
     - `test_cataclysmic_hyper_mutation_step_size`: verifies enlarged perturbation when `is_cataclysmic=True`.
  2. Implement `AdaptiveVolatilityMutator.mutate()`.
  3. Verify tests pass.
  4. Commit Task 2.

### Task 3: Continuous APD-Progress Rechenberg Volatility Adaptation
- **Files:**
  - Modify `src/quant/analytics/evolutionary_lifecycle.py`
  - Modify `tests/unit/test_evolutionary_lifecycle.py`
- **Steps:**
  1. Write failing tests in `TestRechenbergAdaptation`:
     - `test_rechenberg_expands_on_high_apd_success`: success ratio $r > 0.20$ expands step size by factor $\gamma_{\text{expand}} = 1.10$.
     - `test_rechenberg_contracts_on_low_apd_success`: success ratio $r < 0.20$ contracts step size by factor $\gamma_{\text{contract}} = 0.90$.
     - `test_rechenberg_clamps_to_step_size_bounds`: extreme success or failure series clamp step size within $[\sigma_{\min}, \sigma_{\max}]$.
     - `test_rechenberg_smoothing_momentum`: verifies exponential smoothing of the success ratio.
  2. Implement `AdaptiveVolatilityMutator.adapt_step_size()` and `compute_apd_success_ratio()`.
  3. Verify tests pass.
  4. Commit Task 3.

### Task 4: Dual-Space Phenotypic & Residual Stagnation Detector
- **Files:**
  - Modify `src/quant/analytics/evolutionary_lifecycle.py`
  - Modify `tests/unit/test_evolutionary_lifecycle.py`
- **Steps:**
  1. Write failing tests in `TestStagnationDetector`:
     - `test_stagnation_evaluates_param_diversity_and_residual_correlation`: computes $\bar{D}_{\text{param}}$ and $\bar{\rho}_{\text{pop}}$ accurately.
     - `test_stagnation_counter_increments_only_when_both_bind`: counter only increments when $\bar{D} < \epsilon \land \bar{\rho} > \rho_{\text{thresh}}$.
     - `test_cataclysm_triggers_after_limit_generations`: reaching limit triggers cataclysm flag and resets counter.
     - `test_diverse_population_resets_stagnation_counter`: diverse population resets counter to 0.
  2. Implement `StagnationDetector`.
  3. Verify tests pass.
  4. Commit Task 4.

### Task 5: Master $(\mu + \lambda)$ Generational Lifecycle Engine
- **Files:**
  - Modify `src/quant/analytics/evolutionary_lifecycle.py`
  - Modify `src/quant/analytics/__init__.py`
  - Modify `tests/unit/test_evolutionary_lifecycle.py`
- **Steps:**
  1. Write failing tests in `TestGenerationalLifecycleEngine`:
     - `test_lifecycle_step_conserves_population_size`: $N_{t+1} == N_t$ exactly (`INV-LIFE-001`).
     - `test_mu_plus_lambda_preserves_monotonic_pareto_frontier`: Front-1 hypervolume / dominance never degrades (`INV-LIFE-003`).
     - `test_multi_generational_evolution_closed_loop`: 5 complete generations run deterministically with state propagation.
     - `test_lifecycle_benchmark_sub_35ms`: 100 individuals take $< 35\text{ms}$ per generation.
  2. Implement `GenerationalLifecycleEngine.step_generation()` and `initialize_state()`.
  3. Export in `src/quant/analytics/__init__.py`.
  4. Verify all tests pass.
  5. Commit Task 5.

### Task 6: Documentation Synchronization & Quality Gates
- Update `Memory.md` (ADR-017, Fault Vectors `ERR-EVO-LIFE-001` through `ERR-EVO-LIFE-004`, Phase 25 Audit Trail).
- Update `Phases.md` (Mark Phase 4 Step 4 and Phase 4 overall as `[COMPLETE]`).
- Update `Architecture.md` (Section 3 module inventory and Section 4.15 data flow pipeline).
- Update `PRD.md` (Evolutionary Manager Phase 4 as Complete).
- Run full quality gates:
  - `ruff check .`
  - `ruff format --check .`
  - `mypy src --strict`
  - `python -m coverage run -m pytest`
  - `python -m coverage report`
- Commit Task 6.
