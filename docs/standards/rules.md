# The Three Fundamental Engineering Rules

This codebase is governed by three non-negotiable architectural and software engineering invariants. Every line of code, configuration, and technical deliverable must strictly adhere to these rules.

---

## Rule 1: Micro-Level Code Transparency & Defensive Invariants

Every unit of source code must be completely self-explicating. Comments must never merely restate the syntax; they must articulate the functional purpose, dependencies, structural coupling, and defensive invariants.

### 1.1 Line-by-Line Annotation Standards
* **Functional Purpose**: State the mathematical, financial, or systems rationale underlying the instruction.
* **Explicit Dependency Tracking**: Declare any external state, config parameters, hardware caches, or injected repositories relied upon.
* **Structural Relationship**: Document the downstream consumer, caller, or subscriber dependent upon this block's outputs or side-effects.
* **Defensive Invariant**: Declare the valid domain bounds (e.g., $High \ge \max(Open, Close)$, strictly positive prices, strictly positive nanosecond timestamps).

### 1.2 Boundary Invariant Enforcement
* Invariants must be enforced at the earliest possible boundary—specifically upon domain value object instantiation (e.g., `PriceBar.__post_init__`).
* Mathematically impossible or corrupt data (crossed candles, negative volume, NaN propagation) must be rejected immediately with explicit, descriptive domain exceptions.

---

## Rule 2: Zero-Execution Deterministic Diagnostics

During production incidents, performance degradations, or critical pipeline failures, engineers must be able to isolate, diagnose, and prescribe remediation for any defect **solely by consulting the technical documentation**, without executing code, launching debuggers, attaching profilers, or inspecting live memory spaces.

### 2.1 The Diagnostic Failure Matrix Protocol
Every operational failure mode across all modules must be cataloged in `Memory.md` following the standardized schema:
1. **Fault Vector ID**: Unique hierarchical identifier (e.g., `ERR-MKT-INVAR-001`, `ERR-MKT-STORAGE-002`).
2. **Component Coordinates**: Absolute file path, Class name, Method name, and exact line/block coordinates.
3. **Nominal Behavior**: Exact mathematical or system state expected during healthy operation.
4. **Malfunction Symptoms**: Visible HTTP status codes, error payload strings, log signatures, or metrics.
5. **Root Cause Etiology**: Complete breakdown of environmental, input-driven, or concurrency factors causing the fault.
6. **Zero-Runtime Verification**: Concrete static checks an engineer performs on configuration files, input feeds, or static source code to confirm the root cause without reproduction.
7. **Remediation Procedure**: Surgical patch instructions, configuration overrides, or rollback steps.
8. **Secondary Failure Modes**: Cascading downstream consequences if left unaddressed.

---

## Rule 3: Strict Quality Gates & Atomic Synchronization

Documentation is not a post-hoc byproduct; it is a compiled artifact subject to the identical quality standards as production binaries.

### 3.1 The Continuous Integration (CI) Invariants
Every pull request and commit must pass four automated verification gates:
1. **Static Linting**: `ruff check .` with zero errors or unhandled warnings.
2. **Code Formatting**: `ruff format --check .` enforcing 100-character line width standards.
3. **Static Type Checking**: `mypy src` in strict mode (`disallow_untyped_defs = true`, `no_implicit_optional = true`) with zero errors across all source files.
4. **Test Coverage Enforcement**: `pytest --cov=quant --cov-fail-under=85` requiring minimum **85% statement coverage** across unit, integration, and API test suites.

### 3.2 Atomic Lifecycle Synchronization
* **Zero Stale Documentation**: Any modification to code, dependencies, or configuration keys requires an atomic, simultaneous update to `PRD.md`, `Architecture.md`, `Phases.md`, `Design.md`, and `Memory.md`.
* **Autonomous Decision Registration**: Every architectural choice, design pattern selection, and library addition made without explicit user direction must be immediately logged as an ADR in `Memory.md`.

---

## Rule 4: Mandatory Adversarial Red-Teaming & "Best of the Best" First-Principles Architecture

No solution may be implemented simply because it is the "standard textbook" or common open-source consensus approach. Quantitative trading systems face an active, adversarial market. Every subsystem, algorithm, and mathematical formulation must be designed, audited, and verified under an uncompromising "best of the best" standard.

### 4.1 Mandatory "Kill-the-Idea" Pre-Execution Stress-Testing
Before writing or approving any design specification or implementation plan, agents must conduct a dedicated adversarial stress-test assuming hostile market conditions:
* **The Adversarial Market Assumption**: Assume extreme fat tails ($\gamma_4 > 3$), jump-diffusion price shocks, sudden illiquidity, order book depletion, non-zero execution delay, and exchange lot-size discretization.
* **First-Principles Scrutiny**: The plan must explain *why* the initial/textbook solution exists, expose its hidden structural failure modes, and formulate the mathematically superior alternative before any code is written.

### 4.2 The "Superior Alternative" Subagent Review Mandate
Subagents performing code or design reviews must NEVER act as passive syntax checkers or rubber-stamp reviewers. In every review report, the reviewer must explicitly answer:
1. **Mathematical & Conceptual Rigor**: Does this implementation rely on fragile heuristics or unexamined assumptions?
2. **Adversarial Failure Modes**: Under what extreme market regime, liquidity shock, or data feed corruption will this code fail?
3. **Superior Alternatives**: Propose the "best of the best" theoretical and algorithmic alternative, even if the current implementation already passes all automated tests.

### 4.3 Blacklist of Prohibited Naive Quantitative Shortcuts
The following fragile heuristics and mathematical shortcuts are strictly prohibited in the production codebase:
1. **No Raw Square-Root-of-Time ($\sqrt{H}$) Horizon Scaling in Fat-Tailed Regimes**: Brownian motion $\sqrt{H}$ variance scaling fails under Poisson jump-diffusions (where jump intensity scales linearly with $H$) and long-memory trending (where variance scales as $H^{2H_e}$). Multi-horizon risk must account for jump processes and fat-tailed stability.
2. **No Raw Cornish-Fisher Polynomials Under High Kurtosis**: Cornish-Fisher expansions invert and become non-monotonic when excess kurtosis $\gamma_4 > 3$, paradoxically predicting lower risk at higher confidence levels. Heavy-tailed modeling must use Semi-Parametric Extreme Value Theory (EVT) Peaks-Over-Threshold (POT) with Generalized Pareto Distribution (GPD).
3. **No Unconstrained or Heuristic Half-Kelly Bet Sizing**: Raw Kelly betting causes catastrophic drawdown cliffs under parameter estimation error ($\hat{p}, \hat{b}$). Fixed Half-Kelly ($\lambda = 0.50$) is an arbitrary scalar. Sizing must use convex risk-constrained optimization with dynamic parameter shrinkage against epistemic uncertainty ($\sigma^2_{\text{epistemic}}$) and non-linear market impact penalties ($\psi_{3/2}$).
4. **No Iterative Numerical Solvers (MLE / Root-Finding) in the Synchronous Hot Path**: Using `scipy.optimize` or iterative Maximum Likelihood Estimation inside the per-bar execution loop violates the sub-$0.20\text{ms}$ SLA and introduces non-convergence crashes. Hot-path tail estimation must use algebraic, closed-form estimators (e.g., Probability Weighted Moments / L-moments).
5. **No Value-at-Risk (VaR) as the Sole Capital Boundary**: VaR violates Artzner's subadditivity axiom ($\text{VaR}(A+B) > \text{VaR}(A) + \text{VaR}(B)$) and ignores loss severity beyond the threshold. Capital and drawdown constraints must enforce coherent, subadditive Expected Shortfall (CVaR).
6. **No Closed-Loop Realized Return Feedback During Trading Halts**: Risk and volatility estimators must track unconditional market and constituent model returns—never realized halted portfolio returns—to prevent the "Zero-Variance Collapse" feedback loop.
