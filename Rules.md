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
