# Design Specification: Phase 5, Step 2 — Epistemic Disagreement Entropy & Circuit Breaker Overlays

- **Phase:** 5 (Institutional Strategy Suite & Online DMA Aggregator)
- **Step:** 2 (Epistemic Disagreement Entropy & Circuit Breaker Overlays)
- **Target File:** `src/quant/analytics/circuit_breakers.py`
- **Test File:** `tests/unit/test_circuit_breakers.py`
- **Parent Subsystem:** Ensemble Alpha Aggregator & Risk Overlays

---

## 1. Executive Summary & Problem Formulation

In Phase 5 Step 1, the Regime-Conditioned Dynamic Model Averaging (RD-DMA) engine decomposed the total predictive uncertainty of $K$ candidate strategies into **aleatoric process variance** $\sigma^2_{\text{aleatoric}}$ and **epistemic model disagreement variance** $\sigma^2_{\text{epistemic}}$ via the Law of Total Variance:
$$\sigma^2_{\text{total}} = \sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}$$

While the ensemble point prediction $\hat{\mu}_t = \sum w_k \tilde{y}_k$ blends constituent signals, a critical institutional risk blindness emerges when constituent strategies violently diverge:
- **Consensus Equilibrium**: All models forecast $\tilde{y}_k \approx 0.0$ $\implies \hat{\mu}_t \approx 0.0, \sigma^2_{\text{epistemic}} \approx 0.0$.
- **Bimodal Polarization**: 50% of model weight forecasts $+3.0\%$ (bull cascade) and 50% forecasts $-3.0\%$ (panic crash) $\implies \hat{\mu}_t \approx 0.0, \sigma^2_{\text{epistemic}} \gg 0.0$.

Treating these two states identically exposes an algorithmic trading system to catastrophic tail risk. Polarization indicates that the underlying market is undergoing a structural phase change or macroeconomic regime ambiguity where standard statistical arbitrage fails.

Phase 5 Step 2 implements **Epistemic Disagreement Entropy & Circuit Breaker Overlays**:
1. **Directional Consensus & Shannon Epistemic Entropy**: Information-theoretic quantification of model consensus vs. polarization.
2. **Smooth Continuous Haircut**: A continuous, sigmoid-damped sizing multiplier $\kappa_t \in [0.0, 1.0]$ preventing step-function boundary instability.
3. **Multi-Tier Hard Circuit Breakers (`NORMAL`, `CAUTION`, `DERISK`, `HALT`)**: Discrete institutional risk barriers responding to extreme entropy and thermodynamic turbulence.
4. **Hysteresis Anti-Chattering Protocol**: State-machine dwell time and asymmetric recovery barriers preventing rapid oscillation between active and halt states.
5. **Sub-0.20ms Execution SLA**: Ultra-low-latency vectorized NumPy execution for $K=100$ models.

---

## 2. Invariant Contracts

| Invariant ID | Formal Definition | Description |
| :--- | :--- | :--- |
| **`INV-CB-001`** | $\kappa_t \in [0.0, 1.0] \land \frac{\partial \kappa}{\partial \mathcal{H}} \le 0$ | Continuous haircut is strictly bounded in $[0, 1]$ and monotonically non-increasing in epistemic entropy. |
| **`INV-CB-002`** | $\text{State}_t \in \{\text{NORMAL}, \text{CAUTION}, \text{DERISK}, \text{HALT}\}$ | Discrete circuit breaker state is strictly one of the 4 defined institutional tiers with ordered severity. |
| **`INV-CB-003`** | $\text{State}_{t-1} \in \{\text{DERISK}, \text{HALT}\} \land \Delta t < \tau_{\text{dwell}} \implies \text{State}_t \ne \text{NORMAL}$ | Recovery requires minimum dwell time $\tau_{\text{dwell}}$ and shock $\Xi_t < \theta_{\text{recovery}}$ (anti-chattering hysteresis). |
| **`INV-CB-004`** | $\sum_{s \in \{+, 0, -\}} p_s \equiv 1.0 \pm 10^{-10} \land p_s \ge 0$ | Directional consensus probabilities strictly conserve the unit 3-simplex $\Delta^3$. |
| **`INV-CB-005`** | $\text{isnan}(x) \lor \text{isinf}(x) \implies \text{State}_t = \text{HALT} \land \text{raise DegenerateCircuitBreakerException}$ | Immediate defensive halt on non-finite input data. |
| **`INV-CB-006`** | $\text{Latency}_{\text{eval}} \le 0.20\text{ms}$ for $K=100$ | Vectorized evaluation of entropy, haircut, and state machine within $0.20\text{ms}$ SLA. |

---

## 3. Mathematical Foundations

### 3.1 Directional Consensus on the 3-Simplex
Given candidate model scaled predictions $\tilde{\mathbf{y}}_t = [\tilde{y}_1, \dots, \tilde{y}_K]^T$ and ensemble weights $\mathbf{w}_t \in \Delta^K$:
We define directional sign threshold $\delta_{\text{sign}} \ge 0$ (default $10^{-4}$). The weight allocation is partitioned into positive, negative, and neutral consensus:
$$p_+ = \sum_{k: \tilde{y}_k > \delta_{\text{sign}}} w_k, \quad p_- = \sum_{k: \tilde{y}_k < -\delta_{\text{sign}}} w_k, \quad p_0 = \sum_{k: |\tilde{y}_k| \le \delta_{\text{sign}}} w_k$$
Because $\sum_{k=1}^K w_k \equiv 1.0$ and $w_k > 0$, the vector $\mathbf{p}_{\text{dir}} = [p_+, p_-, p_0]^T$ strictly resides on the 3-simplex $\Delta^3$ (`INV-CB-004`).

The **Directional Shannon Entropy** is:
$$\mathcal{H}_{\text{dir}} = -\sum_{s \in \{+, 0, -\}} p_s \ln(p_s + \epsilon_{\text{log}}), \quad \epsilon_{\text{log}} = 10^{-30}$$
Normalized by maximum theoretical entropy $\ln 3$:
$$\tilde{\mathcal{H}}_{\text{dir}} = \frac{\mathcal{H}_{\text{dir}}}{\ln 3} \in [0.0, 1.0]$$
- $\tilde{\mathcal{H}}_{\text{dir}} = 0.0$: Total unanimous consensus (e.g. $p_+ = 1.0$).
- $\tilde{\mathcal{H}}_{\text{dir}} = 1.0$: Complete maximum directional confusion ($p_+ = p_- = p_0 = \frac{1}{3}$).

### 3.2 Epistemic Uncertainty Ratio & Composite Epistemic Entropy
The fraction of total risk attributable to model disagreement is:
$$\rho_{\text{epistemic}} = \frac{\sigma^2_{\text{epistemic}}}{\sigma^2_{\text{total}}} = \frac{\sigma^2_{\text{epistemic}}}{\sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}} \in [0.0, 1.0]$$

We formulate the **Composite Epistemic Disagreement Entropy**:
$$\mathcal{H}_{\text{epistemic}} = \tilde{\mathcal{H}}_{\text{dir}} \cdot \sqrt{\rho_{\text{epistemic}}} \in [0.0, 1.0]$$
This formulation has two crucial properties:
1. If all models agree on direction ($\tilde{\mathcal{H}}_{\text{dir}} \to 0$), $\mathcal{H}_{\text{epistemic}} \to 0$ even if predictive variance is elevated.
2. If total risk is dominated by process noise ($\sigma^2_{\text{epistemic}} \ll \sigma^2_{\text{aleatoric}} \implies \rho \to 0$), $\mathcal{H}_{\text{epistemic}} \to 0$ because model disagreement is negligible compared to background volatility.
3. $\mathcal{H}_{\text{epistemic}}$ spikes to $1.0$ **only** when models are polarized AND epistemic disagreement drives the uncertainty!

### 3.3 Thermodynamic Composite Shock Score ($\Xi_t$)
To prevent blind spots during sudden regime changes before all $K$ models update, we incorporate Phase 3's macroeconomic ambiguity temperature $\beta_t$:
$$\tilde{\beta}_t = \text{clip}\left(\frac{\beta_t - \beta_{\min}}{\beta_{\max} - \beta_{\min}}, 0.0, 1.0\right)$$
$$\Xi_t = \omega_\mathcal{H} \mathcal{H}_{\text{epistemic}} + \omega_\rho \rho_{\text{epistemic}} + \omega_\beta \tilde{\beta}_t$$
with default weights $\omega_\mathcal{H} = 0.50$, $\omega_\rho = 0.30$, $\omega_\beta = 0.20$ ($\sum \omega = 1.0 \implies \Xi_t \in [0.0, 1.0]$).

### 3.4 Continuous Soft Haircut Function
Instead of crude binary on/off switching, we compute a smooth, continuously differentiable position sizing haircut multiplier $\kappa_t \in [0.0, 1.0]$ using a generalized logistic sigmoid:
$$\kappa_t = \frac{1}{1 + \exp\left(k_{\text{steep}} \cdot (\Xi_t - \Xi_{\text{mid}})\right)}$$
Normalized so that $\kappa_t(\Xi=0) \approx 1.0$ and clamped:
$$\kappa_t = \text{clip}\left(\frac{\kappa_t - \kappa_{\min}}{1.0 - \kappa_{\min}}, 0.0, 1.0\right)$$
- Normal calm conditions ($\Xi_t \le 0.20$): $\kappa_t \approx 1.00$ (full capital efficiency).
- Moderate disagreement ($\Xi_t \approx 0.50$): $\kappa_t \approx 0.50$ (graceful proportional de-leveraging).
- Severe polarization ($\Xi_t \ge 0.80$): $\kappa_t \to 0.00$ (smoothly zeroes execution sizing without discrete slippage spikes).

### 3.5 Multi-Tier Hard Circuit Breakers & Hysteresis State Machine

Four institutional discrete operational tiers:

```
+-------------------------------------------------------------+
|                          HALT (Tier 3)                      |
|             Zero Execution, Cancel Active Orders            |
+-------------------------------------------------------------+
         ^                                           |
         | Extreme Shock (Xi >= 0.90)                 | tau >= tau_dwell AND
         | or CUSUM Shock in Panic                   | Xi < theta_recovery (0.30)
         |                                           v
+-------------------------------------------------------------+
|                         DERISK (Tier 2)                     |
|           Orderly Liquidation into Cash (kappa = 0)         |
+-------------------------------------------------------------+
         ^                                           |
         | Severe Disagreement (Xi >= 0.70)           | tau >= tau_dwell AND
         |                                           | Xi < theta_recovery
         |                                           v
+-------------------------------------------------------------+
|                        CAUTION (Tier 1)                     |
|         Throttle Sizing (kappa <= 0.50), Limit Orders       |
+-------------------------------------------------------------+
         ^                                           |
         | Moderate Disagreement (Xi >= 0.45)         | Xi < theta_recovery (0.35)
         |                                           v
+-------------------------------------------------------------+
|                         NORMAL (Tier 0)                     |
|                       Full Alpha Execution                  |
+-------------------------------------------------------------+
```

#### State Transition Logic:
1. **Escalation** (Instantaneous):
   - If $\Xi_t \ge \theta_{\text{halt}}$ (0.90) or CUSUM jump shock detected in Panic regime: transition to `HALT` immediately, reset dwell counter $\tau_{\text{dwell\_active}} = 0$.
   - Else if $\Xi_t \ge \theta_{\text{derisk}}$ (0.70): transition to `DERISK`, reset dwell counter.
   - Else if $\Xi_t \ge \theta_{\text{caution}}$ (0.45): transition to `CAUTION`.
2. **De-escalation / Recovery** (Hysteresis-Guarded):
   - When in `HALT` or `DERISK`: can only de-escalate if $\tau_{\text{dwell\_active}} \ge \tau_{\text{dwell}}$ (e.g. 5 bars) **AND** $\Xi_t < \theta_{\text{recovery}}$ (0.30).
   - Prevents boundary "chattering" (flipping on and off across consecutive bars) (`INV-CB-003`).

---

## 4. Architecture & Domain Entities

### 4.1 Enums & Domain Dataclasses

```python
class CircuitBreakerTier(IntEnum):
    NORMAL = 0
    CAUTION = 1
    DERISK = 2
    HALT = 3


@dataclass(frozen=True)
class CircuitBreakerConfig:
    sign_threshold: float = 1e-4
    caution_threshold: float = 0.45
    derisk_threshold: float = 0.70
    halt_threshold: float = 0.90
    recovery_threshold: float = 0.30
    dwell_time_bars: int = 5
    haircut_steepness: float = 10.0
    haircut_midpoint: float = 0.50
    weight_entropy: float = 0.50
    weight_epistemic_ratio: float = 0.30
    weight_ambiguity: float = 0.20
    beta_min: float = 1.0
    beta_max: float = 5.0


@dataclass(frozen=True)
class CircuitBreakerState:
    tier: CircuitBreakerTier
    active_bars_in_tier: int
    continuous_haircut: float  # kappa in [0.0, 1.0]
    epistemic_entropy: float  # H_epistemic in [0.0, 1.0]
    directional_entropy: float  # H_dir in [0.0, 1.0]
    epistemic_ratio: float  # rho_epistemic in [0.0, 1.0]
    composite_shock_score: float  # Xi_t in [0.0, 1.0]
    directional_probabilities: np.ndarray  # [p_+, p_-, p_0]
    step_index: int


@dataclass(frozen=True)
class CircuitBreakerDecision:
    action_tier: CircuitBreakerTier
    execution_haircut: float  # Final scalar to multiply target position size
    is_halted: bool
    is_derisking: bool
    is_throttled: bool
    state: CircuitBreakerState
```

### 4.2 Core Analytics Classes

1. **`EpistemicEntropyCalculator`**:
   - `compute_directional_consensus(predictions, weights) -> tuple[np.ndarray, float]`
   - `compute_epistemic_entropy(predictions, weights, aleatoric_var, epistemic_var) -> tuple[float, float, float]`
2. **`ContinuousHaircutCalculator`**:
   - `compute_haircut(composite_shock: float) -> float`
3. **`CircuitBreakerOverlayEngine` (Master Facade)**:
   - `initialize_state() -> CircuitBreakerState`
   - `evaluate(prediction: EnsemblePrediction, ambiguity_beta: float, state: CircuitBreakerState, cusum_shock: bool = False, regime_is_panic: bool = False) -> tuple[CircuitBreakerDecision, CircuitBreakerState]`

---

## 5. Verification Plan

- **Unit Tests (`tests/unit/test_circuit_breakers.py`)**:
  - Test directional consensus and Shannon entropy bounds ($[0, 1]$).
  - Test continuous haircut sigmoid monotonicity and smoothness.
  - Test multi-tier escalation (`NORMAL` $\to$ `CAUTION` $\to$ `DERISK` $\to$ `HALT`).
  - Test anti-chattering hysteresis and dwell-time recovery lockouts.
  - Test non-finite NaN/Inf defense and error vector diagnostics.
  - Test performance benchmark: $K=100$ models evaluation completed in $\le 0.20\text{ms}$ median.
- **Coverage Target**: $\ge 90\%$ line coverage on `src/quant/analytics/circuit_breakers.py`.
- **Static Type Safety**: `mypy src --strict` with 0 errors.
- **Linters**: `ruff check .` and `ruff format --check .` 100% clean.
