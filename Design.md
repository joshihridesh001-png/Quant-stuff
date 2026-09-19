# Quantitative & Technical Design Specification

## 1. News Ingestion & Temporal Memory Kernel

### 1.1 Composite News Event Vector
Each news event is projected into a composite feature space:
$$\mathbf{e}_i = \Big[ \mathbf{v}_{\text{dense}} \;\|\; \mathbf{s}_{\text{sentiment}} \;\|\; \mathbf{u}_{\text{urgency}} \;\|\; \mathbf{c}_{\text{centrality}} \Big] \in \mathbb{R}^D$$
* $\mathbf{v}_{\text{dense}} \in \mathbb{R}^{d_e}$: Transformer embedding (FinBERT).
* $\mathbf{s}_{\text{sentiment}} \in [-1, 1]^3$: Polarity, subjectivity, and novelty.
* $\mathbf{u}_{\text{urgency}} \in [0, 1]$: Transmission velocity prior.
* $\mathbf{c}_{\text{centrality}} \in [0, 1]^K$: Continuous association weights across target assets.

### 1.2 Hybrid Temporal Decay Kernel
The active news state for asset $k$ at time $t$ combines high-frequency exponential decay with long-horizon power-law persistence:
$$\mathbf{S}_{\text{news}}^{(k)}(t) = \sum_{i \in \mathcal{V}_t} c_{i,k} \cdot \mathbf{e}_i \cdot \kappa(t - t_i, \mathbf{u}_i)$$
$$\kappa(\Delta t, u) = \alpha \exp\left(-\frac{\Delta t}{\tau_{\text{fast}} \cdot (1 - u)}\right) + (1 - \alpha)\left(1 + \frac{\Delta t}{\tau_{\text{slow}}}\right)^{-\beta}$$
* **Truncation Boundary**: For computational bounds, events with $\kappa(\Delta t, u) < \epsilon$ ($\epsilon = 10^{-4}$) or $\Delta t > T_{\text{max}}$ are pruned.

---

## 2. Market Data Columnar Schema & Vectorization

### 2.1 Embedded Columnar Storage (DuckDB)
```sql
CREATE TABLE IF NOT EXISTS market_bars (
    asset_id VARCHAR NOT NULL,
    resolution VARCHAR NOT NULL,
    timestamp BIGINT NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    vwap DOUBLE NOT NULL,
    PRIMARY KEY (asset_id, resolution, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_market_bars_lookup 
ON market_bars (asset_id, resolution, timestamp);
```

### 2.2 Instantaneous Realized Volatility Estimator
Computed on closing log returns over sliding window $W$:
$$r_t = \ln(P_t / P_{t-1})$$
$$\sigma_t = \sqrt{\frac{1}{W - 1}\sum_{k=0}^{W-1} \left(r_{t-k} - \bar{r}\right)^2}$$

---

## 3. Econometric Feature Engineering & Target Labeling

### 3.1 Fractional Differentiation $(1 - B)^d$
Preserves long-horizon memory while achieving stationarity:
$$(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = 1 - dB + \frac{d(d-1)}{2!}B^2 - \frac{d(d-1)(d-2)}{3!}B^3 + \dots$$
Weights are generated recursively:
$$\omega_0 = 1, \quad \omega_k = -\omega_{k-1} \frac{d - k + 1}{k}$$
* **Memory Truncation**: Truncate weight expansion at lag $l^*$ where $|\omega_k| < \epsilon$ ($\epsilon = 10^{-4}$), bounding filter length to $l^* \le 0.20 \cdot T$.
* **Zero-Sum DC-Offset Normalization**: Truncated expansions leave $\sum \omega_k > 0$, leaking secular price trends into differenced features. Re-centering $\omega_0^* = -\sum_{k=1}^{l^*} \omega_k$ guarantees zero gain at DC frequency ($\sum_{k=0}^{l^*} \omega_k = 0$).
* **Automated Bisection Search**: Evaluates optimal minimum degree $d^* \in [0, 1]$ satisfying Augmented Dickey-Fuller stationarity ($p \le 0.01$) in $\le 7$ binary iterations.
* **Vectorized 1D FFT Convolution**: Evaluates $\tilde{P}_t = \sum_{k=0}^{l^*} \omega_k P_{t-k}$ in $O(T \log l^*)$ time with $O(T)$ RAM using `scipy.signal.fftconvolve`.
* **Analytical Price Reconstruction**: Inverses differenced values for order pricing and risk calculations:
  $$P_t = \frac{\tilde{P}_t - \sum_{k=1}^{l^*} \omega_k P_{t-k}}{\omega_0}$$

### 3.2 Dynamic Volatility Triple-Barrier Method
Defines path-dependent trade exit thresholds standardized across volatility regimes:
* **Causal Range Volatility (Parkinson)**:
  $$\sigma_t = \sqrt{\frac{1}{4 \ln(2) \cdot W} \sum_{i=0}^{W-1} \left(\ln \frac{H_{t-1-i}}{L_{t-1-i}}\right)^2}$$
  Strictly lagged to $t-1$ to prevent lookahead bias; clamped to $[\sigma_{\text{floor}}, \sigma_{\text{cap}}]$.
* **Geometric Log-Price Space Barriers**:
  $$\ln(\text{Upper}) = \ln(P_{\text{entry}}) + c_1 \sigma_t, \quad \ln(\text{Lower}) = \ln(P_{\text{entry}}) - c_2 \sigma_t$$
  For Short positions, lower barrier is profit target and upper barrier is stop-loss.
* **Vertical Barrier (Expiration)**: $t_{\text{entry}} + H$ trading bars.
* **Discontinuous Opening Gap Fill**: When market opens beyond a barrier, exit price is the actual open ($O_k$) rather than the theoretical barrier line.
* **Pessimistic Collision Policy**: If both High and Low penetrate barriers within the identical candle, stop-loss execution takes absolute priority.
* **Net Realized Payoff**: Deducts round-trip friction:
  $$R_t = \text{side} \cdot \frac{P_{\text{exit}} - P_{\text{entry}}}{P_{\text{entry}}} - 2 \cdot (\text{spread} + \text{fee})$$
* **Categorical Label**: $y_t \in \{1, -1, 0\}$ depending on first barrier touched.

### 3.3 Two-Stage Continuous-Payoff Kelly Meta-Labeling Architecture
1. **Primary Model (Directional Recall)**: Generates high-recall directional orientation $\hat{y}_t \in \{-1, 1\}$.
2. **Payoff-Aware Meta-Labeling**: Evaluates direction-aligned net payoff $\pi_t = \hat{y}_t \cdot R_t^{\text{net}}$, assigning binary success target:
   $$z_t = \mathbb{I}(\pi_t > 0) \in \{0, 1\}$$
   Fairly credits net-profitable vertical timeouts while penalizing fee-eroded trades.
3. **Regularized Platt Calibration**: Maps model margins $m_t$ to monotonic calibrated probability $p_t$:
   $$p_t = \frac{1}{1 + \exp(A \cdot m_t + B)}, \quad A < 0$$
   Validated via mandatory Brier score reduction against baseline prior.
4. **Time-Decayed Fractional Kelly Sizing**: Calculates optimal growth allocation scaled by conservative multiplier $\lambda \in [0.25, 0.50]$:
   $$f^*_t = \frac{p_t \cdot b_t - (1 - p_t)}{b_t}, \quad s_t = \text{sign}(\hat{y}_t) \cdot \min\left( L_{\text{max}}, \frac{\lambda \cdot \max(0, f^*_t)}{\sqrt{\max(1, \tau_t) / \tau_{\text{ref}}} \cdot \max(1, c_t)} \right)$$
   where $b_t$ is the net payoff odds ratio, $\tau_t$ is holding duration, and $c_t$ is active concurrent trade count. Zero allocation is enforced when expected value is non-positive ($f^*_t \le 0$).

---

## 4. Validation Methodology: Combinatorial Purged CV & Deflated Sharpe

### 4.1 Combinatorial Purged Cross-Validation (CPCV)
* Partition $T$ observations into $N$ balanced contiguous chronological blocks $G_0, \dots, G_{N-1}$.
* Form test splits from $\binom{N}{k}$ combinations (or forward-chained causal partitions), utilizing remaining blocks for candidate training.
* **Exact Interval Intersection Purging**: A candidate training observation $i$ with holding period $[t_{i, \text{entry}}, t_{i, \text{exit}}]$ is purged if it overlaps any test block interval $[T_{\text{test, start}}, T_{\text{test, end}}]$:
  $$[t_{i, \text{entry}}, t_{i, \text{exit}}] \cap [T_{\text{test, start}}, T_{\text{test, end}}] \ne \emptyset \iff (t_{i, \text{entry}} \le T_{\text{test, end}}) \land (t_{i, \text{exit}} \ge T_{\text{test, start}})$$
* **Autoregressive Embargoing**: Training observations immediately succeeding test intervals within duration $h_{\text{embargo}} = \lceil T \cdot \text{embargo\_pct} \rceil$ (or explicit bars) are excluded to eliminate residual autoregressive memory leakage.
* **Starvation Guard**: Splits failing retained sample ratio constraint $\frac{|\text{train\_indices}|}{T} \ge \text{min\_train\_ratio}$ are defensively rejected.
* **Continuous Backtest Path Reconstruction**: Folds are stitched into $\phi = \binom{N-1}{k-1}$ continuous out-of-sample backtest paths using greedy positional fold assignment: for each block $g$, its $p$-th test fold occurrence fills path $p$.
* **Empirical Sharpe Variance Vector**: Evaluates empirical Sharpe distribution $\{SR_p\}_{p=1}^\phi$ and its variance $V[\{SR\}] = \frac{1}{\phi - 1} \sum_{p=1}^\phi (SR_p - \overline{SR})^2$, which is supplied directly as the variance parameter to Step 6 (Deflated Sharpe Ratio).

### 4.2 Deflated Sharpe Ratio (DSR) & Statistical Significance
Evaluates backtested Sharpe Ratio ($\widehat{SR}$) conditional on skewness $\hat{\gamma}_3$, Pearson kurtosis $\hat{\gamma}_4$, sample length $T$, effective independent trial count $K_{\text{eff}}$, and cross-trial variance $V[\{\widehat{SR}_k\}]$:

1. **Robust Moment Estimation & Pearson Clamping**:
   Two-sided winsorization suppresses outlier wicks. Kurtosis is mathematically bounded to prevent negative standard error radicals:
   $$\hat{\gamma}_4 \ge 1 + \hat{\gamma}_3^2$$

2. **Probabilistic Sharpe Ratio (PSR)**:
   $$\hat{\sigma}_{SR} = \sqrt{\frac{\max\left(10^{-8}, \; 1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{SR}^2\right)}{T - 1}}$$
   $$\text{PSR}(SR^*) = \Phi\left( \frac{\widehat{SR} - SR^*}{\hat{\sigma}_{SR}} \right)$$

3. **Effective Independent Trials via Spectral Decomposition**:
   Constructs correlation matrix $\mathbf{C} \in \mathbb{R}^{K \times K}$ across candidate models and calculates effective rank:
   $$K_{\text{eff}} = \frac{(\text{tr}(\mathbf{C}))^2}{\text{tr}(\mathbf{C}^2)} = \frac{K^2}{\sum_{i=1}^K \sum_{j=1}^K C_{ij}^2}$$

4. **Expected Maximum Sharpe Ratio Hurdle**:
   $$E\left[\max_{k=1\dots K_{\text{eff}}} \{SR_k\}\right] \approx \overline{SR} + \sqrt{V[\{SR\}]} \cdot \left( (1-\gamma) \Phi^{-1}\left(1 - \frac{1}{K_{\text{eff}}}\right) + \gamma \Phi^{-1}\left(1 - \frac{1}{K_{\text{eff}} e}\right) \right)$$
   where $\gamma \approx 0.5772156649$ is the Euler-Mascheroni constant. Collapses to $\overline{SR}$ when $K_{\text{eff}} \le 1$ or $V \le 0$.

5. **Piecewise Minimum Backtest Length (MinBTL)**:
   $$\text{MinBTL} = \begin{cases}
     1 + \left(1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{SR}^2\right) \left( \frac{\Phi^{-1}(0.95)}{\widehat{SR} - E[\max \{SR\}]} \right)^2 & \text{if } \widehat{SR} > E[\max \{SR\}] \\
     +\infty & \text{if } \widehat{SR} \le E[\max \{SR\}]
   \end{cases}$$

6. **Dual Institutional Gate**:
   $$\text{Certify Strategy} \iff (\text{DSR} \ge 0.95) \land (T \ge \text{MinBTL})$$

7. **Cohort False Discovery Rate (FDR) Controls**:
   Applies Benjamini-Hochberg (BH) or Benjamini-Yekutieli (BY) stepdown procedures across multi-model genetic populations to guarantee:
   $$\text{FDR} \le Q^* = 0.05$$

---

## 5. Game-Theoretic Scenario Generator & Payoff Matrices

Market price response is formulated as a Bayesian Game against Nature/Counterparties:
$$\Gamma = \langle \mathcal{N}, \{\mathcal{A}_i\}, \{\Theta_i\}, \{u_i\} \rangle$$
* **Player 1 Action**: Asset allocation vector $\mathbf{a}_1 \in [-1, 1]^K$.
* **Player 2 Regimes**: Immediate Reversal ($\mathbf{s}_1$), Momentum Cascade ($\mathbf{s}_2$), Liquidity Squeeze ($\mathbf{s}_3$).
* **Scenario Payoff**:
  $$U(i, \mathbf{s}_j) = \mathbf{a}_1^T \mathbb{E}[\mathbf{r} \mid \mathbf{s}_j] - \lambda \mathbf{a}_1^T \mathbf{\Sigma}(\mathbf{s}_j) \mathbf{a}_1 - \mathcal{C}(\mathbf{a}_1, \mathbf{a}_1^{\text{prior}}) - \Omega_{\text{crowding}}(\mathbf{a}_1, \mathbf{s}_j)$$
* **Minimax Regret Robustness**:
  $$V_i = -\max_{\mathbf{s}_j \in \mathcal{S}} \left( \max_{\mathbf{a}^*} U(\mathbf{a}^*, \mathbf{s}_j) - U(\mathbf{a}_i, \mathbf{s}_j) \right)$$

---

## 6. Evolutionary Algorithm: Genotype & Hypergamy Dynamics

### 6.1 Genotype Chromosome Structure
$$\mathbf{g}_k = \langle \mathbf{g}_{\text{repr}}, \mathbf{g}_{\text{game}}, \mathbf{g}_{\text{infer}}, \mathbf{g}_{\text{risk}} \rangle$$
* $\mathbf{g}_{\text{repr}}$: Decay parameters $\tau_{\text{fast}}, \tau_{\text{slow}}, \alpha$ and projection weights.
* $\mathbf{g}_{\text{game}}$: Scenario belief priors $\mathbf{p}_{\text{belief}}$ and risk-aversion scalar $\lambda$.
* $\mathbf{g}_{\text{infer}}$: Model hyper-parameters (attention heads, depth, splitting thresholds).
* $\mathbf{g}_{\text{risk}}$: Volatility targeting scalars, maximum drawdown hard stop, turnover budget.

### 6.2 Hypergamic Assortative Selection
The population is stratified into the **Alpha Cohort** (top $\rho = 20\%$ by multi-objective fitness) and the **Aspirant Cohort** (remaining $80\%$).
* **Asymmetric Mating**: Aspirants preferentially attempt to mate with Alpha models.
* **Gated Acceptance**: An Alpha individual rejects pairing unless the candidate exhibits orthogonal residual errors:
  $$\text{Corr}(\mathbf{e}_{\text{Alpha}}, \mathbf{e}_{\text{Aspirant}}) < \delta_{\text{ortho}}$$
  This prevents phenotypic cloning while actively propagating robust alpha characteristics.

### 6.3 Multi-Objective Fitness Functional
$$\mathcal{F}(\mathcal{I}_i) = \text{DeflatedSharpe}(\mathcal{I}_i) \cdot \exp\left( -\psi \cdot \text{MaxDD}(\mathcal{I}_i) \right) + \omega_1 V_i(\text{Regret}) + \omega_2 \mathcal{H}_{\text{novelty}}(\mathcal{I}_i)$$
* $\mathcal{H}_{\text{novelty}}$: Phenotypic crowding distance relative to $K$-nearest neighbors in chromosome space.

---

## 7. Regime-Conditioned Dynamic Model Averaging (RD-DMA)

### 7.1 Volatility-Adaptive Forgetting Factor
Dynamically modulates memory depth $\alpha_t \in [\alpha_{\min}, \alpha_{\max}]$ in response to normalized volatility innovations $\Delta \sigma_t = (\sigma_t - \sigma_{t-1}) / \sigma_{t-1}$:
$$\alpha_t = \alpha_{\max} - (\alpha_{\max} - \alpha_{\min}) \cdot \frac{1}{1 + \exp\left(-\gamma_{\text{adapt}} \cdot \Delta \sigma_t\right)}$$
* In calm regimes ($\Delta \sigma \le 0$), $\alpha_t \to \alpha_{\max} = 0.99$ (100-bar effective memory).
* In sudden volatility shocks ($\Delta \sigma \gg 0$), $\alpha_t \to \alpha_{\min} = 0.85$ (1–2 bar memory for rapid adaptation).

### 7.2 Forward-Markov Regime Transition Prior
Forecasts next-step regime distribution from current posteriors $\mathbf{p}_t \in \Delta^3$:
$$\mathbf{p}_{t+1|t} = \mathbf{P}_{\text{trans}}^T \mathbf{p}_t$$
where $\mathbf{P}_{\text{trans}}$ is the $3 \times 3$ row-stochastic Markov transition matrix over regimes {Absorption, Momentum, Panic}. The composite predictive prior $\bar{\boldsymbol{\pi}}_{t+1|t}$ is synthesized across regime scenario conditional forecasts.

### 7.3 Asymmetric Downside Loss Scoring
Penalizes downside forecast errors and sign-inversion errors more heavily than upside misses:
$$\ell_{t, k} = (y_t - \tilde{y}_{t, k})^2 + \gamma_{\text{down}} \max(0, -y_t \tilde{y}_{t, k})$$
with asymmetry multiplier $\gamma_{\text{down}} = 2.50$, directly factoring downside semi-variance $\sigma^2_{k, \text{down}}$ into model likelihood updates.

### 7.4 Tikhonov-Regularized Correlation & Simplex Optimization
Evaluates pairwise prediction correlation matrix $\mathbf{C}_t$ with standard deviation floor $\sigma_{\min} = 10^{-8}$ and Tikhonov ridge shrinkage:
$$\mathbf{C}_t^{\text{shrunk}} = (1 - \delta_{\text{ridge}}) \widehat{\mathbf{C}}_t + \delta_{\text{ridge}} \mathbf{I}_K, \quad \delta_{\text{ridge}} = 0.05$$
guaranteeing strict positive definiteness ($\lambda_{\min} \ge 0.05$). Solves optimal model allocation $\mathbf{w}_t^* \in \Delta^K$ via Entropic Mirror Descent under SVD subspace orthogonality penalization ($\lambda_{\text{ortho}} = 0.25$) in $< 0.15\text{ms}$ with immutable Laplace floor smoothing $\epsilon_{\text{floor}} = 0.001 / K$.

### 7.5 Total Variance Decomposition
Decomposes predictive uncertainty via the Law of Total Variance (`INV-ENS-002`):
$$\sigma^2_{\text{total}} = \sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}} = \sum_{k=1}^K w_k \sigma_k^2 + \sum_{k=1}^K w_k (\tilde{y}_k - \bar{y})^2$$
where $\sigma^2_{\text{aleatoric}}$ measures irreducible data noise and $\sigma^2_{\text{epistemic}}$ measures model disagreement.

---

## 8. Epistemic Disagreement Entropy & Circuit Breaker Overlays

### 8.1 3-Simplex Directional Consensus & Normalized Shannon Entropy
Partitions model forecasts into bullish, bearish, and neutral consensus around deadband $\delta_{\text{sign}} = 10^{-4}$:
$$p_+ = \sum_{\tilde{y}_k > \delta} w_k, \quad p_- = \sum_{\tilde{y}_k < -\delta} w_k, \quad p_0 = \sum_{|\tilde{y}_k| \le \delta} w_k$$
Residing on the 3-simplex $\mathbf{p} = [p_+, p_-, p_0]^T \in \Delta^3$ (`INV-CB-004`), normalized Shannon directional entropy is:
$$\widetilde{H}_{\text{dir}} = \frac{-\sum_{s \in \{+, 0, -\}} p_s \ln(p_s + \epsilon)}{\ln 3} \in [0.0, 1.0]$$

### 8.2 Epistemic Uncertainty Ratio & Composite Shock Score
$$\rho_{\text{epistemic}} = \frac{\sigma^2_{\text{epistemic}}}{\sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}} \in [0.0, 1.0)$$
$$H_{\text{epistemic}} = \widetilde{H}_{\text{dir}} \cdot \sqrt{\rho_{\text{epistemic}}} \in [0.0, 1.0]$$
Thermodynamic composite shock score integrates macroeconomic ambiguity temperature $\tilde{\beta}_t$:
$$\Xi_t = \omega_H H_{\text{epistemic}} + \omega_\rho \rho_{\text{epistemic}} + \omega_\beta \tilde{\beta}_t \in [0.0, 1.0]$$

### 8.3 Normalized Continuous Logistic Sigmoid Haircut ($\kappa_t$)
Calibrated to exact boundary anchors $\kappa_t(0) \equiv 1.0000$ and $\kappa_t(1) \equiv 0.0000$ (`INV-CB-001`):
$$\kappa_t = \text{clip}\left(\frac{\sigma_{\text{raw}}(\Xi_t) - \sigma_{\text{raw}}(1)}{\sigma_{\text{raw}}(0) - \sigma_{\text{raw}}(1)}, 0.0, 1.0\right)$$

### 8.4 Multi-Tier Hysteresis State Machine
Four-tier risk hierarchy (`CircuitBreakerTier`): `NORMAL` (0), `CAUTION` (1), `DERISK` (2), `HALT` (3) (`INV-CB-002`).
* **Instantaneous Escalation**: Transitions immediately to higher tiers on entropy or volatility surges. Single-bar emergency `HALT` on joint CUSUM jump and panic regime.
* **Anti-Chattering Dwell Cooling**: Enforces $\tau_{\text{dwell}} \ge 5$ bars in `HALT` and `DERISK` (`INV-CB-003`).
* **Dual-Barrier Recovery**: Requires $\Xi_t < \theta_{\text{recovery}} = 0.30$ and strictly single-tier stepdown transitions.

---

## 9. Semi-Parametric Peaks-Over-Threshold EVT Tail Risk & Unified Convex Sizing

### 9.1 POT Generalized Pareto Distribution (GPD) with Closed-Form PWM
* Evaluates dynamic causal high threshold $u_t = \mu_{t-1} + k_{\text{threshold}} \sigma_{t-1}$ on $[t-W, t-1]$ (`INV-TR-007`).
* Probability Weighted Moments (PWM) estimates GPD parameters algebraically in closed form without iterative solvers (Rule 4.3):
  $$M_0 = \frac{1}{N_u} \sum_{i=1}^{N_u} y_i, \quad M_1 = \sum_{i=1}^{N_u} \frac{N_u - i - 0.35}{N_u(N_u - 0.70)} y_{(i)}$$
  $$\hat{\xi}_{\text{PWM}} = 1 - \frac{M_0}{2(M_0 - 2M_1)}, \quad \hat{\beta}_{\text{PWM}} = \frac{2 M_0 M_1}{M_0 - 2M_1}$$
* **Fréchet Tail Stability Clamping & Tripwire (`INV-TR-002`)**: $\xi \in [0.001, 0.999]$. Theoretical infinite variance $\xi \ge 1.0 \implies \text{InfiniteVarianceException}$ demanding emergency execution `HALT`, backed by Hill pre-filter on top 10% exceedances.
* **Coherent Expected Shortfall (CVaR)**: Strictly satisfies $\text{CVaR}_\alpha \ge \text{VaR}_\alpha$ (`INV-TR-001`) and Artzner subadditivity (`INV-TR-003`):
  $$\text{CVaR}_\alpha = \frac{\text{VaR}_\alpha}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}$$
* **3-Tier Cold-Start Degradation Ladder**: Empirical ($N < 30$) $\to$ Student-t MoM with analytical log-gamma integral ($30 \le N < 250$) $\to$ EVT-GPD PWM ($N \ge 250$).

### 9.2 Unified Strictly Concave Execution Sizing
Maximizes strictly concave objective $\mathcal{L}(\boldsymbol{\nu})$ guaranteeing negative-definite Hessian $\nabla^2 \mathcal{L} \prec 0$ (`INV-TR-004`):
$$\max_{\boldsymbol{\nu}} \mathcal{L}(\boldsymbol{\nu}) = U_{\text{Kelly}}(\boldsymbol{\nu}) - \mathcal{C}_{\text{Impact}}(\boldsymbol{\nu}) - \mathcal{R}_{\text{Epistemic}}(\boldsymbol{\nu})$$
1. **Directional Epistemic Shrinkage**:
   $$\tilde{\mu}_i = \text{sign}(\mu_i) \max\left(0, |\mu_i| - \lambda_{\text{shrink}} \sigma^2_{\text{epistemic}, i}\right)$$
   Shrinks expected edge toward zero under high model disagreement, eliminating sign-flipping short squeeze traps.
2. **Universal 3/2-Power Pseudo-Huber Impact**:
   $$\mathcal{C}_{\text{Impact}}(\boldsymbol{\nu}) = \frac{\eta}{W_t^{1/2}} \sum_{i=1}^N \sigma_i \left( (\nu_i^2 + \delta^2 W_t^2)^{3/4} - (\delta W_t)^{3/2} \right) + \frac{1}{2 W_t} \boldsymbol{\nu}^T \boldsymbol{\Lambda}_{\text{cross}} \boldsymbol{\nu}$$
   with PSD permanent cross-impact tensor $\boldsymbol{\Lambda}_{\text{cross}} \succeq 0$.
3. **Circuit Breaker Regularization**:
   $$\mathcal{R}_{\text{Epistemic}}(\boldsymbol{\nu}) = \frac{1}{2 \kappa_t W_t} \|\boldsymbol{\nu}\|_2^2$$
   smoothly crushes allocation to $\mathbf{0}$ as continuous haircut $\kappa_t \to 0$.

### 9.3 Exact $O(N \log N)$ Dual Projection onto Two $L_1$ Balls (`INV-TR-005`)
Projects unconstrained candidate $\mathbf{y}$ onto the intersection of the gross leverage ball and Expected Shortfall drawdown budget:
$$\mathcal{K} = \left\{ \boldsymbol{\nu} \in \mathbb{R}^N : \|\boldsymbol{\nu}\|_1 \le L_{\max} W_t, \; \mathbf{c}^T |\boldsymbol{\nu}| \le \text{MDD}_{\text{budget}} W_t \right\}$$
Solved via 2D Semismooth Newton active-set iteration with provable Dykstra alternating projections fallback and terminal zero-leakage radial contraction.

### 9.4 Microstructural Randomized Lot Discretization
Bridges continuous targets $\boldsymbol{\nu}^*$ to discrete contract lots $\Delta \nu_i$ via Bernoulli lottery:
$$\tilde{\nu}_i = \text{sign}(\nu_i^*) \cdot \left(\left\lfloor \frac{|\nu_i^*|}{\Delta \nu_i} \right\rfloor + B_i\right) \cdot \Delta \nu_i, \quad B_i \sim \text{Bernoulli}(\text{frac}_i)$$
guaranteeing unbiased expectation $\mathbb{E}[\tilde{\boldsymbol{\nu}}] = \boldsymbol{\nu}^*$ without systemic cash drag or margin rounding bias.

---

## 10. End-to-End Live Replay Simulator & Institutional Benchmarking

### 10.1 Causal Replay Loop & Information Barriers (`INV-SIM-001`)
* Chronological event loop $t=0 \dots T-1$ over historical bars.
* Strictly enforces information barrier filtration $\mathcal{F}_{t-1}$: decisions at bar $t$ condition strictly on slice $[:t]$ (lagged returns, volatilities, regimes), with zero contamination from contemporaneous or future prices.
* Circuit breaker emergency `HALT` collapses target and executed allocations strictly to $\mathbf{0}$ (`INV-SIM-004`).

### 10.2 Non-Linear Execution Friction (`INV-SIM-003`)
* Evaluates non-negative total transaction cost $\mathcal{C}_t = \mathcal{C}_{\text{fee}} + \mathcal{C}_{\text{spread}} + \mathcal{C}_{\text{impact}} \ge 0.0$.
* 3/2-power Kyle-Obizhaeva market impact with hardware-native $x \sqrt{x}$ acceleration:
  $$\mathcal{C}_{\text{impact}} = \sum_{i=1}^N \frac{\lambda_0}{\sqrt{\text{ADV}_i}} \sigma_{i, t} |\Delta \nu_{i, t}|^{3/2}$$
* Exchange fee schedule $\mathcal{C}_{\text{fee}} = \text{fee}_{\text{bps}} \cdot 10^{-4} \cdot \|\Delta \boldsymbol{\nu}\|_1$ and half-spread slippage.

### 10.3 Causal Mark-to-Market Accounting (`INV-SIM-002`)
* Realizes portfolio gross return on bar $t$ strictly from lagged exposure $\boldsymbol{\nu}_{t-1}^T \mathbf{r}_t$.
* Bit-exact capital conservation: $|W_t - (\text{cash}_t + \sum \nu_{i, t})| < 10^{-5}$ and $W_t - W_{t-1} = \text{PnL}_t^{\text{net}}$.
* Terminal ruin tripwire: $W_t \le 0.0 \implies \text{InfeasibleSimulationException(ERR-SIM-003)}$ with immediate position liquidation.
* Tracks running high-water mark $M_t = \max_{s \le t} W_s$ and drawdown $D_t = (M_t - W_t) / M_t \in [0.0, 1.0]$.

### 10.4 Institutional Benchmark Auditor & Statistical Certification
* Evaluates institutional metrics: CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, Realized VaR 95/99, Realized CVaR 95/99, Tail Ratio.
* Certifies statistical validity via `DeflatedSharpeEngine`: verifies DSR $\ge 0.95$ and $T \ge \text{MinBTL}$.
* Multi-benchmark attribution against Equal Weight ($1/N$), Risk Parity (Inverse Volatility), and Cash ($R_f=0$) baselines.
* Prefix-sum centered variance acceleration achieving $< 2.5\text{ms}$ audit execution.
* Emits observer hooks via `SimulationListener` protocol (`on_bar_start`, `on_decision`, `on_fill`, `on_bar_end`).
* Latency SLA: 100 bars $\times$ 10 assets completes in $\approx 15.5\text{ms} \le 25\text{ms}$ (`INV-SIM-006`).

---

## 11. Live Execution Gateway, Order State Machine & Risk Firewall (Phase 6)

### 11.1 Deterministic Order State Machine & Causal Reconciliation
* **Monotonic DAG Transitions (`INV-GW-001`)**:
  $$\text{PENDING\_NEW} \longrightarrow \text{NEW} \longleftrightarrow \text{PARTIALLY\_FILLED} \longrightarrow \text{FILLED} \; / \; \text{CANCELLED} \; / \; \text{REJECTED}$$
* **Terminal State Immutability**: States `FILLED`, `CANCELLED`, `REJECTED`, and `EXPIRED` are strictly absorbing and reject subsequent state mutations.
* **Mass Conservation Invariant (`INV-GW-003`)**:
  $$Q_{\text{filled}} + Q_{\text{leaves}} \equiv Q_{\text{target}} \pm 10^{-7}$$
* **Causal Out-of-Order Packet Reconciliation (`INV-GW-004`)**: If a fill execution report arrives before the parent `NEW` acknowledgement, the FSM transitions directly to `PARTIALLY_FILLED` or `FILLED`, recording the out-of-order fill causally without exception.
* **Cryptographic Idempotency Router (`INV-GW-002`)**: Client order IDs derived deterministically via RFC 4122 UUIDv5 over `(strategy_id, asset_id, client_timestamp_ns, order_quantity)`. Duplicate tokens in the active in-flight ring buffer return cached existing orders.

### 11.2 Microstructural Schedulers & Smart Order Router
* **Poisson TWAP Scheduler (`INV-SOR-001`, `INV-SOR-004`)**:
  Interval durations randomized via Poisson clock with physical floor $\Delta t_{\min} \ge 1\text{ ns}$:
  $$\Delta t_k = \max\left(\Delta t_{\min}, \; \bar{\tau} \cdot (1 + U(-\alpha_t, \alpha_t))\right)$$
  Randomized sizing jitter $q_k = \bar{q} \cdot (1 + U(-\alpha_q, \alpha_q))$ with final slice closure strictly enforcing $\sum q_k \equiv Q_{\text{rem}}$.
* **Volume-Adaptive VWAP Scheduler (`INV-SOR-003`)**:
  Bayesian volume blending $\widehat{V}_k = \omega V_{\text{hist}, k} + (1 - \omega) V_{\text{realtime}, k}$ with hard institutional participation cap $\rho \le 15\%$ across all trading slices:
  $$q_k \le \rho \cdot \widehat{V}_k$$
* **Nonlinear Arrival Price Scheduler**:
  Closed-form Almgren-Chriss (2000) optimal liquidation trajectory under 3/2-power market impact proxy:
  $$x_j = X_0 \cdot \frac{\sinh(\kappa_t (T - t_j))}{\sinh(\kappa_t T)}, \quad \kappa_t = \kappa_0 \cdot \frac{\sigma_t}{\sigma_{\text{baseline}}}$$
  Numerically stabilized using linear Taylor expansion $x_j = X_0 (T - t_j) / T$ when $\kappa T < 10^{-6}$, and exponential ratio formulation $x_j = X_0 \cdot e^{-\kappa t_j} (1 - e^{-2\kappa(T - t_j)}) / (1 - e^{-2\kappa T})$ when $\kappa T > 50.0$ to prevent IEEE 754 floating-point overflow.
* **Smart Order Router (Two-Phase Routing, `INV-SOR-002`, `INV-SOR-006`)**:
  1. *Phase 1 (Dark Midpoint Probing)*: Sequential probing of registered dark pools (`DARK_POOL`) at uncrossed NBBO midpoint $P_{\text{mid}} = (P_{\text{bid}} + P_{\text{ask}}) / 2$ with Minimum Execution Size (MES) filter, capturing half-spread price improvement without information leakage.
  2. *Phase 2 (Lit Algebraic Waterfilling)*: Unfilled residual routed across lit venues (`LIT_EXCHANGE`) sorted by marginal effective net cost (taker fee minus maker rebate) in $O(M \log M)$ closed-form algebraic waterfilling without black-box numerical solvers, achieving $< 0.02\text{ms}$ routing latency.
  3. *Toxic Markout Watchdog*: Evaluates post-trade adverse selection:
     $$\Delta P_{\text{markout}} = \text{sign}(\text{side}) \cdot \frac{P_{\text{post}} - P_{\text{fill}}}{P_{\text{fill}}} \cdot 10^4$$
     Automatically quarantines venues breaching consecutive toxicity limits.

### 11.3 Perold (1988) Implementation Shortfall TCA Decomposition (`INV-SOR-005`)
Strictly preserves exact additive identity for both BUY and SELL sides:
$$\text{Total Shortfall} \equiv \text{Delay Cost} + \text{Price Impact} + \text{Spread Slippage} + \text{Fees Paid} + \text{Opportunity Cost}$$
* **Delay Cost**: $(P_{\text{decision\_arrival}} - P_{\text{arrival}}) \cdot Q_{\text{filled}}$
* **Price Impact**: $(P_{\text{vwap}} - P_{\text{decision\_arrival}}) \cdot Q_{\text{filled}}$
* **Opportunity Cost**: $(P_{\text{terminal}} - P_{\text{arrival}}) \cdot Q_{\text{unfilled}}$

### 11.4 In-Memory Pre-Trade Risk Firewall & Emergency Kill Switch
* **Directional Netting Exposure (`INV-RSK-001` - `INV-RSK-007`)**:
  $$\text{Exposure}_i = \max(|w_i|, |w_i + q_{\text{leaves}, i}|) \cdot P_i$$
  Allows de-risking closing orders to execute unconditionally even when cash balance is negative (`required_margin <= 0.0`).
* **Sub-1.5$\mu$s Hot Path**: All checks (fat finger, gross leverage, net leverage, concentration, margin, intraday drawdown tripwire, non-finite input guards) execute in memory in $\approx 1.5\mu\text{s}$.
* **Emergency Panic Kill Switch (`INV-RSK-008`)**: Multi-trigger tripwire (`MANUAL`, `GATEWAY_DISCONNECT`, `DRAWDOWN_BREACH`, `ROGUE_FILL`) executing concurrent multi-gateway mass cancellation sweep via `asyncio.gather` in $< 5\text{ms} \le 50\text{ms}$ SLA, freezing schedulers, and locking order submission (`ERR-RSK-008`).

---

## 12. Live Execution REST, WebSockets & Terminal Bridge (Phase 7)

### 12.1 Execution & Risk Application Services
* `ExecutionService`: Coordinates parent order lifecycles, maps strategy requests to schedulers (`POISSON_TWAP`, `VOLUME_ADAPTIVE_VWAP`, `ARRIVAL_PRICE`), coordinates slice dispatch through `RiskOrchestrator`, updates child order blotters, and computes Perold (1988) TCA shortfall reports.
* `RiskService`: Aggregates real-time portfolio risk telemetry, updates pre-trade firewall parameters, monitors gateway heartbeats, and handles emergency kill switch triggers/resets.

### 12.2 Full-Duplex WebSockets & Institutional Trading Terminal HUD
* `/api/v1/ws/executions`: Real-time streaming of parent order transitions, child slice dispatches, and fill notifications.
* `/api/v1/ws/risk`: Real-time streaming of portfolio NAV, leverage, margin, drawdowns, watchdog heartbeats, and circuit breaker tripwire alerts.
* `trading_terminal.html`: High-refresh browser HUD featuring WebGL/Canvas order book depth rendering, candlestick chart, strategy population explorer with pagination across 1,000 genotypes, active execution blotter, autonomous swarm controls, and emergency kill switch panel.

---

## 13. Production Live Trading Engine & Autonomous Swarm Daemon (Phase 8)

### 13.1 Institutional Alpaca Gateway Protocol Adapter
* Implements `ExecutionGateway` protocol for Alpaca Markets v2 Live and Paper trading endpoints.
* HTTP/2 connection pooling via `httpx.AsyncClient` with authentication headers (`APCA-API-KEY-ID`, `APCA-API-SECRET-KEY`).
* Monotonic state mapping between Alpaca order states and domain `OrderState`:
  $$\text{new} \to \text{NEW}, \; \text{partially\_filled} \to \text{PARTIALLY\_FILLED}, \; \text{filled} \to \text{FILLED}, \; \text{canceled} \to \text{CANCELLED}, \; \text{rejected} \to \text{REJECTED}$$
* UUIDv5 idempotency token passed as `client_order_id`, eliminating duplicate execution risks across transport retries.
* Account telemetry extraction (`cash`, `portfolio_value`, `buying_power`) and open position reconciliation.

### 13.2 Real-Time Market Data Ingestion Feed & Online Feature Pipeline
* `AlpacaMarketDataFeed`: Streams live OHLCV bars and NBBO consolidated quotes into DuckDB columnar storage (`add_bars_batch`) and `StreamingFracDiffBuffer`.
* Enforces boundary invariants: $H \ge \max(O, C)$, $L \le \min(O, C)$, $P > 0$, $V \ge 0$.
* Deterministic synthetic replay fallback for hermetic offline testing when live API credentials are unavailable.

### 13.3 Autonomous Swarm Clock Loop & Delta Rebalancing Daemon
* `AutonomousTradingEngine`: Continuous async clock loop orchestrating the end-to-end econometric prediction and execution pipeline:
  1. Ingests latest market data and updates online fractional differentiation buffers.
  2. Evaluates Bayesian jump-regime posterior probabilities and CUSUM panic gates.
  3. Computes dynamic model weights $\mathbf{w}_t \in \Delta^K$ via RD-DMA ensemble across elite strategy swarm.
  4. Evaluates epistemic disagreement entropy circuit breakers and continuous logistic haircut $\kappa_t$.
  5. Estimates semi-parametric EVT-POT Generalized Pareto tails, VaR, and Expected Shortfall (CVaR).
  6. Solves unified strictly concave execution sizing objective via exact 2D dual projection onto gross leverage and CVaR drawdown budget.
  7. Computes target dollar positions $\boldsymbol{\nu}^* = \mathbf{x}^* \cdot W_t$ and delta rebalancing quantities $\Delta q_i = (\nu_i^* - w_i P_i) / P_i$.
  8. Applies churn suppression filter: orders with $|\Delta q_i| \cdot P_i < \text{MIN\_TRADE\_NOTIONAL}$ are pruned.
  9. Validates proposed trades against pre-trade risk firewall directional netting matrix.
  10. Routes approved orders through algorithmic meta-order schedulers and the Smart Order Router to the live broker gateway.
* **Emergency Coupling**: When emergency kill switch is activated or drawdown limits are breached, target allocations instantly collapse to $\mathbf{0}$, halting trading loop execution.
* **REST & Operational Controls**: Full lifecycle management via `/api/v1/autonomous` (`status`, `start`, `stop`, `pause`, `resume`, `step`).

