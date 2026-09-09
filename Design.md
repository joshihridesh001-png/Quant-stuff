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
