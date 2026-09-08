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
* **Memory Truncation**: Truncate weight expansion at lag $l^*$ where $|\omega_k| < \epsilon$ ($\epsilon = 10^{-4}$), bounding complexity to $O(T)$.
* **Stationarity Search Heuristic**: Minimum degree $d^* \in (0, 1)$ satisfying Augmented Dickey-Fuller test ($p < 0.01$) while maximizing correlation with original series $P_t$.

### 3.2 Dynamic Volatility Triple-Barrier Method
Defines path-dependent trade exit thresholds standardized across volatility regimes:
* **Upper Barrier (Take Profit)**: $P_t \cdot (1 + c_1 \sigma_t)$
* **Lower Barrier (Stop Loss)**: $P_t \cdot (1 - c_2 \sigma_t)$
* **Vertical Barrier (Expiration)**: $t + \Delta t_{\text{horizon}}$
* **Label**: $y_t \in \{1, -1, 0\}$ depending on which barrier is touched first.

### 3.3 Two-Stage Meta-Labeling Architecture
1. **Primary Model (Directional)**: Generates trade recommendation $\hat{y}_t \in \{-1, 1\}$ with high recall.
2. **Secondary Model (Conviction & Sizing)**: Trains a probability-calibrated binary classifier on feature matrix $\mathbf{X}_t$ to predict:
   $$z_t = \mathbb{I}(y_t = \hat{y}_t) \in \{0, 1\}$$
3. **Bet Sizing**: Position size is parameterized by calibrated probability: $p_t = \mathbb{P}(z_t = 1 \mid \mathbf{X}_t)$.

---

## 4. Validation Methodology: Combinatorial Purged CV & Deflated Sharpe

### 4.1 Combinatorial Purged Cross-Validation (CPCV)
* Partition $T$ observations into $N$ chronological blocks.
* Form test splits from $\binom{N}{k}$ combinations, utilizing remaining $N-k$ blocks for training.
* **Purging**: Drop training observations whose information horizon overlaps with test fold start.
* **Embargoing**: Drop training observations for duration $h_{\text{embargo}}$ immediately following test split to prevent autoregressive leakage.

### 4.2 Deflated Sharpe Ratio (DSR)
Evaluates backtested Sharpe Ratio ($\widehat{SR}$) conditional on skewness $\hat{\gamma}_3$, kurtosis $\hat{\gamma}_4$, sample length $T$, trial count $K$, and variance of trials $V[\{\widehat{SR}_k\}]$:

$$E\left[\max_k \{SR_k\}\right] \approx (1-\gamma) Z^{-1}\left(1 - \frac{1}{K}\right) + \gamma Z^{-1}\left(1 - \frac{1}{K e}\right)$$
$$DSR = \Phi\left( \frac{(\widehat{SR} - E[\max \{SR_k\}]) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{SR}^2}} \right)$$
* **Rejection Rule**: Reject any strategy with $DSR < 0.95$.

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
