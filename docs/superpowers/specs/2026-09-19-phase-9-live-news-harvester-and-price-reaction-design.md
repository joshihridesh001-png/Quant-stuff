# Design Specification: Phase 9 — Real-World Financial News Harvester & Causal Price Reaction Engine

**Author:** Antigravity / DeepMind Advanced Agentic Coding  
**Date:** 2026-09-19  
**Status:** Approved / Ready for Implementation Plan  
**Target Milestone:** Phase 9 (Live News Ingestion, Financial Sentiment & Causal Price Reaction Prediction)  
**Governing Standard:** `Rules.md` (Rules 1, 2, 3, 4)

---

## 1. Executive Summary & Problem Formulation

While Phases 1 through 8 delivered an institutional live execution engine, smart order router, pre-trade risk firewall, and autonomous trading swarm daemon, the core news ingestion pipeline (`EventService`, `POST /api/v1/events/ingest`) currently relies on external or manual event submission.

Financial markets do not move in a statistical vacuum. Unscheduled corporate announcements (quarterly earnings beats/misses, CEO departures, M&A bids, SEC regulatory investigations) and macroeconomic releases (FOMC rate decisions, CPI prints, jobs reports) introduce structural discontinuous volatility jumps and persistent multi-hour price drifts.

To capture alpha from real-world events, the platform requires an autonomous, high-speed, end-to-end **Real-World News Harvester & Causal Price Reaction Prediction Subsystem**:

1. **Automated Zero-Cost Live News Ingestion**: Automatically pulls breaking financial news 24/7 from public institutional RSS wires (Yahoo Finance, SEC EDGAR 8-K filings, CNBC, MarketWatch) with SHA-256 deduplication and point-in-time causality verification.
2. **Specialized Financial Domain Sentiment & Entity Mapping**: General-purpose NLP tokenizers fail on financial prose (e.g. treating standard balance sheet liabilities or costs as negative, while misinterpreting "loss narrowed" or "margin contraction slowed"). The engine utilizes the institutional **Loughran-McDonald Financial Lexicon** with negation parsing, ticker resolution, and continuous centrality weighting ($c_{i, k} \in [0, 1]$).
3. **Causal Price Reaction & Directional Trajectory Prediction**: Rather than outputting passive sentiment scores, the engine formulates a predictive **Drift-Diffusion Price Reaction Model**:
   - Predicts the **Direction** (UP vs. DOWN) and expected dollar move ($\Delta \hat{P}_{\text{expected}}$) scaled by historical volatility $\sigma_t$ and event category elasticity $\gamma_{\text{category}}$.
   - Predicts the **Probability** ($\mathbb{P}(\text{UP}) \in [0, 1]$) of breaching the Upper Profit Barrier before the Lower Stop Barrier in our dynamic Triple-Barrier framework.
   - Models the **Post-Announcement Drift (PEAD)** trajectory over time using the dual exponential and power-law memory decay kernel $\kappa(\Delta t, u)$.
4. **Direct Swarm Execution Feedback**: Injects predicted price reactions directly into the `AutonomousTradingEngine` as forward directional priors $\boldsymbol{\mu}_{\text{news}}$, rebalancing portfolio weights before the market finishes absorbing the event.

---

## 2. Invariant Contracts & Mathematical Formulations

Phase 9 is governed by six strict institutional invariants:

### Invariant 1: Availability Point-in-Time Causality (`INV-NEWS-001`)
- Every harvested news item records both its reported public timestamp $t_{\text{published}}$ and the local monotonic clock timestamp $t_{\text{available}}$ when our engine parsed the packet.
- Ingestion and prediction models strictly condition on $t_{\text{available}} \le t_{\text{decision}}$.
- Articles with future timestamps ($t_{\text{published}} > t_{\text{local}} + \epsilon_{\text{clock}}$) are rejected with `FutureTimestampException` (`ERR-NEWS-003`).

### Invariant 2: Cryptographic Deduplication & Idempotency (`INV-NEWS-002`)
- Inbound articles are fingerprinted via SHA-256 over normalized `(source, headline, clean_text)`.
- An in-memory FIFO ring buffer with 24-hour TTL prevents reprocessing the same story across syndication wires.

### Invariant 3: Bounded Tri-Axial Sentiment Normalization (`INV-NEWS-003`)
- Sentiment scores must strictly satisfy unit hypercube/interval bounds:
  $$\text{Polarity } s \in [-1.0, +1.0], \quad \text{Urgency } u \in [0.0, 1.0], \quad \text{Novelty } \mathcal{H} \in [0.0, 1.0]$$
- Non-finite values (NaN, $\pm\infty$) and boolean inputs are rejected with `NonFiniteSignalException` (`ERR-NEWS-005`).

### Invariant 4: Price Impact & Elasticity Scaling (`INV-NEWS-004`)
- Given current Parkinson range volatility $\sigma_t > 0$, current price $P_t > 0$, and net shock $\mathcal{S}_i = s \cdot u \cdot \mathcal{H} \cdot c$:
  $$\Delta \hat{P}_{\text{expected}} = P_t \cdot \gamma_{\text{category}} \cdot \mathcal{S}_i \cdot \sigma_t$$
- Elasticity multiplier $\gamma_{\text{category}}$ strictly adheres to empirical bounds $\gamma \in [0.1, 5.0]$.

### Invariant 5: Monotonic Barrier Breakout Probability (`INV-NEWS-005`)
- The probability of an upward breakout strictly satisfies logistic drift-diffusion monotonicity:
  $$\mathbb{P}(\text{UP}) = \frac{1}{1 + \exp\left(-\frac{\lambda \cdot \mathcal{S}_i \cdot \gamma_{\text{category}}}{\sigma_t}\right)} \in (0, 1)$$
  $$\frac{\partial \mathbb{P}(\text{UP})}{\partial \mathcal{S}_i} > 0, \quad \lim_{\mathcal{S}_i \to +\infty} \mathbb{P}(\text{UP}) = 1.0, \quad \lim_{\mathcal{S}_i \to -\infty} \mathbb{P}(\text{UP}) = 0.0$$

### Invariant 6: Sub-10ms Pipeline Latency SLA (`INV-NEWS-006`)
- From raw RSS XML parsing to entity extraction, sentiment scoring, and price reaction prediction, the entire pipeline must execute in under **10 milliseconds** per headline ($< 10\text{ms}$).

---

## 3. Deterministic Diagnostic Fault Matrix

Following `Rules.md` Rule 2, all operational error conditions are mapped to deterministic fault codes:

| Fault Code | Exception Class | Trigger Condition | Architectural Action |
| :--- | :--- | :--- | :--- |
| `ERR-NEWS-001` | `NewsFeedUnreachableException` | HTTP connection failure / DNS timeout to external RSS wire. | Fallback to secondary wire / log transport warning. |
| `ERR-NEWS-002` | `CorruptNewsPayloadException` | Malformed XML / empty headline / unparseable payload. | Quarantine payload and skip to next item. |
| `ERR-NEWS-003` | `FutureTimestampException` | Article publication timestamp is in the future ($t > t_{\text{now}}$). | Reject article with zero lookahead alert (`INV-NEWS-001`). |
| `ERR-NEWS-004` | `EmptyUniverseException` | Extracted tickers cannot be resolved against system universe. | Classify as broad macro sentiment or log unmapped asset. |
| `ERR-NEWS-005` | `NonFiniteSignalException` | NaN, $\pm\infty$, or boolean detected in sentiment or price inputs. | Reject calculation (`INV-NEWS-003`). |
| `ERR-NEWS-006` | `ReactionPredictionTimeoutException` | Prediction pipeline exceeds execution deadline ($> 50\text{ms}$). | Abort prediction and return neutral prior. |

---

## 4. Component Topology & System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                        PHASE 9: REAL-WORLD NEWS & PRICE REACTION ENGINE                           |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Live Financial RSS Feeds]                                                                       |
|  |-- Yahoo Finance RSS (General + Ticker Streams)                                                 |
|  |-- SEC EDGAR 8-K Regulatory Wire (Material Events)                                              |
|  \-- Financial News RSS (CNBC, MarketWatch)                                                       |
|         |                                                                                         |
|         v (Async HTTP/2 Polling via httpx)                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | NewsHarvester (src/quant/data/news_harvester.py)                                            |  |
|  | - SHA-256 Deduplication Ring Buffer (INV-NEWS-002)                                          |  |
|  | - HTML Tag Sanitization & Content Normalization                                             |  |
|  | - Causal Point-in-Time Timestamp Verification (INV-NEWS-001)                                 |  |
|  | - Synthetic Replay Fallback Generator (Hermetic Testing)                                    |  |
|  +---------------------------------------------------------------------------------------------+  |
|         |                                                                                         |
|         v (Normalized NewsArticle Stream)                                                         |
|  +---------------------------------------------------------------------------------------------+  |
|  | FinancialSentimentClassifier (src/quant/analytics/news_classifier.py)                       |  |
|  | - Entity Extraction: Ticker Regex ($AAPL) + Company Aliases ("Apple" -> AAPL)               |  |
|  | - Centrality Scoring: Headline (1.0), Context (0.5), Macro Proxy (SPY/QQQ)                   |  |
|  | - Event Taxonomy: EARNINGS, MACRO_FED, M_AND_A, REGULATORY_LEGAL, ANALYST_ACTION, GENERAL   |  |
|  | - Loughran-McDonald Lexicon: Polarity s in [-1, 1] with Negation Handling ("loss narrowed")  |  |
|  | - Urgency u in [0, 1] & Novelty H in [0, 1]                                                 |  |
|  +---------------------------------------------------------------------------------------------+  |
|         |                                                                                         |
|         v (ClassifiedNewsEvent)                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | NewsPriceReactionEngine (src/quant/analytics/price_reaction.py)                             |  |
|  | - Expected Dollar Shock: Delta P_expected = P_t * gamma_cat * S_i * sigma_t                 |  |
|  | - Logistic Breakout Probability: P(UP) = 1 / (1 + exp(-lambda * S_i * gamma / sigma_t))      |  |
|  | - Causal Decay Trajectory: Delta P(Delta t) via Dual Kernel kappa(Delta t, u)               |  |
|  | - Triple-Barrier Up/Down Targets: P_upper = P_t * e^(+c*sigma), P_lower = P_t * e^(-c*sigma)|  |
|  +---------------------------------------------------------------------------------------------+  |
|         |                                                                                         |
|         +---> [EventService.ingest_batch] (Persists NewsEvent & EventCentrality in DB)           |
|         |                                                                                         |
|         +---> [AutonomousTradingEngine] (Injects Forward Alpha Prior mu_news into RD-DMA Swarm)   |
|         |                                                                                         |
|         v                                                                                         |
|  [Presentation Layer: /api/v1/news & Trading Terminal HUD Feed Widget]                            |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Mathematical Formulations

### 5.1 Composite News Shock Tensor
For news item $i$ targeting asset $k$:
$$\mathcal{S}_{i, k} = \text{Polarity}_i \cdot \text{Urgency}_i \cdot \text{Novelty}_i \cdot c_{i, k}$$
where:
* $\text{Polarity}_i = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{pos}} + N_{\text{neg}} + \epsilon} \in [-1.0, 1.0]$ using Loughran-McDonald dictionaries with two-word forward lookaround for negation phrases (*"not profitable"*, *"failed to deliver"*, *"loss narrowed"*).
* $\text{Urgency}_i \in [0.1, 0.95]$ dynamically assigned by category priority:
  $$\text{Urgency} = \begin{cases} 0.95 & \text{SEC 8-K Unscheduled Filings} \\ 0.90 & \text{Earnings Releases} \\ 0.85 & \text{FOMC / Fed Rate Decision} \\ 0.60 & \text{Analyst Rating Changes} \\ 0.30 & \text{General Industry Commentary} \end{cases}$$
* $\text{Novelty}_i = \exp(-\Delta t_{\text{similar}} / \tau_{\text{novelty}}) \in [0.1, 1.0]$.
* $c_{i, k} \in [0.0, 1.0]$ denotes entity centrality.

### 5.2 Category Elasticity ($\gamma_{\text{category}}$)
The empirical price multiplier scales how violently a given news class impacts price discovery relative to baseline Parkinson volatility $\sigma_t$:
$$\gamma = \begin{cases} 3.5 & \text{M\_AND\_A (Mergers \& Acquisitions)} \\ 2.5 & \text{EARNINGS (Quarterly Reports \& Guidance)} \\ 2.0 & \text{MACRO\_FED (Interest Rates \& CPI)} \\ 1.2 & \text{REGULATORY\_LEGAL (Antitrust / FDA)} \\ 0.8 & \text{ANALYST\_ACTION (Upgrades / Downgrades)} \\ 0.4 & \text{GENERAL\_MARKET (Broad Market Sentiment)} \end{cases}$$

### 5.3 Predicted Directional Price Move & Triple-Barrier Probabilities
The expected price change at the peak of the post-announcement reaction is:
$$\Delta \hat{P}_{\text{expected}} = P_t \cdot \left( \gamma_{\text{category}} \cdot \mathcal{S}_{i, k} \cdot \sigma_t \right)$$
$$\hat{P}_{\text{target}} = P_t + \Delta \hat{P}_{\text{expected}}$$

Under the dynamic volatility Triple-Barrier framework with barrier width multiplier $c$ (default $c = 2.0$):
* **Upper Profit Barrier**: $B_{\text{upper}} = P_t \cdot \exp(+c \cdot \sigma_t)$
* **Lower Stop Barrier**: $B_{\text{lower}} = P_t \cdot \exp(-c \cdot \sigma_t)$

The probability of hitting the Upper Barrier first follows a drift-diffusion logistic distribution:
$$\mathbb{P}(\text{Market Goes UP}) = \frac{1}{1 + \exp\left(-\frac{\lambda \cdot \mathcal{S}_{i, k} \cdot \gamma_{\text{category}}}{\sigma_t}\right)}$$
$$\mathbb{P}(\text{Market Goes DOWN}) = 1.0 - \mathbb{P}(\text{Market Goes UP})$$
- $\mathbb{P}(\text{UP}) \ge 0.65 \implies \mathbf{BUY\ SIGNAL}$
- $\mathbb{P}(\text{UP}) \le 0.35 \implies \mathbf{SELL\ /\ SHORT\ SIGNAL}$
- $0.35 < \mathbb{P}(\text{UP}) < 0.65 \implies \mathbf{NEUTRAL\ /\ NO\ TRADE}$

### 5.4 Temporal Reaction & Post-Announcement Drift (PEAD) Trajectory
The expected price path over elapsed time $\Delta t$ following publication is modeled by modulating the peak shock with the dual-decay kernel $\kappa(\Delta t, u)$:
$$\hat{P}(t + \Delta t) = P_t + \Delta \hat{P}_{\text{expected}} \cdot \kappa(\Delta t, u)$$
$$\kappa(\Delta t, u) = \alpha \exp\left(-\frac{\Delta t}{\tau_{\text{fast}}(1-u)}\right) + (1-\alpha)\left(1 + \frac{\Delta t}{\tau_{\text{slow}}}\right)^{-\beta}$$
This formulation captures:
1. **Immediate Reaction Jump**: Rapid absorption over horizon $\tau_{\text{fast}}$ (minutes).
2. **Post-Announcement Drift**: Long-memory momentum drift over $\tau_{\text{slow}}$ (hours/days).

---

## 6. Verification & Quality Gates

1. **Unit Testing Suite**:
   - `test_news_harvester.py`: RSS XML parsing, deduplication ring buffer, point-in-time timestamp causality, error handling, synthetic replay fallback.
   - `test_news_classifier.py`: Entity extraction, Loughran-McDonald sentiment, negation handling, category classification, non-finite rejection.
   - `test_price_reaction.py`: Directional shock computation, category elasticity scaling, logistic breakout probability bounds, decay trajectory, and Triple-Barrier alignment.
2. **API & Integration Testing**:
   - `test_news_api.py`: `/api/v1/news/harvest`, `/api/v1/news/predict`, `/api/v1/news/latest` endpoints with RBAC authentication.
   - Test coupling with `AutonomousTradingEngine` verifying forward alpha prior injection.
3. **Static Analysis**:
   - 100% Python 3.13 strict static typing (`mypy src --strict` with 0 errors).
   - Zero linter or formatting deviations (`ruff check .`, `ruff format --check .`).
4. **Latency Benchmark**:
   - Pipeline latency $< 10\text{ms}$ per article execution (`INV-NEWS-006`).
