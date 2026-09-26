# Data Flow Diagram (DFD) Specification

This document provides the formal **Data Flow Diagram (DFD)** decomposition for the Quantitative Research & Autonomous Live Trading Engine, tracing data movement from external ingestion feeds to alpha models, pre-trade risk validation, order execution, and double-entry accounting.

---

## 1. Level 0: Context Diagram (System Environment)

The Level 0 Context Diagram establishes the operational boundary of the Quantitative Engine and defines interactions with external entities, brokers, news providers, and research operators.

```mermaid
flowchart TD
    %% External Entities
    E1["External Market Feeds (Alpaca, Yahoo, Polygon)"]
    E2["Financial News Sources (Finnhub, RSS, SEC Filings)"]
    E3["Execution Venues & Brokers (Alpaca, FIX 4.4 / 5.0 Gateway)"]
    E4["Quantitative Researchers & Operators (Web / CLI)"]
    E5["Autonomous AI Agents (MCP 2024-11-05 Tools)"]

    %% Core System
    CORE[["QUANTITATIVE TRADING & RESEARCH ENGINE (Core System)"]]

    %% Level 0 Data Flows
    E1 -->|"Raw L1/L2 Quotes & Trade Ticks"| CORE
    E2 -->|"Unstructured News Headlines & RSS Payloads"| CORE
    
    CORE -->|"Child Orders & Mass Cancellation Sweeps"| E3
    E3 -->|"Execution Reports, Fill Events & COD Heartbeats"| CORE

    E4 -->|"Strategy Configs, Manual Rebalancing, Panic Hotkeys"| CORE
    CORE -->|"60 FPS Lightweight Charts, Risk HUD, CFA Tearsheets"| E4

    E5 -->|"JSON-RPC Tool Invocations (pre-trade, regime query)"| CORE
    CORE -->|"Structured Telemetry, Order Book Depth & Risk Verifications"| E5
```

---

## 2. Level 1: System Decomposition DFD (Subsystems & Data Stores)

Level 1 breaks down the engine into **8 primary transformational processes** and **5 core data stores**, mapping how data flows from ingestion through mathematical models to order execution and accounting.

```mermaid
flowchart TD
    %% External Entities
    EE_FEED["Market Data Providers"]
    EE_NEWS["News & Macro Providers"]
    EE_VENUE["Brokers / Venues"]
    EE_USER["Researcher / Browser"]

    %% Data Stores
    subgraph DATA_STORES ["Data Stores"]
        D1[("D1: DuckDB Parquet Historical Lake")]
        D2[("D2: PostgreSQL / SQLite Double-Entry Ledger")]
        D3[("D3: In-Memory L2 Order Book Ring Buffers")]
        D4[("D4: Genotype Chromosome Repository")]
        D5[("D5: Risk Audit Trail & Event Store")]
    end

    %% Processes
    subgraph INGESTION ["1. Ingestion & Feature Engineering"]
        P1["1.0 Ingestion & Monotonicity Normalizer"]
        P2["2.0 Causal News & Sentiment Decay Engine"]
        P3["3.0 Econometric Stationarity & Volatility Rig"]
    end

    subgraph ALPHA ["2. Alpha Generation & Optimization"]
        P4["4.0 Bayesian Regime & Jump Classifier"]
        P5["5.0 Evolutionary Swarm & Pareto Optimizer"]
        P6["6.0 Convex Portfolio Allocator & Risk Sizing"]
    end

    subgraph EXECUTION ["3. Risk Clearance & Execution"]
        P7["7.0 Pre-Trade Risk Firewall & Kill Switch"]
        P8["8.0 Smart Order Router & Execution Slicers"]
        P9["9.0 Double-Entry Ledger & TCA Attribution"]
    end

    %% Data Flows
    EE_FEED -->|"Raw OHLCV & Ticks"| P1
    P1 -->|"Clean Chronological Bars"| D1
    P1 -->|"Real-Time Tick Stream"| D3
    P1 -->|"Stationary Log-Returns"| P3

    EE_NEWS -->|"Raw Articles & Headlines"| P2
    P2 -->|"Loughran-McDonald Sentiment & Decay Vectors"| P4

    D1 -->|"Historical Price Memory"| P3
    P3 -->|"FFD Memory-Preserved Features & Garman-Klass Vol"| P4

    P4 -->|"Posterior Regime Probabilities"| P5
    P4 -->|"Thermodynamic Ambiguity"| P6
    D4 <-->|"Load & Persist Chromosomes"| P5
    P5 -->|"Optimized Multi-Strategy Weights"| P6

    D2 -->|"Cold-Start Positions & Cash Balance"| P6
    P6 -->|"Target Portfolio Delta Orders"| P7

    P7 -->|"Leaves Reservation & Tripwire Clearance"| P8
    P7 -->|"Emergency Cancel Sweeps / Lockdown"| EE_VENUE
    P7 -->|"Audit Violations & Rejection Logs"| D5

    P8 -->|"TWAP / VWAP Sliced Child Orders"| EE_VENUE
    EE_VENUE -->|"Fill Notifications & Partial Executions"| P8
    P8 -->|"Executed Fills & Slippage Metrics"| P9

    P9 -->|"Debit / Credit Postings & Tax-Lots"| D2
    P9 -->|"TCA Attribution & Deflated Sharpe Reports"| EE_USER
    D3 -->|"Real-Time Canvas Candlesticks & Markers"| EE_USER
```

---

## 3. Level 2: Real-Time Execution & Risk Hot-Path (Sub-100μs Microstructure Loop)

Level 2 details the mission-critical hot-path within the autonomous trading loop, pre-trade risk firewall, and smart order router.

```mermaid
flowchart LR
    %% Live Loop Inputs
    INPUT_BAR["PriceBar Stream (t)"] --> STEP1["7.1 Mark-to-Market Price Update"]

    %% Step 1: MTM & Drawdown Check
    subgraph RISK_LOOP ["Pre-Trade Risk Clearance Phase (< 10us)"]
        STEP1 --> STEP2{"Drawdown > Limit?"}
        STEP2 -- Yes --> KILL["7.2 Panic Trigger (EmergencyKillSwitch)"]
        STEP2 -- No --> STEP3{"Watchdog Timeout?"}
        STEP3 -- Yes --> KILL
        STEP3 -- No --> STEP4["7.3 Epistemic Uncertainty Haircut Scaling"]
        
        STEP4 --> STEP5["7.4 Sizing Check: Max Notional & Leverage Limits"]
        STEP5 -- Reject --> ERR["ERR-RSK-002: Limit Breach -> Log Rejection"]
        STEP5 -- Pass --> STEP6["7.5 Atomic Leaves Reservation"]
    end

    %% Step 2: Order Slicing Phase
    subgraph ROUTING_LOOP ["Microstructural Routing Phase (< 50us)"]
        STEP6 --> STEP7{"Algorithm Sizer"}
        STEP7 -- TWAP --> SLICE1["8.1 Poisson Random Interval Slicer"]
        STEP7 -- VWAP --> SLICE2["8.2 Volume Profile Curve Slicer"]
        STEP7 -- Arrival --> SLICE3["8.3 Almgren-Chriss Impact Minimizer"]

        SLICE1 --> DISPATCH["8.4 Gateway Socket Dispatch"]
        SLICE2 --> DISPATCH
        SLICE3 --> DISPATCH
    end

    %% Step 3: Gateway & Fill Reconciliation
    subgraph FILL_LOOP ["Reconciliation & Ledger Phase"]
        DISPATCH --> VENUE["Exchange / Broker (Alpaca, FIX)"]
        VENUE --> ACK["Fill Report / Reject"]
        ACK -- Rejection --> ROLLBACK["7.6 Atomic Leaves Rollback"]
        ACK -- Fill Event --> RECORD["9.1 FIFO Tax-Lot Allocation & Double-Entry Journal"]
        RECORD --> TELEM["Broadcast WebSocket Telemetry (HUD & Chart)"]
    end
```

---

## 4. Data Store Architecture

| Store | Technology | Key Models / Entities | Purpose |
|:---|:---|:---|:---|
| **D1: Historical Lake** | DuckDB + Parquet | `CandlestickBarDTO`, `PriceBar` | Nanosecond-resolution multi-asset historical bar lake partitioned by symbol. |
| **D2: Relational Ledger** | PostgreSQL / SQLite (WAL) | `DBLedgerAccount`, `DBTaxLot`, `DBJournalEntry` | Balanced double-entry financial ledger conserving capital ($\sum \text{Debit} = \sum \text{Credit}$). |
| **D3: Order Book Cache** | In-Memory Ring Buffer | `OrderBookSnapshot`, `ConsolidatedQuote` | Sub-microsecond L2 bid/ask depth and rolling bar cache for indicators. |
| **D4: Genotype Pool** | SQLAlchemy ORM / JSON | `DBGenotype`, `ChromosomeRepr`, `ChromosomeGame` | Multi-generation evolutionary strategy chromosomes and Pareto frontiers. |
| **D5: Audit & Event Store** | Append-Only Table | `KillSwitchEvent`, `PreTradeDecision` | Immutable regulatory audit log tracking pre-trade decisions and cancellations. |

---

## 5. Architectural Defensive Invariants

1. **`INV-DATA-004` (Strict Monotonicity)**: Every timestamp flowing from Ingestion ($1.0$) into Historical Lake ($D1$) must satisfy $t_i > t_{i-1}$; duplicate timestamps trigger deterministic rejection.
2. **`INV-RSK-001` (Zero Leakage Clearance)**: No order can be dispatched across network sockets without successfully acquiring atomic leaves reservation from Process $7.5$.
3. **`INV-RSK-008` (Sub-50ms Kill Switch Mass Cancellation)**: Activation of the Emergency Kill Switch triggers concurrent async cancellation across all broker gateways with socket timeout shields.
4. **`INV-LEDGER-001` (Conservation of Value)**: All executions create balanced compound journal entries with explicit tax-lot depletion (FIFO/LIFO) and zero unbacked cash creation.
