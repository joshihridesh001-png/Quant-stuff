# Institutional Model Context Protocol (MCP) AI Agent Integration Guide

## 1. Executive Summary & Architectural Mandate

The Model Context Protocol (MCP 2024-11-05) establishes an institutional-grade, cryptographically authenticated bridge between autonomous frontier Large Language Models (LLMs)—including Claude Desktop, Cursor, Gemini, and Google Antigravity—and the quantitative execution engine.

Rather than granting LLMs ambient, unconstrained access to execution gateways or relying on fragile ad-hoc function calling, the institutional MCP architecture enforces:
1. **Zero-Trust Cryptographic Access Control**: Role-based access control (RBAC) via short-lived HMAC-SHA256 JWT tokens distinguishing read-only `RESEARCHER` capabilities from mutating `ADMIN` powers.
2. **Deterministic Pre-Flight Schema Verification (`INV-MCP-001`)**: Client-side rejection of hallucinated keys, missing required parameters, and non-finite floats (`NaN`, `Inf`) prior to network transmission.
3. **Mandatory Bayesian Pre-Trade Risk Simulation (`INV-GATE-001` to `INV-GATE-004`)**: Autonomous agents are structurally prohibited from placing live orders without first submitting candidate trades to a 5-dimensional Bayesian log-odds decision gate.
4. **Deterministic Diagnostic Error Codes (Rule 2)**: Strict mapping of operational failures to standardized diagnostic codes (`ERR-MCP-001` through `ERR-MCP-005`).

---

## 2. First-Principles Critique: Naive Tool Calling vs. Institutional MCP

Under **Rule 4 (Mandatory Adversarial Red-Teaming & "Best of the Best" First-Principles Architecture)**, trading systems must actively assume an adversarial market and hostile operational conditions. The table below exposes the structural flaws in standard LLM tool calling and articulates the institutional MCP solution.

| Vulnerability Dimension | Naive Tool Calling (Textbook / Consensus) | Structural Failure Mode | Institutional MCP Alternative ("Best of the Best") |
| :--- | :--- | :--- | :--- |
| **Authentication & Authorization** | Ambient tool execution with no caller authentication or hardcoded API keys shared across tools. | Prompt injection or context drift allows an unprivileged assistant to invoke emergency liquidation or parameter resets. | **Cryptographic JWT Tokens + Scoped RBAC**: Read-only tools (`quant_portfolio_telemetry`, `quant_market_orderbook`) accessible to `RESEARCHER`; destructive kill switches (`quant_emergency_panic`, `quant_kill_switch_reset`) restricted strictly to `ADMIN` credentials. |
| **Parameter Integrity** | Unchecked parsing of LLM JSON tool arguments; string-to-float conversions directly into order payloads. | Hallucinated `NaN`, negative quantities, or missing keys cause `ValueError`, zero-division errors, or corrupted order parameters in the hot path. | **Two-Tier Pre-Flight Schema Validation**: `MCPClient._validate_tool_call` enforces non-empty strings, finite positive floats, and strict range clamping (`OBI` in $[-1.0, 1.0]$) before dispatch (`ERR-MCP-005`). |
| **Order Placement Safety** | LLM agent directly calls `place_market_order(symbol, qty, side)` without state verification. | Flash crash slippage, adverse selection into toxic order book queues, or order placement during circuit-breaker halts or inverted yield curve regimes. | **Mandatory Closed-Form Pre-Trade Decision Gate**: Agents must invoke `quant_evaluate_pre_trade` which scores data freshness, order book toxicity, macro yield spread, momentum, and drawdown budgets in $< 50\mu\text{s}$ before any order can be routed. |
| **Transport & Session Drops** | Fragile HTTP `fetch` or unmanaged TCP sockets with indefinite timeouts; silent agent hangs. | Network disconnects or gateway stalls cause agent hangs, unclosed positions, and unmonitored risk exposure. | **Dual-Transport Layer with Auto-Renewal**: Supports both local non-blocking stdio subprocess pipes and authenticated HTTP JSON-RPC (`POST /api/v1/mcp/rpc`) with automated 401 token refresh (`ERR-MCP-001` / `ERR-MCP-003`). |
| **Diagnostics & Observability** | Arbitrary Python traceback strings returned to the LLM. | Agent enters infinite hallucination retry loops attempting to correct syntax errors without understanding underlying risk constraints. | **Deterministic Diagnostic Failure Codes**: Standard JSON-RPC 2.0 error payloads (`-32600` to `-32603`) mapped to hierarchical codes `ERR-MCP-001` through `ERR-MCP-005` with exact root causes cataloged in `Memory.md`. |

---

## 3. Protocol Specification & Tool Catalog

The quantitative MCP server exposes 7 institutional tools conforming to JSON-RPC 2.0 and MCP protocol version `2024-11-05`:

### 3.1 `quant_portfolio_telemetry`
* **Role Required**: `RESEARCHER` or `ADMIN`
* **Purpose**: Query real-time firm-wide portfolio valuation, cash balance, current positions, leverage, and circuit breaker status.
* **Input Schema**: `{}` (No parameters required)
* **Output Payload**:
  ```json
  {
    "nav": 10000.0,
    "peak_nav": 10000.0,
    "cash": 3965.27,
    "margin_used": 6034.73,
    "free_margin": 3965.27,
    "gross_notional": 6034.73,
    "net_notional": 6034.73,
    "gross_leverage": 0.60,
    "net_leverage": 0.60,
    "intraday_drawdown_pct": 0.0,
    "is_kill_switch_active": false,
    "kill_switch_trigger": null,
    "open_leaves_count": 0
  }
  ```

### 3.2 `quant_macro_regimes`
* **Role Required**: `RESEARCHER` or `ADMIN`
* **Purpose**: Inspect macroeconomic indicators from the Federal Reserve (FRED), specifically the 10Y-2Y Treasury yield curve spread (`T10Y2Y`), effective Federal Funds rate (`DFF`), and curve inversion status.
* **Input Schema**: `{}`
* **Output Payload**:
  ```json
  {
    "t10y2y_yield_spread": 0.18,
    "t10y2y_date": "2026-09-24",
    "dff_effective_rate": 5.25,
    "dff_date": "2026-09-24",
    "is_yield_curve_inverted": false
  }
  ```

### 3.3 `quant_market_orderbook`
* **Role Required**: `RESEARCHER` or `ADMIN`
* **Purpose**: Query consolidated quote, high/low, volume, and VWAP for a target equity symbol.
* **Input Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "symbol": { "type": "string", "description": "Ticker symbol (e.g. SPY, AAPL)" }
    },
    "required": ["symbol"]
  }
  ```
* **Output Payload**:
  ```json
  {
    "symbol": "SPY",
    "close": 508.78,
    "high": 510.15,
    "low": 507.40,
    "volume": 13300,
    "vwap": 509.39,
    "timestamp_ns": 1727200000000000000
  }
  ```

### 3.4 `quant_evaluate_pre_trade`
* **Role Required**: `RESEARCHER` or `ADMIN`
* **Purpose**: Evaluate a proposed order through the 5-dimensional Bayesian log-odds decision gate before execution.
* **Input Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "symbol": { "type": "string" },
      "action": { "type": "string", "enum": ["BUY", "SELL", "CLOSE", "LIQUIDATE"] },
      "quantity": { "type": "number", "description": "Positive share quantity" },
      "reference_price": { "type": "number", "description": "Current market price" },
      "is_position_exit": { "type": "boolean", "default": false },
      "market_spread_bps": { "type": "number", "default": 5.0 },
      "order_book_imbalance": { "type": "number", "default": 0.0 },
      "macro_yield_spread": { "type": "number", "default": 0.18 }
    },
    "required": ["symbol", "action", "quantity", "reference_price"]
  }
  ```
* **Output Payload**:
  ```json
  {
    "decision_id": "DEC-1727200123456",
    "allowed": true,
    "decision": "PASS",
    "toxicity_probability": 0.10,
    "confidence": 0.846,
    "primary_code": "OK",
    "reason": "Order passed Bayesian gate (Toxicity=10.0%)",
    "latency_us": 50.6,
    "checks": [
      { "name": "data_freshness", "passed": true, "score": 0.0, "details": "Data age 1.0s" },
      { "name": "microstructure_toxicity", "passed": true, "score": 0.0, "details": "OBI=0.08, Spread=4.5bp" },
      { "name": "macro_regime", "passed": true, "score": 0.0, "details": "T10Y2Y yield spread=0.18%" },
      { "name": "momentum_alignment", "passed": true, "score": 0.0, "details": "Bar return=0.05%, Vol=13.3k" },
      { "name": "capital_risk_budget", "passed": true, "score": 0.0, "details": "Streak=0, Drawdown=0.0%" }
    ]
  }
  ```

### 3.5 `quant_swarm_status`
* **Role Required**: `RESEARCHER` or `ADMIN`
* **Purpose**: Inspect the autonomous trading swarm daemon lifecycle state, iteration counter, and allocated universe.
* **Input Schema**: `{}`
* **Output Payload**:
  ```json
  {
    "state": "IDLE",
    "iteration": 0,
    "universe": ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]
  }
  ```

### 3.6 `quant_emergency_panic`
* **Role Required**: `ADMIN` or `SYSTEM` (**Strictly Restricted**)
* **Purpose**: EMERGENCY KILL SWITCH. Immediately cancels all resting orders across all execution gateways and halts new order submissions.
* **Input Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "reason": { "type": "string", "description": "Etiology for emergency panic halt" },
      "details": { "type": "string", "description": "Additional context or incident details" }
    },
    "required": ["reason"]
  }
  ```

### 3.7 `quant_kill_switch_reset`
* **Role Required**: `ADMIN` or `SYSTEM` (**Strictly Restricted**)
* **Purpose**: Disarm emergency kill switch back to `ARMED_STANDBY` using authenticated administrator secret token.
* **Input Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "admin_token": { "type": "string", "description": "Cryptographic administrator secret token" }
    },
    "required": ["admin_token"]
  }
  ```

---

## 4. Claude Desktop Integration

Claude Desktop integrates natively with the quantitative engine over standard input/output (stdio) transport.

### 4.1 Configuration File Location
* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json` (e.g. `C:\Users\<User>\AppData\Roaming\Claude\claude_desktop_config.json`)
* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Linux**: `~/.config/Claude/claude_desktop_config.json`

### 4.2 Exact `claude_desktop_config.json` Configuration

#### Windows Configuration:
```json
{
  "mcpServers": {
    "quant-engine": {
      "command": "C:\\Users\\jishu\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": [
        "-m",
        "quant.mcp.server"
      ],
      "cwd": "C:\\Users\\jishu\\OneDrive\\Documents\\quant",
      "env": {
        "PYTHONPATH": "C:\\Users\\jishu\\OneDrive\\Documents\\quant\\src",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

#### macOS / Linux Configuration:
```json
{
  "mcpServers": {
    "quant-engine": {
      "command": "/usr/local/bin/python3",
      "args": [
        "-m",
        "quant.mcp.server"
      ],
      "cwd": "/path/to/quant",
      "env": {
        "PYTHONPATH": "/path/to/quant/src",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

---

## 5. Cursor IDE Integration

Cursor IDE natively supports Model Context Protocol servers configured through its MCP settings.

### 5.1 Configuration via Cursor Settings
1. Open Cursor Settings: `Ctrl + ,` (or `Cmd + ,` on macOS).
2. Navigate to **Features** -> **MCP**.
3. Click **Add New MCP Server**.
4. Configure as follows:
   * **Name**: `quant-engine`
   * **Type**: `command` (stdio)
   * **Command**: `python -m quant.mcp.server`

### 5.2 Cursor MCP JSON Configuration (`.cursor/mcp.json`)
In the repository root or project workspace, create `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "quant-engine": {
      "command": "python",
      "args": [
        "-m",
        "quant.mcp.server"
      ],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Alternatively, if connecting over the authenticated HTTP Gateway:
```json
{
  "mcpServers": {
    "quant-engine-http": {
      "url": "http://127.0.0.1:8000/api/v1/mcp/rpc",
      "headers": {
        "Authorization": "Bearer <YOUR_JWT_ACCESS_TOKEN>"
      }
    }
  }
}
```

---

## 6. Deterministic Diagnostic Fault Matrix (Rule 2)

Operational errors across MCP client and server components are assigned deterministic diagnostic codes:

| Fault Vector ID | Component | Malfunction Symptoms | Root Cause Etiology | Zero-Runtime Verification | Remediation Procedure |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ERR-MCP-001`** | `MCPClient.login` / `_send_rpc_http` | `MCPAuthFailureException`: 401/403 HTTP response | Invalid credentials or expired JWT unable to be renewed automatically. | Verify `username`, `password`, and JWT expiration timestamp in claims payload. | Re-authenticate using valid credentials via `POST /api/v1/auth/token`. |
| **`ERR-MCP-002`** | `MCPClient._send_rpc` | `MCPProtocolException`: Missing `result`, bad ID, or version mismatch | Upstream server emitted non-JSON-RPC 2.0 payload or JSON parsing error. | Check that response contains valid `jsonrpc: "2.0"` and matching `id`. | Update MCP server to conform strictly to JSON-RPC 2.0 and MCP 2024-11-05 specifications. |
| **`ERR-MCP-003`** | `MCPClient.connect` / `_send_rpc` | `MCPTransportException`: Connection refused, timeout, or stdio pipe EOF | Web server offline, subprocess crashed, or network timeout exceeded. | Test endpoint connectivity with `curl http://127.0.0.1:8000/healthz` or inspect child process exit code. | Restart API server daemon or check system resources and network gateway. |
| **`ERR-MCP-004`** | `MCPClient.call_tool` | `MCPToolExecutionException`: `isError: true` in response | Target tool logic threw an unhandled exception or rejected operation. | Inspect `content[0].text` error message in tool result. | Fix underlying engine condition (e.g. invalid symbol, database lock, or permission denial). |
| **`ERR-MCP-005`** | `MCPClient._validate_tool_call` | `MCPSchemaValidationException`: Argument failed bounds check | Non-finite scalar (`NaN`, `Inf`), negative size, or missing required parameter. | Inspect client arguments against tool schema requirements. | Sanitize agent inputs; ensure positive finite numbers and valid enum actions. |

---

## 7. Running the Autonomous Agent Decision Loop

The workspace provides an autonomous agent script `scripts/run_mcp_agent.py` executing continuous observation and pre-trade simulation cycles.

### 7.1 Running via Stdio Transport (Local Standalone)
```bash
python scripts/run_mcp_agent.py --transport stdio --symbol SPY --quantity 25.0
```

### 7.2 Running via HTTP Gateway (Authenticated Remote)
First ensure the FastAPI server is running:
```bash
uvicorn quant.main:app --host 127.0.0.1 --port 8000
```
Then run the agent loop:
```bash
python scripts/run_mcp_agent.py --transport http --base-url http://127.0.0.1:8000 --role ADMIN --symbol SPY --quantity 10.0
```

### 7.3 Multi-Cycle Continuous Observation
To run continuous background monitoring (e.g. 5 iterations every 10 seconds):
```bash
python scripts/run_mcp_agent.py --transport stdio --symbol NVDA --quantity 50.0 --iterations 5 --interval 10.0
```
