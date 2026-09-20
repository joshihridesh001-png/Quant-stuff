// TypeScript Domain Types matching backend Pydantic schemas

export interface RiskMetrics {
  nav: number;
  peak_nav: number;
  cash: number;
  margin_used: number;
  free_margin: number;
  gross_notional: number;
  net_notional: number;
  gross_leverage: number;
  net_leverage: number;
  intraday_drawdown_pct: number;
  is_kill_switch_active: boolean;
  kill_switch_trigger?: string | null;
  open_leaves_count: number;
  positions?: Record<string, number>;
  current_prices?: Record<string, number>;
  unrealized_pnl?: number;
  realized_pnl?: number;
}

export interface GatewayHealthDTO {
  gateway_name: string;
  is_connected: boolean;
  last_heartbeat_timestamp_ns: number;
  latency_ms: number;
  unacknowledged_orders: number;
}

export interface RiskWebSocketMessage {
  type: 'SNAPSHOT' | 'RISK_UPDATE' | 'KILL_SWITCH_EVENT' | 'PONG';
  timestamp_ns?: number;
  risk?: RiskMetrics;
  gateways?: GatewayHealthDTO[];
  trigger_reason?: string;
  is_active?: boolean;
}

export interface AutonomousStatus {
  state: 'IDLE' | 'RUNNING' | 'PAUSED' | 'STOPPED' | 'ERROR';
  iteration: number;
  universe: string[];
  target_allocations: Record<string, number>;
  last_step?: {
    iteration: number;
    timestamp_ns: number;
    universe: string[];
    target_allocations: Record<string, number>;
    current_positions: Record<string, number>;
    orders_dispatched: string[];
    duration_ms: number;
    haircut: number;
    is_kill_switch_active: boolean;
  } | null;
}

export interface ChildOrderDTO {
  child_id: string;
  quantity: number;
  price: number;
  fee: number;
  timestamp_ns: number;
  spread_slippage: number;
}

export interface ParentOrder {
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  total_quantity: number;
  filled_quantity: number;
  leaves_quantity: number;
  arrival_price: number;
  decision_price: number;
  vwap_execution_price: number;
  total_fees_paid: number;
  max_duration_seconds: number;
  start_time_ns: number;
  is_closed: boolean;
  child_fills: ChildOrderDTO[];
  algorithm?: string;
}

export interface ExecWebSocketMessage {
  type: 'SNAPSHOT' | 'ORDER_UPDATE' | 'PONG';
  timestamp_ns?: number;
  orders?: ParentOrder[];
  order?: ParentOrder;
  event_type?: string;
}

export interface TcaReport {
  order_id: string;
  symbol: string;
  side: string;
  total_quantity: number;
  filled_quantity: number;
  decision_price: number;
  arrival_price: number;
  execution_vwap: number;
  terminal_price: number;
  delay_cost: number;
  price_impact: number;
  spread_slippage: number;
  fees_paid: number;
  opportunity_cost: number;
  total_shortfall: number;
  total_shortfall_bps: number;
  is_additive_conserved: boolean;
}

export interface Genotype {
  id: string;
  generation: number;
  cohort: string;
  chromosome_repr: {
    tau_slow: number;
    tau_ratio: number;
    tau_fast: number;
    alpha: number;
    alpha_decay: number;
    fractional_d: number;
  };
  chromosome_game: {
    ambiguity_temp: number;
    risk_aversion: number;
    risk_aversion_lambda: number;
    predatory_intensity: number;
    logit_bull: number;
    logit_bear: number;
    logit_panic: number;
    belief_prior: number[];
  };
  chromosome_infer: {
    execution_horizon: number;
    hyperbolic_decay: number;
    profit_take_mult: number;
    stop_loss_mult: number;
    holding_period: number;
    meta_label_thresh: number;
    model_depth: number;
    sensitivity: number;
  };
  chromosome_risk: {
    vol_target: number;
    max_weight: number;
    max_drawdown_limit: number;
    turnover_budget: number;
  };
  fitness_score?: number | null;
  deflated_sharpe?: number | null;
  max_drawdown?: number | null;
  regret_score?: number | null;
  novelty_score?: number | null;
  created_at: string;
}

export interface GenotypeItem {
  genotype: Genotype;
  pareto_rank: number;
  crowding_distance: number;
}

export interface SimulationRunRequest {
  asset_id: string;
  bar_count: number;
  initial_capital: number;
  fee_bps: number;
  spread_bps: number;
  impact_coefficient: number;
}

export interface SimulationRunResponse {
  asset_id: string;
  bar_count: number;
  initial_capital: number;
  terminal_equity: number;
  total_return_pct: number;
  cagr_pct: number;
  annualized_volatility_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown_pct: number;
  realized_var_95_pct: number;
  realized_cvar_95_pct: number;
  deflated_sharpe_ratio: number;
  is_statistically_significant: boolean;
  total_friction_cost: number;
  equity_curve: number[];
  benchmark_equity_curve?: number[];
}

export interface NewsItem {
  headline: string;
  summary?: string;
  source: string;
  url?: string;
  datetime: number;
  sentiment_score?: number;
  sentiment_label?: string;
}

export interface NewsDecayState {
  ticker: string;
  event_count: number;
  state_vector: number[];
  fast_half_life_hours: number;
  slow_half_life_hours: number;
}

export interface PreTradeDecision {
  decision_id: string;
  symbol: string;
  action: string;
  allowed: boolean;
  decision: string;
  toxicity_probability: number;
  latency_us: number;
  reason: string;
  timestamp_ns: number;
}
