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
  gateway_id: string;
  gateway_name?: string;
  status?: string;
  is_connected: boolean;
  last_heartbeat_timestamp?: number;
  last_heartbeat_timestamp_ns?: number;
  last_latency_ms?: number;
  latency_ms?: number;
  missed_sequence_count?: number;
  unacknowledged_orders?: number;
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
  type: 'SNAPSHOT' | 'ORDER_UPDATE' | 'CHILD_FILL' | 'PONG';
  timestamp_ns?: number;
  orders?: ParentOrder[];
  order?: ParentOrder;
  event_type?: string;
  parent_id?: string;
  fill?: ChildOrderDTO;
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

export interface SimulationBenchmarkDTO {
  name: string;
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  alpha: number;
  beta: number;
  information_ratio: number;
}

export interface SimulationRunResponse {
  asset_id: string;
  bar_count: number;
  initial_capital: number;
  final_equity: number;
  total_return_pct: number;
  cagr_pct: number;
  annualized_volatility_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown_pct: number;
  realized_cvar_95_pct: number;
  deflated_sharpe_ratio: number;
  is_statistically_significant: boolean;
  total_friction_cost: number;
  equity_curve: number[];
  benchmarks: SimulationBenchmarkDTO[];
}

export interface NewsItem {
  headline: string;
  summary?: string;
  source: string;
  url?: string;
  datetime?: number;
  published_at?: string;
  sentiment_score?: number;
  sentiment_label?: string;
}

export interface PriceReactionPredictionDTO {
  ticker: string;
  event_type: string;
  current_price: number;
  expected_delta_price: number;
  target_price: number;
  prob_up: number;
  prob_down: number;
  barrier_upper: number;
  barrier_lower: number;
  signal: string;
  confidence: number;
  predicted_trajectory: [number, number][];
  calculation_latency_ms: number;
  created_at: string;
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

export interface FFDScanPoint {
  d: number;
  adf_stat: number;
  adf_pvalue: number;
  correlation: number;
  is_stationary: boolean;
}

export interface FFDScanResponse {
  symbol: string;
  optimal_d: number;
  threshold_pvalue: number;
  points: FFDScanPoint[];
}

export interface VolatilitySeriesPoint {
  timestamp_ns: number;
  close: number;
  parkinson_vol: number;
  garman_klass_vol: number;
}

export interface VolatilityResponse {
  symbol: string;
  window: number;
  latest_close: number;
  annualized_parkinson_pct: number;
  annualized_garman_klass_pct: number;
  series: VolatilitySeriesPoint[];
}

export interface TripleBarrierSimulateRequest {
  symbol: string;
  profit_multiplier: number;
  stop_multiplier: number;
  horizon_bars: number;
  volatility_window: number;
  side: number;
}

export interface BarrierTrajectoryPoint {
  bar_index: number;
  close_price: number;
  upper_barrier: number;
  lower_barrier: number;
  event_type: 'TAKE_PROFIT' | 'STOP_LOSS' | 'EXPIRATION' | null;
}

export interface TripleBarrierSimulateResponse {
  symbol: string;
  total_events: number;
  take_profit_hits: number;
  stop_loss_hits: number;
  vertical_expiration_hits: number;
  take_profit_pct: number;
  stop_loss_pct: number;
  vertical_expiration_pct: number;
  average_holding_bars: number;
  average_net_return_pct: number;
  sample_trajectory: BarrierTrajectoryPoint[];
}

export interface CUSUMPoint {
  timestamp_ns: number;
  bar_index: number;
  s_pos: number;
  s_neg: number;
  return_pct: number;
  is_shock: boolean;
}

export interface RegimeHistoryPoint {
  timestamp_ns: number;
  bar_index: number;
  p_absorption: number;
  p_momentum: number;
  p_panic: number;
}

export interface RegimeStatusResponse {
  symbol: string;
  current_regime: 'LOW_VOL_ABSORPTION' | 'MOMENTUM_CASCADE' | 'PANIC_LIQUIDITY_TRAP';
  p_absorption: number;
  p_momentum: number;
  p_panic: number;
  dirichlet_alphas: number[];
  confidence_pct: number;
  thermodynamic_beta: number;
  cusum_alarm: boolean;
  cusum_threshold_h: number;
  cusum_s_pos: number;
  cusum_s_neg: number;
  cusum_series: CUSUMPoint[];
  regime_history: RegimeHistoryPoint[];
}

export interface PayoffMatrixResponse {
  actions: string[];
  counterparties: string[];
  payoff_matrix: number[][];
  regret_matrix: number[][];
  worst_case_regrets: number[];
  optimal_action: string;
  worst_case_counterparty_probs: number[];
}

export interface BacktestSummaryMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  annualized_return: number;
  annualized_volatility: number;
  win_rate: number;
  profit_factor: number;
  deflated_sharpe_ratio?: number | null;
  alpha?: number | null;
  beta?: number | null;
  total_trades: number;
  initial_capital: number;
  final_equity: number;
  net_pnl: number;
}

export interface BacktestStatusResponse {
  backtest_id: string;
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  progress: number;
  message: string;
  strategy_type: string;
  symbols: string[];
  metrics?: BacktestSummaryMetrics | null;
  equity_curve?: number[] | null;
  benchmark_equity_curve?: number[] | null;
  drawdown_series?: number[] | null;
  monthly_matrix?: Record<string, Record<string, number>> | null;
  created_at: string;
  completed_at?: string | null;
  html_report_path?: string | null;
}

export interface BacktestRunRequest {
  strategy_type: string;
  symbols: string[];
  start_date?: string | null;
  end_date?: string | null;
  initial_cash: number;
  benchmark_symbol: string;
  cost_bps: number;
  parameters?: Record<string, any>;
}

export interface BacktestRunResponse {
  backtest_id: string;
  status: string;
  message: string;
  estimated_duration_sec: number;
}
