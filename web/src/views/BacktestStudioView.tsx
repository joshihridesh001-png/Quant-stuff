import React, { useState, useEffect, useCallback } from 'react';
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  BarChart3,
  Play,
  Download,
  Calendar,
  TrendingUp,
  Percent,
  Clock,
  ExternalLink,
} from 'lucide-react';
import { quantApi } from '../api/client';
import {
  BacktestStatusResponse,
  BacktestRunRequest,
  BacktestRunResponse,
} from '../types/quant';

const STRATEGIES = [
  { id: 'FracDiff_Swarm', label: 'Composite Swarm Meta-Strategy (RD-DMA & Mirror Descent)' },
  { id: 'Kalman_StatArb', label: 'Kalman Filter Cointegration Pairs Trading' },
  { id: 'FracDiff_Momentum', label: 'Memory-Preserving FracDiff Momentum' },
  { id: 'Loughran_Sentiment', label: 'Loughran-McDonald News Sentiment Breakout' },
  { id: 'Vol_Breakout', label: 'Volatility Breakout & Bollinger Range Squeeze' },
];

export const BacktestStudioView: React.FC = () => {
  // Form State
  const [strategyType, setStrategyType] = useState<string>('FracDiff_Swarm');
  const [symbolsInput, setSymbolsInput] = useState<string>('SPY, QQQ, AAPL, NVDA, MSFT');
  const [initialCash, setInitialCash] = useState<number>(100000);
  const [benchmarkSymbol, setBenchmarkSymbol] = useState<string>('SPY');
  const [costBps, setCostBps] = useState<number>(2.0);

  // Execution & Results State
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [activeBacktest, setActiveBacktest] = useState<BacktestStatusResponse | null>(null);
  const [history, setHistory] = useState<BacktestStatusResponse[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Fetch recent history
  const fetchHistory = useCallback(async () => {
    try {
      const res = await quantApi.getBacktestHistory<BacktestStatusResponse[]>();
      if (Array.isArray(res)) {
        setHistory(res);
        if (res.length > 0 && !activeBacktest) {
          setActiveBacktest(res[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch backtest history:', err);
    }
  }, [activeBacktest]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Execute Backtest
  const handleRunBacktest = async () => {
    setIsRunning(true);
    setErrorMessage(null);

    const parsedSymbols = symbolsInput
      .split(',')
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s.length > 0);

    const payload: BacktestRunRequest = {
      strategy_type: strategyType,
      symbols: parsedSymbols.length > 0 ? parsedSymbols : ['SPY', 'QQQ'],
      initial_cash: initialCash,
      benchmark_symbol: benchmarkSymbol.trim().toUpperCase() || 'SPY',
      cost_bps: costBps,
      parameters: { min_warmup_bars: 45 },
    };

    try {
      const runRes = await quantApi.runBacktest<BacktestRunResponse>(payload);
      if (runRes && runRes.backtest_id) {
        // Fetch detailed results
        const statusRes = await quantApi.getBacktestStatus<BacktestStatusResponse>(runRes.backtest_id);
        setActiveBacktest(statusRes);
        fetchHistory();
      }
    } catch (err: any) {
      console.error('Backtest run error:', err);
      setErrorMessage(err?.message || 'Backtest execution failed. Inspect logs.');
    } finally {
      setIsRunning(false);
    }
  };

  // Transform equity curves for Recharts
  const equityChartData = React.useMemo(() => {
    if (!activeBacktest || !activeBacktest.equity_curve) return [];
    const eq = activeBacktest.equity_curve;
    const bench = activeBacktest.benchmark_equity_curve || [];

    return eq.map((val, idx) => ({
      bar: idx,
      strategy: Math.round(val * 100) / 100,
      benchmark: bench[idx] ? Math.round(bench[idx] * 100) / 100 : null,
    }));
  }, [activeBacktest]);

  // Transform drawdowns for Recharts
  const drawdownChartData = React.useMemo(() => {
    if (!activeBacktest || !activeBacktest.drawdown_series) return [];
    return activeBacktest.drawdown_series.map((val, idx) => ({
      bar: idx,
      drawdown: Math.round(val * 10000) / 100, // In percentage (-5.2%)
    }));
  }, [activeBacktest]);

  const metrics = activeBacktest?.metrics;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-card p-5 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border border-slate-800">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <BarChart3 className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                Backtest & Attribution Studio
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-500/20 text-cyan-300 border border-cyan-500/30">
                  CFA-GRADE
                </span>
              </h1>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Zero-lookahead simulation replay, fractional differentiation alpha signals & standalone HTML tearsheets
              </p>
            </div>
          </div>
        </div>

        {activeBacktest?.backtest_id && (
          <div className="flex items-center gap-3">
            <a
              href={`/api/v1/backtest/${activeBacktest.backtest_id}/tearsheet.html`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 font-mono text-xs font-semibold border border-slate-700 transition shadow-sm"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span>Open Tearsheet HTML</span>
            </a>
            <a
              href={`/api/v1/backtest/${activeBacktest.backtest_id}/tearsheet.html`}
              download={`tearsheet_${activeBacktest.backtest_id}.html`}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-mono text-xs font-semibold border border-cyan-400 transition shadow-md shadow-cyan-600/20"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Report</span>
            </a>
          </div>
        )}
      </div>

      {/* Grid: Controls & Metric Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Column: Configuration Controls */}
        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-4 lg:col-span-1">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 font-mono flex items-center gap-2 border-b border-slate-800 pb-3">
            <Play className="w-4 h-4 text-cyan-400" />
            Strategy Parameters
          </h2>

          <div className="space-y-3 font-mono text-xs">
            <div>
              <label className="block text-slate-400 mb-1 font-semibold">Alpha Strategy</label>
              <select
                value={strategyType}
                onChange={(e) => setStrategyType(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-slate-400 mb-1 font-semibold">Asset Universe (Comma-separated)</label>
              <input
                type="text"
                value={symbolsInput}
                onChange={(e) => setSymbolsInput(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
                placeholder="SPY, QQQ, AAPL, NVDA, MSFT"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-slate-400 mb-1 font-semibold">Initial Cash ($)</label>
                <input
                  type="number"
                  value={initialCash}
                  onChange={(e) => setInitialCash(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1 font-semibold">Benchmark</label>
                <input
                  type="text"
                  value={benchmarkSymbol}
                  onChange={(e) => setBenchmarkSymbol(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
                />
              </div>
            </div>

            <div>
              <label className="block text-slate-400 mb-1 font-semibold">Friction / Cost (BPS)</label>
              <input
                type="number"
                step="0.5"
                value={costBps}
                onChange={(e) => setCostBps(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
              />
            </div>

            <button
              onClick={handleRunBacktest}
              disabled={isRunning}
              className={`w-full py-2.5 rounded-lg font-bold font-mono text-xs flex items-center justify-center gap-2 transition shadow-lg ${
                isRunning
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-gradient-to-r from-blue-600 via-cyan-600 to-teal-500 hover:from-blue-500 hover:to-teal-400 text-white shadow-cyan-500/20 border border-cyan-400/40'
              }`}
            >
              {isRunning ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                  <span>SIMULATING BACKTEST...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>RUN CFA BACKTEST</span>
                </>
              )}
            </button>

            {errorMessage && (
              <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-[11px]">
                {errorMessage}
              </div>
            )}
          </div>

          {/* Past Runs Drawer */}
          {history.length > 0 && (
            <div className="pt-4 border-t border-slate-800">
              <h3 className="text-xs font-bold uppercase text-slate-400 mb-2 flex items-center gap-1.5 font-mono">
                <Clock className="w-3.5 h-3.5 text-slate-500" />
                Recent Simulations
              </h3>
              <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                {history.slice(0, 5).map((item) => (
                  <button
                    key={item.backtest_id}
                    onClick={() => setActiveBacktest(item)}
                    className={`w-full text-left p-2 rounded-lg text-xs font-mono transition flex items-center justify-between ${
                      activeBacktest?.backtest_id === item.backtest_id
                        ? 'bg-cyan-950/40 border border-cyan-500/40 text-cyan-300'
                        : 'bg-slate-950/60 hover:bg-slate-900 border border-slate-800/80 text-slate-400'
                    }`}
                  >
                    <span className="truncate max-w-[120px] font-semibold">{item.strategy_type}</span>
                    <span className="text-[10px] text-slate-500">
                      {item.created_at ? new Date(item.created_at).toLocaleTimeString() : ''}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Columns: CFA Metrics & Performance Curves */}
        <div className="lg:col-span-3 space-y-6">
          {/* CFA Metrics Ribbon */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Sharpe Ratio</span>
              <div className={`text-xl font-bold font-mono mt-1 ${metrics && metrics.sharpe_ratio >= 1.0 ? 'text-emerald-400' : 'text-slate-100'}`}>
                {metrics ? metrics.sharpe_ratio.toFixed(2) : '--'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">Annualized Excess</span>
            </div>

            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Sortino Ratio</span>
              <div className="text-xl font-bold font-mono mt-1 text-cyan-400">
                {metrics ? metrics.sortino_ratio.toFixed(2) : '--'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">Downside Dev Semi-Var</span>
            </div>

            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Max Drawdown</span>
              <div className={`text-xl font-bold font-mono mt-1 ${metrics && metrics.max_drawdown <= -0.15 ? 'text-rose-400' : 'text-slate-100'}`}>
                {metrics ? `${(metrics.max_drawdown * 100).toFixed(2)}%` : '--'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">Peak-to-Trough Loss</span>
            </div>

            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">CAGR Return</span>
              <div className={`text-xl font-bold font-mono mt-1 ${metrics && metrics.annualized_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {metrics ? `${(metrics.annualized_return * 100).toFixed(2)}%` : '--'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">Compound Annual Growth</span>
            </div>

            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Win Rate</span>
              <div className="text-xl font-bold font-mono mt-1 text-slate-100">
                {metrics ? `${(metrics.win_rate * 100).toFixed(1)}%` : '--'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">Profitable Bars Ratio</span>
            </div>

            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Profit Factor</span>
              <div className="text-xl font-bold font-mono mt-1 text-slate-100">
                {metrics ? metrics.profit_factor.toFixed(2) : '--'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">Gross Wins / Gross Losses</span>
            </div>

            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">CAPM Alpha (Ann.)</span>
              <div className="text-xl font-bold font-mono mt-1 text-cyan-400">
                {metrics && metrics.alpha !== null && metrics.alpha !== undefined
                  ? `${(metrics.alpha * 100).toFixed(2)}%`
                  : 'N/A'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">vs. {activeBacktest?.symbols[0] || 'Benchmark'}</span>
            </div>

            <div className="glass-card p-3.5 rounded-xl border border-slate-800">
              <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold">CAPM Beta</span>
              <div className="text-xl font-bold font-mono mt-1 text-slate-100">
                {metrics && metrics.beta !== null && metrics.beta !== undefined
                  ? metrics.beta.toFixed(2)
                  : 'N/A'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">Systematic Exposure</span>
            </div>
          </div>

          {/* Equity Curve Chart vs Benchmark */}
          <div className="glass-card p-5 rounded-2xl border border-slate-800">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 font-mono flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                Cumulative Equity Curve vs. Benchmark
              </h3>
              {metrics && (
                <div className="text-xs font-mono font-semibold text-slate-400">
                  Initial: <span className="text-slate-200">${metrics.initial_capital.toLocaleString()}</span> &bull; Final:{' '}
                  <span className={metrics.net_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                    ${Math.round(metrics.final_equity).toLocaleString()}
                  </span>
                </div>
              )}
            </div>

            <div className="h-72 w-full">
              {equityChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={equityChartData}>
                    <defs>
                      <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#38BDF8" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#38BDF8" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.6} />
                    <XAxis dataKey="bar" stroke="#64748B" tick={{ fontSize: 10, fill: '#64748B' }} />
                    <YAxis
                      stroke="#64748B"
                      tick={{ fontSize: 10, fill: '#64748B' }}
                      domain={['auto', 'auto']}
                      tickFormatter={(v) => `$${Math.round(v / 1000)}k`}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#090D16', borderColor: '#1E293B', borderRadius: '8px' }}
                      formatter={(val: any) => [`$${Number(val).toLocaleString()}`, '']}
                    />
                    <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace' }} />
                    <Area
                      type="monotone"
                      dataKey="strategy"
                      name={`${activeBacktest?.strategy_type || 'Strategy'} NAV`}
                      stroke="#38BDF8"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#eqGrad)"
                    />
                    {equityChartData[0]?.benchmark !== null && (
                      <Line
                        type="monotone"
                        dataKey="benchmark"
                        name="Benchmark (SPY)"
                        stroke="#94A3B8"
                        strokeWidth={1.5}
                        strokeDasharray="4 4"
                        dot={false}
                      />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-xs font-mono text-slate-500">
                  Select parameters and click "RUN CFA BACKTEST" to simulate.
                </div>
              )}
            </div>
          </div>

          {/* Underwater Drawdown Chart */}
          <div className="glass-card p-5 rounded-2xl border border-slate-800">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 font-mono mb-4 flex items-center gap-2">
              <Percent className="w-4 h-4 text-rose-400" />
              Underwater Peak-to-Trough Drawdown Series (%)
            </h3>
            <div className="h-44 w-full">
              {drawdownChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={drawdownChartData}>
                    <defs>
                      <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#EF4444" stopOpacity={0.0} />
                        <stop offset="95%" stopColor="#EF4444" stopOpacity={0.3} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" opacity={0.6} />
                    <XAxis dataKey="bar" stroke="#64748B" tick={{ fontSize: 10, fill: '#64748B' }} />
                    <YAxis
                      stroke="#64748B"
                      tick={{ fontSize: 10, fill: '#64748B' }}
                      domain={['dataMin', 0]}
                      tickFormatter={(v) => (typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(1)}%` : '0.0%')}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#090D16', borderColor: '#1E293B', borderRadius: '8px' }}
                      formatter={(val: any) => [`${Number(val).toFixed(2)}%`, 'Drawdown']}
                    />
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      stroke="#EF4444"
                      strokeWidth={1.5}
                      fillOpacity={1}
                      fill="url(#ddGrad)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-xs font-mono text-slate-500">
                  No drawdown data available.
                </div>
              )}
            </div>
          </div>

          {/* Calendar Monthly Return Heatmap Grid */}
          {activeBacktest?.monthly_matrix && Object.keys(activeBacktest.monthly_matrix).length > 0 && (
            <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200 font-mono flex items-center gap-2">
                <Calendar className="w-4 h-4 text-cyan-400" />
                Calendar Monthly Return Matrix (%)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono text-center border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                      <th className="py-2 px-2 text-left">Year</th>
                      {Array.from({ length: 12 }, (_, i) => (
                        <th key={i} className="py-2 px-2">
                          M{String(i + 1).padStart(2, '0')}
                        </th>
                      ))}
                      <th className="py-2 px-2 font-bold text-slate-200">YTD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(activeBacktest.monthly_matrix)
                      .sort(([y1], [y2]) => y2.localeCompare(y1))
                      .map(([year, months]) => {
                        const ytdVal = months['YTD'] || 0;
                        return (
                          <tr key={year} className="border-b border-slate-900/60 hover:bg-slate-900/40">
                            <td className="py-2 px-2 text-left font-bold text-slate-300">{year}</td>
                            {Array.from({ length: 12 }, (_, i) => {
                              const mKey = `M${String(i + 1).padStart(2, '0')}`;
                              const val = months[mKey];
                              const hasVal = val !== undefined && val !== null;
                              return (
                                <td key={mKey} className="py-2 px-2">
                                  {hasVal ? (
                                    <span
                                      className={`inline-block w-full py-1 px-1 rounded font-semibold text-[11px] ${
                                        val > 0
                                          ? 'bg-emerald-500/15 text-emerald-400'
                                          : val < 0
                                          ? 'bg-rose-500/15 text-rose-400'
                                          : 'text-slate-500'
                                      }`}
                                    >
                                      {val > 0 ? `+${val.toFixed(1)}%` : `${val.toFixed(1)}%`}
                                    </span>
                                  ) : (
                                    <span className="text-slate-700">-</span>
                                  )}
                                </td>
                              );
                            })}
                            <td className="py-2 px-2">
                              <span
                                className={`inline-block w-full py-1 px-1.5 rounded font-bold text-xs ${
                                  ytdVal > 0
                                    ? 'bg-emerald-500/25 text-emerald-300 border border-emerald-500/30'
                                    : ytdVal < 0
                                    ? 'bg-rose-500/25 text-rose-300 border border-rose-500/30'
                                    : 'text-slate-400'
                                }`}
                              >
                                {ytdVal > 0 ? `+${ytdVal.toFixed(1)}%` : `${ytdVal.toFixed(1)}%`}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
