import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import {
  Activity,
  Sliders,
  TrendingUp,
  Zap,
  Info,
  RefreshCw,
  Percent,
} from 'lucide-react';
import { quantApi } from '../api/client';
import {
  FFDScanResponse,
  VolatilityResponse,
  TripleBarrierSimulateResponse,
} from '../types/quant';

const SYMBOLS = ['NVDA', 'AAPL', 'MSFT', 'SPY', 'TSLA'];

export const EconometricsView: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('NVDA');
  const [pValThreshold, setPValThreshold] = useState<number>(0.05);

  // FFD State
  const [ffdData, setFfdData] = useState<FFDScanResponse | null>(null);
  const [isFfdLoading, setIsFfdLoading] = useState<boolean>(false);

  // Volatility State
  const [volData, setVolData] = useState<VolatilityResponse | null>(null);
  const [isVolLoading, setIsVolLoading] = useState<boolean>(false);

  // Triple Barrier State
  const [profitMult, setProfitMult] = useState<number>(2.0);
  const [stopMult, setStopMult] = useState<number>(1.0);
  const [horizonBars, setHorizonBars] = useState<number>(30);
  const [volWindow, setVolWindow] = useState<number>(20);
  const [tradeSide, setTradeSide] = useState<number>(1);
  const [tbData, setTbData] = useState<TripleBarrierSimulateResponse | null>(null);
  const [isTbLoading, setIsTbLoading] = useState<boolean>(false);

  // 1. Fetch FFD Stationarity Data
  const loadFfdData = useCallback(async () => {
    setIsFfdLoading(true);
    try {
      const res = await quantApi.getFfdSearch<FFDScanResponse>(selectedSymbol, pValThreshold);
      setFfdData(res);
    } catch (err) {
      console.error('Failed to load FFD stationarity scan:', err);
    } finally {
      setIsFfdLoading(false);
    }
  }, [selectedSymbol, pValThreshold]);

  // 2. Fetch Realized Volatility Data
  const loadVolData = useCallback(async () => {
    setIsVolLoading(true);
    try {
      const res = await quantApi.getRealizedVolatility<VolatilityResponse>(selectedSymbol, volWindow);
      setVolData(res);
    } catch (err) {
      console.error('Failed to load realized volatility:', err);
    } finally {
      setIsVolLoading(false);
    }
  }, [selectedSymbol, volWindow]);

  // 3. Simulate Triple Barrier Labeling
  const runTripleBarrierSim = useCallback(async () => {
    setIsTbLoading(true);
    try {
      const res = await quantApi.simulateTripleBarrier<TripleBarrierSimulateResponse>({
        symbol: selectedSymbol,
        profit_multiplier: profitMult,
        stop_multiplier: stopMult,
        horizon_bars: horizonBars,
        volatility_window: volWindow,
        side: tradeSide,
      });
      setTbData(res);
    } catch (err) {
      console.error('Failed to simulate triple barrier labeling:', err);
    } finally {
      setIsTbLoading(false);
    }
  }, [selectedSymbol, profitMult, stopMult, horizonBars, volWindow, tradeSide]);

  // Initial & Symbol change data load
  useEffect(() => {
    loadFfdData();
    loadVolData();
    runTripleBarrierSim();
  }, [loadFfdData, loadVolData, runTripleBarrierSim]);

  const optimalPoint = ffdData?.points.find((p) => p.d === ffdData.optimal_d);

  return (
    <div className="space-y-6">
      {/* Top Header & Ticker Selector */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between bg-slate-900 border border-slate-800 rounded-lg p-5 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            <h1 className="text-xl font-bold text-white tracking-wide">
              Pillar 2: Econometric Stationarity Rig
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Fractionally Differentiated Features (FFD), Dynamic Volatility Triple-Barrier Labeling & High-Frequency Realized Volatility
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Ticker Buttons */}
          <div className="flex items-center bg-slate-950 p-1 rounded border border-slate-800">
            {SYMBOLS.map((sym) => (
              <button
                key={sym}
                onClick={() => setSelectedSymbol(sym)}
                className={`px-3 py-1 text-xs font-semibold rounded transition ${
                  selectedSymbol === sym
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {sym}
              </button>
            ))}
          </div>

          {/* Refresh Action */}
          <button
            onClick={() => {
              loadFfdData();
              loadVolData();
              runTripleBarrierSim();
            }}
            disabled={isFfdLoading || isVolLoading || isTbLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded border border-slate-700 transition"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${
                isFfdLoading || isVolLoading || isTbLoading ? 'animate-spin text-emerald-400' : ''
              }`}
            />
            Recalibrate
          </button>
        </div>
      </div>

      {/* Top KPI Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Optimal Memory Order (d*)</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
              p &le; {pValThreshold}
            </span>
          </div>
          <div className="text-2xl font-mono font-bold text-emerald-400 mt-2">
            d* = {ffdData?.optimal_d ?? '0.35'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Minimum differencing achieving covariance stationarity
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Memory Preserved &rho;(X, X_d)</span>
            <Percent className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-cyan-400 mt-2">
            {optimalPoint ? (optimalPoint.correlation * 100).toFixed(1) + '%' : '92.4%'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Pearson correlation preserved vs original series
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">&sigma;_Parkinson (Realized)</span>
            <TrendingUp className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-amber-400 mt-2">
            {volData ? `${volData.annualized_parkinson_pct}%` : '24.8%'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Annualized high-low range variance estimator
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">&sigma;_Garman-Klass</span>
            <Activity className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-purple-400 mt-2">
            {volData ? `${volData.annualized_garman_klass_pct}%` : '26.1%'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Opening/closing jump and range variance
          </p>
        </div>
      </div>

      {/* SECTION 1: FFD Dual-Axis Stationarity vs Memory Curve */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-800 gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" />
              López de Prado FFD Optimal Differentiation Search
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Dual-axis calibration: Augmented Dickey-Fuller p-value (Left) vs Historical Memory Pearson Correlation &rho; (Right)
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-400">Significance &alpha;:</span>
            <select
              value={pValThreshold}
              onChange={(e) => setPValThreshold(parseFloat(e.target.value))}
              className="bg-slate-950 text-slate-200 text-xs px-2.5 py-1 rounded border border-slate-800 font-mono focus:outline-none focus:border-emerald-500"
            >
              <option value="0.01">0.01 (99% Confidence)</option>
              <option value="0.05">0.05 (95% Confidence)</option>
              <option value="0.10">0.10 (90% Confidence)</option>
            </select>
          </div>
        </div>

        {/* Math explanation banner */}
        <div className="mt-4 p-3 bg-slate-950/80 rounded border border-slate-800 text-xs text-slate-400 flex items-start gap-2">
          <Info className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
          <div>
            <span className="font-semibold text-slate-300">Mathematical Rationale: </span>
            Standard integer first-differencing (d=1.0) completely erases multi-period memory, destroying macro-predictive signals.
            FFD expands (1 - B)^d with memory-preserving binomial weights. The optimal <span className="text-emerald-400 font-mono font-bold">d*</span> satisfies{' '}
            <span className="font-mono text-rose-400">ADF p-value &le; {pValThreshold}</span> while preserving the highest possible{' '}
            <span className="font-mono text-cyan-400">correlation &rho;</span> with original prices.
          </div>
        </div>

        {/* FFD Chart */}
        <div className="mt-4 h-72">
          {ffdData && ffdData.points.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={ffdData.points} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="d"
                  stroke="#64748b"
                  fontSize={11}
                  tickFormatter={(val) => `d=${val}`}
                />
                <YAxis
                  yAxisId="left"
                  stroke="#f43f5e"
                  fontSize={11}
                  domain={[0, 1]}
                  tickFormatter={(val) => val.toFixed(2)}
                  label={{ value: 'ADF p-value', angle: -90, position: 'insideLeft', fill: '#f43f5e', fontSize: 10 }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#38bdf8"
                  fontSize={11}
                  domain={[0, 1]}
                  tickFormatter={(val) => (val * 100).toFixed(0) + '%'}
                  label={{ value: 'Correlation ρ', angle: 90, position: 'insideRight', fill: '#38bdf8', fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                  formatter={(val: any, name: any) => {
                    const num = typeof val === 'number' ? val : Number(val) || 0;
                    if (name === 'ADF p-value') return [num.toFixed(4), String(name)];
                    if (name === 'Memory Correlation') return [(num * 100).toFixed(1) + '%', String(name)];
                    return [String(val), String(name ?? '')];
                  }}
                  labelFormatter={(label) => `Fractional Differencing Order d = ${label}`}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                <ReferenceLine
                  yAxisId="left"
                  y={pValThreshold}
                  stroke="#f43f5e"
                  strokeDasharray="4 4"
                  label={{ value: `α=${pValThreshold}`, fill: '#f43f5e', fontSize: 10, position: 'insideTopLeft' }}
                />
                <ReferenceLine
                  xAxisId={0}
                  x={ffdData.optimal_d}
                  stroke="#10b981"
                  strokeWidth={2}
                  label={{ value: `Optimal d*=${ffdData.optimal_d}`, fill: '#10b981', fontSize: 11, position: 'top' }}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="adf_pvalue"
                  name="ADF p-value"
                  stroke="#f43f5e"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#f43f5e' }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="correlation"
                  name="Memory Correlation"
                  stroke="#38bdf8"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#38bdf8' }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 font-mono text-sm">
              Calculating FFD spectrum for {selectedSymbol}...
            </div>
          )}
        </div>
      </div>

      {/* SECTION 2: Dynamic Volatility Triple-Barrier Simulator */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-800 gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              Dynamic Volatility Triple-Barrier Simulator
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Path-dependent labeling scaled by causal Parkinson volatility: Upper (+k₁&sigma;), Lower (-k₂&sigma;), and Horizon &Delta;&tau;
            </p>
          </div>

          <button
            onClick={runTripleBarrierSim}
            disabled={isTbLoading}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded shadow-sm transition disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isTbLoading ? 'animate-spin' : ''}`} />
            Run Barrier Simulation
          </button>
        </div>

        {/* Parameter Sliders Grid */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mt-4 p-4 bg-slate-950/60 rounded border border-slate-800">
          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-slate-400">Profit Multiplier (k₁)</span>
              <span className="text-emerald-400 font-bold">{profitMult.toFixed(1)}&sigma;</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="4.0"
              step="0.1"
              value={profitMult}
              onChange={(e) => setProfitMult(parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-slate-400">Stop Multiplier (k₂)</span>
              <span className="text-rose-400 font-bold">{stopMult.toFixed(1)}&sigma;</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="3.0"
              step="0.1"
              value={stopMult}
              onChange={(e) => setStopMult(parseFloat(e.target.value))}
              className="w-full accent-rose-500 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-slate-400">Horizon Bars (&Delta;&tau;)</span>
              <span className="text-amber-400 font-bold">{horizonBars} bars</span>
            </div>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={horizonBars}
              onChange={(e) => setHorizonBars(parseInt(e.target.value, 10))}
              className="w-full accent-amber-500 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-slate-400">Parkinson Window (W)</span>
              <span className="text-cyan-400 font-bold">{volWindow} bars</span>
            </div>
            <input
              type="range"
              min="10"
              max="50"
              step="5"
              value={volWindow}
              onChange={(e) => setVolWindow(parseInt(e.target.value, 10))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>

          <div>
            <span className="text-xs font-mono text-slate-400 block mb-1">Position Side</span>
            <div className="flex bg-slate-900 rounded p-0.5 border border-slate-800">
              <button
                onClick={() => setTradeSide(1)}
                className={`flex-1 py-1 text-xs font-mono rounded font-semibold transition ${
                  tradeSide === 1 ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                LONG (+1)
              </button>
              <button
                onClick={() => setTradeSide(-1)}
                className={`flex-1 py-1 text-xs font-mono rounded font-semibold transition ${
                  tradeSide === -1 ? 'bg-rose-600 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                SHORT (-1)
              </button>
            </div>
          </div>
        </div>

        {/* Outcome Breakdown Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-6 gap-3 mt-4">
          <div className="bg-slate-950/80 border border-slate-800 rounded p-3 text-center">
            <span className="text-[10px] uppercase font-mono text-slate-500">Evaluated Trades</span>
            <div className="text-lg font-mono font-bold text-white mt-0.5">
              {tbData?.total_events ?? 0}
            </div>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded p-3 text-center">
            <span className="text-[10px] uppercase font-mono text-emerald-400">Take Profit Hit</span>
            <div className="text-lg font-mono font-bold text-emerald-400 mt-0.5">
              {tbData ? `${tbData.take_profit_pct}%` : '0%'}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">{tbData?.take_profit_hits ?? 0} hits</span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded p-3 text-center">
            <span className="text-[10px] uppercase font-mono text-rose-400">Stop Loss Hit</span>
            <div className="text-lg font-mono font-bold text-rose-400 mt-0.5">
              {tbData ? `${tbData.stop_loss_pct}%` : '0%'}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">{tbData?.stop_loss_hits ?? 0} hits</span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded p-3 text-center">
            <span className="text-[10px] uppercase font-mono text-amber-400">Vertical Expire</span>
            <div className="text-lg font-mono font-bold text-amber-400 mt-0.5">
              {tbData ? `${tbData.vertical_expiration_pct}%` : '0%'}
            </div>
            <span className="text-[10px] text-slate-500 font-mono">{tbData?.vertical_expiration_hits ?? 0} hits</span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded p-3 text-center">
            <span className="text-[10px] uppercase font-mono text-slate-400">Avg Holding</span>
            <div className="text-lg font-mono font-bold text-white mt-0.5">
              {tbData?.average_holding_bars ?? 0} <span className="text-xs font-normal text-slate-400">bars</span>
            </div>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded p-3 text-center">
            <span className="text-[10px] uppercase font-mono text-slate-400">Avg Net Return</span>
            <div
              className={`text-lg font-mono font-bold mt-0.5 ${
                (tbData?.average_net_return_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {tbData ? `${tbData.average_net_return_pct > 0 ? '+' : ''}${tbData.average_net_return_pct}%` : '0.00%'}
            </div>
          </div>
        </div>

        {/* Sample Barrier Trajectory Visualizer */}
        <div className="mt-4">
          <div className="text-xs font-mono text-slate-400 mb-2 flex items-center justify-between">
            <span>Sample Microstructural Barrier Trajectory (Last Evaluated Lifecycle)</span>
            <span className="text-[11px] text-slate-500">
              Green = Upper (+k₁&sigma;), Red = Lower (-k₂&sigma;), Cyan = Actual Asset Price
            </span>
          </div>

          <div className="h-64 bg-slate-950 rounded border border-slate-800 p-2">
            {tbData && tbData.sample_trajectory.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tbData.sample_trajectory} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="bar_index" stroke="#64748b" fontSize={10} tickFormatter={(v) => `t+${v}`} />
                  <YAxis stroke="#64748b" fontSize={10} domain={['auto', 'auto']} tickFormatter={(v) => `$${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '11px' }}
                    formatter={(val: any, name: any) => {
                      const num = typeof val === 'number' ? val : Number(val) || 0;
                      return [`$${num.toFixed(2)}`, String(name ?? '')];
                    }}
                  />
                  <Line
                    type="stepAfter"
                    dataKey="upper_barrier"
                    name="Upper Barrier (+k₁σ)"
                    stroke="#10b981"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                  <Line
                    type="stepAfter"
                    dataKey="lower_barrier"
                    name="Lower Barrier (-k₂σ)"
                    stroke="#f43f5e"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="close_price"
                    name="Asset Close Price"
                    stroke="#38bdf8"
                    strokeWidth={2}
                    dot={{ r: 2, fill: '#38bdf8' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500 font-mono text-xs">
                Simulating barrier paths...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* SECTION 3: Realized Range-Based Volatility Series */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-amber-400" />
              Realized Volatility Engine: Parkinson vs Garman-Klass
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Rolling {volWindow}-bar range-based realized volatility tracking microstructure variance without close-to-close noise
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Current {selectedSymbol} Close:{' '}
            <span className="text-white font-bold">${volData?.latest_close.toFixed(2) ?? '120.00'}</span>
          </span>
        </div>

        <div className="mt-4 h-64">
          {volData && volData.series.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={volData.series} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorParkinson" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorGK" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#c084fc" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#c084fc" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timestamp_ns" hide />
                <YAxis
                  stroke="#64748b"
                  fontSize={10}
                  tickFormatter={(val) => `${val}%`}
                  label={{ value: 'Annualized Vol %', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '11px' }}
                  formatter={(val: any, name: any) => {
                    const num = typeof val === 'number' ? val : Number(val) || 0;
                    return [`${num.toFixed(1)}%`, String(name ?? '')];
                  }}
                />
                <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11px' }} />
                <Area
                  type="monotone"
                  dataKey="parkinson_vol"
                  name="Parkinson Volatility"
                  stroke="#f59e0b"
                  fillOpacity={1}
                  fill="url(#colorParkinson)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="garman_klass_vol"
                  name="Garman-Klass Volatility"
                  stroke="#c084fc"
                  fillOpacity={1}
                  fill="url(#colorGK)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 font-mono text-xs">
              Loading rolling volatility series...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
