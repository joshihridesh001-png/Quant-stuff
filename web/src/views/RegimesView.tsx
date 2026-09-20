import React, { useState, useEffect, useCallback } from 'react';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import {
  Brain,
  Zap,
  AlertTriangle,
  RefreshCw,
  Layers,
  Thermometer,
  Crosshair,
} from 'lucide-react';
import { quantApi } from '../api/client';
import {
  RegimeStatusResponse,
  PayoffMatrixResponse,
} from '../types/quant';

const SYMBOLS = ['NVDA', 'AAPL', 'MSFT', 'SPY', 'TSLA'];

export const RegimesView: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('NVDA');
  const [cusumThreshold, setCusumThreshold] = useState<number>(3.0);
  const [cusumDrift, setCusumDrift] = useState<number>(0.5);

  // Payoff Matrix Controls
  const [ambiguityBeta, setAmbiguityBeta] = useState<number>(1.5);
  const [riskAversion, setRiskAversion] = useState<number>(2.0);

  // Data states
  const [regimeData, setRegimeData] = useState<RegimeStatusResponse | null>(null);
  const [payoffData, setPayoffData] = useState<PayoffMatrixResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Fetch Regime Status
  const loadRegimeStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await quantApi.getRegimeStatus<RegimeStatusResponse>(
        selectedSymbol,
        cusumThreshold,
        cusumDrift
      );
      setRegimeData(res);
    } catch (err) {
      console.error('Failed to load regime status:', err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedSymbol, cusumThreshold, cusumDrift]);

  // Fetch Payoff Matrix
  const loadPayoffMatrix = useCallback(async () => {
    try {
      const res = await quantApi.computePayoffMatrix<PayoffMatrixResponse>(
        ambiguityBeta,
        riskAversion
      );
      setPayoffData(res);
    } catch (err) {
      console.error('Failed to load payoff matrix:', err);
    }
  }, [ambiguityBeta, riskAversion]);

  useEffect(() => {
    loadRegimeStatus();
  }, [loadRegimeStatus]);

  useEffect(() => {
    loadPayoffMatrix();
  }, [loadPayoffMatrix]);

  const getRegimeBadge = (regime: string) => {
    switch (regime) {
      case 'LOW_VOL_ABSORPTION':
        return {
          label: 'Low-Vol Absorption',
          color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
          desc: 'Market efficiently absorbs liquidity without price impact',
        };
      case 'MOMENTUM_CASCADE':
        return {
          label: 'Momentum Cascade',
          color: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
          desc: 'Directional momentum with persistent order book imbalance',
        };
      case 'PANIC_LIQUIDITY_TRAP':
        return {
          label: 'Panic Liquidity Trap',
          color: 'bg-rose-500/10 text-rose-400 border-rose-500/30 animate-pulse',
          desc: 'Severe liquidity withdrawal and violent microstructural spread blowout',
        };
      default:
        return {
          label: regime,
          color: 'bg-slate-800 text-slate-300 border-slate-700',
          desc: 'Standard regime',
        };
    }
  };

  const badge = getRegimeBadge(regimeData?.current_regime ?? 'LOW_VOL_ABSORPTION');

  // Dirichlet Bar Data
  const dirichletData = regimeData
    ? [
        { name: 'Absorption (α₁)', alpha: regimeData.dirichlet_alphas[0], prob: regimeData.p_absorption, fill: '#10b981' },
        { name: 'Momentum (α₂)', alpha: regimeData.dirichlet_alphas[1], prob: regimeData.p_momentum, fill: '#38bdf8' },
        { name: 'Panic (α₃)', alpha: regimeData.dirichlet_alphas[2], prob: regimeData.p_panic, fill: '#f43f5e' },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between bg-slate-900 border border-slate-800 rounded-lg p-5 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            <h1 className="text-xl font-bold text-white tracking-wide">
              Pillar 3: Bayesian Game Theory & Jump Regimes
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time 3-simplex posterior probabilities, Dirichlet belief updates, two-sided CUSUM shock detector, and Stackelberg Minimax Regret
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Ticker Selector */}
          <div className="flex items-center bg-slate-950 p-1 rounded border border-slate-800">
            {SYMBOLS.map((sym) => (
              <button
                key={sym}
                onClick={() => setSelectedSymbol(sym)}
                className={`px-3 py-1 text-xs font-semibold rounded transition ${
                  selectedSymbol === sym
                    ? 'bg-purple-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {sym}
              </button>
            ))}
          </div>

          <button
            onClick={() => {
              loadRegimeStatus();
              loadPayoffMatrix();
            }}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded border border-slate-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-purple-400' : ''}`} />
            Recalculate
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Dominant Market State</span>
            <Layers className="w-4 h-4 text-purple-400" />
          </div>
          <div className="mt-2">
            <span
              className={`inline-block text-xs px-2.5 py-1 rounded font-mono font-bold border ${badge.color}`}
            >
              {badge.label}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 mt-2">{badge.desc}</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Posterior Confidence</span>
            <span className="text-xs font-mono text-purple-400 font-bold">
              {typeof regimeData?.confidence_pct === 'number' ? regimeData.confidence_pct.toFixed(1) : '75.0'}%
            </span>
          </div>
          <div className="text-2xl font-mono font-bold text-white mt-2">
            {regimeData
              ? `${(Math.max(regimeData.p_absorption ?? 0, regimeData.p_momentum ?? 0, regimeData.p_panic ?? 0) * 100).toFixed(1)}%`
              : '70.0%'}
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="bg-purple-500 h-full rounded-full transition-all"
              style={{ width: `${regimeData?.confidence_pct ?? 70}%` }}
            />
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Two-Sided CUSUM Shock</span>
            <AlertTriangle
              className={`w-4 h-4 ${regimeData?.cusum_alarm ? 'text-rose-400 animate-bounce' : 'text-slate-600'}`}
            />
          </div>
          <div className="mt-2">
            {regimeData?.cusum_alarm ? (
              <span className="inline-block text-xs font-mono font-bold px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse">
                ALARM ACTIVE (S &ge; {cusumThreshold})
              </span>
            ) : (
              <span className="inline-block text-xs font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                NORMAL (Hysteresis Safe)
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 mt-2 font-mono">
            S+ = {typeof regimeData?.cusum_s_pos === 'number' ? regimeData.cusum_s_pos.toFixed(2) : '0.00'} | S- ={' '}
            {typeof regimeData?.cusum_s_neg === 'number' ? regimeData.cusum_s_neg.toFixed(2) : '0.00'} (h={cusumThreshold})
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Thermodynamic Ambiguity (&beta;)</span>
            <Thermometer className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-amber-400 mt-2">
            &beta; = {typeof regimeData?.thermodynamic_beta === 'number' ? regimeData.thermodynamic_beta.toFixed(2) : '1.50'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Fournier-Guillin concentration radius temperature
          </p>
        </div>
      </div>

      {/* SECTION 1: 3-Simplex Probability & Dirichlet Belief Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Simplex Probability History */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-lg p-5">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-400" />
                3-Simplex Posterior Regime Dynamics
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Causal Bayesian probability evolution satisfying conservation: P_Abs + P_Mom + P_Pan &equiv; 1.0
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="text-emerald-400">● P_Abs</span>
              <span className="text-cyan-400">● P_Mom</span>
              <span className="text-rose-400">● P_Pan</span>
            </div>
          </div>

          <div className="mt-4 h-64">
            {regimeData && regimeData.regime_history.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={regimeData.regime_history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="bar_index" stroke="#64748b" fontSize={10} tickFormatter={(v) => `t-${regimeData.regime_history.length - v}`} />
                  <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '11px' }}
                    formatter={(val: any, name: any) => [`${(Number(val) * 100).toFixed(1)}%`, String(name ?? '')]}
                  />
                  <Area
                    type="monotone"
                    dataKey="p_absorption"
                    stackId="1"
                    name="Low-Vol Absorption"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.6}
                  />
                  <Area
                    type="monotone"
                    dataKey="p_momentum"
                    stackId="1"
                    name="Momentum Cascade"
                    stroke="#38bdf8"
                    fill="#38bdf8"
                    fillOpacity={0.6}
                  />
                  <Area
                    type="monotone"
                    dataKey="p_panic"
                    stackId="1"
                    name="Panic Liquidity Trap"
                    stroke="#f43f5e"
                    fill="#f43f5e"
                    fillOpacity={0.6}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500 font-mono text-xs">
                Computing Bayesian regime history...
              </div>
            )}
          </div>
        </div>

        {/* Dirichlet Belief Hyperparameters */}
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2 pb-3 border-b border-slate-800">
              <Brain className="w-4 h-4 text-cyan-400" />
              Dirichlet Belief Vector &alpha;
            </h2>
            <p className="text-xs text-slate-400 mt-2">
              Concentration hyperparameters scaling posterior weight across latent states
            </p>

            <div className="mt-4 h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dirichletData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={9} />
                  <YAxis stroke="#64748b" fontSize={10} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '11px' }}
                    formatter={(val: any) => [`α = ${Number(val).toFixed(2)}`, 'Dirichlet Prior']}
                  />
                  <Bar dataKey="alpha" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-3 bg-slate-950 rounded border border-slate-800 mt-3 font-mono text-xs text-slate-400 space-y-1">
            <div className="flex justify-between">
              <span>P(Absorption):</span>
              <span className="text-emerald-400 font-bold">{((regimeData?.p_absorption ?? 0.7) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span>P(Momentum):</span>
              <span className="text-cyan-400 font-bold">{((regimeData?.p_momentum ?? 0.2) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span>P(Panic Trap):</span>
              <span className="text-rose-400 font-bold">{((regimeData?.p_panic ?? 0.1) * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 2: Two-Sided CUSUM Structural Break & Jump Detector */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-800 gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" />
              Two-Sided CUSUM Structural Break & Jump Detector
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Accumulates standardized log-return deviations S_t^+ and S_t^- with allowance drift k and threshold h
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-400">Threshold h:</span>
              <input
                type="number"
                min="1.5"
                max="6.0"
                step="0.5"
                value={cusumThreshold}
                onChange={(e) => setCusumThreshold(parseFloat(e.target.value))}
                className="w-16 bg-slate-950 text-slate-200 text-xs px-2 py-1 rounded border border-slate-800 font-mono text-center"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-400">Drift k:</span>
              <input
                type="number"
                min="0.1"
                max="1.5"
                step="0.1"
                value={cusumDrift}
                onChange={(e) => setCusumDrift(parseFloat(e.target.value))}
                className="w-16 bg-slate-950 text-slate-200 text-xs px-2 py-1 rounded border border-slate-800 font-mono text-center"
              />
            </div>
          </div>
        </div>

        <div className="mt-4 h-64">
          {regimeData && regimeData.cusum_series.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={regimeData.cusum_series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="bar_index" stroke="#64748b" fontSize={10} tickFormatter={(v) => `t-${regimeData.cusum_series.length - v}`} />
                <YAxis stroke="#64748b" fontSize={10} domain={[0, 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '11px' }}
                  formatter={(val: any, name: any) => [Number(val).toFixed(2), String(name ?? '')]}
                />
                <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine
                  y={cusumThreshold}
                  stroke="#f43f5e"
                  strokeDasharray="4 4"
                  label={{ value: `Shock Limit h=${cusumThreshold}`, fill: '#f43f5e', fontSize: 10, position: 'insideTopLeft' }}
                />
                <Line
                  type="monotone"
                  dataKey="s_pos"
                  name="S+ Positive Drift"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="s_neg"
                  name="S- Negative Drift (Crash)"
                  stroke="#f43f5e"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 font-mono text-xs">
              Streaming CUSUM series...
            </div>
          )}
        </div>
      </div>

      {/* SECTION 3: Stackelberg Minimax Regret Payoff Matrix */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-800 gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Crosshair className="w-4 h-4 text-emerald-400" />
              Stackelberg Minimax Regret Payoff Matrix against Adversarial Counterparties
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Distributionally robust execution game minimizing maximum regret: R(a, &theta;) = max_a' U(a', &theta;) - U(a, &theta;)
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div>
              <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
                <span>Ambiguity &beta;: {ambiguityBeta.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="5.0"
                step="0.1"
                value={ambiguityBeta}
                onChange={(e) => setAmbiguityBeta(parseFloat(e.target.value))}
                className="w-28 accent-amber-500 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
                <span>Risk Aversion &lambda;: {riskAversion.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="5.0"
                step="0.5"
                value={riskAversion}
                onChange={(e) => setRiskAversion(parseFloat(e.target.value))}
                className="w-28 accent-cyan-500 cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Payoff & Regret Interactive Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
          {/* Table 1: Raw Utility Payoffs U(a, theta) */}
          <div className="bg-slate-950 rounded border border-slate-800 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-white">
                Institutional Payoff Matrix U(a, &theta;) [bps]
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Higher is Better</span>
            </div>

            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-left">
                  <th className="pb-2">Strategic Action</th>
                  {payoffData?.counterparties.map((cp) => (
                    <th key={cp} className="pb-2 text-right">
                      {cp}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {payoffData?.actions.map((act, rIdx) => (
                  <tr
                    key={act}
                    className={
                      act === payoffData.optimal_action
                        ? 'bg-emerald-500/10 text-emerald-300 font-semibold'
                        : 'text-slate-300'
                    }
                  >
                    <td className="py-2.5 flex items-center gap-1.5">
                      {act === payoffData.optimal_action && (
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      )}
                      {act}
                    </td>
                    {payoffData.payoff_matrix[rIdx].map((val, cIdx) => (
                      <td
                        key={cIdx}
                        className={`py-2.5 text-right font-bold ${
                          (val ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {(val ?? 0) >= 0 ? `+${(val ?? 0).toFixed(1)}` : (val ?? 0).toFixed(1)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Table 2: Regret Matrix R(a, theta) & Worst-Case Regret */}
          <div className="bg-slate-950 rounded border border-slate-800 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-white">
                Regret Matrix R(a, &theta;) & Max Regret
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Lower is Better</span>
            </div>

            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-left">
                  <th className="pb-2">Action</th>
                  {payoffData?.counterparties.map((cp) => (
                    <th key={cp} className="pb-2 text-right">
                      {cp}
                    </th>
                  ))}
                  <th className="pb-2 text-right text-amber-400 font-bold">Max Regret</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {payoffData?.actions.map((act, rIdx) => {
                  const isOptimal = act === payoffData.optimal_action;
                  const maxRegret = payoffData.worst_case_regrets?.[rIdx] ?? 0;
                  return (
                    <tr
                      key={act}
                      className={
                        isOptimal
                          ? 'bg-emerald-500/10 text-emerald-300 font-semibold'
                          : 'text-slate-300'
                      }
                    >
                      <td className="py-2.5 flex items-center gap-1.5">
                        {isOptimal && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
                        {act}
                      </td>
                      {payoffData.regret_matrix[rIdx].map((val, cIdx) => (
                        <td key={cIdx} className="py-2.5 text-right text-slate-400">
                          {(val ?? 0).toFixed(1)}
                        </td>
                      ))}
                      <td
                        className={`py-2.5 text-right font-bold ${
                          isOptimal ? 'text-emerald-400 text-sm' : 'text-amber-400'
                        }`}
                      >
                        {(maxRegret ?? 0).toFixed(1)} bps
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Robust Decision Certification Banner */}
        <div className="mt-4 p-4 bg-emerald-950/40 rounded border border-emerald-500/30 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">
                Certified Minimax Regret Solution
              </span>
              <span className="text-xs text-slate-400">
                Stackelberg equilibrium guaranteed against worst-case counterparty mix
              </span>
            </div>
            <div className="text-base font-bold text-white mt-1">
              Optimal Robust Strategy:{' '}
              <span className="text-emerald-400 font-mono font-extrabold">
                {payoffData?.optimal_action ?? 'Stealth Adaptive VWAP'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-slate-400 border-t sm:border-t-0 sm:border-l border-slate-800 pt-2 sm:pt-0 sm:pl-4">
            <div>
              <span className="block text-[10px] text-slate-500">Worst-Case Counterparty Probs:</span>
              <div className="flex gap-2 mt-0.5">
                <span className="text-slate-300">
                  Noise: {((payoffData?.worst_case_counterparty_probs[0] ?? 0.33) * 100).toFixed(0)}%
                </span>
                <span className="text-rose-300">
                  Predatory: {((payoffData?.worst_case_counterparty_probs[1] ?? 0.33) * 100).toFixed(0)}%
                </span>
                <span className="text-amber-300">
                  Latency: {((payoffData?.worst_case_counterparty_probs[2] ?? 0.33) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
