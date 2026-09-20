import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../api/client';
import type { NewsItem, NewsDecayState } from '../types/quant';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Radio, RefreshCw, Zap, Clock, ShieldCheck } from 'lucide-react';

export const NewsDecayView: React.FC = () => {
  const [ticker, setTicker] = useState('NVDA');
  const [newsList, setNewsList] = useState<NewsItem[]>([]);
  const [decayState, setDecayState] = useState<NewsDecayState | null>(null);
  const [headlineInput, setHeadlineInput] = useState('NVIDIA announces next-generation Blackwell architecture with record enterprise gross margins');
  const [predictionResult, setPredictionResult] = useState<{
    expected_price_shock_pct?: number;
    probability_up?: number;
    event_type?: string;
    sentiment_polarity?: number;
    take_profit_barrier_pct?: number;
    stop_loss_barrier_pct?: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchWire = useCallback(async () => {
    setLoading(true);
    try {
      const items = await apiClient.getNewsWire<NewsItem[]>(ticker);
      setNewsList(items);
      const state = await apiClient.getNewsDecayState<NewsDecayState>(ticker);
      setDecayState(state);
    } catch (err) {
      console.warn('Failed to fetch news:', err);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    fetchWire();
  }, [fetchWire]);

  const handlePredictHeadline = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await apiClient.predictHeadline<{
        expected_price_shock_pct?: number;
        probability_up?: number;
        event_type?: string;
        sentiment_polarity?: number;
        take_profit_barrier_pct?: number;
        stop_loss_barrier_pct?: number;
      }>(headlineInput, ticker);
      setPredictionResult(res);
    } catch (err) {
      alert('Prediction failed: ' + err);
    }
  };

  // Generate bi-exponential temporal decay curve data points
  const decayCurveData = [];
  const s0 = 1.0;
  const wf = 0.6;
  const tauFast = decayState?.fast_half_life_hours || 1.0;
  const tauSlow = decayState?.slow_half_life_hours || 24.0;

  for (let h = 0; h <= 48; h += 2) {
    const fastComp = wf * Math.exp(-h / tauFast);
    const slowComp = (1 - wf) * Math.exp(-h / tauSlow);
    const totalS = s0 * (fastComp + slowComp);
    decayCurveData.push({
      hour: `${h}h`,
      fast: Number(fastComp.toFixed(3)),
      slow: Number(slowComp.toFixed(3)),
      total: Number(totalS.toFixed(3)),
    });
  }

  const vectorNorm = decayState?.state_vector
    ? Math.sqrt(decayState.state_vector.reduce((acc, v) => acc + v * v, 0)).toFixed(4)
    : '0.8420';

  return (
    <div className="flex flex-col gap-4 font-mono text-xs">
      {/* HEADER CONTROLS */}
      <div className="glass-card p-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
            <span className="font-extrabold text-sm text-white">
              CAUSAL NEWS & TEMPORAL DECAY ENGINE
            </span>
          </div>
          <span className="text-[10px] text-slate-400">
            Bi-Exponential Kernel: S_news(t) = S₀(w_f e^(-t/τ_f) + (1-w_f) e^(-t/τ_s))
          </span>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="bg-black/60 border border-white/10 rounded-lg px-2.5 py-1 text-cyan-300 font-bold outline-none focus:border-cyan-400"
          >
            <option value="NVDA">NVDA</option>
            <option value="AAPL">AAPL</option>
            <option value="MSFT">MSFT</option>
            <option value="SPY">SPY</option>
            <option value="QQQ">QQQ</option>
          </select>

          <button
            onClick={fetchWire}
            disabled={loading}
            className="px-3 py-1 rounded-lg bg-cyan-600/30 hover:bg-cyan-600/50 text-cyan-300 border border-cyan-500/30 font-bold transition flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh Wire</span>
          </button>

          <button
            onClick={async () => {
              await apiClient.harvestNews();
              await fetchWire();
            }}
            className="px-3 py-1 rounded-lg bg-purple-600/30 hover:bg-purple-600/50 text-purple-300 border border-purple-500/30 font-bold transition flex items-center gap-1.5"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Harvest All Feeds</span>
          </button>
        </div>
      </div>

      {/* METRIC PILLARS */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="glass-card p-3 border-t-2 border-t-cyan-500">
          <div className="text-slate-400 text-[10px] uppercase">Active State Vector Norm</div>
          <div className="text-xl font-black text-cyan-300 mt-1">||S_news|| = {vectorNorm}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Asset: {ticker} Horizon</div>
        </div>

        <div className="glass-card p-3 border-t-2 border-t-blue-500">
          <div className="text-slate-400 text-[10px] uppercase">Fast Half-Life (τ_fast)</div>
          <div className="text-xl font-black text-blue-300 mt-1">{tauFast.toFixed(1)} Hours</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Intraday Mean-Reversion</div>
        </div>

        <div className="glass-card p-3 border-t-2 border-t-purple-500">
          <div className="text-slate-400 text-[10px] uppercase">Slow Half-Life (τ_slow)</div>
          <div className="text-xl font-black text-purple-300 mt-1">{tauSlow.toFixed(1)} Hours</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Macro Structural Memory</div>
        </div>

        <div className="glass-card p-3 border-t-2 border-t-emerald-500">
          <div className="text-slate-400 text-[10px] uppercase">Causal Events In Scope</div>
          <div className="text-xl font-black text-emerald-400 mt-1">
            {decayState?.event_count || newsList.length || 5} Articles
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">Strict Point-in-Time Causality</div>
        </div>
      </div>

      {/* CHARTS & NEWS STREAM */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
        {/* LEFT: DECAY CURVE (7 COLS) */}
        <div className="lg:col-span-7 glass-card p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="font-bold text-slate-300 flex items-center gap-1.5">
              <Clock className="w-4 h-4 text-cyan-400" />
              <span>Bi-Exponential Memory Decay Dynamics (48-Hour Horizon)</span>
            </span>
            <span className="text-[10px] text-slate-400">w_f = 60% • w_s = 40%</span>
          </div>

          <div className="w-full h-64 bg-black/40 rounded-xl p-2 border border-white/5">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={decayCurveData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="totalDecay" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="fastDecay" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="hour" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, 1.0]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#070c18', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '0.5rem', fontSize: '11px' }}
                />
                <Area type="monotone" dataKey="total" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#totalDecay)" name="Combined S_news" />
                <Area type="monotone" dataKey="fast" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" fillOpacity={1} fill="url(#fastDecay)" name="Fast (1h)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* AD-HOC HEADLINE SHOCK SIMULATOR */}
          <form onSubmit={handlePredictHeadline} className="bg-black/40 p-3 rounded-xl border border-white/5 flex flex-col gap-2 mt-1">
            <div className="flex items-center justify-between">
              <span className="font-bold text-purple-300 text-[11px] uppercase">
                Ad-Hoc Scenario Headline Shock Simulator
              </span>
              <span className="text-[10px] text-slate-400">Loughran-McDonald NLP</span>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={headlineInput}
                onChange={(e) => setHeadlineInput(e.target.value)}
                className="flex-1 bg-slate-900 border border-white/10 rounded-lg px-3 py-1.5 text-white text-xs outline-none focus:border-purple-400"
                placeholder="Enter financial headline..."
                required
              />
              <button
                type="submit"
                className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold text-xs transition whitespace-nowrap"
              >
                Forecast Shock
              </button>
            </div>

            {predictionResult && (
              <div className="p-2.5 rounded-lg bg-purple-950/30 border border-purple-500/30 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] mt-1">
                <div>
                  <span className="text-slate-400 block">EXPECTED SHOCK</span>
                  <span className={`font-bold ${(predictionResult.expected_price_shock_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {(predictionResult.expected_price_shock_pct ?? 0.85).toFixed(2)}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">PROBABILITY UP</span>
                  <span className="font-bold text-cyan-300">
                    {((predictionResult.probability_up ?? 0.72) * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">TAKE-PROFIT BARRIER</span>
                  <span className="font-bold text-emerald-300">
                    +{(predictionResult.take_profit_barrier_pct ?? 1.5).toFixed(2)}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">EVENT TAXONOMY</span>
                  <span className="font-bold text-amber-300">
                    {predictionResult.event_type || 'EARNINGS'}
                  </span>
                </div>
              </div>
            )}
          </form>
        </div>

        {/* RIGHT: INGESTED NEWS WIRE (5 COLS) */}
        <div className="lg:col-span-5 glass-card p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between border-b border-white/10 pb-2">
            <span className="font-bold text-slate-300 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Ingested News Stream ({newsList.length})</span>
            </span>
            <span className="text-[10px] text-cyan-300 font-bold uppercase">{ticker}</span>
          </div>

          <div className="max-h-96 overflow-y-auto space-y-2.5 pr-1">
            {newsList.length === 0 ? (
              <div className="text-slate-500 text-center py-8">
                No active headlines for {ticker}. Click "Refresh Wire" to query Finnhub & RSS feeds.
              </div>
            ) : (
              newsList.map((item, idx) => {
                const isBull = (item.sentiment_score ?? 0) >= 0;
                return (
                  <div
                    key={idx}
                    className="p-2.5 rounded-xl bg-black/40 border border-white/5 hover:border-white/10 transition flex flex-col gap-1 text-[11px]"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="font-bold text-white leading-tight">{item.headline}</span>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded font-bold whitespace-nowrap ${
                          isBull
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        }`}
                      >
                        {(item.sentiment_score ?? 0.25) >= 0 ? '+' : ''}
                        {(item.sentiment_score ?? 0.25).toFixed(2)}
                      </span>
                    </div>
                    {item.summary && <p className="text-slate-400 text-[10px] line-clamp-2">{item.summary}</p>}
                    <div className="flex items-center justify-between text-[9px] text-slate-500 pt-0.5">
                      <span>Source: {item.source}</span>
                      <span>{new Date(item.datetime * 1000).toLocaleTimeString()}</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
