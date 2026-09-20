import React, { useState } from 'react';
import { useQuantContext } from '../api/context';
import { MetricCard } from './MetricCard';
import { DisarmModal } from './DisarmModal';
import { Play, Pause, StepForward, Square, AlertOctagon, ShieldCheck, ShieldAlert } from 'lucide-react';

export const HUD: React.FC = () => {
  const {
    riskStatus,
    execStatus,
    risk,
    swarm,
    isPanic,
    triggerPanic,
    startSwarm,
    pauseSwarm,
    stepSwarm,
    stopSwarm,
  } = useQuantContext();

  const [showDisarm, setShowDisarm] = useState(false);

  // Formatted Metrics
  const nav = risk?.nav ?? 10000.0;
  const returnPct = ((nav - 10000.0) / 10000.0) * 100.0;
  const cash = risk?.cash ?? 10000.0;
  const freeMargin = risk?.free_margin ?? 10000.0;
  const unrealizedPnl = risk?.unrealized_pnl ?? 0.0;
  const realizedPnl = risk?.realized_pnl ?? 0.0;
  const grossLev = risk?.gross_leverage ?? 0.0;
  const netLev = risk?.net_leverage ?? 0.0;
  const drawdown = (risk?.intraday_drawdown_pct ?? 0.0) * 100.0;
  const activePositionsCount = Object.values(risk?.positions ?? {}).filter((q) => Math.abs(q) > 1e-6).length;

  const swarmState = swarm?.state ?? 'IDLE';
  const swarmIter = swarm?.iteration ?? 0;

  return (
    <header className="flex flex-col gap-3">
      {/* 1. TOP STATUS & ORCHESTRATION BAR */}
      <div className="glass-card px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 font-mono">
        {/* Brand & Connection Telemetry */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-md bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center font-black text-[10px] text-black shadow-lg shadow-cyan-500/30">
              Q
            </div>
            <span className="font-extrabold text-sm tracking-wider bg-gradient-to-r from-white via-slate-200 to-cyan-300 bg-clip-text text-transparent">
              QUANT ALPHA WORKBENCH
            </span>
            <span className="text-[9px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-bold">
              PROD v2.0
            </span>
          </div>

          <div className="h-4 w-px bg-white/10 hidden sm:block"></div>

          {/* WebSocket indicators */}
          <div className="flex items-center gap-2 text-[10px]">
            <div className="flex items-center gap-1.5 bg-black/40 px-2 py-0.5 rounded-lg border border-white/5">
              <span
                className={`w-2 h-2 rounded-full ${
                  riskStatus === 'CONNECTED' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
                }`}
              ></span>
              <span className="text-slate-400">WS Risk:</span>
              <span className={`font-bold ${riskStatus === 'CONNECTED' ? 'text-emerald-400' : 'text-rose-400'}`}>
                {riskStatus}
              </span>
            </div>

            <div className="flex items-center gap-1.5 bg-black/40 px-2 py-0.5 rounded-lg border border-white/5">
              <span
                className={`w-2 h-2 rounded-full ${
                  execStatus === 'CONNECTED' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
                }`}
              ></span>
              <span className="text-slate-400">WS Exec:</span>
              <span className={`font-bold ${execStatus === 'CONNECTED' ? 'text-emerald-400' : 'text-rose-400'}`}>
                {execStatus}
              </span>
            </div>
          </div>
        </div>

        {/* Autonomous Swarm Controls & Panic Button */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <div className="flex items-center gap-2 bg-black/50 px-2.5 py-1 rounded-xl border border-white/10 text-[11px]">
            <span className="text-slate-400 font-bold">SWARM DAEMON:</span>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-extrabold uppercase tracking-wide ${
                swarmState === 'RUNNING'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 animate-pulse'
                  : swarmState === 'PAUSED'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              {swarmState}
            </span>
            <span className="text-cyan-400 font-bold">Iter #{swarmIter}</span>
          </div>

          <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/5 text-xs">
            <button
              onClick={startSwarm}
              disabled={swarmState === 'RUNNING'}
              className="flex items-center gap-1 px-2 py-1 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-300 font-bold border border-emerald-500/30 transition disabled:opacity-30 text-[11px]"
              title="Start Continuous Autonomous Rebalancing Swarm"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>START</span>
            </button>
            <button
              onClick={pauseSwarm}
              disabled={swarmState !== 'RUNNING'}
              className="p-1.5 rounded-lg bg-amber-600/20 hover:bg-amber-600/40 text-amber-300 font-bold border border-amber-500/30 transition disabled:opacity-30"
              title="Pause Swarm"
            >
              <Pause className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={stepSwarm}
              className="p-1.5 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/40 text-cyan-300 font-bold border border-cyan-500/30 transition"
              title="Step Single Cycle"
            >
              <StepForward className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={stopSwarm}
              disabled={swarmState === 'STOPPED' || swarmState === 'IDLE'}
              className="flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold border border-white/10 transition disabled:opacity-30 text-[11px]"
              title="Stop Swarm"
            >
              <Square className="w-3 h-3" />
              <span>STOP</span>
            </button>
          </div>

          <div className="h-4 w-px bg-white/10 hidden md:block"></div>

          {/* Kill Switch Panic / Reset */}
          {!isPanic ? (
            <button
              onClick={triggerPanic}
              className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-rose-600 to-red-700 hover:from-rose-500 hover:to-red-600 text-white font-extrabold text-[11px] tracking-wider border border-rose-400/40 shadow-lg shadow-rose-600/30 transition flex items-center gap-1.5 active:scale-95"
            >
              <AlertOctagon className="w-3.5 h-3.5 animate-pulse" />
              <span>🚨 PANIC LOCKOUT</span>
            </button>
          ) : (
            <button
              onClick={() => setShowDisarm(true)}
              className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-amber-600 to-yellow-600 hover:from-amber-500 hover:to-yellow-500 text-white font-extrabold text-[11px] tracking-wider border border-amber-400/40 shadow-lg shadow-amber-600/30 transition flex items-center gap-1.5 active:scale-95"
            >
              <ShieldAlert className="w-3.5 h-3.5 animate-bounce" />
              <span>🔓 DISARM LOCKOUT</span>
            </button>
          )}
        </div>
      </div>

      {/* 2. SIX FINANCIAL KPI METRIC CARDS */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 font-mono">
        <MetricCard
          title="Portfolio NAV"
          value={`$${nav.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subValue={`Return: ${returnPct >= 0 ? '+' : ''}${returnPct.toFixed(2)}%`}
          badgeText="LIVE"
          badgeVariant="green"
          borderColor="border-t-emerald-500"
        />

        <MetricCard
          title="Cash & Margin"
          value={`$${cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subValue={`Free: $${freeMargin.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          badgeText="LIQUID"
          badgeVariant="blue"
          borderColor="border-t-cyan-500"
        />

        <MetricCard
          title="Unrealized P&L"
          value={`${unrealizedPnl >= 0 ? '+' : ''}$${unrealizedPnl.toFixed(2)}`}
          subValue={`Open: ${activePositionsCount} Positions`}
          badgeText="MTM"
          badgeVariant={unrealizedPnl >= 0 ? 'green' : 'rose'}
          borderColor="border-t-blue-500"
        />

        <MetricCard
          title="Realized P&L"
          value={`${realizedPnl >= 0 ? '+' : ''}$${realizedPnl.toFixed(2)}`}
          subValue="Settled Banked"
          badgeText="CLOSED"
          badgeVariant={realizedPnl >= 0 ? 'green' : 'amber'}
          borderColor="border-t-teal-500"
        />

        <MetricCard
          title="Firm Leverage"
          value={`${grossLev.toFixed(2)}x Gross`}
          subValue={`Net: ${netLev >= 0 ? '+' : ''}${netLev.toFixed(2)}x`}
          badgeText="CAP 2.0x"
          badgeVariant="purple"
          borderColor="border-t-purple-500"
        />

        <MetricCard
          title="Intraday Drawdown"
          value={`${drawdown.toFixed(2)}%`}
          subValue={isPanic ? 'LOCKED OUT' : 'ARMED (< 5.0%)'}
          badgeText={isPanic ? 'TRIPPED' : 'SECURE'}
          badgeVariant={isPanic ? 'rose' : 'green'}
          borderColor={isPanic ? 'border-t-rose-500' : 'border-t-amber-500'}
          icon={isPanic ? ShieldAlert : ShieldCheck}
        />
      </div>

      <DisarmModal isOpen={showDisarm} onClose={() => setShowDisarm(false)} />
    </header>
  );
};
