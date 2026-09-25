import React, { useState, useEffect, useCallback } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts';
import {
  Dna,
  Play,
  RefreshCw,
  Sparkles,
  Award,
  ShieldCheck,
  TrendingUp,
  Cpu,
  CheckCircle2,
} from 'lucide-react';
import { quantApi } from '../api/client';
import { GenotypeItem, Genotype } from '../types/quant';

export const StrategySwarmView: React.FC = () => {
  const [generation, setGeneration] = useState<number>(0);
  const [genotypes, setGenotypes] = useState<GenotypeItem[]>([]);
  const [selectedGenotype, setSelectedGenotype] = useState<Genotype | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSeeding, setIsSeeding] = useState<boolean>(false);
  const [isStepping, setIsStepping] = useState<boolean>(false);

  // Fetch Pareto Ranked Genotypes
  const loadGenotypes = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await quantApi.getParetoGenotypes<GenotypeItem[]>(generation);
      setGenotypes(data || []);
      if (data && data.length > 0 && !selectedGenotype) {
        setSelectedGenotype(data[0].genotype);
      }
    } catch (err) {
      console.error('Failed to load Pareto genotypes:', err);
    } finally {
      setIsLoading(false);
    }
  }, [generation, selectedGenotype]);

  // Seed Population
  const handleSeedPopulation = async (size: number = 30) => {
    setIsSeeding(true);
    try {
      await quantApi.seedGenotypes(size);
      setGeneration(0);
      const data = await quantApi.getParetoGenotypes<GenotypeItem[]>(0);
      setGenotypes(data || []);
      if (data && data.length > 0) {
        setSelectedGenotype(data[0].genotype);
      }
    } catch (err) {
      console.error('Failed to seed population:', err);
    } finally {
      setIsSeeding(false);
    }
  };

  // Step Generation
  const handleStepGeneration = async () => {
    setIsStepping(true);
    try {
      const nextGen = generation + 1;
      const data = await quantApi.stepGenotypes<GenotypeItem[]>(generation, 20);
      setGeneration(nextGen);
      if (data && data.length > 0) {
        setGenotypes(data);
        setSelectedGenotype(data[0].genotype);
      }
    } catch (err) {
      console.error('Failed to step swarm generation:', err);
    } finally {
      setIsStepping(false);
    }
  };

  useEffect(() => {
    loadGenotypes();
  }, [loadGenotypes]);

  // Aggregate Stats
  const totalCount = genotypes.length;
  const eliteCount = genotypes.filter((g) => g.pareto_rank === 1).length;
  const maxDsr = genotypes.length > 0
    ? Math.max(...genotypes.map((g) => g.genotype.deflated_sharpe ?? 0))
    : 0;
  const minDrawdown = genotypes.length > 0
    ? Math.min(...genotypes.map((g) => g.genotype.max_drawdown ?? 1))
    : 0;

  // Prepare Scatter Data for Pareto Frontier
  // X: Max Drawdown % (lower is better)
  // Y: Deflated Sharpe Ratio (higher is better)
  const rank1Data = genotypes
    .filter((g) => g.pareto_rank === 1)
    .map((g) => ({
      x: ((g.genotype.max_drawdown ?? 0.1) * 100),
      y: (g.genotype.deflated_sharpe ?? 1.5),
      id: g.genotype.id,
      cohort: g.genotype.cohort,
      rank: g.pareto_rank,
      crowding: g.crowding_distance,
      genotype: g.genotype,
    }));

  const rank2Data = genotypes
    .filter((g) => g.pareto_rank === 2)
    .map((g) => ({
      x: ((g.genotype.max_drawdown ?? 0.15) * 100),
      y: (g.genotype.deflated_sharpe ?? 1.2),
      id: g.genotype.id,
      cohort: g.genotype.cohort,
      rank: g.pareto_rank,
      crowding: g.crowding_distance,
      genotype: g.genotype,
    }));

  const rankOtherData = genotypes
    .filter((g) => g.pareto_rank > 2)
    .map((g) => ({
      x: ((g.genotype.max_drawdown ?? 0.2) * 100),
      y: (g.genotype.deflated_sharpe ?? 0.9),
      id: g.genotype.id,
      cohort: g.genotype.cohort,
      rank: g.pareto_rank,
      crowding: g.crowding_distance,
      genotype: g.genotype,
    }));

  // Build Radar Chart Data for Selected Genotype's 20-Gene Chromosome
  const radarData = selectedGenotype
    ? [
        { gene: 'Frac-Diff d', value: Math.min(100, (selectedGenotype.chromosome_repr.fractional_d ?? 0.3) * 100) },
        { gene: 'Alpha Decay', value: Math.min(100, (selectedGenotype.chromosome_repr.alpha_decay ?? 0.05) * 500) },
        { gene: 'Tau Ratio', value: Math.min(100, (selectedGenotype.chromosome_repr.tau_ratio ?? 2.0) * 20) },
        { gene: 'Ambiguity β', value: Math.min(100, (selectedGenotype.chromosome_game.ambiguity_temp ?? 1.5) * 25) },
        { gene: 'Risk Aversion', value: Math.min(100, (selectedGenotype.chromosome_game.risk_aversion ?? 2.0) * 20) },
        { gene: 'Profit Take k₁', value: Math.min(100, (selectedGenotype.chromosome_infer.profit_take_mult ?? 2.0) * 25) },
        { gene: 'Stop Loss k₂', value: Math.min(100, (selectedGenotype.chromosome_infer.stop_loss_mult ?? 1.0) * 33) },
        { gene: 'Vol Target', value: Math.min(100, (selectedGenotype.chromosome_risk.vol_target ?? 0.15) * 300) },
        { gene: 'Max Leverage', value: Math.min(100, (selectedGenotype.chromosome_risk.max_weight ?? 0.25) * 200) },
        { gene: 'Turnover Budget', value: Math.min(100, (selectedGenotype.chromosome_risk.turnover_budget ?? 0.5) * 100) },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between bg-slate-900 border border-slate-800 rounded-lg p-5 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Dna className="w-5 h-5 text-emerald-400" />
            <h1 className="text-xl font-bold text-white tracking-wide">
              Pillar 4: Evolutionary Strategy Swarm
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Multi-objective NSGA-II Non-Dominated Pareto Frontier & 20-Gene Chromosome Deep Inspector
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Generation indicator */}
          <div className="flex items-center bg-slate-950 px-3 py-1.5 rounded border border-slate-800 text-xs font-mono text-slate-300">
            <span>Gen:</span>
            <span className="text-emerald-400 font-bold ml-1.5">{generation}</span>
          </div>

          {/* Seed Population */}
          <button
            onClick={() => handleSeedPopulation(30)}
            disabled={isSeeding}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded border border-slate-700 transition disabled:opacity-50"
          >
            <Sparkles className={`w-3.5 h-3.5 ${isSeeding ? 'animate-spin text-amber-400' : 'text-amber-400'}`} />
            Seed Population (30)
          </button>

          {/* Step Evolution */}
          <button
            onClick={handleStepGeneration}
            disabled={isStepping || totalCount === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded shadow-sm transition disabled:opacity-50"
          >
            <Play className={`w-3.5 h-3.5 ${isStepping ? 'animate-spin' : ''}`} />
            Step Generation
          </button>

          {/* Refresh */}
          <button
            onClick={loadGenotypes}
            disabled={isLoading}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-emerald-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Total Population</span>
            <Cpu className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-white mt-2">{totalCount}</div>
          <p className="text-[11px] text-slate-500 mt-1">Stratified Alpha & Aspirant individuals</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Elite Pareto Front (Rank 1)</span>
            <Award className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-amber-400 mt-2">
            {eliteCount}{' '}
            <span className="text-xs font-normal text-slate-400">
              ({totalCount > 0 ? ((eliteCount / totalCount) * 100).toFixed(0) : 0}%)
            </span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Non-dominated optimal chromosomes</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Max Deflated Sharpe (DSR)</span>
            <TrendingUp className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-cyan-400 mt-2">
            {maxDsr.toFixed(2)}{' '}
            <span className="text-xs font-normal text-slate-400">
              {maxDsr >= 1.0 ? '★ p<0.05' : ''}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Bailey-López de Prado significance certified</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">Min Pareto Drawdown</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-mono font-bold text-emerald-400 mt-2">
            {(minDrawdown * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Worst-case equity drawdown on frontier</p>
        </div>
      </div>

      {/* SECTION 1: 2D NSGA-II Non-Dominated Pareto Frontier Scatter Plot */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-800 gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Award className="w-4 h-4 text-amber-400" />
              2D NSGA-II Non-Dominated Pareto Frontier
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Multi-objective optimization: Deflated Sharpe Ratio (Maximize Y) vs Maximum Drawdown % (Minimize X). Click any dot to inspect chromosome.
            </p>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="flex items-center gap-1 text-emerald-400">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Rank 1 (Elite)
            </span>
            <span className="flex items-center gap-1 text-cyan-400">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" /> Rank 2
            </span>
            <span className="flex items-center gap-1 text-slate-500">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-500" /> Rank 3+
            </span>
          </div>
        </div>

        <div className="mt-4 h-80">
          {totalCount > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Max Drawdown"
                  unit="%"
                  stroke="#64748b"
                  fontSize={11}
                  domain={[0, 'auto']}
                  label={{ value: 'Max Drawdown % (Risk Objective → Lower is Better)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Deflated Sharpe Ratio"
                  stroke="#64748b"
                  fontSize={11}
                  domain={[0, 'auto']}
                  label={{ value: 'Deflated Sharpe Ratio (Return Objective → Higher is Better)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '12px' }}
                  formatter={(val: any, name: any) => [
                    name === 'Max Drawdown' ? `${Number(val).toFixed(2)}%` : Number(val).toFixed(3),
                    String(name ?? ''),
                  ]}
                />
                <Scatter
                  name="Rank 1 Elite"
                  data={rank1Data}
                  fill="#10b981"
                  stroke="#059669"
                  strokeWidth={1.5}
                  onClick={(node: any) => {
                    if (node && node.genotype) setSelectedGenotype(node.genotype);
                  }}
                  className="cursor-pointer"
                />
                <Scatter
                  name="Rank 2"
                  data={rank2Data}
                  fill="#38bdf8"
                  onClick={(node: any) => {
                    if (node && node.genotype) setSelectedGenotype(node.genotype);
                  }}
                  className="cursor-pointer"
                />
                <Scatter
                  name="Rank 3+"
                  data={rankOtherData}
                  fill="#64748b"
                  onClick={(node: any) => {
                    if (node && node.genotype) setSelectedGenotype(node.genotype);
                  }}
                  className="cursor-pointer"
                />
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 font-mono text-sm gap-3">
              <span>No strategies registered in Generation {generation}.</span>
              <button
                onClick={() => handleSeedPopulation(30)}
                disabled={isSeeding}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded font-sans text-xs font-semibold shadow transition"
              >
                Seed Generation 0 Population (30 Stratified Chromosomes)
              </button>
            </div>
          )}
        </div>
      </div>

      {/* SECTION 2: 20-Gene Chromosome Deep Inspector */}
      {selectedGenotype && (
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-4 border-b border-slate-800 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <Dna className="w-5 h-5 text-emerald-400" />
                <h2 className="text-base font-bold text-white">
                  20-Gene Chromosome Deep Inspector
                </h2>
                <span className="text-xs px-2 py-0.5 rounded font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
                  {selectedGenotype.id.substring(0, 8)}...
                </span>
                <span className="text-xs px-2 py-0.5 rounded font-mono bg-purple-500/10 text-purple-400 border border-purple-500/20">
                  Cohort: {selectedGenotype.cohort}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Decomposition across 4 institutional functional quadrants: Representation, Game Theory, Inference & Risk
              </p>
            </div>

            <div className="flex items-center gap-3 text-xs font-mono text-slate-300">
              <span>Fitness: <strong className="text-emerald-400">{selectedGenotype.fitness_score?.toFixed(3) ?? 'N/A'}</strong></span>
              <span>DSR: <strong className="text-cyan-400">{selectedGenotype.deflated_sharpe?.toFixed(2) ?? 'N/A'}</strong></span>
              <span>Max DD: <strong className="text-rose-400">{((selectedGenotype.max_drawdown ?? 0) * 100).toFixed(1)}%</strong></span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-4">
            {/* 4 Functional Quadrants */}
            <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Quadrant 1: Representation */}
              <div className="bg-slate-950 rounded border border-slate-800 p-3.5">
                <span className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider block mb-2">
                  Quadrant 1: Representation
                </span>
                <div className="space-y-1.5 text-xs font-mono text-slate-400">
                  <div className="flex justify-between">
                    <span>tau_slow:</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_repr.tau_slow}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>tau_fast:</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_repr.tau_fast}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>tau_ratio:</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_repr.tau_ratio}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>fractional_d:</span>
                    <span className="text-emerald-400 font-bold">{selectedGenotype.chromosome_repr.fractional_d}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>alpha_decay:</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_repr.alpha_decay}</span>
                  </div>
                </div>
              </div>

              {/* Quadrant 2: Game Theory */}
              <div className="bg-slate-950 rounded border border-slate-800 p-3.5">
                <span className="text-xs font-mono font-bold text-purple-400 uppercase tracking-wider block mb-2">
                  Quadrant 2: Game Theory
                </span>
                <div className="space-y-1.5 text-xs font-mono text-slate-400">
                  <div className="flex justify-between">
                    <span>ambiguity_temp (β):</span>
                    <span className="text-amber-400 font-bold">{selectedGenotype.chromosome_game.ambiguity_temp}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>risk_aversion (λ):</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_game.risk_aversion}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>predatory_intensity:</span>
                    <span className="text-rose-400 font-bold">{selectedGenotype.chromosome_game.predatory_intensity}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>logit_bull / bear / panic:</span>
                    <span className="text-slate-200 font-bold">
                      {selectedGenotype.chromosome_game.logit_bull} / {selectedGenotype.chromosome_game.logit_bear} / {selectedGenotype.chromosome_game.logit_panic}
                    </span>
                  </div>
                </div>
              </div>

              {/* Quadrant 3: Inference */}
              <div className="bg-slate-950 rounded border border-slate-800 p-3.5">
                <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider block mb-2">
                  Quadrant 3: Inference & Labeling
                </span>
                <div className="space-y-1.5 text-xs font-mono text-slate-400">
                  <div className="flex justify-between">
                    <span>profit_take_mult (k₁):</span>
                    <span className="text-emerald-400 font-bold">{selectedGenotype.chromosome_infer.profit_take_mult}σ</span>
                  </div>
                  <div className="flex justify-between">
                    <span>stop_loss_mult (k₂):</span>
                    <span className="text-rose-400 font-bold">{selectedGenotype.chromosome_infer.stop_loss_mult}σ</span>
                  </div>
                  <div className="flex justify-between">
                    <span>holding_period (Δτ):</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_infer.holding_period} bars</span>
                  </div>
                  <div className="flex justify-between">
                    <span>execution_horizon:</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_infer.execution_horizon}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>meta_label_thresh:</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_infer.meta_label_thresh}</span>
                  </div>
                </div>
              </div>

              {/* Quadrant 4: Risk Firewall */}
              <div className="bg-slate-950 rounded border border-slate-800 p-3.5">
                <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider block mb-2">
                  Quadrant 4: Risk Firewall
                </span>
                <div className="space-y-1.5 text-xs font-mono text-slate-400">
                  <div className="flex justify-between">
                    <span>vol_target:</span>
                    <span className="text-slate-200 font-bold">{((selectedGenotype.chromosome_risk.vol_target ?? 0.15) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>max_weight (leverage):</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_risk.max_weight}x</span>
                  </div>
                  <div className="flex justify-between">
                    <span>max_drawdown_limit:</span>
                    <span className="text-rose-400 font-bold">{((selectedGenotype.chromosome_risk.max_drawdown_limit ?? 0.1) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>turnover_budget:</span>
                    <span className="text-slate-200 font-bold">{selectedGenotype.chromosome_risk.turnover_budget}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Polar Radar Chart of Chromosome */}
            <div className="bg-slate-950 rounded border border-slate-800 p-3 flex flex-col justify-between">
              <span className="text-xs font-mono font-bold text-slate-300 block text-center">
                Normalized Chromosome Footprint
              </span>

              <div className="h-60 mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                    <PolarGrid stroke="#1e293b" />
                    <PolarAngleAxis dataKey="gene" stroke="#64748b" fontSize={9} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#334155" fontSize={8} />
                    <Radar
                      name="Chromosome Gene Level"
                      dataKey="value"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.4}
                    />
                    <Legend wrapperStyle={{ fontSize: '10px' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 3: Population Registry Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
          <span className="text-base font-bold text-white flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-400" />
            Genotype Population Registry (Generation {generation})
          </span>
          <span className="text-xs font-mono text-slate-400">
            {totalCount} Individuals Sorted by Pareto Dominance
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-left">
                <th className="pb-2">Genotype ID</th>
                <th className="pb-2">Cohort</th>
                <th className="pb-2 text-center">Pareto Rank</th>
                <th className="pb-2 text-right">Crowding Distance</th>
                <th className="pb-2 text-right">Deflated Sharpe</th>
                <th className="pb-2 text-right">Max Drawdown</th>
                <th className="pb-2 text-right">Fitness Score</th>
                <th className="pb-2 text-center">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {genotypes.map((item) => {
                const g = item.genotype;
                const isSelected = selectedGenotype?.id === g.id;
                return (
                  <tr
                    key={g.id}
                    className={`hover:bg-slate-800/40 transition cursor-pointer ${
                      isSelected ? 'bg-emerald-500/10 font-semibold' : ''
                    }`}
                    onClick={() => setSelectedGenotype(g)}
                  >
                    <td className="py-2.5 flex items-center gap-1.5">
                      {item.pareto_rank === 1 && (
                        <Award className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                      )}
                      <span className={isSelected ? 'text-emerald-400 font-bold' : 'text-slate-300'}>
                        {g.id.substring(0, 8)}...
                      </span>
                    </td>
                    <td className="py-2.5 text-slate-400">{g.cohort}</td>
                    <td className="py-2.5 text-center">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[11px] font-bold ${
                          item.pareto_rank === 1
                            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                            : item.pareto_rank === 2
                            ? 'bg-cyan-500/20 text-cyan-400'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        Rank {item.pareto_rank}
                      </span>
                    </td>
                    <td className="py-2.5 text-right text-slate-400">
                      {item.crowding_distance === Infinity ? '∞' : typeof item.crowding_distance === 'number' && Number.isFinite(item.crowding_distance) ? item.crowding_distance.toFixed(3) : '0.000'}
                    </td>
                    <td
                      className={`py-2.5 text-right font-bold ${
                        (g.deflated_sharpe ?? 0) >= 1.0 ? 'text-emerald-400' : 'text-slate-300'
                      }`}
                    >
                      {g.deflated_sharpe?.toFixed(2) ?? 'N/A'}
                    </td>
                    <td className="py-2.5 text-right text-rose-400 font-bold">
                      {g.max_drawdown ? `${(g.max_drawdown * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="py-2.5 text-right text-slate-200">
                      {g.fitness_score?.toFixed(3) ?? 'N/A'}
                    </td>
                    <td className="py-2.5 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedGenotype(g);
                        }}
                        className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300"
                      >
                        {isSelected ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 inline" />
                        ) : (
                          'Select'
                        )}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
