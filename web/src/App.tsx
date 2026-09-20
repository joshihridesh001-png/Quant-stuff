import React, { useState } from 'react';
import { QuantProvider } from './api/context';
import { HUD } from './components/HUD';
import { Navigation, ActivePillar } from './components/Navigation';
import { NewsDecayView } from './views/NewsDecayView';
import { EconometricsView } from './views/EconometricsView';
import { RegimesView } from './views/RegimesView';
import { StrategySwarmView } from './views/StrategySwarmView';
import { SimulationRiskView } from './views/SimulationRiskView';
import { Activity, Shield, Terminal, Cpu } from 'lucide-react';

import { ErrorBoundary } from './components/ErrorBoundary';

const WorkbenchContent: React.FC = () => {
  const [activePillar, setActivePillar] = useState<ActivePillar>('news');

  const renderActiveView = () => {
    switch (activePillar) {
      case 'news':
        return <NewsDecayView />;
      case 'econometrics':
        return <EconometricsView />;
      case 'regimes':
        return <RegimesView />;
      case 'swarm':
        return <StrategySwarmView />;
      case 'simulation':
        return <SimulationRiskView />;
      default:
        return <NewsDecayView />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Pinned Top Telemetry HUD */}
      <div className="sticky top-0 z-40 bg-slate-950/90 backdrop-blur-md border-b border-slate-800/80 px-4 sm:px-6 py-3">
        <HUD />
      </div>

      {/* Navigation Tab Bar */}
      <div className="px-4 sm:px-6 pt-3 pb-1">
        <Navigation activePillar={activePillar} onSelectPillar={setActivePillar} />
      </div>

      {/* Main Quantitative Pillar View Container */}
      <main className="flex-1 px-4 sm:px-6 py-4 max-w-[1800px] w-full mx-auto">
        <ErrorBoundary key={activePillar} fallbackTitle={`Pillar: ${activePillar.toUpperCase()} Error`}>
          {renderActiveView()}
        </ErrorBoundary>
      </main>

      {/* Bottom Quantitative System Status Bar */}
      <footer className="border-t border-slate-900 bg-slate-950/80 px-4 sm:px-6 py-2 flex flex-col sm:flex-row items-center justify-between text-[11px] font-mono text-slate-500 gap-2">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-emerald-400">
            <Activity className="w-3 h-3" /> Engine: Online (FastAPI + React 19)
          </span>
          <span className="flex items-center gap-1.5 text-slate-400">
            <Shield className="w-3 h-3 text-cyan-400" /> Pre-Trade Gate: Bayesian Log-Odds Active
          </span>
          <span className="flex items-center gap-1.5 text-slate-400">
            <Cpu className="w-3 h-3 text-purple-400" /> NSGA-II Swarm: Stratified
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-slate-500">
            <Terminal className="w-3 h-3" /> Port 8000 | WebSocket Dual-Feed | v2.0-PROD
          </span>
        </div>
      </footer>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <QuantProvider>
      <WorkbenchContent />
    </QuantProvider>
  );
};

export default App;
