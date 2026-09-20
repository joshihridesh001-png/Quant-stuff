import React from 'react';
import { Newspaper, LineChart, Dices, Dna, FlaskConical } from 'lucide-react';

export type ActivePillar = 'news' | 'econometrics' | 'regimes' | 'swarm' | 'simulation';

interface NavigationProps {
  activePillar: ActivePillar;
  onSelectPillar: (pillar: ActivePillar) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ activePillar, onSelectPillar }) => {
  const tabs = [
    { id: 'news' as ActivePillar, label: '1. Causal News & Decay', icon: Newspaper },
    { id: 'econometrics' as ActivePillar, label: '2. Econometric Rig', icon: LineChart },
    { id: 'regimes' as ActivePillar, label: '3. Game Theory & Regimes', icon: Dices },
    { id: 'swarm' as ActivePillar, label: '4. Evolutionary Swarm', icon: Dna },
    { id: 'simulation' as ActivePillar, label: '5. Simulation & Risk Studio', icon: FlaskConical },
  ];

  return (
    <nav className="glass-card p-1.5 flex items-center gap-1.5 overflow-x-auto font-mono text-xs">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activePillar === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onSelectPillar(tab.id)}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg font-bold transition whitespace-nowrap ${
              isActive
                ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-md shadow-cyan-500/20 border border-cyan-400/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-200' : 'text-slate-400'}`} />
            <span>{tab.label}</span>
          </button>
        );
      })}
    </nav>
  );
};
