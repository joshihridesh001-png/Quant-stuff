import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string;
  subValue?: string;
  badgeText?: string;
  badgeVariant?: 'green' | 'blue' | 'purple' | 'amber' | 'rose';
  icon?: LucideIcon;
  borderColor?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subValue,
  badgeText,
  badgeVariant = 'green',
  icon: Icon,
  borderColor = 'border-t-emerald-500',
}) => {
  const badgeStyles = {
    green: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    rose: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  }[badgeVariant];

  return (
    <div className={`glass-card p-3 flex flex-col justify-between border-t-2 ${borderColor} transition hover:bg-slate-900/60`}>
      <div className="flex items-center justify-between text-slate-400 text-[10px] font-mono uppercase font-bold tracking-wider">
        <span>{title}</span>
        {badgeText && (
          <span className={`text-[9px] px-1.5 py-0.5 rounded border font-bold ${badgeStyles}`}>
            {badgeText}
          </span>
        )}
        {Icon && !badgeText && <Icon className="w-3.5 h-3.5 text-slate-400" />}
      </div>
      <div className="text-lg sm:text-xl font-black font-mono text-white mt-1 tracking-tight">
        {value}
      </div>
      {subValue && (
        <div className="text-[11px] font-mono text-slate-400 mt-1 flex items-center justify-between">
          {subValue}
        </div>
      )}
    </div>
  );
};
