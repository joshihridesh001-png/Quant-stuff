import React, { useState } from 'react';
import { ShieldAlert, KeyRound, X } from 'lucide-react';
import { useQuantContext } from '../api/context';

interface DisarmModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DisarmModal: React.FC<DisarmModalProps> = ({ isOpen, onClose }) => {
  const { resetKillSwitch } = useQuantContext();
  const [token, setToken] = useState('QUANT_SECRET_2026_PROD_RECOVERY_KEY');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await resetKillSwitch(token.trim());
      onClose();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="glass-card max-w-md w-full p-5 border border-amber-500/40 shadow-2xl relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2.5 text-amber-400 font-mono font-black text-sm">
          <ShieldAlert className="w-5 h-5 text-amber-400" />
          <span>DISARM EMERGENCY KILL SWITCH</span>
        </div>

        <p className="text-xs text-slate-300 font-mono mt-2">
          Enter cryptographically verified HMAC administrator secret to disarm the circuit breaker lockout and restore trading desk operations.
        </p>

        <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3 font-mono">
          <div>
            <label className="text-[10px] text-slate-400 block uppercase font-bold mb-1">
              ADMIN HMAC RECOVERY KEY
            </label>
            <div className="relative">
              <KeyRound className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
              <input
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                className="w-full bg-black/60 border border-white/10 rounded-xl pl-9 pr-3 py-2 text-xs text-white outline-none focus:border-amber-400"
                placeholder="Recovery secret..."
                required
              />
            </div>
          </div>

          <div className="flex items-center justify-end gap-2 mt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-bold hover:bg-slate-700 transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-black shadow-lg shadow-amber-600/30 transition disabled:opacity-50"
            >
              {loading ? 'Disarming...' : 'Disarm & Re-enable Desk'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
