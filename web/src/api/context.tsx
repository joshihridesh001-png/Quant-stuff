import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { useRiskWebSocket, useExecutionsWebSocket, ConnectionStatus } from './useWebSocket';
import { apiClient } from './client';
import type { RiskMetrics, GatewayHealthDTO, ParentOrder, AutonomousStatus } from '../types/quant';

interface QuantContextValue {
  riskStatus: ConnectionStatus;
  execStatus: ConnectionStatus;
  risk: RiskMetrics | null;
  gateways: GatewayHealthDTO[];
  orders: ParentOrder[];
  swarm: AutonomousStatus | null;
  isPanic: boolean;
  refreshSwarm: () => Promise<void>;
  refreshOrders: () => Promise<void>;
  triggerPanic: () => Promise<void>;
  resetKillSwitch: (secret: string) => Promise<void>;
  startSwarm: () => Promise<void>;
  pauseSwarm: () => Promise<void>;
  stepSwarm: () => Promise<void>;
  stopSwarm: () => Promise<void>;
}

const QuantContext = createContext<QuantContextValue | null>(null);

export const QuantProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { status: riskStatus, risk, gateways, setRisk } = useRiskWebSocket();
  const { status: execStatus, orders, setOrders } = useExecutionsWebSocket();
  const [swarm, setSwarm] = useState<AutonomousStatus | null>(null);

  const refreshSwarm = useCallback(async () => {
    try {
      const data = await apiClient.getSwarmStatus<AutonomousStatus>();
      setSwarm(data);
    } catch {
      // ignore
    }
  }, []);

  const refreshOrders = useCallback(async () => {
    try {
      const data = await apiClient.getOrders<ParentOrder[]>();
      setOrders(data);
    } catch {
      // ignore
    }
  }, [setOrders]);

  const triggerPanic = useCallback(async () => {
    try {
      await apiClient.triggerPanic();
      setRisk((prev) => (prev ? { ...prev, is_kill_switch_active: true } : null));
    } catch (err) {
      alert('Panic trigger error: ' + err);
    }
  }, [setRisk]);

  const resetKillSwitch = useCallback(async (secret: string) => {
    try {
      await apiClient.resetKillSwitch(secret);
      setRisk((prev) => (prev ? { ...prev, is_kill_switch_active: false } : null));
      await refreshSwarm();
    } catch (err) {
      alert('Kill switch reset error: ' + err);
    }
  }, [setRisk, refreshSwarm]);

  const startSwarm = useCallback(async () => {
    try {
      await apiClient.startSwarm();
      await refreshSwarm();
      await refreshOrders();
    } catch (err) {
      alert('Start swarm error: ' + err);
    }
  }, [refreshSwarm, refreshOrders]);

  const pauseSwarm = useCallback(async () => {
    try {
      await apiClient.pauseSwarm();
      await refreshSwarm();
    } catch (err) {
      alert('Pause swarm error: ' + err);
    }
  }, [refreshSwarm]);

  const stepSwarm = useCallback(async () => {
    try {
      await apiClient.stepSwarm();
      await refreshSwarm();
      await refreshOrders();
    } catch (err) {
      alert('Step swarm error: ' + err);
    }
  }, [refreshSwarm, refreshOrders]);

  const stopSwarm = useCallback(async () => {
    try {
      await apiClient.stopSwarm();
      await refreshSwarm();
    } catch (err) {
      alert('Stop swarm error: ' + err);
    }
  }, [refreshSwarm]);

  useEffect(() => {
    refreshSwarm();
    const timer = setInterval(refreshSwarm, 3000);
    return () => clearInterval(timer);
  }, [refreshSwarm]);

  const isPanic = !!risk?.is_kill_switch_active;

  return (
    <QuantContext.Provider
      value={{
        riskStatus,
        execStatus,
        risk,
        gateways,
        orders,
        swarm,
        isPanic,
        refreshSwarm,
        refreshOrders,
        triggerPanic,
        resetKillSwitch,
        startSwarm,
        pauseSwarm,
        stepSwarm,
        stopSwarm,
      }}
    >
      {children}
    </QuantContext.Provider>
  );
};

export function useQuantContext(): QuantContextValue {
  const ctx = useContext(QuantContext);
  if (!ctx) {
    throw new Error('useQuantContext must be used within a QuantProvider');
  }
  return ctx;
}

export const useQuant = useQuantContext;
