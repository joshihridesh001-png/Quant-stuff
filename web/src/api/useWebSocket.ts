import { useEffect, useRef, useState, useCallback } from 'react';
import type { RiskMetrics, GatewayHealthDTO, ParentOrder, RiskWebSocketMessage, ExecWebSocketMessage } from '../types/quant';

export type ConnectionStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED';

export function useRiskWebSocket() {
  const [status, setStatus] = useState<ConnectionStatus>('CONNECTING');
  const [risk, setRisk] = useState<RiskMetrics | null>(null);
  const [gateways, setGateways] = useState<GatewayHealthDTO[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    // In dev Vite proxy redirects /ws to backend ws://127.0.0.1:8000/api/v1/ws
    // In prod FastAPI serves from host
    const url = `${proto}//${host}/api/v1/ws/risk`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      setStatus('CONNECTING');

      ws.onopen = () => {
        setStatus('CONNECTED');
      };

      ws.onmessage = (event) => {
        try {
          const msg: RiskWebSocketMessage = JSON.parse(event.data);
          if (msg.risk) {
            setRisk(msg.risk);
          }
          if (msg.gateways) {
            setGateways(msg.gateways);
          }
          if (msg.type === 'KILL_SWITCH_EVENT') {
            const active = msg.is_active !== undefined ? !!msg.is_active : !!msg.risk?.is_kill_switch_active;
            setRisk((prev) => (prev ? { ...prev, is_kill_switch_active: active } : null));
          }
        } catch (err) {
          console.warn('Risk WS message parse error:', err);
        }
      };

      ws.onclose = () => {
        setStatus('DISCONNECTED');
        wsRef.current = null;
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = () => {
        setStatus('DISCONNECTED');
      };
    } catch (err) {
      console.warn('Risk WS connection error:', err);
      setStatus('DISCONNECTED');
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { status, risk, gateways, setRisk };
}

export function useExecutionsWebSocket() {
  const [status, setStatus] = useState<ConnectionStatus>('CONNECTING');
  const [orders, setOrders] = useState<ParentOrder[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    const url = `${proto}//${host}/api/v1/ws/executions`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      setStatus('CONNECTING');

      ws.onopen = () => {
        setStatus('CONNECTED');
      };

      ws.onmessage = (event) => {
        try {
          const msg: ExecWebSocketMessage = JSON.parse(event.data);
          if (msg.type === 'SNAPSHOT' && Array.isArray(msg.orders)) {
            setOrders(msg.orders);
          } else if (msg.type === 'ORDER_UPDATE' && msg.order) {
            const updated = msg.order;
            setOrders((prev) => {
              const idx = prev.findIndex((o) => o.order_id === updated.order_id);
              if (idx !== -1) {
                const next = [...prev];
                next[idx] = updated;
                return next;
              }
              return [updated, ...prev];
            });
          } else if (msg.type === 'CHILD_FILL' && msg.parent_id && msg.fill) {
            const childFill = msg.fill;
            setOrders((prev) => {
              const idx = prev.findIndex((o) => o.order_id === msg.parent_id);
              if (idx === -1) return prev;
              const parentOrder = prev[idx];
              const childFills = parentOrder.child_fills ? [...parentOrder.child_fills] : [];
              const childIdx = childFills.findIndex((c) => c.child_id === childFill.child_id);
              if (childIdx !== -1) {
                // Duplicate packet or fill update: avoid double-counting
                childFills[childIdx] = childFill;
                const next = [...prev];
                next[idx] = { ...parentOrder, child_fills: childFills };
                return next;
              }
              childFills.push(childFill);
              const fillQty = childFill.quantity || 0;
              const newFilled = parentOrder.filled_quantity + fillQty;
              const newLeaves = Math.max(0, parentOrder.leaves_quantity - fillQty);
              const updatedParent: ParentOrder = {
                ...parentOrder,
                filled_quantity: newFilled,
                leaves_quantity: newLeaves,
                is_closed: newLeaves <= 0,
                child_fills: childFills,
              };
              const next = [...prev];
              next[idx] = updatedParent;
              return next;
            });
          }
        } catch (err) {
          console.warn('Exec WS message parse error:', err);
        }
      };

      ws.onclose = () => {
        setStatus('DISCONNECTED');
        wsRef.current = null;
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = () => {
        setStatus('DISCONNECTED');
      };
    } catch (err) {
      console.warn('Exec WS connection error:', err);
      setStatus('DISCONNECTED');
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { status, orders, setOrders };
}
