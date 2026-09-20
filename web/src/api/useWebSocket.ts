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
          if (msg.type === 'KILL_SWITCH_EVENT' && msg.is_active !== undefined) {
            setRisk((prev) => (prev ? { ...prev, is_kill_switch_active: !!msg.is_active } : null));
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
