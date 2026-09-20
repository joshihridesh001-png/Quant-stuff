// Authenticated REST API Client for Quantitative Research Workbench

let cachedToken: string | null = null;

export async function getAuthToken(): Promise<string | null> {
  if (cachedToken) return cachedToken;
  try {
    const res = await fetch('/api/v1/auth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: 'admin',
        password: 'quant-secret-pass',
        role: 'ADMIN',
      }),
    });
    if (res.ok) {
      const data = await res.json();
      cachedToken = data.access_token;
      return cachedToken;
    }
  } catch (err) {
    console.error('Failed to negotiate auth token:', err);
  }
  return null;
}

export async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  let token = cachedToken;
  if (!token) {
    token = await getAuthToken();
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let res = await fetch(endpoint, { ...options, headers });

  // If 401, refresh token once and retry
  if (res.status === 401) {
    cachedToken = null;
    token = await getAuthToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
      res = await fetch(endpoint, { ...options, headers });
    }
  }

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status} ${res.statusText}`;
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail?.message || errorJson.detail || errorDetail;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return res.json() as Promise<T>;
}

export const apiClient = {
  get: <T>(url: string) => apiFetch<T>(url, { method: 'GET' }),
  post: <T>(url: string, body?: unknown) =>
    apiFetch<T>(url, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
    }),
  put: <T>(url: string, body?: unknown) =>
    apiFetch<T>(url, {
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined,
    }),
  delete: <T>(url: string) => apiFetch<T>(url, { method: 'DELETE' }),

  // Swarm
  startSwarm: () => apiFetch('/api/v1/autonomous/start', { method: 'POST' }),
  pauseSwarm: () => apiFetch('/api/v1/autonomous/pause', { method: 'POST' }),
  stepSwarm: () => apiFetch('/api/v1/autonomous/step', { method: 'POST' }),
  stopSwarm: () => apiFetch('/api/v1/autonomous/stop', { method: 'POST' }),
  getSwarmStatus: <T>() => apiFetch<T>('/api/v1/autonomous/status'),

  // Risk & Kill Switch
  triggerPanic: () =>
    apiFetch('/api/v1/risk/panic', {
      method: 'POST',
      body: JSON.stringify({ reason: 'Operator Emergency Lockdown' }),
    }),
  resetKillSwitch: (secret: string) =>
    apiFetch('/api/v1/risk/reset', {
      method: 'POST',
      body: JSON.stringify({ admin_token: secret }),
    }),
  getRiskStatus: <T>() => apiFetch<T>('/api/v1/risk/status'),

  // Genotypes
  getParetoGenotypes: <T>(generation: number = 0) =>
    apiFetch<T>(`/api/v1/genotypes/pareto?generation=${generation}`),
  seedGenotypes: (count: number) =>
    apiFetch('/api/v1/genotypes/seed', {
      method: 'POST',
      body: JSON.stringify({ population_size: count }),
    }),

  // Orders
  getOrders: <T>() => apiFetch<T>('/api/v1/orders'),
  submitOrder: <T>(orderPayload: unknown) =>
    apiFetch<T>('/api/v1/orders', {
      method: 'POST',
      body: JSON.stringify(orderPayload),
    }),
  getOrderTca: <T>(orderId: string) => apiFetch<T>(`/api/v1/orders/${orderId}/tca`),

  // Simulation
  runSimulation: <T>(simPayload: unknown) =>
    apiFetch<T>('/api/v1/simulation/run', {
      method: 'POST',
      body: JSON.stringify(simPayload),
    }),

  // News & Events
  getNewsWire: <T>(ticker: string) =>
    apiFetch<T>(`/api/v1/providers/company-news/${ticker}`),
  getLatestNews: <T>(limit: number = 50) =>
    apiFetch<T>(`/api/v1/news/latest?limit=${limit}`),
  getNewsDecayState: <T>(ticker: string) =>
    apiFetch<T>(`/api/v1/events/state/${ticker}`),
  harvestNews: () => apiFetch('/api/v1/news/harvest', { method: 'POST' }),
  predictHeadline: <T>(headline: string, ticker: string) =>
    apiFetch<T>('/api/v1/news/predict', {
      method: 'POST',
      body: JSON.stringify({
        headline,
        ticker: ticker,
        current_price: 150.0,
      }),
    }),

  // Econometrics
  getFfdSearch: <T>(symbol: string, threshold: number = 0.05) =>
    apiFetch<T>(`/api/v1/econometrics/ffd/search?symbol=${encodeURIComponent(symbol)}&threshold=${threshold}`),
  getRealizedVolatility: <T>(symbol: string, window: number = 20) =>
    apiFetch<T>(`/api/v1/econometrics/volatility/parkinson?symbol=${encodeURIComponent(symbol)}&window=${window}`),
  simulateTripleBarrier: <T>(params: {
    symbol: string;
    profit_multiplier: number;
    stop_multiplier: number;
    horizon_bars: number;
    volatility_window: number;
    side: number;
  }) =>
    apiFetch<T>('/api/v1/econometrics/triple-barrier/simulate', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  // Game Theory & Regimes
  getRegimeStatus: <T>(symbol: string, cusumThreshold: number = 3.0, cusumDrift: number = 0.5) =>
    apiFetch<T>(`/api/v1/game-theory/regimes?symbol=${encodeURIComponent(symbol)}&cusum_threshold=${cusumThreshold}&cusum_drift=${cusumDrift}`),
  computePayoffMatrix: <T>(ambiguityBeta: number, riskAversion: number = 2.0) =>
    apiFetch<T>('/api/v1/game-theory/payoff-matrix', {
      method: 'POST',
      body: JSON.stringify({ ambiguity_beta: ambiguityBeta, risk_aversion: riskAversion }),
    }),
};

export const quantApi = apiClient;
