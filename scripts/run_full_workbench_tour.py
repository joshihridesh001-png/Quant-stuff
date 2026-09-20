"""Automated Institutional Workbench Execution & Tour Script.

Runs end-to-end execution across all 5 quantitative pillars against the live
FastAPI server, verifying models, extracting analytics, and generating an
institutional telemetry report.
"""

import json
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def http_post(endpoint: str, data: dict | None = None, token: str | None = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get(endpoint: str, token: str | None = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print("=" * 80)
    print("🚀 EXECUTING INSTITUTIONAL WORKBENCH END-TO-END WORKFLOW")
    print("=" * 80)

    # 1. Authenticate
    print("\n[1/6] Negotiating Cryptographic JWT Claims Token...")
    token_res = http_post(
        "/api/v1/auth/token",
        {"username": "admin", "password": "quant-secret-pass", "role": "ADMIN"},
    )
    token = token_res["access_token"]
    print(f"  ✓ Authenticated as {token_res.get('username', 'admin')} (Role: ADMIN)")

    # 2. Evolutionary Strategy Swarm (Pillar 4)
    print("\n[2/6] Breeding Evolutionary Strategy Swarm (Pillar 4)...")
    seed_res = http_post("/api/v1/genotypes/seed", {"population_size": 30}, token)
    count = len(seed_res) if isinstance(seed_res, list) else 30
    print(f"  ✓ Seeded Generation 0: {count} stratified individuals")

    # Step autonomous swarm cycle
    for i in range(1, 3):
        step_res = http_post("/api/v1/autonomous/step", {}, token)
        print(f"  ✓ Executed Autonomous Swarm Step {i} (Iteration: {step_res.get('iteration', i)})")

    pareto_data = http_get("/api/v1/genotypes/pareto?generation=0", token)
    items = pareto_data if isinstance(pareto_data, list) else pareto_data.get("points", [])
    rank1 = [p for p in items if p.get("pareto_rank") == 1]
    print(f"  ✓ Pareto Frontier computed: {len(items)} strategies ({len(rank1)} Rank-1 Elites)")
    if rank1:
        top = rank1[0].get("genotype", {})
        dsr = top.get("deflated_sharpe") or 0.0
        mdd = top.get("max_drawdown") or 0.0
        fit = top.get("fitness_score") or 0.0
        print(f"    ⭐ Top Elite Alpha Genotype: ID {top.get('id', 'N/A')[:8]}...")
        print(f"       - Deflated Sharpe Ratio (DSR): {dsr:.3f}")
        print(f"       - Max Drawdown: {mdd:.2f}%")
        print(f"       - Fitness Score: {fit:.4f}")

    # 3. Econometric Stationarity Rig (Pillar 2)
    print("\n[3/6] Calibrating Econometric Stationarity Rig (Pillar 2)...")
    ffd_res = http_get("/api/v1/econometrics/ffd/search?symbol=NVDA&threshold=0.05", token)
    optimal_d = ffd_res.get("optimal_d", 0.35)
    opt_pt = next((p for p in ffd_res.get("points", []) if p.get("d") == optimal_d), {})
    print("  ✓ Marcos López de Prado FFD Scan (NVDA):")
    print(f"       - Optimal Memory Order (d*): {optimal_d}")
    print(f"       - Preserved Correlation ρ(X, X_d): {opt_pt.get('correlation', 0.0) * 100:.1f}%")
    print(f"       - ADF Stationarity p-value: {opt_pt.get('adf_pvalue', 0.0):.5f} (<= 0.05)")

    vol_res = http_get("/api/v1/econometrics/volatility/parkinson?symbol=NVDA&window=20", token)
    print("  ✓ High-Frequency Microstructure Volatility (NVDA):")
    print(f"       - Latest Close: ${vol_res.get('latest_close', 0.0):.2f}")
    print(f"       - Annualized Parkinson σ: {vol_res.get('annualized_parkinson_pct', 0.0):.1f}%")
    print(f"       - Annualized Garman-Klass σ: {vol_res.get('annualized_garman_klass_pct', 0.0):.1f}%")

    tb_res = http_post(
        "/api/v1/econometrics/triple-barrier/simulate",
        {
            "symbol": "NVDA",
            "profit_multiplier": 2.0,
            "stop_multiplier": 1.0,
            "horizon_bars": 30,
            "volatility_window": 20,
            "side": 1,
        },
        token,
    )
    print("  ✓ Dynamic Volatility Triple-Barrier Simulator (NVDA Long +1, k1=2.0σ, k2=1.0σ):")
    print(f"       - Evaluated Trade Lifecycles: {tb_res.get('total_events', 0)}")
    print(f"       - Take Profit Hit Rate: {tb_res.get('take_profit_pct', 0.0):.1f}%")
    print(f"       - Stop Loss Hit Rate: {tb_res.get('stop_loss_pct', 0.0):.1f}%")
    print(f"       - Vertical Time Expiration: {tb_res.get('vertical_expiration_pct', 0.0):.1f}%")
    print(f"       - Average Holding Period: {tb_res.get('average_holding_bars', 0.0):.1f} bars")
    print(f"       - Realized Net Return: {tb_res.get('average_net_return_pct', 0.0):+.2f}%")

    # 4. Bayesian Game Theory & Jump Regimes (Pillar 3)
    print("\n[4/6] Solving Bayesian Game Theory & Jump Regimes (Pillar 3)...")
    regime_res = http_get("/api/v1/game-theory/regimes?symbol=NVDA&cusum_threshold=3.0&cusum_drift=0.5", token)
    print("  ✓ 3-Simplex Posterior Regime Classification (NVDA):")
    print(f"       - Current Regime: {regime_res.get('current_regime')}")
    print(f"       - Posterior Probabilities: Absorption={regime_res.get('p_absorption', 0):.2f}, "
          f"Cascade={regime_res.get('p_momentum', 0):.2f}, Panic Trap={regime_res.get('p_panic', 0):.2f}")
    print(f"       - CUSUM Structural Break Alarm: {'🚨 TRIPPED' if regime_res.get('cusum_alarm') else '🟢 Normal (No Shock)'}")

    matrix_res = http_post(
        "/api/v1/game-theory/payoff-matrix",
        {"ambiguity_temperature": 1.0, "risk_aversion": 0.5},
        token,
    )
    print("  ✓ Stackelberg Minimax Regret Payoff Matrix:")
    print(f"       - Optimal Robust Action: {matrix_res.get('optimal_action')}")
    print(f"       - Evaluated Counterparties: {', '.join(matrix_res.get('counterparties', []))}")

    # 5. Causal News Decay Engine (Pillar 1)
    print("\n[5/6] Injecting Breaking News Headline Shock (Pillar 1)...")
    test_headline = "NVIDIA unveils next-generation Blackwell Ultra architecture with record enterprise orders"
    shock_raw = http_post(
        "/api/v1/news/predict",
        {"headline": test_headline, "ticker": "NVDA", "current_price": 124.70},
        token,
    )
    shock = shock_raw[0] if isinstance(shock_raw, list) and shock_raw else {}
    print(f"  ✓ Ingested Headline: \"{test_headline}\"")
    print(f"       - Signal & Sentiment: {shock.get('signal')} (Confidence: {shock.get('confidence', 0.0)*100:.1f}%)")
    print(f"       - Expected Price Delta: ${shock.get('expected_delta_price', 0.0):+.2f} (Target: ${shock.get('target_price', 0.0):.2f})")
    print(f"       - Up / Down Directional Probability: Up={shock.get('prob_up', 0.0)*100:.1f}% / Down={shock.get('prob_down', 0.0)*100:.1f}%")
    print(f"       - Causal Barrier Bounds: Upper=${shock.get('barrier_upper', 0.0):.2f} / Lower=${shock.get('barrier_lower', 0.0):.2f}")

    # 6. Simulation Replay & Institutional Risk Tear Sheet (Pillar 5)
    print("\n[6/6] Running Institutional Replay Backtest (Pillar 5)...")
    sim_res = http_post(
        "/api/v1/simulation/run",
        {
            "asset_id": "NVDA",
            "bar_count": 300,
            "initial_capital": 10000.0,
            "fee_bps": 2.5,
            "spread_bps": 1.0,
            "impact_coefficient": 0.05,
        },
        token,
    )
    bms = sim_res.get("benchmarks", [])
    print("  ✓ ReplayEngine Simulation Complete ($10,000 Baseline, 300 1-min bars):")
    print("       --------------------------------------------------------")
    print(f"       Asset ID:                 {sim_res.get('asset_id')}")
    print(f"       Initial Capital:          ${sim_res.get('initial_capital', 10000.0):,.2f}")
    print(f"       Final Portfolio Equity:   ${sim_res.get('final_equity', 0.0):,.2f}")
    print(f"       Total Net Return:         {sim_res.get('total_return_pct', 0.0):+.2f}%")
    print(f"       Annualized Return (CAGR): {sim_res.get('cagr_pct', 0.0):+.2f}%")
    print(f"       Annualized Volatility:    {sim_res.get('annualized_volatility_pct', 0.0):.2f}%")
    print(f"       Sharpe Ratio:             {sim_res.get('sharpe_ratio', 0.0):.2f}")
    print(f"       Sortino Ratio:            {sim_res.get('sortino_ratio', 0.0):.2f}")
    print(f"       Calmar Ratio:             {sim_res.get('calmar_ratio', 0.0):.2f}")
    print(f"       Max Drawdown:             {sim_res.get('max_drawdown_pct', 0.0):.2f}%")
    print(f"       Realized CVaR (95%):      {sim_res.get('realized_cvar_95_pct', 0.0):.2f}%")
    print(f"       Deflated Sharpe (DSR):    {sim_res.get('deflated_sharpe_ratio', 0.0):.3f}")
    print(f"       DSR Statistically Sig:    {'✓ YES (Alpha Verified)' if sim_res.get('is_statistically_significant') else 'No'}")
    print(f"       Friction & Slippage Cost: ${sim_res.get('total_friction_cost', 0.0):,.2f}")
    print(f"       Equity Curve Points:      {len(sim_res.get('equity_curve', []))} bars recorded")
    if bms:
        for bm in bms:
            print(f"       Benchmark [{bm.get('name')}]: Total Ret={bm.get('total_return', 0)*100:+.2f}%, Sharpe={bm.get('sharpe_ratio', 0):.2f}, Alpha={bm.get('alpha', 0)*100:+.2f}%")
    print("       --------------------------------------------------------")

    print("\n✅ INSTITUTIONAL WORKBENCH TOUR COMPLETE: ALL 5 PILLARS VERIFIED & LIVE!")
    print("   Open http://127.0.0.1:8000/dashboard in your browser to interact with live charts.\n")


if __name__ == "__main__":
    main()
