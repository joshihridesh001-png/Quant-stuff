"""CLI Historical Market Data Ingestion Runner & Columnar Verification Rig.

Functional Purpose:
    Downloads real multi-year historical bar batches and corporate actions from Yahoo Finance,
    Polygon.io, or Alpaca Market Data v2. Persists observations into embedded DuckDB columnar storage
    (data/market_data.duckdb) with idempotent upsert semantics. Runs ultra-low latency verification
    benchmarks (< 5ms read SLA) and displays a quantitative summary report.

Explicit Dependency Tracking:
    - quant.data.historical: HistoricalDataHub, YahooFinanceHistoricalProvider, PolygonHistoricalProvider, AlpacaHistoricalProvider.
    - quant.infrastructure.database.duckdb_session: DuckDBManager.
    - quant.infrastructure.repositories.duckdb_historical_repository: DuckDBHistoricalRepository, AlignedReturnsResult.
    - quant.domain.historical: HistoricalBarBatch, HistoricalDataError.
    - quant.domain.models: Resolution.

Structural Relationship:
    Standalone research CLI tool in scripts/ invoked by operators or automated schedulers to hydrate
    the historical lake for offline backtesting and cross-asset signal modeling.

Defensive Invariants:
    - Validates date boundaries (start <= end) and strictly finite prices.
    - Reports exact diagnostic error codes (ERR-DATA-*) upon ingestion failure.
    - Benchmarks sub-5ms DuckDB read latency to verify zero-copy PyArrow efficiency.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np

from quant.data.institutional_providers import (
    AlpacaHistoricalProvider,
    HistoricalDataHub,
    PolygonHistoricalProvider,
)
from quant.data.yahoo_provider import YahooFinanceHistoricalProvider
from quant.domain.historical import HistoricalBarBatch, HistoricalDataError
from quant.domain.models import Resolution
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.repositories.duckdb_historical_repository import (
    DuckDBHistoricalRepository,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_historical")

RESOLUTION_MAP: Final[dict[str, Resolution]] = {
    "1d": Resolution.ONE_DAY,
    "1h": Resolution.ONE_HOUR,
    "15m": Resolution.FIFTEEN_MINUTES,
    "5m": Resolution.FIVE_MINUTES,
    "1m": Resolution.ONE_MINUTE,
}


def parse_date(date_str: str) -> datetime:
    """Parse ISO date YYYY-MM-DD or full ISO datetime into timezone-aware UTC datetime."""
    try:
        if len(date_str) == 10:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            dt = datetime.fromisoformat(date_str)
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    except Exception as e:
        raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD: {e}") from e


def run_ingestion(
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    resolution: Resolution,
    provider_name: str,
    db_path: str,
) -> int:
    """Execute historical data ingestion and latency verification pipeline."""
    print("=" * 80)
    print("  QUANT ENGINE: HISTORICAL DATA LAKE INGESTION & COLUMNAR STORAGE RIG")
    print("=" * 80)
    print(f"  Target Universe : {', '.join(symbols)}")
    print(f"  Time Interval   : {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")
    print(f"  Bar Resolution  : {resolution.value}")
    print(f"  Primary Provider: {provider_name.upper()}")
    print(f"  Storage Target  : {db_path}")
    print("=" * 80)

    # Initialize Storage Repository
    manager = DuckDBManager(database_path=db_path)
    repository = DuckDBHistoricalRepository(manager=manager)

    # Select Provider
    if provider_name.lower() == "yahoo":
        provider = YahooFinanceHistoricalProvider()
    elif provider_name.lower() == "polygon":
        provider = PolygonHistoricalProvider()
    elif provider_name.lower() == "alpaca":
        provider = AlpacaHistoricalProvider()
    else:
        # Default zero-failure fallback cascade
        provider = HistoricalDataHub()

    total_bars_ingested = 0
    results_summary: list[dict[str, str | int | float]] = []

    try:
        for sym in symbols:
            sym_clean = sym.strip().upper()
            if not sym_clean:
                continue

            print(
                f"\n[+] Fetching '{sym_clean}' from {provider_name.upper()}...", end="", flush=True
            )
            t_start_fetch = time.perf_counter()

            try:
                batch: HistoricalBarBatch = provider.fetch_historical_bars(
                    symbol=sym_clean,
                    start=start_dt,
                    end=end_dt,
                    resolution=resolution,
                )
                t_fetch_ms = (time.perf_counter() - t_start_fetch) * 1000.0

                # Ingest into DuckDB
                t_start_write = time.perf_counter()
                saved_count = repository.save_batch_sync(batch)
                t_write_ms = (time.perf_counter() - t_start_write) * 1000.0

                # Benchmark read latency SLA (< 5ms)
                start_ns = batch.start_timestamp
                end_ns = batch.end_timestamp
                t_start_read = time.perf_counter()
                _ = repository.get_bars_range_sync(sym_clean, start_ns, end_ns, resolution)
                t_read_ms = (time.perf_counter() - t_start_read) * 1000.0

                first_dt = datetime.fromtimestamp(batch.start_timestamp / 1e9, tz=UTC).strftime(
                    "%Y-%m-%d"
                )
                last_dt = datetime.fromtimestamp(batch.end_timestamp / 1e9, tz=UTC).strftime(
                    "%Y-%m-%d"
                )

                total_bars_ingested += saved_count
                print(f" OK ({saved_count} bars, {len(batch.corporate_actions)} splits/divs)")

                results_summary.append(
                    {
                        "symbol": sym_clean,
                        "status": "SUCCESS",
                        "bars": saved_count,
                        "first_bar": first_dt,
                        "last_bar": last_dt,
                        "fetch_ms": f"{t_fetch_ms:.1f}ms",
                        "write_ms": f"{t_write_ms:.2f}ms",
                        "read_ms": f"{t_read_ms:.2f}ms",
                        "sla_pass": "PASS" if t_read_ms < 5.0 else "WARN",
                    }
                )

            except HistoricalDataError as h_err:
                print(f" FAILED [{h_err.code}]: {h_err.message}")
                results_summary.append(
                    {
                        "symbol": sym_clean,
                        "status": f"FAILED ({h_err.code})",
                        "bars": 0,
                        "first_bar": "-",
                        "last_bar": "-",
                        "fetch_ms": "-",
                        "write_ms": "-",
                        "read_ms": "-",
                        "sla_pass": "FAIL",
                    }
                )
            except Exception as e:
                print(f" ERROR: {e}")
                results_summary.append(
                    {
                        "symbol": sym_clean,
                        "status": "ERROR",
                        "bars": 0,
                        "first_bar": "-",
                        "last_bar": "-",
                        "fetch_ms": "-",
                        "write_ms": "-",
                        "read_ms": "-",
                        "sla_pass": "FAIL",
                    }
                )

        # Print Summary Table
        print("\n" + "=" * 90)
        print("  HISTORICAL INGESTION SUMMARY REPORT")
        print("=" * 90)
        print(
            f"  {'Symbol':<8} {'Status':<16} {'Bars':<8} {'Earliest':<12} {'Latest':<12} "
            f"{'Fetch':<10} {'Write':<10} {'Read':<10} {'SLA < 5ms'}"
        )
        print("  " + "-" * 88)
        for r in results_summary:
            print(
                f"  {r['symbol']:<8} {r['status']:<16} {str(r['bars']):<8} {str(r['first_bar']):<12} "
                f"{str(r['last_bar']):<12} {str(r['fetch_ms']):<10} {str(r['write_ms']):<10} "
                f"{str(r['read_ms']):<10} {r['sla_pass']}"
            )
        print("=" * 90)

        # Multi-Asset Aligned Matrix Verification
        active_symbols = [str(r["symbol"]) for r in results_summary if r["status"] == "SUCCESS"]
        if len(active_symbols) >= 2:
            print("\n[+] Benchmarking Vectorized Aligned Returns Matrix Extraction...")
            start_ns = int(start_dt.timestamp() * 1e9)
            end_ns = int(end_dt.timestamp() * 1e9)
            t_matrix_start = time.perf_counter()

            aligned_res = repository.get_aligned_returns_matrix_sync(
                symbols=active_symbols,
                start_time=start_ns,
                end_time=end_ns,
                resolution=resolution,
                price_field="adj_close",
                return_type="simple",
            )
            t_matrix_ms = (time.perf_counter() - t_matrix_start) * 1000.0

            n_periods, n_assets = aligned_res.matrix.shape
            cov_matrix = np.cov(aligned_res.matrix, rowvar=False)
            corr_matrix = np.corrcoef(aligned_res.matrix, rowvar=False)
            ann_vols = np.sqrt(np.maximum(0.0, np.diag(cov_matrix)) * 252.0) * 100.0

            print(f"  Aligned Matrix Dimensions : {n_periods} periods x {n_assets} assets")
            print(f"  Matrix Extraction Latency : {t_matrix_ms:.2f} ms")
            print("  Asset Annualized Volatility:")
            for idx, s in enumerate(aligned_res.symbols):
                print(f"    {s:<6}: {ann_vols[idx]:>6.2f}%")
            print("\n  Cross-Asset Correlation Matrix:")
            header_str = "        " + "  ".join(f"{s:>7}" for s in aligned_res.symbols)
            print(header_str)
            for i, s1 in enumerate(aligned_res.symbols):
                row_vals = "  ".join(
                    f"{corr_matrix[i, j]:>7.3f}" if math.isfinite(corr_matrix[i, j]) else "    NaN"
                    for j in range(n_assets)
                )
                print(f"  {s1:>6}: {row_vals}")

        print("\n[V] Ingestion run completed successfully. Total bars stored:", total_bars_ingested)
        return 0
    finally:
        repository.close()


def main() -> None:
    """CLI entrypoint parsing options and executing ingestion."""
    parser = argparse.ArgumentParser(
        description="Institutional Historical Market Data Ingestion Runner"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,AAPL,NVDA,MSFT",
        help="Comma-separated ticker symbols to ingest (e.g. SPY,QQQ,AAPL,NVDA,MSFT)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now(UTC) - timedelta(days=365 * 3)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format (default: 3 years ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1d",
        choices=list(RESOLUTION_MAP.keys()),
        help="Bar resolution: 1d, 1h, 15m, 5m, 1m (default: 1d)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="hub",
        choices=["hub", "yahoo", "polygon", "alpaca"],
        help="Data provider to query (default: hub with zero-failure fallback cascade)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/market_data.duckdb",
        help="Target embedded DuckDB path (default: data/market_data.duckdb)",
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("Error: No valid symbols specified.", file=sys.stderr)
        sys.exit(1)

    try:
        start_dt = parse_date(args.start)
        end_dt = parse_date(args.end)
    except ValueError as val_err:
        print(f"Error: {val_err}", file=sys.stderr)
        sys.exit(1)

    if start_dt > end_dt:
        print(
            f"Error: Start date ({args.start}) cannot be after end date ({args.end}).",
            file=sys.stderr,
        )
        sys.exit(1)

    resolution = RESOLUTION_MAP[args.resolution]

    exit_code = run_ingestion(
        symbols=symbols,
        start_dt=start_dt,
        end_dt=end_dt,
        resolution=resolution,
        provider_name=args.provider,
        db_path=args.db_path,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
