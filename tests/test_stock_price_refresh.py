"""Regression test for overlapping daily stock-price updates."""

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def test_stock_pipeline_refreshes_existing_date(monkeypatch, tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "src" / "pipeline" / "08_incremental_stock_prices.py"
    )

    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location(
        "incremental_stock_prices_test", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    old = pd.DataFrame([
        {"date": "2026-11-13", "ticker": "AAPL", "open": 100.0,
         "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1000},
        {"date": "2026-11-16", "ticker": "AAPL", "open": 103.0,
         "high": 104.0, "low": 101.0, "close": 103.0, "volume": 500},
    ])
    old["date"] = pd.to_datetime(old["date"])

    incoming = pd.DataFrame([
        {"date": "2026-11-16", "ticker": "AAPL", "open": 103.0,
         "high": 108.0, "low": 101.0, "close": 107.0, "volume": 2000},
    ])
    incoming["date"] = pd.to_datetime(incoming["date"])

    stock_file = tmp_path / "stocks.parquet"
    old.to_parquet(stock_file, index=False)

    monkeypatch.setattr(module, "STOCK_FILE", stock_file)
    monkeypatch.setattr(
        module, "fetch_incremental",
        lambda existing, overlap_days=7: incoming.copy(),
    )
    monkeypatch.setattr(
        sys, "argv", ["08_incremental_stock_prices.py", "--apply"]
    )

    module.main()

    result = pd.read_parquet(stock_file)

    assert len(result) == 2
    assert not result.duplicated(["ticker", "date"]).any()

    latest = result.loc[
        result["date"] == pd.Timestamp("2026-11-16")
    ].iloc[0]

    assert latest["close"] == 107.0
    assert latest["volume"] == 2000

    historical = result.loc[
        result["date"] == pd.Timestamp("2026-11-13")
    ].iloc[0]

    assert historical["close"] == 102.0
