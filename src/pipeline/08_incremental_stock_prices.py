"""Incremental Yahoo Finance stock collection with safe preview mode."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import yfinance as yf

from incremental_store import merge_incremental


ROOT = Path(__file__).resolve().parents[2]
STOCK_FILE = ROOT / "data" / "stock_prices.parquet"


def fetch_incremental(existing, overlap_days=7):
    existing = existing.copy()
    existing["date"] = pd.to_datetime(existing["date"])

    tickers = sorted(existing["ticker"].unique())
    latest = existing["date"].max()
    start = (latest - pd.Timedelta(days=overlap_days)).date().isoformat()
    end = (
        pd.Timestamp.now(tz="America/New_York")
        + pd.Timedelta(days=1)
    ).date().isoformat()

    print("Tickers:", len(tickers))
    print("Last stored date:", latest.date())
    print("Download start:", start)
    print("Download end (exclusive):", end)

    if pd.Timestamp(start) >= pd.Timestamp(end):
        print("No eligible download window.")
        return existing.iloc[0:0].copy()

    downloaded = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    parts = []
    required = ["Open", "High", "Low", "Close", "Volume"]

    for ticker in tickers:
        if ticker not in downloaded.columns.get_level_values(0):
            raise RuntimeError(f"Missing ticker in download: {ticker}")

        frame = downloaded[ticker].copy()

        if not set(required).issubset(frame.columns):
            raise RuntimeError(f"Missing OHLCV columns for {ticker}")

        frame = frame[required].dropna(subset=["Close"])
        frame = frame.reset_index()
        frame["ticker"] = ticker

        frame = frame.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        parts.append(
            frame[["date", "ticker", "open", "high",
                   "low", "close", "volume"]]
        )

    incoming = pd.concat(parts, ignore_index=True)
    incoming["date"] = pd.to_datetime(incoming["date"]).dt.tz_localize(None)

    return incoming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Write validated merged data; default is preview.")
    parser.add_argument("--overlap-days", type=int, default=7)
    args = parser.parse_args()

    if args.overlap_days < 1:
        raise ValueError("overlap-days must be at least 1.")

    existing = pd.read_parquet(STOCK_FILE)
    incoming = fetch_incremental(existing, args.overlap_days)

    # Preserve historical records; append only previously unseen keys.
    existing_keys = pd.MultiIndex.from_frame(
        existing[["ticker", "date"]].assign(
            date=pd.to_datetime(existing["date"])
        )
    )
    incoming_keys = pd.MultiIndex.from_frame(
        incoming[["ticker", "date"]].assign(
            date=pd.to_datetime(incoming["date"])
        )
    )

    new_only = incoming.loc[~incoming_keys.isin(existing_keys)].copy()

    merged = merge_incremental(
        existing, new_only, ["ticker", "date"]
    )

    expected = set(existing["ticker"].unique())
    if set(merged["ticker"].unique()) != expected:
        raise RuntimeError("Ticker universe changed unexpectedly.")

    if merged["date"].max() < pd.to_datetime(existing["date"]).max():
        raise RuntimeError("Latest available date moved backwards.")

    print("Existing rows:", len(existing))
    print("Downloaded rows:", len(incoming))
    print("Merged rows:", len(merged))
    print("New net rows:", len(merged) - len(existing))
    print("Latest merged date:", merged["date"].max())

    if not args.apply:
        print("PREVIEW ONLY - existing Parquet file unchanged.")
        return

    fd, temp_name = tempfile.mkstemp(
        suffix=".parquet", dir=STOCK_FILE.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        merged.to_parquet(temp_path, index=False)
        verified = pd.read_parquet(temp_path)
        if len(verified) != len(merged):
            raise RuntimeError("Written file verification failed.")
        os.replace(temp_path, STOCK_FILE)
    finally:
        temp_path.unlink(missing_ok=True)

    print("UPDATED:", STOCK_FILE)


if __name__ == "__main__":
    main()
