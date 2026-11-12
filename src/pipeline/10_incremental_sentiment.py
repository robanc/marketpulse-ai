"""Incrementally append Alternative.me sentiment aligned to stock trading dates."""

import argparse
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests


DATA_DIR = Path("data")
STOCK_FILE = DATA_DIR / "stock_prices.parquet"
SENTIMENT_FILE = DATA_DIR / "market_sentiment.parquet"
API_URL = "https://api.alternative.me/fng/"


def fetch_sentiment():
    response = requests.get(
        API_URL,
        params={"limit": 45, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    observations = pd.DataFrame(payload["data"])

    if observations.empty:
        raise ValueError("Sentiment API returned no observations")

    observations["source_date"] = pd.to_datetime(
        pd.to_numeric(observations["timestamp"]),
        unit="s",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    observations["fear_greed_index"] = pd.to_numeric(
        observations["value"], errors="raise"
    ).astype(int)

    observations["value_classification"] = (
        observations["value_classification"].astype(str)
    )

    observations["available_date"] = (
        observations["source_date"] + pd.Timedelta(days=1)
    ).astype("datetime64[ns]")

    observations = observations.sort_values("available_date")

    if observations["available_date"].duplicated().any():
        raise ValueError("Duplicate API availability dates")

    return observations[
        ["available_date", "fear_greed_index", "value_classification"]
    ]


def build_incremental(stocks, existing, observations):
    trading_dates = pd.DataFrame({
        "date": pd.to_datetime(stocks["date"]).astype("datetime64[ns]")
    }).drop_duplicates().sort_values("date")

    existing = existing.copy()
    existing["date"] = pd.to_datetime(
        existing["date"]
    ).astype("datetime64[ns]")

    if existing["date"].duplicated().any():
        raise ValueError("Duplicate existing sentiment dates")

    latest_existing = existing["date"].max()

    new_dates = trading_dates.loc[
        trading_dates["date"] > latest_existing
    ].copy()

    if new_dates.empty:
        return existing, existing.iloc[0:0].copy()

    earliest_available = observations["available_date"].min()

    if earliest_available > latest_existing:
        raise ValueError(
            "API history does not overlap the existing dataset; "
            "cannot verify continuity"
        )

    aligned = pd.merge_asof(
        new_dates,
        observations.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    ).drop(columns="available_date")

    if aligned[["fear_greed_index", "value_classification"]].isna().any().any():
        raise ValueError("Missing aligned sentiment values")

    incoming = aligned[
        ["date", "fear_greed_index", "value_classification"]
    ].copy()

    incoming["fear_greed_index"] = (
        incoming["fear_greed_index"].astype(int)
    )

    combined = pd.concat(
        [existing, incoming], ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    if combined["date"].duplicated().any():
        raise ValueError("Duplicate merged sentiment dates")

    return combined, incoming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist new sentiment rows; default is preview",
    )
    args = parser.parse_args()

    stocks = pd.read_parquet(STOCK_FILE)
    existing = pd.read_parquet(SENTIMENT_FILE)
    observations = fetch_sentiment()

    combined, incoming = build_incremental(
        stocks, existing, observations
    )

    print("\nMARKETPULSE AI - INCREMENTAL SENTIMENT")
    print("=" * 70)
    print("Source: Alternative.me Crypto Fear & Greed")
    print("Availability lag: 1 calendar day")
    print("Existing rows:", len(existing))
    print("New rows:", len(incoming))
    print("Merged rows:", len(combined))
    print("Latest merged date:", combined["date"].max().date())

    if not incoming.empty:
        print("\nNew sentiment observations:")
        print(incoming.to_string(index=False))

    if args.apply and not incoming.empty:
        temp_file = SENTIMENT_FILE.with_name(
            SENTIMENT_FILE.stem + "." + uuid4().hex + ".tmp.parquet"
        )
        try:
            combined.to_parquet(temp_file, index=False)
            verified = pd.read_parquet(temp_file)

            if verified.shape != combined.shape:
                raise ValueError(
                    "Temporary Parquet row/column count mismatch"
                )

            if list(verified.columns) != list(combined.columns):
                raise ValueError(
                    "Temporary Parquet schema mismatch"
                )

            if verified.duplicated(['date']).any():
                raise ValueError(
                    "Duplicate keys in temporary Parquet file"
                )

            os.replace(temp_file, SENTIMENT_FILE)
        finally:
            if temp_file.exists():
                temp_file.unlink()
        print("\nUPDATED:", SENTIMENT_FILE.resolve())
    elif args.apply:
        print("\nNO UPDATE NEEDED")
    else:
        print("\nPREVIEW ONLY - Existing files unchanged")


if __name__ == "__main__":
    main()
