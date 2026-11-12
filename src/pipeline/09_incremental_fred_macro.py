"""Append-only incremental FRED macroeconomic updater."""

import argparse
import os
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STOCK_FILE = DATA / "stock_prices.parquet"
MACRO_FILE = DATA / "fred_macro.parquet"

SERIES = {
    "DGS10": ("treasury_10y", 1),
    "DGS2": ("treasury_2y", 1),
    "FEDFUNDS": ("fed_funds_rate", 7),
    "CPIAUCSL": ("cpi", 30),
    "UNRATE": ("unemployment_rate", 30),
}


def download_observations():
    observations = {}

    for series_id, (column, delay) in SERIES.items():
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

        response = requests.get(
            url,
            params={"id": series_id},
            timeout=30,
        )
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        if len(df.columns) != 2:
            raise ValueError(f"Unexpected FRED schema: {series_id}")

        df.columns = ["date", column]
        df["date"] = pd.to_datetime(df["date"])
        df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=[column]).copy()

        df["available_date"] = (
            df["date"] + pd.Timedelta(days=delay)
        ).astype("datetime64[ns]")

        df = df.sort_values("available_date")

        if df.empty:
            raise ValueError(f"No valid FRED data: {series_id}")

        observations[column] = df[["available_date", column]]

        print(
            f"Downloaded {series_id}: "
            f"{len(df)} valid observations"
        )

    return observations


def build_incremental(stocks, macro, observations):
    stock_dates = pd.DatetimeIndex(
        stocks["date"].drop_duplicates().sort_values()
    )

    latest_stored = macro["date"].max()

    missing_dates = stock_dates[stock_dates > latest_stored]

    if len(missing_dates) == 0:
        return macro.iloc[0:0].copy()

    aligned = pd.DataFrame({"date": missing_dates})

    for column, _ in SERIES.values():
        aligned = pd.merge_asof(
            aligned.sort_values("date"),
            observations[column],
            left_on="date",
            right_on="available_date",
            direction="backward",
        ).drop(columns="available_date")

    aligned = aligned.dropna().reset_index(drop=True)

    return aligned[macro.columns.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist validated incremental records",
    )
    parser.add_argument(
        "--hold-unverified",
        action="store_true",
        help="Carry forward stored CPI and unemployment values",
    )
    args = parser.parse_args()

    stocks = pd.read_parquet(STOCK_FILE)
    macro = pd.read_parquet(MACRO_FILE)

    stocks["date"] = pd.to_datetime(
        stocks["date"]
    ).astype("datetime64[ns]")

    macro["date"] = pd.to_datetime(
        macro["date"]
    ).astype("datetime64[ns]")

    if macro["date"].duplicated().any():
        raise ValueError("Duplicate dates in existing FRED data")

    observations = download_observations()
    incoming = build_incremental(stocks, macro, observations)

    if args.hold_unverified and not incoming.empty:
        for column in ["cpi", "unemployment_rate"]:
            incoming[column] = macro.sort_values("date")[
                column
            ].iloc[-1]

        print(
            "SAFEGUARD: CPI and unemployment held at "
            "last stored values; publication dates unverified."
        )

    if args.apply and not args.hold_unverified:
        raise ValueError(
            "Refusing default apply: CPI/unemployment "
            "publication dates have not been verified. "
            "Use --hold-unverified --apply."
        )

    merged = pd.concat(
        [macro, incoming],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)

    if merged["date"].duplicated().any():
        raise ValueError("Duplicate dates after incremental merge")

    if merged.isna().any().any():
        raise ValueError("Missing values in merged macro data")

    if len(merged) != len(macro) + len(incoming):
        raise ValueError("Unexpected merged row count")

    print("\nMARKETPULSE AI - INCREMENTAL FRED")
    print("=" * 65)
    print("Existing rows:", len(macro))
    print("New rows:", len(incoming))
    print("Merged rows:", len(merged))
    print("Latest merged date:", merged["date"].max().date())

    if incoming.empty:
        print("No new complete macro records.")
        return

    print("\nNew macro observations:")
    print(incoming.to_string(index=False))

    if not args.apply:
        print("\nPREVIEW ONLY - Existing files unchanged")
        return

    fd, temp_name = tempfile.mkstemp(
        suffix=".parquet",
        prefix="fred_incremental_",
        dir=DATA,
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        merged.to_parquet(temp_path, index=False)

        check = pd.read_parquet(temp_path)

        if len(check) != len(merged):
            raise ValueError("Saved Parquet verification failed")

        os.replace(temp_path, MACRO_FILE)
        print("\nUPDATED:", MACRO_FILE)

    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
