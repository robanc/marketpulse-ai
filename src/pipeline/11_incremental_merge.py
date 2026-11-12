"""Append newly available stock, macro, and sentiment records."""

import argparse
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd


DATA_DIR = Path("data")
STOCK_FILE = DATA_DIR / "stock_prices.parquet"
MACRO_FILE = DATA_DIR / "fred_macro.parquet"
SENTIMENT_FILE = DATA_DIR / "market_sentiment.parquet"
MERGED_FILE = DATA_DIR / "marketpulse_merged.parquet"


def normalize_dates(df):
    result = df.copy()
    result["date"] = pd.to_datetime(
        result["date"]
    ).astype("datetime64[ns]")
    return result


def build_incremental(stocks, macro, sentiment, existing):
    stocks = normalize_dates(stocks)
    macro = normalize_dates(macro)
    sentiment = normalize_dates(sentiment)
    existing = normalize_dates(existing)

    if stocks.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate stock date/ticker keys")

    for name, frame in [
        ("macro", macro),
        ("sentiment", sentiment),
    ]:
        if frame["date"].duplicated().any():
            raise ValueError(f"Duplicate {name} dates")

    if existing.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate existing merged keys")

    latest_existing = existing["date"].max()

    new_stocks = stocks.loc[
        stocks["date"] > latest_existing
    ].copy()

    if new_stocks.empty:
        return existing, existing.iloc[0:0].copy()

    expected_dates = set(new_stocks["date"])

    for name, frame in [
        ("macro", macro),
        ("sentiment", sentiment),
    ]:
        missing_dates = expected_dates - set(frame["date"])
        if missing_dates:
            raise ValueError(
                f"{name} missing dates: {sorted(missing_dates)}"
            )

    incoming = new_stocks.merge(
        macro,
        on="date",
        how="inner",
        validate="many_to_one",
    ).merge(
        sentiment,
        on="date",
        how="inner",
        validate="many_to_one",
    )

    incoming = incoming[existing.columns.tolist()]

    if len(incoming) != len(new_stocks):
        raise ValueError("Unexpected row loss during merge")

    if incoming.isna().any().any():
        raise ValueError("Missing values in incoming records")

    combined = pd.concat(
        [existing, incoming],
        ignore_index=True,
    ).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    if combined.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate merged date/ticker keys")

    if combined.isna().any().any():
        raise ValueError("Missing values in merged dataset")

    return combined, incoming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Save appended records; default is preview",
    )
    args = parser.parse_args()

    stocks = pd.read_parquet(STOCK_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    sentiment = pd.read_parquet(SENTIMENT_FILE)
    existing = pd.read_parquet(MERGED_FILE)

    combined, incoming = build_incremental(
        stocks, macro, sentiment, existing
    )

    print("\nMARKETPULSE AI - INCREMENTAL MERGED DATA")
    print("=" * 70)
    print("Existing records:", len(existing))
    print("New records:", len(incoming))
    print("Merged records:", len(combined))
    print("Columns:", len(combined.columns))
    print("Unique tickers:", combined["ticker"].nunique())
    print("Latest date:", combined["date"].max().date())

    if not incoming.empty:
        print("\nNew records by trading date:")
        print(
            incoming.groupby("date")
            .size()
            .to_string()
        )

        print("\nSample new records:")
        print(
            incoming.sort_values(["date", "ticker"])
            .head(5)
            .to_string(index=False)
        )

    if args.apply and not incoming.empty:
        temp_file = MERGED_FILE.with_name(
            MERGED_FILE.stem + "." + uuid4().hex + ".tmp.parquet"
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

            if verified.duplicated(['ticker', 'date']).any():
                raise ValueError(
                    "Duplicate keys in temporary Parquet file"
                )

            os.replace(temp_file, MERGED_FILE)
        finally:
            if temp_file.exists():
                temp_file.unlink()
        print("\nUPDATED:", MERGED_FILE.resolve())
    elif args.apply:
        print("\nNO UPDATE NEEDED")
    else:
        print("\nPREVIEW ONLY - Existing files unchanged")


if __name__ == "__main__":
    main()
