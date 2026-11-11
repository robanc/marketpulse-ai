"""Generate latest-date MarketPulse AI research trading signals."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    config = json.loads(
        (MODELS_DIR / "marketpulse_trading_config.json")
        .read_text(encoding="utf-8-sig")
    )

    model = joblib.load(
        MODELS_DIR / "marketpulse_rf_market_only.joblib"
    )

    feature_columns = config["features"]
    threshold = float(config["signal_threshold"])

    if list(model.feature_names_in_) != feature_columns:
        raise ValueError("Model and configuration features do not match.")

    features = pd.read_parquet(
        DATA_DIR / "marketpulse_features.parquet"
    )

    required_columns = ["date", "ticker"] + feature_columns
    missing_columns = [
        col for col in required_columns
        if col not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    features["date"] = pd.to_datetime(features["date"])

    stocks = features.loc[
        features["ticker"] != "SPY"
    ].copy()

    if stocks.empty:
        raise ValueError("No stock observations available.")

    latest_date = stocks["date"].max()

    latest = stocks.loc[
        stocks["date"] == latest_date
    ].copy()

    if latest["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers on latest date.")

    X = latest[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        invalid = latest.loc[
            X.isna().any(axis=1), "ticker"
        ].tolist()

        raise ValueError(
            f"Invalid latest features for: {invalid}"
        )

    probabilities = model.predict_proba(X)[:, 1]

    output = latest[["date", "ticker"]].copy()
    output["predicted_probability"] = probabilities
    output["signal"] = (
        output["predicted_probability"] >= threshold
    ).astype(int)

    output["prediction_horizon_days"] = int(
        config["prediction_horizon_days"]
    )

    output = output.sort_values(
        ["signal", "predicted_probability"],
        ascending=[False, False]
    ).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "marketpulse_latest_predictions.csv"

    output.to_csv(output_path, index=False)

    print("MARKETPULSE AI - LATEST PREDICTIONS")
    print("=" * 65)
    print("Market data date:", latest_date.date())
    print("Stocks evaluated:", len(output))
    print("Signal threshold:", threshold)
    print("Selected signals:", int(output["signal"].sum()))
    print()
    print(output.to_string(index=False))
    print()
    print("Saved:", output_path)
    print("Latest inference: PASSED")


if __name__ == "__main__":
    main()
