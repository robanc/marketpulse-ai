"""Validate ML signals and prepare non-executable paper-order previews."""

from datetime import date
import math

import pandas as pd


def validate_signals(predictions, as_of_date=None, market_open=False, max_age_days=3):
    """Validate prediction data before creating any order preview."""
    today = as_of_date or date.today()
    required = {
        "date", "ticker", "predicted_probability",
        "signal", "prediction_horizon_days",
    }

    if not required.issubset(predictions.columns):
        raise ValueError("Prediction file is missing required columns")

    if predictions.empty or predictions[list(required)].isna().any().any():
        raise ValueError("Prediction file is empty or contains missing values")

    dates = pd.to_datetime(predictions["date"], errors="raise").dt.date
    if dates.nunique() != 1:
        raise ValueError("Predictions must share one trading date")

    prediction_date = dates.iloc[0]
    age = (today - prediction_date).days

    if not 0 <= age <= max_age_days:
        raise ValueError("Predictions are stale or future-dated")

    if market_open and prediction_date == today:
        raise ValueError("Today's market session is incomplete")

    if predictions["ticker"].duplicated().any():
        raise ValueError("Duplicate ticker predictions detected")

    if not predictions["ticker"].astype(str).str.fullmatch(r"[A-Z]{1,5}").all():
        raise ValueError("Invalid ticker symbols detected")

    probabilities = pd.to_numeric(
        predictions["predicted_probability"], errors="raise"
    )
    if not probabilities.between(0, 1).all():
        raise ValueError("Invalid predicted probabilities")

    if not predictions["signal"].isin([0, 1]).all():
        raise ValueError("Signals must be 0 or 1")

    if not (predictions["prediction_horizon_days"] == 10).all():
        raise ValueError("Unexpected prediction horizon")

    return prediction_date


def preview_cash_orders(predictions, prices, cash, allocation_fraction=0.10):
    """Create cash-limited previews; never submit brokerage orders."""
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError("Available cash must be positive and finite")

    if not 0 < allocation_fraction <= 0.10:
        raise ValueError("Allocation fraction must be between 0 and 10%")

    selected = predictions.loc[predictions["signal"] == 1].copy()

    if selected.empty:
        return pd.DataFrame(
            columns=["ticker", "quantity", "reference_price", "estimated_cost"]
        )

    if prices["ticker"].duplicated().any():
        raise ValueError("Duplicate price records detected")

    selected = selected.merge(
        prices[["ticker", "close"]],
        on="ticker",
        how="left",
        validate="one_to_one",
    )

    if selected["close"].isna().any():
        raise ValueError("Missing reference prices")

    if not selected["close"].apply(
        lambda value: math.isfinite(float(value)) and float(value) > 0
    ).all():
        raise ValueError("Invalid reference prices")

    budget = cash * allocation_fraction
    per_stock_budget = budget / len(selected)

    selected["quantity"] = (
        per_stock_budget // selected["close"]
    ).astype(int)

    selected["reference_price"] = selected["close"]
    selected["estimated_cost"] = (
        selected["quantity"] * selected["reference_price"]
    )

    return selected.loc[
        selected["quantity"] > 0,
        ["ticker", "quantity", "reference_price", "estimated_cost"],
    ].reset_index(drop=True)


def exclude_existing_exposure(preview, positions, open_orders):
    """Exclude tickers already held or associated with open orders.

    Read-only filtering. Does not submit or cancel brokerage orders.
    """
    if preview["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers in order preview")

    held = {position.symbol for position in positions}
    pending = {order.symbol for order in open_orders}
    excluded = held | pending

    safe_preview = preview.loc[
        ~preview["ticker"].isin(excluded)
    ].copy()

    blocked = preview.loc[
        preview["ticker"].isin(excluded), "ticker"
    ].tolist()

    return safe_preview.reset_index(drop=True), blocked
