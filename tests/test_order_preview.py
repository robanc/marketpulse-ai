"""Automated tests for non-executable Alpaca order previews."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from src.broker.order_preview import (
    validate_signals,
    preview_cash_orders,
    exclude_existing_exposure,
)


@pytest.fixture
def predictions():
    return pd.DataFrame({
        "date": ["2026-11-13", "2026-11-13"],
        "ticker": ["AAPL", "MSFT"],
        "predicted_probability": [0.65, 0.55],
        "signal": [1, 0],
        "prediction_horizon_days": [10, 10],
    })


@pytest.fixture
def prices():
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "close": [250.0, 500.0],
    })


def test_valid_predictions(predictions):
    result = validate_signals(
        predictions, as_of_date=date(2026, 11, 16)
    )
    assert result == date(2026, 11, 13)


def test_intraday_predictions_rejected(predictions):
    with pytest.raises(ValueError, match="incomplete"):
        validate_signals(
            predictions.assign(date="2026-11-16"),
            as_of_date=date(2026, 11, 16),
            market_open=True,
        )


def test_future_predictions_rejected(predictions):
    with pytest.raises(ValueError, match="stale or future"):
        validate_signals(
            predictions.assign(date="2026-11-17"),
            as_of_date=date(2026, 11, 16),
        )


def test_stale_predictions_rejected(predictions):
    with pytest.raises(ValueError, match="stale or future"):
        validate_signals(
            predictions.assign(date="2026-11-10"),
            as_of_date=date(2026, 11, 16),
        )


def test_duplicate_predictions_rejected(predictions):
    duplicated = pd.concat([predictions, predictions], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_signals(
            duplicated, as_of_date=date(2026, 11, 16)
        )


def test_invalid_probability_rejected(predictions):
    with pytest.raises(ValueError, match="probabilities"):
        validate_signals(
            predictions.assign(predicted_probability=1.5),
            as_of_date=date(2026, 11, 16),
        )


def test_invalid_signal_rejected(predictions):
    with pytest.raises(ValueError, match="Signals"):
        validate_signals(
            predictions.assign(signal=2),
            as_of_date=date(2026, 11, 16),
        )


def test_cash_allocation_limit(predictions, prices):
    preview = preview_cash_orders(
        predictions, prices, cash=100000, allocation_fraction=0.10
    )
    assert len(preview) == 1
    assert preview.iloc[0]["ticker"] == "AAPL"
    assert preview.iloc[0]["quantity"] == 40
    assert preview["estimated_cost"].sum() <= 10000


def test_excessive_allocation_rejected(predictions, prices):
    with pytest.raises(ValueError, match="Allocation"):
        preview_cash_orders(
            predictions, prices, cash=100000, allocation_fraction=0.50
        )


def test_missing_price_rejected(predictions, prices):
    with pytest.raises(ValueError, match="Missing reference"):
        preview_cash_orders(
            predictions, prices.iloc[0:0], cash=100000
        )


def test_existing_exposure_excluded():
    preview = pd.DataFrame({
        "ticker": ["NVDA", "COST", "BAC"],
        "quantity": [5, 1, 10],
        "reference_price": [160.0, 1100.0, 50.0],
        "estimated_cost": [800.0, 1100.0, 500.0],
    })
    positions = [SimpleNamespace(symbol="NVDA")]
    orders = [SimpleNamespace(symbol="COST")]

    safe, blocked = exclude_existing_exposure(
        preview, positions, orders
    )

    assert set(blocked) == {"NVDA", "COST"}
    assert safe["ticker"].tolist() == ["BAC"]


def test_duplicate_preview_rejected():
    preview = pd.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "quantity": [1, 1],
    })

    with pytest.raises(ValueError, match="Duplicate"):
        exclude_existing_exposure(preview, [], [])
