"""Safety tests for Alpaca paper-order execution."""

from unittest.mock import MagicMock

import pytest

from src.broker.paper_execution import (
    PaperOrderExecutionDisabled,
    submit_paper_orders,
)


@pytest.mark.parametrize("confirmed", [False, True])
def test_order_submission_disabled(confirmed):
    client = MagicMock()

    with pytest.raises(PaperOrderExecutionDisabled, match="disabled"):
        submit_paper_orders(
            client,
            [{"ticker": "AAPL", "quantity": 1}],
            confirmed=confirmed,
        )

    client.submit_order.assert_not_called()

from src.broker.paper_execution import validate_paper_orders


def test_valid_paper_order():
    orders = [
        {"ticker": "AAPL", "quantity": 2, "reference_price": 250.0}
    ]
    assert validate_paper_orders(orders, 10000) == 500.0


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True])
def test_invalid_quantity_rejected(quantity):
    orders = [
        {"ticker": "AAPL", "quantity": quantity, "reference_price": 250.0}
    ]
    with pytest.raises(ValueError, match="Quantity"):
        validate_paper_orders(orders, 10000)


def test_duplicate_ticker_rejected():
    orders = [
        {"ticker": "AAPL", "quantity": 1, "reference_price": 250.0},
        {"ticker": "AAPL", "quantity": 1, "reference_price": 250.0},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_paper_orders(orders, 10000)


def test_excessive_allocation_rejected():
    orders = [
        {"ticker": "AAPL", "quantity": 5, "reference_price": 250.0}
    ]
    with pytest.raises(ValueError, match="allocation"):
        validate_paper_orders(orders, 10000)


@pytest.mark.parametrize("price", [0, -10, float("nan"), float("inf")])
def test_invalid_reference_price_rejected(price):
    orders = [
        {"ticker": "AAPL", "quantity": 1, "reference_price": price}
    ]
    with pytest.raises(ValueError, match="Reference price"):
        validate_paper_orders(orders, 10000)

@pytest.mark.parametrize("cash", [None, "invalid", float("nan"), float("inf"), -100])
def test_invalid_cash_rejected(cash):
    orders = [{"ticker": "AAPL", "quantity": 1, "reference_price": 250.0}]
    with pytest.raises(ValueError, match="Available cash"):
        validate_paper_orders(orders, cash)


@pytest.mark.parametrize("quantity", ["2", None, 2.0])
def test_noninteger_quantity_rejected(quantity):
    orders = [{"ticker": "AAPL", "quantity": quantity, "reference_price": 250.0}]
    with pytest.raises(ValueError, match="Quantity"):
        validate_paper_orders(orders, 10000)


@pytest.mark.parametrize("price", [None, "invalid"])
def test_malformed_reference_price_rejected(price):
    orders = [{"ticker": "AAPL", "quantity": 1, "reference_price": price}]
    with pytest.raises(ValueError, match="Reference price"):
        validate_paper_orders(orders, 10000)
