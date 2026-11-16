"""Controlled Alpaca paper-order execution.

Order submission remains disabled until validation and tests are complete.
"""

import math
import re


class PaperOrderExecutionDisabled(RuntimeError):
    """Raised when paper-order submission is not enabled."""


def _positive_finite_number(value, field_name):
    """Convert a numeric input and reject invalid or nonfinite values."""
    try:
        if value is None or isinstance(value, (bool, str)):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"{field_name} must be positive and finite"
        ) from None

    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{field_name} must be positive and finite"
        )

    return number


def validate_paper_orders(orders, available_cash, max_allocation=0.10):
    """Validate proposed paper orders without submitting them."""
    cash = _positive_finite_number(available_cash, "Available cash")

    try:
        allocation = float(max_allocation)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Maximum allocation must be between 0 and 10%") from None

    if not math.isfinite(allocation) or not 0 < allocation <= 0.10:
        raise ValueError("Maximum allocation must be between 0 and 10%")

    if not isinstance(orders, (list, tuple)) or not orders:
        raise ValueError("No valid orders provided")

    seen = set()
    total_cost = 0.0

    for order in orders:
        if not isinstance(order, dict):
            raise ValueError("Each order must be a dictionary")

        ticker = order.get("ticker")
        quantity = order.get("quantity")
        price = order.get("reference_price")

        if not isinstance(ticker, str) or not re.fullmatch(r"[A-Z]{1,5}", ticker):
            raise ValueError("Invalid ticker symbol")

        if ticker in seen:
            raise ValueError("Duplicate ticker in orders")
        seen.add(ticker)

        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer")

        reference_price = _positive_finite_number(price, "Reference price")
        total_cost += quantity * reference_price

    if not math.isfinite(total_cost) or total_cost > cash * allocation:
        raise ValueError("Orders exceed maximum cash allocation")

    return total_cost


def submit_paper_orders(client, orders, *, confirmed=False):
    """Fail closed until paper-order execution safeguards are implemented."""
    raise PaperOrderExecutionDisabled(
        "Paper-order submission is disabled pending validation and tests."
    )
