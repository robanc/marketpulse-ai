"""Read-only Alpaca paper-trading account integration."""

from pathlib import Path

from alpaca.trading.client import TradingClient


def load_paper_credentials(env_path=".env"):
    """Load credentials without printing or logging secrets."""
    path = Path(env_path)
    if not path.is_file():
        raise FileNotFoundError("Alpaca .env configuration file not found")

    config = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()

    if config.get("ALPACA_PAPER", "").lower() != "true":
        raise ValueError("Alpaca paper-trading mode must be enabled")

    api_key = config.get("ALPACA_API_KEY")
    secret_key = config.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("Alpaca paper-trading credentials are missing")

    return api_key, secret_key


def create_paper_client(env_path=".env"):
    """Create an Alpaca client restricted to paper trading."""
    api_key, secret_key = load_paper_credentials(env_path)

    return TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
    )


def get_paper_account_summary(env_path=".env"):
    """Retrieve account details without submitting orders."""
    client = create_paper_client(env_path)
    account = client.get_account()

    return {
        "status": str(account.status),
        "currency": account.currency,
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "trading_blocked": account.trading_blocked,
    }


def get_open_paper_orders(client):
    """Retrieve only open paper orders; never submit or cancel orders."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    return client.get_orders(filter=request)
