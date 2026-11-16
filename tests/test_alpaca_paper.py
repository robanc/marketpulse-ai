"""Safety tests for the Alpaca paper-trading configuration."""

import pytest

from src.broker.alpaca_paper import load_paper_credentials


def test_valid_paper_credentials(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "ALPACA_API_KEY=test_key\n"
        "ALPACA_SECRET_KEY=test_secret\n"
        "ALPACA_PAPER=true\n",
        encoding="utf-8",
    )

    assert load_paper_credentials(env) == ("test_key", "test_secret")


def test_live_trading_configuration_rejected(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "ALPACA_API_KEY=test_key\n"
        "ALPACA_SECRET_KEY=test_secret\n"
        "ALPACA_PAPER=false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paper-trading mode"):
        load_paper_credentials(env)


def test_missing_credentials_rejected(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ALPACA_PAPER=true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="credentials are missing"):
        load_paper_credentials(env)


def test_missing_env_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_paper_credentials(tmp_path / "missing.env")


def test_trading_client_always_uses_paper_mode(tmp_path, monkeypatch):
    """Verify that live brokerage connections cannot be created."""
    from unittest.mock import MagicMock

    import src.broker.alpaca_paper as broker

    env = tmp_path / ".env"
    env.write_text(
        "ALPACA_API_KEY=dummy_key\n"
        "ALPACA_SECRET_KEY=dummy_secret\n"
        "ALPACA_PAPER=true\n",
        encoding="utf-8",
    )

    mock_client = MagicMock()
    monkeypatch.setattr(broker, "TradingClient", mock_client)

    broker.create_paper_client(env)

    mock_client.assert_called_once_with(
        api_key="dummy_key",
        secret_key="dummy_secret",
        paper=True,
    )
