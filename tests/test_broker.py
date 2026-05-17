"""
test_broker.py
==============
בדיקות לחיבור ל-Alpaca, קריאת .env, וטיפול בשגיאות.
"""

import os
from unittest.mock import MagicMock, patch, call
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# 1. קריאת .env וחיבור
# ══════════════════════════════════════════════════════════════════════════════

class TestBrokerConnection:

    def test_missing_api_key_raises(self, monkeypatch):
        """EnvironmentError כשאין ALPACA_API_KEY."""
        monkeypatch.delenv("ALPACA_API_KEY",    raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

        with patch("broker_api.TradingClient"), \
             patch("broker_api.StockHistoricalDataClient"):
            from broker_api import AlpacaBrokerAPI
            with pytest.raises(EnvironmentError, match="ALPACA_API_KEY"):
                AlpacaBrokerAPI(paper=True)

    def test_missing_secret_key_raises(self, monkeypatch):
        """EnvironmentError כשאין ALPACA_SECRET_KEY."""
        monkeypatch.setenv("ALPACA_API_KEY",    "some_key")
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

        with patch("broker_api.TradingClient"), \
             patch("broker_api.StockHistoricalDataClient"):
            from broker_api import AlpacaBrokerAPI
            with pytest.raises(EnvironmentError, match="ALPACA_SECRET_KEY"):
                AlpacaBrokerAPI(paper=True)

    def test_paper_uses_paper_url(self, monkeypatch):
        """paper=True → base_url מכיל 'paper-api'."""
        monkeypatch.setenv("ALPACA_API_KEY",    "key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        monkeypatch.delenv("ALPACA_BASE_URL", raising=False)

        trading_mock = MagicMock()
        with patch("broker_api.TradingClient", return_value=trading_mock) as tc, \
             patch("broker_api.StockHistoricalDataClient"):
            from broker_api import AlpacaBrokerAPI
            b = AlpacaBrokerAPI(paper=True, auto_approve=True)

        assert "paper-api" in b.base_url

    def test_env_base_url_overrides_flag(self, monkeypatch):
        """ALPACA_BASE_URL ב-.env גובר על flag paper=True."""
        custom_url = "https://custom-broker.example.com"
        monkeypatch.setenv("ALPACA_API_KEY",    "key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        monkeypatch.setenv("ALPACA_BASE_URL",   custom_url)

        with patch("broker_api.TradingClient"), \
             patch("broker_api.StockHistoricalDataClient"):
            from broker_api import AlpacaBrokerAPI
            b = AlpacaBrokerAPI(paper=True, auto_approve=True)

        assert b.base_url == custom_url

    def test_trading_client_called_with_paper_flag(self, monkeypatch):
        """TradingClient נקרא עם paper=True."""
        monkeypatch.setenv("ALPACA_API_KEY",    "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.delenv("ALPACA_BASE_URL", raising=False)

        with patch("broker_api.TradingClient") as tc_cls, \
             patch("broker_api.StockHistoricalDataClient"):
            from broker_api import AlpacaBrokerAPI
            AlpacaBrokerAPI(paper=True, auto_approve=True)

        _, kwargs = tc_cls.call_args
        assert kwargs.get("paper") is True

    def test_network_error_on_get_account(self, broker):
        """שגיאת רשת ב-get_account לא קורסת את התהליך."""
        broker._trading.get_account.side_effect = ConnectionError("timeout")
        with pytest.raises(ConnectionError):
            broker.get_account()

    def test_auth_error_on_submit_order(self, broker):
        """שגיאת אותנטיקציה מוחזרת כ-dict עם status=ERROR."""
        broker._trading.submit_order.side_effect = Exception("403 Forbidden")
        result = broker._submit_order("AAPL", 1, "buy")
        assert result["status"] == "ERROR"
        assert "403" in result["error"]

    def test_auto_approve_default_is_false(self, monkeypatch):
        """ברירת מחדל auto_approve=False."""
        monkeypatch.setenv("ALPACA_API_KEY",    "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.delenv("ALPACA_BASE_URL", raising=False)

        with patch("broker_api.TradingClient"), \
             patch("broker_api.StockHistoricalDataClient"):
            from broker_api import AlpacaBrokerAPI
            b = AlpacaBrokerAPI(paper=True)

        assert b.auto_approve is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. חשבון ופוזיציות
# ══════════════════════════════════════════════════════════════════════════════

class TestAccountInfo:

    def test_get_account_returns_floats(self, broker, mock_alpaca_account):
        """get_account() מחזיר float לכל שדה."""
        result = broker.get_account()
        for key in ("cash", "equity", "buying_power", "portfolio_value"):
            assert isinstance(result[key], float), f"{key} should be float"

    def test_get_cash_matches_account(self, broker):
        """get_cash() = get_account()['cash']."""
        assert broker.get_cash() == broker.get_account()["cash"]

    def test_get_positions_empty(self, broker):
        """ללא פוזיציות → dict ריק."""
        broker._trading.get_all_positions.return_value = []
        assert broker.get_positions() == {}

    def test_get_positions_with_holdings(self, broker, mock_alpaca_position):
        """פוזיציה אחת → dict עם ticker ו-shares."""
        broker._trading.get_all_positions.return_value = [mock_alpaca_position]
        pos = broker.get_positions()
        assert "AAPL" in pos
        assert pos["AAPL"] == 10.0

    def test_get_positions_error_returns_empty(self, broker):
        """שגיאה ב-get_all_positions → dict ריק (לא קריסה)."""
        broker._trading.get_all_positions.side_effect = Exception("API down")
        assert broker.get_positions() == {}
