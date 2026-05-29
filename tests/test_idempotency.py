"""
test_idempotency.py
===================
בדיקות למניעת פקודות כפולות (idempotency) ב-AlpacaBrokerAPI.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def broker_with_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_API_KEY",    "FAKE_KEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_SECRET")

    trading_mock = MagicMock()
    order_mock   = MagicMock()
    order_mock.id     = "order-123"
    order_mock.status = "accepted"
    trading_mock.submit_order.return_value        = order_mock
    trading_mock.get_account.return_value         = MagicMock(
        cash="50000", equity="100000", buying_power="50000",
        portfolio_value="100000", status="ACTIVE"
    )
    trading_mock.get_all_positions.return_value   = []

    import broker_api as ba
    monkeypatch.setattr(ba, "LOG_FILE",   str(tmp_path / "orders.log"))
    monkeypatch.setattr(ba, "TRADES_CSV", str(tmp_path / "trades.csv"))

    with patch("broker_api.TradingClient",              return_value=trading_mock), \
         patch("broker_api.StockHistoricalDataClient",  return_value=MagicMock()):
        from broker_api import AlpacaBrokerAPI
        b = AlpacaBrokerAPI(paper=True, auto_approve=True)
        b._trading = trading_mock

    return b, trading_mock


class TestIdempotency:

    def test_first_order_submitted(self, broker_with_mock):
        """פקודה ראשונה תמיד עוברת."""
        broker, trading = broker_with_mock
        result = broker._submit_order("AAPL", 10, "buy")
        assert result["status"] != "DUPLICATE_BLOCKED"
        trading.submit_order.assert_called_once()

    def test_duplicate_order_blocked(self, broker_with_mock):
        """פקודה זהה ביום אחד נחסמת."""
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("AAPL", 10, "buy")
        assert result["status"] == "DUPLICATE_BLOCKED"
        # Alpaca API נקרא רק פעם אחת
        assert trading.submit_order.call_count == 1

    def test_different_ticker_allowed(self, broker_with_mock):
        """פקודות על מניות שונות מותרות."""
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("MSFT", 10, "buy")
        assert result["status"] != "DUPLICATE_BLOCKED"
        assert trading.submit_order.call_count == 2

    def test_different_side_allowed(self, broker_with_mock):
        """קנייה ומכירה של אותה מניה מותרות."""
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("AAPL", 10, "sell")
        assert result["status"] != "DUPLICATE_BLOCKED"

    def test_different_qty_allowed(self, broker_with_mock):
        """אותה מניה, כמות שונה = לא כפול."""
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("AAPL", 20, "buy")
        assert result["status"] != "DUPLICATE_BLOCKED"

    def test_order_key_format(self, broker_with_mock):
        """מפתח ה-idempotency מכיל תאריך+מניה+כיוון+כמות."""
        broker, _ = broker_with_mock
        key = broker._order_key("AAPL", "buy", 10)
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[1] == "AAPL"
        assert parts[2] == "BUY"
        assert parts[3] == "10"

    def test_submitted_keys_cleared_on_new_instance(self, monkeypatch, tmp_path):
        """מופע חדש של הברוקר = אין זיכרון מהסשן הקודם."""
        monkeypatch.setenv("ALPACA_API_KEY",    "KEY2")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "SECRET2")
        import broker_api as ba
        monkeypatch.setattr(ba, "LOG_FILE",   str(tmp_path / "o2.log"))
        monkeypatch.setattr(ba, "TRADES_CSV", str(tmp_path / "t2.csv"))

        with patch("broker_api.TradingClient",             return_value=MagicMock(
                    get_account=MagicMock(return_value=MagicMock(
                        cash="50000", equity="100000", buying_power="50000",
                        portfolio_value="100000", status="ACTIVE"
                    ))
                )), \
             patch("broker_api.StockHistoricalDataClient", return_value=MagicMock()):
            from broker_api import AlpacaBrokerAPI
            b1 = AlpacaBrokerAPI(paper=True, auto_approve=True)
            b2 = AlpacaBrokerAPI(paper=True, auto_approve=True)

        assert b1._submitted_keys is not b2._submitted_keys
        assert len(b2._submitted_keys) == 0
