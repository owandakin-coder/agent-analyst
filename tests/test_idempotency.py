"""
Tests for duplicate-order protection and broker reconciliation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def broker_with_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE_KEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_SECRET")

    trading_mock = MagicMock()
    order_mock = MagicMock()
    order_mock.id = "order-123"
    order_mock.status = "accepted"
    order_mock.client_order_id = "ATZMA-2026-05-29-AAPL-BUY-10"
    trading_mock.submit_order.return_value = order_mock
    trading_mock.get_account.return_value = MagicMock(
        cash="50000",
        equity="100000",
        buying_power="50000",
        portfolio_value="100000",
        status="ACTIVE",
    )
    trading_mock.get_all_positions.return_value = []
    trading_mock.get_orders.return_value = []

    import broker_api as ba

    monkeypatch.setattr(ba, "LOG_FILE", str(tmp_path / "orders.log"))
    monkeypatch.setattr(ba, "TRADES_CSV", str(tmp_path / "trades.csv"))
    monkeypatch.setattr(ba, "SUBMITTED_ORDERS_FILE", str(tmp_path / "submitted_orders.json"))

    with patch("broker_api.TradingClient", return_value=trading_mock), patch(
        "broker_api.StockHistoricalDataClient", return_value=MagicMock()
    ):
        from broker_api import AlpacaBrokerAPI

        broker = AlpacaBrokerAPI(paper=True, auto_approve=True)
        broker._trading = trading_mock

    return broker, trading_mock


class TestIdempotency:
    def test_first_order_submitted(self, broker_with_mock):
        broker, trading = broker_with_mock
        result = broker._submit_order("AAPL", 10, "buy")
        assert result["status"] != "DUPLICATE_BLOCKED"
        trading.submit_order.assert_called_once()

    def test_duplicate_order_blocked_same_instance(self, broker_with_mock):
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("AAPL", 10, "buy")
        assert result["status"] == "DUPLICATE_BLOCKED"
        assert result["source"] == "local_state"
        assert trading.submit_order.call_count == 1

    def test_duplicate_order_blocked_new_instance_from_state_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ALPACA_API_KEY", "KEY2")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "SECRET2")

        trading_mock = MagicMock()
        trading_mock.submit_order.return_value = MagicMock(id="order-1", status="accepted")
        trading_mock.get_account.return_value = MagicMock(
            cash="50000",
            equity="100000",
            buying_power="50000",
            portfolio_value="100000",
            status="ACTIVE",
        )
        trading_mock.get_all_positions.return_value = []
        trading_mock.get_orders.return_value = []

        import broker_api as ba

        state_file = tmp_path / "submitted_orders.json"
        monkeypatch.setattr(ba, "LOG_FILE", str(tmp_path / "o.log"))
        monkeypatch.setattr(ba, "TRADES_CSV", str(tmp_path / "t.csv"))
        monkeypatch.setattr(ba, "SUBMITTED_ORDERS_FILE", str(state_file))

        with patch("broker_api.TradingClient", return_value=trading_mock), patch(
            "broker_api.StockHistoricalDataClient", return_value=MagicMock()
        ):
            from broker_api import AlpacaBrokerAPI

            first = AlpacaBrokerAPI(paper=True, auto_approve=True)
            first._trading = trading_mock
            first._submit_order("AAPL", 10, "buy")

            second = AlpacaBrokerAPI(paper=True, auto_approve=True)
            second._trading = trading_mock
            result = second._submit_order("AAPL", 10, "buy")

        assert result["status"] == "DUPLICATE_BLOCKED"
        assert result["source"] == "local_state"
        assert trading_mock.submit_order.call_count == 1

    def test_duplicate_order_blocked_from_broker_history(self, broker_with_mock):
        broker, trading = broker_with_mock
        existing = MagicMock()
        existing.symbol = "AAPL"
        existing.side = "buy"
        existing.qty = "10"
        existing.client_order_id = broker._client_order_id_from_key(broker._order_key("AAPL", "buy", 10))
        existing.submitted_at = datetime.now(timezone.utc)
        trading.get_orders.return_value = [existing]

        result = broker._submit_order("AAPL", 10, "buy")
        assert result["status"] == "DUPLICATE_BLOCKED"
        assert result["source"] == "broker_history"
        trading.submit_order.assert_not_called()

    def test_duplicate_check_network_error_falls_back_to_submit(self, broker_with_mock):
        broker, trading = broker_with_mock
        trading.get_orders.side_effect = ConnectionError("timeout")

        result = broker._submit_order("AAPL", 10, "buy")
        assert result["status"] != "DUPLICATE_BLOCKED"
        trading.submit_order.assert_called_once()

    def test_different_ticker_allowed(self, broker_with_mock):
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("MSFT", 10, "buy")
        assert result["status"] != "DUPLICATE_BLOCKED"
        assert trading.submit_order.call_count == 2

    def test_different_side_allowed(self, broker_with_mock):
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("AAPL", 10, "sell", account_snapshot={"positions": {"AAPL": 10}})
        assert result["status"] != "DUPLICATE_BLOCKED"

    def test_different_qty_allowed(self, broker_with_mock):
        broker, trading = broker_with_mock
        broker._submit_order("AAPL", 10, "buy")
        result = broker._submit_order("AAPL", 20, "buy")
        assert result["status"] != "DUPLICATE_BLOCKED"
        assert trading.submit_order.call_count == 2

    def test_order_key_format(self, broker_with_mock):
        broker, _ = broker_with_mock
        key = broker._order_key("AAPL", "buy", 10)
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[1] == "AAPL"
        assert parts[2] == "BUY"
        assert parts[3] == "10"

    def test_reconcile_snapshot_contains_positions(self, broker_with_mock):
        broker, trading = broker_with_mock
        position = MagicMock(symbol="AAPL", qty="5")
        trading.get_all_positions.return_value = [position]
        snapshot = broker.reconcile_account_state()
        assert snapshot["cash"] == 50000.0
        assert snapshot["positions"]["AAPL"] == 5.0

    def test_reconcile_snapshot_uses_cached_state_on_retry_exhaustion(self, broker_with_mock):
        broker, trading = broker_with_mock
        first = broker.reconcile_account_state()
        trading.get_account.side_effect = ConnectionError("broker unavailable")
        trading.get_all_positions.side_effect = ConnectionError("broker unavailable")
        second = broker.reconcile_account_state()
        assert second["cash"] == first["cash"]
        assert second["status"] == first["status"]
