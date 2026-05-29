"""
End-to-end tests for the live trading flow.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class FixedActionModel:
    def __init__(self, action):
        self._action = np.array(action, dtype=float)

    def predict(self, obs, deterministic=True):
        return self._action.copy(), None


class IdentityVecNorm:
    def normalize_obs(self, obs):
        return obs


def _make_broker(monkeypatch, tmp_path, trading_mock):
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE_KEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_SECRET")

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

    return broker


def _make_trader(multi_featured, broker):
    from data_manager import DataManager
    from live_trader import LiveTrader
    from risk_manager import RiskManager

    dm = DataManager.__new__(DataManager)
    dm.load_all = MagicMock(return_value=multi_featured)
    dm.tickers = ["AAPL", "MSFT", "GOOGL"]
    dm.start = dm.end = ""

    trader = LiveTrader(
        model=FixedActionModel([1.0, 0.0, 0.0]),
        broker=broker,
        data_manager=dm,
        risk_manager=RiskManager(100_000.0),
        vec_norm=IdentityVecNorm(),
        tickers=["AAPL", "MSFT", "GOOGL"],
        initial_capital=100_000.0,
    )
    trader._fetch_fresh_data = MagicMock(return_value=multi_featured)
    trader._detect_regime = MagicMock(return_value=None)
    trader._alt_fetcher.fetch_all = MagicMock(return_value={})
    trader._send_daily_summary = MagicMock()
    trader._telegram = MagicMock()
    return trader


@pytest.fixture
def trading_mock():
    mock = MagicMock()
    mock.get_account.return_value = MagicMock(
        cash="100000",
        equity="100000",
        buying_power="100000",
        portfolio_value="100000",
        status="ACTIVE",
    )
    mock.get_all_positions.return_value = []
    mock.get_orders.return_value = []
    order = MagicMock()
    order.id = "order-123"
    order.status = "accepted"
    order.client_order_id = "ATZMA-2026-05-29-AAPL-BUY-633"
    mock.submit_order.return_value = order
    return mock


class TestLiveTradingEndToEnd:
    def test_run_once_places_order_and_persists_state(self, monkeypatch, tmp_path, multi_featured, trading_mock):
        broker = _make_broker(monkeypatch, tmp_path, trading_mock)
        broker.get_latest_prices = MagicMock(return_value={"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 130.0})

        trader = _make_trader(multi_featured, broker)
        trader.run_once()

        trading_mock.submit_order.assert_called_once()
        state_path = tmp_path / "submitted_orders.json"
        assert state_path.exists()
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        assert len(saved) == 1
        saved_entry = next(iter(saved.values()))
        assert saved_entry["ticker"] == "AAPL"
        assert saved_entry["side"] == "BUY"

    def test_second_run_new_process_blocks_duplicate(self, monkeypatch, tmp_path, multi_featured, trading_mock):
        broker1 = _make_broker(monkeypatch, tmp_path, trading_mock)
        broker1.get_latest_prices = MagicMock(return_value={"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 130.0})
        trader1 = _make_trader(multi_featured, broker1)
        trader1.run_once()
        assert trading_mock.submit_order.call_count == 1

        trading_mock_2 = MagicMock()
        trading_mock_2.get_account.return_value = trading_mock.get_account.return_value
        trading_mock_2.get_all_positions.return_value = []
        trading_mock_2.get_orders.return_value = []
        trading_mock_2.submit_order.return_value = trading_mock.submit_order.return_value

        broker2 = _make_broker(monkeypatch, tmp_path, trading_mock_2)
        broker2.get_latest_prices = MagicMock(return_value={"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 130.0})
        trader2 = _make_trader(multi_featured, broker2)
        trader2.run_once()

        trading_mock_2.submit_order.assert_not_called()

    def test_reconciliation_failure_skips_cycle(self, monkeypatch, tmp_path, multi_featured, trading_mock):
        broker = _make_broker(monkeypatch, tmp_path, trading_mock)
        broker.reconcile_account_state = MagicMock(side_effect=ConnectionError("broker down"))
        broker.get_latest_prices = MagicMock(return_value={"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 130.0})

        trader = _make_trader(multi_featured, broker)
        trader.run_once()

        trading_mock.submit_order.assert_not_called()
