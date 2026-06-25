from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest


class _DummyVecNorm:
    def normalize_obs(self, obs):
        return obs


class _DummyModel:
    def predict(self, obs, deterministic=True):
        return np.zeros((1, 3), dtype=float), None


class _DummyDataManager:
    start = None
    end = None

    def load_all(self, force_download=True):
        raise AssertionError("load_all should be patched in the test")


def _make_trader(monkeypatch, multi_featured):
    from broker_api import BrokerAPIStub
    from live_trader import LiveTrader
    from risk_manager import RiskManager

    broker = BrokerAPIStub()
    broker.set_cash(100_000.0)
    risk_manager = RiskManager(initial_capital=100_000.0)
    trader = LiveTrader(
        model=_DummyModel(),
        broker=broker,
        data_manager=_DummyDataManager(),
        risk_manager=risk_manager,
        vec_norm=_DummyVecNorm(),
        tickers=["AAPL", "MSFT", "GOOGL"],
        initial_capital=100_000.0,
    )
    monkeypatch.setattr(trader, "_fetch_fresh_data", lambda: multi_featured)
    monkeypatch.setattr(trader, "_detect_regime", lambda _fresh_data: None)
    monkeypatch.setattr(trader._alt_fetcher, "fetch_all", lambda: {})
    monkeypatch.setattr(trader, "_send_daily_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("live_trader.write_last_decision", lambda payload: None)
    return trader


def test_stale_quotes_abort_execution(monkeypatch, multi_featured):
    trader = _make_trader(monkeypatch, multi_featured)
    monkeypatch.setenv("ATZMA_REQUIRE_FRESH_QUOTES", "1")
    monkeypatch.setattr("live_trader.load_control_state", lambda: {"trading_enabled": True, "emergency_stop": False})
    monkeypatch.setattr("live_trader.can_trade", lambda state: (True, None))
    monkeypatch.setattr(trader.broker, "get_latest_prices", lambda tickers: {ticker: 100.0 for ticker in tickers})
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    monkeypatch.setattr(trader.broker, "get_latest_quotes_info", lambda tickers: {
        ticker: {"price": 100.0, "timestamp": stale, "source": "alpaca_latest_quote"} for ticker in tickers
    })
    monkeypatch.setattr(trader.broker, "reconcile_account_state", lambda: {"cash": 100_000.0, "equity": 100_000.0, "positions": {}, "position_details": {}})
    executed = {"called": False}
    monkeypatch.setattr(trader, "_execute_actions", lambda *args, **kwargs: executed.__setitem__("called", True))

    assert trader.run_once() is None
    assert executed["called"] is False


def test_recent_observed_at_allows_execution_when_exchange_timestamp_is_old(monkeypatch, multi_featured):
    trader = _make_trader(monkeypatch, multi_featured)
    monkeypatch.setenv("ATZMA_REQUIRE_FRESH_QUOTES", "1")
    monkeypatch.setattr("live_trader.load_control_state", lambda: {"trading_enabled": True, "emergency_stop": False})
    monkeypatch.setattr("live_trader.can_trade", lambda state: (True, None))
    monkeypatch.setattr(trader.broker, "get_latest_prices", lambda tickers: {ticker: 100.0 for ticker in tickers})
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    fresh = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(trader.broker, "get_latest_quotes_info", lambda tickers: {
        ticker: {
            "price": 100.0,
            "timestamp": old,
            "observed_at": fresh,
            "source": "alpaca_latest_trade",
        }
        for ticker in tickers
    })
    monkeypatch.setattr(trader.broker, "reconcile_account_state", lambda: {"cash": 100_000.0, "equity": 100_000.0, "positions": {}, "position_details": {}})
    executed = {"called": False}
    monkeypatch.setattr(trader, "_execute_actions", lambda *args, **kwargs: executed.__setitem__("called", True) or [])

    trader.run_once()
    assert executed["called"] is True


def test_control_plane_outage_fails_closed(monkeypatch, multi_featured):
    trader = _make_trader(monkeypatch, multi_featured)
    monkeypatch.setenv("ATZMA_FAIL_CLOSED_CONTROL", "1")
    monkeypatch.setattr("live_trader.load_control_state", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    executed = {"called": False}
    monkeypatch.setattr(trader, "_execute_actions", lambda *args, **kwargs: executed.__setitem__("called", True))

    assert trader.run_once() is None
    assert executed["called"] is False


def test_emergency_stop_blocks_execution(monkeypatch, multi_featured):
    trader = _make_trader(monkeypatch, multi_featured)
    monkeypatch.setattr("live_trader.load_control_state", lambda: {"trading_enabled": False, "emergency_stop": True})
    monkeypatch.setattr("live_trader.can_trade", lambda state: (False, "emergency_stop"))
    executed = {"called": False}
    monkeypatch.setattr(trader, "_execute_actions", lambda *args, **kwargs: executed.__setitem__("called", True))

    assert trader.run_once() is None
    assert executed["called"] is False


def test_daily_loss_breach_halts_execution(monkeypatch, multi_featured):
    trader = _make_trader(monkeypatch, multi_featured)
    monkeypatch.setenv("ATZMA_DAILY_REALIZED_LOSS_LIMIT", "1000")
    monkeypatch.setenv("ATZMA_DAILY_UNREALIZED_LOSS_LIMIT", "1000")
    monkeypatch.setattr("live_trader.load_control_state", lambda: {"trading_enabled": True, "emergency_stop": False})
    monkeypatch.setattr("live_trader.can_trade", lambda state: (True, None))
    now = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(trader.broker, "get_latest_prices", lambda tickers: {ticker: 100.0 for ticker in tickers})
    monkeypatch.setattr(trader.broker, "get_latest_quotes_info", lambda tickers: {
        ticker: {"price": 100.0, "timestamp": now, "source": "alpaca_latest_quote"} for ticker in tickers
    })
    monkeypatch.setattr(trader.broker, "reconcile_account_state", lambda: {
        "cash": 95_000.0,
        "equity": 95_000.0,
        "positions": {"AAPL": 10.0},
        "position_details": {"AAPL": {"qty": 10.0, "unrealized_pl": -2_500.0}},
    })
    executed = {"called": False}
    monkeypatch.setattr(trader, "_execute_actions", lambda *args, **kwargs: executed.__setitem__("called", True))

    assert trader.run_once() is None
    assert executed["called"] is False
