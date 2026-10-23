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

    def _compute_features(self, raw, ticker):
        from data_manager import DataManager
        dm = DataManager.__new__(DataManager)
        return dm._compute_features(raw, ticker)


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


def _build_trader():
    from broker_api import BrokerAPIStub
    from live_trader import LiveTrader
    from risk_manager import RiskManager

    broker = BrokerAPIStub()
    broker.set_cash(100_000.0)
    risk_manager = RiskManager(initial_capital=100_000.0)
    return LiveTrader(
        model=_DummyModel(),
        broker=broker,
        data_manager=_DummyDataManager(),
        risk_manager=risk_manager,
        vec_norm=_DummyVecNorm(),
        tickers=["AAPL", "MSFT", "GOOGL"],
        initial_capital=100_000.0,
    )


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


def test_recent_download_cannot_make_an_old_exchange_quote_fresh(monkeypatch, multi_featured):
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
    assert executed["called"] is False


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


def test_closed_position_daily_loss_survives_worker_restart(monkeypatch, multi_featured):
    trader = _make_trader(monkeypatch, multi_featured)
    monkeypatch.setenv("ATZMA_DAILY_REALIZED_LOSS_LIMIT", "1000")
    # No remaining positions or unrealized loss: the old algebra always
    # returned zero and allowed another cycle despite a $2,000 daily loss.
    snapshot = {"equity": 98_000, "last_equity": 100_000, "position_details": {}}
    assert trader._enforce_daily_loss_limits(snapshot, 98_000) is True


def test_daily_gain_does_not_trigger_loss_limit(monkeypatch, multi_featured):
    trader = _make_trader(monkeypatch, multi_featured)
    snapshot = {"equity": 102_000, "last_equity": 100_000, "position_details": {"AAPL": {"unrealized_pl": 10_000}}}
    assert trader._enforce_daily_loss_limits(snapshot, 102_000) is False


def test_fetch_fresh_data_falls_back_to_broker_bars(monkeypatch, long_raw_ohlcv):
    raw_ohlcv = long_raw_ohlcv
    trader = _build_trader()
    monkeypatch.setattr(trader.data_manager, "load_all", lambda force_download=True: (_ for _ in ()).throw(ValueError("yf down")))
    monkeypatch.setattr(
        trader.broker,
        "get_historical_bars",
        lambda tickers, start, end: {ticker: raw_ohlcv.copy() for ticker in tickers},
        raising=False,
    )

    data = trader._fetch_fresh_data()

    assert set(data.keys()) == {"AAPL", "MSFT", "GOOGL"}
    assert all("close" in df.columns for df in data.values())
    assert all(len(df) >= 200 for df in data.values())


@pytest.mark.parametrize("primary_fails", [False, True])
def test_fetch_requires_trend_history_after_primary_and_fallback(monkeypatch, primary_fails):
    import pandas as pd
    from live_trader import LiveTrader

    trader = LiveTrader.__new__(LiveTrader)
    trader.tickers = ["AAPL"]
    short = {"AAPL": pd.DataFrame({"close": np.linspace(100, 120, 199)})}
    def primary(force_download):
        if primary_fails:
            raise ValueError("primary unavailable")
        return short
    trader.data_manager = SimpleNamespace(load_all=primary)
    monkeypatch.setattr(trader, "_fetch_broker_feature_data", lambda **kwargs: short)
    with pytest.raises(ValueError, match="200 rows required"):
        trader._fetch_fresh_data()


def test_fetch_backfills_short_primary_with_real_trend_history(monkeypatch):
    import pandas as pd
    from live_trader import LiveTrader

    trader = LiveTrader.__new__(LiveTrader)
    trader.tickers = ["AAPL"]
    short = pd.DataFrame({"close": np.linspace(100, 120, 80)})
    complete = pd.DataFrame({"close": np.linspace(100, 140, 200)})
    trader.data_manager = SimpleNamespace(load_all=lambda force_download: {"AAPL": short})
    calls = []
    def fallback(**kwargs):
        calls.append(kwargs)
        return {"AAPL": complete}
    monkeypatch.setattr(trader, "_fetch_broker_feature_data", fallback)
    assert trader._fetch_fresh_data()["AAPL"] is complete
    assert calls[0]["tickers"] == ["AAPL"]
    start = datetime.fromisoformat(calls[0]["start"])
    end = datetime.fromisoformat(calls[0]["end"])
    assert (end - start).days >= 730


def test_detect_regime_uses_existing_spy_data(monkeypatch, featured_df):
    trader = _build_trader()
    spy_df = featured_df.copy()
    spy_df["close"] = np.linspace(100, 140, len(spy_df))

    # _detect_regime also fetches VIX (a separate, legitimate yfinance call
    # unrelated to this test's concern) — stub it out directly instead of
    # letting it hit the network or trip the fake SPY-download module below.
    monkeypatch.setattr(trader._alt_fetcher, "fetch_vix_raw", lambda: 18.0)

    called = {"downloaded": False}

    def fail_download(*args, **kwargs):
        called["downloaded"] = True
        raise AssertionError("yfinance should not be called when SPY is already present")

    monkeypatch.setitem(__import__("sys").modules, "yfinance", type("YF", (), {"download": staticmethod(fail_download)}))

    signal = trader._detect_regime({"SPY": spy_df})

    assert signal is not None
    assert called["downloaded"] is False
