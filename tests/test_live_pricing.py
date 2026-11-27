from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import broker_api
import live_trader


def _broker(monkeypatch, quotes, trades):
    monkeypatch.setattr(broker_api, "StockLatestQuoteRequest", lambda **kw: "quotes")
    monkeypatch.setattr(broker_api, "StockLatestTradeRequest", lambda **kw: "trades")
    broker = broker_api.AlpacaBrokerAPI.__new__(broker_api.AlpacaBrokerAPI)
    broker._data = SimpleNamespace(get_stock_latest_quote=lambda r: quotes, get_stock_latest_trade=lambda r: trades)
    broker._call_with_retry = lambda name, fn, *args, **kwargs: fn(*args, **kwargs)
    return broker


def _quote(bid, ask, age=1):
    return SimpleNamespace(bid_price=bid, ask_price=ask, timestamp=datetime.now(timezone.utc) - timedelta(seconds=age))


def _trade(price, age=1):
    return SimpleNamespace(price=price, timestamp=datetime.now(timezone.utc) - timedelta(seconds=age))


def test_recent_trade_beats_a_wide_quote_midpoint(monkeypatch):
    broker = _broker(monkeypatch, {"JPM": _quote(320.0, 339.46)}, {"JPM": _trade(338.5)})
    info = broker.get_latest_quotes_info(["JPM"])["JPM"]
    assert info["price"] == 338.5
    assert info["source"] == "alpaca_latest_trade"
    assert info["reliable"] is True


@pytest.mark.parametrize("bid,ask,reliable", [(338.0, 338.4, True), (320.0, 339.46, False), (0.0, 339.0, False)])
def test_without_a_recent_trade_only_a_tight_quote_is_reliable(monkeypatch, bid, ask, reliable):
    broker = _broker(monkeypatch, {"JPM": _quote(bid, ask)}, {"JPM": _trade(338.5, age=600)})
    info = broker.get_latest_quotes_info(["JPM"])["JPM"]
    assert info["source"] == "alpaca_latest_quote"
    assert info["reliable"] is reliable


def test_an_unreliable_price_neither_stops_nor_trades():
    trader = live_trader.LiveTrader.__new__(live_trader.LiveTrader)
    trader.tickers, trader.num_stocks, trader.stop_loss_pct = ["JPM"], 1, 0.06
    trader._entry_prices, trader._trailing_highs = {"JPM": 350.0}, {"JPM": 354.0}
    trader._unreliable_tickers = {"JPM"}
    trader.broker = SimpleNamespace(recent_sell_tickers=lambda days: set())
    trader._place_order = lambda *a: pytest.fail("no order may be placed on an unreliable price")
    trader._portfolio_market_value = lambda prices, positions: 12 * prices["JPM"]
    trader._auto_deleverage_if_needed = lambda prices, positions, cash, nw, events: (cash, positions)
    events = trader._execute_actions(np.array([-1.0]), {"JPM": 320.0}, 50_000.0, {"JPM": 12.0})
    assert events == []
    assert trader._trailing_highs["JPM"] == 354.0


def test_a_trailing_high_above_anything_traded_is_lowered():
    trader = live_trader.LiveTrader.__new__(live_trader.LiveTrader)
    trader._entry_prices = {"SPY": 740.0}
    trader._trailing_highs = {"SPY": 774.35}
    bars = {"SPY": pd.DataFrame({"high": [757.0, 749.8, 739.9], "close": [754.0, 740.0, 716.0]})}
    trader._clamp_trailing_highs(bars, {"SPY": 727.0})
    assert trader._trailing_highs["SPY"] == 757.0


def test_an_unreliable_price_never_becomes_a_trailing_high():
    trader = live_trader.LiveTrader.__new__(live_trader.LiveTrader)
    trader._unreliable_tickers = {"JPM"}
    trader._entry_prices, trader._trailing_highs = {}, {}
    snapshot = {"position_details": {"JPM": {"avg_entry_price": 320.0}}}
    trader._hydrate_entry_state_from_broker({"JPM": 5.0}, snapshot, {"JPM": 353.0})
    assert trader._trailing_highs["JPM"] == 320.0
    trader._trailing_highs["JPM"] = 353.0
    trader._clamp_trailing_highs({"JPM": pd.DataFrame({"high": [330.0, 325.0]})}, {"JPM": 353.0})
    assert trader._trailing_highs["JPM"] == 330.0
