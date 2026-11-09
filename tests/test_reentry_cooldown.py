"""
test_reentry_cooldown.py
========================
A ticker sold inside the cooldown window must not be bought back. Seen live on
2026-11-05: a trailing stop sold all 26 JNJ and the next cycle bought it again.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tests.test_live_trader_execution import _build_trader

PRICES = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
POSITIONS = {"AAPL": 0.0, "MSFT": 0.0, "GOOGL": 0.0}
BUY_ALL = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _bought(events):
    return {e["ticker"] for e in events if e.get("side") == "BUY" or e.get("event_type") == "buy_signal"}


def test_recently_sold_ticker_is_not_bought_back():
    trader = _build_trader()
    trader.broker.recent_sell_tickers = lambda days: {"AAPL"}
    bought = _bought(trader._execute_actions(BUY_ALL, PRICES, 100_000.0, dict(POSITIONS)))
    assert "AAPL" not in bought
    assert bought >= {"MSFT", "GOOGL"}


def test_without_a_recent_sell_the_ticker_is_bought():
    trader = _build_trader()
    trader.broker.recent_sell_tickers = lambda days: set()
    assert "AAPL" in _bought(trader._execute_actions(BUY_ALL, PRICES, 100_000.0, dict(POSITIONS)))


def test_failed_lookup_does_not_block_buys():
    trader = _build_trader()

    def boom(days):
        raise RuntimeError("broker down")

    trader.broker.recent_sell_tickers = boom
    assert "AAPL" in _bought(trader._execute_actions(BUY_ALL, PRICES, 100_000.0, dict(POSITIONS)))


def test_zero_days_disables_the_cooldown(monkeypatch):
    import live_trader

    monkeypatch.setattr(live_trader, "REENTRY_COOLDOWN_DAYS", 0)
    trader = _build_trader()
    trader.broker.recent_sell_tickers = lambda days: {"AAPL"}
    assert "AAPL" in _bought(trader._execute_actions(BUY_ALL, PRICES, 100_000.0, dict(POSITIONS)))


def test_alpaca_lookup_counts_live_and_filled_sells_only():
    from broker_api import AlpacaBrokerAPI

    orders = [
        SimpleNamespace(symbol="JNJ", side="OrderSide.SELL", status="OrderStatus.FILLED"),
        SimpleNamespace(symbol="JPM", side="OrderSide.SELL", status="OrderStatus.PENDING_NEW"),
        SimpleNamespace(symbol="XOM", side="OrderSide.SELL", status="OrderStatus.CANCELED"),
        SimpleNamespace(symbol="AAPL", side="OrderSide.BUY", status="OrderStatus.FILLED"),
    ]
    broker = AlpacaBrokerAPI.__new__(AlpacaBrokerAPI)
    broker._trading = SimpleNamespace(get_orders=lambda filter=None: orders)
    assert broker.recent_sell_tickers(3) == {"JNJ", "JPM"}
