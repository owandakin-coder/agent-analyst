"""trend_signal.trend_actions: buy leaders in an uptrend, exit on a 200-day break."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_signal import BUY_ACTION, trend_actions


def _frame(prices):
    return pd.DataFrame({"close": np.asarray(prices, dtype=float)})


def _uptrend(n=260, drift=0.002):
    return _frame(100 * np.exp(np.arange(n) * drift))


def _downtrend(n=260, drift=-0.002):
    return _frame(100 * np.exp(np.arange(n) * drift))


def test_uptrend_not_held_is_bought():
    a = trend_actions({"UP": _uptrend()}, ["UP"], {})
    assert a[0] == BUY_ACTION


def test_downtrend_not_held_is_ignored():
    assert trend_actions({"DN": _downtrend()}, ["DN"], {})[0] == 0.0


def test_held_stock_below_200_day_average_is_sold():
    assert trend_actions({"DN": _downtrend()}, ["DN"], {"DN": 10.0})[0] == -1.0


def test_held_stock_in_uptrend_is_topped_up():
    assert trend_actions({"UP": _uptrend()}, ["UP"], {"UP": 10.0})[0] == BUY_ACTION


def test_short_history_gives_no_signal():
    assert trend_actions({"X": _uptrend(150)}, ["X"], {})[0] == 0.0


def test_only_the_strongest_momentum_names_fill_the_open_slots():
    data = {f"S{k}": _uptrend(drift=0.001 + 0.0005 * k) for k in range(4)}
    a = trend_actions(data, list(data), {}, top_n=2)
    assert list(np.flatnonzero(a)) == [2, 3]


def test_slots_already_held_reduce_new_buys():
    data = {f"S{k}": _uptrend(drift=0.001 + 0.0005 * k) for k in range(3)}
    a = trend_actions(data, list(data), {"S0": 5.0}, top_n=2)
    assert a[0] == BUY_ACTION and a[2] == BUY_ACTION and a[1] == 0.0


def test_trend_source_replaces_model_action_and_turns_kelly_off(monkeypatch):
    import live_trader
    from tests.test_live_trader_execution import _build_trader

    monkeypatch.setattr(live_trader, "SIGNAL_SOURCE", "trend")
    trader = _build_trader()
    assert trader.risk_manager.use_kelly is False
    data = {t: _uptrend() for t in trader.tickers}
    model_says = np.full(len(trader.tickers), -0.9)
    out = trader._apply_signal_source(model_says, data, {})
    assert (out > 0).all()


def test_model_source_leaves_the_model_action_alone(monkeypatch):
    import live_trader
    from tests.test_live_trader_execution import _build_trader

    monkeypatch.setattr(live_trader, "SIGNAL_SOURCE", "model")
    trader = _build_trader()
    raw = np.array([0.3, -0.2, 0.0])
    assert trader._apply_signal_source(raw, {}, {}) is raw
