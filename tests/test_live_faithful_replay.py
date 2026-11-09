"""SimBroker accounting for live_faithful_replay.py (no market data needed)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from live_faithful_replay import SimBroker


def _broker():
    b = SimBroker(cash=10_000.0, slippage=0.001, commission=0.002)
    b.fill_prices = {"AAA": 100.0}
    b.now = datetime(2025, 3, 3, tzinfo=timezone.utc)
    return b


def test_buy_fills_above_open_and_pays_commission():
    b = _broker()
    r = b.buy("AAA", 10, 99.0)
    assert r["filled_avg_price"] == 100.1
    assert b.positions["AAA"] == 10
    assert abs(b.cash - (10_000 - 10 * 100.1 - 10 * 100.1 * 0.002)) < 1e-9


def test_sell_fills_below_open_and_caps_at_holdings():
    b = _broker()
    b.buy("AAA", 5, 100.0)
    r = b.sell("AAA", 50, 100.0)
    assert r["filled_qty"] == 5 and r["filled_avg_price"] == 99.9
    assert b.positions["AAA"] == 0
    assert b.sell("AAA", 1, 100.0)["status"] == "REJECTED"


def test_recent_sell_window_follows_simulated_clock():
    b = _broker()
    b.buy("AAA", 5, 100.0)
    b.sell("AAA", 5, 100.0)
    assert b.recent_sell_tickers(3) == {"AAA"}
    b.now += timedelta(days=4)
    assert b.recent_sell_tickers(3) == set()
