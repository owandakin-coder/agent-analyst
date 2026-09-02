"""
test_entry_state_recovery.py
=============================
בדיקות ל-LiveTrader._hydrate_entry_state_from_broker():
- אחרי קריסת worker, פוזיציה קיימת מקבלת entry_price אמיתי מהברוקר,
  לא נשארת בלי הגנת trailing-stop.
- ticker שכבר במעקב לא נדרס.
- אם הברוקר לא מחזיר avg_entry_price, נופלים חזרה למחיר נוכחי (לא נתקעים).
- פוזיציה סגורה (held<=0) לא נרשמת.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _DummyVecNorm:
    def normalize_obs(self, obs):
        return obs


class _DummyModel:
    def predict(self, obs, deterministic=True):
        return np.zeros((1, 3), dtype=float), None


class _DummyDataManager:
    pass


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


class TestEntryStateRecovery:

    def test_recovers_entry_price_after_restart(self):
        trader = _build_trader()
        assert trader._entry_prices == {}

        positions = {"AAPL": 10.0}
        snapshot = {"position_details": {"AAPL": {"avg_entry_price": 150.0}}}
        current_prices = {"AAPL": 165.0}

        trader._hydrate_entry_state_from_broker(positions, snapshot, current_prices)

        assert trader._entry_prices["AAPL"] == 150.0
        assert trader._trailing_highs["AAPL"] == 165.0  # max(entry, current)

    def test_does_not_overwrite_already_tracked_entry(self):
        trader = _build_trader()
        trader._entry_prices["AAPL"] = 100.0
        trader._trailing_highs["AAPL"] = 200.0

        positions = {"AAPL": 10.0}
        snapshot = {"position_details": {"AAPL": {"avg_entry_price": 150.0}}}
        current_prices = {"AAPL": 165.0}

        trader._hydrate_entry_state_from_broker(positions, snapshot, current_prices)

        # Untouched — a live-tracked trailing high must never be clobbered
        # by re-deriving from the broker mid-session.
        assert trader._entry_prices["AAPL"] == 100.0
        assert trader._trailing_highs["AAPL"] == 200.0

    def test_falls_back_to_current_price_when_broker_has_no_entry(self):
        trader = _build_trader()

        positions = {"AAPL": 10.0}
        snapshot = {"position_details": {}}
        current_prices = {"AAPL": 172.0}

        trader._hydrate_entry_state_from_broker(positions, snapshot, current_prices)

        assert trader._entry_prices["AAPL"] == 172.0
        assert trader._trailing_highs["AAPL"] == 172.0

    def test_skips_positions_with_no_holding(self):
        trader = _build_trader()

        positions = {"AAPL": 0.0}
        snapshot = {"position_details": {"AAPL": {"avg_entry_price": 150.0}}}
        current_prices = {"AAPL": 165.0}

        trader._hydrate_entry_state_from_broker(positions, snapshot, current_prices)

        assert "AAPL" not in trader._entry_prices

    def test_trailing_stop_check_no_longer_skips_recovered_position(self):
        """End-to-end: a held position with no local entry_price gets one
        from the broker, so _execute_actions' trailing-stop loop can act on it
        instead of silently skipping (entry<=0 guard)."""
        trader = _build_trader()
        positions = {"AAPL": 10.0}
        snapshot = {"position_details": {"AAPL": {"avg_entry_price": 150.0}}}
        # Price has since dropped hard from the recovered entry — trailing
        # stop should be able to see this once state is hydrated.
        current_prices = {"AAPL": 130.0}

        trader._hydrate_entry_state_from_broker(positions, snapshot, current_prices)

        assert trader._entry_prices.get("AAPL", 0.0) > 0
        drop_pct = (trader._trailing_highs["AAPL"] - current_prices["AAPL"]) / trader._trailing_highs["AAPL"]
        assert drop_pct >= trader.stop_loss_pct
