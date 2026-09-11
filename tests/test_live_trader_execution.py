"""
test_live_trader_execution.py
==============================
בדיקה ל-LiveTrader._execute_actions(): אותו באג ביטול מתמטי שתוקן ב-
trading_env.py (ראה tests/test_trading_env.py) קיים גם כאן — זה הנתיב
האמיתי שמבצע קניות ב-Paper/Live. תקציב קנייה חייב להגיב לצמצום אחיד
(RiskManager.REDUCED, regime multiplier), לא להישאר זהה.
"""

from __future__ import annotations

import numpy as np


class _DummyVecNorm:
    def normalize_obs(self, obs):
        return obs


class _DummyModel:
    def predict(self, obs, deterministic=True):
        return np.zeros((1, 3), dtype=float), None


class _DummyDataManager:
    pass


def _build_trader(initial_capital: float = 100_000.0):
    from broker_api import BrokerAPIStub
    from live_trader import LiveTrader
    from risk_manager import RiskManager

    broker = BrokerAPIStub()
    risk_manager = RiskManager(initial_capital=initial_capital)
    return LiveTrader(
        model=_DummyModel(),
        broker=broker,
        data_manager=_DummyDataManager(),
        risk_manager=risk_manager,
        vec_norm=_DummyVecNorm(),
        tickers=["AAPL", "MSFT", "GOOGL"],
        initial_capital=initial_capital,
    )


def _dollars_spent(order_events: list[dict]) -> float:
    return sum(
        float(e.get("shares", 0)) * float(e.get("price", 0))
        for e in order_events
        if e.get("side") == "BUY" or e.get("event_type") == "buy_signal"
    )


class TestBudgetRespondsToUniformScaling:

    def test_scaling_all_buy_signals_down_reduces_dollars_spent(self):
        """Small enough conviction/capital that the concentration and
        single-order-notional caps don't bind in either case, so this
        isolates the proportional-split formula itself."""
        trader = _build_trader(initial_capital=100_000.0)
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
        positions = {"AAPL": 0.0, "MSFT": 0.0, "GOOGL": 0.0}

        full_action = np.array([0.15, 0.15, 0.15], dtype=np.float32)
        events_full = trader._execute_actions(full_action, prices, 100_000.0, dict(positions))
        spent_full = _dollars_spent(events_full)

        scaled_action = full_action * 0.5  # e.g. RiskManager.REDUCED
        events_scaled = trader._execute_actions(scaled_action, prices, 100_000.0, dict(positions))
        spent_scaled = _dollars_spent(events_scaled)

        assert spent_full > 0, "sanity: the unscaled case should actually buy something"
        assert spent_scaled < spent_full * 0.75, (
            f"Scaling all buy signals by 0.5x should meaningfully reduce dollars spent "
            f"(full=${spent_full:.2f}, scaled=${spent_scaled:.2f}) — if these are close, "
            f"the cancellation bug is back."
        )
