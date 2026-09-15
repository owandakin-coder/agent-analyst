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


class TestMarginLeverageCeiling:
    """Regression coverage for wiring broker buying_power (margin) into
    order sizing: leverage must actually be deployable, but the hard
    max_gross_exposure ceiling must never be crossed regardless of how much
    buying_power the broker grants or how bullish the action vector is."""

    def test_buying_power_above_cash_is_actually_deployed(self):
        """With no_margin=False, a margin account's buying_power (> cash)
        should be usable, not silently capped at raw cash. Per-order and
        per-ticker concentration caps are widened for this test alone so
        they can't mask the thing actually being isolated: whether
        available_cash is sourced from buying_power or from raw cash."""
        import live_trader as lt

        trader = _build_trader(initial_capital=100_000.0)
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
        positions = {"AAPL": 0.0, "MSFT": 0.0, "GOOGL": 0.0}
        action = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        cash = 20_000.0
        original_no_margin = lt.NO_MARGIN
        original_notional = lt.MAX_SINGLE_ORDER_NOTIONAL_PCT
        original_concentration = lt.MAX_CONCENTRATION
        lt.MAX_SINGLE_ORDER_NOTIONAL_PCT = 1.0
        lt.MAX_CONCENTRATION = 1.0
        try:
            no_margin_events = trader._execute_actions(action, prices, cash, dict(positions), buying_power=cash)
            spent_no_margin = _dollars_spent(no_margin_events)

            lt.NO_MARGIN = False
            margin_events = trader._execute_actions(
                action, prices, cash, dict(positions), buying_power=cash * 2.0
            )
        finally:
            lt.NO_MARGIN = original_no_margin
            lt.MAX_SINGLE_ORDER_NOTIONAL_PCT = original_notional
            lt.MAX_CONCENTRATION = original_concentration
        spent_with_margin = _dollars_spent(margin_events)

        assert spent_with_margin > spent_no_margin, (
            "buying_power above cash should let the trader deploy more capital "
            "under margin mode — if these are equal, buying_power isn't wired in "
            "and leverage can never actually be used."
        )

    def test_gross_exposure_ceiling_holds_even_with_ample_buying_power(self):
        """Even if the broker grants huge buying_power, total deployed
        capital must never exceed net_worth * MAX_GROSS_EXPOSURE — the cap
        is on the account's own equity, not on borrowed buying power."""
        import live_trader as lt

        trader = _build_trader(initial_capital=100_000.0)
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
        positions = {"AAPL": 0.0, "MSFT": 0.0, "GOOGL": 0.0}
        max_conviction = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        cash = 100_000.0

        original = lt.NO_MARGIN
        lt.NO_MARGIN = False
        try:
            events = trader._execute_actions(
                max_conviction, prices, cash, dict(positions), buying_power=cash * 10.0
            )
        finally:
            lt.NO_MARGIN = original

        spent = _dollars_spent(events)
        net_worth = cash
        assert spent <= net_worth * lt.MAX_GROSS_EXPOSURE + 1e-6, (
            f"Deployed ${spent:.2f} against net worth ${net_worth:.2f} — exceeds the "
            f"hard {lt.MAX_GROSS_EXPOSURE}x gross-exposure ceiling even "
            f"though buying_power (${cash * 10.0:.2f}) would have allowed more."
        )
