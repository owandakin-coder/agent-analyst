from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def test_broker_rejects_buy_when_cash_guard_would_be_breached(broker):
    broker._trading.get_account.return_value = MagicMock(
        cash="1000",
        equity="1000",
        buying_power="1000",
        portfolio_value="1000",
        status="ACTIVE",
    )
    broker._trading.get_all_positions.return_value = []

    result = broker.buy("AAPL", shares=10, price=100.0)

    assert result["status"] == "REJECTED"
    assert result["reason"] == "insufficient_cash"
    broker._trading.submit_order.assert_not_called()


class _DummyModel:
    def predict(self, obs, deterministic=True):
        return np.zeros((1, 3), dtype=float), None


class _DummyVecNorm:
    def normalize_obs(self, obs):
        return obs


class _DummyDataManager:
    def load_all(self, force_download=True):
        return {}


class _ConservativeStubBroker:
    def __init__(self, cash: float, positions: dict[str, float], position_details: dict[str, dict] | None = None):
        self.cash = cash
        self.positions = dict(positions)
        self.position_details = position_details or {
            ticker: {"qty": qty, "unrealized_pl": 0.0}
            for ticker, qty in positions.items()
        }
        self.sell_calls: list[tuple[str, int, float]] = []
        self.buy_calls: list[tuple[str, int, float]] = []

    def sell(self, ticker: str, shares: int, price: float):
        held = self.positions.get(ticker, 0.0)
        sold = min(int(shares), int(held))
        if sold <= 0:
            return {"status": "REJECTED", "reason": "no_position"}
        self.positions[ticker] = held - sold
        self.cash += sold * price
        self.sell_calls.append((ticker, sold, price))
        return {"status": "FILLED_STUB", "ticker": ticker, "shares": sold, "side": "SELL", "price": price}

    def buy(self, ticker: str, shares: int, price: float):
        self.buy_calls.append((ticker, shares, price))
        self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
        self.cash -= shares * price
        return {"status": "FILLED_STUB", "ticker": ticker, "shares": shares, "side": "BUY", "price": price}

    def get_position_details(self):
        return self.position_details


def _make_trader(broker):
    from live_trader import LiveTrader
    from risk_manager import RiskManager

    trader = LiveTrader(
        model=_DummyModel(),
        broker=broker,
        data_manager=_DummyDataManager(),
        risk_manager=RiskManager(initial_capital=100_000.0),
        vec_norm=_DummyVecNorm(),
        tickers=["AAPL", "MSFT", "GOOGL"],
        initial_capital=100_000.0,
    )
    trader._telegram = MagicMock()
    return trader


def test_execute_actions_skips_new_buys_when_reserved_cash_not_available():
    broker = _ConservativeStubBroker(cash=600.0, positions={})
    trader = _make_trader(broker)

    orders = trader._execute_actions(
        np.array([1.0, 0.0, 0.0]),
        {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0},
        600.0,
        {},
    )

    assert orders == []
    assert broker.buy_calls == []


def test_execute_actions_auto_deleverages_negative_cash_and_high_exposure():
    broker = _ConservativeStubBroker(
        cash=-1000.0,
        positions={"AAPL": 100.0, "MSFT": 50.0},
        position_details={
            "AAPL": {"qty": 100.0, "unrealized_pl": -1200.0},
            "MSFT": {"qty": 50.0, "unrealized_pl": -300.0},
        },
    )
    trader = _make_trader(broker)

    orders = trader._execute_actions(
        np.array([0.0, 0.0, 0.0]),
        {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0},
        -1000.0,
        {"AAPL": 100.0, "MSFT": 50.0},
    )

    assert any(order.get("event_type") == "auto_deleverage" for order in orders)
    assert broker.sell_calls
    assert broker.cash > -1000.0
