"""Safety regressions for ambiguous receipts, partial fills and stale inputs."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from order_evidence import confirmed_fill, requires_reconciliation
from tests.test_live_trader_execution import _build_trader


@pytest.mark.parametrize("status", ["accepted", "new", "ERROR", "reconciliation_pending", "OrderStatus.PARTIALLY_FILLED"])
def test_unresolved_receipts_never_imply_a_full_fill(status):
    receipt = {"status": status, "shares": 10}
    assert confirmed_fill(receipt) == (0, 0)
    assert requires_reconciliation(receipt)


@pytest.mark.parametrize("qty,price", [(float("nan"), 100), (3, float("inf")), (11, 100), (-1, 100), (3, 0)])
def test_invalid_fill_evidence_is_not_counted(qty, price):
    assert confirmed_fill({"shares": 10, "filled_qty": qty, "filled_avg_price": price}) == (0, 0)


def test_accepted_stop_keeps_position_protection_and_kelly_unchanged():
    trader = _build_trader()
    trader._entry_prices["AAPL"] = 100
    trader._trailing_highs["AAPL"] = 120
    trader.broker.sell = MagicMock(return_value={"status": "accepted", "shares": 10})
    trader.broker.buy = MagicMock()
    trader.risk_manager.record_trade_outcome = MagicMock()
    held = {"AAPL": 10}
    events = trader._execute_actions(np.ones(3), {"AAPL": 100, "MSFT": 100, "GOOGL": 100}, 100000, held)
    assert len(events) == 1 and held["AAPL"] == 10
    assert trader._entry_prices["AAPL"] == 100 and trader._trailing_highs["AAPL"] == 120
    trader.risk_manager.record_trade_outcome.assert_not_called()
    trader.broker.buy.assert_not_called()


def test_partial_stop_uses_only_confirmed_shares_and_execution_price():
    trader = _build_trader()
    trader._entry_prices["AAPL"] = 100
    trader._trailing_highs["AAPL"] = 120
    trader._telegram = MagicMock()
    trader.broker.sell = MagicMock(return_value={"status": "partially_filled", "shares": 10,
                                                "filled_qty": 3, "filled_avg_price": 90})
    trader.broker.buy = MagicMock()
    trader.risk_manager.record_trade_outcome = MagicMock()
    held = {"AAPL": 10}
    trader._execute_actions(np.ones(3), {"AAPL": 100, "MSFT": 100, "GOOGL": 100}, 100000, held)
    assert held["AAPL"] == 7 and "AAPL" in trader._trailing_highs
    trader.risk_manager.record_trade_outcome.assert_called_once_with("AAPL", -.1)
    trader.broker.buy.assert_not_called()


def test_accepted_buy_does_not_invent_entry_or_place_another_order():
    trader = _build_trader()
    trader.broker.buy = MagicMock(return_value={"status": "accepted", "shares": 40})
    held = {}
    trader._execute_actions(np.ones(3), {t: 100 for t in trader.tickers}, 100000, held)
    trader.broker.buy.assert_called_once()
    assert held == {} and trader._entry_prices == {}


def test_existing_broker_orders_block_new_submissions_across_cycles():
    trader = _build_trader()
    trader.broker.get_open_orders = MagicMock(return_value=[{"symbol": "OTHER", "status": "new"}])
    trader.broker.buy = MagicMock()
    trader._execute_actions(np.ones(3), {t: 100 for t in trader.tickers}, 100000, {})
    trader.broker.buy.assert_not_called()


def test_expensive_whole_share_cannot_exceed_order_budget():
    trader = _build_trader()
    trader.broker.buy = MagicMock()
    trader._execute_actions(np.ones(3), {t: 9000 for t in trader.tickers}, 100000, {})
    trader.broker.buy.assert_not_called()


@pytest.mark.parametrize("actions", [np.array([np.nan, 0, 0]), np.array([2, 0, 0]), np.array([.1])])
def test_invalid_actions_abort_before_any_order(actions):
    trader = _build_trader()
    trader.broker.buy = MagicMock()
    with pytest.raises(ValueError, match="Invalid execution action"):
        trader._execute_actions(actions, {t: 100 for t in trader.tickers}, 100000, {})
    trader.broker.buy.assert_not_called()


def test_quote_expiring_during_inference_blocks_submission(monkeypatch):
    trader = _build_trader()
    trader.broker.buy = MagicMock()
    monkeypatch.setenv("ATZMA_REQUIRE_FRESH_QUOTES", "1")
    monkeypatch.setattr("live_trader.load_control_state", lambda: {})
    monkeypatch.setattr("live_trader.can_trade", lambda _: (True, None))
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    trader._execution_quotes = {t: {"price": 100, "timestamp": old} for t in trader.tickers}
    assert trader._place_order("buy", "AAPL", 1, 100)["reason"] == "stale_quotes_before_submission"
    trader.broker.buy.assert_not_called()


def test_pause_during_inference_blocks_submission(monkeypatch):
    trader = _build_trader()
    trader.broker.buy = MagicMock()
    trader._execution_quotes = {}
    monkeypatch.setattr("live_trader.load_control_state", lambda: {})
    monkeypatch.setattr("live_trader.can_trade", lambda _: (False, "emergency_stop"))
    assert trader._place_order("buy", "AAPL", 1, 100)["reason"] == "emergency_stop"
    trader.broker.buy.assert_not_called()
