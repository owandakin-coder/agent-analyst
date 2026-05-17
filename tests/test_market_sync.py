"""
test_market_sync.py
===================
בדיקות לסנכרון שוק: is_market_open(), next_market_open(),
ולוגיקת LiveTrader שלא מנסה לסחור כשהשוק סגור.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest


class TestIsMarketOpen:

    def test_returns_true_when_open(self, broker, mock_alpaca_clock_open):
        broker._trading.get_clock.return_value = mock_alpaca_clock_open
        assert broker.is_market_open() is True

    def test_returns_false_when_closed(self, broker, mock_alpaca_clock_closed):
        broker._trading.get_clock.return_value = mock_alpaca_clock_closed
        assert broker.is_market_open() is False

    def test_returns_bool_not_truthy(self, broker, mock_alpaca_clock_open):
        """ודא שהערך הוא bool אמיתי, לא רק truthy."""
        broker._trading.get_clock.return_value = mock_alpaca_clock_open
        result = broker.is_market_open()
        assert isinstance(result, bool)

    def test_api_error_returns_false(self, broker):
        """כישלון API → False (לא קריסה)."""
        broker._trading.get_clock.side_effect = Exception("API down")
        result = broker.is_market_open()
        assert result is False

    def test_stub_always_open(self):
        """BrokerAPIStub מחזיר תמיד True (לצורך בדיקות)."""
        from broker_api import BrokerAPIStub
        stub = BrokerAPIStub()
        assert stub.is_market_open() is True


class TestNextMarketOpen:

    def test_returns_datetime(self, broker, mock_alpaca_clock_closed):
        broker._trading.get_clock.return_value = mock_alpaca_clock_closed
        result = broker.next_market_open()
        assert isinstance(result, datetime)

    def test_api_error_returns_none(self, broker):
        """כישלון API → None (לא קריסה)."""
        broker._trading.get_clock.side_effect = Exception("timeout")
        result = broker.next_market_open()
        assert result is None


class TestLiveLoopMarketCheck:

    def test_run_once_skipped_when_halted(self, multi_featured, tiny_model_and_norm):
        """
        כשה-RiskManager ב-HALTED, run_once לא שולח פקודות.
        """
        model, vec_norm, _, _ = tiny_model_and_norm
        from live_trader  import LiveTrader
        from risk_manager import RiskManager
        from broker_api   import BrokerAPIStub
        from data_manager import DataManager

        stub     = BrokerAPIStub()
        stub.set_cash(100_000.0)
        risk_mgr = RiskManager(100_000.0)
        risk_mgr.update(80_000.0)   # → HALTED (20%)

        dm = DataManager.__new__(DataManager)
        dm.load_all   = MagicMock(return_value=multi_featured)
        dm.tickers    = ["AAPL", "MSFT", "GOOGL"]
        dm.start = dm.end = ""

        trader = LiveTrader(
            model=model, broker=stub, data_manager=dm,
            risk_manager=risk_mgr, vec_norm=vec_norm,
            tickers=["AAPL", "MSFT", "GOOGL"],
            initial_capital=100_000.0,
        )

        # mock _fetch_fresh_data להחזיר נתוני fixture
        trader._fetch_fresh_data = MagicMock(return_value=multi_featured)
        stub.get_latest_prices   = MagicMock(
            return_value={"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 130.0}
        )
        stub.get_positions       = MagicMock(return_value={})
        stub.get_cash            = MagicMock(return_value=80_000.0)

        orders_before = stub.order_counter
        trader.run_once()
        assert stub.order_counter == orders_before, \
            "HALTED: לא אמורה להישלח שום פקודה"

    def test_loop_polls_until_open(self, broker, mock_alpaca_clock_closed):
        """
        LiveTrader.run_loop ממתין בלולאה כשהשוק סגור.
        בדיקה: אחרי קריאה אחת לקריאה סגורה + KeyboardInterrupt.
        """
        import time
        from live_trader  import LiveTrader
        from risk_manager import RiskManager
        from broker_api   import BrokerAPIStub
        from data_manager import DataManager

        stub     = BrokerAPIStub()
        risk_mgr = RiskManager(100_000.0)
        dm       = DataManager.__new__(DataManager)

        class FakeModel:
            def predict(self, *a, **kw): return [[0.0, 0.0, 0.0]], None

        class FakeNorm:
            obs_rms = MagicMock()
            ret_rms = MagicMock()
            def normalize_obs(self, o): return o

        trader = LiveTrader(
            model=FakeModel(), broker=stub, data_manager=dm,
            risk_manager=risk_mgr, vec_norm=FakeNorm(),
            tickers=["AAPL", "MSFT", "GOOGL"],
            initial_capital=100_000.0,
        )

        calls = []

        def fake_is_open():
            calls.append(1)
            if len(calls) >= 2:
                raise KeyboardInterrupt   # יצא מהלולאה
            return False

        stub.is_market_open    = fake_is_open
        stub.next_market_open  = lambda: None

        with patch("time.sleep"):   # לא ממתין באמת
            trader.run_loop(poll_seconds=1)

        assert len(calls) >= 1, "is_market_open() חייב להיקרא לפחות פעם אחת"
