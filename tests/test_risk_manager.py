"""
test_risk_manager.py
====================
בדיקות ל-RiskManager: רמות סיכון, scale_action, עצירה אוטומטית.
"""

import numpy as np
import pytest
from risk_manager import RiskManager, RiskLevel

INITIAL = 100_000.0


class TestRiskLevels:

    def test_initial_level_is_normal(self, risk_manager):
        """מצב התחלתי = NORMAL."""
        assert risk_manager.risk_level == RiskLevel.NORMAL

    def test_normal_below_10_pct(self, risk_manager):
        """drawdown < 10% → NORMAL."""
        risk_manager.update(INITIAL * 0.92)   # 8% drawdown
        assert risk_manager.risk_level == RiskLevel.NORMAL

    def test_reduced_at_10_pct(self, risk_manager):
        """drawdown >= 10% → REDUCED."""
        risk_manager.update(INITIAL * 0.90)   # 10% drawdown
        assert risk_manager.risk_level == RiskLevel.REDUCED

    def test_reduced_between_10_15_pct(self, risk_manager):
        """10% <= drawdown < 15% → REDUCED."""
        risk_manager.update(INITIAL * 0.87)   # 13%
        assert risk_manager.risk_level == RiskLevel.REDUCED

    def test_halted_at_15_pct(self, risk_manager):
        """drawdown >= 15% → HALTED."""
        risk_manager.update(INITIAL * 0.85)   # 15%
        assert risk_manager.risk_level == RiskLevel.HALTED

    def test_halted_above_15_pct(self, risk_manager):
        """drawdown > 15% → HALTED."""
        risk_manager.update(INITIAL * 0.70)   # 30%
        assert risk_manager.risk_level == RiskLevel.HALTED

    def test_recovery_to_normal(self, risk_manager):
        """אחרי REDUCED ועלייה חזרה → NORMAL."""
        risk_manager.update(INITIAL * 0.88)   # → REDUCED
        assert risk_manager.risk_level == RiskLevel.REDUCED
        risk_manager.peak_value = INITIAL * 0.88  # סמלק peak לערך הנוכחי
        risk_manager.update(INITIAL * 0.88)        # drawdown=0% → NORMAL
        assert risk_manager.risk_level == RiskLevel.NORMAL

    def test_is_halted_property(self, risk_manager):
        """is_halted נכון/שגוי לפי המצב."""
        assert risk_manager.is_halted is False
        risk_manager.update(INITIAL * 0.84)
        assert risk_manager.is_halted is True


class TestScaleAction:

    def test_normal_passes_unchanged(self, risk_manager):
        """NORMAL → action ללא שינוי."""
        action = np.array([0.8, -0.3, 0.5])
        scaled = risk_manager.scale_action(action)
        np.testing.assert_array_almost_equal(scaled, action)

    def test_reduced_halves_buy_signals(self, risk_manager):
        """REDUCED → רק ערכים חיוביים (קנייה) מוכפלים ב-0.5."""
        risk_manager.update(INITIAL * 0.88)   # → REDUCED
        action = np.array([0.8, -0.4, 0.6])
        scaled = risk_manager.scale_action(action)

        assert scaled[0] == pytest.approx(0.4)   # 0.8 * 0.5
        assert scaled[1] == pytest.approx(-0.4)  # מכירה לא מוגבלת
        assert scaled[2] == pytest.approx(0.3)   # 0.6 * 0.5

    def test_halted_returns_zeros(self, risk_manager):
        """HALTED → כל הפעולות אפס (אין מסחר)."""
        risk_manager.update(INITIAL * 0.84)   # → HALTED
        action = np.array([0.9, -0.5, 0.7])
        scaled = risk_manager.scale_action(action)
        np.testing.assert_array_equal(scaled, np.zeros(3))

    def test_scale_called_before_broker(self):
        """
        LiveTrader מפעיל scale_action לפני שליחה לברוקר.
        בדיקה: סוכן שנתן פקודה גדולה → קטנה ב-REDUCED.
        """
        from live_trader  import LiveTrader
        from broker_api   import BrokerAPIStub
        from data_manager import DataManager

        stub = BrokerAPIStub()
        stub.set_cash(100_000.0)

        rm = RiskManager(100_000.0)
        rm.update(90_000.0)   # → REDUCED (10%)

        from data_manager import DataManager as DM

        dm  = DM.__new__(DM)

        class FakeModel:
            def predict(self, obs, deterministic=True):
                return np.array([[1.0, 1.0, 1.0]]), None

        class FakeNorm:
            def normalize_obs(self, obs):
                return obs

        trader = LiveTrader(
            model=FakeModel(), broker=stub, data_manager=dm,
            risk_manager=rm, vec_norm=FakeNorm(),
            tickers=["AAPL", "MSFT", "GOOGL"],
            initial_capital=100_000.0,
        )

        action_raw = np.array([1.0, 1.0, 1.0])
        scaled     = rm.scale_action(action_raw)

        # כל קנייה חתוכה ל-50%
        assert all(scaled[scaled > 0] <= 0.5 + 1e-9)

    def test_peak_value_tracks_maximum(self):
        """peak_value לא יורד אחרי ירידת מחיר."""
        rm = RiskManager(100_000.0)
        rm.update(120_000.0)   # עלייה
        rm.update(90_000.0)    # ירידה
        assert rm.peak_value == 120_000.0

    def test_drawdown_computed_from_peak(self):
        """drawdown מחושב מהשיא, לא מההון ההתחלתי."""
        rm = RiskManager(100_000.0)
        rm.update(120_000.0)   # peak = 120K
        rm.update(102_000.0)   # drawdown = (120K - 102K) / 120K = 15%
        assert rm.risk_level == RiskLevel.HALTED
