"""
test_observation.py
===================
בדיקות ל-_build_observation() ב-LiveTrader:
- צורת מטריצה נכונה
- ערכים נרמלים (z-score בתוך חלון)
- ללא NaN / Inf
- תאימות לסביבת האימון
"""

import numpy as np
import pandas as pd
import pytest

from live_trader import WINDOW_SIZE, FEATURE_COLS


# ── fixtures מקומיים ──────────────────────────────────────────────────────────

def _make_trader(multi_featured, tiny_model_and_norm):
    """מחזיר LiveTrader מאותחל עם stub broker."""
    model, vec_norm, _, _ = tiny_model_and_norm
    from live_trader  import LiveTrader
    from risk_manager import RiskManager
    from broker_api   import BrokerAPIStub
    from data_manager import DataManager

    broker   = BrokerAPIStub()
    broker.set_cash(100_000.0)
    risk_mgr = RiskManager(100_000.0)
    dm       = DataManager.__new__(DataManager)

    return LiveTrader(
        model=model, broker=broker, data_manager=dm,
        risk_manager=risk_mgr, vec_norm=vec_norm,
        tickers=["AAPL", "MSFT", "GOOGL"],
        initial_capital=100_000.0,
    )


class TestObservationShape:

    def test_output_is_ndarray(self, multi_featured, tiny_model_and_norm):
        trader = _make_trader(multi_featured, tiny_model_and_norm)
        obs    = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        assert isinstance(obs, np.ndarray)

    def test_shape_rows_equals_window_size(self, multi_featured, tiny_model_and_norm):
        """מספר שורות = WINDOW_SIZE."""
        trader = _make_trader(multi_featured, tiny_model_and_norm)
        obs    = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        assert obs.shape[0] == WINDOW_SIZE

    def test_shape_cols(self, multi_featured, tiny_model_and_norm):
        """
        מספר עמודות = num_stocks * num_features + 3 (portfolio state).
        """
        trader      = _make_trader(multi_featured, tiny_model_and_norm)
        num_stocks  = 3
        num_feats   = len([c for c in FEATURE_COLS if c in multi_featured["AAPL"].columns])
        expected_cols = num_stocks * num_feats + 3

        obs = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        assert obs.shape[1] == expected_cols

    def test_dtype_is_float32(self, multi_featured, tiny_model_and_norm):
        """dtype = float32 (כמו בסביבת האימון)."""
        trader = _make_trader(multi_featured, tiny_model_and_norm)
        obs    = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        assert obs.dtype == np.float32


class TestObservationValues:

    def test_no_nan(self, multi_featured, tiny_model_and_norm):
        """אין NaN בתצפית."""
        trader = _make_trader(multi_featured, tiny_model_and_norm)
        obs    = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        assert not np.isnan(obs).any(), "NaN values detected in observation"

    def test_no_inf(self, multi_featured, tiny_model_and_norm):
        """אין Inf בתצפית."""
        trader = _make_trader(multi_featured, tiny_model_and_norm)
        obs    = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        assert not np.isinf(obs).any(), "Inf values detected in observation"

    def test_stock_features_are_normalized(self, multi_featured, tiny_model_and_norm):
        """
        עמודות פיצ'רי מניה (לא portfolio) עוברות z-score בתוך החלון.
        הממוצע אמור להיות קרוב ל-0 (±רווח).
        """
        trader     = _make_trader(multi_featured, tiny_model_and_norm)
        num_feats  = len([c for c in FEATURE_COLS if c in multi_featured["AAPL"].columns])
        obs        = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        # בדוק רק את עמודות הפיצ'רים (לא 3 עמודות ה-portfolio בסוף)
        feature_cols = obs[:, :-3]
        col_means    = feature_cols.mean(axis=0)
        # לאחר z-score, ממוצע בתוך חלון ≈ 0
        assert np.abs(col_means).max() < 2.0, \
            f"Feature means too large: {col_means.max():.3f}"

    def test_portfolio_features_in_correct_range(self, multi_featured, tiny_model_and_norm):
        """
        3 עמודות portfolio: cash_ratio ∈ [0,1], pnl ∈ [-1,∞), drawdown ∈ [0,1].
        """
        trader = _make_trader(multi_featured, tiny_model_and_norm)
        obs    = trader._build_observation(
            multi_featured, cash=30_000.0, net_worth=100_000.0, drawdown=0.05
        )
        # כל שורות ה-portfolio שוות (tile)
        portfolio = obs[:, -3:]
        cash_ratio = portfolio[0, 0]
        drawdown   = portfolio[0, 2]

        assert 0.0 <= cash_ratio <= 1.0,  f"cash_ratio={cash_ratio}"
        assert 0.0 <= drawdown  <= 1.0,   f"drawdown={drawdown}"

    def test_insufficient_data_returns_none(self, tiny_model_and_norm):
        """פחות מ-WINDOW_SIZE שורות → מחזיר None (לא קריסה)."""
        model, vec_norm, _, _ = tiny_model_and_norm
        from live_trader  import LiveTrader
        from risk_manager import RiskManager
        from broker_api   import BrokerAPIStub
        from data_manager import DataManager

        broker   = BrokerAPIStub()
        risk_mgr = RiskManager(100_000.0)
        dm       = DataManager.__new__(DataManager)

        trader = LiveTrader(
            model=model, broker=broker, data_manager=dm,
            risk_manager=risk_mgr, vec_norm=vec_norm,
            tickers=["AAPL"], initial_capital=100_000.0,
        )

        # רק 5 שורות (< WINDOW_SIZE=30)
        short_data = {"AAPL": pd.DataFrame({"returns": range(5)})}
        result = trader._build_observation(
            short_data, cash=100.0, net_worth=100.0, drawdown=0.0
        )
        assert result is None

    def test_matches_trading_env_observation(self, multi_featured, tiny_model_and_norm):
        """
        הצורה חייבת להתאים ל-observation_space של TradingEnvironment.
        """
        from trading_env import TradingEnvironment

        env = TradingEnvironment(multi_featured)
        obs_space_shape = env.observation_space.shape  # (window_size, width)

        model, vec_norm, _, _ = tiny_model_and_norm
        from live_trader  import LiveTrader
        from risk_manager import RiskManager
        from broker_api   import BrokerAPIStub
        from data_manager import DataManager

        broker   = BrokerAPIStub()
        risk_mgr = RiskManager(100_000.0)
        dm       = DataManager.__new__(DataManager)
        trader   = LiveTrader(
            model=model, broker=broker, data_manager=dm,
            risk_manager=risk_mgr, vec_norm=vec_norm,
            tickers=["AAPL", "MSFT", "GOOGL"], initial_capital=100_000.0,
        )
        obs = trader._build_observation(
            multi_featured, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        assert obs.shape == obs_space_shape, (
            f"LiveTrader obs shape {obs.shape} != "
            f"TradingEnv obs_space {obs_space_shape}"
        )
