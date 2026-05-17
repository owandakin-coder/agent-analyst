"""
test_model.py
=============
בדיקות למודל RL:
- טעינה תקינה מקובץ
- predict() מחזיר צורה נכונה
- ערכים ב-[-1, 1]
- ללא NaN
- עקביות deterministic
"""

import numpy as np
import pytest
from pathlib import Path


class TestModelLoading:

    def test_model_loads_without_error(self, tiny_model_and_norm):
        """המודל נטען בלי חריגה."""
        model, _, _, _ = tiny_model_and_norm
        assert model is not None

    def test_model_file_exists_after_save(self, tiny_model_and_norm):
        """קובץ המודל קיים בתיקייה."""
        _, _, _, tmp = tiny_model_and_norm
        assert (tmp / "test_model.zip").exists()

    def test_vec_normalize_file_exists(self, tiny_model_and_norm):
        """קובץ VecNormalize קיים."""
        _, _, _, tmp = tiny_model_and_norm
        assert (tmp / "vec_norm.pkl").exists()

    def test_load_from_disk(self, tiny_model_and_norm):
        """טעינה ישירה מהדיסק (לא מה-fixture)."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from trading_env import TradingEnvironment

        _, _, raw_data, tmp = tiny_model_and_norm
        model = PPO.load(str(tmp / "test_model"))
        assert model is not None

        dummy   = DummyVecEnv([lambda: TradingEnvironment(raw_data)])
        loaded_norm = VecNormalize.load(str(tmp / "vec_norm.pkl"), dummy)
        assert loaded_norm is not None


class TestModelPrediction:

    def _make_obs(self, model, vec_norm, raw_data):
        """בונה תצפית ממדים נכונים."""
        from live_trader  import LiveTrader, WINDOW_SIZE, FEATURE_COLS
        from risk_manager import RiskManager
        from broker_api   import BrokerAPIStub
        from data_manager import DataManager

        stub     = BrokerAPIStub()
        risk_mgr = RiskManager(100_000.0)
        dm       = DataManager.__new__(DataManager)
        tickers  = list(raw_data.keys())

        trader = LiveTrader(
            model=model, broker=stub, data_manager=dm,
            risk_manager=risk_mgr, vec_norm=vec_norm,
            tickers=tickers, initial_capital=100_000.0,
        )
        return trader._build_observation(
            raw_data, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )

    def test_predict_returns_array(self, tiny_model_and_norm):
        """predict() מחזיר np.ndarray."""
        model, vec_norm, raw_data, _ = tiny_model_and_norm
        obs    = self._make_obs(model, vec_norm, raw_data)
        action, _ = model.predict(obs[np.newaxis], deterministic=True)
        assert isinstance(np.array(action), np.ndarray)

    def test_predict_shape_matches_num_stocks(self, tiny_model_and_norm):
        """פעולה: (num_stocks,) = (3,)."""
        model, vec_norm, raw_data, _ = tiny_model_and_norm
        obs    = self._make_obs(model, vec_norm, raw_data)
        action, _ = model.predict(obs[np.newaxis], deterministic=True)
        action = np.array(action).flatten()
        assert action.shape == (3,), f"Expected (3,), got {action.shape}"

    def test_predict_values_in_range(self, tiny_model_and_norm):
        """ערכי הפעולה ∈ [-1, 1]."""
        model, vec_norm, raw_data, _ = tiny_model_and_norm
        obs    = self._make_obs(model, vec_norm, raw_data)
        action, _ = model.predict(obs[np.newaxis], deterministic=True)
        action = np.array(action).flatten()
        assert action.min() >= -1.0 - 1e-6, f"min={action.min()}"
        assert action.max() <=  1.0 + 1e-6, f"max={action.max()}"

    def test_predict_no_nan(self, tiny_model_and_norm):
        """ללא NaN בפלט."""
        model, vec_norm, raw_data, _ = tiny_model_and_norm
        obs    = self._make_obs(model, vec_norm, raw_data)
        action, _ = model.predict(obs[np.newaxis], deterministic=True)
        assert not np.isnan(np.array(action)).any()

    def test_predict_deterministic_consistent(self, tiny_model_and_norm):
        """deterministic=True → אותה תצפית נותנת אותה פעולה בכל קריאה."""
        model, vec_norm, raw_data, _ = tiny_model_and_norm
        obs = self._make_obs(model, vec_norm, raw_data)
        a1, _ = model.predict(obs[np.newaxis], deterministic=True)
        a2, _ = model.predict(obs[np.newaxis], deterministic=True)
        np.testing.assert_array_equal(np.array(a1), np.array(a2))

    def test_multiple_observations_no_crash(self, tiny_model_and_norm):
        """100 קריאות predict() רצופות ללא קריסה."""
        model, vec_norm, raw_data, _ = tiny_model_and_norm
        obs = self._make_obs(model, vec_norm, raw_data)
        for _ in range(100):
            action, _ = model.predict(obs[np.newaxis], deterministic=True)
            assert not np.isnan(np.array(action)).any()


class TestVecNormalize:

    def test_normalize_obs_preserves_shape(self, tiny_model_and_norm):
        """VecNormalize לא משנה את צורת התצפית."""
        model, vec_norm, raw_data, _ = tiny_model_and_norm
        from live_trader  import LiveTrader
        from risk_manager import RiskManager
        from broker_api   import BrokerAPIStub
        from data_manager import DataManager

        stub     = BrokerAPIStub()
        risk_mgr = RiskManager(100_000.0)
        dm       = DataManager.__new__(DataManager)

        trader = LiveTrader(
            model=model, broker=stub, data_manager=dm,
            risk_manager=risk_mgr, vec_norm=vec_norm,
            tickers=list(raw_data.keys()), initial_capital=100_000.0,
        )
        obs = trader._build_observation(
            raw_data, cash=50_000.0, net_worth=100_000.0, drawdown=0.0
        )
        original_shape = obs.shape
        normalized     = vec_norm.normalize_obs(obs[np.newaxis])
        assert normalized.shape[1:] == original_shape
