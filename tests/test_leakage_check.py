"""
test_leakage_check.py
======================
בדיקות ל-leakage_check.py — במיוחד לשלוש הבדיקות שנכתבו/תוקנו:
- check_feature_lookahead: עכשיו בדיקה אמיתית (חישוב כפול + השוואה לפי תאריך),
  לא רק אוטוקורלציה.
- check_normalization_leak: עכשיו משווה obs_rms.count לתקציב אימון ידוע,
  לא מחזירה True תמיד.
- check_target_leakage: לא הייתה קיימת בכלל קודם — reward לא תלוי בעתיד.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from leakage_check import (
    check_feature_lookahead,
    check_normalization_leak,
    check_target_leakage,
    check_train_test_overlap,
)


def _make_ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    high = close * (1 + np.abs(rng.normal(0.005, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0.005, 0.005, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def featured_multi():
    from data_manager import DataManager
    dm = DataManager.__new__(DataManager)
    return {t: dm._compute_features(_make_ohlcv(400, seed=ord(t[0])), t) for t in ("AAPL", "MSFT")}


class TestFeatureLookahead:

    def test_real_features_have_no_lookahead(self, featured_multi):
        # This check rebuilds the standard indicator set from OHLCV and
        # compares full vs truncated computation — it validates
        # data_manager._compute_features itself, not arbitrary extra
        # columns a caller might have added to the dict.
        assert check_feature_lookahead(featured_multi, verbose=False) is True


class TestNormalizationLeak:

    def _write_vec_norm(self, tmp_path, count: float):
        from types import SimpleNamespace

        vec_norm = SimpleNamespace(
            obs_rms=SimpleNamespace(count=count, mean=np.zeros(3), var=np.ones(3))
        )
        norm_path = tmp_path / "vec_normalize.pkl"
        with open(norm_path, "wb") as f:
            pickle.dump(vec_norm, f)

    def test_count_within_budget_passes(self, tmp_path, monkeypatch):
        import config_loader

        self._write_vec_norm(tmp_path, count=500_100.0)
        monkeypatch.setitem(config_loader.CFG._data["paths"], "models", str(tmp_path))
        monkeypatch.setitem(config_loader.CFG._data["training"], "ensemble_timesteps", 500_000)

        assert check_normalization_leak() is True

    def test_count_far_above_budget_flagged(self, tmp_path, monkeypatch):
        import config_loader

        self._write_vec_norm(tmp_path, count=2_000_000.0)  # way more than the training budget
        monkeypatch.setitem(config_loader.CFG._data["paths"], "models", str(tmp_path))
        monkeypatch.setitem(config_loader.CFG._data["training"], "ensemble_timesteps", 500_000)

        assert check_normalization_leak() is False


class TestTargetLeakage:

    def test_real_reward_has_no_target_leakage(self, featured_multi):
        assert check_target_leakage(featured_multi, verbose=False) is True


class TestTrainTestOverlap:

    def test_no_overlap_when_windows_are_disjoint(self, featured_multi, monkeypatch):
        import config_loader
        idx = featured_multi["AAPL"].index
        split = len(idx) // 2
        periods = config_loader.CFG._data["periods"]
        monkeypatch.setitem(periods, "train_start", str(idx[0].date()))
        monkeypatch.setitem(periods, "train_end", str(idx[split - 5].date()))
        monkeypatch.setitem(periods, "test_start", str(idx[split].date()))
        monkeypatch.setitem(periods, "test_end", str(idx[-1].date()))
        assert check_train_test_overlap(featured_multi) is True

    def test_overlap_detected_when_windows_share_dates(self, featured_multi, monkeypatch):
        import config_loader
        idx = featured_multi["AAPL"].index
        periods = config_loader.CFG._data["periods"]
        monkeypatch.setitem(periods, "train_start", str(idx[0].date()))
        monkeypatch.setitem(periods, "train_end", str(idx[-1].date()))
        monkeypatch.setitem(periods, "test_start", str(idx[0].date()))
        monkeypatch.setitem(periods, "test_end", str(idx[-1].date()))
        assert check_train_test_overlap(featured_multi) is False
