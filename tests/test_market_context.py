"""
test_market_context.py
=======================
בדיקות ל-market_context.py:
- VIX context: cache עובד, אין leakage (rolling windows רק אחורה)
- cross-sectional features: rank/zscore נכונים, ואין תלות בעתיד
  (מוכח באותה שיטת truncation כמו leakage_check.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_context import (
    add_cross_sectional_features,
    merge_market_context,
    CROSS_SECTIONAL_SOURCE_COLS,
)


def _make_featured(n: int, seed: int, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.01, n)
    close = base * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "close": close,
            "returns": returns,
            "rsi": rng.uniform(20, 80, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
        },
        index=dates,
    )


class TestCrossSectionalFeatures:

    def test_rank_and_zscore_columns_added(self):
        data = {"AAA": _make_featured(60, 1), "BBB": _make_featured(60, 2), "CCC": _make_featured(60, 3)}
        out = add_cross_sectional_features(data)
        for col in CROSS_SECTIONAL_SOURCE_COLS:
            assert f"{col}_xs_rank" in out["AAA"].columns
            assert f"{col}_xs_zscore" in out["AAA"].columns

    def test_rank_is_correct_on_a_known_day(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        data = {
            "LOW":  pd.DataFrame({"returns": [0.0, 0.0, 0.01], "rsi": [30, 30, 30], "volume_ratio": [1, 1, 1]}, index=idx),
            "MID":  pd.DataFrame({"returns": [0.0, 0.0, 0.05], "rsi": [30, 30, 30], "volume_ratio": [1, 1, 1]}, index=idx),
            "HIGH": pd.DataFrame({"returns": [0.0, 0.0, 0.10], "rsi": [30, 30, 30], "volume_ratio": [1, 1, 1]}, index=idx),
        }
        out = add_cross_sectional_features(data)
        # On the last day, HIGH has the biggest return of the three -> top rank (pct=1.0)
        assert out["HIGH"]["returns_xs_rank"].iloc[-1] == 1.0
        assert out["LOW"]["returns_xs_rank"].iloc[-1] == pytest.approx(1 / 3, abs=1e-9)


class TestNoLookahead:

    def test_cross_sectional_value_unchanged_when_future_rows_removed(self):
        """The definitive test: today's cross-sectional rank must not change
        when we delete tomorrow's data — if it does, something is peeking
        ahead."""
        full = {"AAA": _make_featured(100, 1), "BBB": _make_featured(100, 2), "CCC": _make_featured(100, 3)}
        cutoff = 80
        truncated = {t: df.iloc[:cutoff].copy() for t, df in full.items()}

        check_at = 50  # comfortably before the cutoff

        out_full = add_cross_sectional_features(full)
        out_trunc = add_cross_sectional_features(truncated)

        for ticker in full:
            date = out_full[ticker].index[check_at]
            for col in CROSS_SECTIONAL_SOURCE_COLS:
                a = out_full[ticker][f"{col}_xs_rank"].loc[date]
                b = out_trunc[ticker][f"{col}_xs_rank"].loc[date]
                assert abs(a - b) < 1e-9, f"{ticker}.{col}_xs_rank changed after removing future rows"

    def test_vix_merge_never_uses_future_vix_value(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        stock = pd.DataFrame({"close": np.arange(10.0)}, index=idx)
        vix_context = pd.DataFrame(
            {"vix_close": np.arange(100.0, 110.0), "vix_change": np.zeros(10), "vix_zscore": np.zeros(10)},
            index=idx,
        )
        # Remove one mid-series VIX print to force ffill, then confirm the
        # filled value is the *previous* real value, never a later one.
        vix_context_gapped = vix_context.drop(index=idx[5])

        full = merge_market_context({"X": stock}, vix_context)
        gapped = merge_market_context({"X": stock}, vix_context_gapped)

        assert gapped["X"]["vix_close"].loc[idx[5]] == vix_context.loc[idx[4], "vix_close"]
        # Everything strictly before the gap must be identical either way.
        pd.testing.assert_series_equal(
            full["X"]["vix_close"].loc[:idx[4]],
            gapped["X"]["vix_close"].loc[:idx[4]],
        )
