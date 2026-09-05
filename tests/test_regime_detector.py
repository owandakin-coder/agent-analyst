"""
test_regime_detector.py
========================
בדיקות ל-RegimeDetector.detect(vix_value=...):
- ברירת מחדל (בלי VIX) — מתנהג בדיוק כמו קודם (proxy ממחיר).
- VIX אמיתי גובר על ה-proxy, ומשנה את הסיווג בפועל (לא רק את המספר המדווח).
- VIX לא תקין (None/0/שלילי) → נופל חזרה ל-proxy במקום לקרוס.

וגם ל-AlternativeDataFetcher.fetch_vix_raw()'s cache: לא קורא לרשת שוב
בתוך חלון ה-cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_detector import RegimeDetector, Regime


def _calm_benchmark(n: int = 260, start: float = 100.0) -> pd.DataFrame:
    """Low, steady volatility (well under vol_high_threshold=0.28) with a
    mild uptrend, so the price-derived vol_20 proxy alone would classify
    this as TRENDING_UP, not HIGH_VOLATILITY."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0006, 0.004, n)  # ~6% annualized vol — very calm
    close = start * np.cumprod(1 + returns)
    return pd.DataFrame({"close": close})


class TestVixOverride:

    def test_no_vix_matches_previous_price_based_behavior(self):
        df = _calm_benchmark()
        detector = RegimeDetector()
        signal = detector.detect(df)
        assert signal.regime != Regime.HIGH_VOLATILITY  # calm price series, no VIX given

    def test_real_vix_overrides_the_calm_price_proxy(self):
        """The core validated behavior: a genuinely volatile VIX print
        should flip the regime even when the benchmark's own recent price
        action looks calm — this is exactly the gap VIX was added to close."""
        df = _calm_benchmark()
        detector = RegimeDetector()
        signal = detector.detect(df, vix_value=45.0)  # 45 -> 0.45 vol_20, crisis-level
        assert signal.regime == Regime.HIGH_VOLATILITY
        assert signal.vol_20 == pytest.approx(0.45)

    def test_low_vix_does_not_falsely_trigger_high_volatility(self):
        df = _calm_benchmark()
        detector = RegimeDetector()
        signal = detector.detect(df, vix_value=12.0)  # 12 -> 0.12, calm
        assert signal.regime != Regime.HIGH_VOLATILITY
        assert signal.vol_20 == pytest.approx(0.12)

    def test_none_vix_falls_back_to_price_proxy(self):
        df = _calm_benchmark()
        detector = RegimeDetector()
        with_none = detector.detect(df, vix_value=None)
        without_arg = detector.detect(df)
        assert with_none.vol_20 == pytest.approx(without_arg.vol_20)

    def test_invalid_vix_falls_back_instead_of_crashing(self):
        df = _calm_benchmark()
        detector = RegimeDetector()
        baseline = detector.detect(df)
        for bad_value in (0.0, -5.0):
            signal = detector.detect(df, vix_value=bad_value)
            assert signal.vol_20 == pytest.approx(baseline.vol_20)


class TestVixCache:

    def test_fetch_vix_raw_uses_cache_within_ttl(self, monkeypatch):
        from alternative_data import AlternativeDataFetcher

        fetcher = AlternativeDataFetcher()
        calls = {"n": 0}

        def fake_download(*args, **kwargs):
            calls["n"] += 1
            df = pd.DataFrame({"Close": [21.5]})
            return df

        monkeypatch.setitem(__import__("sys").modules, "yfinance",
                             type("YF", (), {"download": staticmethod(fake_download)}))

        first = fetcher.fetch_vix_raw()
        second = fetcher.fetch_vix_raw()

        assert first == pytest.approx(21.5)
        assert second == pytest.approx(21.5)
        assert calls["n"] == 1  # second call served from cache, no new network hit
