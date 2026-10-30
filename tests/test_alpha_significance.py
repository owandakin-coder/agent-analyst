"""alpha_significance: block-bootstrap of daily excess return vs SPY (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_significance import bootstrap_excess, daily_returns


def _noise(n=400, seed=1, scale=0.01):
    return np.random.default_rng(seed).normal(0.0004, scale, n)


def test_daily_returns_from_equity():
    r = daily_returns([100.0, 110.0, 99.0])
    assert r == pytest.approx([0.10, -0.10])


def test_identical_series_has_zero_excess_and_ci_around_zero():
    s = _noise()
    r = bootstrap_excess(s, s.copy())
    assert r["annualised_excess"] == pytest.approx(0.0, abs=1e-12)
    assert r["ci95_low"] <= 0.0 <= r["ci95_high"]


def test_constant_outperformance_is_detected():
    s = _noise()
    r = bootstrap_excess(s + 0.001, s)
    assert r["annualised_excess"] == pytest.approx(0.001 * 252, rel=1e-6)
    assert r["ci95_low"] > 0
    assert r["p_excess_le_0"] == 0.0


def test_noisy_zero_mean_difference_includes_zero():
    rng = np.random.default_rng(5)
    s = _noise(seed=2)
    r = bootstrap_excess(s + rng.normal(0.0, 0.004, len(s)), s)
    assert r["ci95_low"] < 0 < r["ci95_high"]


def test_is_deterministic_for_a_seed():
    s, m = _noise(seed=3), _noise(seed=4)
    assert bootstrap_excess(m, s, seed=7) == bootstrap_excess(m, s, seed=7)


def test_too_few_days_is_rejected():
    with pytest.raises(ValueError):
        bootstrap_excess(np.zeros(30), np.zeros(30), block=20)
