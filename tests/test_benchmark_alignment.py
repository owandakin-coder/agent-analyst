"""SPY benchmarks must cover the days the env trades (window_size days after the data starts)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark import spy_buy_hold, trading_window


def _data(n=100):
    idx = pd.date_range("2025-01-02", periods=n, freq="B")
    close = np.linspace(100.0, 100.0 + n - 1, n)
    return {"SPY": pd.DataFrame({"close": close}, index=idx), "AAPL": pd.DataFrame({"close": close}, index=idx)}


def test_trading_window_drops_the_first_window_size_days():
    w = trading_window(_data(100), window_size=30)
    assert all(len(df) == 70 for df in w.values())
    assert w["SPY"]["close"].iloc[0] == 130.0


def test_spy_return_is_measured_from_the_first_traded_day():
    data = _data(100)
    full = spy_buy_hold(data, 100_000.0)
    aligned = spy_buy_hold(trading_window(data, window_size=30), 100_000.0)
    assert len(aligned) == len(full) - 30
    assert aligned[0] == 100_000.0
    assert np.isclose(aligned[-1] / aligned[0] - 1, (199.0 - 130.0) / 130.0)
    assert not np.isclose(aligned[-1] / aligned[0], full[-1] / full[0])
