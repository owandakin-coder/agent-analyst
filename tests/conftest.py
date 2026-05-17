"""
conftest.py
===========
Fixtures משותפים לכל קבצי הבדיקות.
"""

from __future__ import annotations
import sys, os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

# ── ודא שהשורש של הפרויקט נמצא ב-sys.path ──────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─── נתוני שוק סינתטיים ──────────────────────────────────────────────────────

def _make_price_series(n: int, start: float = 150.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices  = start * np.cumprod(1 + returns)
    return pd.Series(prices)


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    close  = _make_price_series(n, seed=seed).values          # numpy array
    high   = close * rng.uniform(1.001, 1.02, n)
    low    = close * rng.uniform(0.98, 0.999, n)
    open_  = np.concatenate([[close[0]], close[:-1]])          # prev close
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)
    dates  = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def raw_ohlcv():
    """DataFrame OHLCV גולמי (120 ימים)."""
    return _make_ohlcv(120)


@pytest.fixture
def featured_df():
    """
    DataFrame עם פיצ'רים מחושבים (כפי שמחזיר DataManager).
    120 שורות × עמודות פיצ'ר.
    """
    from data_manager import DataManager
    raw = _make_ohlcv(120)
    dm  = DataManager.__new__(DataManager)
    return dm._compute_features(raw, "TEST")


@pytest.fixture
def multi_featured(featured_df):
    """
    dict של 3 מניות עם אותה DataFrame (לסביבת Gym ו-LiveTrader).
    """
    return {"AAPL": featured_df.copy(), "MSFT": featured_df.copy(), "GOOGL": featured_df.copy()}


# ─── Alpaca mock objects ──────────────────────────────────────────────────────

@pytest.fixture
def mock_alpaca_account():
    """Mock של TradingClient.get_account()."""
    acc = MagicMock()
    acc.cash            = "50000.00"
    acc.equity          = "100000.00"
    acc.buying_power    = "50000.00"
    acc.portfolio_value = "100000.00"
    return acc


@pytest.fixture
def mock_alpaca_order():
    """Mock של פקודה שהוחזרה מ-Alpaca."""
    order = MagicMock()
    order.id     = "order-uuid-1234"
    order.status = "accepted"
    return order


@pytest.fixture
def mock_alpaca_position(ticker="AAPL", qty="10"):
    pos = MagicMock()
    pos.symbol = ticker
    pos.qty    = qty
    return pos


@pytest.fixture
def mock_alpaca_clock_open():
    clock = MagicMock()
    clock.is_open   = True
    clock.next_open = datetime(2025, 1, 7, 14, 30, tzinfo=timezone.utc)
    return clock


@pytest.fixture
def mock_alpaca_clock_closed():
    clock = MagicMock()
    clock.is_open   = False
    clock.next_open = datetime(2025, 1, 7, 14, 30, tzinfo=timezone.utc)
    return clock


# ─── Broker fixture (כבר מאותחל עם credentials מדומים) ──────────────────────

@pytest.fixture
def broker(mock_alpaca_account, mock_alpaca_order, tmp_path, monkeypatch):
    """
    AlpacaBrokerAPI עם TradingClient מדומה לחלוטין.
    לא מבצע שום קריאת רשת אמיתית.
    """
    monkeypatch.setenv("ALPACA_API_KEY",    "FAKE_KEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_SECRET")

    trading_mock = MagicMock()
    trading_mock.get_account.return_value  = mock_alpaca_account
    trading_mock.submit_order.return_value = mock_alpaca_order
    trading_mock.get_all_positions.return_value = []
    trading_mock.get_clock.return_value    = MagicMock(is_open=True)

    data_mock = MagicMock()

    # הפנה את לוג הפקודות לתיקייה זמנית
    import broker_api as ba
    monkeypatch.setattr(ba, "LOG_FILE", str(tmp_path / "paper_orders.log"))

    with patch("broker_api.TradingClient", return_value=trading_mock), \
         patch("broker_api.StockHistoricalDataClient", return_value=data_mock):

        from broker_api import AlpacaBrokerAPI
        b = AlpacaBrokerAPI(paper=True, auto_approve=True)
        b._trading = trading_mock
        b._data    = data_mock

    return b


# ─── Risk Manager fixture ─────────────────────────────────────────────────────

@pytest.fixture
def risk_manager():
    from risk_manager import RiskManager
    return RiskManager(initial_capital=100_000.0)


# ─── מודל מאומן מינימלי (PPO על סביבה מינימלית) ──────────────────────────────

@pytest.fixture(scope="session")
def tiny_model_and_norm(tmp_path_factory):
    """
    מאמן PPO מינימלי (2K צעדים) ומחזיר (model, vec_norm).
    scope=session → נוצר פעם אחת לכל ריצת הבדיקות.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from data_manager      import DataManager
    from trading_env       import TradingEnvironment
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # נתונים סינתטיים מספיקים לאימון
    n = 200
    raw_data = {}
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        raw = _make_ohlcv(n, seed=ord(ticker[0]))
        dm  = DataManager.__new__(DataManager)
        raw_data[ticker] = dm._compute_features(raw, ticker)

    env_fn   = lambda: TradingEnvironment(raw_data)
    vec_env  = DummyVecEnv([env_fn])
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_norm, verbose=0,
                n_steps=64, batch_size=32, n_epochs=2)
    model.learn(total_timesteps=2_000)

    tmp = tmp_path_factory.mktemp("model")
    model.save(str(tmp / "test_model"))
    vec_norm.save(str(tmp / "vec_norm.pkl"))

    vec_norm.training    = False
    vec_norm.norm_reward = False

    return model, vec_norm, raw_data, tmp
