"""
Fetches historical OHLCV data from Binance public API.
No API key required — uses public klines endpoint.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


BINANCE_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR = Path("data_cache")


def generate_synthetic_ohlcv(n_candles: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic BTC-like OHLCV data using geometric Brownian motion.
    Used for testing when Binance API is unavailable.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 8760  # hourly
    mu = 0.5       # annual drift
    sigma = 0.8    # annual volatility (BTC-like)

    prices = [30_000.0]
    for _ in range(n_candles - 1):
        ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal()
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)
    noise = rng.uniform(0.001, 0.005, n_candles)

    df = pd.DataFrame({
        "open":   prices * (1 - noise / 2),
        "high":   prices * (1 + noise),
        "low":    prices * (1 - noise),
        "close":  prices,
        "volume": rng.uniform(100, 2000, n_candles),
    })

    df.index = pd.date_range("2022-01-01", periods=n_candles, freq="1h")
    df.index.name = "timestamp"
    return _add_features(df).dropna()


def fetch_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2022-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Download OHLCV candles from Binance and cache locally.
    Handles pagination automatically (Binance limit = 1000 candles/request).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{interval}_{start}_{end}.parquet"

    if cache_file.exists():
        print(f"[Data] Loading from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"[Data] Fetching {symbol} {interval} from Binance...")

    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)

    all_candles: list[list] = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        try:
            resp = requests.get(BINANCE_URL, params=params, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Data] Binance unavailable ({e}). Using synthetic data.")
            return generate_synthetic_ohlcv()
        candles = resp.json()

        if not candles:
            break

        all_candles.extend(candles)
        start_ts = candles[-1][0] + 1  # next candle after last

    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = _add_features(df)
    df = df.dropna()

    df.to_parquet(cache_file)
    print(f"[Data] Saved {len(df)} candles → {cache_file}")
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as observation features."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Trend
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_12"] = close.ewm(span=12).mean()
    df["ema_26"] = close.ewm(span=26).mean()

    # Momentum
    df["rsi"] = _rsi(close, 14)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volatility
    df["atr"] = _atr(high, low, close, 14)
    df["bb_upper"] = df["sma_20"] + 2 * close.rolling(20).std()
    df["bb_lower"] = df["sma_20"] - 2 * close.rolling(20).std()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]

    # Volume
    df["volume_ma"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_ma"]

    # Price returns
    df["returns_1"] = close.pct_change(1)
    df["returns_3"] = close.pct_change(3)
    df["returns_12"] = close.pct_change(12)

    # Normalize price-based features relative to current price
    df["price_to_sma20"] = close / df["sma_20"] - 1
    df["price_to_sma50"] = close / df["sma_50"] - 1
    df["price_to_bb_upper"] = close / df["bb_upper"] - 1
    df["price_to_bb_lower"] = close / df["bb_lower"] - 1

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# Feature columns used by the environment (order matters — defines obs space)
FEATURE_COLS = [
    "returns_1", "returns_3", "returns_12",
    "rsi",
    "macd_hist",
    "price_to_sma20", "price_to_sma50",
    "price_to_bb_upper", "price_to_bb_lower",
    "bb_width",
    "atr",
    "volume_ratio",
]
