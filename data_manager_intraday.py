"""
data_manager_intraday.py
========================
Fetches 1-hour OHLCV bars from Alpaca and computes intraday features.

Why intraday?
-------------
Daily models see one candle per day — they miss intraday momentum, gap-ups,
and volume surges that often precede multi-day moves.  A 1h model trained on
~78 candles (≈ 2 trading weeks × 6.5 h/day) captures these dynamics while
still being tractable to train and run on free infra.

Data source: Alpaca Markets Historical Data API (free tier, up to 5 years
of 1h bars for US equities).

Features per bar
----------------
  returns, log_returns,
  price_to_vwap,             # distance from intraday VWAP
  rsi,                       # 14-bar RSI
  macd_hist,                 # MACD histogram
  boll_pct, boll_width,      # Bollinger Band position and width
  atr_pct,                   # ATR as % of price
  volume_ratio,              # bar volume / 20-bar avg volume
  hour_sin, hour_cos,        # cyclical encoding of hour-of-day
"""

from __future__ import annotations

import os
import logging
import pickle
from datetime import date, timedelta, datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("IntradayDM")

CACHE_DIR    = "cache_1h"
WINDOW_SIZE  = 78   # ~2 trading weeks of 1h bars
MIN_BARS     = 500  # minimum bars required after dropna


def _ta(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean helper."""
    return series.rolling(window).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes intraday TA features on a 1h OHLCV dataframe."""
    c = df["close"]
    v = df["volume"]

    # Basic returns
    df["returns"]     = c.pct_change()
    df["log_returns"] = np.log(c / c.shift(1))

    # VWAP proxy (rolling 20-bar)
    typical = (df["high"] + df["low"] + c) / 3
    df["price_to_vwap"] = c / (typical.rolling(20).mean() + 1e-9) - 1

    # RSI
    df["rsi"] = _compute_rsi(c) / 100  # normalise to [0,1]

    # MACD histogram (12/26/9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - signal) / (c + 1e-9)

    # Bollinger Bands (20-bar)
    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std() + 1e-9
    df["boll_pct"]   = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)
    df["boll_width"] = 4 * std20 / (ma20 + 1e-9)

    # ATR
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - c.shift(1)).abs()
    lpc = (df["low"]  - c.shift(1)).abs()
    atr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    df["atr_pct"] = atr / (c + 1e-9)

    # Volume ratio
    df["volume_ratio"] = v / (v.rolling(20).mean() + 1e-9)

    # Cyclical time features (hour of day 0–6 for US market 9:30–16:00)
    if hasattr(df.index, "hour"):
        hour_norm = (df.index.hour - 9) / 7      # ≈ 0..1
        df["hour_sin"] = np.sin(2 * np.pi * hour_norm)
        df["hour_cos"] = np.cos(2 * np.pi * hour_norm)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0

    return df.replace([np.inf, -np.inf], np.nan)


FEATURE_COLS_1H = [
    "returns", "log_returns", "price_to_vwap",
    "rsi", "macd_hist", "boll_pct", "boll_width",
    "atr_pct", "volume_ratio", "hour_sin", "hour_cos",
]


class IntradayDataManager:
    """
    Downloads and caches 1-hour bars from Alpaca.

    Parameters
    ----------
    tickers : list[str]
    days_back : int
        How many calendar days of history to fetch (default 365 = ~1 year).
    """

    def __init__(
        self,
        tickers: list[str],
        days_back: int = 365,
    ):
        self.tickers   = tickers
        self.days_back = days_back
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    def load_all(self, force_download: bool = False) -> dict[str, pd.DataFrame]:
        """Returns {ticker: DataFrame} with computed features."""
        data = {}
        for ticker in self.tickers:
            df = self._load_ticker(ticker, force_download)
            if df is not None and len(df) >= MIN_BARS:
                data[ticker] = df
            else:
                log.warning(
                    f"{ticker}: only {len(df) if df is not None else 0} bars "
                    f"(need {MIN_BARS}). Skipping."
                )
        log.info(f"IntradayDM loaded {len(data)}/{len(self.tickers)} tickers.")
        return data

    def get_aligned_data(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Aligns all tickers to their common datetime index."""
        common = None
        for df in data.values():
            common = df.index if common is None else common.intersection(df.index)
        if common is None or len(common) == 0:
            raise ValueError("No common timestamps across tickers.")
        common = common.sort_values()
        return {t: df.loc[common].copy() for t, df in data.items()}

    # ──────────────────────────────────────────────────────────────────────────
    def _load_ticker(
        self, ticker: str, force_download: bool
    ) -> pd.DataFrame | None:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}_1h.pkl")

        if not force_download and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    df = pickle.load(f)
                log.debug(f"{ticker}: loaded from 1h cache ({len(df)} bars).")
                return df
            except Exception:
                pass

        return self._download(ticker, cache_path)

    def _download(self, ticker: str, cache_path: str) -> pd.DataFrame | None:
        """Downloads 1h bars from Alpaca historical data API."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from dotenv import load_dotenv

            load_dotenv()
            api_key    = os.getenv("ALPACA_API_KEY", "")
            secret_key = os.getenv("ALPACA_SECRET_KEY", "")

            client = StockHistoricalDataClient(
                api_key=api_key, secret_key=secret_key
            )

            end_dt   = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=self.days_back)

            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Hour,
                start=start_dt,
                end=end_dt,
                adjustment="all",   # split + dividend adjusted
            )
            bars = client.get_stock_bars(req).df

            if bars.empty:
                log.warning(f"{ticker}: Alpaca returned empty data.")
                return None

            # Alpaca returns MultiIndex (symbol, timestamp) — flatten
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(ticker, level="symbol")

            bars.index = pd.to_datetime(bars.index, utc=True).tz_localize(None)
            bars.columns = [c.lower() for c in bars.columns]

            # Keep only OHLCV
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in bars.columns:
                    log.warning(f"{ticker}: missing column '{col}'.")
                    return None

            df = _compute_features(bars[["open", "high", "low", "close", "volume"]].copy())
            df = df.dropna()

            with open(cache_path, "wb") as f:
                pickle.dump(df, f)

            log.info(f"{ticker}: downloaded {len(df)} 1h bars from Alpaca.")
            return df

        except Exception as exc:
            log.error(f"{ticker}: Alpaca 1h download failed — {exc}")
            return None
