"""
market_context.py
==================
תוספות מידע שמעבר ל-OHLCV הבודד של כל מניה:
  1. VIX היסטורי אמיתי (עם היסטוריה שמורה — לא כמו alternative_data.py
     שרק מביא ערך חי, בלי אפשרות backtest).
  2. פיצ'רים חתכיים (cross-sectional): איך המניה מתפקדת היום *ביחס*
     לשאר האוניברס — לפעמים יש שם אות שאין באינדיקטור הגולמי.

⚠️ כלל ברזל: פיצ'ר חתכי בזמן t מותר להשתמש רק בנתוני זמן t של שאר
המניות (מידע זמין באותו רגע בפועל) — אף פעם לא בנתוני t+1 של אף מניה,
כולל לא של המניה עצמה. זה נבדק ב-tests/test_market_context.py באותה
שיטת "חשב פעמיים, מלא מול חתוך" כמו leakage_check.py.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

CACHE_DIR = "cache"


def fetch_market_context(start: str, end: str, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """VIX היסטורי אמיתי, עם cache — בניגוד ל-alternative_data.fetch_vix_proxy
    שמביא רק את הערך הנוכחי ולא ניתן ל-backtest בכלל."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"VIX_CONTEXT_{start}_{end}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    import yfinance as yf

    raw = yf.download("^VIX", start=start, end=end, progress=False)
    if raw.empty:
        raise ValueError("No VIX data returned from yfinance")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    close = raw["Close"].astype(float)
    context = pd.DataFrame(index=raw.index)
    context["vix_close"] = close
    context["vix_change"] = close.pct_change()
    rolling_mean = close.rolling(60).mean()
    rolling_std = close.rolling(60).std() + 1e-9
    context["vix_zscore"] = (close - rolling_mean) / rolling_std

    with open(cache_file, "wb") as f:
        pickle.dump(context, f)
    return context


def merge_market_context(
    all_data: dict[str, pd.DataFrame], context: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Left-joins VIX context columns onto every ticker's frame by date.
    Values are forward-filled only from *earlier* VIX rows — never backward
    — so a date without a same-day VIX print (rare) never reaches into the
    future for one.
    """
    out = {}
    for ticker, df in all_data.items():
        merged = df.join(context, how="left")
        merged[context.columns] = merged[context.columns].ffill()
        out[ticker] = merged
    return out


CROSS_SECTIONAL_SOURCE_COLS = ["returns", "rsi", "volume_ratio"]


def add_cross_sectional_features(all_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """For each date, ranks every ticker against the others in the universe
    on returns/RSI/volume_ratio — "is this stock unusually strong/weak
    *relative to its peers today*", not just in isolation.

    Only ever compares same-date rows across tickers. No ticker's value at
    t depends on any ticker's value at t+1 or later — see
    tests/test_market_context.py for the truncation-based proof.
    """
    tickers = list(all_data.keys())
    cols = [c for c in CROSS_SECTIONAL_SOURCE_COLS if all(c in all_data[t].columns for t in tickers)]
    if not cols or len(tickers) < 2:
        return all_data

    panels = {c: pd.DataFrame({t: all_data[t][c] for t in tickers}) for c in cols}

    ranks = {}
    zscores = {}
    for c, panel in panels.items():
        ranks[c] = panel.rank(axis=1, pct=True)  # 0..1 within that day's universe
        row_mean = panel.mean(axis=1)
        row_std = panel.std(axis=1) + 1e-9
        zscores[c] = panel.sub(row_mean, axis=0).div(row_std, axis=0)

    out = {}
    for ticker, df in all_data.items():
        merged = df.copy()
        for c in cols:
            merged[f"{c}_xs_rank"] = ranks[c][ticker]
            merged[f"{c}_xs_zscore"] = zscores[c][ticker]
        out[ticker] = merged
    return out
