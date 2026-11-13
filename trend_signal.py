"""
trend_signal.py
===============
Rules-based action vector used when live.signal_source = "trend".

Why it exists: over 2020-2026 the PPO model's raw action had no predictive power
for forward returns (rank IC about 0, buy signals returned no more than the
average day; see ic_probe.py). What the portfolio
earned came from being long equities, not from picking them. This rule makes
that exposure explicit and removes it when the trend breaks, with no fitted
parameters: the 50/200-day averages and 6-month momentum are the textbook
definitions.

  buy   a stock whose close is above its 200-day average and whose 50-day
        average is above the 200-day: new names ranked by 6-month momentum,
        top_n positions at most; names already held are topped up until the
        per-name concentration cap stops the order
  sell  a held stock whose close drops below its 200-day average
  hold  everything else

Sizing, stops, caps, the Trend/Defense approval and the risk manager still apply
downstream, so this only replaces the model's raw signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMA_SLOW = 200
SMA_FAST = 50
MOMENTUM_DAYS = 126
BUY_ACTION = 0.6


def trend_actions(
    fresh_data: dict[str, pd.DataFrame],
    tickers: list[str],
    held: dict[str, float],
    top_n: int = 8,
) -> np.ndarray:
    action = np.zeros(len(tickers), dtype=float)
    candidates: list[tuple[float, int]] = []
    for i, ticker in enumerate(tickers):
        df = fresh_data.get(ticker)
        if df is None or "close" not in df.columns:
            continue
        close = df["close"].dropna()
        if len(close) < SMA_SLOW + 1:
            continue
        price = float(close.iloc[-1])
        slow = float(close.iloc[-SMA_SLOW:].mean())
        fast = float(close.iloc[-SMA_FAST:].mean())
        is_held = held.get(ticker, 0.0) > 0
        if is_held and price < slow:
            action[i] = -1.0
        elif price > slow and fast > slow and is_held:
            action[i] = BUY_ACTION
        elif price > slow and fast > slow:
            base = float(close.iloc[-MOMENTUM_DAYS - 1]) if len(close) > MOMENTUM_DAYS else float(close.iloc[0])
            candidates.append((price / base - 1.0, i))
    held_on = sum(1 for t in tickers if held.get(t, 0.0) > 0)
    room = max(0, top_n - held_on)
    for _, i in sorted(candidates, reverse=True)[:room]:
        action[i] = BUY_ACTION
    return action
