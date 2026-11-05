"""Bounded, vectorized quantitative research agents; no broker or voting authority.

Agents are parameter variants sharing the same price history, not independent
models or independent evidence. Their counts must never become release votes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ResearchAgentPanel:
    MAX_AGENTS = 10_000
    FAMILIES = ("momentum", "trend_distance", "mean_reversion")

    def __init__(self, count: int = 0):
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.MAX_AGENTS:
            raise ValueError(f"research agent count must be an integer from 0 to {self.MAX_AGENTS}")
        self.count = count
        indexes = np.arange(count)
        self._families = indexes % 3
        # Up to 10,000 distinct family/window/threshold combinations.
        self._windows = 5 + (indexes // 3 % 196)
        self._thresholds = 0.001 * (1 + indexes // (3 * 196))

    def evaluate(self, fresh_data: dict[str, pd.DataFrame]) -> dict:
        results = {}
        if self.count:
            for ticker, frame in fresh_data.items():
                if "close" not in frame or len(frame) < 200:
                    results[ticker] = {"status": "insufficient_history", "evaluated": 0}
                    continue
                close = pd.to_numeric(frame["close"].iloc[-200:], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(close).all() or (close <= 0).any():
                    results[ticker] = {"status": "invalid_history", "evaluated": 0}
                    continue
                cumulative = np.concatenate(([0.0], np.cumsum(close)))
                means = (cumulative[-1] - cumulative[200 - self._windows]) / self._windows
                momentum = close[-1] / close[-self._windows] - 1
                distance = close[-1] / means - 1
                signal = np.where(self._families == 0, momentum,
                                  np.where(self._families == 1, distance, -distance))
                votes = np.where(signal > self._thresholds, 1,
                                 np.where(signal < -self._thresholds, -1, 0))
                results[ticker] = {
                    "status": "evaluated", "evaluated": self.count,
                    "families": {
                        name: {"buy": int(np.sum(votes[self._families == i] == 1)),
                               "hold": int(np.sum(votes[self._families == i] == 0)),
                               "sell": int(np.sum(votes[self._families == i] == -1))}
                        for i, name in enumerate(self.FAMILIES)
                    },
                }
        return {"configured_agents": self.count, "execution_authority": False,
                "scope": "correlated_parameter_variants_research_only", "tickers": results}
