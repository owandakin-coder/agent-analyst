"""
Market regime detection for ATZMA.

Classifies the market into execution-aware regimes:
- TRENDING_UP
- TRENDING_DOWN
- RANGE_BOUND
- HIGH_VOLATILITY
- CRASH_CORRECTION
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger("RegimeDetector")


class Regime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRASH_CORRECTION = "CRASH_CORRECTION"

    def position_multiplier(self) -> float:
        return {
            Regime.TRENDING_UP: 1.0,
            Regime.RANGE_BOUND: 0.65,
            Regime.TRENDING_DOWN: 0.45,
            Regime.HIGH_VOLATILITY: 0.35,
            Regime.CRASH_CORRECTION: 0.0,
        }[self]

    def strategy_mode(self) -> str:
        return {
            Regime.TRENDING_UP: "trend_following",
            Regime.TRENDING_DOWN: "defensive_trend",
            Regime.RANGE_BOUND: "mean_reversion",
            Regime.HIGH_VOLATILITY: "capital_preservation",
            Regime.CRASH_CORRECTION: "move_to_cash",
        }[self]

    def legacy_label(self) -> str:
        return {
            Regime.TRENDING_UP: "BULL",
            Regime.RANGE_BOUND: "SIDEWAYS",
            Regime.TRENDING_DOWN: "BEAR",
            Regime.HIGH_VOLATILITY: "BEAR",
            Regime.CRASH_CORRECTION: "BEAR",
        }[self]


@dataclass
class RegimeSignal:
    regime: Regime
    confidence: float
    ma50: float
    ma200: float
    vol_20: float
    pct_from_high: float
    trailing_return_20: float
    description: str


class RegimeDetector:
    def __init__(
        self,
        vol_high_threshold: float = 0.28,
        crash_drawdown_threshold: float = -0.12,
        crash_return_threshold: float = -0.08,
    ):
        self.vol_high_threshold = vol_high_threshold
        self.crash_drawdown_threshold = crash_drawdown_threshold
        self.crash_return_threshold = crash_return_threshold

    def detect(self, benchmark_df: pd.DataFrame) -> RegimeSignal:
        close = benchmark_df["close"].dropna()
        if len(close) < 200:
            price = float(close.iloc[-1]) if len(close) else 0.0
            return RegimeSignal(
                regime=Regime.RANGE_BOUND,
                confidence=0.0,
                ma50=price,
                ma200=price,
                vol_20=0.0,
                pct_from_high=0.0,
                trailing_return_20=0.0,
                description="Insufficient data; defaulting to range-bound regime.",
            )

        price = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        returns = close.pct_change().dropna()
        vol_20 = float(returns.iloc[-20:].std() * np.sqrt(252))
        high_252 = float(close.iloc[-252:].max())
        pct_from_high = (price - high_252) / high_252 if high_252 else 0.0
        trailing_return_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 21 else 0.0
        ma_gap = (ma50 - ma200) / ma200 if ma200 else 0.0

        if pct_from_high <= self.crash_drawdown_threshold and trailing_return_20 <= self.crash_return_threshold:
            regime = Regime.CRASH_CORRECTION
            score = 1.0
        elif vol_20 >= self.vol_high_threshold:
            regime = Regime.HIGH_VOLATILITY
            score = min(vol_20 / max(self.vol_high_threshold, 1e-6), 2.0) / 2.0
        elif ma50 > ma200 and price > ma50 and trailing_return_20 > 0:
            regime = Regime.TRENDING_UP
            score = min((ma_gap * 8.0) + max(trailing_return_20, 0.0) * 3.0, 1.0)
        elif ma50 < ma200 and price < ma50 and trailing_return_20 < 0:
            regime = Regime.TRENDING_DOWN
            score = min((abs(ma_gap) * 8.0) + abs(min(trailing_return_20, 0.0)) * 3.0, 1.0)
        else:
            regime = Regime.RANGE_BOUND
            score = max(0.35, 1.0 - min(abs(ma_gap) * 10.0 + abs(trailing_return_20) * 4.0, 0.65))

        confidence = float(np.clip(score, 0.0, 1.0))
        description = (
            f"{regime.value} | mode={regime.strategy_mode()} | "
            f"MA50={ma50:.2f} vs MA200={ma200:.2f} | "
            f"20dRet={trailing_return_20:+.1%} | Vol20={vol_20:.1%} | "
            f"FromHigh={pct_from_high:+.1%}"
        )
        signal = RegimeSignal(
            regime=regime,
            confidence=confidence,
            ma50=ma50,
            ma200=ma200,
            vol_20=vol_20,
            pct_from_high=pct_from_high,
            trailing_return_20=trailing_return_20,
            description=description,
        )
        log.info("Regime detected: %s", description)
        return signal

    @staticmethod
    def fit_hmm(returns: pd.Series, n_states: int = 3):
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError("pip install hmmlearn to use HMM-based detection.") from exc

        x = returns.values.reshape(-1, 1)
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(x)
        means = model.means_.flatten()
        order = np.argsort(means)
        labels = {
            order[0]: Regime.TRENDING_DOWN,
            order[1]: Regime.RANGE_BOUND,
            order[2]: Regime.TRENDING_UP,
        }
        return model, labels
