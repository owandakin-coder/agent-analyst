"""
regime_detector.py
==================
Detects the current market regime (Bull / Bear / Sideways) from price data.

Why it matters
--------------
An RL model trained on mixed data doesn't automatically know whether we're
in a 2020-style crash or a 2023 bull run.  By detecting the regime first,
we can:
  - Scale down position sizes in Bear markets
  - Allow full exposure in confirmed Bull markets
  - Stay defensive (reduce buys, hold cash) in Sideways markets

Detection method
----------------
Rule-based using SPY as the benchmark (simple, interpretable, no refit):
  - 50-day MA vs 200-day MA (Golden / Death Cross)
  - Current price distance from 52-week high
  - 20-day realised volatility

For production a Hidden Markov Model (HMM) with 3 states would be more
robust — the framework is laid out in `fit_hmm()` below as a drop-in upgrade.
"""

from __future__ import annotations

import logging
from enum import Enum
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("RegimeDetector")


class Regime(Enum):
    BULL     = "BULL"      # trend up, low vol → full exposure
    BEAR     = "BEAR"      # trend down, high vol → defensive
    SIDEWAYS = "SIDEWAYS"  # no clear trend → reduced exposure

    def position_multiplier(self) -> float:
        """How much of the model's suggested position to actually use."""
        return {
            Regime.BULL:     1.0,
            Regime.SIDEWAYS: 0.6,
            Regime.BEAR:     0.3,
        }[self]


@dataclass
class RegimeSignal:
    regime:     Regime
    confidence: float          # 0..1
    ma50:       float
    ma200:      float
    vol_20:     float          # annualised realised vol
    pct_from_high: float       # distance from 52-week high (negative = below)
    description: str


class RegimeDetector:
    """
    Classifies the current market regime using SPY (or any benchmark ticker).

    Parameters
    ----------
    vol_bear_threshold : float
        Annualised volatility above this → Bear signal contribution.
    high_bear_threshold : float
        If price is this far below the 52-week high → Bear signal.
    """

    def __init__(
        self,
        vol_bear_threshold: float  = 0.25,   # 25% annualised vol
        high_bear_threshold: float = -0.15,  # 15% below 52-week high
    ):
        self.vol_bear_threshold  = vol_bear_threshold
        self.high_bear_threshold = high_bear_threshold

    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, spy_df: pd.DataFrame) -> RegimeSignal:
        """
        Detects regime from a DataFrame with a 'close' column.

        Parameters
        ----------
        spy_df : pd.DataFrame
            Daily OHLCV for SPY (or benchmark), sorted ascending.

        Returns
        -------
        RegimeSignal
        """
        close = spy_df["close"].dropna()
        if len(close) < 200:
            log.warning(
                f"Only {len(close)} bars available (need 200). "
                "Defaulting to SIDEWAYS."
            )
            return RegimeSignal(
                regime=Regime.SIDEWAYS, confidence=0.0,
                ma50=float(close.iloc[-1]), ma200=float(close.iloc[-1]),
                vol_20=0.0, pct_from_high=0.0,
                description="Insufficient data — defaulting to SIDEWAYS",
            )

        price  = float(close.iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        ma200  = float(close.rolling(200).mean().iloc[-1])

        # 20-day realised volatility (annualised)
        rets   = close.pct_change().dropna()
        vol_20 = float(rets.iloc[-20:].std() * np.sqrt(252))

        # Distance from 52-week high
        high_52w      = float(close.iloc[-252:].max())
        pct_from_high = (price - high_52w) / high_52w

        # ── Score: +1 = Bull signal, -1 = Bear signal ────────────────────
        score = 0.0

        # Golden cross (MA50 > MA200) → +2
        if ma50 > ma200:
            score += 2.0
        else:
            score -= 2.0

        # Price above MA50 → +1
        if price > ma50:
            score += 1.0
        else:
            score -= 1.0

        # Low volatility → +1
        if vol_20 < self.vol_bear_threshold:
            score += 1.0
        else:
            score -= 1.0

        # Near 52-week high → +1
        if pct_from_high > -0.05:
            score += 1.0
        elif pct_from_high < self.high_bear_threshold:
            score -= 1.0

        # ── Map score to regime ───────────────────────────────────────────
        # score range roughly -5 .. +5
        if score >= 2.0:
            regime = Regime.BULL
        elif score <= -1.0:
            regime = Regime.BEAR
        else:
            regime = Regime.SIDEWAYS

        confidence = min(abs(score) / 5.0, 1.0)

        description = (
            f"MA50={'↑' if ma50>ma200 else '↓'} vs MA200 | "
            f"Vol={vol_20:.0%} | "
            f"From52wHigh={pct_from_high:+.1%} | "
            f"Score={score:+.1f}"
        )

        signal = RegimeSignal(
            regime=regime, confidence=confidence,
            ma50=ma50, ma200=ma200,
            vol_20=vol_20, pct_from_high=pct_from_high,
            description=description,
        )

        log.info(
            f"Regime detected: {regime.value} (confidence={confidence:.0%}) | "
            f"{description}"
        )
        return signal

    # ──────────────────────────────────────────────────────────────────────────
    # Optional: HMM-based regime detection (requires hmmlearn)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def fit_hmm(returns: pd.Series, n_states: int = 3):
        """
        Fits a Gaussian HMM with n_states on historical returns.
        Returns (model, state_labels) where state_labels maps
        state index → Regime enum.

        Install: pip install hmmlearn
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("pip install hmmlearn to use HMM-based detection.")

        X = returns.values.reshape(-1, 1)
        model = GaussianHMM(
            n_components=n_states, covariance_type="full",
            n_iter=100, random_state=42,
        )
        model.fit(X)

        # Sort states by mean return: low=Bear, mid=Sideways, high=Bull
        means = model.means_.flatten()
        order = np.argsort(means)
        labels = {
            order[0]: Regime.BEAR,
            order[1]: Regime.SIDEWAYS,
            order[2]: Regime.BULL,
        }
        return model, labels
