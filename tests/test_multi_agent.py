from __future__ import annotations

import numpy as np
import pandas as pd

from multi_agent import AgentDirection, MultiAgentDecisionEngine
from regime_detector import Regime, RegimeDetector


def _frame(close, rsi, boll_pct, macd_hist, atr_pct=0.02, vol=0.12):
    size = len(close)
    return pd.DataFrame({
        "close": close,
        "rsi": np.full(size, rsi, dtype=float),
        "boll_pct": np.full(size, boll_pct, dtype=float),
        "macd_hist": np.full(size, macd_hist, dtype=float),
        "atr_pct": np.full(size, atr_pct, dtype=float),
        "volatility_20": np.full(size, vol, dtype=float),
    })


def test_regime_detector_classifies_crash():
    close = np.concatenate([
        np.linspace(100, 120, 240),
        np.linspace(118, 96, 20),
    ])
    df = pd.DataFrame({"close": close})
    signal = RegimeDetector().detect(df)
    assert signal.regime == Regime.CRASH_CORRECTION
    assert signal.regime.position_multiplier() == 0.0


def test_multi_agent_unanimous_buy():
    trend_close = np.linspace(100, 140, 260)
    ticker_df = _frame(trend_close, rsi=34, boll_pct=0.18, macd_hist=0.8)
    spy_df = pd.DataFrame({"close": np.linspace(100, 150, 260)})
    regime_signal = RegimeDetector().detect(spy_df)
    assert regime_signal.regime == Regime.TRENDING_UP

    engine = MultiAgentDecisionEngine()
    bundle = engine.evaluate(
        tickers=["AAPL"],
        fresh_data={"AAPL": ticker_df},
        raw_action=np.array([0.7]),
        regime_signal=regime_signal,
        positions={},
        entry_prices={},
        trailing_highs={},
        current_drawdown=0.02,
    )

    decision = bundle.decisions[0]
    assert decision.unanimous is True
    assert decision.direction == AgentDirection.BUY
    assert decision.final_action > 0


def test_multi_agent_blocks_buy_in_high_volatility():
    spy_df = pd.DataFrame({"close": np.concatenate([np.linspace(100, 130, 230), np.linspace(130, 122, 30)])})
    signal = RegimeDetector().detect(spy_df)
    signal.regime = Regime.HIGH_VOLATILITY
    signal.description = "Forced high volatility for test"

    ticker_df = _frame(np.linspace(100, 120, 260), rsi=33, boll_pct=0.10, macd_hist=0.9, atr_pct=0.08, vol=0.38)
    engine = MultiAgentDecisionEngine()
    bundle = engine.evaluate(
        tickers=["AAPL"],
        fresh_data={"AAPL": ticker_df},
        raw_action=np.array([0.9]),
        regime_signal=signal,
        positions={},
        entry_prices={},
        trailing_highs={},
        current_drawdown=0.03,
    )

    decision = bundle.decisions[0]
    assert decision.unanimous is False
    assert decision.final_action == 0.0
    assert "Defense Agent=HOLD" in decision.explanation
