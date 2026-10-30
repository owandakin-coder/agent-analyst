from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_agent import AgentDirection, MultiAgentDecisionEngine
from regime_detector import Regime, RegimeDetector


@pytest.mark.parametrize("rows", [30, 199])
def test_trend_does_not_substitute_spot_for_missing_history(rows):
    from multi_agent import TrendAgent
    df = _frame(np.full(rows, 100.0), rsi=30, boll_pct=0.1, macd_hist=1)
    vote = TrendAgent().vote("AAPL", df, None, AgentDirection.BUY, 0.7)
    assert vote.direction == AgentDirection.HOLD
    assert vote.confidence == 0
    assert "200-session" in vote.reason
    assert "ma200" not in vote.metrics


def test_trend_uses_full_history_at_200_session_boundary():
    from multi_agent import TrendAgent
    df = _frame(np.linspace(100, 140, 200), rsi=30, boll_pct=0.1, macd_hist=1)
    vote = TrendAgent().vote("AAPL", df, None, AgentDirection.BUY, 0.7)
    assert vote.direction == AgentDirection.BUY
    assert vote.metrics["ma200"] == pytest.approx(120)
    assert vote.metrics["ma200"] != vote.metrics["price"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0, -1.0])
def test_trend_blocks_approval_for_invalid_history(invalid):
    from multi_agent import TrendAgent
    df = _frame(np.linspace(100, 140, 220), rsi=30, boll_pct=0.1, macd_hist=1)
    df.loc[50, "close"] = invalid
    vote = TrendAgent().vote("AAPL", df, None, AgentDirection.BUY, 0.7)
    assert vote.direction == AgentDirection.HOLD
    assert vote.confidence == 0


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


def _buy_candidate(rule):
    """A buy the Trend and Defense agents like but the Entry agent vetoes
    (RSI 70: not a pullback)."""
    ticker_df = _frame(np.linspace(100, 140, 260), rsi=70, boll_pct=0.60, macd_hist=0.8)
    signal = RegimeDetector().detect(pd.DataFrame({"close": np.linspace(100, 150, 260)}))
    assert signal.regime == Regime.TRENDING_UP
    bundle = MultiAgentDecisionEngine(approval_rule=rule).evaluate(
        tickers=["AAPL"], fresh_data={"AAPL": ticker_df}, raw_action=np.array([0.7]),
        regime_signal=signal, positions={}, entry_prices={}, trailing_highs={}, current_drawdown=0.02,
    )
    return bundle.decisions[0], bundle


def test_unanimous_rule_lets_the_entry_agent_veto():
    decision, _ = _buy_candidate("unanimous")
    assert decision.unanimous is False
    assert decision.final_action == 0.0
    assert decision.direction == AgentDirection.HOLD


def test_trend_defense_rule_ignores_the_entry_agent_veto():
    decision, bundle = _buy_candidate("trend_defense")
    assert decision.unanimous is False
    assert decision.final_action == pytest.approx(0.7)
    assert decision.direction == AgentDirection.BUY
    assert bundle.top_summary() == decision.explanation


def test_trend_defense_rule_still_blocked_by_the_defense_agent():
    spy_df = pd.DataFrame({"close": np.concatenate([np.linspace(100, 130, 230), np.linspace(130, 122, 30)])})
    signal = RegimeDetector().detect(spy_df)
    signal.regime = Regime.HIGH_VOLATILITY
    ticker_df = _frame(np.linspace(100, 140, 260), rsi=40, boll_pct=0.30, macd_hist=0.8, atr_pct=0.08, vol=0.38)
    bundle = MultiAgentDecisionEngine(approval_rule="trend_defense").evaluate(
        tickers=["AAPL"], fresh_data={"AAPL": ticker_df}, raw_action=np.array([0.9]),
        regime_signal=signal, positions={}, entry_prices={}, trailing_highs={}, current_drawdown=0.03,
    )
    assert bundle.decisions[0].final_action == 0.0


def test_unknown_approval_rule_is_rejected():
    with pytest.raises(ValueError):
        MultiAgentDecisionEngine(approval_rule="majority")
