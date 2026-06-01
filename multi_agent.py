"""
Multi-agent orchestration layer for ATZMA.

The RL model still proposes a directional action, but execution only happens
after three specialised agents vote unanimously:
- TrendAgent   -> long-term direction
- EntryAgent   -> timing and tactical entry/exit
- DefenseAgent -> risk veto / forced defense
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class AgentDirection(Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class AgentVote:
    agent: str
    direction: AgentDirection
    confidence: float
    reason: str
    metrics: dict[str, float | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = self.direction.value
        return payload


@dataclass
class TickerDecision:
    ticker: str
    proposed_action: float
    final_action: float
    unanimous: bool
    direction: AgentDirection
    votes: list[AgentVote]
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "proposed_action": float(self.proposed_action),
            "final_action": float(self.final_action),
            "unanimous": bool(self.unanimous),
            "direction": self.direction.value,
            "votes": [vote.as_dict() for vote in self.votes],
            "explanation": self.explanation,
        }


@dataclass
class MultiAgentDecisionBundle:
    regime: str
    strategy_mode: str
    decisions: list[TickerDecision]
    raw_action: list[float]

    def final_action_vector(self, tickers: list[str]) -> np.ndarray:
        by_ticker = {decision.ticker: decision.final_action for decision in self.decisions}
        return np.array([by_ticker.get(ticker, 0.0) for ticker in tickers], dtype=float)

    def top_summary(self) -> str:
        if not self.decisions:
            return "No trade candidates."
        unanimous = [d for d in self.decisions if d.unanimous and abs(d.final_action) > 0]
        if unanimous:
            top = max(unanimous, key=lambda d: abs(d.final_action))
            return top.explanation
        vetoed = max(self.decisions, key=lambda d: abs(d.proposed_action))
        return f"No unanimous trade. {vetoed.ticker}: {vetoed.explanation}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "strategy_mode": self.strategy_mode,
            "raw_action": [float(x) for x in self.raw_action],
            "decisions": [decision.as_dict() for decision in self.decisions],
            "summary": self.top_summary(),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _latest(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in df.columns or df.empty:
        return default
    return _safe_float(df[column].iloc[-1], default)


def _rolling_close(df: pd.DataFrame, window: int) -> float:
    close = df.get("close")
    if close is None or len(close) < window:
        return _safe_float(close.iloc[-1] if close is not None and len(close) else 0.0)
    return _safe_float(close.rolling(window).mean().iloc[-1], 0.0)


class TrendAgent:
    def vote(self, ticker: str, df: pd.DataFrame, regime_signal, proposed: AgentDirection, rl_action: float) -> AgentVote:
        price = _latest(df, "close")
        ma50 = _rolling_close(df, 50)
        ma200 = _rolling_close(df, 200)
        macd_hist = _latest(df, "macd_hist")

        direction = AgentDirection.HOLD
        confidence = 0.35
        reason = "Long-term trend is mixed."

        if proposed == AgentDirection.BUY and ma50 >= ma200 and price >= ma50 and macd_hist >= 0:
            direction = AgentDirection.BUY
            confidence = min(0.55 + abs(rl_action) * 0.3, 0.95)
            reason = f"Trend up: price>{ma50:.2f}, MA50>{ma200:.2f}, MACD positive."
        elif proposed == AgentDirection.SELL and (ma50 < ma200 or price < ma50 or macd_hist < 0):
            direction = AgentDirection.SELL
            confidence = min(0.55 + abs(rl_action) * 0.3, 0.95)
            reason = f"Trend weak: price<{ma50:.2f} or MA50<{ma200:.2f}, MACD soft."

        return AgentVote(
            agent="Trend Agent",
            direction=direction,
            confidence=confidence,
            reason=reason,
            metrics={"price": round(price, 4), "ma50": round(ma50, 4), "ma200": round(ma200, 4), "macd_hist": round(macd_hist, 4)},
        )


class EntryAgent:
    def vote(self, ticker: str, df: pd.DataFrame, regime_signal, proposed: AgentDirection, rl_action: float) -> AgentVote:
        rsi = _latest(df, "rsi", 50.0)
        boll_pct = _latest(df, "boll_pct", 0.5)
        regime_name = getattr(regime_signal.regime, "value", "RANGE_BOUND") if regime_signal else "UNSPECIFIED"

        direction = AgentDirection.HOLD
        confidence = 0.30
        reason = "Entry timing is not compelling."

        if proposed == AgentDirection.BUY:
            if regime_name == "TRENDING_UP" and rsi <= 55 and boll_pct <= 0.70:
                direction = AgentDirection.BUY
                confidence = min(0.5 + abs(rl_action) * 0.25, 0.9)
                reason = f"Trend pullback entry: RSI={rsi:.1f}, Bollinger={boll_pct:.2f}."
            elif regime_name == "RANGE_BOUND" and rsi <= 35 and boll_pct <= 0.20:
                direction = AgentDirection.BUY
                confidence = min(0.58 + abs(rl_action) * 0.2, 0.88)
                reason = f"Mean-reversion buy: RSI={rsi:.1f}, Bollinger={boll_pct:.2f}."
            elif regime_name == "UNSPECIFIED" and rsi <= 60 and boll_pct <= 0.85:
                direction = AgentDirection.BUY
                confidence = min(0.46 + abs(rl_action) * 0.22, 0.82)
                reason = f"Fallback entry buy: RSI={rsi:.1f}, Bollinger={boll_pct:.2f}."
        elif proposed == AgentDirection.SELL:
            if regime_name in {"TRENDING_DOWN", "CRASH_CORRECTION"} and rsi >= 45:
                direction = AgentDirection.SELL
                confidence = min(0.5 + abs(rl_action) * 0.25, 0.9)
                reason = f"Downtrend exit: RSI rebound={rsi:.1f} into weakness."
            elif regime_name == "RANGE_BOUND" and rsi >= 65 and boll_pct >= 0.80:
                direction = AgentDirection.SELL
                confidence = min(0.58 + abs(rl_action) * 0.2, 0.88)
                reason = f"Range sell: RSI={rsi:.1f}, Bollinger={boll_pct:.2f}."
            elif regime_name == "UNSPECIFIED" and rsi >= 40:
                direction = AgentDirection.SELL
                confidence = min(0.46 + abs(rl_action) * 0.22, 0.82)
                reason = f"Fallback exit sell: RSI={rsi:.1f}."

        return AgentVote(
            agent="Entry Agent",
            direction=direction,
            confidence=confidence,
            reason=reason,
            metrics={"rsi": round(rsi, 2), "boll_pct": round(boll_pct, 4), "regime": regime_name},
        )


class DefenseAgent:
    def __init__(self, stop_loss_pct: float = 0.08):
        self.stop_loss_pct = stop_loss_pct

    def vote(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime_signal,
        proposed: AgentDirection,
        rl_action: float,
        position: float,
        entry_price: float,
        trailing_high: float,
        current_drawdown: float,
    ) -> AgentVote:
        price = _latest(df, "close")
        atr_pct = _latest(df, "atr_pct")
        vol20 = _latest(df, "volatility_20")
        regime_name = getattr(regime_signal.regime, "value", "RANGE_BOUND") if regime_signal else "RANGE_BOUND"

        if regime_name == "CRASH_CORRECTION":
            direction = AgentDirection.SELL if position > 0 else AgentDirection.HOLD
            return AgentVote(
                agent="Defense Agent",
                direction=direction,
                confidence=0.95,
                reason="Crash regime: move to cash.",
                metrics={"atr_pct": round(atr_pct, 4), "volatility_20": round(vol20, 4)},
            )

        if trailing_high > 0 and price > 0 and position > 0:
            drop = (trailing_high - price) / trailing_high
            if drop >= self.stop_loss_pct:
                return AgentVote(
                    agent="Defense Agent",
                    direction=AgentDirection.SELL,
                    confidence=0.95,
                    reason=f"Trailing stop violated: drop={drop:.1%} from high.",
                    metrics={"price": round(price, 4), "trailing_high": round(trailing_high, 4)},
                )

        if current_drawdown >= 0.10 and proposed == AgentDirection.BUY:
            return AgentVote(
                agent="Defense Agent",
                direction=AgentDirection.HOLD,
                confidence=0.82,
                reason=f"Drawdown={current_drawdown:.1%}. New buys paused.",
                metrics={"drawdown": round(current_drawdown, 4)},
            )

        if regime_name == "HIGH_VOLATILITY" and proposed == AgentDirection.BUY:
            return AgentVote(
                agent="Defense Agent",
                direction=AgentDirection.HOLD,
                confidence=0.84,
                reason="High-volatility regime: block fresh longs.",
                metrics={"atr_pct": round(atr_pct, 4), "volatility_20": round(vol20, 4)},
            )

        direction = proposed if proposed != AgentDirection.HOLD else AgentDirection.HOLD
        confidence = min(0.45 + abs(rl_action) * 0.2, 0.85)
        reason = "Risk posture allows the proposed direction."
        return AgentVote(
            agent="Defense Agent",
            direction=direction,
            confidence=confidence,
            reason=reason,
            metrics={"atr_pct": round(atr_pct, 4), "volatility_20": round(vol20, 4), "entry_price": round(entry_price, 4)},
        )


class MultiAgentDecisionEngine:
    def __init__(self, buy_threshold: float = 0.05, sell_threshold: float = -0.05, stop_loss_pct: float = 0.08):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.trend_agent = TrendAgent()
        self.entry_agent = EntryAgent()
        self.defense_agent = DefenseAgent(stop_loss_pct=stop_loss_pct)

    def _proposed_direction(self, rl_action: float) -> AgentDirection:
        if rl_action >= self.buy_threshold:
            return AgentDirection.BUY
        if rl_action <= self.sell_threshold:
            return AgentDirection.SELL
        return AgentDirection.HOLD

    def evaluate(
        self,
        tickers: list[str],
        fresh_data: dict[str, pd.DataFrame],
        raw_action: np.ndarray,
        regime_signal,
        positions: dict[str, float],
        entry_prices: dict[str, float],
        trailing_highs: dict[str, float],
        current_drawdown: float,
    ) -> MultiAgentDecisionBundle:
        decisions: list[TickerDecision] = []
        raw_vector = np.array(raw_action, dtype=float).flatten()

        for index, ticker in enumerate(tickers):
            df = fresh_data.get(ticker)
            rl_action = float(raw_vector[index]) if index < len(raw_vector) else 0.0
            proposed = self._proposed_direction(rl_action)

            if df is None or df.empty:
                decisions.append(TickerDecision(
                    ticker=ticker,
                    proposed_action=rl_action,
                    final_action=0.0,
                    unanimous=False,
                    direction=AgentDirection.HOLD,
                    votes=[],
                    explanation="Missing market data; no action.",
                ))
                continue

            if proposed == AgentDirection.HOLD:
                decisions.append(TickerDecision(
                    ticker=ticker,
                    proposed_action=rl_action,
                    final_action=0.0,
                    unanimous=False,
                    direction=AgentDirection.HOLD,
                    votes=[],
                    explanation="RL conviction below action threshold.",
                ))
                continue

            votes = [
                self.trend_agent.vote(ticker, df, regime_signal, proposed, rl_action),
                self.entry_agent.vote(ticker, df, regime_signal, proposed, rl_action),
                self.defense_agent.vote(
                    ticker=ticker,
                    df=df,
                    regime_signal=regime_signal,
                    proposed=proposed,
                    rl_action=rl_action,
                    position=float(positions.get(ticker, 0.0)),
                    entry_price=float(entry_prices.get(ticker, 0.0)),
                    trailing_high=float(trailing_highs.get(ticker, 0.0)),
                    current_drawdown=current_drawdown,
                ),
            ]
            unanimous = all(v.direction == proposed for v in votes)
            final_action = rl_action if unanimous else 0.0
            explanation = self._explain(ticker, proposed, votes, unanimous)
            decisions.append(TickerDecision(
                ticker=ticker,
                proposed_action=rl_action,
                final_action=final_action,
                unanimous=unanimous,
                direction=proposed if unanimous else AgentDirection.HOLD,
                votes=votes,
                explanation=explanation,
            ))

        regime_name = getattr(regime_signal.regime, "value", "RANGE_BOUND") if regime_signal else "RANGE_BOUND"
        strategy_mode = getattr(regime_signal.regime, "strategy_mode", lambda: "adaptive")() if regime_signal else "adaptive"
        return MultiAgentDecisionBundle(
            regime=regime_name,
            strategy_mode=strategy_mode,
            decisions=decisions,
            raw_action=[float(x) for x in raw_vector.tolist()],
        )

    @staticmethod
    def _explain(ticker: str, proposed: AgentDirection, votes: list[AgentVote], unanimous: bool) -> str:
        if unanimous:
            dominant = max(votes, key=lambda vote: vote.confidence)
            return f"{proposed.value} {ticker}: unanimous approval. {dominant.agent}: {dominant.reason}"
        blocks = [f"{vote.agent}={vote.direction.value}" for vote in votes]
        return f"HOLD {ticker}: vote split ({', '.join(blocks)})."
