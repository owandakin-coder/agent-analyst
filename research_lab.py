"""Independent, long-only shadow portfolios. These never send broker orders.

All portfolios mark previous holdings before computing their next allocation.
Costs are explicit estimates, and account results are kept separate from these
simulations. A cash/benchmark pair provides comparisons at the same exposure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
COST_BPS = 10.0  # 5 bps commission/fees allowance + 5 bps assumed slippage
MAX_EXPOSURE = 0.5


def independent_weights(data: dict, prices: dict[str, float]) -> dict[str, dict[str, float]]:
    momentum, reversal, defensive = {}, {}, {}
    for ticker, frame in data.items():
        if ticker not in prices or len(frame) < 60 or "close" not in frame:
            continue
        close = frame["close"].astype(float).dropna()
        if len(close) < 60 or not np.isfinite(close.iloc[-60:]).all() or (close.iloc[-60:] <= 0).any():
            continue
        last, ma20, ma60 = float(close.iloc[-1]), float(close.iloc[-20:].mean()), float(close.iloc[-60:].mean())
        ret20 = last / float(close.iloc[-21]) - 1
        volatility = float(close.pct_change().iloc[-20:].std())
        if ret20 > 0 and last > ma60:
            momentum[ticker] = ret20
        if ma60 > float(close.iloc[-60]) and last < ma20 * 0.98:
            reversal[ticker] = max(0.0, ma20 / last - 1)
        if last > ma60 and volatility > 0:
            defensive[ticker] = 1 / volatility

    def normalize(scores):
        top = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5])
        total = sum(top.values())
        return {ticker: MAX_EXPOSURE * score / total for ticker, score in top.items()} if total else {}

    return {"momentum": normalize(momentum), "mean_reversion": normalize(reversal), "defensive": normalize(defensive)}


def advance_portfolio(previous: dict | None, weights: dict[str, float], prices: dict[str, float], timestamp: str,
                      initial_equity: float = 100_000, cost_bps: float = COST_BPS) -> tuple[dict, dict]:
    """Mark holdings, then rebalance at this tick. No return is credited upfront."""
    if not math.isfinite(initial_equity) or initial_equity <= 0 or not 0 <= cost_bps < 1000:
        raise ValueError("Invalid portfolio capital/cost")
    marks = {t: float(p) for t, p in prices.items() if math.isfinite(float(p)) and float(p) > 0}
    prior = previous or {"cash": initial_equity, "quantities": {}, "initial_equity": initial_equity, "costs": 0, "peak": initial_equity}
    if prior.get("timestamp") and datetime.fromisoformat(timestamp.replace("Z", "+00:00")) <= datetime.fromisoformat(prior["timestamp"].replace("Z", "+00:00")):
        raise ValueError("Shadow timestamps must increase")
    held = prior.get("quantities", {})
    if any(not math.isfinite(float(q)) or float(q) < 0 for q in held.values()):
        raise ValueError("Invalid shadow holdings")
    if any(t not in marks and float(q) != 0 for t, q in held.items()):
        raise ValueError("Missing price for an existing shadow holding")
    nav = float(prior["cash"]) + sum(float(q) * marks[t] for t, q in held.items() if float(q) != 0)
    if not math.isfinite(nav) or nav <= 0:
        raise ValueError("Invalid shadow equity")
    valid = {t: float(w) for t, w in weights.items() if t in marks and math.isfinite(float(w)) and float(w) > 0}
    total = sum(valid.values())
    if total > MAX_EXPOSURE:
        valid = {t: w * MAX_EXPOSURE / total for t, w in valid.items()}
    def turnover_at(equity):
        return sum(abs(equity * valid.get(t, 0) - float(held.get(t, 0)) * marks[t]) for t in set(valid) | set(held) if t in marks)
    # Solve costs against post-cost equity: targeting 50% of pre-cost NAV
    # silently exceeds the exposure ceiling after fees are deducted.
    low, high = 0.0, nav
    for _ in range(60):
        mid = (low + high) / 2
        if mid + turnover_at(mid) * cost_bps / 10_000 > nav:
            high = mid
        else:
            low = mid
    equity = (low + high) / 2
    cost = turnover_at(equity) * cost_bps / 10_000
    equity = nav - cost
    target = {t: equity * w / marks[t] for t, w in valid.items()}
    cash = equity - sum(q * marks[t] for t, q in target.items())
    peak = max(float(prior.get("peak", nav)), equity)
    state = {"cash": cash, "quantities": target, "initial_equity": float(prior["initial_equity"]),
             "costs": float(prior.get("costs", 0)) + cost, "peak": peak, "timestamp": timestamp,
             "max_drawdown": max(float(prior.get("max_drawdown", 0)), (peak - equity) / peak)}
    sample = {"timestamp": timestamp, "equity": equity, "return": equity / state["initial_equity"] - 1,
              "costs": state["costs"], "cost_this_tick": cost, "cost_bps": cost_bps,
              "max_drawdown": state["max_drawdown"], "exposure": sum(q * marks[t] for t, q in target.items()) / equity,
              "weights": valid, "source": "shadow_simulation", "schema_version": SCHEMA_VERSION}
    return state, sample


def evaluate_shadows(state: dict, data: dict, prices: dict, champion_weights: dict,
                     comparators: dict[str, dict] | None = None, timestamp: str | None = None) -> tuple[dict, list[dict]]:
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    portfolios = dict(state)
    allocations = independent_weights(data, prices)
    allocations["champion"] = champion_weights
    # Same maximum exposure and cost assumptions for every comparator.
    if "SPY" in prices:
        allocations["benchmark_spy"] = {"SPY": MAX_EXPOSURE}
    allocations["cash"] = {}
    if comparators:
        allocations.update(comparators)
    samples = []
    for name, weights in allocations.items():
        updated, sample = advance_portfolio(portfolios.get(name), weights, prices, timestamp)
        portfolios[name] = updated
        samples.append({"strategy": name, **sample})
    return portfolios, samples


def positive_action_weights(tickers: list[str], actions: Any) -> dict[str, float]:
    scores = {t: max(0.0, float(a)) for t, a in zip(tickers, np.asarray(actions).flatten()) if math.isfinite(float(a))}
    total = sum(scores.values())
    return {t: MAX_EXPOSURE * value / max(total, 1.0) for t, value in scores.items() if value > 0}
