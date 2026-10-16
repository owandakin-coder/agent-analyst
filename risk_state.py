"""Validated state shared by sequential workers for one broker/strategy scope."""
from __future__ import annotations
import math


def _positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Invalid persisted risk value")
    return result


def restore(trader, state: dict) -> None:
    if not state:
        trader._shadow_portfolios = {}
        return
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported persisted risk schema")
    trader._peak_net_worth = _positive(state["peak_net_worth"])
    trader.risk_manager.peak_value = _positive(state["risk_peak"])
    trader._entry_prices = {t: _positive(p) for t, p in state.get("entry_prices", {}).items()}
    trader._trailing_highs = {t: _positive(p) for t, p in state.get("trailing_highs", {}).items()}
    outcomes = state.get("trade_outcomes", {})
    for values in outcomes.values():
        if not isinstance(values, list) or any(not math.isfinite(float(v)) for v in values):
            raise ValueError("Invalid persisted trade outcomes")
    trader.risk_manager._trade_outcomes = {t: list(v[-100:]) for t, v in outcomes.items()}
    trader._shadow_portfolios = dict(state.get("shadow_portfolios", {}))


def snapshot(trader) -> dict:
    return {"schema_version": 1, "peak_net_worth": trader._peak_net_worth,
            "risk_peak": trader.risk_manager.peak_value,
            "entry_prices": dict(trader._entry_prices), "trailing_highs": dict(trader._trailing_highs),
            "trade_outcomes": trader.risk_manager._trade_outcomes,
            "shadow_portfolios": trader._shadow_portfolios,
            "research_samples": getattr(trader, "_latest_research_samples", []),
            "account_sample": getattr(trader, "_latest_account_sample", None)}
