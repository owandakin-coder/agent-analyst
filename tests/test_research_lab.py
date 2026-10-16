from datetime import date
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from research_lab import advance_portfolio, independent_weights, evaluate_shadows
import risk_state
from scripts.rolling_training_config import periods


def test_shadow_marks_previous_holdings_and_charges_costs():
    state, first = advance_portfolio(None, {"SPY": .5}, {"SPY": 100}, "2026-10-01T14:00:00Z")
    assert first["equity"] == pytest.approx(99_950)
    state, second = advance_portfolio(state, {}, {"SPY": 110}, "2026-10-02T14:00:00Z")
    assert second["equity"] == pytest.approx(104_895)
    assert state["quantities"] == {}
    assert state["costs"] == pytest.approx(105)


def test_first_shadow_tick_cannot_claim_the_historical_return():
    _, sample = advance_portfolio(None, {"SPY": .5}, {"SPY": 1000}, "2026-10-01T14:00:00Z")
    assert sample["return"] == pytest.approx(-0.0005)


def test_shadow_prices_and_timestamps_fail_closed():
    state, _ = advance_portfolio(None, {"SPY": .5}, {"SPY": 100}, "2026-10-01T14:00:00Z")
    with pytest.raises(ValueError, match="Missing price"):
        advance_portfolio(state, {}, {}, "2026-10-02T14:00:00Z")
    with pytest.raises(ValueError, match="increase"):
        advance_portfolio(state, {}, {"SPY": 100}, "2026-10-01T14:00:00Z")


def test_strategies_are_independent_of_rl_and_bounded():
    data = {"SPY": pd.DataFrame({"close": np.linspace(80, 120, 80)}),
            "AAPL": pd.DataFrame({"close": list(np.linspace(80, 120, 79)) + [110]})}
    allocations = independent_weights(data, {"SPY": 120, "AAPL": 110})
    assert allocations["momentum"]
    assert allocations["mean_reversion"]
    assert allocations["defensive"]
    for weights in allocations.values():
        assert sum(weights.values()) <= .50000001


def test_risk_restore_preserves_peak_stops_and_kelly_across_workers():
    state = {"schema_version": 1, "peak_net_worth": 150000, "risk_peak": 150000,
             "entry_prices": {"AAPL": 100}, "trailing_highs": {"AAPL": 130},
             "trade_outcomes": {"AAPL": [.1, -.05]}, "shadow_portfolios": {}}
    trader = SimpleNamespace(risk_manager=SimpleNamespace())
    risk_state.restore(trader, state)
    saved = risk_state.snapshot(trader)
    assert saved["risk_peak"] == 150000 and saved["trailing_highs"]["AAPL"] == 130
    assert saved["trade_outcomes"]["AAPL"] == [.1, -.05]
    with pytest.raises(ValueError):
        risk_state.restore(trader, {**state, "risk_peak": float("nan")})


def test_rolling_windows_exclude_today_and_do_not_overlap():
    windows = periods(date(2026, 10, 15))
    assert windows["train_end"] < windows["val_start"] <= windows["val_end"] < windows["test_start"]
    assert windows["test_end"] < "2026-10-15"


def test_nonfinite_backtest_metrics_cannot_be_promoted():
    from promote_model import should_promote
    candidate = {"sharpe": float("nan"), "max_drawdown": .1, "annualized_return": .1}
    assert should_promote(candidate, None)[0] is False


def test_missing_spy_does_not_fabricate_a_flat_benchmark():
    _, samples = evaluate_shadows({}, {}, {"AAPL":100}, {}, timestamp="2026-10-01T14:00:00Z")
    assert "benchmark_spy" not in {row["strategy"] for row in samples}


@pytest.mark.parametrize("state_available", [True, False])
def test_live_cycle_persists_evidence_before_broker_mutation(monkeypatch, multi_featured, state_available):
    from tests.test_execution_safety import _make_trader
    import live_trader
    trader = _make_trader(monkeypatch, multi_featured)
    prices = {t: float(frame["close"].iloc[-1]) for t, frame in multi_featured.items()}
    monkeypatch.setattr(live_trader, "load_control_state", lambda:{"trading_enabled":True,"emergency_stop":False})
    monkeypatch.setattr(live_trader, "can_trade", lambda state:(True,None))
    monkeypatch.setattr(trader.broker, "get_latest_prices", lambda tickers:prices)
    monkeypatch.setenv("ATZMA_REQUIRE_FRESH_QUOTES", "0")
    saved, orders = [], []
    def save(state):
        if not state_available:
            raise ConnectionError("State backend unavailable")
        saved.append(state)
    def execute(*_):
        assert saved and saved[-1]["research_samples"]
        assert saved[-1]["account_sample"]["source"] == "paper_broker_snapshot"
        orders.append(1)
        return []
    monkeypatch.setattr(live_trader, "save_execution_state", save)
    monkeypatch.setattr(trader, "_execute_actions", execute)
    if state_available:
        result = trader.run_once()
        assert result["research_samples"] and orders == [1]
        assert len(result["feature_snapshot_hash"]) == 64
    else:
        with pytest.raises(ConnectionError, match="State backend unavailable"):
            trader.run_once()
        assert orders == []


def test_risk_halt_still_records_the_observed_account_equity(monkeypatch, multi_featured):
    from tests.test_execution_safety import _make_trader
    import live_trader
    trader = _make_trader(monkeypatch, multi_featured)
    trader.risk_manager.peak_value = 200000
    monkeypatch.setattr(live_trader, "load_control_state", lambda:{"trading_enabled":True,"emergency_stop":False})
    monkeypatch.setattr(live_trader, "can_trade", lambda state:(True,None))
    monkeypatch.setattr(trader.broker,"get_latest_prices",lambda tickers:{t:float(df["close"].iloc[-1]) for t,df in multi_featured.items()})
    monkeypatch.setenv("ATZMA_REQUIRE_FRESH_QUOTES","0")
    monkeypatch.setattr(trader,"_notify_halt",lambda *_:None)
    saved=[]
    monkeypatch.setattr(live_trader,"save_execution_state",saved.append)
    assert trader.run_once() is None
    assert saved[-1]["account_sample"]["equity"] == 100000
