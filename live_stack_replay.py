"""
live_stack_replay.py
=====================
Replays the production decision stack over the out-of-sample window
(forward_eval.FORWARD_START..today): model -> regime detector -> the three-agent
unanimity veto (multi_agent.py) -> RiskManager -> env. forward_eval.py scores the
model WITHOUT the veto; this shows what the live cycle would have traded.

Not replicated: the 200-session history gate in live_trader, order-size
minimums, the daily order limit and broker fills. So treat the trade counts as
an upper bound on what live would do.

Usage:  python live_stack_replay.py
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _apply_rule(bundle, tickers, rule: str) -> np.ndarray:
    """Final action vector under an approval rule. "unanimous" is production.
    "majority": two of the three agents. "trend_defense": Trend and Defense
    agree and the Entry agent is not consulted. "none": the raw model."""
    if rule == "unanimous":
        return bundle.final_action_vector(tickers)
    out = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        d = next(x for x in bundle.decisions if x.ticker == t)
        if rule == "none":
            out[i] = d.proposed_action
        elif d.votes:
            want = "BUY" if d.proposed_action > 0 else "SELL"
            by_agent = {v.agent: v.direction.value == want for v in d.votes}
            if rule == "majority" and sum(by_agent.values()) >= 2:
                out[i] = d.proposed_action
            elif rule == "trend_defense" and by_agent.get("Trend Agent") and by_agent.get("Defense Agent"):
                out[i] = d.proposed_action
    return out


def replay(rule: str = "unanimous") -> dict:
    from stable_baselines3.common.vec_env import DummyVecEnv
    from benchmark import spy_buy_hold, trading_window
    from config_loader import CFG
    from data_manager import DataManager
    from ensemble_agent import load_ensemble
    from forward_eval import FORWARD_START
    from market_context import fetch_market_context
    from multi_agent import MultiAgentDecisionEngine
    from promote_model import metrics_from_equity
    from regime_detector import RegimeDetector
    from risk_manager import RiskManager
    from trading_env import TradingEnvironment
    from datetime import date

    today = date.today().isoformat()
    dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=today)
    dm.load_all(force_download=False)
    full = dm.get_aligned_data()
    data = {t: df[df.index >= FORWARD_START] for t, df in full.items()}
    tickers = list(data.keys())

    vec_env = DummyVecEnv([lambda: TradingEnvironment(data, max_drawdown_stop=1.0)])
    inner = vec_env.envs[0]
    window = inner.window_size
    index = inner._aligned_index
    agent = load_ensemble(dummy_data=data)
    engine = MultiAgentDecisionEngine()
    detector = RegimeDetector()
    vix = fetch_market_context(CFG.data_start, today)
    risk = RiskManager(CFG.initial_capital)

    obs = vec_env.reset()
    equity_live, equity_raw = [CFG.initial_capital], [CFG.initial_capital]
    entry_price: dict[str, float] = {}
    trail_high: dict[str, float] = {}
    proposed_n, approved_n = Counter(), Counter()
    trade_days = 0
    done, step, net_worth = False, 0, CFG.initial_capital
    peak = CFG.initial_capital

    while not done:
        date_now = index[min(step + window, len(index) - 1)]
        spy_hist = full["SPY"][full["SPY"].index <= date_now]
        vix_past = vix[vix.index <= date_now]["vix_close"]
        signal = detector.detect(spy_hist, vix_value=float(vix_past.iloc[-1]) if len(vix_past) else None)
        risk.set_regime_multiplier(signal.regime.position_multiplier())

        raw = agent.predict(obs, deterministic=True)
        raw = np.asarray(raw).flatten()
        fresh = {t: full[t][full[t].index <= date_now] for t in tickers}
        prices = {t: float(fresh[t]["close"].iloc[-1]) for t in tickers}
        held = {t: float(inner.holdings[i]) for i, t in enumerate(tickers)}
        for t in tickers:
            if held[t] > 0:
                entry_price.setdefault(t, prices[t])
                trail_high[t] = max(trail_high.get(t, prices[t]), prices[t])
            else:
                entry_price.pop(t, None)
                trail_high.pop(t, None)
        peak = max(peak, net_worth)
        drawdown = (peak - net_worth) / peak if peak else 0.0

        bundle = engine.evaluate(
            tickers=tickers, fresh_data=fresh, raw_action=raw, regime_signal=signal,
            positions=held, entry_prices=entry_price, trailing_highs=trail_high,
            current_drawdown=drawdown,
        )
        action = _apply_rule(bundle, tickers, rule)
        for i, d in enumerate(bundle.decisions):
            if d.votes:
                side = "BUY" if d.proposed_action > 0 else "SELL"
                proposed_n[side] += 1
                if action[i] != 0:
                    approved_n[side] += 1
        if np.any(action != 0):
            trade_days += 1

        risk.update(net_worth)
        action = risk.scale_action(action)
        obs, _, dones, infos = vec_env.step(action[np.newaxis])
        done = bool(dones[0])
        net_worth = infos[0].get("net_worth", net_worth)
        equity_live.append(net_worth)
        step += 1

    spy = spy_buy_hold(trading_window(data), CFG.initial_capital)
    return {
        "days": step, "trade_days": trade_days,
        "proposed": dict(proposed_n), "approved": dict(approved_n),
        "live_stack": metrics_from_equity(np.array(equity_live)),
        "spy": metrics_from_equity(np.asarray(spy, dtype=float)),
    }


def main() -> int:
    rules = [("unanimous", "production: all 3 agents"), ("majority", "2 of 3 agents"),
             ("trend_defense", "Trend + Defense only"), ("none", "no veto (what forward_eval scores)")]
    print(f"{'rule':34s} {'action days':>11s} {'buys ok':>8s} {'sells ok':>9s} {'return':>8s} {'sharpe':>7s} {'maxDD':>7s}")
    spy = None
    for rule, label in rules:
        r = replay(rule)
        m, spy = r["live_stack"], r["spy"]
        print(f"{label:34s} {r['trade_days']:>5d}/{r['days']:<5d} {r['approved'].get('BUY', 0):>8d} "
              f"{r['approved'].get('SELL', 0):>9d} {m['total_return']:>+8.1%} {m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%}")
    print(f"{'SPY, same days':34s} {'':>11s} {'':>8s} {'':>9s} {spy['total_return']:>+8.1%} {spy['sharpe']:>7.2f} {spy['max_drawdown']:>7.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
