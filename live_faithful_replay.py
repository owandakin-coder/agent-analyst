"""
live_faithful_replay.py
=======================
Replays the live Paper decision path over the out-of-sample window, one decision
per trading day, using the production code itself instead of a re-implementation:

    LiveTrader._build_observation -> model -> RegimeDetector -> MultiAgentDecisionEngine
    -> RiskManager.scale_action -> LiveTrader._execute_actions (order sizing,
    trailing stop, concentration / gross / cash-buffer caps, re-entry cooldown)

live_stack_replay.py stops before _execute_actions, so it ignores every live
sizing cap and shows a portfolio that is far more invested than the real account.
Here orders go to a simulated broker: decided on day t's close, filled at day
t+1's open with config slippage and commission.

Differences from live, stated so nobody over-reads the numbers:
  * one decision per day (live runs a cycle every ~15 min);
  * no daily-order limit, broker rejections, partial fills or stale quotes;
  * the model was trained before FORWARD_START, so this is out of sample.

Usage:
    python live_faithful_replay.py                      # production config
    python live_faithful_replay.py BUY_THRESHOLD=0.05   # override a live constant
    python live_faithful_replay.py START=2022-01-01 END=2024-12-31   # another window
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np

if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class _NoData:
    """LiveTrader's data_manager is only used by the fetch path, which is not called."""


class SimBroker:
    """BrokerAPIStub-compatible: instant fills at the next open, with costs."""

    broker_name = "sim"

    def __init__(self, cash: float, slippage: float, commission: float):
        self.cash = cash
        self.positions: dict[str, float] = {}
        self.slippage = slippage
        self.commission = commission
        self.fill_prices: dict[str, float] = {}
        self.now = datetime.now(timezone.utc)
        self.sells: dict[str, datetime] = {}
        self.orders: list[tuple] = []

    def _fill(self, side: str, ticker: str, shares: float) -> dict:
        base = self.fill_prices[ticker]
        price = base * (1 + self.slippage) if side == "BUY" else base * (1 - self.slippage)
        notional = shares * price
        fee = notional * self.commission
        self.cash += (-notional if side == "BUY" else notional) - fee
        self.orders.append((self.now, side, ticker, shares, price))
        return {
            "order_id": f"SIM-{len(self.orders):06d}", "status": "FILLED_STUB", "side": side,
            "ticker": ticker, "shares": shares, "price": price,
            "filled_qty": shares, "filled_avg_price": price,
        }

    def buy(self, ticker, shares, price):
        self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
        return self._fill("BUY", ticker, shares)

    def sell(self, ticker, shares, price):
        shares = min(shares, self.positions.get(ticker, 0.0))
        if shares <= 0:
            return {"status": "REJECTED", "reason": "no_position"}
        self.positions[ticker] -= shares
        self.sells[ticker] = self.now
        return self._fill("SELL", ticker, shares)

    def reconcile_account_state(self) -> dict:
        return {"cash": self.cash, "equity": self.cash, "buying_power": self.cash,
                "positions": dict(self.positions), "position_details": {}}

    def recent_sell_tickers(self, days: int) -> set[str]:
        cutoff = self.now - timedelta(days=days)
        return {t for t, when in self.sells.items() if when >= cutoff}


def replay(overrides: dict | None = None, approval_rule: str | None = None,
           start: str | None = None, end: str | None = None) -> dict:
    import live_trader
    from benchmark import spy_buy_hold, trading_window
    from config_loader import CFG
    from data_manager import DataManager
    from ensemble_agent import load_ensemble
    from forward_eval import FORWARD_START
    from market_context import fetch_market_context
    from promote_model import metrics_from_equity
    from risk_manager import RiskManager

    start = start or FORWARD_START
    saved = {}
    patch = dict(overrides or {})
    if approval_rule:
        patch["APPROVAL_RULE"] = approval_rule
    for k in [k for k in patch if k.startswith("TREND_")]:
        import trend_signal
        setattr(trend_signal, k[6:], patch.pop(k))
    risk_over = {k[5:].lower(): bool(patch.pop(k)) for k in [k for k in patch if k.startswith("RISK_")]}
    for name, value in patch.items():
        if not hasattr(live_trader, name):
            raise ValueError(f"unknown live_trader constant: {name}")
        saved[name] = getattr(live_trader, name)
        setattr(live_trader, name, value)
    try:
        today = date.today().isoformat()
        dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=today)
        dm.load_all(force_download=False)
        full = dm.get_aligned_data()
        tickers = list(full.keys())
        index = full[tickers[0]].index
        vix = fetch_market_context(CFG.data_start, today)

        agent = load_ensemble(dummy_data={t: df[df.index >= FORWARD_START] for t, df in full.items()})
        broker = SimBroker(CFG.initial_capital, CFG.slippage_pct, CFG.commission_pct)
        risk = RiskManager(CFG.initial_capital)
        for name, value in risk_over.items():
            setattr(risk, name, value)
        trader = live_trader.LiveTrader(
            model=agent, broker=broker, data_manager=_NoData(), risk_manager=risk,
            vec_norm=None, tickers=tickers, initial_capital=CFG.initial_capital,
        )

        days = [i for i, d in enumerate(index) if str(d.date()) >= start and (end is None or str(d.date()) <= end)]
        equity, gross, peak = [CFG.initial_capital], [0.0], CFG.initial_capital
        raws, dates = [], []
        for k, i in enumerate(days[:-1]):
            d = index[i]
            broker.now = d.to_pydatetime().replace(tzinfo=timezone.utc)
            fresh = {t: full[t].iloc[: i + 1] for t in tickers}
            prices = {t: float(fresh[t]["close"].iloc[-1]) for t in tickers}
            nxt = index[i + 1]
            broker.fill_prices = {
                t: float(full[t]["open"].loc[nxt]) if "open" in full[t].columns else float(full[t]["close"].loc[nxt])
                for t in tickers
            }
            positions = dict(broker.positions)
            cash = broker.cash
            net_worth = cash + sum(positions.get(t, 0.0) * prices[t] for t in tickers)
            peak = max(peak, net_worth)
            drawdown = (peak - net_worth) / peak if peak else 0.0
            risk.update(net_worth)

            vix_past = vix[vix.index <= d]["vix_close"]
            signal = trader._regime_detector.detect(
                fresh["SPY"].reset_index(drop=True), vix_value=float(vix_past.iloc[-1]) if len(vix_past) else None)
            risk.set_regime_multiplier(signal.regime.position_multiplier())

            obs = trader._build_observation(fresh, cash, net_worth, drawdown)
            raw = np.asarray(agent.predict(obs[np.newaxis], deterministic=True)).flatten()
            raws.append(raw.copy())
            raw = trader._apply_signal_source(raw, fresh, positions)
            dates.append(d)
            bundle = trader._multi_agent.evaluate(
                tickers=tickers, fresh_data=fresh, raw_action=raw, regime_signal=signal,
                positions=positions, entry_prices=trader._entry_prices,
                trailing_highs=trader._trailing_highs, current_drawdown=drawdown)
            action = risk.scale_action(
                bundle.final_action_vector(tickers), tickers=tickers,
                price_history={t: fresh[t]["close"] for t in tickers})
            trader._execute_actions(action, prices, cash, positions, cash)

            nxt_prices = {t: float(full[t]["close"].loc[nxt]) for t in tickers}
            value = sum(broker.positions.get(t, 0.0) * nxt_prices[t] for t in tickers)
            nw = broker.cash + value
            equity.append(nw)
            gross.append(value / nw if nw else 0.0)

        spy_data = {t: df[(df.index >= start) & ((df.index <= end) if end else True)] for t, df in full.items()}
        spy = spy_buy_hold(trading_window(spy_data), CFG.initial_capital)
        return {
            "days": len(equity) - 1, "orders": len(broker.orders),
            "avg_gross": float(np.mean(gross)), "end_gross": gross[-1],
            "live": metrics_from_equity(np.array(equity)),
            "spy": metrics_from_equity(np.asarray(spy, dtype=float)),
            "equity": equity, "raw": np.array(raws), "dates": dates, "tickers": tickers,
        }
    finally:
        for name, value in saved.items():
            setattr(live_trader, name, value)


def _parse(args: list[str]) -> dict:
    out = {}
    for a in args:
        k, v = a.split("=", 1)
        out[k] = float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
    return out


def row(label: str, r: dict) -> str:
    m = r["live"]
    return (f"{label:34s} {r['orders']:>6d} {r['avg_gross']:>7.0%} {m['total_return']:>+8.1%} "
            f"{m['sharpe']:>7.2f} {m['max_drawdown']:>7.1%}")


HEADER = f"{'variant':34s} {'orders':>6s} {'avg inv':>7s} {'return':>8s} {'sharpe':>7s} {'maxDD':>7s}"


def main() -> int:
    args = _parse(sys.argv[1:])
    start, end = args.pop("START", None), args.pop("END", None)
    r = replay(args, start=start, end=end)
    print(HEADER)
    print(row("production config" if len(sys.argv) == 1 else " ".join(sys.argv[1:]), r))
    s = r["spy"]
    print(f"{'SPY, same days':34s} {'':>6s} {'':>7s} {s['total_return']:>+8.1%} {s['sharpe']:>7.2f} {s['max_drawdown']:>7.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
