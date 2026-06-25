from __future__ import annotations

import math
import os

from broker_api import AlpacaBrokerAPI

try:
    from config_loader import CFG

    TARGET_CASH_BUFFER_PCT = CFG.live_cash_buffer_pct
    TARGET_GROSS_EXPOSURE = CFG.live_max_gross_exposure
    TARGET_MAX_POSITIONS = CFG.live_max_positions
except Exception:
    TARGET_CASH_BUFFER_PCT = 0.30
    TARGET_GROSS_EXPOSURE = 0.65
    TARGET_MAX_POSITIONS = 8


def main() -> int:
    broker = AlpacaBrokerAPI(paper=True, auto_approve=True)
    snapshot = broker.reconcile_account_state()
    prices = broker.get_latest_prices(list(snapshot["positions"].keys()))
    details = snapshot.get("position_details") or {}

    cash = float(snapshot.get("cash", 0.0))
    equity = float(snapshot.get("equity", 0.0))
    positions = {ticker: float(qty) for ticker, qty in (snapshot.get("positions") or {}).items() if float(qty or 0.0) > 0}
    market_value = sum(positions.get(ticker, 0.0) * prices.get(ticker, 0.0) for ticker in positions)
    reserve_cash = equity * TARGET_CASH_BUFFER_PCT

    ranked = []
    for ticker, qty in positions.items():
        price = prices.get(ticker, 0.0)
        if price <= 0:
            continue
        unrealized = float((details.get(ticker) or {}).get("unrealized_pl", 0.0))
        ranked.append((unrealized, ticker, qty, price))
    ranked.sort(key=lambda item: item[0])

    sold = []
    for unrealized, ticker, qty, price in ranked:
        positions_count = sum(1 for value in positions.values() if value > 0)
        gross_exposure = market_value / equity if equity > 0 else 0.0
        cash_ok = cash >= reserve_cash
        exposure_ok = gross_exposure <= TARGET_GROSS_EXPOSURE
        positions_ok = positions_count <= TARGET_MAX_POSITIONS
        if cash_ok and exposure_ok and positions_ok:
            break

        if positions_count > TARGET_MAX_POSITIONS:
            shares_to_sell = int(qty)
        else:
            target_sale_value = max(reserve_cash - cash, 0.0)
            target_excess_value = max(market_value - (equity * TARGET_GROSS_EXPOSURE), 0.0)
            required_value = max(target_sale_value, target_excess_value, price)
            shares_to_sell = min(int(qty), max(1, int(math.ceil(required_value / price))))

        result = broker.sell(ticker, shares_to_sell, price)
        if result.get("status") in ("ERROR", "REJECTED", "REJECTED_BY_USER"):
            continue

        sold.append({"ticker": ticker, "shares": shares_to_sell, "price": price, "unrealized_pl": unrealized})
        positions[ticker] = max(0.0, positions[ticker] - shares_to_sell)
        cash += shares_to_sell * price
        market_value -= shares_to_sell * price

    print(
        {
            "sold": sold,
            "cash_after_estimate": round(cash, 2),
            "gross_exposure_after_estimate": round((market_value / equity) if equity > 0 else 0.0, 4),
            "positions_after_estimate": sum(1 for value in positions.values() if value > 0),
            "target_cash_buffer_pct": TARGET_CASH_BUFFER_PCT,
            "target_gross_exposure": TARGET_GROSS_EXPOSURE,
            "target_max_positions": TARGET_MAX_POSITIONS,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
