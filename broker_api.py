"""
Broker integrations for Alpaca paper/live trading and a local stub.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    TradingClient = None
    StockHistoricalDataClient = None
    GetOrdersRequest = None
    MarketOrderRequest = None
    OrderSide = None
    QueryOrderStatus = None
    TimeInForce = None
    _ALPACA_AVAILABLE = False

try:
    from config_loader import CFG
    LOG_FILE = str(Path(CFG.logs_dir) / "paper_orders.log")
    TRADES_CSV = "trades_history.csv"
    SUBMITTED_ORDERS_FILE = CFG.broker_submitted_orders_file
    DUPLICATE_WINDOW_DAYS = CFG.broker_duplicate_window_days
    RECENT_ORDERS_LIMIT = CFG.broker_recent_orders_limit
except Exception:
    LOG_FILE = "paper_orders.log"
    TRADES_CSV = "trades_history.csv"
    SUBMITTED_ORDERS_FILE = "logs/submitted_orders.json"
    DUPLICATE_WINDOW_DAYS = 1
    RECENT_ORDERS_LIMIT = 100

load_dotenv()

log = logging.getLogger("BrokerAPI")

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


class AlpacaBrokerAPI:
    def __init__(self, paper: bool = True, auto_approve: bool = False):
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv("ALPACA_BASE_URL", "") or (PAPER_BASE_URL if paper else LIVE_BASE_URL)
        self.paper = paper
        self.auto_approve = auto_approve

        self._idempotency_path = Path(SUBMITTED_ORDERS_FILE)
        self._submitted_orders = self._load_submitted_orders()
        self._submitted_keys = set(self._submitted_orders.keys())

        self._validate_credentials()
        self._init_client()

        mode_label = "PAPER (virtual money)" if paper else "LIVE (REAL MONEY!)"
        log.info(
            "AlpacaBrokerAPI initialized | Mode: %s | auto_approve=%s | base_url=%s",
            mode_label,
            auto_approve,
            self.base_url,
        )

        if not paper:
            self._live_warning()

    def _validate_credentials(self):
        missing = []
        if not self.api_key:
            missing.append("ALPACA_API_KEY")
        if not self.secret_key:
            missing.append("ALPACA_SECRET_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing environment variables: {missing}\n"
                "Copy .env.example to .env and fill in your Alpaca credentials."
            )

    def _init_client(self):
        if not _ALPACA_AVAILABLE:
            raise ImportError("alpaca-py is not installed. Run: pip install alpaca-py")
        self._trading = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=self.paper)
        self._data = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)

    @staticmethod
    def _live_warning():
        print("\n" + "!" * 60)
        print("  WARNING: LIVE MODE - REAL MONEY AT RISK")
        print("  All orders will affect your real brokerage account.")
        print("!" * 60 + "\n")

    def buy(self, ticker: str, shares: float, price: float | None = None) -> dict:
        shares = max(1, int(shares))
        order_info = {
            "side": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        if not self._request_approval(order_info):
            return self._log_rejected(order_info)
        return self._submit_order(ticker, shares, "buy")

    def sell(self, ticker: str, shares: float, price: float | None = None) -> dict:
        shares = max(1, int(shares))
        snapshot = self.reconcile_account_state()
        held = snapshot["positions"].get(ticker, 0.0)
        if held <= 0:
            log.warning("SELL rejected: no position in %s", ticker)
            return {"status": "REJECTED", "reason": "no_position"}

        shares = min(shares, int(held))
        order_info = {
            "side": "SELL",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        if not self._request_approval(order_info):
            return self._log_rejected(order_info)
        return self._submit_order(ticker, shares, "sell", account_snapshot=snapshot)

    def hold(self, ticker: str):
        log.info("HOLD %s (no order submitted)", ticker)

    def _request_approval(self, order_info: dict) -> bool:
        msg = (
            f"\n>>> Order pending approval:\n"
            f"    {order_info['side']} {order_info['shares']} {order_info['ticker']}"
        )
        if order_info.get("price"):
            msg += f" @ ${order_info['price']:.2f}"
        msg += f"\n    Time: {order_info['time']}"

        self._notify_telegram(msg)

        if self.auto_approve:
            log.info(
                "AUTO-APPROVED: %s %s %s",
                order_info["side"],
                order_info["shares"],
                order_info["ticker"],
            )
            return True

        print(msg)
        try:
            response = input("    Approve? [y/N]: ").strip().lower()
        except EOFError:
            response = "n"

        approved = response in ("y", "yes")
        status = "APPROVED" if approved else "REJECTED"
        log.info("%s: %s %s %s", status, order_info["side"], order_info["shares"], order_info["ticker"])
        return approved

    def _submit_order(
        self,
        ticker: str,
        shares: int,
        side: str,
        account_snapshot: dict | None = None,
    ) -> dict:
        if shares <= 0:
            return {"status": "REJECTED", "reason": "invalid_shares"}

        key = self._order_key(ticker, side, shares)
        client_order_id = self._client_order_id_from_key(key)

        duplicate = self._check_duplicate_order(ticker, side, shares, client_order_id)
        if duplicate:
            log.warning(
                "DUPLICATE ORDER BLOCKED | %s %s %s | source=%s",
                side.upper(),
                shares,
                ticker,
                duplicate["source"],
            )
            return {
                "status": "DUPLICATE_BLOCKED",
                "ticker": ticker,
                "side": side.upper(),
                "shares": shares,
                "source": duplicate["source"],
                "client_order_id": client_order_id,
            }

        snapshot = account_snapshot or self.reconcile_account_state()
        if side == "sell" and snapshot["positions"].get(ticker, 0.0) < shares:
            log.warning("SELL rejected after reconciliation: insufficient shares in %s", ticker)
            return {"status": "REJECTED", "reason": "insufficient_position"}

        alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=ticker,
            qty=int(shares),
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )

        try:
            order = self._trading.submit_order(request)
            now_iso = datetime.now(timezone.utc).isoformat()
            result = {
                "order_id": str(order.id),
                "client_order_id": str(getattr(order, "client_order_id", client_order_id) or client_order_id),
                "status": str(order.status),
                "side": side.upper(),
                "ticker": ticker,
                "shares": shares,
                "time": now_iso,
            }
            log.info(
                "ORDER SUBMITTED | %s %s %s | id=%s status=%s client_order_id=%s",
                side.upper(),
                shares,
                ticker,
                order.id,
                order.status,
                result["client_order_id"],
            )
            self._record_order_key(key, result)
            self._write_log(result)
            return result
        except Exception as exc:
            log.error("ORDER FAILED | %s %s %s | %s", side.upper(), shares, ticker, exc)
            return {"status": "ERROR", "error": str(exc)}

    def _order_key(self, ticker: str, side: str, shares: int) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"{today}:{ticker}:{side.upper()}:{shares}"

    def _client_order_id_from_key(self, key: str) -> str:
        return f"ATZMA-{key.replace(':', '-')}"

    def _check_duplicate_order(self, ticker: str, side: str, shares: int, client_order_id: str) -> dict | None:
        key = self._order_key(ticker, side, shares)
        if key in self._submitted_keys:
            return {"source": "local_state"}

        try:
            order = self._find_matching_broker_order(ticker, side, shares, client_order_id)
        except Exception as exc:
            log.warning("Duplicate check against broker failed: %s", exc)
            return None

        if order is not None:
            return {"source": "broker_history", "order_id": getattr(order, "id", None)}
        return None

    def _find_matching_broker_order(self, ticker: str, side: str, shares: int, client_order_id: str):
        if GetOrdersRequest is None or QueryOrderStatus is None:
            return None

        after = datetime.now(timezone.utc) - timedelta(days=max(DUPLICATE_WINDOW_DAYS, 1))
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=RECENT_ORDERS_LIMIT, after=after)
        orders = self._trading.get_orders(filter=request)

        for order in orders or []:
            order_side = str(getattr(order, "side", "")).split(".")[-1].upper()
            order_symbol = getattr(order, "symbol", "")
            order_qty = int(float(getattr(order, "qty", 0) or 0))
            order_client_id = getattr(order, "client_order_id", "")
            if order_client_id == client_order_id:
                return order
            if order_symbol == ticker and order_side == side.upper() and order_qty == shares:
                submitted_at = getattr(order, "submitted_at", None)
                if submitted_at is None or self._same_utc_day(submitted_at, datetime.now(timezone.utc)):
                    return order
        return None

    def _same_utc_day(self, dt1, dt2: datetime) -> bool:
        if dt1 is None:
            return False
        if getattr(dt1, "tzinfo", None) is None:
            dt1 = dt1.replace(tzinfo=timezone.utc)
        return dt1.astimezone(timezone.utc).date() == dt2.astimezone(timezone.utc).date()

    def _load_submitted_orders(self) -> dict[str, dict]:
        try:
            if self._idempotency_path.exists():
                with open(self._idempotency_path, encoding="utf-8") as handle:
                    data = json.load(handle) or {}
                    if isinstance(data, dict):
                        return data
        except Exception as exc:
            log.warning("Could not load submitted order state: %s", exc)
        return {}

    def _save_submitted_orders(self):
        try:
            self._idempotency_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._idempotency_path, "w", encoding="utf-8") as handle:
                json.dump(self._submitted_orders, handle, ensure_ascii=True, indent=2, sort_keys=True)
        except Exception as exc:
            log.warning("Could not save submitted order state: %s", exc)

    def _record_order_key(self, key: str, result: dict):
        self._submitted_keys.add(key)
        self._submitted_orders[key] = {
            "time": result.get("time"),
            "ticker": result.get("ticker"),
            "side": result.get("side"),
            "shares": result.get("shares"),
            "client_order_id": result.get("client_order_id"),
            "order_id": result.get("order_id"),
            "status": result.get("status"),
        }
        self._submitted_orders = self._prune_submitted_orders(self._submitted_orders)
        self._submitted_keys = set(self._submitted_orders.keys())
        self._save_submitted_orders()

    def _prune_submitted_orders(self, submitted: dict[str, dict]) -> dict[str, dict]:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(DUPLICATE_WINDOW_DAYS, 1))
        kept: dict[str, dict] = {}
        for key, payload in submitted.items():
            key_date = key.split(":", 1)[0]
            try:
                day = datetime.strptime(key_date, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day >= cutoff:
                kept[key] = payload
        return kept

    def get_account(self) -> dict:
        acc = self._trading.get_account()
        return {
            "cash": float(acc.cash),
            "equity": float(acc.equity),
            "buying_power": float(acc.buying_power),
            "portfolio_value": float(acc.portfolio_value),
            "status": str(acc.status),
        }

    def get_cash(self) -> float:
        return self.get_account()["cash"]

    def get_positions(self) -> dict[str, float]:
        positions: dict[str, float] = {}
        try:
            for pos in self._trading.get_all_positions():
                symbol = getattr(pos, "symbol", None)
                qty = getattr(pos, "qty", None)
                if not symbol or qty in (None, ""):
                    continue
                positions[symbol] = float(qty)
        except Exception as exc:
            log.warning("Could not fetch positions: %s", exc)
        return positions

    def reconcile_account_state(self) -> dict:
        account = self.get_account()
        positions = self.get_positions()
        return {
            "cash": account["cash"],
            "equity": account["equity"],
            "buying_power": account["buying_power"],
            "portfolio_value": account["portfolio_value"],
            "status": account["status"],
            "positions": positions,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=tickers)
            quotes = self._data.get_stock_latest_quote(request)
            for ticker in tickers:
                if ticker not in quotes:
                    continue
                quote = quotes[ticker]
                bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
                ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
                if bid > 0 and ask > 0:
                    prices[ticker] = (bid + ask) / 2
                elif ask > 0:
                    prices[ticker] = ask
                elif bid > 0:
                    prices[ticker] = bid
        except Exception as exc:
            log.warning("Could not fetch prices: %s", exc)
        return prices

    def is_market_open(self) -> bool:
        try:
            return bool(self._trading.get_clock().is_open)
        except Exception:
            return False

    def next_market_open(self) -> datetime | None:
        try:
            return self._trading.get_clock().next_open
        except Exception:
            return None

    def cancel_all_orders(self):
        try:
            self._trading.cancel_orders()
            log.info("All open orders cancelled.")
        except Exception as exc:
            log.warning("Could not cancel orders: %s", exc)

    def get_account_summary(self) -> dict:
        snapshot = self.reconcile_account_state()
        return {"mode": "PAPER" if self.paper else "LIVE", "auto_approve": self.auto_approve, **snapshot}

    def _get_held_shares(self, ticker: str) -> float:
        return self.get_positions().get(ticker, 0.0)

    def _notify_telegram(self, message: str):
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        try:
            import urllib.parse
            import urllib.request

            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
            urllib.request.urlopen(url, data, timeout=5)
        except Exception as exc:
            log.debug("Telegram notification failed: %s", exc)

    def _log_rejected(self, order_info: dict) -> dict:
        result = {**order_info, "status": "REJECTED_BY_USER"}
        self._write_log(result)
        return result

    def _write_log(self, entry: dict):
        log.info("TRADE RECORD | %s", entry)
        try:
            Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except Exception as exc:
            log.warning("Failed to write to trade audit log (%s): %s", LOG_FILE, exc)
        self._write_csv(entry)

    def _write_csv(self, entry: dict):
        fieldnames = ["time", "side", "ticker", "shares", "price", "order_id", "client_order_id", "status"]
        write_header = not os.path.exists(TRADES_CSV)
        try:
            with open(TRADES_CSV, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow({key: entry.get(key, "") for key in fieldnames})
        except Exception as exc:
            log.warning("Failed to write trade to CSV (%s): %s", TRADES_CSV, exc)


class BrokerAPIStub:
    def __init__(self, account_id: str = "PAPER_ACCOUNT_001"):
        self.account_id = account_id
        self.positions: dict[str, float] = {}
        self.cash = 0.0
        self.order_counter = 0
        log.info("[STUB] BrokerAPIStub initialized. Account: %s", account_id)

    def buy(self, ticker: str, shares: float, price: float) -> dict:
        self.order_counter += 1
        order_id = f"STUB-{self.order_counter:06d}"
        self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
        self.cash -= shares * price
        return {
            "order_id": order_id,
            "status": "FILLED_STUB",
            "side": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": price,
        }

    def sell(self, ticker: str, shares: float, price: float) -> dict:
        held = self.positions.get(ticker, 0.0)
        shares = min(shares, held)
        if shares <= 0:
            return {"status": "REJECTED", "reason": "no_position"}
        self.order_counter += 1
        order_id = f"STUB-{self.order_counter:06d}"
        self.positions[ticker] = max(0.0, held - shares)
        self.cash += shares * price
        return {
            "order_id": order_id,
            "status": "FILLED_STUB",
            "side": "SELL",
            "ticker": ticker,
            "shares": shares,
            "price": price,
        }

    def hold(self, ticker: str):
        log.info("[STUB] HOLD %s", ticker)

    def get_cash(self) -> float:
        return self.cash

    def set_cash(self, amount: float):
        self.cash = amount

    def get_positions(self) -> dict:
        return dict(self.positions)

    def reconcile_account_state(self) -> dict:
        cash = self.get_cash()
        positions = self.get_positions()
        equity = cash
        return {
            "cash": cash,
            "equity": equity,
            "buying_power": cash,
            "portfolio_value": equity,
            "status": "ACTIVE",
            "positions": dict(positions),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def is_market_open(self) -> bool:
        return True

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        return {}

    def get_account(self) -> dict:
        snapshot = self.reconcile_account_state()
        return {
            "cash": snapshot["cash"],
            "equity": snapshot["equity"],
            "buying_power": snapshot["buying_power"],
            "portfolio_value": snapshot["portfolio_value"],
            "status": snapshot["status"],
        }

    def get_account_summary(self) -> dict:
        return {"mode": "STUB", "cash": self.cash, "positions": self.positions}


BrokerAPI = BrokerAPIStub
