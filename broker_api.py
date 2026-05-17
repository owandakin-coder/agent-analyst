"""
broker_api.py
=============
ממשק ברוקר Alpaca – תומך במצב Paper ו-Live.

⚠️  DISCLAIMER
    מצב Live מחובר לכסף אמיתי. כל פקודה שתבוצע היא באחריות
    בלעדית של המשתמש. היוצרים אינם אחראים להפסדים כלשהם.
    לעולם אל תגדיר auto_approve=True אלא אם הבנת לחלוטין את הסיכון.
"""

from __future__ import annotations

import csv
import os
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    TradingClient = None
    StockHistoricalDataClient = None
    _ALPACA_AVAILABLE = False

# ─── טעינת משתני סביבה מקובץ .env ─────────────────────────────────────────
load_dotenv()

# ─── לוגר ──────────────────────────────────────────────────────────────────
LOG_FILE   = "paper_orders.log"
TRADES_CSV = "trades_history.csv"
log = logging.getLogger("BrokerAPI")

# ─── קבועים ────────────────────────────────────────────────────────────────
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL  = "https://api.alpaca.markets"


# ══════════════════════════════════════════════════════════════════════════════
# מחלקת Alpaca Broker
# ══════════════════════════════════════════════════════════════════════════════

class AlpacaBrokerAPI:
    """
    ממשק מסחר מחובר ל-Alpaca.

    Parameters
    ----------
    paper : bool
        True  = חשבון Paper (כסף וירטואלי, בטוח לבדיקות).
        False = חשבון Live  (כסף אמיתי – מסוכן!).
    auto_approve : bool
        False (ברירת מחדל) = כל פקודה תחכה לאישור ידני.
        True               = פקודות מבוצעות אוטומטית.
        ⚠️ לא לשנות ל-True אלא בהחלטה מודעת!
    """

    def __init__(
        self,
        paper: bool = True,
        auto_approve: bool = False,
    ):
        # ── טעינת קרדנציאלים ─────────────────────────────────────────────
        self.api_key    = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url_env    = os.getenv("ALPACA_BASE_URL", "")

        # base_url: env גובר על ה-flag paper
        if base_url_env:
            self.base_url = base_url_env
        else:
            self.base_url = PAPER_BASE_URL if paper else LIVE_BASE_URL

        self.paper        = paper
        self.auto_approve = auto_approve  # ⚠️ לא לשנות ל-True בלי הבנה מלאה

        self._validate_credentials()
        self._init_client()

        mode_label = "PAPER (virtual money)" if paper else "LIVE (REAL MONEY!)"
        log.info(f"AlpacaBrokerAPI initialized | Mode: {mode_label} | "
                 f"auto_approve={auto_approve} | base_url={self.base_url}")

        if not paper:
            self._live_warning()

    # ──────────────────────────────────────────────────────────────────────────
    # אתחול
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_credentials(self):
        """מוודא שהמפתחות קיימים."""
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
        """יוצר את לקוחות Alpaca."""
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            )
        self._trading = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )
        self._data = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

    @staticmethod
    def _live_warning():
        print("\n" + "!" * 60)
        print("  WARNING: LIVE MODE – REAL MONEY AT RISK")
        print("  All orders will affect your real brokerage account.")
        print("!" * 60 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # פקודות מסחר
    # ──────────────────────────────────────────────────────────────────────────

    def buy(self, ticker: str, shares: float, price: float | None = None) -> dict:
        """
        פקודת קנייה.
        אם auto_approve=False, מחכה לאישור המשתמש לפני ביצוע.
        """
        shares = max(1, int(shares))  # Alpaca דורש מניות שלמות
        order_info = {
            "side":    "BUY",
            "ticker":  ticker,
            "shares":  shares,
            "price":   price,
            "time":    datetime.now(timezone.utc).isoformat(),
        }

        if not self._request_approval(order_info):
            return self._log_rejected(order_info)

        return self._submit_order(ticker, shares, "buy")

    def sell(self, ticker: str, shares: float, price: float | None = None) -> dict:
        """
        פקודת מכירה.
        בודק שיש אחזקה מספקת לפני הגשה.
        """
        shares = max(1, int(shares))
        held   = self._get_held_shares(ticker)

        if held <= 0:
            log.warning(f"SELL rejected: no position in {ticker}")
            return {"status": "REJECTED", "reason": "no_position"}

        shares = min(shares, held)  # לא למכור יותר ממה שיש
        order_info = {
            "side":    "SELL",
            "ticker":  ticker,
            "shares":  shares,
            "price":   price,
            "time":    datetime.now(timezone.utc).isoformat(),
        }

        if not self._request_approval(order_info):
            return self._log_rejected(order_info)

        return self._submit_order(ticker, shares, "sell")

    def hold(self, ticker: str):
        """החזק – אין פעולה."""
        log.info(f"HOLD {ticker} (no order submitted)")

    # ──────────────────────────────────────────────────────────────────────────
    # אישור פקודה
    # ──────────────────────────────────────────────────────────────────────────

    def _request_approval(self, order_info: dict) -> bool:
        """
        מחכה לאישור המשתמש (או מאשר אוטומטית אם auto_approve=True).
        החזרת True = מאושר, False = נדחה.
        """
        msg = (
            f"\n>>> Order pending approval:\n"
            f"    {order_info['side']} {order_info['shares']} {order_info['ticker']}"
        )
        if order_info.get("price"):
            msg += f" @ ${order_info['price']:.2f}"
        msg += f"\n    Time: {order_info['time']}"

        # שליחה לטלגרם (אם מוגדר)
        self._notify_telegram(msg)

        if self.auto_approve:
            log.info(f"AUTO-APPROVED: {order_info['side']} {order_info['shares']} {order_info['ticker']}")
            return True

        # המתנה לאישור ידני דרך הקונסול
        print(msg)
        try:
            response = input("    Approve? [y/N]: ").strip().lower()
        except EOFError:
            # אין טרמינל אינטראקטיבי (למשל CI) – דחה
            response = "n"

        approved = response in ("y", "yes")
        status   = "APPROVED" if approved else "REJECTED"
        log.info(f"{status}: {order_info['side']} {order_info['shares']} {order_info['ticker']}")
        return approved

    # ──────────────────────────────────────────────────────────────────────────
    # גשת פקודה ל-Alpaca
    # ──────────────────────────────────────────────────────────────────────────

    def _submit_order(self, ticker: str, shares: int, side: str) -> dict:
        """מגיש פקודת Market Order ל-Alpaca."""
        alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        req = MarketOrderRequest(
            symbol=ticker,
            qty=int(shares),
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
        )

        try:
            order = self._trading.submit_order(req)
            result = {
                "order_id": str(order.id),
                "status":   str(order.status),
                "side":     side.upper(),
                "ticker":   ticker,
                "shares":   shares,
                "time":     datetime.now(timezone.utc).isoformat(),
            }
            log.info(
                f"ORDER SUBMITTED | {side.upper()} {shares} {ticker} | "
                f"id={order.id} status={order.status}"
            )
            self._write_log(result)
            return result

        except Exception as exc:
            log.error(f"ORDER FAILED | {side.upper()} {shares} {ticker} | {exc}")
            return {"status": "ERROR", "error": str(exc)}

    # ──────────────────────────────────────────────────────────────────────────
    # מידע על חשבון ופוזיציות
    # ──────────────────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """מחזיר פרטי חשבון (מזומן, equity וכו')."""
        acc = self._trading.get_account()
        return {
            "cash":            float(acc.cash),
            "equity":          float(acc.equity),
            "buying_power":    float(acc.buying_power),
            "portfolio_value": float(acc.portfolio_value),
            "status":          str(acc.status),
        }

    def get_cash(self) -> float:
        return self.get_account()["cash"]

    def get_positions(self) -> dict[str, float]:
        """מחזיר {ticker: shares} לכל הפוזיציות הפתוחות."""
        positions = {}
        try:
            for pos in self._trading.get_all_positions():
                positions[pos.symbol] = float(pos.qty)
        except Exception as exc:
            log.warning(f"Could not fetch positions: {exc}")
        return positions

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        """מחזיר מחיר אחרון עבור רשימת מניות."""
        prices = {}
        try:
            req    = StockLatestQuoteRequest(symbol_or_symbols=tickers)
            quotes = self._data.get_stock_latest_quote(req)
            for ticker in tickers:
                if ticker in quotes:
                    q = quotes[ticker]
                    bid = float(q.bid_price)
                    ask = float(q.ask_price)
                    # Use mid-price only when both sides are valid (market hours).
                    # Pre/post-market one side may be 0 — fall back gracefully.
                    if bid > 0 and ask > 0:
                        prices[ticker] = (bid + ask) / 2
                    elif ask > 0:
                        prices[ticker] = ask
                    elif bid > 0:
                        prices[ticker] = bid
                    # else: omit ticker; validate_prices() will use last-close fallback
        except Exception as exc:
            log.warning(f"Could not fetch prices: {exc}")
        return prices

    def is_market_open(self) -> bool:
        """בודק אם השוק פתוח כעת."""
        try:
            clock = self._trading.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False

    def next_market_open(self) -> datetime | None:
        """מחזיר מתי יפתח השוק הבא."""
        try:
            clock = self._trading.get_clock()
            return clock.next_open
        except Exception:
            return None

    def cancel_all_orders(self):
        """מבטל את כל הפקודות הפתוחות."""
        try:
            self._trading.cancel_orders()
            log.info("All open orders cancelled.")
        except Exception as exc:
            log.warning(f"Could not cancel orders: {exc}")

    def get_account_summary(self) -> dict:
        acc = self.get_account()
        return {
            "mode":       "PAPER" if self.paper else "LIVE",
            "auto_approve": self.auto_approve,
            **acc,
            "positions":  self.get_positions(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # עזר פנימי
    # ──────────────────────────────────────────────────────────────────────────

    def _get_held_shares(self, ticker: str) -> float:
        """כמה מניות של ticker מוחזקות כרגע."""
        return self.get_positions().get(ticker, 0.0)

    def _notify_telegram(self, message: str):
        """שולח הודעה לטלגרם אם הוגדרו פרטים (אופציונלי)."""
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        try:
            import urllib.request, urllib.parse, json
            url  = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
            urllib.request.urlopen(url, data, timeout=5)
        except Exception as exc:
            log.debug(f"Telegram notification failed: {exc}")

    def _log_rejected(self, order_info: dict) -> dict:
        result = {**order_info, "status": "REJECTED_BY_USER"}
        self._write_log(result)
        return result

    def _write_log(self, entry: dict):
        """Appends a structured trade record to the audit log and CSV.

        paper_orders.log is the explicit trade audit trail (written here).
        agent_analyst.log (configured in main.py) is the general process log.
        These are intentionally separate files.
        """
        log.info(f"TRADE RECORD | {entry}")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(str(entry) + "\n")
        except Exception as exc:
            log.warning(f"Failed to write to trade audit log ({LOG_FILE}): {exc}")
        self._write_csv(entry)

    def _write_csv(self, entry: dict):
        """Appends a trade record to trades_history.csv for later analysis."""
        fieldnames = ["time", "side", "ticker", "shares", "price",
                      "order_id", "status"]
        write_header = not os.path.exists(TRADES_CSV)
        try:
            with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow({k: entry.get(k, "") for k in fieldnames})
        except Exception as exc:
            log.warning(f"Failed to write trade to CSV ({TRADES_CSV}): {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Stub ישן – נשמר לתאימות אחורה ולמצב --mode live_stub
# ══════════════════════════════════════════════════════════════════════════════

class BrokerAPIStub:
    """
    ממשק מדומה (ללא חיבור לברוקר).
    משמש ב---mode live_stub לבדיקות ללא API.
    """

    def __init__(self, account_id: str = "PAPER_ACCOUNT_001"):
        self.account_id    = account_id
        self.positions: dict[str, float] = {}
        self.cash: float   = 0.0
        self.order_counter = 0
        log.info(f"[STUB] BrokerAPIStub initialized. Account: {account_id}")

    def buy(self, ticker: str, shares: float, price: float) -> dict:
        self.order_counter += 1
        oid = f"STUB-{self.order_counter:06d}"
        msg = f"[STUB] BUY {shares:.0f} {ticker} @ ${price:.2f}"
        log.info(msg)
        self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
        self.cash -= shares * price
        return {"order_id": oid, "status": "FILLED_STUB", "side": "BUY",
                "ticker": ticker, "shares": shares, "price": price}

    def sell(self, ticker: str, shares: float, price: float) -> dict:
        held = self.positions.get(ticker, 0.0)
        shares = min(shares, held)
        if shares <= 0:
            return {"status": "REJECTED", "reason": "no_position"}
        self.order_counter += 1
        oid = f"STUB-{self.order_counter:06d}"
        msg = f"[STUB] SELL {shares:.0f} {ticker} @ ${price:.2f}"
        log.info(msg)
        self.positions[ticker] = max(0.0, held - shares)
        self.cash += shares * price
        return {"order_id": oid, "status": "FILLED_STUB", "side": "SELL",
                "ticker": ticker, "shares": shares, "price": price}

    def hold(self, ticker: str):
        log.info(f"[STUB] HOLD {ticker}")

    def get_cash(self) -> float:
        return self.cash

    def set_cash(self, amount: float):
        self.cash = amount

    def get_positions(self) -> dict:
        return dict(self.positions)

    def is_market_open(self) -> bool:
        return True

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        return {}

    def get_account_summary(self) -> dict:
        return {"mode": "STUB", "cash": self.cash, "positions": self.positions}


# ── ייצוא ידידותי ──────────────────────────────────────────────────────────
# BrokerAPI = BrokerAPIStub  (ברירת מחדל ישנה)
BrokerAPI = BrokerAPIStub
