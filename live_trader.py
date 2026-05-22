"""
live_trader.py
==============
לולאת המסחר החי (Live / Paper-Live).
מחבר את מודל ה-RL לממשק Alpaca ומריץ מחזורי החלטה יומיים.

⚠️  לצרכי מחקר בלבד. כל פעולה על כסף אמיתי באחריות המשתמש.
"""

from __future__ import annotations

import os
import time
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regime_detector import RegimeDetector
from alternative_data import AlternativeDataFetcher, enrich_observation

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from broker_api import AlpacaBrokerAPI
    from risk_manager import RiskManager
    from data_manager import DataManager
    from ensemble_agent import EnsembleAgent

log = logging.getLogger("LiveTrader")

# ─── קבועים ────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 30   # חלון תצפית (ימים) – חייב להתאים לאימון
FEATURE_COLS   = [    # פיצ'רים שהמודל ראה באימון
    "returns", "log_returns",
    "price_to_ma20", "price_to_ma50", "ma_cross",
    "rsi", "macd_hist", "boll_pct", "boll_width",
    "atr_pct", "volume_ratio", "volatility_20",
]
MIN_DATA_DAYS  = 200  # enough rows after dropna (z-score warmup ~78 + WINDOW_SIZE 30)


# ══════════════════════════════════════════════════════════════════════════════
class LiveTrader:
    """
    מנהל מחזור ה-RL בזמן אמת:
    1. שולף נתונים היסטוריים עדכניים
    2. בונה תצפית (observation)
    3. מקבל החלטה מהמודל
    4. מתרגם לפקודות ברוקר
    5. מפעיל ניהול סיכונים
    6. שולח לברוקר (עם/בלי אישור)
    """

    def __init__(
        self,
        model: "PPO | EnsembleAgent",
        broker: "AlpacaBrokerAPI",
        data_manager: "DataManager",
        risk_manager: "RiskManager",
        vec_norm: "VecNormalize | None",
        tickers: list[str],
        initial_capital: float = 100_000.0,
    ):
        self.model          = model
        self.broker         = broker
        self.data_manager   = data_manager
        self.risk_manager   = risk_manager
        self.vec_norm       = vec_norm      # None when using EnsembleAgent
        self.tickers        = tickers
        self.initial_capital = initial_capital
        self.num_stocks     = len(tickers)

        # Detect ensemble mode (EnsembleAgent has its own normalisation)
        self._is_ensemble = hasattr(model, "members")

        # Regime detection + alternative data
        self._regime_detector = RegimeDetector()
        self._alt_fetcher     = AlternativeDataFetcher()

        # מעקב שווי תיק
        self._peak_net_worth = initial_capital

        # Per-stock trailing stop-loss tracking
        self._entry_prices:   dict[str, float] = {}
        self._trailing_highs: dict[str, float] = {}   # max price since entry
        self.stop_loss_pct: float = 0.08   # sell if price drops 8% from trailing high

    # ──────────────────────────────────────────────────────────────────────────
    # לולאה ראשית
    # ──────────────────────────────────────────────────────────────────────────

    def run_loop(self, poll_seconds: int = 60):
        """
        לולאה אינסופית: בכל פעם שהשוק פתוח – מריץ החלטה אחת ומחכה ליום הבא.

        Parameters
        ----------
        poll_seconds : int
            כמה שניות לחכות בין בדיקות האם השוק פתוח.
        """
        log.info("Live trading loop started. Press Ctrl+C to stop.")
        self._notify_startup()
        try:
            while True:
                try:
                    if self.broker.is_market_open():
                        log.info("Market is OPEN – running decision cycle.")
                        self.run_once()
                        # ישן עד לפתיחת השוק הבאה (+ 5 דקות חיץ)
                        sleep_secs = self._seconds_until_next_open(buffer_minutes=5)
                        log.info(
                            f"Decision executed. Sleeping {sleep_secs/3600:.1f}h "
                            f"until next market open."
                        )
                        time.sleep(sleep_secs)
                    else:
                        next_open = self.broker.next_market_open()
                        log.info(
                            f"Market CLOSED. Next open: {next_open}. "
                            f"Checking again in {poll_seconds}s."
                        )
                        time.sleep(poll_seconds)

                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error(f"Unexpected error in live loop: {exc}", exc_info=True)
                    log.info(f"Retrying in {poll_seconds}s ...")
                    time.sleep(poll_seconds)

        except KeyboardInterrupt:
            log.info("Live loop stopped by user (Ctrl+C).")
        finally:
            self._notify_shutdown()

    def run_once(self):
        """
        מחזור החלטה יחיד:
        בנה תצפית → חזה פעולה → בדוק סיכון → שלח פקודות.
        """
        # ── נתונים עדכניים ──────────────────────────────────────────────────
        log.info("Fetching latest market data ...")
        try:
            fresh_data = self._fetch_fresh_data()
        except Exception as exc:
            log.error(f"Failed to fetch data: {exc}")
            return

        # ── מחירים נוכחיים ──────────────────────────────────────────────────
        current_prices = self.broker.get_latest_prices(self.tickers)
        if not current_prices:
            current_prices = {
                t: float(fresh_data[t]["close"].iloc[-1])
                for t in self.tickers if t in fresh_data
            }
        current_prices = self.validate_prices(current_prices, fresh_data)
        log.info(f"Prices: {current_prices}")

        # ── שווי תיק עדכני ──────────────────────────────────────────────────
        positions = self.broker.get_positions()
        cash      = self.broker.get_cash()
        net_worth = cash + sum(
            positions.get(t, 0.0) * current_prices.get(t, 0.0)
            for t in self.tickers
        )
        self._peak_net_worth = max(self._peak_net_worth, net_worth)
        drawdown = (self._peak_net_worth - net_worth) / (self._peak_net_worth + 1e-9)

        log.info(
            f"Portfolio | Net Worth: ${net_worth:,.0f} | "
            f"Cash: ${cash:,.0f} | Drawdown: {drawdown:.1%}"
        )

        # ── ניהול סיכונים ───────────────────────────────────────────────────
        prev_halted = self.risk_manager.is_halted
        risk_level  = self.risk_manager.update(net_worth)
        if self.risk_manager.is_halted:
            log.warning("TRADING HALTED by RiskManager. Skipping this cycle.")
            if not prev_halted:
                self._notify_halt(net_worth, drawdown)
            return

        # ── Regime detection ─────────────────────────────────────────────────
        regime_signal = self._detect_regime(fresh_data)
        if regime_signal is not None:
            multiplier = regime_signal.regime.position_multiplier()
            self.risk_manager.set_regime_multiplier(multiplier)
            log.info(
                f"Regime: {regime_signal.regime.value} "
                f"(conf={regime_signal.confidence:.0%}, multiplier={multiplier}) | "
                f"{regime_signal.description}"
            )

        # ── Alternative data ─────────────────────────────────────────────────
        alt_data = self._alt_fetcher.fetch_all()

        # ── תצפית למודל ─────────────────────────────────────────────────────
        obs = self._build_observation(fresh_data, cash, net_worth, drawdown)
        if obs is None:
            log.error("Could not build observation. Skipping cycle.")
            return

        # ── חיזוי פעולה ─────────────────────────────────────────────────────
        # NOTE: enrich_observation (alt data features) is only appended AFTER
        # normalisation, and only when the model was trained with those extra
        # features (i.e. after a retrain that includes alt-data in obs space).
        # The current deployed model was trained on the original 183-feature obs,
        # so we skip enrichment here to avoid shape mismatch with VecNormalize.
        # Alt data still influences decisions via the regime multiplier above.
        if self._is_ensemble:
            # EnsembleAgent handles normalisation internally per member
            action = self.model.predict(obs[np.newaxis], deterministic=True)
            log.info(f"Ensemble vote summary: {self.model.vote_summary(obs[np.newaxis])}")
        else:
            obs_norm = self.vec_norm.normalize_obs(obs[np.newaxis])  # (1, W, F)
            action, _ = self.model.predict(obs_norm, deterministic=True)
            action = np.array(action).flatten()

        # התאמת גודל פוזיציה לפי סיכון (with regime multiplier + kelly + correlation)
        price_history = {t: fresh_data[t]["close"] for t in self.tickers if t in fresh_data}
        action = self.risk_manager.scale_action(
            action,
            tickers=self.tickers,
            price_history=price_history,
        )
        log.info(f"Scaled action: {action.round(3)}")

        # ── המרה לפקודות ────────────────────────────────────────────────────
        self._execute_actions(action, current_prices, cash, positions)

        # ── דוח יומי לטלגרם ─────────────────────────────────────────────────
        self._send_daily_summary(net_worth, cash, drawdown, action, current_prices)

    def _seconds_until_next_open(self, buffer_minutes: int = 5) -> float:
        """מחשב שניות עד לפתיחת השוק הבאה (+ חיץ בדקות)."""
        try:
            next_open = self.broker.next_market_open()
            if next_open is not None:
                now = datetime.now(timezone.utc)
                if next_open.tzinfo is None:
                    next_open = next_open.replace(tzinfo=timezone.utc)
                delta = (next_open - now).total_seconds() + buffer_minutes * 60
                return max(delta, 60.0)
        except Exception as exc:
            log.warning(f"Could not get next market open: {exc}")
        # fallback: 23 שעות
        return 23 * 3600

    # ──────────────────────────────────────────────────────────────────────────
    # שליפת נתונים היסטוריים עדכניים
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_fresh_data(self) -> dict[str, pd.DataFrame]:
        """
        מוריד MIN_DATA_DAYS+ ימים של נתונים היסטוריים (מ-yfinance)
        ומחשב פיצ'רים. זהו אותו DataManager שהשתמשנו בו באימון.
        """
        from datetime import date
        end   = date.today().isoformat()
        # לוקח 20% יותר ימים כדי לפצות על ימי חופשה
        days_back = int(MIN_DATA_DAYS * 1.3)
        start = (date.today() - timedelta(days=days_back)).isoformat()

        # עדכון טמפורלי של DataManager
        self.data_manager.start = start
        self.data_manager.end   = end

        data = self.data_manager.load_all(force_download=True)
        return data

    def validate_prices(
        self,
        prices: dict[str, float],
        fresh_data: dict[str, pd.DataFrame],
        min_price: float = 1.0,
        max_price: float = 10_000.0,
    ) -> dict[str, float]:
        """
        בודק מחירים חריגים ומחליף ב-fallback (סגירה אחרונה מהנתונים).
        מחיר תקין: min_price <= price <= max_price.
        """
        validated = {}
        for ticker, price in prices.items():
            if min_price <= price <= max_price:
                validated[ticker] = price
            else:
                # fallback: מחיר סגירה אחרון מה-fresh_data
                fallback = None
                if ticker in fresh_data and not fresh_data[ticker].empty:
                    fallback = float(fresh_data[ticker]["close"].iloc[-1])
                if fallback and min_price <= fallback <= max_price:
                    log.warning(
                        f"PRICE INVALID: {ticker}=${price:.2f} out of "
                        f"[{min_price}, {max_price}]. Using fallback ${fallback:.2f}."
                    )
                    validated[ticker] = fallback
                else:
                    log.warning(
                        f"PRICE INVALID: {ticker}=${price:.2f}, no valid fallback. Skipping."
                    )
        return validated

    # ──────────────────────────────────────────────────────────────────────────
    # בניית תצפית
    # ──────────────────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        data: dict[str, pd.DataFrame],
        cash: float,
        net_worth: float,
        drawdown: float,
    ) -> np.ndarray | None:
        """
        בונה מטריצת תצפית (window_size, num_stocks * features + 3)
        בדיוק כפי שה-TradingEnvironment בונה אותה בזמן אימון.
        """
        frames = []
        for ticker in self.tickers:
            df = data.get(ticker)
            if df is None or len(df) < WINDOW_SIZE:
                log.error(f"Not enough data for {ticker}: {len(df) if df is not None else 0} rows")
                return None

            # בחר עמודות קיימות בלבד
            available = [c for c in FEATURE_COLS if c in df.columns]
            if len(available) < len(FEATURE_COLS):
                missing = set(FEATURE_COLS) - set(available)
                log.warning(f"Missing features for {ticker}: {missing}")

            slice_df = df[available].iloc[-WINDOW_SIZE:].copy()

            # z-score נרמול תוך-חלון (כמו בסביבת האימון)
            normed = (slice_df - slice_df.mean()) / (slice_df.std() + 1e-9)
            frames.append(normed.values)

        if not frames:
            return None

        # פיצ'רי תיק (3 ערכים): cash_ratio, pnl, drawdown
        cash_ratio     = cash / (net_worth + 1e-9)
        unrealized_pnl = (net_worth - self.initial_capital) / self.initial_capital
        portfolio_row  = np.array([cash_ratio, unrealized_pnl, drawdown])
        portfolio_block = np.tile(portfolio_row, (WINDOW_SIZE, 1))

        obs = np.concatenate(frames + [portfolio_block], axis=1).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # ביצוע פקודות
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_actions(
        self,
        action: np.ndarray,
        prices: dict[str, float],
        cash: float,
        positions: dict[str, float],
    ):
        """
        ממיר וקטור פעולות [-1,1] לפקודות קנייה/מכירה בפועל.

        אסטרטגיה:
        - action[i] < -0.05  → מכור פרופורציה מהאחזקה
        - action[i] >  0.05  → קנה פרופורציה מהמזומן הפנוי
        - ריכוזיות > 30%     → מדלג על קנייה
        - אחרת → החזק
        """
        BUY_THRESHOLD  =  0.05
        SELL_THRESHOLD = -0.05

        # ── Trailing Stop-Loss ────────────────────────────────────────────────
        # Updates the high-water mark per ticker and sells if price drops
        # more than stop_loss_pct below that mark (protects accumulated gains).
        for ticker in list(positions.keys()):
            held  = positions.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            entry = self._entry_prices.get(ticker, 0.0)
            if held <= 0 or entry <= 0 or price <= 0:
                continue

            # Update trailing high-water mark
            prev_high = self._trailing_highs.get(ticker, entry)
            trail_high = max(prev_high, price)
            self._trailing_highs[ticker] = trail_high

            # Check if price has fallen stop_loss_pct below the trailing high
            drop_pct = (trail_high - price) / trail_high
            if drop_pct >= self.stop_loss_pct:
                log.warning(
                    f"TRAILING STOP triggered: {ticker} dropped {drop_pct:.1%} "
                    f"from high ${trail_high:.2f} → now ${price:.2f}. Selling all."
                )
                result = self.broker.sell(ticker, int(held), price)
                if result.get("status") not in ("ERROR", "REJECTED", "REJECTED_BY_USER"):
                    # Record Kelly outcome
                    entry = self._entry_prices.get(ticker, 0.0)
                    if entry > 0:
                        pnl_pct = (price - entry) / entry
                        self.risk_manager.record_trade_outcome(ticker, pnl_pct)
                    self._entry_prices.pop(ticker, None)
                    self._trailing_highs.pop(ticker, None)
                    self._telegram(
                        f"🛑 *Trailing Stop triggered*\n"
                        f"{ticker}: -{drop_pct:.1%} from high\n"
                        f"High: ${trail_high:.2f} → Now: ${price:.2f}\n"
                        f"Sold {int(held)} shares."
                    )

        # ── מכירות קודם (לשחרר מזומן) ──────────────────────────────────────
        sells_executed = 0
        for i, ticker in enumerate(self.tickers):
            act   = float(action[i])
            held  = positions.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)

            if act < SELL_THRESHOLD and held > 0 and price > 0:
                shares_to_sell = max(1, int(held * abs(act)))
                log.info(f"Action {act:.3f} -> SELL {shares_to_sell} {ticker} @ ${price:.2f}")
                result = self.broker.sell(ticker, shares_to_sell, price)
                if result.get("status") in ("ERROR", "REJECTED", "REJECTED_BY_USER"):
                    log.warning(f"SELL {ticker} failed: {result}")
                else:
                    sells_executed += 1
                    # Record outcome for Kelly Criterion — P&L from entry to exit
                    entry = self._entry_prices.get(ticker, 0.0)
                    if entry > 0:
                        pnl_pct = (price - entry) / entry
                        self.risk_manager.record_trade_outcome(ticker, pnl_pct)
                        log.debug(f"Kelly record: {ticker} pnl={pnl_pct:+.2%}")

        # ── רענון מזומן אחרי מכירות ─────────────────────────────────────────
        if sells_executed > 0:
            try:
                cash = self.broker.get_cash()
                log.info(f"Cash after sells: ${cash:,.0f}")
            except Exception as exc:
                log.warning(f"Could not refresh cash after sells: {exc}")

        # ── קניות אחר כך ─────────────────────────────────────────────────────
        MAX_CONCENTRATION = 0.30   # max 30% of portfolio per stock

        buy_actions = [(i, t) for i, t in enumerate(self.tickers)
                       if float(action[i]) > BUY_THRESHOLD]
        total_buy_signal = sum(float(action[i]) for i, _ in buy_actions) or 1.0

        # חישוב שווי תיק כולל (מזומן + פוזיציות)
        net_worth_total = cash + sum(
            positions.get(t, 0.0) * prices.get(t, 0.0) for t in self.tickers
        )

        for i, ticker in buy_actions:
            act   = float(action[i])
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue

            # בדיקת ריכוזיות: שווי נוכחי + קנייה מתוכננת לא יעלו על 30%
            current_value = positions.get(ticker, 0.0) * price
            max_allowed   = MAX_CONCENTRATION * net_worth_total
            if current_value >= max_allowed:
                log.warning(
                    f"CONCENTRATION LIMIT: {ticker} already at "
                    f"{current_value/net_worth_total:.1%} (max {MAX_CONCENTRATION:.0%}). Skipping BUY."
                )
                continue

            # חלוקת מזומן פרופורציונלית לעוצמת הסיגנל
            budget = cash * (act / total_buy_signal) * 0.95  # 5% buffer

            # הגבל את התקציב כך שלא נחרוג מ-30%
            budget = min(budget, max_allowed - current_value)
            shares_to_buy = max(1, int(budget / price))

            log.info(f"Action {act:.3f} -> BUY {shares_to_buy} {ticker} @ ${price:.2f}")
            result = self.broker.buy(ticker, shares_to_buy, price)
            if result.get("status") in ("ERROR", "REJECTED", "REJECTED_BY_USER"):
                log.warning(f"BUY {ticker} failed: {result}")
            else:
                # עדכון מחיר כניסה ממוצע (weighted average)
                prev_held  = positions.get(ticker, 0.0)
                prev_entry = self._entry_prices.get(ticker, price)
                total_shares = prev_held + shares_to_buy
                if total_shares > 0:
                    self._entry_prices[ticker] = (
                        (prev_held * prev_entry + shares_to_buy * price) / total_shares
                    )
                    # Reset trailing high to current price on new buy
                    self._trailing_highs[ticker] = max(
                        self._trailing_highs.get(ticker, price), price
                    )

    # ──────────────────────────────────────────────────────────────────────────
    # התראות Telegram
    # ──────────────────────────────────────────────────────────────────────────

    def _telegram(self, msg: str):
        """שליחת הודעה גנרית לטלגרם."""
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        try:
            url  = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id":    chat_id,
                "text":       msg,
                "parse_mode": "Markdown",
            }).encode()
            urllib.request.urlopen(url, data, timeout=10)
        except Exception as exc:
            log.debug(f"Telegram send failed: {exc}")

    def _notify_startup(self):
        """הודעה בהפעלת הסוכן."""
        try:
            equity = self.broker.get_account().get("equity", self.initial_capital)
        except Exception:
            equity = self.initial_capital
        self._telegram(
            f"🟢 *Agent STARTED*\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Tickers: {len(self.tickers)} stocks\n"
            f"Equity: ${equity:,.0f}\n"
            f"Mode: Paper Trading"
        )
        log.info("Startup notification sent to Telegram.")

    def _notify_shutdown(self):
        """הודעה בסגירת הסוכן."""
        self._telegram(
            f"🔴 *Agent STOPPED*\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Reason: process exited (manual stop or error)\n"
            f"Restart: run `python main.py --mode live_paper --auto-approve`"
        )
        log.info("Shutdown notification sent to Telegram.")

    def _notify_halt(self, net_worth: float, drawdown: float):
        """שולח התראה דחופה לטלגרם כשהמסחר מועצר."""
        pnl = net_worth - self.initial_capital
        self._telegram(
            f"🚨 *TRADING HALTED*\n"
            f"\n"
            f"Drawdown reached {drawdown:.1%} — automatic halt triggered.\n"
            f"\n"
            f"💰 Net Worth: ${net_worth:,.0f}\n"
            f"📉 Loss from peak: ${pnl:,.0f} ({drawdown:.1%})\n"
            f"\n"
            f"No orders will be placed until manually restarted.\n"
            f"Restart: `python main.py --mode live_paper --auto-approve`"
        )
        log.info("HALT alert sent to Telegram.")

    # ──────────────────────────────────────────────────────────────────────────
    # Regime detection helper
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_regime(self, fresh_data: dict[str, "pd.DataFrame"]):
        """
        Downloads SPY data and runs RegimeDetector.
        Returns RegimeSignal or None on failure.

        Handles both yfinance column formats:
        - Simple:      Open, High, Low, Close, Volume
        - Multi-level: (Open, SPY), (Close, SPY), ...  [newer yfinance]
        """
        try:
            import yfinance as yf
            spy_raw = yf.download("SPY", period="300d", progress=False, auto_adjust=True)
            if spy_raw.empty:
                log.warning("SPY data empty — skipping regime detection.")
                return None

            # Flatten multi-level columns: ('Close', 'SPY') → 'close'
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = [col[0].lower() for col in spy_raw.columns]
            else:
                spy_raw.columns = [c.lower() for c in spy_raw.columns]

            spy_df = spy_raw.reset_index()
            # Ensure date column is named 'date'
            date_col = [c for c in spy_df.columns if "date" in c.lower() or c.lower() == "index"]
            if date_col and date_col[0] != "date":
                spy_df = spy_df.rename(columns={date_col[0]: "date"})

            return self._regime_detector.detect(spy_df)
        except Exception as exc:
            log.warning(f"Regime detection failed: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # דוח יומי
    # ──────────────────────────────────────────────────────────────────────────

    def _send_daily_summary(
        self,
        net_worth: float,
        cash: float,
        drawdown: float,
        action: np.ndarray,
        prices: dict[str, float],
    ):
        """שולח סיכום יומי לטלגרם."""
        pnl_pct = (net_worth - self.initial_capital) / self.initial_capital * 100
        pnl_abs = net_worth - self.initial_capital
        sign    = "+" if pnl_abs >= 0 else ""

        rs = self.risk_manager.get_status()
        lines = [
            f"📊 *Daily Trading Report* — {datetime.now().strftime('%Y-%m-%d')}",
            f"",
            f"💰 Net Worth:  ${net_worth:,.0f}",
            f"💵 Cash:       ${cash:,.0f}",
            f"📈 Total P&L:  {sign}{pnl_abs:,.0f} ({sign}{pnl_pct:.1f}%)",
            f"📉 Drawdown:   {drawdown:.1%}",
            f"⚠️  Risk Level: {rs['risk_level']}",
            f"🌐 Regime mult: {rs['regime_multiplier']:.2f}",
            f"",
            f"🎯 *Actions taken:*",
        ]

        for i, ticker in enumerate(self.tickers):
            act   = float(action[i])
            price = prices.get(ticker, 0.0)
            if act > 0.05:
                lines.append(f"  BUY  {ticker} (signal={act:+.2f}, ${price:.2f})")
            elif act < -0.05:
                lines.append(f"  SELL {ticker} (signal={act:+.2f}, ${price:.2f})")

        self._telegram("\n".join(lines))
        log.info("Daily summary sent to Telegram.")
