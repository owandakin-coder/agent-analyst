"""
live_trader.py
==============
לולאת המסחר החי (Live / Paper-Live).
מחבר את מודל ה-RL לממשק Alpaca ומריץ מחזורי החלטה יומיים.

⚠️  לצרכי מחקר בלבד. כל פעולה על כסף אמיתי באחריות המשתמש.
"""

from __future__ import annotations

import time
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from control_plane import can_trade, load_control_state
from decision_journal import write_last_decision
from execution_runtime import emit_execution_event, emit_risk_event
from multi_agent import MultiAgentDecisionEngine
from notifications import send_operator_alert
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


def _fail_closed_control_enabled() -> bool:
    return os.getenv("ATZMA_FAIL_CLOSED_CONTROL", "1").strip().lower() not in {"0", "false", "no"}


def _require_fresh_quotes() -> bool:
    return os.getenv("ATZMA_REQUIRE_FRESH_QUOTES", "1").strip().lower() not in {"0", "false", "no"}


def _fresh_quote_max_age_seconds() -> int:
    try:
        return max(5, int(os.getenv("ATZMA_MAX_QUOTE_AGE_SECONDS", "60")))
    except Exception:
        return 60


def _daily_realized_loss_limit() -> float:
    try:
        return max(0.0, float(os.getenv("ATZMA_DAILY_REALIZED_LOSS_LIMIT", "2500")))
    except Exception:
        return 2500.0


def _daily_unrealized_loss_limit() -> float:
    try:
        return max(0.0, float(os.getenv("ATZMA_DAILY_UNREALIZED_LOSS_LIMIT", "4000")))
    except Exception:
        return 4000.0

# ─── קבועים (מ-config.yaml) ─────────────────────────────────────────────────
try:
    from config_loader import CFG as _CFG
    WINDOW_SIZE = _CFG.window_size
    MIN_DATA_DAYS = max(_CFG.live_min_data_days, WINDOW_SIZE * 7)
    BUY_THRESHOLD = _CFG.live_buy_threshold
    SELL_THRESHOLD = _CFG.live_sell_threshold
    MAX_CONCENTRATION = _CFG.live_max_concentration
    MAX_GROSS_EXPOSURE = _CFG.live_max_gross_exposure
    MAX_POSITIONS = _CFG.live_max_positions
    MAX_SINGLE_ORDER_NOTIONAL_PCT = _CFG.live_max_single_order_notional_pct
    CASH_BUFFER_PCT = _CFG.live_cash_buffer_pct
    STOP_LOSS_PCT = _CFG.live_stop_loss_pct
    NO_MARGIN = _CFG.live_no_margin
    AUTO_DELEVERAGE = _CFG.live_auto_deleverage
    PRICE_MIN = _CFG.live_price_min
    PRICE_MAX = _CFG.live_price_max
    DEFAULT_POLL_SECONDS = _CFG.live_poll_seconds
    MARKET_OPEN_BUFFER_MINUTES = _CFG.live_market_open_buffer_minutes
    MIN_TRADE_VALUE = _CFG.min_trade_value
except Exception:
    WINDOW_SIZE = 30
    MIN_DATA_DAYS = 200
    BUY_THRESHOLD = 0.05
    SELL_THRESHOLD = -0.05
    MAX_CONCENTRATION = 0.30
    MAX_GROSS_EXPOSURE = 0.65
    MAX_POSITIONS = 8
    MAX_SINGLE_ORDER_NOTIONAL_PCT = 0.04
    CASH_BUFFER_PCT = 0.05
    STOP_LOSS_PCT = 0.08
    NO_MARGIN = True
    AUTO_DELEVERAGE = True
    PRICE_MIN = 1.0
    PRICE_MAX = 10_000.0
    DEFAULT_POLL_SECONDS = 60
    MARKET_OPEN_BUFFER_MINUTES = 5
    MIN_TRADE_VALUE = 500.0

FEATURE_COLS   = [    # פיצ'רים שהמודל ראה באימון – חייב להתאים ל-DataManager
    "returns", "log_returns",
    "price_to_ma20", "price_to_ma50", "ma_cross",
    "rsi", "macd_hist", "boll_pct", "boll_width",
    "atr_pct", "volume_ratio", "volatility_20",
]


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
        self._multi_agent = MultiAgentDecisionEngine(
            buy_threshold=BUY_THRESHOLD,
            sell_threshold=SELL_THRESHOLD,
            stop_loss_pct=STOP_LOSS_PCT,
        )
        self._alt_fetcher     = AlternativeDataFetcher()
        self.last_cycle_result: dict | None = None

        # מעקב שווי תיק
        self._peak_net_worth = initial_capital

        # Per-stock trailing stop-loss tracking
        self._entry_prices:   dict[str, float] = {}
        self._trailing_highs: dict[str, float] = {}   # max price since entry
        self.stop_loss_pct: float = STOP_LOSS_PCT

    def _portfolio_market_value(self, prices: dict[str, float], positions: dict[str, float]) -> float:
        return float(sum(max(0.0, positions.get(t, 0.0)) * max(0.0, prices.get(t, 0.0)) for t in self.tickers))

    def _active_positions_count(self, positions: dict[str, float]) -> int:
        return sum(1 for qty in positions.values() if qty and qty > 0)

    def _auto_deleverage_if_needed(
        self,
        prices: dict[str, float],
        positions: dict[str, float],
        cash: float,
        net_worth_total: float,
        order_events: list[dict],
    ) -> tuple[float, dict[str, float]]:
        if not AUTO_DELEVERAGE or net_worth_total <= 0:
            return cash, positions

        reserve_cash = net_worth_total * CASH_BUFFER_PCT
        market_value = self._portfolio_market_value(prices, positions)
        gross_exposure = market_value / net_worth_total if net_worth_total > 0 else 0.0
        needs_deleverage = (NO_MARGIN and cash < reserve_cash) or gross_exposure > MAX_GROSS_EXPOSURE
        if not needs_deleverage:
            return cash, positions

        details = self.broker.get_position_details() if hasattr(self.broker, "get_position_details") else {}
        ranking: list[tuple[float, str, float, float]] = []
        for ticker, held in positions.items():
            price = prices.get(ticker, 0.0)
            if held <= 0 or price <= 0:
                continue
            unrealized = float((details.get(ticker) or {}).get("unrealized_pl", 0.0))
            ranking.append((unrealized, ticker, held, price))

        ranking.sort(key=lambda item: item[0])
        for _, ticker, held, price in ranking:
            market_value = self._portfolio_market_value(prices, positions)
            gross_exposure = market_value / net_worth_total if net_worth_total > 0 else 0.0
            if cash >= reserve_cash and gross_exposure <= MAX_GROSS_EXPOSURE:
                break
            shares_to_sell = max(1, int(np.ceil(held * 0.25)))
            log.warning(
                "AUTO-DELEVERAGING %s: selling %s shares to restore cash/exposure limits.",
                ticker,
                shares_to_sell,
            )
            result = self.broker.sell(ticker, shares_to_sell, price)
            order_events.append({"event_type": "auto_deleverage", **result})
            if result.get("status") in ("ERROR", "REJECTED", "REJECTED_BY_USER"):
                continue
            positions[ticker] = max(0.0, positions.get(ticker, 0.0) - shares_to_sell)
            cash += shares_to_sell * price

        return cash, positions

    # ──────────────────────────────────────────────────────────────────────────
    # לולאה ראשית
    # ──────────────────────────────────────────────────────────────────────────

    def run_loop(self, poll_seconds: int = DEFAULT_POLL_SECONDS):
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
                        sleep_secs = self._seconds_until_next_open(
                            buffer_minutes=MARKET_OPEN_BUFFER_MINUTES
                        )
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
        emit_execution_event("execution_started", {"tickers": self.tickers, "initial_capital": self.initial_capital})
        try:
            state = load_control_state()
            allowed, reason = can_trade(state)
            emit_execution_event("control_validated", {
                "allowed": allowed,
                "reason": reason,
                "mode": state.get("mode"),
                "trading_enabled": state.get("trading_enabled"),
                "emergency_stop": state.get("emergency_stop"),
                "command_version": state.get("command_version"),
            })
        except Exception as exc:
            log.warning(f"Control plane unavailable: {exc}")
            allowed, reason = (False, "control_plane_unavailable") if _fail_closed_control_enabled() else (True, None)

        if not allowed:
            status = "emergency stop" if reason == "emergency_stop" else "paused"
            if reason == "control_plane_unavailable":
                status = "control plane unavailable"
            log.warning(f"Trading skipped by control plane: {status}")
            emit_execution_event("execution_blocked", {"reason": reason or status})
            self._telegram(
                f"⏸ *Agent skipped* — control plane is {status}.\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            return

        log.info("Fetching latest market data ...")
        try:
            fresh_data = self._fetch_fresh_data()
        except Exception as exc:
            log.error(f"Failed to fetch data: {exc}")
            return

        # ── מחירים נוכחיים ──────────────────────────────────────────────────
        current_prices = self.broker.get_latest_prices(self.tickers)
        quote_info = self.broker.get_latest_quotes_info(self.tickers) if hasattr(self.broker, "get_latest_quotes_info") else {}
        if not current_prices:
            current_prices = {
                t: float(fresh_data[t]["close"].iloc[-1])
                for t in self.tickers if t in fresh_data
            }
        current_prices = self.validate_prices(current_prices, fresh_data)
        if not self._quotes_are_fresh(quote_info, current_prices):
            emit_execution_event("execution_aborted", {"reason": "stale_quotes", "quotes": quote_info})
            log.error("Aborting execution due to stale or missing live quotes.")
            return
        log.info(f"Prices: {current_prices}")

        # ── שווי תיק עדכני ──────────────────────────────────────────────────
        snapshot = self._reconcile_snapshot()
        if snapshot is None:
            return

        positions = snapshot["positions"]
        cash      = snapshot["cash"]
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
        emit_risk_event("risk_updated", {"risk_level": risk_level.value, "drawdown": drawdown, "net_worth": net_worth})
        if self._enforce_daily_loss_limits(snapshot, net_worth):
            emit_risk_event("daily_loss_breached", {
                "net_worth": net_worth,
                "drawdown": drawdown,
            })
            log.warning("Daily loss circuit breaker triggered. Skipping this cycle.")
            return
        if self.risk_manager.is_halted:
            log.warning("TRADING HALTED by RiskManager. Skipping this cycle.")
            if not prev_halted:
                self._notify_halt(net_worth, drawdown)
            emit_risk_event("risk_halted", {"drawdown": drawdown, "net_worth": net_worth})
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
            emit_execution_event("regime_detected", {
                "regime": regime_signal.regime.value,
                "confidence": regime_signal.confidence,
                "description": regime_signal.description,
            })

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
            raw_action = self.model.predict(obs[np.newaxis], deterministic=True)
            log.info(f"Ensemble vote summary: {self.model.vote_summary(obs[np.newaxis])}")
        else:
            obs_norm = self.vec_norm.normalize_obs(obs[np.newaxis])  # (1, W, F)
            raw_action, _ = self.model.predict(obs_norm, deterministic=True)
        raw_action = np.array(raw_action).flatten()

        decision_bundle = self._multi_agent.evaluate(
            tickers=self.tickers,
            fresh_data=fresh_data,
            raw_action=raw_action,
            regime_signal=regime_signal,
            positions=positions,
            entry_prices=self._entry_prices,
            trailing_highs=self._trailing_highs,
            current_drawdown=drawdown,
        )
        action = decision_bundle.final_action_vector(self.tickers)
        log.info("Multi-agent summary: %s", decision_bundle.top_summary())
        emit_execution_event("decision_ready", {
            "regime": decision_bundle.regime,
            "strategy_mode": decision_bundle.strategy_mode,
            "summary": decision_bundle.top_summary(),
            "decisions": [decision.as_dict() for decision in decision_bundle.decisions],
        })

        # התאמת גודל פוזיציה לפי סיכון (with regime multiplier + kelly + correlation)
        price_history = {t: fresh_data[t]["close"] for t in self.tickers if t in fresh_data}
        action = self.risk_manager.scale_action(
            action,
            tickers=self.tickers,
            price_history=price_history,
        )
        log.info(f"Scaled action: {action.round(3)}")

        # ── המרה לפקודות ────────────────────────────────────────────────────
        broker_orders = self._execute_actions(action, current_prices, cash, positions)
        self.last_cycle_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": os.getenv("ATZMA_MODEL_VERSION", "unknown"),
            "strategy_version": os.getenv("ATZMA_STRATEGY_VERSION", "default"),
            "regime": decision_bundle.regime,
            "strategy_mode": decision_bundle.strategy_mode,
            "net_worth": float(net_worth),
            "cash": float(cash),
            "drawdown": float(drawdown),
            "risk_level": self.risk_manager.risk_level.value,
            "market_data_source": "alpaca_latest_quote",
            "broker_snapshot_hash": self._snapshot_hash(snapshot),
            "market_snapshot_hash": self._snapshot_hash({"prices": current_prices, "quotes": quote_info}),
            "feature_snapshot_hash": self._snapshot_hash({"tickers": self.tickers, "obs_shape": list(obs.shape)}),
            "broker_snapshot": snapshot,
            "market_snapshot": self._build_market_snapshot(fresh_data, current_prices, quote_info),
            "feature_snapshot": {
                "observation": obs.tolist(),
                "tickers": list(self.tickers),
                "window_size": WINDOW_SIZE,
                "feature_columns": list(FEATURE_COLS),
            },
            "raw_action": [float(x) for x in raw_action.tolist()],
            "scaled_action": [float(x) for x in action.tolist()],
            "summary": decision_bundle.top_summary(),
            "decisions": [decision.as_dict() for decision in decision_bundle.decisions],
            "broker_orders": broker_orders,
        }
        write_last_decision(self.last_cycle_result)

        # ── דוח יומי לטלגרם ─────────────────────────────────────────────────
        self._send_daily_summary(net_worth, cash, drawdown, action, current_prices, decision_bundle)
        emit_execution_event("execution_completed", {"summary": self.last_cycle_result.get("summary"), "broker_orders": broker_orders})
        return self.last_cycle_result

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
        try:
            data = self.data_manager.load_all(force_download=True)
        except Exception as exc:
            log.warning("Primary market data fetch failed: %s. Trying broker bars fallback.", exc)
            data = self._fetch_broker_feature_data(start=start, end=end, tickers=self.tickers)
            if not data:
                raise
            return data

        missing = [
            ticker for ticker in self.tickers
            if ticker not in data or data[ticker] is None or len(data[ticker]) < WINDOW_SIZE
        ]
        if missing:
            log.warning("Market data missing/incomplete for %s. Backfilling from broker bars.", missing)
            fallback = self._fetch_broker_feature_data(start=start, end=end, tickers=missing)
            data.update(fallback)
        return data

    def _fetch_broker_feature_data(self, *, start: str, end: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
        get_bars = getattr(self.broker, "get_historical_bars", None)
        compute_features = getattr(self.data_manager, "_compute_features", None)
        if not callable(get_bars) or not callable(compute_features):
            return {}

        frames = get_bars(tickers, start=start, end=end)
        featured: dict[str, pd.DataFrame] = {}
        for ticker, raw in frames.items():
            if raw is None or raw.empty:
                continue
            try:
                featured[ticker] = compute_features(raw.copy(), ticker)
            except Exception as exc:
                log.warning("Broker bar feature build failed for %s: %s", ticker, exc)
        return featured

    def validate_prices(
        self,
        prices: dict[str, float],
        fresh_data: dict[str, pd.DataFrame],
        min_price: float = PRICE_MIN,
        max_price: float = PRICE_MAX,
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

    def _quotes_are_fresh(self, quote_info: dict[str, dict], prices: dict[str, float]) -> bool:
        if not _require_fresh_quotes():
            return True
        if not prices:
            return False
        now = datetime.now(timezone.utc)
        max_age = _fresh_quote_max_age_seconds()
        for ticker in self.tickers:
            if ticker not in prices:
                return False
            info = quote_info.get(ticker) or {}
            candidates = [info.get("observed_at"), info.get("timestamp")]
            fresh = False
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    quote_time = datetime.fromisoformat(str(candidate).replace("Z", "+00:00"))
                except Exception:
                    continue
                if quote_time.tzinfo is None:
                    quote_time = quote_time.replace(tzinfo=timezone.utc)
                age = (now - quote_time.astimezone(timezone.utc)).total_seconds()
                if 0 <= age <= max_age:
                    fresh = True
                    break
            if not fresh:
                return False
        return True

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
    ) -> list[dict]:
        """
        ממיר וקטור פעולות [-1,1] לפקודות קנייה/מכירה בפועל.

        אסטרטגיה:
        - action[i] < -0.05  → מכור פרופורציה מהאחזקה
        - action[i] >  0.05  → קנה פרופורציה מהמזומן הפנוי
        - ריכוזיות > 30%     → מדלג על קנייה
        - אחרת → החזק
        """
        order_events: list[dict] = []
        buy_threshold = BUY_THRESHOLD
        sell_threshold = SELL_THRESHOLD

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
                order_events.append({"event_type": "trailing_stop_exit", **result})
                if result.get("status") not in ("ERROR", "REJECTED", "REJECTED_BY_USER"):
                    # Record Kelly outcome
                    entry = self._entry_prices.get(ticker, 0.0)
                    if entry > 0:
                        pnl_pct = (price - entry) / entry
                        self.risk_manager.record_trade_outcome(ticker, pnl_pct)
                    self._entry_prices.pop(ticker, None)
                    self._trailing_highs.pop(ticker, None)
                    positions[ticker] = 0.0
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

            if act < sell_threshold and held > 0 and price > 0:
                shares_to_sell = max(1, int(held * abs(act)))
                log.info(f"Action {act:.3f} -> SELL {shares_to_sell} {ticker} @ ${price:.2f}")
                result = self.broker.sell(ticker, shares_to_sell, price)
                order_events.append({"event_type": "sell_signal", **result})
                if result.get("status") in ("ERROR", "REJECTED", "REJECTED_BY_USER"):
                    log.warning(f"SELL {ticker} failed: {result}")
                else:
                    sells_executed += 1
                    positions[ticker] = max(0.0, held - shares_to_sell)
                    # Record outcome for Kelly Criterion — P&L from entry to exit
                    entry = self._entry_prices.get(ticker, 0.0)
                    if entry > 0:
                        pnl_pct = (price - entry) / entry
                        self.risk_manager.record_trade_outcome(ticker, pnl_pct)
                        log.debug(f"Kelly record: {ticker} pnl={pnl_pct:+.2%}")

        # ── רענון מזומן אחרי מכירות ─────────────────────────────────────────
        if sells_executed > 0:
            try:
                snapshot = self._reconcile_snapshot()
                if snapshot is not None:
                    cash = snapshot["cash"]
                    positions.update(snapshot["positions"])
                    log.info(f"Cash after sells: ${cash:,.0f}")
            except Exception as exc:
                log.warning(f"Could not refresh cash after sells: {exc}")

        net_worth_total = cash + self._portfolio_market_value(prices, positions)
        cash, positions = self._auto_deleverage_if_needed(prices, positions, cash, net_worth_total, order_events)

        # ── קניות אחר כך ─────────────────────────────────────────────────────
        max_concentration = MAX_CONCENTRATION

        buy_actions = [(i, t) for i, t in enumerate(self.tickers)
                       if float(action[i]) > buy_threshold]
        total_buy_signal = sum(float(action[i]) for i, _ in buy_actions) or 1.0

        # חישוב שווי תיק כולל (מזומן + פוזיציות)
        net_worth_total = cash + self._portfolio_market_value(prices, positions)
        reserve_cash = max(0.0, net_worth_total * CASH_BUFFER_PCT)
        available_cash = max(0.0, cash - reserve_cash) if NO_MARGIN else max(0.0, cash)
        remaining_gross_room = max(
            0.0,
            net_worth_total * MAX_GROSS_EXPOSURE - self._portfolio_market_value(prices, positions),
        )
        if NO_MARGIN and available_cash < MIN_TRADE_VALUE:
            log.info("Skipping buys: available cash after reserve is below minimum trade value.")
            return order_events

        for i, ticker in buy_actions:
            act   = float(action[i])
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue
            if positions.get(ticker, 0.0) <= 0 and self._active_positions_count(positions) >= MAX_POSITIONS:
                log.info("Skipping BUY %s: max active positions limit reached (%s).", ticker, MAX_POSITIONS)
                continue

            # בדיקת ריכוזיות: שווי נוכחי + קנייה מתוכננת לא יעלו על 30%
            current_value = positions.get(ticker, 0.0) * price
            max_allowed   = max_concentration * net_worth_total
            if current_value >= max_allowed:
                log.warning(
                    f"CONCENTRATION LIMIT: {ticker} already at "
                    f"{current_value/net_worth_total:.1%} (max {max_concentration:.0%}). Skipping BUY."
                )
                continue

            # חלוקת מזומן פרופורציונלית לעוצמת הסיגנל
            budget = available_cash * (act / total_buy_signal)

            # הגבל את התקציב כך שלא נחרוג מ-30%
            budget = min(
                budget,
                max_allowed - current_value,
                net_worth_total * MAX_SINGLE_ORDER_NOTIONAL_PCT,
                remaining_gross_room,
            )
            if budget < MIN_TRADE_VALUE:
                log.info(
                    f"Skipping BUY {ticker}: budget ${budget:.2f} below minimum ${MIN_TRADE_VALUE:.2f}"
                )
                continue
            shares_to_buy = max(1, int(budget / price))

            log.info(f"Action {act:.3f} -> BUY {shares_to_buy} {ticker} @ ${price:.2f}")
            result = self.broker.buy(ticker, shares_to_buy, price)
            order_events.append({"event_type": "buy_signal", **result})
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
                    positions[ticker] = total_shares
                    cash = cash - shares_to_buy * price
                    available_cash = max(0.0, cash - reserve_cash) if NO_MARGIN else max(0.0, cash)
                    remaining_gross_room = max(
                        0.0,
                        net_worth_total * MAX_GROSS_EXPOSURE - self._portfolio_market_value(prices, positions),
                    )

        return order_events

    def _reconcile_snapshot(self) -> dict | None:
        try:
            snapshot = self.broker.reconcile_account_state()
        except Exception as exc:
            log.error(f"Account reconciliation failed: {exc}")
            return None

        if not isinstance(snapshot, dict):
            log.error("Account reconciliation failed: invalid snapshot type")
            return None
        if "cash" not in snapshot or "positions" not in snapshot:
            log.error("Account reconciliation failed: missing cash or positions")
            return None
        return snapshot

    def _enforce_daily_loss_limits(self, snapshot: dict, net_worth: float) -> bool:
        position_details = snapshot.get("position_details") or {}
        unrealized_pnl = 0.0
        if isinstance(position_details, dict):
            for details in position_details.values():
                if isinstance(details, dict):
                    unrealized_pnl += float(details.get("unrealized_pl", 0.0) or 0.0)
        baseline = float(snapshot.get("equity", net_worth) or net_worth) - unrealized_pnl
        realized_pnl = float(net_worth) - baseline - unrealized_pnl
        realized_loss = max(0.0, -realized_pnl)
        unrealized_loss = max(0.0, -unrealized_pnl)
        emit_risk_event("daily_loss_state", {
            "trading_day": datetime.now(timezone.utc).date().isoformat(),
            "baseline_equity": baseline,
            "current_equity": net_worth,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "realized_loss_limit": _daily_realized_loss_limit(),
            "unrealized_loss_limit": _daily_unrealized_loss_limit(),
        })
        return realized_loss >= _daily_realized_loss_limit() or unrealized_loss >= _daily_unrealized_loss_limit()

    def _snapshot_hash(self, payload: dict) -> str:
        import hashlib
        import json

        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_market_snapshot(
        self,
        fresh_data: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        quote_info: dict[str, dict],
    ) -> dict:
        snapshot: dict[str, dict] = {
            "prices": {ticker: float(price) for ticker, price in current_prices.items()},
            "quotes": quote_info,
            "tickers": list(self.tickers),
        }
        bars: dict[str, dict] = {}
        for ticker, df in fresh_data.items():
            if df is None or df.empty:
                continue
            tail = df.tail(WINDOW_SIZE).copy()
            bars[ticker] = {
                "columns": [str(col) for col in tail.columns],
                "index": [str(idx) for idx in tail.index.tolist()],
                "rows": tail.astype(float, errors="ignore").replace({np.nan: None}).values.tolist(),
            }
        snapshot["bars"] = bars
        return snapshot

    # ──────────────────────────────────────────────────────────────────────────
    # התראות Telegram
    # ──────────────────────────────────────────────────────────────────────────

    def _telegram(self, msg: str):
        """שליחת הודעה גנרית לטלגרם."""
        send_operator_alert(msg, markdown=True)

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
        Uses SPY from the current market snapshot when available and only
        falls back to a direct download if needed.
        Returns RegimeSignal or None on failure.
        """
        try:
            spy_df = fresh_data.get("SPY")
            if spy_df is not None and not spy_df.empty and "close" in spy_df.columns:
                return self._regime_detector.detect(spy_df.reset_index(drop=True))

            import yfinance as yf
            spy_raw = yf.download("SPY", period="300d", progress=False, auto_adjust=True)
            if spy_raw.empty:
                log.warning("SPY data empty - skipping regime detection.")
                return None
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = [col[0].lower() for col in spy_raw.columns]
            else:
                spy_raw.columns = [c.lower() for c in spy_raw.columns]
            return self._regime_detector.detect(spy_raw.reset_index(drop=True))
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
        decision_bundle=None,
    ):
        """שולח סיכום יומי לטלגרם."""
        pnl_pct = (net_worth - self.initial_capital) / self.initial_capital * 100
        pnl_abs = net_worth - self.initial_capital
        sign = "+" if pnl_abs >= 0 else ""

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
