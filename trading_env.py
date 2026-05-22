"""
trading_env.py
==============
סביבת מסחר מבוססת OpenAI Gym לסוכן RL.
⚠️ לצרכי מחקר בלבד. אין שימוש בכסף אמיתי.
"""

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

log = logging.getLogger("TradingEnv")

# ─── קבועים ───────────────────────────────────────────────────────────────────
COMMISSION_RATE   = 0.001   # 0.1% עמלת עסקה
SLIPPAGE_RATE     = 0.0005  # 0.05% החלקת מחיר
INITIAL_CASH      = 100_000.0  # הון התחלתי בדולרים
WINDOW_SIZE       = 30      # חלון תצפית (ימים)
SHARPE_WINDOW     = 21      # חלון לחישוב Sharpe בתגמול
MAX_DRAWDOWN_STOP = 0.15    # עצירה ב-15% drawdown


class TradingEnvironment(gym.Env):
    """
    סביבת מסחר רב-מניות.

    מרחב פעולות:
        Box(-1, 1, shape=(num_stocks,)) – ערכים שליליים=מכירה, חיוביים=קנייה.
        0 = החזק.

    מרחב תצפיות:
        (window_size, num_features_per_stock * num_stocks + portfolio_features)
        portfolio_features: [cash_ratio, unrealized_pnl_ratio, drawdown]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        window_size: int = WINDOW_SIZE,
        initial_cash: float = INITIAL_CASH,
        commission: float = COMMISSION_RATE,
        slippage: float = SLIPPAGE_RATE,
        max_drawdown_stop: float = MAX_DRAWDOWN_STOP,
        features_per_stock: Optional[list[str]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.tickers = list(data.keys())
        self.num_stocks = len(self.tickers)
        self.data = data
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.max_drawdown_stop = max_drawdown_stop
        self.render_mode = render_mode

        # בחירת פיצ'רים רלוונטיים לתצפית
        if features_per_stock is None:
            self._feature_cols = self._default_features()
        else:
            self._feature_cols = features_per_stock

        self.num_features = len(self._feature_cols)

        # אינדקסים משותפים לכל המניות
        self._aligned_index = self._get_common_index()
        self.total_steps = len(self._aligned_index) - window_size - 1

        # ── מרחב פעולות ──────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_stocks,),
            dtype=np.float32,
        )

        # ── מרחב תצפיות ──────────────────────────────────────────────────
        # portfolio state: cash_ratio, unrealized_pnl, drawdown (3 ערכים)
        obs_width = self.num_stocks * self.num_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, obs_width),
            dtype=np.float32,
        )

        # ── Pre-compute numpy arrays for fast step() access ──────────────
        # pandas .loc per-step is ~100x slower than direct numpy indexing.
        # Pre-computing once here makes step() pure numpy → major speedup.
        self._price_arr = np.array(
            [data[t]["close"].reindex(self._aligned_index).values for t in self.tickers],
            dtype=np.float64,
        )  # shape: (num_stocks, n_timestamps)

        self._feature_arr = np.array(
            [data[t][self._feature_cols].reindex(self._aligned_index).values
             for t in self.tickers],
            dtype=np.float32,
        )  # shape: (num_stocks, n_timestamps, num_features)

        # Replace any NaN from reindex with 0
        np.nan_to_num(self._price_arr,   nan=0.0, copy=False)
        np.nan_to_num(self._feature_arr, nan=0.0, copy=False)

        # ── מצב פנימי (מאוחר יותר ב-reset) ─────────────────────────────
        self.current_step: int = 0
        self.cash: float = initial_cash
        self.holdings: np.ndarray = np.zeros(self.num_stocks)
        self.net_worth: float = initial_cash
        self.peak_net_worth: float = initial_cash
        self.daily_returns: list[float] = []
        self._done: bool = False
        self._info: dict = {}

        # היסטוריה לאקוויטי קרב
        self.equity_curve: list[float] = []
        self.action_history: list[np.ndarray] = []

    # ──────────────────────────────────────────────────────────────────────────
    # gym API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.num_stocks)
        self.net_worth = self.initial_cash
        self.peak_net_worth = self.initial_cash
        self.daily_returns = []
        self._done = False
        self.equity_curve = [self.initial_cash]
        self.action_history = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        if self._done:
            raise RuntimeError("קרא reset() לפני step() חדש.")

        # שמירת שווי לפני ביצוע פעולה
        prev_net_worth = self.net_worth
        prices = self._get_current_prices()

        # ── ביצוע פעולות ──────────────────────────────────────────────────
        action = np.clip(action, -1.0, 1.0)
        self._execute_actions(action, prices)
        self.action_history.append(action.copy())

        # ── קדם לצעד הבא ─────────────────────────────────────────────────
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self._done = True

        # ── חשב שווי נוכחי ────────────────────────────────────────────────
        new_prices = self._get_current_prices()
        self.net_worth = self._compute_net_worth(new_prices)
        self.equity_curve.append(self.net_worth)
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)

        # תשואה יומית
        daily_ret = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-9)
        self.daily_returns.append(daily_ret)

        # ── בדיקת עצירת drawdown ─────────────────────────────────────────
        drawdown = (self.peak_net_worth - self.net_worth) / (self.peak_net_worth + 1e-9)
        if drawdown >= self.max_drawdown_stop:
            self._done = True
            log.debug(f"[ENV] Episode stopped: drawdown reached {drawdown:.1%}")

        # ── תגמול ─────────────────────────────────────────────────────────
        reward = self._compute_reward(daily_ret, drawdown)

        obs  = self._get_observation()
        info = self._get_info(drawdown=drawdown, daily_ret=daily_ret)

        return obs, reward, self._done, False, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"Step={self.current_step:4d} | "
                f"NetWorth=${self.net_worth:,.0f} | "
                f"Cash=${self.cash:,.0f} | "
                f"Holdings={self.holdings}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # פעולות מסחר
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_actions(self, action: np.ndarray, prices: np.ndarray):
        """
        ממיר וקטור פעולות לקניות/מכירות בפועל.
        action[i] > 0  → קנה (פרופורציה מהמזומן הפנוי)
        action[i] < 0  → מכור (פרופורציה מהאחזקה הנוכחית)
        action[i] = 0  → החזק
        """
        # קודם מכירות (לשחרר מזומן)
        for i, (act, price) in enumerate(zip(action, prices)):
            if act < 0 and self.holdings[i] > 0:
                shares_to_sell = self.holdings[i] * abs(act)
                # החלקת מחיר: מחיר מכירה מעט נמוך יותר
                exec_price = price * (1 - self.slippage)
                proceeds   = shares_to_sell * exec_price
                commission  = proceeds * self.commission
                self.cash     += proceeds - commission
                self.holdings[i] -= shares_to_sell

        # אחר כך קניות
        for i, (act, price) in enumerate(zip(action, prices)):
            if act > 0 and self.cash > 0:
                # משקיע פרופורציה מהמזומן הזמין
                budget = self.cash * act / max(np.sum(action[action > 0]), 1.0)
                budget = min(budget, self.cash)  # לא יותר מהמזומן

                # החלקת מחיר: מחיר קנייה מעט גבוה יותר
                exec_price  = price * (1 + self.slippage)
                commission_per_unit = exec_price * self.commission
                total_per_unit = exec_price + commission_per_unit

                if total_per_unit > 0:
                    shares_to_buy = budget / total_per_unit
                    cost = shares_to_buy * exec_price
                    commission = shares_to_buy * commission_per_unit

                    if cost + commission <= self.cash:
                        self.holdings[i] += shares_to_buy
                        self.cash -= cost + commission

    def _compute_net_worth(self, prices: np.ndarray) -> float:
        return self.cash + float(np.sum(self.holdings * prices))

    # ──────────────────────────────────────────────────────────────────────────
    # תגמול
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, daily_ret: float, drawdown: float) -> float:
        """
        תגמול משולב:
        1. תשואה יומית
        2. Sharpe Ratio על חלון של 21 ימים (תגמל עקביות)
        3. קנס על drawdown
        """
        # רכיב 1: תשואה יומית (ממוספרת)
        ret_reward = daily_ret * 100

        # רכיב 2: Sharpe על חלון גלגל
        sharpe_reward = 0.0
        if len(self.daily_returns) >= SHARPE_WINDOW:
            window_rets = np.array(self.daily_returns[-SHARPE_WINDOW:])
            mean_ret = window_rets.mean()
            std_ret  = window_rets.std() + 1e-9
            sharpe_reward = (mean_ret / std_ret) * np.sqrt(252) * 0.1

        # רכיב 3: קנס drawdown
        drawdown_penalty = -drawdown * 5.0 if drawdown > 0.05 else 0.0

        return float(ret_reward + sharpe_reward + drawdown_penalty)

    # ──────────────────────────────────────────────────────────────────────────
    # תצפיות
    # ──────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """בונה מטריצת תצפית (window_size, features) — pure numpy, no pandas."""
        start = self.current_step
        end   = self.current_step + self.window_size

        # _feature_arr: (num_stocks, n_timestamps, num_features)
        # slice window: (num_stocks, window_size, num_features)
        window = self._feature_arr[:, start:end, :]  # (S, W, F)

        frames = []
        for i in range(self.num_stocks):
            s = window[i]                             # (W, F)
            mean = s.mean(axis=0)
            std  = s.std(axis=0) + 1e-9
            frames.append((s - mean) / std)           # z-score per feature

        # פיצ'רי תיק (3 ערכים, קבועים לאורך החלון)
        prices = self._get_current_prices()
        total  = self._compute_net_worth(prices)
        cash_ratio       = self.cash / (total + 1e-9)
        unrealized_pnl   = (total - self.initial_cash) / self.initial_cash
        drawdown         = (self.peak_net_worth - total) / (self.peak_net_worth + 1e-9)

        portfolio_row   = np.array([cash_ratio, unrealized_pnl, drawdown], dtype=np.float32)
        portfolio_block = np.tile(portfolio_row, (self.window_size, 1))

        obs = np.concatenate(frames + [portfolio_block], axis=1).astype(np.float32)
        np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return obs

    def _get_current_prices(self) -> np.ndarray:
        """מחיר סגירה נוכחי לכל מניה — direct numpy index, no pandas."""
        step_idx = self.current_step + self.window_size
        return self._price_arr[:, step_idx]   # (num_stocks,)

    def _get_info(self, drawdown: float = 0.0, daily_ret: float = 0.0) -> dict:
        return {
            "net_worth":   self.net_worth,
            "cash":        self.cash,
            "holdings":    self.holdings.copy(),
            "drawdown":    drawdown,
            "daily_ret":   daily_ret,
            "step":        self.current_step,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # עזר
    # ──────────────────────────────────────────────────────────────────────────

    def _get_common_index(self) -> pd.Index:
        """מחזיר אינדקס משותף לכל המניות."""
        idx = None
        for df in self.data.values():
            idx = df.index if idx is None else idx.intersection(df.index)
        return idx.sort_values()

    @staticmethod
    def _default_features() -> list[str]:
        """פיצ'רים ברירת-מחדל לתצפית."""
        return [
            "returns", "log_returns",
            "price_to_ma20", "price_to_ma50", "ma_cross",
            "rsi", "macd_hist", "boll_pct", "boll_width",
            "atr_pct", "volume_ratio", "volatility_20",
        ]
