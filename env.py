"""
CryptoTradingEnv — Gymnasium environment simulating a crypto exchange.

Action space:  Box([-1, 1]) — continuous value where:
  -1.0 = sell 100% of position
   0.0 = hold
  +1.0 = buy up to max_position_pct of portfolio

Observation space: market features + portfolio state (position, PnL, etc.)

Reward: risk-adjusted return (Sortino-like) to discourage excessive drawdown.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from data import FEATURE_COLS


class CryptoTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        trading_fee: float = 0.001,       # 0.1% per trade (Binance taker fee)
        max_position_pct: float = 0.95,   # max 95% of balance in crypto
        window_size: int = 1,             # lookback (1 = current candle only)
        reward_scaling: float = 100.0,
    ) -> None:
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = FEATURE_COLS
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.max_position_pct = max_position_pct
        self.window_size = window_size
        self.reward_scaling = reward_scaling

        # Number of market features + 4 portfolio state features
        n_features = len(self.feature_cols) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Continuous action: fraction of portfolio to be in crypto [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )

        self._reset_state()

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        target_position_pct = float(np.clip(action[0], -1.0, 1.0))

        # Map [-1,1] → [0, max_position_pct]  (no short selling for simplicity)
        target_crypto_pct = ((target_position_pct + 1.0) / 2.0) * self.max_position_pct

        prev_portfolio_value = self._portfolio_value()
        self._execute_trade(target_crypto_pct)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Update crypto value at new price
        current_price = self._current_price()
        self.crypto_value = self.crypto_units * current_price

        new_portfolio_value = self._portfolio_value()

        # Risk-adjusted reward: penalise downside variance
        step_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward = self._compute_reward(step_return)

        # Track for metrics
        self.portfolio_history.append(new_portfolio_value)
        self.trade_history.append({
            "step": self.current_step,
            "action": target_position_pct,
            "price": current_price,
            "portfolio_value": new_portfolio_value,
            "cash": self.cash,
            "crypto_value": self.crypto_value,
        })

        obs = self._get_obs()
        info = {
            "portfolio_value": new_portfolio_value,
            "return": step_return,
            "cash": self.cash,
            "crypto_units": self.crypto_units,
        }
        return obs, reward, done, False, info

    def render(self) -> None:
        pv = self._portfolio_value()
        roi = (pv / self.initial_balance - 1) * 100
        print(
            f"Step {self.current_step:5d} | "
            f"Price: {self._current_price():10.2f} | "
            f"Portfolio: {pv:10.2f} | "
            f"ROI: {roi:+.2f}%"
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.current_step: int = 50  # start after warmup window for indicators
        self.cash: float = self.initial_balance
        self.crypto_units: float = 0.0
        self.crypto_value: float = 0.0
        self.portfolio_history: list[float] = [self.initial_balance]
        self.trade_history: list[dict] = []
        self._return_history: list[float] = []

    def _current_price(self) -> float:
        return float(self.df.loc[self.current_step, "close"])

    def _portfolio_value(self) -> float:
        return self.cash + self.crypto_value

    def _execute_trade(self, target_crypto_pct: float) -> None:
        """Rebalance portfolio to match target_crypto_pct allocation."""
        total_value = self._portfolio_value()
        target_crypto_value = total_value * target_crypto_pct
        current_crypto_value = self.crypto_value
        price = self._current_price()

        diff = target_crypto_value - current_crypto_value

        if diff > 0:  # Buy
            # Can only spend available cash (minus fee buffer)
            max_spend = self.cash / (1 + self.trading_fee)
            spend = min(diff, max_spend)
            fee = spend * self.trading_fee
            units_bought = spend / price
            self.cash -= spend + fee
            self.crypto_units += units_bought

        elif diff < 0:  # Sell
            units_to_sell = min(abs(diff) / price, self.crypto_units)
            proceeds = units_to_sell * price
            fee = proceeds * self.trading_fee
            self.cash += proceeds - fee
            self.crypto_units -= units_to_sell

        self.crypto_value = self.crypto_units * price

    def _compute_reward(self, step_return: float) -> float:
        """
        Sortino-inspired reward:
        - Reward positive returns normally
        - Penalise negative returns more heavily (2x)
        - Scale to keep gradients stable
        """
        self._return_history.append(step_return)

        if step_return >= 0:
            reward = step_return * self.reward_scaling
        else:
            # Downside penalty — extra discouragement of losses
            reward = step_return * self.reward_scaling * 2.0

        # Max drawdown penalty
        if len(self.portfolio_history) > 1:
            peak = max(self.portfolio_history)
            current = self._portfolio_value()
            drawdown = (peak - current) / peak
            if drawdown > 0.15:  # 15% drawdown threshold
                reward -= drawdown * 10.0

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        """Build observation vector: market features + portfolio state."""
        row = self.df.loc[self.current_step]

        market_features = np.array(
            [row[col] for col in self.feature_cols], dtype=np.float32
        )

        # Clip extreme values to prevent exploding gradients
        market_features = np.clip(market_features, -10.0, 10.0)

        total_value = self._portfolio_value()
        portfolio_features = np.array([
            self.crypto_value / total_value if total_value > 0 else 0.0,  # crypto allocation
            self.cash / self.initial_balance,                              # normalised cash
            total_value / self.initial_balance - 1.0,                     # total ROI
            self._max_drawdown(),                                          # current drawdown
        ], dtype=np.float32)

        return np.concatenate([market_features, portfolio_features])

    def _max_drawdown(self) -> float:
        if len(self.portfolio_history) < 2:
            return 0.0
        peak = max(self.portfolio_history)
        current = self._portfolio_value()
        return float((peak - current) / peak) if peak > 0 else 0.0

    def get_metrics(self) -> dict:
        """Compute performance metrics after an episode."""
        values = np.array(self.portfolio_history)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0] - 1) * 100
        sharpe = (
            np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(8760)  # annualised hourly
            if len(returns) > 1 else 0.0
        )

        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = float(drawdown.max()) * 100

        # Win rate
        win_rate = float(np.mean(returns > 0)) * 100 if len(returns) > 0 else 0.0

        return {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "final_value": round(float(values[-1]), 2),
            "n_steps": len(self.trade_history),
        }
