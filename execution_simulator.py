"""
execution_simulator.py
======================
סימולטור ביצוע (Paper Trading): מריץ את המודל על נתוני טסט,
מחשב מדדי ביצוע ומייצר גרפים.
⚠️ לצרכי מחקר בלבד. אין שימוש בכסף אמיתי.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ← backend ללא GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import TradingEnvironment
from risk_manager import RiskManager

RESULTS_DIR = "results"


class ExecutionSimulator:
    """
    מריץ את המודל המאומן על נתוני טסט ומחשב מדדי ביצוע.
    """

    def __init__(
        self,
        model: PPO,
        test_data: dict[str, pd.DataFrame],
        vec_norm: VecNormalize,
        initial_capital: float = 100_000.0,
    ):
        self.model          = model
        self.test_data      = test_data
        self.vec_norm       = vec_norm
        self.initial_capital = initial_capital
        self.tickers        = list(test_data.keys())

        os.makedirs(RESULTS_DIR, exist_ok=True)

        # תוצאות
        self.equity_curve: list[float]      = []
        self.actions_history: list[np.ndarray] = []
        self.dates: list[pd.Timestamp]      = []
        self.metrics: dict                  = {}

    # ──────────────────────────────────────────────────────────────────────────
    # API ציבורי
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """מריץ סימולציה מלאה ומחשב מדדים."""
        print("\n[Simulator] Running Paper Trading on test data ...")

        # בניית סביבה – inner env לגישה ל-net_worth ול-window_size
        inner_env = TradingEnvironment(self.test_data)
        common_idx = inner_env._get_common_index()
        window_size = inner_env.window_size

        env_vec = DummyVecEnv([lambda: TradingEnvironment(self.test_data)])
        env_norm = VecNormalize(env_vec, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)
        env_norm.obs_rms = self.vec_norm.obs_rms
        env_norm.ret_rms = self.vec_norm.ret_rms

        # מנהל סיכונים
        risk_mgr = RiskManager(self.initial_capital)

        obs   = env_norm.reset()
        done  = False
        step  = 0
        current_net_worth = self.initial_capital

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action    = action[0]  # DummyVecEnv עוטף ב-batch=1

            # ניהול סיכונים: התאמת פעולה
            risk_mgr.update(current_net_worth)
            action = risk_mgr.scale_action(action)

            obs, _, dones, infos = env_norm.step(action[np.newaxis])
            done = dones[0]
            info = infos[0]
            current_net_worth = info["net_worth"]

            # שמירת נתונים
            self.equity_curve.append(current_net_worth)
            self.actions_history.append(action.copy())

            # תאריך נוכחי
            date_idx = step + window_size
            if date_idx < len(common_idx):
                self.dates.append(common_idx[date_idx])

            step += 1

        env_norm.close()
        print(f"[Simulator] Completed: {step} trading days.")

        # ── חישוב מדדים ─────────────────────────────────────────────────────
        self.metrics = self._compute_metrics()
        self._print_metrics()
        return self.metrics

    def plot_all(self):
        """מייצר את כל הגרפים ושומר לתיקיית results."""
        self._plot_equity_curve()
        self._plot_actions_heatmap()
        self._plot_returns_distribution()
        print(f"[Simulator] Charts saved to '{RESULTS_DIR}'.")

    # ──────────────────────────────────────────────────────────────────────────
    # מדדי ביצוע
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-9)

        # ── Sharpe Ratio ─────────────────────────────────────────────────────
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # ── Sortino Ratio ─────────────────────────────────────────────────────
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino = 0.0

        # ── Max Drawdown ──────────────────────────────────────────────────────
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-9)
        max_dd = drawdown.max()

        # ── Win Rate ──────────────────────────────────────────────────────────
        win_rate = float(np.mean(returns > 0))

        # ── Profit Factor ─────────────────────────────────────────────────────
        gains  = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / (losses + 1e-9)

        # ── תשואה כוללת ───────────────────────────────────────────────────────
        total_return = (equity[-1] - equity[0]) / equity[0]

        # ── Calmar Ratio (annualised return / max drawdown) ───────────────────
        n_years = len(returns) / 252
        annualised_return = (1 + total_return) ** (1 / max(n_years, 1e-9)) - 1
        calmar = annualised_return / (max_dd + 1e-9)

        # ── Buy & Hold ─────────────────────────────────────────────────────────
        bh_return = self._buy_hold_return()

        spy_label = "SPY" if "SPY" in self.test_data else "avg_stocks"
        return {
            "total_return":    total_return,
            "annualised_return": annualised_return,
            "sharpe":          sharpe,
            "sortino":         sortino,
            "calmar":          calmar,
            "max_drawdown":    max_dd,
            "win_rate":        win_rate,
            "profit_factor":   profit_factor,
            "buy_hold_return": bh_return,
            "buy_hold_label":  spy_label,
            "final_equity":    equity[-1],
            "num_trades":      self._count_trades(),
        }

    def _buy_hold_return(self) -> float:
        """תשואת Buy & Hold על SPY (הבנצ'מרק הסטנדרטי). Fallback לממוצע כל המניות."""
        if "SPY" in self.test_data:
            df = self.test_data["SPY"]
            if len(df) >= 2:
                return float(
                    (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
                )
        # fallback אם SPY לא בנתונים
        returns = []
        for ticker, df in self.test_data.items():
            if len(df) < 2:
                continue
            r = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
            returns.append(r)
        return float(np.mean(returns)) if returns else 0.0

    def _count_trades(self) -> int:
        """סופר מספר פעולות שאינן 'החזק'."""
        count = 0
        for act in self.actions_history:
            if np.any(np.abs(act) > 0.05):
                count += 1
        return count

    def _print_metrics(self):
        m = self.metrics
        print("\n" + "=" * 50)
        print("  Paper Trading Results")
        print("=" * 50)
        print(f"  Total Return:      {m['total_return']:+.1%}")
        print(f"  Annualised Return: {m['annualised_return']:+.1%}")
        print(f"  Buy & Hold ({m.get('buy_hold_label','SPY')}): {m['buy_hold_return']:+.1%}")
        print(f"  Sharpe Ratio:      {m['sharpe']:.2f}")
        print(f"  Sortino Ratio:     {m['sortino']:.2f}")
        print(f"  Calmar Ratio:      {m['calmar']:.2f}")
        print(f"  Max Drawdown:      {m['max_drawdown']:.1%}")
        print(f"  Win Rate:          {m['win_rate']:.1%}")
        print(f"  Profit Factor:     {m['profit_factor']:.2f}")
        print(f"  Number of Trades:  {m['num_trades']}")
        print(f"  Final Equity:      ${m['final_equity']:,.0f}")
        print("=" * 50)

    # ──────────────────────────────────────────────────────────────────────────
    # גרפים
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_equity_curve(self):
        """עקומת אקוויטי מול Buy & Hold."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle("Portfolio Performance vs Buy & Hold", fontsize=14, fontweight="bold")

        equity = np.array(self.equity_curve)
        dates  = self.dates[: len(equity)]

        # ─── גרף עליון: אקוויטי ───────────────────────────────────────────
        ax = axes[0]
        ax.plot(dates, equity / equity[0] * 100, label="Agent Portfolio", color="#2196F3", lw=2)

        # Buy & Hold — SPY benchmark + top-3 by volume (AAPL, MSFT, NVDA)
        bh_show = []
        if "SPY" in self.test_data:
            bh_show.append(("SPY", "#FF9800"))
        for t, col in [("AAPL", "#4CAF50"), ("MSFT", "#9C27B0"), ("NVDA", "#F44336")]:
            if t in self.test_data and t != "SPY":
                bh_show.append((t, col))
        bh_show = bh_show[:4]   # max 4 B&H lines for readability

        for ticker, col in bh_show:
            close = self.test_data[ticker]["close"].values
            bh = close / close[0] * 100
            bh_dates = self.test_data[ticker].index[: len(dates)]
            ax.plot(bh_dates, bh[: len(dates)], label=f"B&H {ticker}",
                    color=col, lw=1.2, alpha=0.7, linestyle="--")

        ax.set_ylabel("ערך מנורמל (התחלה=100)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(100, color="gray", linestyle=":", lw=1)

        # ─── גרף תחתון: Drawdown ──────────────────────────────────────────
        ax2 = axes[1]
        peak     = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-9) * 100
        ax2.fill_between(dates, -drawdown, 0, color="#F44336", alpha=0.6, label="Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("תאריך")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "equity_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Simulator] → {path}")

    def _plot_actions_heatmap(self):
        """מפת חום של פעולות לאורך הזמן."""
        if not self.actions_history:
            return

        actions_arr = np.array(self.actions_history).T  # (num_stocks, time)
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(
            actions_arr,
            cmap="RdYlGn",
            center=0,
            xticklabels=max(1, len(self.dates) // 30),
            yticklabels=self.tickers,
            ax=ax,
            cbar_kws={"label": "Action (-1=מכור, 1=קנה)"},
        )
        ax.set_title("מפת חום של פעולות", fontsize=12)
        ax.set_xlabel("ימי מסחר")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "actions_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Simulator] → {path}")

    def _plot_returns_distribution(self):
        """התפלגות תשואות יומיות."""
        equity  = np.array(self.equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-9) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("התפלגות תשואות יומיות", fontsize=13, fontweight="bold")

        # היסטוגרמה
        ax = axes[0]
        ax.hist(returns, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean={returns.mean():.2f}%")
        ax.axvline(0, color="black", linestyle=":", lw=1)
        ax.set_xlabel("תשואה יומית (%)")
        ax.set_ylabel("תדירות")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats as scipy_stats
        ax2 = axes[1]
        (osm, osr), (slope, intercept, r) = scipy_stats.probplot(returns)
        ax2.plot(osm, osr, "o", color="#2196F3", alpha=0.5, markersize=3)
        ax2.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2, label=f"R²={r**2:.3f}")
        ax2.set_xlabel("Theoretical Quantiles")
        ax2.set_ylabel("Sample Quantiles")
        ax2.set_title("Q-Q Plot (נורמליות)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "returns_distribution.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Simulator] → {path}")
