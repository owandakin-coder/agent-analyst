"""
evaluate.py — Backtest a trained model on unseen data and produce performance charts.

Usage:
    python evaluate.py                            # evaluate default model on test data
    python evaluate.py --model-name ppo_crypto    # specify model
    python evaluate.py --export-onnx              # also export to ONNX for deployment
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from data import fetch_ohlcv
from env import CryptoTradingEnv
from agent import load_agent, export_to_onnx, make_env
from stable_baselines3.common.vec_env import DummyVecEnv


PLOTS_DIR = Path("plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--start", default="2024-01-01")   # Out-of-sample test period
    p.add_argument("--end", default="2024-12-01")
    p.add_argument("--initial-balance", type=float, default=10_000)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--model-name", default="ppo_crypto")
    p.add_argument("--export-onnx", action="store_true")
    return p.parse_args()


def run_backtest(agent, env: CryptoTradingEnv) -> dict:
    obs, _ = env.reset()
    done = False
    actions = []

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        actions.append(float(action[0]))

    metrics = env.get_metrics()
    metrics["actions"] = actions
    metrics["portfolio_history"] = env.portfolio_history
    metrics["trade_history"] = env.trade_history
    return metrics


def compute_buy_and_hold(df, initial_balance: float) -> list[float]:
    """Benchmark: buy at start, hold until end."""
    entry_price = df.iloc[50]["close"]  # same start step as env
    units = initial_balance / entry_price
    return [units * df.iloc[i]["close"] for i in range(50, len(df))]


def plot_results(metrics: dict, bh_values: list[float], df, model_name: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    portfolio = np.array(metrics["portfolio_history"])
    bh = np.array(bh_values[:len(portfolio)])
    actions = metrics["actions"]
    steps = np.arange(len(portfolio))

    prices = df["close"].values[50:50 + len(portfolio)]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Backtest: {model_name}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── 1. Portfolio value vs Buy&Hold ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, portfolio, label="PPO Agent", color="#2196F3", linewidth=1.5)
    ax1.plot(steps, bh, label="Buy & Hold", color="#FF9800", linewidth=1.5, linestyle="--")
    ax1.axhline(y=10_000, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_title("Portfolio Value vs Buy & Hold")
    ax1.set_ylabel("USD")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 2. Drawdown ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / peak * 100
    ax2.fill_between(steps, -drawdown, 0, color="#F44336", alpha=0.5)
    ax2.set_title("Drawdown %")
    ax2.set_ylabel("%")
    ax2.grid(True, alpha=0.3)

    # ── 3. Position allocation over time ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(np.arange(len(actions)), actions, color="#4CAF50", linewidth=0.8)
    ax3.axhline(y=0, color="gray", linewidth=0.8)
    ax3.fill_between(np.arange(len(actions)), actions, 0,
                     where=[a > 0 for a in actions], color="#4CAF50", alpha=0.3, label="Long")
    ax3.fill_between(np.arange(len(actions)), actions, 0,
                     where=[a < 0 for a in actions], color="#F44336", alpha=0.3, label="Short/Cash")
    ax3.set_title("Agent Position Allocation [-1=cash, +1=full]")
    ax3.set_ylabel("Position")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── 4. Returns distribution ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    returns = np.diff(portfolio) / portfolio[:-1] * 100
    ax4.hist(returns, bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax4.axvline(x=0, color="red", linewidth=1)
    ax4.set_title("Step Returns Distribution (%)")
    ax4.set_xlabel("%")
    ax4.grid(True, alpha=0.3)

    # ── 5. Price + action overlay ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5_twin = ax5.twinx()
    ax5.plot(np.arange(len(prices)), prices, color="#607D8B", linewidth=0.8, alpha=0.7)
    ax5_twin.plot(np.arange(len(actions)), actions, color="#FF5722", linewidth=0.5, alpha=0.6)
    ax5.set_title("Price vs Agent Actions")
    ax5.set_ylabel("BTC Price")
    ax5_twin.set_ylabel("Action", color="#FF5722")
    ax5.grid(True, alpha=0.3)

    output_path = PLOTS_DIR / f"{model_name}_backtest.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Eval] Chart saved → {output_path}")
    plt.close()


def print_report(metrics: dict, bh_values: list[float], initial_balance: float) -> None:
    bh_return = (bh_values[-1] / initial_balance - 1) * 100

    print("\n═══════════════════════════════════════════════════")
    print("  BACKTEST REPORT")
    print("═══════════════════════════════════════════════════")
    print(f"  Total Return:        {metrics['total_return_pct']:>+8.2f}%")
    print(f"  Buy & Hold Return:   {bh_return:>+8.2f}%")
    print(f"  Alpha vs B&H:        {metrics['total_return_pct'] - bh_return:>+8.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.3f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Win Rate:            {metrics['win_rate_pct']:>8.2f}%")
    print(f"  Final Portfolio:     ${metrics['final_value']:>10,.2f}")
    print(f"  Steps:               {metrics['n_steps']:>8,}")
    print("═══════════════════════════════════════════════════\n")


def main():
    args = parse_args()

    df = fetch_ohlcv(args.symbol, args.interval, args.start, args.end)
    print(f"[Eval] Test data: {len(df)} candles ({args.start} → {args.end})")

    env_kwargs = {"initial_balance": args.initial_balance, "trading_fee": args.fee}
    test_env = CryptoTradingEnv(df, **env_kwargs)

    # Load agent with VecNormalize (for obs normalisation during inference)
    dummy_vec = DummyVecEnv([make_env(df, **env_kwargs)])
    agent, vec_norm = load_agent(dummy_vec, args.model_name)

    # Backtest
    metrics = run_backtest(agent, test_env)
    bh_values = compute_buy_and_hold(df, args.initial_balance)

    print_report(metrics, bh_values, args.initial_balance)
    plot_results(metrics, bh_values, df, args.model_name)

    if args.export_onnx:
        export_to_onnx(agent, args.model_name)


if __name__ == "__main__":
    main()
