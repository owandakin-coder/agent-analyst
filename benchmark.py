"""
benchmark.py
============
השוואת ביצועי הסוכן מול בנצ'מרקים סטנדרטיים:
  - SPY Buy & Hold
  - Equal-Weight Buy & Hold (כל המניות ב-universe)
  - Momentum פשוט (רכישת 5 המניות הטובות ביותר מדי חודש)

מייצר דוח טקסט + גרפים ל-results/plots/benchmark_*.png

שימוש:
    python benchmark.py                        # השוואה מלאה
    python benchmark.py --no-plot              # ללא גרפים
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows terminals often default stdout to a non-UTF-8 codepage (e.g. cp1255),
# which crashes on the box-drawing characters used in the report below.
if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from config_loader import CFG


RESULTS_DIR = Path(CFG.results_dir)
PLOTS_DIR   = Path(CFG.plots_dir)


# ══════════════════════════════════════════════════════════════════
# Benchmark strategies
# ══════════════════════════════════════════════════════════════════

def spy_buy_hold(test_data: dict[str, pd.DataFrame], initial_capital: float) -> np.ndarray:
    """SPY Buy & Hold — הבנצ'מרק הסטנדרטי."""
    if "SPY" not in test_data:
        raise ValueError("SPY missing from test_data")
    prices = test_data["SPY"]["close"].values
    units  = initial_capital / prices[0]
    return units * prices


def equal_weight_buy_hold(test_data: dict[str, pd.DataFrame], initial_capital: float) -> np.ndarray:
    """Equal-Weight Buy & Hold — שווי שוק שווה לכל מניה."""
    n = len(test_data)
    alloc = initial_capital / n
    portfolios = []
    for ticker, df in test_data.items():
        prices = df["close"].values
        units  = alloc / prices[0]
        portfolios.append(units * prices)

    min_len = min(len(p) for p in portfolios)
    return np.sum([p[:min_len] for p in portfolios], axis=0)


def momentum_strategy(test_data: dict[str, pd.DataFrame], initial_capital: float,
                      top_n: int = 5, rebalance_days: int = 21) -> np.ndarray:
    """
    Momentum פשוט: כל rebalance_days ימים, רוכש top_n מניות
    לפי ביצוע 90 הימים האחרונים.
    """
    tickers = [t for t in test_data if t != "SPY"]
    min_len = min(len(df) for df in test_data.values())
    portfolio = np.zeros(min_len)
    portfolio[0] = initial_capital

    cash       = initial_capital
    positions  = {}   # ticker → units

    for day in range(1, min_len):
        # Rebalance?
        if day % rebalance_days == 0 and day >= 90:
            # Liquidate all
            for ticker, units in positions.items():
                if ticker in test_data and day < len(test_data[ticker]["close"]):
                    cash += units * test_data[ticker]["close"].iloc[day]
            positions = {}

            # Rank by 90-day return
            scores = {}
            for ticker in tickers:
                df = test_data[ticker]
                if day >= 90 and day < len(df):
                    ret = (df["close"].iloc[day] - df["close"].iloc[day - 90]) / df["close"].iloc[day - 90]
                    scores[ticker] = ret

            if scores:
                top = sorted(scores, key=scores.get, reverse=True)[:top_n]
                alloc = cash / len(top)
                for ticker in top:
                    price = test_data[ticker]["close"].iloc[day]
                    positions[ticker] = alloc / price
                cash = 0.0

        # Mark to market
        value = cash
        for ticker, units in positions.items():
            if day < len(test_data[ticker]["close"]):
                value += units * test_data[ticker]["close"].iloc[day]
        portfolio[day] = value

    return portfolio


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

def compute_metrics(equity: np.ndarray, label: str = "") -> dict:
    """מחשב מדדי ביצוע סטנדרטיים מעקומת ה-equity."""
    returns = np.diff(equity) / (equity[:-1] + 1e-9)
    total_return    = (equity[-1] - equity[0]) / equity[0]
    n_years         = len(returns) / 252
    ann_return      = (1 + total_return) ** (1 / max(n_years, 1e-9)) - 1
    sharpe          = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    downside        = returns[returns < 0]
    sortino         = (returns.mean() / (downside.std() + 1e-9)) * np.sqrt(252) if len(downside) > 0 else 0.0
    peak            = np.maximum.accumulate(equity)
    drawdown        = (peak - equity) / (peak + 1e-9)
    max_dd          = drawdown.max()
    calmar          = ann_return / (max_dd + 1e-9)
    win_rate        = float(np.mean(returns > 0))
    volatility      = returns.std() * np.sqrt(252)

    return {
        "label":          label,
        "total_return":   total_return,
        "ann_return":     ann_return,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "max_drawdown":   max_dd,
        "win_rate":       win_rate,
        "volatility":     volatility,
        "final_value":    equity[-1],
    }


def compute_alpha_beta(agent_equity: np.ndarray, spy_equity: np.ndarray) -> tuple[float, float]:
    """Alpha ו-Beta של הסוכן יחסית ל-SPY."""
    min_len = min(len(agent_equity), len(spy_equity))
    a_ret = np.diff(agent_equity[:min_len]) / (agent_equity[:min_len][:-1] + 1e-9)
    s_ret = np.diff(spy_equity[:min_len])   / (spy_equity[:min_len][:-1]   + 1e-9)

    cov    = np.cov(a_ret, s_ret)
    beta   = cov[0, 1] / (cov[1, 1] + 1e-9)
    rf     = 0.05 / 252   # risk-free rate יומי
    alpha  = (a_ret.mean() - rf) - beta * (s_ret.mean() - rf)
    alpha_ann = alpha * 252
    return alpha_ann, beta


def compute_information_ratio(agent_equity: np.ndarray, spy_equity: np.ndarray) -> float:
    """Information Ratio: alpha יומי / tracking error."""
    min_len = min(len(agent_equity), len(spy_equity))
    a_ret = np.diff(agent_equity[:min_len]) / (agent_equity[:min_len][:-1] + 1e-9)
    s_ret = np.diff(spy_equity[:min_len])   / (spy_equity[:min_len][:-1]   + 1e-9)
    active = a_ret - s_ret
    if active.std() < 1e-9:
        return 0.0
    return float((active.mean() / active.std()) * np.sqrt(252))


# ══════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════

def print_report(results: list[dict], agent_equity: np.ndarray, spy_equity: np.ndarray):
    alpha, beta = compute_alpha_beta(agent_equity, spy_equity)
    ir          = compute_information_ratio(agent_equity, spy_equity)

    W = 58
    print("\n" + "═" * W)
    print(f"  BENCHMARK COMPARISON REPORT")
    print("═" * W)
    header = f"{'Metric':<22} " + " ".join(f"{r['label']:>10}" for r in results)
    print(header)
    print("─" * W)

    rows = [
        ("Total Return",   "total_return",  "{:>+9.1%}"),
        ("Ann. Return",    "ann_return",    "{:>+9.1%}"),
        ("Sharpe Ratio",   "sharpe",        "{:>9.2f}"),
        ("Sortino Ratio",  "sortino",       "{:>9.2f}"),
        ("Calmar Ratio",   "calmar",        "{:>9.2f}"),
        ("Max Drawdown",   "max_drawdown",  "{:>9.1%}"),
        ("Win Rate",       "win_rate",      "{:>9.1%}"),
        ("Volatility",     "volatility",    "{:>9.1%}"),
        ("Final Value",    "final_value",   "${:>9,.0f}"),
    ]

    for label, key, fmt in rows:
        vals = " ".join(fmt.format(r[key]) for r in results)
        print(f"  {label:<20} {vals}")

    print("─" * W)
    agent_m = results[0]
    spy_m   = results[1]
    alpha_pct = agent_m["ann_return"] - spy_m["ann_return"]
    print(f"  {'Alpha vs SPY':<20} {alpha_pct:>+9.1%}")
    print(f"  {'Beta':<20} {beta:>9.2f}")
    print(f"  {'Information Ratio':<20} {ir:>9.2f}")
    print("═" * W)

    # Verdict
    print("\n  VERDICT:")
    beats_spy    = agent_m["ann_return"] > spy_m["ann_return"]
    better_sharp = agent_m["sharpe"]     > spy_m["sharpe"]
    less_dd      = agent_m["max_drawdown"] < spy_m["max_drawdown"]

    if beats_spy and better_sharp:
        print("  ✅ Agent BEATS SPY on both return and Sharpe ratio")
    elif beats_spy:
        print("  ⚠️  Agent beats SPY on return but has lower Sharpe (more risk)")
    elif better_sharp:
        print("  ⚠️  Agent has better Sharpe but lower return than SPY")
    else:
        print("  ❌ Agent UNDERPERFORMS SPY on both return and Sharpe")

    if less_dd:
        print("  ✅ Lower max drawdown than SPY (better capital preservation)")
    else:
        print("  ⚠️  Higher max drawdown than SPY")

    if ir > 0.5:
        print(f"  ✅ Strong Information Ratio ({ir:.2f}) — consistent alpha generation")
    elif ir > 0:
        print(f"  ⚠️  Positive but weak Information Ratio ({ir:.2f})")
    else:
        print(f"  ❌ Negative Information Ratio ({ir:.2f}) — agent adds no alpha")
    print()


# ══════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════

def plot_comparison(equities: list[tuple[str, np.ndarray]], results: list[dict],
                    agent_equity: np.ndarray, spy_equity: np.ndarray):
    if not HAS_PLOT:
        print("[Benchmark] matplotlib not available — skipping plots.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    colors = ["#00d4aa", "#ff9f0a", "#0a84ff", "#bf5af2"]

    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0f")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("ATZMA Agent vs Benchmarks", fontsize=14, fontweight="bold",
                 color="white", y=0.98)

    ax_style = dict(facecolor="#111113", tick_params=dict(colors="#636366"))

    # ── 1. Equity curves ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#111113")
    ax1.tick_params(colors="#636366")
    for (label, eq), col in zip(equities, colors):
        ax1.plot(eq / eq[0] * 100, label=label, color=col, linewidth=1.5)
    ax1.axhline(100, color="#3a3a3c", linewidth=0.8, linestyle="--")
    ax1.set_title("Equity Curves (normalized to 100)", color="white", fontsize=11)
    ax1.set_ylabel("Value", color="#aeaeb2")
    ax1.legend(facecolor="#1a1a1c", edgecolor="#3a3a3c", labelcolor="white")
    ax1.grid(True, alpha=0.15, color="#3a3a3c")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#3a3a3c")

    # ── 2. Drawdowns ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#111113")
    ax2.tick_params(colors="#636366")
    for (label, eq), col in zip(equities[:2], colors):
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / (peak + 1e-9) * 100
        ax2.fill_between(range(len(dd)), -dd, 0, alpha=0.35, color=col, label=label)
        ax2.plot(-dd, color=col, linewidth=0.8, alpha=0.8)
    ax2.set_title("Drawdown (%)", color="white", fontsize=11)
    ax2.set_ylabel("%", color="#aeaeb2")
    ax2.legend(facecolor="#1a1a1c", edgecolor="#3a3a3c", labelcolor="white", fontsize=9)
    ax2.grid(True, alpha=0.15, color="#3a3a3c")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#3a3a3c")

    # ── 3. Metrics bar chart ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#111113")
    ax3.tick_params(colors="#636366")
    metrics_keys = ["sharpe", "sortino", "calmar"]
    labels_m     = ["Sharpe", "Sortino", "Calmar"]
    x = np.arange(len(metrics_keys))
    width = 0.8 / len(results)
    for i, (res, col) in enumerate(zip(results, colors)):
        vals = [res[k] for k in metrics_keys]
        ax3.bar(x + i * width - (len(results) - 1) * width / 2,
                vals, width, label=res["label"], color=col, alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_m, color="#aeaeb2")
    ax3.set_title("Risk-Adjusted Metrics", color="white", fontsize=11)
    ax3.axhline(0, color="#3a3a3c", linewidth=0.8)
    ax3.legend(facecolor="#1a1a1c", edgecolor="#3a3a3c", labelcolor="white", fontsize=9)
    ax3.grid(True, alpha=0.15, color="#3a3a3c", axis="y")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#3a3a3c")

    out = PLOTS_DIR / "benchmark_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0f")
    plt.close()
    print(f"[Benchmark] Chart saved → {out}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def run_benchmark(agent_equity: np.ndarray | None = None,
                  test_data: dict | None = None,
                  plot: bool = True) -> dict:
    """
    מריץ השוואת בנצ'מרק מלאה.
    אם agent_equity לא סופק, נטען מ-results/equity_data.pkl.
    """
    # טעינת equity הסוכן
    if agent_equity is None:
        eq_path = RESULTS_DIR / "equity_data.pkl"
        if not eq_path.exists():
            print(f"[Benchmark] {eq_path} not found. Run simulate first.")
            return {}
        with open(eq_path, "rb") as f:
            agent_equity, dates = pickle.load(f)
        agent_equity = np.array(agent_equity)

    # טעינת נתוני טסט
    if test_data is None:
        from data_manager import DataManager
        dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=CFG.data_end)
        dm.load_all(force_download=False)
        all_data  = dm.get_aligned_data()
        test_data = {t: df[(df.index >= CFG.test_start) & (df.index <= CFG.test_end)]
                     for t, df in all_data.items()}

    initial_capital = float(agent_equity[0])

    # בנצ'מרקים
    spy_eq   = spy_buy_hold(test_data, initial_capital)
    ew_eq    = equal_weight_buy_hold(test_data, initial_capital)
    mom_eq   = momentum_strategy(test_data, initial_capital)

    # חיתוך לאורך הקצר ביותר
    n = min(len(agent_equity), len(spy_eq), len(ew_eq), len(mom_eq))
    agent_eq = agent_equity[:n]
    spy_eq   = spy_eq[:n]
    ew_eq    = ew_eq[:n]
    mom_eq   = mom_eq[:n]

    # מדדים
    results = [
        compute_metrics(agent_eq, "ATZMA Agent"),
        compute_metrics(spy_eq,   "SPY B&H"),
        compute_metrics(ew_eq,    "Equal-Weight"),
        compute_metrics(mom_eq,   "Momentum"),
    ]

    print_report(results, agent_eq, spy_eq)

    if plot and HAS_PLOT:
        equities = [
            ("ATZMA Agent",   agent_eq),
            ("SPY B&H",       spy_eq),
            ("Equal-Weight",  ew_eq),
            ("Momentum",      mom_eq),
        ]
        plot_comparison(equities, results, agent_eq, spy_eq)

    # שמירת תוצאות
    RESULTS_DIR.mkdir(exist_ok=True)
    import json
    out = {r["label"]: {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in r.items()} for r in results}
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Benchmark] Results saved → {RESULTS_DIR}/benchmark_results.json")

    return out


def parse_args():
    p = argparse.ArgumentParser(description="ATZMA Benchmark Comparison")
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(plot=not args.no_plot)
