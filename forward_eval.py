"""
forward_eval.py
================
Daily out-of-sample scorecard: the production model (load_ensemble(), exactly
what the live worker loads) rolled over market data from FORWARD_START to
today, against SPY buy & hold, appended to forward_log/forward_performance.csv.

FORWARD_START is after config data_end (2024-12-31), so none of this data was
used for training, validation or the promotion gate. It is a simulation on
real prices (same env / RiskManager as the promotion gate), not broker fills.

Usage:  python forward_eval.py
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FORWARD_START = "2025-01-01"
LOG_PATH = Path("forward_log") / "forward_performance.csv"
FIELDS = [
    "run_date", "data_last_date", "forward_start", "trading_days", "members",
    "model_return", "spy_return", "alpha",
    "model_sharpe", "spy_sharpe", "model_max_dd", "spy_max_dd", "git_sha",
]


def build_row(model_equity, spy_equity, *, run_date: str, data_last_date: str,
              trading_days: int, members: int, git_sha: str = "") -> dict:
    from promote_model import metrics_from_equity

    model = metrics_from_equity(np.asarray(model_equity, dtype=float))
    spy = metrics_from_equity(np.asarray(spy_equity, dtype=float))
    return {
        "run_date": run_date,
        "data_last_date": data_last_date,
        "forward_start": FORWARD_START,
        "trading_days": trading_days,
        "members": members,
        "model_return": round(model["total_return"], 6),
        "spy_return": round(spy["total_return"], 6),
        "alpha": round(model["total_return"] - spy["total_return"], 6),
        "model_sharpe": round(model["sharpe"], 4),
        "spy_sharpe": round(spy["sharpe"], 4),
        "model_max_dd": round(model["max_drawdown"], 6),
        "spy_max_dd": round(spy["max_drawdown"], 6),
        "git_sha": git_sha,
    }


def append_row(path: Path, row: dict) -> None:
    """One row per run_date: a re-run on the same day replaces that day's row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["run_date"] != row["run_date"]]
    rows.append({k: row[k] for k in FIELDS})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def run() -> dict:
    from config_loader import CFG
    from data_manager import DataManager
    from ensemble_agent import load_ensemble
    from benchmark import spy_buy_hold
    from promote_model import backtest_ensemble_equity_curve

    today = date.today().isoformat()
    dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=today)
    dm.load_all(force_download=False)
    aligned = dm.get_aligned_data()
    data = {t: df[df.index >= FORWARD_START] for t, df in aligned.items()}
    days = len(next(iter(data.values())))
    if days < 30:
        raise RuntimeError(f"Only {days} trading days since {FORWARD_START}; refusing to score.")

    agent = load_ensemble(dummy_data=data)
    model_equity = backtest_ensemble_equity_curve(agent, data)
    spy_equity = spy_buy_hold(data, CFG.initial_capital)

    return build_row(
        model_equity, spy_equity,
        run_date=today,
        data_last_date=str(next(iter(data.values())).index[-1].date()),
        trading_days=days,
        members=len(agent.members),
        git_sha=_git_sha(),
    )


def main() -> int:
    row = run()
    append_row(LOG_PATH, row)
    print(f"[Forward] {row['forward_start']} -> {row['data_last_date']} ({row['trading_days']} days, "
          f"{row['members']} member(s))")
    print(f"[Forward] model {row['model_return']:+.1%} (sharpe {row['model_sharpe']:.2f}, "
          f"maxDD {row['model_max_dd']:.1%}) | SPY {row['spy_return']:+.1%} "
          f"(sharpe {row['spy_sharpe']:.2f}, maxDD {row['spy_max_dd']:.1%}) | alpha {row['alpha']:+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
