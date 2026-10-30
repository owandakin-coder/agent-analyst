"""
alpha_significance.py
======================
Is the production model's daily excess return over SPY distinguishable from
zero, or is it noise? Circular block bootstrap of the daily return difference
(blocks keep short-run autocorrelation), over the same out-of-sample window as
forward_eval.py (FORWARD_START..today).

Usage:  python alpha_significance.py
"""

from __future__ import annotations

import sys

import numpy as np

if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRADING_DAYS = 252


def daily_returns(equity) -> np.ndarray:
    e = np.asarray(equity, dtype=float)
    return e[1:] / e[:-1] - 1.0


def bootstrap_excess(model_ret, spy_ret, *, block: int = 20, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Circular block bootstrap of mean(model - spy), annualised. Series are
    aligned from the start and truncated to the shorter one."""
    m = np.asarray(model_ret, dtype=float)
    s = np.asarray(spy_ret, dtype=float)
    n = min(len(m), len(s))
    if n < 2 * block:
        raise ValueError(f"Need at least {2 * block} daily returns, got {n}.")
    d = m[:n] - s[:n]

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = d[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1) * TRADING_DAYS

    observed = float(d.mean() * TRADING_DAYS)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "days": n,
        "annualised_excess": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_excess_le_0": float(np.mean(means <= 0.0)),
        "p_excess_ge_0": float(np.mean(means >= 0.0)),
        "tracking_error": float(d.std(ddof=1) * np.sqrt(TRADING_DAYS)),
    }


def main() -> int:
    from forward_eval import compute_equities, FORWARD_START

    _, _, model_equity, spy_equity, today = compute_equities()
    r = bootstrap_excess(daily_returns(model_equity), daily_returns(spy_equity))
    print(f"[Significance] {FORWARD_START} -> {today}: {r['days']} daily returns, 20-day blocks, 10,000 resamples")
    print(f"[Significance] annualised excess return vs SPY: {r['annualised_excess']:+.1%} "
          f"(95% CI {r['ci95_low']:+.1%} .. {r['ci95_high']:+.1%}), tracking error {r['tracking_error']:.1%}")
    print(f"[Significance] P(excess <= 0) = {r['p_excess_le_0']:.2f}   P(excess >= 0) = {r['p_excess_ge_0']:.2f}")
    if r["ci95_low"] > 0:
        print("[Significance] verdict: positive excess return, unlikely to be noise.")
    elif r["ci95_high"] < 0:
        print("[Significance] verdict: negative excess return, unlikely to be noise.")
    else:
        print("[Significance] verdict: the interval includes zero. This window cannot tell the model "
              "apart from SPY either way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
