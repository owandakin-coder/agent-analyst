"""
promote_model.py
=================
Gate between "a fresh model finished training" and "this model is what live
trading actually uses".

Without this gate, `.github/workflows/retrain.yml` runs unattended once a
month, trains a new model on a fixed step budget, and commits + pushes
`models/final_model.zip` straight to `main` regardless of how the new model
performs. A bad seed, a regime shift in the training window, or an
overfit run would go live with zero human review and zero automated check —
the exact failure mode the rest of this codebase's capital-preservation
controls (RiskManager drawdown halts, MultiAgentDecisionEngine unanimous
voting) are designed to guard against at the trade level, but not at the
model-deployment level.

This module backtests the freshly trained candidate against the model it
would replace, on the same held-out test period used everywhere else
(CFG.test_start .. CFG.test_end), and only allows promotion if the
candidate does not regress beyond a small tolerance on Sharpe, max
drawdown, or annualized return — and does not badly lag a trivial SPY
buy-and-hold over the same window. That second check exists because
"doesn't regress vs. the previous model" alone lets mediocrity persist
indefinitely if the very first deployed model was never actually good:
each retrain only has to hold its own against a baseline that might
itself be bad. Comparing against SPY catches a model that's actively
losing money while the market just sits there, which "beat the last
model" alone would miss.

Usage (wired into retrain.yml around the existing training steps):
    python promote_model.py --previous models/previous_model.zip \
                             --previous-norm models/previous_vec_normalize.pkl
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# Windows terminals often default stdout to a non-UTF-8 codepage (e.g. cp1255),
# which crashes on the emoji used in the alerts/prints below.
if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config_loader import CFG

MODEL_DIR = Path(CFG.model_dir)

# A candidate must not regress beyond these tolerances vs. the model it
# would replace. Tolerances exist because re-training on a slightly
# different data window naturally produces noisy metrics even for an
# equally good policy — only a real regression should block promotion.
SHARPE_TOLERANCE = 0.10
DRAWDOWN_TOLERANCE_PP = 0.03
RETURN_TOLERANCE_PP = 0.05

# Deliberately loose: a risk-managed strategy can legitimately lag a raw,
# unmanaged index by a lot in absolute return while being far better on a
# risk-adjusted basis. This tolerance exists only to catch the pathological
# case — a model that loses real money while the market does nothing bad —
# not to demand the model beat a naive benchmark on every run.
SPY_UNDERPERFORMANCE_TOLERANCE_PP = 0.20


def should_promote(candidate: dict, previous: dict | None, spy: dict | None = None) -> tuple[bool, str]:
    """Pure decision function so the policy is unit-testable without a GPU or market data."""
    reasons = []

    if previous is not None:
        if candidate["sharpe"] < previous["sharpe"] - SHARPE_TOLERANCE:
            reasons.append(
                f"sharpe {candidate['sharpe']:.2f} < production {previous['sharpe']:.2f} - {SHARPE_TOLERANCE}"
            )
        if candidate["max_drawdown"] > previous["max_drawdown"] + DRAWDOWN_TOLERANCE_PP:
            reasons.append(
                f"max_drawdown {candidate['max_drawdown']:.1%} > production "
                f"{previous['max_drawdown']:.1%} + {DRAWDOWN_TOLERANCE_PP:.0%}"
            )
        if candidate["annualized_return"] < previous["annualized_return"] - RETURN_TOLERANCE_PP:
            reasons.append(
                f"annualized_return {candidate['annualized_return']:+.1%} < production "
                f"{previous['annualized_return']:+.1%} - {RETURN_TOLERANCE_PP:.0%}"
            )

    if spy is not None:
        if candidate["annualized_return"] < spy["annualized_return"] - SPY_UNDERPERFORMANCE_TOLERANCE_PP:
            reasons.append(
                f"annualized_return {candidate['annualized_return']:+.1%} is more than "
                f"{SPY_UNDERPERFORMANCE_TOLERANCE_PP:.0%} below SPY buy-and-hold "
                f"({spy['annualized_return']:+.1%}) over the same window"
            )

    if reasons:
        return False, "; ".join(reasons)
    if previous is None:
        return True, "no existing production model to compare against — promoting first candidate"
    return True, "candidate meets or exceeds production on sharpe, drawdown and return, and isn't badly lagging SPY"


def _load_test_data() -> dict:
    from data_manager import DataManager

    dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=CFG.data_end)
    dm.load_all(force_download=False)
    all_data = dm.get_aligned_data()
    return {
        ticker: df[(df.index >= CFG.test_start) & (df.index <= CFG.test_end)]
        for ticker, df in all_data.items()
    }


def _sub_period_bounds(n: int = 3, start=None, end=None) -> list[tuple]:
    """Splits a date range (CFG.test_start..CFG.test_end by default) into n
    roughly-equal calendar sub-periods, so a candidate's aggregate win/loss
    over the whole window can be checked for whether it actually holds up
    piece by piece, or is an artifact of one stretch dominating the
    full-period number. (See the 2026-09-17 walk-forward finding: a
    candidate that cleared the aggregate single-window gate showed no
    improvement across independent windows — this is a lighter-weight,
    no-retrain way to get some of that same signal on the *specific* test
    window the gate already uses.)"""
    import pandas as pd

    start = pd.Timestamp(start if start is not None else CFG.test_start)
    end = pd.Timestamp(end if end is not None else CFG.test_end)
    edges = pd.date_range(start, end, periods=n + 1)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def sub_period_report(candidate_model: Path, candidate_norm: Path,
                       previous_model: Path, previous_norm: Path,
                       n: int = 3, test_data: dict | None = None) -> list[dict]:
    """Diagnostic only — does not affect should_promote(). Backtests the
    already-trained candidate/previous models (no retraining) on n slices
    of the same test window, to surface whether an aggregate win is
    consistent or driven by a single sub-period. test_data overrides the
    default CFG-driven load, for testing against synthetic data."""
    full_test_data = test_data if test_data is not None else _load_test_data()
    any_df = next(iter(full_test_data.values()))
    bounds_kwargs = {} if test_data is None else {"start": any_df.index.min(), "end": any_df.index.max()}
    rows = []
    for period_start, period_end in _sub_period_bounds(n, **bounds_kwargs):
        sub_data = {
            ticker: df[(df.index >= period_start) & (df.index <= period_end)]
            for ticker, df in full_test_data.items()
        }
        if not sub_data or any(len(df) < 30 for df in sub_data.values()):
            continue
        row = {
            "start": period_start.date().isoformat(),
            "end": period_end.date().isoformat(),
            "candidate": evaluate_model_metrics(candidate_model, candidate_norm, sub_data),
        }
        if previous_model.exists() and previous_norm.exists():
            row["previous"] = evaluate_model_metrics(previous_model, previous_norm, sub_data)
        spy = compute_spy_baseline(sub_data)
        if spy is not None:
            row["spy"] = spy
        rows.append(row)
    return rows


def backtest_equity_curve(model, vec_norm_stats, test_data: dict) -> np.ndarray:
    """Rolls a trained model deterministically over test_data.

    Mirrors the walk_forward_eval.py rollout (same env, same RiskManager
    scaling) so promotion decisions use the identical methodology as the
    rest of the evaluation pipeline instead of a bespoke shortcut.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trading_env import TradingEnvironment
    from risk_manager import RiskManager

    # max_drawdown_stop=1.0: this gate must compare full-period performance,
    # not whichever candidate happens to survive longer before hitting the
    # training-time hard stop. RiskManager.scale_action() below is the real,
    # live-matching risk control (reduce/halt/recover) — see
    # execution_simulator.py for the full reasoning.
    vec_env = DummyVecEnv([lambda: TradingEnvironment(test_data, max_drawdown_stop=1.0)])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    norm_env.obs_rms = vec_norm_stats.obs_rms
    norm_env.ret_rms = vec_norm_stats.ret_rms

    risk_mgr = RiskManager(CFG.initial_capital)
    obs = norm_env.reset()
    done = False
    equity = [CFG.initial_capital]
    net_worth = CFG.initial_capital

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = action[0]
        risk_mgr.update(net_worth)
        action = risk_mgr.scale_action(action)
        obs, _, dones, infos = norm_env.step(action[np.newaxis])
        done = bool(dones[0])
        net_worth = infos[0].get("net_worth", net_worth)
        equity.append(net_worth)

    norm_env.close()
    return np.array(equity)


def metrics_from_equity(equity: np.ndarray) -> dict:
    from benchmark import compute_metrics
    from walk_forward_eval import _annualized_return

    metrics = compute_metrics(equity)
    metrics["annualized_return"] = _annualized_return(metrics["total_return"], len(equity) - 1)
    return metrics


def compute_spy_baseline(test_data: dict) -> dict | None:
    """SPY buy-and-hold over the same window, as the 'did nothing' baseline."""
    from benchmark import spy_buy_hold

    try:
        equity = spy_buy_hold(test_data, CFG.initial_capital)
    except (ValueError, KeyError):
        return None
    return metrics_from_equity(np.asarray(equity, dtype=float))


def evaluate_model_metrics(model_path: Path, vec_norm_path: Path, test_data: dict) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trading_env import TradingEnvironment

    model = PPO.load(str(model_path))
    dummy_env = DummyVecEnv([lambda: TradingEnvironment(test_data)])
    vec_norm = VecNormalize.load(str(vec_norm_path), dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False

    equity = backtest_equity_curve(model, vec_norm, test_data)
    return metrics_from_equity(equity)


def run_gate(
    candidate_model: Path,
    candidate_norm: Path,
    previous_model: Path,
    previous_norm: Path,
) -> tuple[bool, str, dict, dict | None, dict | None]:
    test_data = _load_test_data()

    candidate_metrics = evaluate_model_metrics(candidate_model, candidate_norm, test_data)

    previous_metrics = None
    if previous_model.exists() and previous_norm.exists():
        previous_metrics = evaluate_model_metrics(previous_model, previous_norm, test_data)

    spy_metrics = compute_spy_baseline(test_data)

    promote_ok, reason = should_promote(candidate_metrics, previous_metrics, spy_metrics)
    return promote_ok, reason, candidate_metrics, previous_metrics, spy_metrics


def _fmt(metrics: dict) -> str:
    return (
        f"sharpe={metrics['sharpe']:.2f} "
        f"return={metrics['annualized_return']:+.1%} "
        f"max_dd={metrics['max_drawdown']:.1%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a freshly trained model before it goes live.")
    parser.add_argument("--candidate", default=str(MODEL_DIR / "final_model.zip"))
    parser.add_argument("--candidate-norm", default=str(MODEL_DIR / "vec_normalize.pkl"))
    parser.add_argument("--previous", default=str(MODEL_DIR / "previous_model.zip"))
    parser.add_argument("--previous-norm", default=str(MODEL_DIR / "previous_vec_normalize.pkl"))
    parser.add_argument("--sub-periods", type=int, default=0,
                         help="Also print a diagnostic breakdown of the candidate/previous/SPY "
                              "over N slices of the test window, to show whether an aggregate "
                              "win is consistent or driven by one stretch. Informational only — "
                              "does not affect the promote/reject decision.")
    args = parser.parse_args()

    candidate_model = Path(args.candidate)
    candidate_norm = Path(args.candidate_norm)
    previous_model = Path(args.previous)
    previous_norm = Path(args.previous_norm)

    if not candidate_model.exists() or not candidate_norm.exists():
        print(f"[Promote] Candidate model not found at {candidate_model} — nothing to gate.")
        return 1

    from notifications import send_operator_alert

    promote_ok, reason, candidate_metrics, previous_metrics, spy_metrics = run_gate(
        candidate_model, candidate_norm, previous_model, previous_norm
    )

    print(f"[Promote] Candidate: {_fmt(candidate_metrics)}")
    if previous_metrics:
        print(f"[Promote] Previous : {_fmt(previous_metrics)}")
    if spy_metrics:
        print(f"[Promote] SPY B&H  : {_fmt(spy_metrics)}")

    if args.sub_periods > 1:
        print(f"\n[Promote] Sub-period breakdown ({args.sub_periods} slices of the test window, informational only):")
        for row in sub_period_report(candidate_model, candidate_norm, previous_model, previous_norm, n=args.sub_periods):
            line = f"  {row['start']} -> {row['end']} | candidate: {_fmt(row['candidate'])}"
            if "previous" in row:
                line += f" | previous: {_fmt(row['previous'])}"
            if "spy" in row:
                line += f" | spy: {_fmt(row['spy'])}"
            print(line)

    if promote_ok:
        print(f"[Promote] PROMOTED — {reason}")
        send_operator_alert(f"✅ Monthly retrain promoted to production — {reason}")
        return 0

    print(f"[Promote] REJECTED — {reason}")
    if previous_model.exists() and previous_norm.exists():
        shutil.copy2(previous_model, candidate_model)
        shutil.copy2(previous_norm, candidate_norm)
        print("[Promote] Reverted final_model.zip / vec_normalize.pkl to the previous production model.")
    send_operator_alert(
        f"⚠️ Monthly retrain REJECTED, production model unchanged — {reason}"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
