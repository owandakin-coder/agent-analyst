"""
walk_forward_eval.py
====================
הערכת Walk-Forward אמיתית — מאמן ומעריך על מספר חלונות עצמאיים.

במקום אימון אחד על כל הנתונים:
  חלון 1: train 2015-2017 → test 2018 H1
  חלון 2: train 2015-2018 → test 2019 H1
  חלון 3: train 2015-2019 → test 2020 H1
  חלון 4: train 2015-2020 → test 2021 H1

כל חלון עצמאי לחלוטין. זה מונע data leakage.

שימוש:
    python walk_forward_eval.py              # הרצה מלאה
    python walk_forward_eval.py --fast       # 50K steps (לבדיקה מהירה)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Windows terminals often default stdout to a non-UTF-8 codepage (e.g. cp1255),
# which crashes on the box-drawing characters used in the banners below.
if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from config_loader import CFG


RESULTS_DIR = Path(CFG.results_dir)
PLOTS_DIR   = Path(CFG.plots_dir)


# ══════════════════════════════════════════════════════════════════
# Window generator
# ══════════════════════════════════════════════════════════════════

def generate_windows(
    start_date: str = "2015-01-01",
    n_windows:  int = 4,
    train_months: int = 36,
    test_months:  int = 6,
) -> list[dict]:
    """
    יוצר חלונות walk-forward עם train expanding.
    """
    windows = []
    base = datetime.strptime(start_date, "%Y-%m-%d")

    for i in range(n_windows):
        test_start = base + relativedelta(months=train_months + i * test_months)
        test_end   = test_start + relativedelta(months=test_months) - relativedelta(days=1)

        windows.append({
            "window":      i + 1,
            "train_start": start_date,
            "train_end":   (test_start - relativedelta(days=1)).strftime("%Y-%m-%d"),
            "test_start":  test_start.strftime("%Y-%m-%d"),
            "test_end":    test_end.strftime("%Y-%m-%d"),
        })

    return windows


# ══════════════════════════════════════════════════════════════════
# Single window evaluation
# ══════════════════════════════════════════════════════════════════

def evaluate_window(
    window: dict,
    all_data: dict[str, pd.DataFrame],
    timesteps: int = 100_000,
    seed: int = 42,
    model_params: dict | None = None,
    use_transformer_policy: bool = False,
) -> dict:
    """מאמן ומעריך על חלון אחד. מחזיר מדדים.

    By default trains a plain MlpPolicy with fixed hyperparameters — fast,
    but not the architecture live trading actually uses. Pass
    use_transformer_policy=True (and optionally model_params from
    models/training_meta.pkl's best_params) to match the production
    training_pipeline.py setup, so this evaluates whether the *deployed*
    architecture generalizes across windows, not just a cheap proxy.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trading_env import TradingEnvironment
    from risk_manager import RiskManager

    w = window["window"]
    print(f"\n{'─' * 50}")
    print(f"  Window {w}: Train {window['train_start']} → {window['train_end']}")
    print(f"            Test  {window['test_start']}  → {window['test_end']}")
    print(f"{'─' * 50}")

    # פילוח נתונים
    train_data = {t: df[(df.index >= window["train_start"]) &
                        (df.index <= window["train_end"])].copy()
                  for t, df in all_data.items()}
    test_data  = {t: df[(df.index >= window["test_start"]) &
                        (df.index <= window["test_end"])].copy()
                  for t, df in all_data.items()}

    # בדיקה שיש מספיק נתונים
    for ticker, df in train_data.items():
        if len(df) < 60:
            print(f"  [WARN] {ticker} has only {len(df)} train days — skipping window")
            return {"window": w, "skipped": True}

    # ── אימון ──────────────────────────────────────────────────────
    train_env = DummyVecEnv([lambda: TradingEnvironment(train_data)])
    norm_env  = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ppo_kwargs = dict(model_params or {})
    ppo_kwargs.setdefault("learning_rate", 3e-4)
    ppo_kwargs.setdefault("n_steps", 2048)
    ppo_kwargs.setdefault("batch_size", 64)
    ppo_kwargs.setdefault("n_epochs", 10)
    ppo_kwargs.setdefault("gamma", 0.99)

    policy_kwargs = None
    if use_transformer_policy:
        from transformer_policy import TransformerExtractor
        from training_pipeline import TRANSFORMER_CONFIG, POLICY_HEAD_ARCH
        policy_kwargs = dict(
            features_extractor_class=TransformerExtractor,
            features_extractor_kwargs=dict(TRANSFORMER_CONFIG),
            net_arch=list(POLICY_HEAD_ARCH),
        )

    model = PPO(
        "MlpPolicy", norm_env,
        verbose=0, seed=seed,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=timesteps)
    norm_env.training    = False
    norm_env.norm_reward = False

    # ── הערכה ─────────────────────────────────────────────────────
    # max_drawdown_stop=1.0: don't let the training-time hard episode-stop
    # (CFG.drawdown_halt) truncate the *reported* test-window performance.
    # Live trading never uses that hard stop — only RiskManager.scale_action()
    # below does, and it can reduce/halt/recover, so this window's numbers
    # should reflect that instead of a premature cutoff at the worst point.
    test_env_raw = DummyVecEnv([lambda: TradingEnvironment(test_data, max_drawdown_stop=1.0)])
    test_env     = VecNormalize(test_env_raw, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)
    test_env.obs_rms = norm_env.obs_rms
    test_env.ret_rms = norm_env.ret_rms

    risk_mgr  = RiskManager(CFG.initial_capital)
    obs       = test_env.reset()
    done      = False
    equity    = [CFG.initial_capital]
    net_worth = CFG.initial_capital

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action    = action[0]
        risk_mgr.update(net_worth)
        action = risk_mgr.scale_action(action)
        obs, _, dones, infos = test_env.step(action[np.newaxis])
        done      = dones[0]
        net_worth = infos[0].get("net_worth", net_worth)
        equity.append(net_worth)

    test_env.close()
    equity = np.array(equity)

    # ── מדדים ─────────────────────────────────────────────────────
    returns = np.diff(equity) / (equity[:-1] + 1e-9)
    total_return = (equity[-1] - equity[0]) / equity[0]
    sharpe       = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    peak         = np.maximum.accumulate(equity)
    max_dd       = ((peak - equity) / (peak + 1e-9)).max()
    win_rate     = float(np.mean(returns > 0))
    annualized_return = _annualized_return(total_return, len(returns))
    calmar = annualized_return / (max_dd + 1e-9)

    # ── SPY benchmark ─────────────────────────────────────────────
    spy_return = 0.0
    if "SPY" in test_data and len(test_data["SPY"]) >= 2:
        spy_prices = test_data["SPY"]["close"].values
        spy_return = (spy_prices[-1] - spy_prices[0]) / spy_prices[0]

    result = {
        "window":       w,
        "train_start":  window["train_start"],
        "train_end":    window["train_end"],
        "test_start":   window["test_start"],
        "test_end":     window["test_end"],
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "max_drawdown": max_dd,
        "win_rate":     win_rate,
        "spy_return":   spy_return,
        "alpha":        total_return - spy_return,
        "equity":       equity.tolist(),
        "skipped":      False,
    }

    print(f"  Return: {total_return:+.1%} | SPY: {spy_return:+.1%} | "
          f"Alpha: {result['alpha']:+.1%} | Sharpe: {sharpe:.2f} | "
          f"Calmar: {calmar:.2f} | MaxDD: {max_dd:.1%}")
    return result


# ══════════════════════════════════════════════════════════════════
# Full walk-forward run
# ══════════════════════════════════════════════════════════════════

def run_walk_forward(
    timesteps: int = 100_000,
    use_transformer_policy: bool = False,
    model_params: dict | None = None,
) -> list[dict]:
    """מריץ את כל חלונות ה-Walk-Forward."""

    print("\n" + "═" * 55)
    print("  WALK-FORWARD EVALUATION")
    policy_label = "transformer (production)" if use_transformer_policy else "MLP (fast proxy)"
    print(f"  {CFG.wf_n_windows} windows · {timesteps:,} steps each · {policy_label}")
    print("═" * 55)

    # טעינת נתונים
    from data_manager import DataManager
    dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=CFG.data_end)
    dm.load_all(force_download=False)
    all_data = dm.get_aligned_data()

    # יצירת חלונות
    windows = generate_windows(
        start_date    = CFG.train_start,
        n_windows     = CFG.wf_n_windows,
        train_months  = CFG.wf_train_months,
        test_months   = CFG.wf_test_months,
    )

    print(f"\n  Generated {len(windows)} windows:")
    for w in windows:
        print(f"    Window {w['window']}: train {w['train_start']}→{w['train_end']} | "
              f"test {w['test_start']}→{w['test_end']}")

    # הרצת כל חלון
    results = []
    for window in windows:
        res = evaluate_window(
            window, all_data, timesteps=timesteps,
            model_params=model_params, use_transformer_policy=use_transformer_policy,
        )
        results.append(res)

    # ── סיכום ─────────────────────────────────────────────────────
    valid = [r for r in results if not r.get("skipped")]
    summary = _build_summary(valid)
    _print_summary(summary)
    _save_results(valid, summary)
    if HAS_PLOT:
        _plot_results(valid)

    return results


def _annualized_return(total_return: float, n_periods: int, periods_per_year: int = 252) -> float:
    years = max(n_periods / periods_per_year, 1e-9)
    return (1 + total_return) ** (1 / years) - 1


def _build_summary(results: list[dict]) -> dict:
    if not results:
        return {}

    returns = np.array([r["total_return"] for r in results], dtype=float)
    alphas = np.array([r["alpha"] for r in results], dtype=float)
    sharpes = np.array([r["sharpe"] for r in results], dtype=float)
    calmars = np.array([r.get("calmar", 0.0) for r in results], dtype=float)
    max_dds = np.array([r["max_drawdown"] for r in results], dtype=float)
    annualized = np.array([r.get("annualized_return", 0.0) for r in results], dtype=float)
    return {
        "windows_evaluated": len(results),
        "avg_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "avg_alpha": float(np.mean(alphas)),
        "positive_alpha_windows": int(np.sum(alphas > 0)),
        "avg_sharpe": float(np.mean(sharpes)),
        "avg_calmar": float(np.mean(calmars)),
        "avg_max_drawdown": float(np.mean(max_dds)),
        "avg_annualized_return": float(np.mean(annualized)),
        "positive_return_windows": int(np.sum(returns > 0)),
        "consistency_ratio": float(np.mean(returns > 0)),
    }


def _print_summary(summary: dict):
    if not summary:
        print("[WalkForward] No valid windows to summarize.")
        return

    print("\n" + "═" * 55)
    print("  WALK-FORWARD SUMMARY")
    print("═" * 55)
    print(f"  Windows evaluated    : {summary['windows_evaluated']}")
    print(f"  Avg Return / window  : {summary['avg_return']:>+8.1%}")
    print(f"  Median Return        : {summary['median_return']:>+8.1%}")
    print(f"  Avg Annualized Ret   : {summary['avg_annualized_return']:>+8.1%}")
    print(f"  Avg Alpha vs SPY     : {summary['avg_alpha']:>+8.1%}")
    print(f"  Positive Alpha       : {summary['positive_alpha_windows']}/{summary['windows_evaluated']}")
    print(f"  Avg Sharpe           : {summary['avg_sharpe']:>8.2f}")
    print(f"  Avg Calmar           : {summary['avg_calmar']:>8.2f}")
    print(f"  Avg Max Drawdown     : {summary['avg_max_drawdown']:>8.1%}")
    print(f"  Consistency (>0 ret) : {summary['positive_return_windows']}/{summary['windows_evaluated']}")
    print("═" * 55)

    if summary["avg_alpha"] > 0 and summary["positive_alpha_windows"] > summary["windows_evaluated"] / 2:
        print("  ✅ Model generates positive alpha consistently")
    else:
        print("  ⚠️  Inconsistent alpha — review features and data leakage")


def _save_results(results: list[dict], summary: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    save = [{k: v for k, v in r.items() if k != "equity"} for r in results]
    import json
    with open(RESULTS_DIR / "walk_forward_results.json", "w") as f:
        json.dump(save, f, indent=2, default=float)
    with open(RESULTS_DIR / "walk_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    pd.DataFrame(save).to_csv(RESULTS_DIR / "walk_forward_results.csv", index=False)
    print(f"\n[WalkForward] Results → {RESULTS_DIR}/walk_forward_results.json")
    print(f"[WalkForward] Summary → {RESULTS_DIR}/walk_forward_summary.json")


def _plot_results(results: list[dict]):
    if not results:
        return
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#0d0d0f")
    fig.suptitle("Walk-Forward Evaluation", color="white", fontsize=13)

    windows = [r["window"] for r in results]

    # Returns
    returns = [r["total_return"] * 100 for r in results]
    spy_ret = [r["spy_return"]   * 100 for r in results]
    ax = axes[0]
    ax.set_facecolor("#111113")
    ax.bar(windows, returns, color=["#34c759" if v > 0 else "#ff3b30" for v in returns],
           alpha=0.8, label="Agent")
    ax.plot(windows, spy_ret, "o--", color="#ff9f0a", linewidth=1.5, label="SPY")
    ax.axhline(0, color="#3a3a3c", linewidth=0.8)
    ax.set_title("Return per Window (%)", color="white", fontsize=10)
    ax.tick_params(colors="#636366")
    ax.legend(facecolor="#1a1a1c", labelcolor="white", fontsize=8)
    ax.grid(True, alpha=0.15, color="#3a3a3c")

    # Alpha
    alphas = [r["alpha"] * 100 for r in results]
    ax = axes[1]
    ax.set_facecolor("#111113")
    ax.bar(windows, alphas, color=["#34c759" if v > 0 else "#ff3b30" for v in alphas], alpha=0.8)
    ax.axhline(0, color="#3a3a3c", linewidth=0.8)
    ax.set_title("Alpha vs SPY (%)", color="white", fontsize=10)
    ax.tick_params(colors="#636366")
    ax.grid(True, alpha=0.15, color="#3a3a3c")

    # Sharpe
    sharpes = [r["sharpe"] for r in results]
    ax = axes[2]
    ax.set_facecolor("#111113")
    ax.bar(windows, sharpes, color="#0a84ff", alpha=0.8)
    ax.axhline(1.0, color="#ff9f0a", linewidth=1, linestyle="--", label="Sharpe=1")
    ax.set_title("Sharpe Ratio", color="white", fontsize=10)
    ax.tick_params(colors="#636366")
    ax.legend(facecolor="#1a1a1c", labelcolor="white", fontsize=8)
    ax.grid(True, alpha=0.15, color="#3a3a3c")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a3c")

    out = PLOTS_DIR / "walk_forward.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0f")
    plt.close()
    print(f"[WalkForward] Chart → {out}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Walk-Forward Evaluation")
    p.add_argument("--fast", action="store_true",
                   help="50K steps per window (quick test)")
    p.add_argument("--steps", type=int, default=None,
                   help="Custom timesteps per window")
    p.add_argument("--production-policy", action="store_true",
                   help="Use the transformer policy + tuned hyperparameters from "
                        "models/training_meta.pkl (matches the deployed model's "
                        "architecture) instead of the fast plain-MLP proxy")
    return p.parse_args()


def _load_production_params() -> dict | None:
    meta_path = os.path.join(CFG.model_dir, "training_meta.pkl")
    if not os.path.exists(meta_path):
        print(f"[WalkForward] No {meta_path} found — using default PPO hyperparameters.")
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    best_params = meta.get("best_params")
    if best_params:
        print(f"[WalkForward] Loaded tuned hyperparameters from {meta_path}: {best_params}")
    return best_params or None


if __name__ == "__main__":
    args    = parse_args()
    steps   = args.steps or (CFG.trial_timesteps if args.fast else CFG.timesteps // 4)
    params  = _load_production_params() if args.production_policy else None
    run_walk_forward(timesteps=steps, use_transformer_policy=args.production_policy, model_params=params)
