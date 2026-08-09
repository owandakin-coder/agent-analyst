"""
main.py
=======
נקודת כניסה ראשית למערכת המסחר האוטונומית.

⚠️  DISCLAIMER
    מערכת זו מיועדת לצרכי מחקר ולמידה בלבד.
    שימוש בכסף אמיתי (--mode live) הוא באחריות בלעדית של המשתמש.
    לעולם אל תגדיר auto_approve=True אלא אם הבנת לחלוטין את הסיכון.

שימוש:
    python main.py --mode download          # הורד נתונים
    python main.py --mode train             # אמן מודל
    python main.py --mode simulate          # Paper Backtest (ללא API)
    python main.py --mode full              # כל השלבים ברצף
    python main.py --mode live_stub         # Live Stub (מדומה, ללא API)
    python main.py --mode live_paper        # Live על חשבון Paper Alpaca
    python main.py --mode live              # Live על חשבון אמיתי (DANGER)
    python main.py --mode dashboard         # הפעל דשבורד Streamlit
"""

import os
import sys
import pickle
import argparse
import warnings
from datetime import datetime, timezone

# ── UTF-8 לטרמינל Windows ──────────────────────────────────────────────────
if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and getattr(sys.stderr, "encoding", None) and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging

import numpy as np
import pandas as pd
from logging_setup import configure_logging

warnings.filterwarnings("ignore")

# ── Centralised logging ────────────────────────────────────────────────────
# Configure once here so all module loggers (BrokerAPI, LiveTrader, etc.)
# inherit the same format and handlers. Modules must NOT call basicConfig().
configure_logging()

# ─── ייבוא מודולים מקומיים ────────────────────────────────────────────────
from config_loader       import CFG                          # ← config.yaml
from data_manager        import DataManager
from trading_env         import TradingEnvironment
from training_pipeline   import TrainingPipeline
from execution_simulator import ExecutionSimulator
from broker_api          import BrokerAPIStub, AlpacaBrokerAPI
from control_plane       import can_trade, load_control_state
from risk_manager        import RiskManager
from live_trader         import LiveTrader

# ─── קבועים גלובליים (מ-config.yaml) ─────────────────────────────────────
TICKERS         = CFG.tickers
START_DATE      = CFG.data_start
END_DATE        = CFG.data_end
INITIAL_CAPITAL = CFG.initial_capital
RESULTS_DIR     = CFG.results_dir
MODEL_DIR       = CFG.model_dir
TEST_START      = CFG.test_start
TEST_END        = CFG.test_end

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)


def _fail_closed_control_enabled() -> bool:
    return os.getenv("ATZMA_FAIL_CLOSED_CONTROL", "1").strip().lower() not in {"0", "false", "no"}


# ══════════════════════════════════════════════════════════════════════════════
# שלב 1: הורדת נתונים
# ══════════════════════════════════════════════════════════════════════════════

def step_download(force: bool = False) -> dict[str, pd.DataFrame]:
    """מוריד ומעבד נתונים מ-yfinance."""
    print_banner("Step 1: Download Data")
    dm = DataManager(tickers=TICKERS, start=START_DATE, end=END_DATE)
    dm.load_all(force_download=force)
    aligned = dm.get_aligned_data()

    for ticker, df in aligned.items():
        print(f"  {ticker}: {len(df)} days | {len(df.columns)} features | "
              f"{df.index[0].date()} - {df.index[-1].date()}")

    print(f"\n  Features (sample): {list(df.columns)[:8]} ...")
    return aligned


# ══════════════════════════════════════════════════════════════════════════════
# שלב 2: אימון
# ══════════════════════════════════════════════════════════════════════════════

def step_train(aligned_data: dict[str, pd.DataFrame], n_optuna_trials: int = 15):
    """מאמן PPO עם Walk-Forward + Optuna."""
    print_banner("Step 2: Train (Walk-Forward + Optuna)")
    pipeline = TrainingPipeline(aligned_data, n_optuna_trials=n_optuna_trials)
    model    = pipeline.run()

    meta = {
        "best_params":  pipeline.best_params,
        "best_validation_summary": pipeline.best_validation_summary,
        "tickers":      TICKERS,
        "train_period": (CFG.train_start, CFG.train_end),
        "val_period":   (CFG.val_start, CFG.val_end),
        "test_period":  (CFG.test_start, CFG.test_end),
        "trained_at":   datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "training_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return model, pipeline


# ══════════════════════════════════════════════════════════════════════════════
# שלב 3: Backtest Simulation (ללא API)
# ══════════════════════════════════════════════════════════════════════════════

def step_simulate(model, aligned_data: dict[str, pd.DataFrame], vec_norm):
    """מריץ Paper Backtest על נתוני טסט ושומר תוצאות."""
    print_banner("Step 3: Paper Backtest Simulation")

    test_data = {
        ticker: df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
        for ticker, df in aligned_data.items()
    }

    for ticker, df in test_data.items():
        if len(df) < 60:
            raise ValueError(f"Insufficient test data for {ticker}: {len(df)} days")

    sim     = ExecutionSimulator(model, test_data, vec_norm, INITIAL_CAPITAL)
    metrics = sim.run()
    sim.plot_all()
    _save_dashboard_data(sim, metrics)
    return metrics


def _save_dashboard_data(sim: ExecutionSimulator, metrics: dict):
    with open(os.path.join(RESULTS_DIR, "metrics.pkl"),      "wb") as f:
        pickle.dump(metrics, f)
    with open(os.path.join(RESULTS_DIR, "equity_data.pkl"),  "wb") as f:
        pickle.dump((sim.equity_curve, sim.dates), f)
    with open(os.path.join(RESULTS_DIR, "actions_data.pkl"), "wb") as f:
        pickle.dump((sim.actions_history, sim.tickers, sim.dates), f)
    print(f"[Main] Dashboard data saved to '{RESULTS_DIR}'.")


# ══════════════════════════════════════════════════════════════════════════════
# שלב 4: Live Stub (מדומה – ללא API)
# ══════════════════════════════════════════════════════════════════════════════

def step_live_stub(model, aligned_data: dict[str, pd.DataFrame]):
    """הדגמת Live Mode עם BrokerAPIStub – ללא חיבור לשום ברוקר."""
    print_banner("Step 4: Live Mode Stub (SIMULATED – no real orders)")

    broker   = BrokerAPIStub(account_id="RESEARCH_PAPER_001")
    broker.set_cash(INITIAL_CAPITAL)
    risk_mgr = RiskManager(INITIAL_CAPITAL)

    last_prices = {t: float(df["close"].iloc[-1]) for t, df in aligned_data.items()}
    print(f"  Last prices: {last_prices}\n")

    for day in range(1, 4):
        print(f"--- Day {day} ---")
        np.random.seed(day)
        action = np.random.uniform(-0.5, 0.5, len(TICKERS))
        risk_mgr.update(INITIAL_CAPITAL * (1 - 0.03 * day))
        action = risk_mgr.scale_action(action)

        for i, ticker in enumerate(TICKERS):
            act   = float(action[i])
            price = last_prices[ticker] * (1 + np.random.uniform(-0.01, 0.01))
            if act > 0.05:
                broker.buy(ticker, max(1, int(1000 * act)), round(price, 2))
            elif act < -0.05:
                held = broker.positions.get(ticker, 0.0)
                shares = max(0, int(held * abs(act)))
                if shares > 0:
                    broker.sell(ticker, shares, round(price, 2))
            else:
                broker.hold(ticker)
        print()

    summary = broker.get_account_summary()
    print(f"  Account summary (stub): {summary}")
    print("  Order log -> paper_orders.log")


# ══════════════════════════════════════════════════════════════════════════════
# שלב 5: Live Paper – Alpaca Paper Account
# ══════════════════════════════════════════════════════════════════════════════

def step_live_paper(model, vec_norm, auto_approve: bool = False, ensemble=False):
    """
    מריץ לולאת מסחר חיה מול חשבון Paper של Alpaca.
    כסף וירטואלי בלבד – אך API אמיתי.

    Parameters
    ----------
    auto_approve : bool
        False (ברירת מחדל) = כל פקודה תחכה לאישור.
        True               = פקודות אוטומטיות.
    """
    print_banner("Step 5: Live Paper Trading (Alpaca Paper API)")
    print("  Mode: PAPER (virtual money, real API)\n")

    # ── בניית ברוקר ────────────────────────────────────────────────────────
    try:
        control_state = load_control_state()
        trade_allowed, block_reason = can_trade(control_state)
    except Exception as exc:
        print(f"  [WARN] Control state unavailable: {exc}")
        trade_allowed, block_reason = (False, "control_plane_unavailable") if _fail_closed_control_enabled() else (True, None)

    if not trade_allowed:
        status = "EMERGENCY STOP" if block_reason == "emergency_stop" else "PAUSED"
        if block_reason == "control_plane_unavailable":
            status = "CONTROL PLANE UNAVAILABLE"
        print(f"  Trading skipped by control plane: {status}.")
        return

    try:
        broker = AlpacaBrokerAPI(paper=True, auto_approve=auto_approve)
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}")
        print("  Create a .env file from .env.example and fill in your credentials.")
        sys.exit(1)

    # ── אחזור שווי התחלתי ──────────────────────────────────────────────────
    try:
        account_info = broker.get_account()
        initial_capital = account_info["equity"]
        print(f"  Alpaca account equity: ${initial_capital:,.2f}")
    except Exception:
        initial_capital = INITIAL_CAPITAL
        print(f"  Could not fetch account equity. Using default: ${initial_capital:,.0f}")

    # ── DataManager לשליפת נתונים עדכניים ──────────────────────────────────
    dm       = DataManager(tickers=TICKERS)
    risk_mgr = RiskManager(initial_capital)

    trader = LiveTrader(
        model          = model,
        broker         = broker,
        data_manager   = dm,
        risk_manager   = risk_mgr,
        vec_norm       = vec_norm,
        tickers        = TICKERS,
        initial_capital = initial_capital,
    )

    print("  Starting live paper trading loop. Press Ctrl+C to stop.\n")
    trader.run_loop(poll_seconds=60)


# ══════════════════════════════════════════════════════════════════════════════
# שלב 6: Live Real – Alpaca Live Account (DANGER)
# ══════════════════════════════════════════════════════════════════════════════

def step_live_once(model, vec_norm, auto_approve: bool = True):
    """
    Runs ONE trading decision and exits.
    Designed for GitHub Actions / scheduled cloud runs.
    No infinite loop — load model → decide → submit orders → done.
    """
    print_banner("Live Once – Single Decision Cycle (GitHub Actions / Cloud)")
    print("  Mode: PAPER (virtual money, real API)\n")

    try:
        broker = AlpacaBrokerAPI(paper=True, auto_approve=auto_approve)
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    try:
        initial_capital = broker.get_account().get("equity", INITIAL_CAPITAL)
        print(f"  Account equity: ${initial_capital:,.2f}")
    except Exception:
        initial_capital = INITIAL_CAPITAL

    dm       = DataManager(tickers=TICKERS)
    risk_mgr = RiskManager(initial_capital)

    trader = LiveTrader(
        model           = model,
        broker          = broker,
        data_manager    = dm,
        risk_manager    = risk_mgr,
        vec_norm        = vec_norm,
        tickers         = TICKERS,
        initial_capital = initial_capital,
    )

    if not broker.is_market_open():
        print("  Market is CLOSED right now. No orders placed.")
        trader._telegram(
            f"⏸ *Agent skipped* — market closed at run time.\n"
            f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        sys.exit(0)

    print("  Market is OPEN. Running decision cycle...")
    result = trader.run_once()
    print("  Decision cycle complete. Exiting.")
    return result


def step_live_real(model, vec_norm, auto_approve: bool):
    """
    מריץ לולאת מסחר חיה מול חשבון Live של Alpaca.

    ⚠️  REAL MONEY – כל פקודה משפיעה על כסף אמיתי.
        המשתמש אחראי באופן בלעדי לכל הפסד.

    auto_approve=True נדרש להפעלה אוטומטית.
    """
    print_banner("Step 6: LIVE TRADING (REAL MONEY)")
    print("!" * 60)
    print("  WARNING: This mode trades REAL MONEY.")
    print("  All losses are your sole responsibility.")
    print("!" * 60 + "\n")

    # ── חובה auto_approve=True ל-live ────────────────────────────────────
    if not auto_approve:
        print(
            "[LIVE] auto_approve=False detected.\n"
            "  In --mode live you must explicitly pass --auto-approve\n"
            "  to confirm you understand the risk of automated trading.\n"
            "  Exiting."
        )
        sys.exit(1)

    # אישור כפול מהמשתמש
    print("  Type 'I UNDERSTAND' to proceed with LIVE trading: ", end="")
    try:
        confirm = input().strip()
    except EOFError:
        confirm = ""
    if confirm != "I UNDERSTAND":
        print("  Aborted.")
        sys.exit(0)

    try:
        broker = AlpacaBrokerAPI(paper=False, auto_approve=True)
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    try:
        account_info    = broker.get_account()
        initial_capital = account_info["equity"]
        print(f"  Live account equity: ${initial_capital:,.2f}")
    except Exception:
        initial_capital = INITIAL_CAPITAL

    dm       = DataManager(tickers=TICKERS)
    risk_mgr = RiskManager(initial_capital)

    trader = LiveTrader(
        model          = model,
        broker         = broker,
        data_manager   = dm,
        risk_manager   = risk_mgr,
        vec_norm       = vec_norm,
        tickers        = TICKERS,
        initial_capital = initial_capital,
    )

    trader.run_loop(poll_seconds=60)


# ══════════════════════════════════════════════════════════════════════════════
# טעינת מודל מאומן
# ══════════════════════════════════════════════════════════════════════════════

def load_trained_model_and_norm(dummy_data: dict | None = None):
    """
    טוען מודל PPO ו-VecNormalize קיימים.
    אם dummy_data=None, יוצר סביבה מינימלית לטעינת ה-VecNormalize.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_path = os.path.join(MODEL_DIR, "final_model.zip")
    norm_path  = os.path.join(MODEL_DIR, "vec_normalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: python main.py --mode train"
        )
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"VecNormalize not found: {norm_path}\n"
            "Run: python main.py --mode train"
        )

    print("[Main] Loading trained model ...")
    model = PPO.load(model_path)

    # סביבה זמנית לטעינת נרמול
    if dummy_data is None:
        dm = DataManager(tickers=TICKERS, start="2022-01-01", end="2024-12-31")
        dm.load_all(force_download=False)
        dummy_data = dm.get_aligned_data()

    dummy_env = DummyVecEnv([lambda: TradingEnvironment(dummy_data)])
    vec_norm  = VecNormalize.load(norm_path, dummy_env)
    vec_norm.training    = False
    vec_norm.norm_reward = False

    return model, vec_norm


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def step_dashboard():
    print_banner("Streamlit Dashboard")
    print("  Launch: streamlit run dashboard.py")
    os.system("streamlit run dashboard.py")


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_disclaimer():
    print("""
+--------------------------------------------------------------+
|   Agent Analyst - Research Trading System                    |
|                                                              |
|  [!] For research and educational purposes only.             |
|  [!] Real-money trading (--mode live) is the user's sole    |
|      responsibility. Use at your own risk.                   |
+--------------------------------------------------------------+
    """)


# ══════════════════════════════════════════════════════════════════════════════
# argparse
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Agent Analyst - Autonomous Trading Research System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "download", "train", "train_ensemble", "simulate", "full",
            "live_stub", "live_paper", "live_once", "live_ensemble", "live", "dashboard",
            "benchmark", "walk_forward", "leakage_check",
        ],
        default="full",
        help=(
            "download       - Download market data\n"
            "train          - Train PPO model\n"
            "simulate       - Backtest on test data (no API)\n"
            "full           - download + train + simulate\n"
            "benchmark      - Compare vs SPY, Equal-Weight, Momentum\n"
            "walk_forward   - Multi-window Walk-Forward evaluation\n"
            "leakage_check  - Check for data leakage issues\n"
            "live_stub      - Demo with stub broker (no API)\n"
            "live_paper     - Live loop on Alpaca Paper account\n"
            "live_once      - ONE decision then exit (for GitHub Actions / cron)\n"
            "live           - Live loop on Alpaca REAL account (DANGER)\n"
            "dashboard      - Launch Streamlit dashboard"
        ),
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Force re-download (ignore cache)",
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=10,
        help="Number of Optuna trials (default: 10)",
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help=(
            "Execute orders without manual approval.\n"
            "Required for --mode live. USE WITH EXTREME CAUTION."
        ),
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_disclaimer()
    args = parse_args()

    aligned_data = None
    model        = None
    vec_norm     = None

    # ── Download ──────────────────────────────────────────────────────────────
    if args.mode in ("download", "train", "train_ensemble", "simulate", "full", "live_stub"):
        aligned_data = step_download(force=args.force_download)

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.mode in ("train", "full"):
        model, pipeline = step_train(aligned_data, n_optuna_trials=args.optuna_trials)
        vec_norm = pipeline.vec_norm

    # ── Train Ensemble ────────────────────────────────────────────────────────
    if args.mode == "train_ensemble":
        meta_path = os.path.join(MODEL_DIR, "training_meta.pkl")
        if os.path.exists(meta_path):
            # Reuse best params from previous run — skip Optuna + base training
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            best_params = meta.get("best_params", {})
            if best_params:
                print_banner("Train Ensemble (reusing saved best params)")
                print(f"  Loaded best_params: {best_params}")
                pipeline = TrainingPipeline(aligned_data, n_optuna_trials=0)
                pipeline.best_params = best_params
                pipeline.train_ensemble()
                print_banner("Ensemble training complete")
                print("  Models saved: models/ensemble_0.zip, ensemble_1.zip, ensemble_2.zip")
                return
        # No saved params: run full Optuna + base first, then ensemble
        print_banner("Train Ensemble (running Optuna first — no saved params found)")
        _, pipeline = step_train(aligned_data, n_optuna_trials=args.optuna_trials)
        pipeline.train_ensemble()
        print_banner("Ensemble training complete")
        print("  Models saved: models/ensemble_0.zip, ensemble_1.zip, ensemble_2.zip")
        return

    # ── Simulate (Backtest) ───────────────────────────────────────────────────
    if args.mode == "simulate":
        try:
            model, vec_norm = load_trained_model_and_norm(aligned_data)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        step_simulate(model, aligned_data, vec_norm)

    elif args.mode == "full":
        step_simulate(model, aligned_data, vec_norm)

    # ── Live Stub ─────────────────────────────────────────────────────────────
    if args.mode == "live_stub":
        try:
            model, _ = load_trained_model_and_norm(aligned_data)
        except FileNotFoundError:
            print("[WARN] No trained model found. Using random actions for demo.")
            model = None
        step_live_stub(model, aligned_data)

    # ── Live Ensemble ─────────────────────────────────────────────────────────
    if args.mode == "live_ensemble":
        from ensemble_agent import load_ensemble
        ensemble = load_ensemble()
        step_live_paper(ensemble, vec_norm=None, auto_approve=args.auto_approve)

    # ── Live Once (GitHub Actions / cron) ────────────────────────────────────
    if args.mode == "live_once":
        try:
            model, vec_norm = load_trained_model_and_norm()
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        step_live_once(model, vec_norm, auto_approve=args.auto_approve)

    # ── Live Paper (Alpaca Paper API) ─────────────────────────────────────────
    if args.mode == "live_paper":
        try:
            model, vec_norm = load_trained_model_and_norm()
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        step_live_paper(model, vec_norm, auto_approve=args.auto_approve)

    # ── Live Real (Alpaca Live API) – DANGER ──────────────────────────────────
    if args.mode == "live":
        try:
            model, vec_norm = load_trained_model_and_norm()
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        step_live_real(model, vec_norm, auto_approve=args.auto_approve)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if args.mode == "benchmark":
        from benchmark import run_benchmark
        run_benchmark()
        return

    # ── Walk-Forward Evaluation ───────────────────────────────────────────────
    if args.mode == "walk_forward":
        from walk_forward_eval import run_walk_forward
        steps = args.optuna_trials * 10_000   # reuse --optuna-trials as step multiplier
        run_walk_forward(timesteps=max(steps, 50_000))
        return

    # ── Leakage Check ─────────────────────────────────────────────────────────
    if args.mode == "leakage_check":
        from leakage_check import run_all_checks
        run_all_checks(verbose=True)
        return

    # ── Dashboard ─────────────────────────────────────────────────────────────
    if args.mode == "dashboard":
        step_dashboard()
        return  # dashboard blocks; skip footer

    # ── footer ────────────────────────────────────────────────────────────────
    if args.mode not in ("live_paper", "live", "live_once"):
        print_banner("Done")
        print("  Dashboard : streamlit run dashboard.py")
        print("  Order log : paper_orders.log")
        print("  Charts    : results/")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Google Colab helper
# ══════════════════════════════════════════════════════════════════════════════

def colab_setup():
    """הפעל ב-Colab: from main import colab_setup; colab_setup()"""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "gymnasium", "stable-baselines3", "yfinance", "optuna",
        "seaborn", "plotly", "streamlit", "scipy", "alpaca-py", "python-dotenv",
    ], check=True)
    print("[Colab] All packages installed.")


if __name__ == "__main__":
    main()
