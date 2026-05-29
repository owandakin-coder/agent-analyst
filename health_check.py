"""
health_check.py
===============
בדיקת תקינות מהירה של כל מרכיבי המערכת.
מריץ לפני כל deploy או debugging session.

שימוש:
    python health_check.py           # בדיקה מלאה
    python health_check.py --fast    # ללא בדיקת חיבור Alpaca

יציאה עם exit code 0 = הכל תקין, 1 = יש בעיות.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


OK   = "✅"
WARN = "⚠️ "
FAIL = "❌"


def check(label: str, ok: bool, detail: str = ""):
    sym = OK if ok else FAIL
    msg = f"  {sym} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok


def warn(label: str, detail: str = ""):
    msg = f"  {WARN} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ══════════════════════════════════════════════════════════════════
# 1. Config
# ══════════════════════════════════════════════════════════════════

def check_config() -> bool:
    print("\n[1] Config")
    try:
        from config_loader import CFG
        ok = True
        ok &= check("config.yaml loaded", True)
        ok &= check("Tickers defined",    len(CFG.tickers) > 0,
                    f"{len(CFG.tickers)} tickers")
        ok &= check("Benchmark in tickers", CFG.benchmark in CFG.tickers,
                    CFG.benchmark)
        ok &= check("Capital > 0",         CFG.initial_capital > 0,
                    f"${CFG.initial_capital:,.0f}")
        ok &= check("Kelly fraction valid", 0 < CFG.kelly_fraction <= 1.0,
                    f"{CFG.kelly_fraction}×")
        ok &= check("DD halt > DD reduce",  CFG.drawdown_halt > CFG.drawdown_reduce,
                    f"{CFG.drawdown_reduce:.0%} / {CFG.drawdown_halt:.0%}")
        ok &= check("Train before test",
                    CFG.train_end < CFG.test_start,
                    f"{CFG.train_end} < {CFG.test_start}")
        return ok
    except Exception as e:
        check("config.yaml", False, str(e))
        return False


# ══════════════════════════════════════════════════════════════════
# 2. Dependencies
# ══════════════════════════════════════════════════════════════════

def check_dependencies() -> bool:
    print("\n[2] Dependencies")
    required = {
        "numpy":         "numpy",
        "pandas":        "pandas",
        "gymnasium":     "gymnasium",
        "stable_baselines3": "stable_baselines3",
        "optuna":        "optuna",
        "yfinance":      "yfinance",
        "yaml":          "pyyaml",
        "dotenv":        "python-dotenv",
        "scipy":         "scipy",
        "matplotlib":    "matplotlib",
    }
    optional = {
        "alpaca":        "alpaca-py",
        "telegram":      "python-telegram-bot",
    }
    ok = True
    for module, pkg in required.items():
        try:
            __import__(module)
            check(pkg, True)
        except ImportError:
            check(pkg, False, f"pip install {pkg}")
            ok = False

    for module, pkg in optional.items():
        try:
            __import__(module)
            check(f"{pkg} (optional)", True)
        except ImportError:
            warn(f"{pkg} (optional)", f"pip install {pkg}")

    return ok


# ══════════════════════════════════════════════════════════════════
# 3. Files & Directories
# ══════════════════════════════════════════════════════════════════

def check_files() -> bool:
    print("\n[3] Files & Directories")
    from config_loader import CFG
    ok = True

    # קבצי קוד
    required_files = [
        "main.py", "trading_env.py", "training_pipeline.py",
        "execution_simulator.py", "data_manager.py", "risk_manager.py",
        "broker_api.py", "live_trader.py", "config.yaml", "config_loader.py",
        ".env",
    ]
    for f in required_files:
        exists = Path(f).exists()
        if f == ".env" and not exists:
            warn(".env missing", "copy .env.example → .env and fill credentials")
        else:
            ok &= check(f, exists)

    # תיקיות
    for d in [CFG.model_dir, CFG.results_dir, CFG.logs_dir]:
        Path(d).mkdir(exist_ok=True)
        check(f"dir: {d}/", True)

    # מודל מאומן
    model_path = Path(CFG.model_dir) / "final_model.zip"
    norm_path  = Path(CFG.model_dir) / "vec_normalize.pkl"
    if model_path.exists() and norm_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        check("Trained model exists", True, f"{size_mb:.1f} MB")
    else:
        warn("No trained model", "run: python main.py --mode train")

    return ok


# ══════════════════════════════════════════════════════════════════
# 4. Environment Variables
# ══════════════════════════════════════════════════════════════════

def check_env() -> bool:
    print("\n[4] Environment Variables")
    from dotenv import load_dotenv
    load_dotenv()

    ok = True
    vars_required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    vars_optional = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ALPACA_BASE_URL"]

    for var in vars_required:
        val = os.getenv(var, "")
        if val:
            check(var, True, f"{'*' * 6}{val[-4:]}")
        else:
            check(var, False, "not set in .env")
            ok = False

    for var in vars_optional:
        val = os.getenv(var, "")
        if val:
            check(f"{var} (optional)", True)
        else:
            warn(f"{var} (optional)", "not set")

    return ok


# ══════════════════════════════════════════════════════════════════
# 5. Alpaca Connection
# ══════════════════════════════════════════════════════════════════

def check_alpaca() -> bool:
    print("\n[5] Alpaca API Connection")
    try:
        from broker_api import AlpacaBrokerAPI
        broker = AlpacaBrokerAPI(paper=True, auto_approve=False)
        acc    = broker.get_account()
        equity = acc.get("equity", 0)
        check("Alpaca Paper connected", True, f"equity=${equity:,.0f}")

        is_open = broker.is_market_open()
        if is_open:
            check("Market status", True, "OPEN")
        else:
            next_open = broker.next_market_open()
            warn("Market status", f"CLOSED · next open: {next_open}")

        return True

    except EnvironmentError as e:
        warn("Alpaca skipped", "credentials not set")
        return True   # אל תכשיל אם אין credentials
    except Exception as e:
        check("Alpaca connection", False, str(e))
        return False


# ══════════════════════════════════════════════════════════════════
# 6. Model Loading
# ══════════════════════════════════════════════════════════════════

def check_model() -> bool:
    print("\n[6] Model Loading")
    from config_loader import CFG
    model_path = Path(CFG.model_dir) / "final_model.zip"
    norm_path  = Path(CFG.model_dir) / "vec_normalize.pkl"

    if not model_path.exists():
        warn("Model not found", "skipping load test")
        return True

    try:
        t0 = time.time()
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))
        elapsed = time.time() - t0
        check("Model loads", True, f"{elapsed:.1f}s")

        # בדיקת observation shape
        obs_shape = model.observation_space.shape
        check("Observation space defined", obs_shape is not None, str(obs_shape))

        # בדיקת action space
        act_shape = model.action_space.shape
        check("Action space defined", act_shape is not None, str(act_shape))

        return True
    except Exception as e:
        check("Model load", False, str(e))
        return False


# ══════════════════════════════════════════════════════════════════
# 7. GitHub Actions
# ══════════════════════════════════════════════════════════════════

def check_github_actions() -> bool:
    print("\n[7] GitHub Actions")
    try:
        import urllib.request, json
        url = "https://api.github.com/repos/owandakin-coder/agent-analyst/actions/runs?per_page=5"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        runs = data.get("workflow_runs", [])
        if runs:
            last = runs[0]
            status = last.get("conclusion") or last.get("status", "?")
            name   = last["name"]
            date   = last["created_at"][:10]
            ok = last.get("conclusion") in ("success", None)
            check(f"Last run: {name}", True if ok else False,
                  f"{date} · {status}")
        else:
            warn("No workflow runs found")
        return True
    except Exception as e:
        warn("GitHub Actions check", f"skipped ({e})")
        return True


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main(fast: bool = False) -> int:
    print("=" * 55)
    print("  ATZMA — System Health Check")
    print("=" * 55)

    results = []
    results.append(check_config())
    results.append(check_dependencies())
    results.append(check_files())
    results.append(check_env())

    if not fast:
        results.append(check_alpaca())
        results.append(check_model())
        results.append(check_github_actions())

    passed = sum(results)
    total  = len(results)

    print(f"\n{'=' * 55}")
    if passed == total:
        print(f"  ✅ All {total} checks passed — system is healthy")
    else:
        print(f"  ❌ {total - passed}/{total} checks failed — fix issues above")
    print("=" * 55 + "\n")

    return 0 if passed == total else 1


def parse_args():
    p = argparse.ArgumentParser(description="ATZMA Health Check")
    p.add_argument("--fast", action="store_true",
                   help="Skip Alpaca + model loading (no network)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(fast=args.fast))
