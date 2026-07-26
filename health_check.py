"""
Fast system health checks for ATZMA.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def check(label: str, ok: bool, detail: str = "") -> bool:
    prefix = OK if ok else FAIL
    message = f"  {prefix} {label}"
    if detail:
        message += f" ({detail})"
    print(message)
    return ok


def warn(label: str, detail: str = ""):
    message = f"  {WARN} {label}"
    if detail:
        message += f" ({detail})"
    print(message)


def remote_api_base() -> str:
    return os.getenv("ATZMA_REMOTE_API_BASE", "https://sofowpweliticltlbxrj.supabase.co/functions/v1/api").rstrip("/")


def classify_broker_error(exc: Exception) -> tuple[str, bool]:
    text = str(exc).strip()
    lower = text.lower()
    if "unauthorized" in lower or "forbidden" in lower or "403" in lower or "401" in lower:
        return "broker credentials rejected by Alpaca", False
    if "timeout" in lower:
        return "broker request timed out", False
    if "missing environment variables" in lower:
        return "broker credentials missing locally", True
    return text or exc.__class__.__name__, False


def check_config() -> bool:
    print("\n[1] Config")
    try:
        from config_loader import CFG
    except Exception as exc:
        return check("config.yaml", False, str(exc))

    ok = True
    ok &= check("config.yaml loaded", True)
    ok &= check("Tickers defined", len(CFG.tickers) > 0, f"{len(CFG.tickers)} tickers")
    ok &= check("Benchmark in universe", CFG.benchmark in CFG.tickers, CFG.benchmark)
    ok &= check("Capital positive", CFG.initial_capital > 0, f"${CFG.initial_capital:,.0f}")
    ok &= check("Train before validation", CFG.train_end < CFG.val_start)
    ok &= check("Validation before test", CFG.val_end < CFG.test_start)
    ok &= check("Drawdown hierarchy", CFG.drawdown_halt > CFG.drawdown_reduce)
    ok &= check("Trade minimum positive", CFG.min_trade_value > 0, f"${CFG.min_trade_value:,.0f}")
    ok &= check("Live thresholds ordered", CFG.live_sell_threshold < 0 < CFG.live_buy_threshold)
    return ok


def check_dependencies() -> bool:
    print("\n[2] Dependencies")
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "gymnasium": "gymnasium",
        "stable_baselines3": "stable-baselines3",
        "optuna": "optuna",
        "yfinance": "yfinance",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
    }
    optional = {
        "alpaca": "alpaca-py",
    }
    ok = True
    for module, package in required.items():
        try:
            found = importlib.util.find_spec(module) is not None
            if not found:
                raise ImportError(module)
            check(package, True)
        except ImportError:
            ok &= check(package, False, f"pip install {package}")
    for module, package in optional.items():
        try:
            found = importlib.util.find_spec(module) is not None
            if not found:
                raise ImportError(module)
            check(f"{package} (optional)", True)
        except ImportError:
            warn(f"{package} (optional)", f"pip install {package}")
    return ok


def check_files() -> bool:
    print("\n[3] Files")
    from config_loader import CFG

    required = [
        "README.md",
        "main.py",
        "broker_api.py",
        "live_trader.py",
        "trading_env.py",
        "training_pipeline.py",
        "config.yaml",
        "config_loader.py",
        "worker_entry.py",
        "render.yaml",
        "runtime.txt",
    ]
    ok = True
    for filename in required:
        ok &= check(filename, Path(filename).exists())

    for directory in [CFG.model_dir, CFG.results_dir, CFG.logs_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        check(f"dir: {directory}/", True)

    submitted_orders = Path(CFG.broker_submitted_orders_file)
    if submitted_orders.exists():
        check("submitted_orders.json", True, str(submitted_orders))
    else:
        warn("submitted_orders.json missing", "will be created on first order")

    app_index = Path("dashboard_app/index.html")
    root_index = Path("index.html")
    if app_index.exists() and root_index.exists():
        sync_ok = app_index.read_bytes() == root_index.read_bytes()
        ok &= check("frontend build synced", sync_ok, "index.html == dashboard_app/index.html" if sync_ok else "root and dashboard_app HTML differ")

    return ok


def check_env() -> bool:
    print("\n[4] Environment")
    from dotenv import load_dotenv

    load_dotenv(".env")
    ok = True
    for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
        value = os.getenv(key, "")
        ok &= check(key, bool(value), "set" if value else "missing")
    for key in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ALPACA_BASE_URL"]:
        value = os.getenv(key, "")
        if value:
            check(f"{key} (optional)", True)
        else:
            warn(f"{key} (optional)", "not set")
    for key in ["SUPABASE_ACCESS_TOKEN", "ATZMA_WORKER_SHARED_TOKEN"]:
        value = os.getenv(key, "")
        if value:
            check(f"{key} (launch)", True)
        else:
            warn(f"{key} (launch)", "not set")
    broker_key = os.getenv("ATZMA_BROKER_CREDENTIAL_KEY", "")
    if broker_key:
        check("ATZMA_BROKER_CREDENTIAL_KEY (local optional)", True)
    else:
        warn("ATZMA_BROKER_CREDENTIAL_KEY (local optional)", "not set locally; remote edge secret must be enabled")
    return ok


def check_alpaca() -> bool:
    print("\n[5] Broker")
    try:
        from broker_api import AlpacaBrokerAPI

        broker = AlpacaBrokerAPI(paper=True, auto_approve=False)
        snapshot = broker.reconcile_account_state()
        ok = True
        ok &= check("Broker snapshot", isinstance(snapshot, dict))
        ok &= check("Snapshot has cash", "cash" in snapshot, f"${snapshot.get('cash', 0):,.0f}")
        ok &= check("Snapshot has positions", "positions" in snapshot)
        return ok
    except EnvironmentError:
        warn("Broker check skipped", "credentials not set")
        return True
    except Exception as exc:
        detail, is_warning = classify_broker_error(exc)
        if is_warning:
            warn("Broker check skipped", detail)
            return True
        return check("Broker connection", False, detail)


def check_model() -> bool:
    print("\n[6] Model")
    from config_loader import CFG

    model_path = Path(CFG.model_dir) / "final_model.zip"
    if not model_path.exists():
        warn("Model not found", "run: python main.py --mode train")
        return True

    try:
        from stable_baselines3 import PPO

        started = time.time()
        model = PPO.load(str(model_path))
        elapsed = time.time() - started
        ok = True
        ok &= check("Model loads", True, f"{elapsed:.1f}s")
        ok &= check("Observation space", model.observation_space.shape is not None, str(model.observation_space.shape))
        ok &= check("Action space", model.action_space.shape is not None, str(model.action_space.shape))
        return ok
    except Exception as exc:
        return check("Model load", False, str(exc))


def check_worker_runtime() -> bool:
    print("\n[7] Worker Runtime")
    base = remote_api_base()
    url = f"{base}/control"
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
        executor_ok = payload.get("executor") == "worker_pool"
        dispatch_ok = payload.get("can_dispatch") is False
        ok = check(
            "Remote control runtime",
            executor_ok and dispatch_ok,
            f"executor={payload.get('executor')} | can_dispatch={payload.get('can_dispatch')}",
        )
        health_request = urllib.request.Request(f"{base}/health", headers={"Accept": "application/json"})
        with urllib.request.urlopen(health_request, timeout=10) as response:
            health_payload = json.loads(response.read())
        encryption_ok = check(
            "Broker credential encryption",
            health_payload.get("broker_credentials_encryption") is True,
            f"enabled={health_payload.get('broker_credentials_encryption')}",
        )
        return ok and encryption_ok
    except Exception as exc:
        return check("Remote control runtime", False, str(exc))


def check_remote_app() -> bool:
    print("\n[8] Remote App")
    base = remote_api_base()
    checks = [
        ("Health endpoint", f"{base}/health", 200),
        ("Control endpoint", f"{base}/control", 200),
        ("Auth me unauthorized", f"{base}/auth/me", 401),
        ("Portfolio locked without auth", f"{base}/account", 401),
    ]
    ok = True
    for label, url, expected in checks:
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            urllib.request.urlopen(request, timeout=10)
            status = 200
        except urllib.error.HTTPError as exc:
            status = exc.code
        except Exception as exc:
            ok &= check(label, False, str(exc))
            continue
        ok &= check(label, status == expected, f"expected {expected}, got {status}")
    return ok


def main(fast: bool = False) -> int:
    print("=" * 55)
    print("  ATZMA - System Health Check")
    print("=" * 55)

    results = [
        check_config(),
        check_dependencies(),
        check_files(),
        check_env(),
    ]
    if not fast:
        results.extend([check_alpaca(), check_model(), check_worker_runtime(), check_remote_app()])

    passed = sum(1 for result in results if result)
    total = len(results)
    print("\n" + "=" * 55)
    if passed == total:
        print(f"  {OK} All {total} checks passed")
    else:
        print(f"  {FAIL} {total - passed}/{total} checks failed")
    print("=" * 55)
    return 0 if passed == total else 1


def parse_args():
    parser = argparse.ArgumentParser(description="ATZMA Health Check")
    parser.add_argument("--fast", action="store_true", help="Skip network-dependent checks")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    sys.exit(main(fast=arguments.fast))
