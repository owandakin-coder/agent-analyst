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

    return ok


def check_env() -> bool:
    print("\n[4] Environment")
    from dotenv import load_dotenv

    load_dotenv()
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
    for key in ["GITHUB_TOKEN", "SUPABASE_ACCESS_TOKEN", "ATZMA_BROKER_CREDENTIAL_KEY"]:
        value = os.getenv(key, "")
        if value:
            check(f"{key} (launch)", True)
        else:
            warn(f"{key} (launch)", "not set")
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
        return check("Broker connection", False, str(exc))


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


def infer_github_repo() -> str:
    from config_loader import CFG

    if CFG.github_repo:
        return CFG.github_repo
    if os.getenv("GITHUB_REPOSITORY"):
        return os.getenv("GITHUB_REPOSITORY", "")
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            check=True,
            capture_output=True,
            text=True,
        )
        remote = result.stdout.strip()
    except Exception:
        return ""

    if remote.endswith(".git"):
        remote = remote[:-4]
    if remote.startswith("git@github.com:"):
        return remote.split("git@github.com:", 1)[1]
    if "github.com/" in remote:
        return remote.split("github.com/", 1)[1].strip("/")
    return ""


def check_github_actions() -> bool:
    print("\n[7] GitHub")
    repo = infer_github_repo()
    if not repo:
        warn("GitHub Actions check skipped", "repo not configured")
        return True

    url = f"https://api.github.com/repos/{repo}/actions/runs?per_page=5"
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
        runs = payload.get("workflow_runs", [])
        if not runs:
            warn("No workflow runs found", repo)
            return True
        run = runs[0]
        status = run.get("conclusion") or run.get("status", "?")
        date = str(run.get("created_at", ""))[:10]
        return check(f"Last run: {run.get('name', 'workflow')}", status in ("success", None), f"{repo} · {date} · {status}")
    except Exception as exc:
        warn("GitHub Actions check skipped", str(exc))
        return True


def check_remote_app() -> bool:
    print("\n[8] Remote App")
    base = "https://sofowpweliticltlbxrj.supabase.co/functions/v1/api"
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
        results.extend([check_alpaca(), check_model(), check_github_actions(), check_remote_app()])

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
