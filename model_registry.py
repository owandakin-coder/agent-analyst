"""Content-addressed model staging, forward gates, and reversible releases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import statistics
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PAIR = ["final_model.zip", "vec_normalize.pkl"]
ENSEMBLE = [name for i in range(3) for name in (f"ensemble_{i}.zip", f"ensemble_norm_{i}.pkl")]


def artifacts(directory: Path) -> list[Path]:
    present = [(directory / name).exists() for name in ENSEMBLE]
    if any(present) and not all(present):
        raise ValueError("Incomplete ensemble release")
    names = ENSEMBLE if all(present) else PAIR
    return [directory / name for name in names]


def artifact_id(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in artifacts(directory):
        if not path.exists():
            return "unavailable"
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:24]


def copy_release(source: Path, target: Path) -> None:
    paths = artifacts(source)
    if not all(p.exists() for p in paths):
        raise ValueError("Incomplete model release")
    target.mkdir(parents=True, exist_ok=True)
    # Remove stale ensemble members when switching to a single model.
    for name in PAIR + ENSEMBLE + ["training_meta.pkl"]:
        (target / name).unlink(missing_ok=True)
    for path in paths:
        shutil.copy2(path, target / path.name)
    meta = source / "training_meta.pkl"
    if meta.exists():
        shutil.copy2(meta, target / meta.name)


def stage(source: Path, champion: Path, backtest: dict) -> dict:
    candidate = champion / "challenger"
    existing_manifest = candidate / "manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        age = datetime.now(timezone.utc) - datetime.fromisoformat(existing["staged_at"])
        if existing.get("champion_id") == artifact_id(champion / "snapshot") and age.days < 45 and artifact_id(candidate) == existing.get("candidate_id"):
            return {**existing, "stage_skipped": "Preserving the current candidate's forward window"}
    copy_release(source, candidate)
    manifest = {"candidate_id": artifact_id(candidate), "champion_id": artifact_id(champion / "snapshot"),
                "staged_at": datetime.now(timezone.utc).isoformat(), "backtest": backtest,
                "status": "shadow_only", "minimum_completed_sessions": 20}
    (candidate / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def shadow_predictions(data: dict, observation) -> dict:
    from execution_runtime import emit_execution_event
    predictions = {}
    active_id = artifact_id(MODEL_DIR)
    for folder, manifest_name, prefix, identity_key in (
        ("challenger", "manifest.json", "candidate:", "candidate_id"),
        ("rollback", "release.json", "rollback:", "previous_id"),
    ):
        directory = MODEL_DIR / folder
        manifest_path = directory / manifest_name
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = artifact_id(directory)
            if identity != manifest[identity_key]:
                raise ValueError("Shadow artifact hash mismatch")
            if folder == "challenger" and manifest["champion_id"] != active_id:
                continue
            if folder == "rollback" and manifest.get("active_id") != active_id:
                continue
            from promote_model import load_ensemble_from_paths
            paths = artifacts(directory)
            agent = load_ensemble_from_paths(list(zip(paths[::2], paths[1::2])), data)
            predictions[prefix + identity] = agent.predict(observation[None], deterministic=True)
        except Exception as exc:
            logging.getLogger("ModelRegistry").error("Shadow inference unavailable: %s", type(exc).__name__)
            emit_execution_event("shadow_unavailable", {"component": folder, "error": type(exc).__name__})
    return predictions


def forward_gate(rows: list[dict], strategy: str, champion_id: str, minimum_sessions: int = 20,
                 now: datetime | None = None, improvement: float = 0.005, require_benchmark: bool = True,
                 not_before: str | None = None) -> tuple[bool, dict]:
    """Compare paired daily marks, never inflate evidence with intraday samples/users."""
    now = now or datetime.now(timezone.utc)
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XNYS")
    start_at = datetime.fromisoformat(not_before.replace("Z", "+00:00")) if not_before else None
    if start_at is not None and start_at.tzinfo is None:
        raise ValueError("Release evidence boundary requires a timezone")
    today = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    paired = {}
    invalid_pairs = set()
    rejected_rows = 0
    required = (strategy, "champion", "benchmark_spy") if require_benchmark else (strategy, "champion")
    for row in rows:
        payload = row.get("payload", {})
        if row.get("strategy") not in {strategy, "champion", "benchmark_spy"} or payload.get("champion_id") != champion_id:
            continue
        if payload.get("source") != "shadow_simulation" or payload.get("cost_bps") != 10 or payload.get("schema_version") != 2:
            continue
        try:
            timestamp = datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("Missing timezone")
            if start_at is not None and timestamp < start_at:
                raise ValueError("Observation predates this candidate/release")
            day = timestamp.astimezone(ZoneInfo("America/New_York")).date().isoformat()
            equity = float(row["equity"])
            if day >= today or not math.isfinite(equity) or equity <= 0 or not calendar.is_session(day):
                raise ValueError("Not a completed exchange session")
            if not calendar.session_open(day).to_pydatetime() <= timestamp <= calendar.session_close(day).to_pydatetime():
                raise ValueError("Observation outside regular session")
            if not row.get("user_id") or not row.get("broker_account_id") or not row.get("execution_request_id"):
                raise ValueError("Missing account/request scope")
        except (ValueError, TypeError, KeyError):
            rejected_rows += 1
            continue
        key = ((row["user_id"], row["broker_account_id"]), row["execution_request_id"])
        group = paired.setdefault(key, {})
        value = (timestamp, day, equity)
        if row["strategy"] in group and group[row["strategy"]] != value:
            invalid_pairs.add(key)
        group[row["strategy"]] = value
    accounts = {}
    for (user, request), group in paired.items():
        if (user, request) in invalid_pairs or not all(name in group for name in required):
            continue
        if len({group[name][0] for name in required}) != 1:
            continue
        timestamp, day, _ = group[strategy]
        prior = accounts.setdefault(user, {}).get(day)
        if prior is None or timestamp > prior[0]:
            accounts[user][day] = (timestamp, {name: group[name][2] for name in group})
    results = []
    for days in accounts.values():
        ordered = [values for _, values in sorted(days.values(), key=lambda item: item[0])]
        if len(ordered) < minimum_sessions + 1:
            continue
        metrics = {}
        for name in required:
            values = [row[name] for row in ordered]
            returns = [b / a - 1 for a, b in zip(values, values[1:])]
            peak, drawdown = values[0], 0.0
            for value in values:
                peak = max(peak, value)
                drawdown = max(drawdown, 1 - value / peak)
            std = statistics.stdev(returns) if len(returns) > 1 else 0
            metrics[name] = {"return": values[-1] / values[0] - 1, "max_drawdown": drawdown,
                             "sharpe": statistics.mean(returns) / std * math.sqrt(252) if std > 1e-12 else 0}
        results.append({"sessions": len(ordered) - 1, "metrics": metrics})
    report = {"eligible_accounts": len(results), "minimum_completed_sessions": minimum_sessions, "source": "paired_forward_shadow",
              "calendar": "XNYS", "schema_version": 2, "rejected_rows": rejected_rows, "accounts": results}
    if not results:
        return False, {**report, "reason": "Insufficient paired, completed forward sessions"}
    def median(name, metric):
        return statistics.median(r["metrics"][name][metric] for r in results)
    passed = (median(strategy, "return") >= median("champion", "return") + improvement
              and (not require_benchmark or median(strategy, "return") >= median("benchmark_spy", "return"))
              and median(strategy, "max_drawdown") <= median("champion", "max_drawdown") + 0.01
              and median(strategy, "sharpe") >= max(0, median("champion", "sharpe")))
    return passed, {**report, "reason": "Forward gate passed" if passed else "Candidate did not clear return/risk comparisons"}


def evidence() -> list[dict]:
    token = os.environ.get("ATZMA_WORKER_SHARED_TOKEN", "")
    if not token:
        raise ValueError("Worker shared token required for evidence")
    base = os.environ.get("ATZMA_REMOTE_API_BASE", "https://sofowpweliticltlbxrj.supabase.co/functions/v1/api")
    rows = []
    while True:
        request = urllib.request.Request(base + "/worker/research/evidence", data=json.dumps({"offset": len(rows)}).encode(),
            headers={"Content-Type": "application/json", "x-atzma-worker-token": token})
        with urllib.request.urlopen(request, timeout=30) as response:
            page = json.load(response)["observations"]
        rows.extend(page)
        if len(page) < 1000:
            return rows


def promote(directory: Path, rows: list[dict]) -> dict:
    candidate = directory / "challenger"
    manifest_path = candidate / "manifest.json"
    if not manifest_path.exists():
        return {"promoted": False, "reason": "No staged candidate"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_id(candidate) != manifest["candidate_id"]:
        raise ValueError("Release artifacts no longer match the staged comparison")
    if artifact_id(directory) != manifest["champion_id"]:
        return {"promoted": False, "reason": "Staged comparison superseded; awaiting the next retrain"}
    passed, report = forward_gate(rows, "candidate:" + manifest["candidate_id"], manifest["champion_id"], not_before=manifest["staged_at"])
    if not passed:
        return {"promoted": False, **report}
    rollback = directory / "rollback"
    copy_release(directory, rollback)
    release = {"active_id": manifest["candidate_id"], "previous_id": manifest["champion_id"],
               "promoted_at": datetime.now(timezone.utc).isoformat(), "evidence": report}
    (rollback / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    copy_release(candidate, directory)
    manifest_path.unlink()
    return {"promoted": True, **release}


def rollback_if_degraded(directory: Path, rows: list[dict]) -> dict:
    backup = directory / "rollback"
    manifest_path = backup / "release.json"
    if not manifest_path.exists():
        return {"rolled_back": False}
    release = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_id(directory) != release["active_id"] or artifact_id(backup) != release["previous_id"]:
        raise ValueError("Rollback release hash mismatch")
    passed, report = forward_gate(rows, "rollback:" + release["previous_id"], release["active_id"], minimum_sessions=5, improvement=0.02,
                                  require_benchmark=False, not_before=release["promoted_at"])
    if not passed:
        return {"rolled_back": False, **report}
    copy_release(backup, directory)
    (backup / "rollback-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path.unlink()
    return {"rolled_back": True, **report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "restore-snapshot", "govern"])
    args = parser.parse_args()
    if args.action == "snapshot":
        copy_release(MODEL_DIR, MODEL_DIR / "snapshot")
    elif args.action == "restore-snapshot":
        copy_release(MODEL_DIR / "snapshot", MODEL_DIR)
    else:
        rows = evidence()
        rollback = rollback_if_degraded(MODEL_DIR, rows)
        result = rollback if rollback.get("rolled_back") else promote(MODEL_DIR, rows)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
