"""
Shared trading control state for dashboard, GitHub Actions, and live trading.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from config_loader import CFG
except Exception:  # pragma: no cover - config fallback for isolated use
    CFG = None


LOCAL_CONTROL_STATE_FILE = Path("runtime/control_state.local.json")
DEFAULT_GITHUB_CONTROL_PATH = "runtime/control_state.json"
DEFAULT_GITHUB_WORKFLOW = "trade.yml"
DEFAULT_GITHUB_BRANCH = "main"
USER_AGENT = "ATZMA-ControlPlane/1.0"
DEFAULT_CONTROL_API_URL = "https://sofowpweliticltlbxrj.supabase.co/functions/v1/api/control"


def _cfg_get(*keys, default=None):
    if CFG is None:
        return default
    return CFG.get(*keys, default=default)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def github_repo() -> str:
    return (
        os.getenv("GITHUB_REPOSITORY", "").strip()
        or os.getenv("ATZMA_GITHUB_REPO", "").strip()
        or str(_cfg_get("github", "repo", default="")).strip()
    )


def github_token() -> str:
    return (
        os.getenv("GITHUB_TOKEN", "").strip()
        or os.getenv("GH_TOKEN", "").strip()
        or os.getenv("GITHUB_PAT", "").strip()
        or os.getenv("ATZMA_GITHUB_TOKEN", "").strip()
    )


def github_control_state_path() -> str:
    return str(_cfg_get("github", "control_state_path", default=DEFAULT_GITHUB_CONTROL_PATH)).strip()


def github_trade_workflow() -> str:
    return str(_cfg_get("github", "trade_workflow", default=DEFAULT_GITHUB_WORKFLOW)).strip()


def github_default_branch() -> str:
    return (
        os.getenv("GITHUB_REF_NAME", "").strip()
        or os.getenv("ATZMA_GITHUB_BRANCH", "").strip()
        or str(_cfg_get("github", "branch", default=DEFAULT_GITHUB_BRANCH)).strip()
    )


def control_api_url() -> str:
    return (
        os.getenv("ATZMA_CONTROL_API_URL", "").strip()
        or str(_cfg_get("supabase", "control_api_url", default="")).strip()
        or DEFAULT_CONTROL_API_URL
    )


def default_control_state() -> dict:
    return {
        "mode": "paper",
        "trading_enabled": True,
        "emergency_stop": False,
        "status": "running",
        "executor": "github_actions",
        "executor_label": "GitHub Actions",
        "last_command": "bootstrap",
        "last_command_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "updated_by": "system",
        "note": "Paper engine is allowed to execute.",
        "command_version": 1,
    }


def normalize_control_state(raw: dict | None) -> dict:
    state = default_control_state()
    if isinstance(raw, dict):
        state.update(raw)

    if state.get("emergency_stop"):
        state["status"] = "stopped"
        state["trading_enabled"] = False
    elif not state.get("trading_enabled", True):
        state["status"] = "paused"
    else:
        state["status"] = "running"

    state["mode"] = str(state.get("mode", "paper")).lower()
    state["updated_at"] = str(state.get("updated_at") or utc_now_iso())
    state["last_command_at"] = str(state.get("last_command_at") or state["updated_at"])
    state["updated_by"] = str(state.get("updated_by") or "system")
    state["note"] = str(state.get("note") or "")
    state["executor"] = str(state.get("executor") or "github_actions")
    state["executor_label"] = str(state.get("executor_label") or "GitHub Actions")
    state["command_version"] = int(state.get("command_version", 1) or 1)
    return state


def _github_headers(token: str, *, json_content: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def _github_contents_url(repo: str, path: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents/{path}"


def _github_request(url: str, *, method: str = "GET", token: str = "", payload: dict | None = None) -> dict:
    data = None
    headers = _github_headers(token, json_content=payload is not None)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def load_remote_control_state() -> dict:
    api_url = control_api_url()
    if api_url:
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body) if body else {}
            normalized = normalize_control_state(payload)
            normalized["_source"] = str(payload.get("_source") or "control_api")
            return normalized
        except Exception:
            pass

    repo = github_repo()
    if not repo:
        raise RuntimeError("GitHub repository is not configured")

    token = github_token()
    url = _github_contents_url(repo, github_control_state_path())
    payload = _github_request(url, token=token)
    content = payload.get("content", "")
    if not content:
        return default_control_state()
    decoded = base64.b64decode(content).decode("utf-8")
    state = json.loads(decoded)
    normalized = normalize_control_state(state)
    normalized["_source"] = "github_repo"
    return normalized


def load_local_control_state() -> dict:
    if LOCAL_CONTROL_STATE_FILE.exists():
        with open(LOCAL_CONTROL_STATE_FILE, encoding="utf-8") as handle:
            state = json.load(handle)
        normalized = normalize_control_state(state)
        normalized["_source"] = "local_file"
        return normalized
    state = default_control_state()
    state["_source"] = "default"
    return state


def load_control_state(prefer_remote: bool = True) -> dict:
    if prefer_remote:
        try:
            return load_remote_control_state()
        except Exception:
            pass
    return load_local_control_state()


def save_local_control_state(state: dict) -> dict:
    LOCAL_CONTROL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_control_state(state)
    with open(LOCAL_CONTROL_STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=True, indent=2, sort_keys=True)
    normalized["_source"] = "local_file"
    return normalized


def save_remote_control_state(state: dict, actor: str = "dashboard") -> dict:
    repo = github_repo()
    token = github_token()
    if not repo or not token:
        raise RuntimeError("GitHub repository or token is missing")

    path = github_control_state_path()
    url = _github_contents_url(repo, path)
    normalized = normalize_control_state(state)
    sha = None
    try:
        existing = _github_request(url, token=token)
        sha = existing.get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    payload = {
        "message": f"ATZMA control update by {actor}",
        "content": base64.b64encode(
            json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        ).decode("utf-8"),
        "branch": github_default_branch(),
    }
    if sha:
        payload["sha"] = sha

    _github_request(url, method="PUT", token=token, payload=payload)
    normalized["_source"] = "github_repo"
    return normalized


def save_control_state(state: dict, actor: str = "dashboard", prefer_remote: bool = True) -> dict:
    if prefer_remote:
        try:
            return save_remote_control_state(state, actor=actor)
        except Exception:
            pass
    return save_local_control_state(state)


def apply_control_action(action: str, actor: str = "dashboard", prefer_remote: bool = True) -> dict:
    normalized_action = action.strip().lower()
    state = load_control_state(prefer_remote=prefer_remote)

    if normalized_action == "pause":
        state["trading_enabled"] = False
        state["emergency_stop"] = False
        state["note"] = "Trading paused by operator."
    elif normalized_action == "resume":
        state["trading_enabled"] = True
        state["emergency_stop"] = False
        state["note"] = "Trading resumed by operator."
    elif normalized_action in ("stop", "emergency_stop"):
        state["trading_enabled"] = False
        state["emergency_stop"] = True
        state["note"] = "Emergency stop is active. No new execution is allowed."
    elif normalized_action == "paper":
        state["mode"] = "paper"
        state["note"] = "Paper mode selected from control surface."
    elif normalized_action == "live":
        state["mode"] = "live"
        state["note"] = "Live mode selected from control surface."
    else:
        raise ValueError(f"Unsupported control action: {action}")

    state["last_command"] = normalized_action
    state["last_command_at"] = utc_now_iso()
    state["updated_at"] = state["last_command_at"]
    state["updated_by"] = actor
    state["command_version"] = int(state.get("command_version", 1) or 1) + 1
    return save_control_state(state, actor=actor, prefer_remote=prefer_remote)


def can_trade(state: dict | None = None) -> tuple[bool, str | None]:
    current = normalize_control_state(state or load_control_state())
    if current.get("emergency_stop"):
        return False, "emergency_stop"
    if not current.get("trading_enabled", True):
        return False, "paused"
    return True, None


def control_status_summary(state: dict | None = None) -> dict:
    current = normalize_control_state(state or load_control_state())
    allowed, reason = can_trade(current)
    current["trade_allowed"] = allowed
    current["block_reason"] = reason
    current["github_repo"] = github_repo()
    current["control_state_path"] = github_control_state_path()
    current["workflow"] = github_trade_workflow()
    current["branch"] = github_default_branch()
    current["can_dispatch"] = bool(github_repo() and github_token())
    return current


def dispatch_trade_workflow(actor: str = "dashboard") -> dict:
    repo = github_repo()
    token = github_token()
    if not repo or not token:
        raise RuntimeError("GitHub repository or token is missing")

    workflow = github_trade_workflow()
    branch = github_default_branch()
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    payload = {"ref": branch}
    _github_request(url, method="POST", token=token, payload=payload)
    return {
        "status": "dispatched",
        "workflow": workflow,
        "branch": branch,
        "repo": repo,
        "actor": actor,
        "dispatched_at": utc_now_iso(),
    }
