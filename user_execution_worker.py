"""
Execute one isolated per-user trading job from the Supabase queue.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from logging_setup import configure_logging


API_BASE = os.getenv("ATZMA_REMOTE_API_BASE", "https://sofowpweliticltlbxrj.supabase.co/functions/v1/api").rstrip("/")
WORKER_TOKEN = os.getenv("ATZMA_WORKER_TOKEN", "")
JOB_ID = os.getenv("ATZMA_JOB_ID", "").strip()

configure_logging()


def _request(path: str, payload: dict | None = None) -> dict:
    headers = {
        "Accept": "application/json",
        "x-atzma-worker-token": WORKER_TOKEN,
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{API_BASE}{path}", data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def complete(job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
    _request("/worker/jobs/complete", {
        "job_id": job_id,
        "status": status,
        "result": result or {},
        "error": error,
    })


def main() -> int:
    if not WORKER_TOKEN:
        print("Missing ATZMA_WORKER_TOKEN", file=sys.stderr)
        return 2
    if not JOB_ID:
        print("Missing ATZMA_JOB_ID", file=sys.stderr)
        return 2

    claim = _request("/worker/jobs/claim", {"job_id": JOB_ID})
    broker = claim.get("broker_connection") or {}
    if not broker:
        complete(JOB_ID, "failed", error="broker connection missing")
        return 1

    os.environ["ALPACA_API_KEY"] = broker.get("api_key", "")
    os.environ["ALPACA_SECRET_KEY"] = broker.get("secret_key", "")
    os.environ["ALPACA_BASE_URL"] = broker.get("base_url", "https://paper-api.alpaca.markets")

    try:
        from main import load_trained_model_and_norm, step_live_once

        model, vec_norm = load_trained_model_and_norm()
        step_live_once(model, vec_norm, auto_approve=True)
        complete(JOB_ID, "succeeded", result={"job_id": JOB_ID, "mode": broker.get("trading_mode", "paper")})
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        if code == 0:
            complete(JOB_ID, "skipped", result={"job_id": JOB_ID, "reason": "market_closed_or_noop"})
            return 0
        complete(JOB_ID, "failed", error=f"system_exit:{code}")
        return code
    except urllib.error.HTTPError as exc:
        complete(JOB_ID, "failed", error=f"http_error:{exc.code}")
        return 1
    except Exception as exc:
        complete(JOB_ID, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
