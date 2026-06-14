"""
Execute isolated per-user trading requests from the ATZMA backend queue.
Supports both the legacy job-id flow and the new poll-based execution_requests flow.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

from execution_runtime import clear_execution_runtime_hooks, set_execution_runtime_hooks
from logging_setup import configure_logging


API_BASE = os.getenv("ATZMA_REMOTE_API_BASE", "https://sofowpweliticltlbxrj.supabase.co/functions/v1/api").rstrip("/")
WORKER_TOKEN = os.getenv("ATZMA_WORKER_TOKEN", "")
WORKER_SHARED_TOKEN = os.getenv("ATZMA_WORKER_SHARED_TOKEN", WORKER_TOKEN)
JOB_ID = os.getenv("ATZMA_JOB_ID", "").strip()
WORKER_ID = os.getenv("ATZMA_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")
RECONCILE_INTERVAL_SECONDS = int(os.getenv("ATZMA_RECONCILE_INTERVAL_SECONDS", "30"))

configure_logging()


def _request(path: str, payload: dict | None = None, *, token: str | None = None) -> dict:
    headers = {
        "Accept": "application/json",
        "x-atzma-worker-token": token or WORKER_TOKEN,
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


def _set_broker_env(broker: dict) -> None:
    os.environ["ALPACA_API_KEY"] = broker.get("api_key", "")
    os.environ["ALPACA_SECRET_KEY"] = broker.get("secret_key", "")
    os.environ["ALPACA_BASE_URL"] = broker.get("base_url", "https://paper-api.alpaca.markets")


class _LeaseHeartbeat:
    def __init__(self, *, worker_id: str, request_id: str, lease_seconds: int, token: str):
        self.worker_id = worker_id
        self.request_id = request_id
        self.lease_seconds = lease_seconds
        self.token = token
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"lease-heartbeat-{self.request_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        interval = max(5, min(self.lease_seconds // 3, 20))
        while not self._stop.wait(interval):
            try:
                _request(
                    "/worker/execution/heartbeat",
                    {
                        "worker_id": self.worker_id,
                        "request_id": self.request_id,
                        "lease_seconds": self.lease_seconds,
                        "active_jobs": 1,
                        "capacity": 1,
                    },
                    token=self.token,
                )
            except Exception:
                # Keep the worker running and let lease ownership decide safety.
                pass


def _result_payload(request_id: str, broker: dict, decision_result: dict | None) -> dict:
    decision_result = decision_result or {}
    return {
        "request_id": request_id,
        "job_id": request_id,
        "mode": broker.get("trading_mode", "paper"),
        "decision_summary": decision_result.get("summary"),
        "regime": decision_result.get("regime"),
        "strategy_mode": decision_result.get("strategy_mode"),
        "decisions": decision_result.get("decisions", []),
        "raw_action": decision_result.get("raw_action", []),
        "scaled_action": decision_result.get("scaled_action", []),
        "drawdown": decision_result.get("drawdown"),
        "risk_level": decision_result.get("risk_level"),
        "broker_orders": decision_result.get("broker_orders", []),
        "payload": decision_result,
    }


def _install_runtime_hooks(request_id: str, token: str) -> None:
    def execution_hook(stage: str, payload: dict) -> None:
        _request("/worker/execution/event", {
            "request_id": request_id,
            "stage": stage,
            "payload": payload,
        }, token=token)

    def risk_hook(stage: str, payload: dict) -> None:
        _request("/worker/risk/event", {
            "request_id": request_id,
            "event_type": stage,
            "payload": payload,
        }, token=token)

    def order_prepare_hook(payload: dict) -> dict | None:
        response = _request("/worker/orders/prepare", {
            "request_id": request_id,
            "order": payload,
        }, token=token)
        return {
            "order_id": response.get("order_id"),
            "client_order_id": response.get("client_order_id"),
        }

    def order_update_hook(payload: dict) -> None:
        _request("/worker/orders/update", {
            "request_id": request_id,
            "order": payload,
        }, token=token)

    set_execution_runtime_hooks(
        execution_hook=execution_hook,
        risk_hook=risk_hook,
        order_prepare_hook=order_prepare_hook,
        order_update_hook=order_update_hook,
    )


def _execute_with_main(request_id: str, broker: dict, on_success, on_skip, on_failure) -> int:
    _set_broker_env(broker)
    try:
        from main import load_trained_model_and_norm, step_live_once

        model, vec_norm = load_trained_model_and_norm()
        decision_result = step_live_once(model, vec_norm, auto_approve=True) or {}
        on_success(_result_payload(request_id, broker, decision_result))
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        if code == 0:
            on_skip({"request_id": request_id, "reason": "market_closed_or_noop"})
            return 0
        on_failure(f"system_exit:{code}")
        return code
    except urllib.error.HTTPError as exc:
        on_failure(f"http_error:{exc.code}")
        return 1
    except Exception as exc:
        on_failure(str(exc))
        raise
    finally:
        clear_execution_runtime_hooks()


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

    return _execute_with_main(
        JOB_ID,
        broker,
        lambda result: complete(JOB_ID, "succeeded", result=result),
        lambda result: complete(JOB_ID, "skipped", result=result),
        lambda error: complete(JOB_ID, "failed", error=error),
    )


def poll_once(*, worker_id: str | None = None, lease_seconds: int = 90) -> int:
    token = WORKER_SHARED_TOKEN or WORKER_TOKEN
    if not token:
        print("Missing ATZMA_WORKER_SHARED_TOKEN", file=sys.stderr)
        return 2

    worker_id = worker_id or WORKER_ID
    claim = _request(
        "/worker/execution/claim-next",
        {"worker_id": worker_id, "lease_seconds": lease_seconds},
        token=token,
    )
    request = claim.get("request") or {}
    if not request:
        return 0

    request_id = str(request.get("id") or "")
    broker = claim.get("broker_connection") or {}
    if not broker:
        _request(
            "/worker/execution/complete",
            {"request_id": request_id, "status": "failed", "error": "broker connection missing", "worker_id": worker_id},
            token=token,
        )
        return 1

    started = _request(
        "/worker/execution/start",
        {"request_id": request_id, "worker_id": worker_id},
        token=token,
    )
    if started.get("error"):
        return 1

    heartbeat = _LeaseHeartbeat(worker_id=worker_id, request_id=request_id, lease_seconds=lease_seconds, token=token)
    heartbeat.start()
    _install_runtime_hooks(request_id, token)

    try:
        return _execute_with_main(
            request_id,
            broker,
            lambda result: _request(
                "/worker/execution/complete",
                {"request_id": request_id, "status": "succeeded", "result": result, "worker_id": worker_id},
                token=token,
            ),
            lambda result: _request(
                "/worker/execution/complete",
                {"request_id": request_id, "status": "skipped", "result": result, "worker_id": worker_id},
                token=token,
            ),
            lambda error: _request(
                "/worker/execution/complete",
                {"request_id": request_id, "status": "failed", "error": error, "worker_id": worker_id},
                token=token,
            ),
        )
    finally:
        heartbeat.stop()


def reconcile_open_orders(*, token: str | None = None, limit: int = 20) -> dict:
    return _request(
        "/worker/reconcile/open",
        {"limit": max(1, min(limit, 100))},
        token=token or WORKER_SHARED_TOKEN or WORKER_TOKEN,
    )


def worker_loop(*, worker_id: str | None = None, poll_seconds: int = 10, lease_seconds: int = 90) -> int:
    worker_id = worker_id or WORKER_ID
    token = WORKER_SHARED_TOKEN or WORKER_TOKEN
    if not token:
        print("Missing ATZMA_WORKER_SHARED_TOKEN", file=sys.stderr)
        return 2

    last_reconcile_at = 0.0
    while True:
        now = time.time()
        if now - last_reconcile_at >= max(RECONCILE_INTERVAL_SECONDS, 5):
            try:
                reconcile_open_orders(token=token)
            except Exception:
                pass
            last_reconcile_at = now
        code = poll_once(worker_id=worker_id, lease_seconds=lease_seconds)
        if code not in (0,):
            time.sleep(min(max(poll_seconds, 5), 60))
            continue
        time.sleep(max(poll_seconds, 1))


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="ATZMA execution worker")
    parser.add_argument("--poll-once", action="store_true", help="Claim and execute one execution_request from the durable queue")
    parser.add_argument("--loop", action="store_true", help="Continuously poll the durable queue")
    parser.add_argument("--worker-id", default=WORKER_ID)
    parser.add_argument("--poll-seconds", type=int, default=int(os.getenv("ATZMA_WORKER_POLL_SECONDS", "10")))
    parser.add_argument("--lease-seconds", type=int, default=int(os.getenv("ATZMA_WORKER_LEASE_SECONDS", "90")))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.loop:
        raise SystemExit(worker_loop(worker_id=args.worker_id, poll_seconds=args.poll_seconds, lease_seconds=args.lease_seconds))
    if args.poll_once:
        raise SystemExit(poll_once(worker_id=args.worker_id, lease_seconds=args.lease_seconds))
    raise SystemExit(main())
