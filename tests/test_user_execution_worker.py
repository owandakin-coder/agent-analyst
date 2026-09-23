import os
import sys
from types import SimpleNamespace

import pytest

import user_execution_worker as worker


@pytest.fixture(autouse=True)
def reset_worker_env(monkeypatch):
    monkeypatch.setattr(worker, "API_BASE", "https://example.test/api")
    monkeypatch.setattr(worker, "WORKER_TOKEN", "claim-token")
    monkeypatch.setattr(worker, "WORKER_SHARED_TOKEN", "shared-token")
    monkeypatch.setattr(worker, "JOB_ID", "job-123")
    monkeypatch.setattr(worker, "WORKER_ID", "worker-1")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    # _execute_with_main loads the model via ensemble_agent.load_ensemble()
    # (falls back to final_model.zip on its own when there's no ensemble on
    # disk) rather than main.load_trained_model_and_norm() directly — stub
    # it so tests never touch real model files. Individual tests still stub
    # main.step_live_once per-case since that's what carries each test's
    # actual scenario.
    fake_ensemble_agent = SimpleNamespace(load_ensemble=lambda: "model")
    monkeypatch.setitem(sys.modules, "ensemble_agent", fake_ensemble_agent)


def test_main_success_sets_broker_env_and_completes(monkeypatch):
    calls = []

    def fake_request(path, payload=None):
        calls.append((path, payload))
        if path == "/worker/jobs/claim":
            return {
                "broker_connection": {
                    "api_key": "user-key",
                    "secret_key": "user-secret",
                    "base_url": "https://paper-api.alpaca.markets",
                    "trading_mode": "paper",
                }
            }
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)

    fake_main = SimpleNamespace(
        step_live_once=lambda model, vec_norm, auto_approve=True: None,
    )
    monkeypatch.setitem(sys.modules, "main", fake_main)

    code = worker.main()

    assert code == 0
    assert os.environ["ALPACA_API_KEY"] == "user-key"
    assert os.environ["ALPACA_SECRET_KEY"] == "user-secret"
    assert os.environ["ALPACA_BASE_URL"] == "https://paper-api.alpaca.markets"
    assert calls[-1][0] == "/worker/jobs/complete"
    payload = calls[-1][1]
    assert payload["job_id"] == "job-123"
    assert payload["status"] == "succeeded"
    assert payload["result"]["job_id"] == "job-123"
    assert payload["result"]["mode"] == "paper"
    assert "decision_summary" in payload["result"]


def test_main_loads_model_via_ensemble_agent_not_single_model(monkeypatch):
    """Regression test: _execute_with_main must go through
    ensemble_agent.load_ensemble(), not main.load_trained_model_and_norm()
    directly — otherwise a retrain whose CI fast path only refreshes the
    ensemble members (train_ensemble's "reuse saved params" branch, which
    never touches final_model.zip) never actually reaches production,
    even though promote_model.py reports it as promoted."""
    def fake_request(path, payload=None):
        if path == "/worker/jobs/claim":
            return {
                "broker_connection": {
                    "api_key": "user-key",
                    "secret_key": "user-secret",
                    "base_url": "https://paper-api.alpaca.markets",
                    "trading_mode": "paper",
                }
            }
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)

    load_ensemble_calls = []
    fake_ensemble_agent = SimpleNamespace(
        load_ensemble=lambda: load_ensemble_calls.append(1) or "ensemble-model"
    )
    monkeypatch.setitem(sys.modules, "ensemble_agent", fake_ensemble_agent)

    received = {}

    def fake_step_live_once(model, vec_norm, auto_approve=True):
        received["model"] = model
        received["vec_norm"] = vec_norm
        return None

    fake_main = SimpleNamespace(step_live_once=fake_step_live_once)
    monkeypatch.setitem(sys.modules, "main", fake_main)

    code = worker.main()

    assert code == 0
    assert load_ensemble_calls == [1]
    assert received["model"] == "ensemble-model"
    assert received["vec_norm"] is None  # EnsembleAgent normalises per-member internally


def test_main_marks_skipped_on_zero_system_exit(monkeypatch):
    recorded = []

    def fake_request(path, payload=None):
        recorded.append((path, payload))
        if path == "/worker/jobs/claim":
            return {
                "broker_connection": {
                    "api_key": "user-key",
                    "secret_key": "user-secret",
                    "base_url": "https://paper-api.alpaca.markets",
                    "trading_mode": "paper",
                }
            }
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)

    def stop_once(*_args, **_kwargs):
        raise SystemExit(0)

    fake_main = SimpleNamespace(
        step_live_once=stop_once,
    )
    monkeypatch.setitem(sys.modules, "main", fake_main)

    code = worker.main()

    assert code == 0
    assert recorded[-1][1]["status"] == "skipped"
    assert recorded[-1][1]["result"]["reason"] == "market_closed_or_noop"


def test_main_marks_failed_when_broker_missing(monkeypatch):
    recorded = []

    def fake_request(path, payload=None):
        recorded.append((path, payload))
        if path == "/worker/jobs/claim":
            return {"broker_connection": None}
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)

    code = worker.main()

    assert code == 1
    assert recorded[-1] == (
        "/worker/jobs/complete",
        {"job_id": "job-123", "status": "failed", "result": {}, "error": "broker connection missing"},
    )


def test_main_returns_two_when_token_missing(monkeypatch, capsys):
    monkeypatch.setattr(worker, "WORKER_TOKEN", "")

    code = worker.main()

    captured = capsys.readouterr()
    assert code == 2
    assert "Missing ATZMA_WORKER_TOKEN" in captured.err


def test_main_marks_failed_on_runtime_error(monkeypatch):
    recorded = []

    def fake_request(path, payload=None):
        recorded.append((path, payload))
        if path == "/worker/jobs/claim":
            return {
                "broker_connection": {
                    "api_key": "user-key",
                    "secret_key": "user-secret",
                    "base_url": "https://paper-api.alpaca.markets",
                    "trading_mode": "live",
                }
            }
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)

    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    fake_main = SimpleNamespace(
        step_live_once=explode,
    )
    monkeypatch.setitem(sys.modules, "main", fake_main)

    with pytest.raises(RuntimeError, match="boom"):
        worker.main()

    assert recorded[-1][1]["status"] == "failed"
    assert recorded[-1][1]["error"] == "boom"


def test_poll_once_claims_request_and_completes(monkeypatch):
    calls = []

    def fake_request(path, payload=None, token=None):
        calls.append((path, payload, token))
        if path == "/worker/execution/claim-next":
            return {
                "request": {"id": "req-1"},
                "broker_connection": {
                    "api_key": "user-key",
                    "secret_key": "user-secret",
                    "base_url": "https://paper-api.alpaca.markets",
                    "trading_mode": "paper",
                },
            }
        if path == "/worker/execution/start":
            return {"ok": True, "request": {"id": "req-1", "status": "running"}}
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)
    fake_main = SimpleNamespace(
        step_live_once=lambda model, vec_norm, auto_approve=True: {
            "summary": "buy AAPL",
            "regime": "TRENDING_UP",
            "strategy_mode": "trend",
            "decisions": [{"ticker": "AAPL"}],
            "broker_orders": [{"order_id": "1", "ticker": "AAPL", "side": "BUY", "shares": 10, "status": "accepted"}],
        },
    )
    monkeypatch.setitem(sys.modules, "main", fake_main)

    code = worker.poll_once(worker_id="worker-1", lease_seconds=120)

    assert code == 0
    assert calls[0][0] == "/worker/execution/claim-next"
    assert calls[1][0] == "/worker/execution/start"
    assert calls[-1][0] == "/worker/execution/complete"
    assert calls[-1][1]["request_id"] == "req-1"
    assert calls[-1][1]["status"] == "succeeded"


def test_poll_once_returns_zero_when_queue_empty(monkeypatch):
    def fake_request(path, payload=None, token=None):
        if path == "/worker/execution/claim-next":
            return {"request": None}
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)
    assert worker.poll_once(worker_id="worker-1") == 0


def test_poll_once_reports_clear_error_on_401(monkeypatch, capsys):
    """A token mismatch should read as a distinct, actionable config error
    (return code 3), not an unhandled traceback that looks like transient
    noise — this is what a bad ATZMA_WORKER_SHARED_TOKEN actually looks like
    in practice (confirmed 2026-09 against the real deployed API)."""
    import urllib.error

    def fake_request(path, payload=None, token=None):
        if path == "/worker/execution/claim-next":
            raise urllib.error.HTTPError(
                "https://example.test/api/worker/execution/claim-next",
                401, "Unauthorized", hdrs=None, fp=None,
            )
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)
    assert worker.poll_once(worker_id="worker-1") == 3
    assert "401" in capsys.readouterr().err


def test_poll_once_reraises_non_401_http_errors(monkeypatch):
    """Only 401 gets the friendly message — any other HTTP error (5xx,
    network-shaped issues, ...) should still surface as a real exception
    rather than being silently swallowed."""
    import urllib.error

    def fake_request(path, payload=None, token=None):
        if path == "/worker/execution/claim-next":
            raise urllib.error.HTTPError(
                "https://example.test/api/worker/execution/claim-next",
                500, "Internal Server Error", hdrs=None, fp=None,
            )
        return {"ok": True}

    monkeypatch.setattr(worker, "_request", fake_request)
    with pytest.raises(urllib.error.HTTPError):
        worker.poll_once(worker_id="worker-1")


def test_worker_loop_runs_reconcile_poller(monkeypatch):
    calls = {"reconcile": 0, "poll": 0}

    monkeypatch.setattr(worker, "RECONCILE_INTERVAL_SECONDS", 1)

    def fake_reconcile(*, token=None, limit=20):
        calls["reconcile"] += 1
        return {"ok": True}

    def fake_poll_once(*, worker_id=None, lease_seconds=90):
        calls["poll"] += 1
        if calls["poll"] >= 2:
            raise KeyboardInterrupt()
        return 0

    timestamps = iter([6.0, 12.0, 18.0, 24.0])
    monkeypatch.setattr(worker, "reconcile_open_orders", fake_reconcile)
    monkeypatch.setattr(worker, "poll_once", fake_poll_once)
    monkeypatch.setattr(worker.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)

    with pytest.raises(KeyboardInterrupt):
        worker.worker_loop(worker_id="worker-1", poll_seconds=1, lease_seconds=30)

    assert calls["reconcile"] >= 1
    assert calls["poll"] >= 2
