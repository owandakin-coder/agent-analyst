import os
import sys
from types import SimpleNamespace

import pytest

import user_execution_worker as worker


@pytest.fixture(autouse=True)
def reset_worker_env(monkeypatch):
    monkeypatch.setattr(worker, "API_BASE", "https://example.test/api")
    monkeypatch.setattr(worker, "WORKER_TOKEN", "claim-token")
    monkeypatch.setattr(worker, "JOB_ID", "job-123")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)


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
        load_trained_model_and_norm=lambda: ("model", "vec"),
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
        load_trained_model_and_norm=lambda: ("model", "vec"),
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
        load_trained_model_and_norm=lambda: ("model", "vec"),
        step_live_once=explode,
    )
    monkeypatch.setitem(sys.modules, "main", fake_main)

    with pytest.raises(RuntimeError, match="boom"):
        worker.main()

    assert recorded[-1][1]["status"] == "failed"
    assert recorded[-1][1]["error"] == "boom"
