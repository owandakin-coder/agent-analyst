from __future__ import annotations

from pathlib import Path

import worker_entry


def test_worker_entry_fails_when_required_env_missing(monkeypatch):
    monkeypatch.delenv("ATZMA_REMOTE_API_BASE", raising=False)
    monkeypatch.delenv("ATZMA_WORKER_SHARED_TOKEN", raising=False)
    assert worker_entry.main() == 2


def test_worker_entry_fails_when_model_artifacts_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ATZMA_REMOTE_API_BASE", "https://example.com")
    monkeypatch.setenv("ATZMA_WORKER_SHARED_TOKEN", "token")
    monkeypatch.setattr(
        worker_entry,
        "_required_model_paths",
        lambda: [Path(tmp_path) / "final_model.zip", Path(tmp_path) / "vec_normalize.pkl"],
    )
    assert worker_entry.main() == 3


def test_worker_entry_starts_loop_with_valid_runtime(monkeypatch, tmp_path):
    model_dir = Path(tmp_path)
    (model_dir / "final_model.zip").write_bytes(b"model")
    (model_dir / "vec_normalize.pkl").write_bytes(b"norm")

    monkeypatch.setenv("ATZMA_REMOTE_API_BASE", "https://example.com")
    monkeypatch.setenv("ATZMA_WORKER_SHARED_TOKEN", "token")
    monkeypatch.setenv("ATZMA_WORKER_POLL_SECONDS", "7")
    monkeypatch.setenv("ATZMA_WORKER_LEASE_SECONDS", "45")
    monkeypatch.setattr(
        worker_entry,
        "_required_model_paths",
        lambda: [model_dir / "final_model.zip", model_dir / "vec_normalize.pkl"],
    )

    called = {}

    def fake_loop(*, worker_id, poll_seconds, lease_seconds):
        called["worker_id"] = worker_id
        called["poll_seconds"] = poll_seconds
        called["lease_seconds"] = lease_seconds
        return 0

    monkeypatch.setattr(worker_entry, "worker_loop", fake_loop)
    assert worker_entry.main() == 0
    assert called == {
        "worker_id": None,
        "poll_seconds": 7,
        "lease_seconds": 45,
    }
