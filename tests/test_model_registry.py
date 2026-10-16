from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pytest
import model_registry as registry


def observations(days=25, candidate_gain=.004):
    rows = []
    for i in range(days):
        when = (datetime(2026, 1, 1, 20, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
        for strategy, gain in [("candidate:new", candidate_gain), ("champion", .001), ("benchmark_spy", .0015)]:
            rows.append({"user_id": "one", "execution_request_id": str(i), "strategy": strategy,
                         "observed_at": when, "equity": 100000 * (1 + gain) ** i,
                         "payload": {"champion_id": "old", "source": "shadow_simulation", "cost_bps": 10, "schema_version": 1}})
    return rows


def test_forward_gate_requires_completed_paired_sessions():
    passed, report = registry.forward_gate(observations(10), "candidate:new", "old")
    assert not passed and report["eligible_accounts"] == 0
    # Duplicating intraday observations/accounts must not manufacture more days.
    passed, _ = registry.forward_gate(observations(10) * 20, "candidate:new", "old")
    assert not passed


def test_forward_gate_blocks_wrong_costs_versions_and_unpaired_marks():
    for change in [lambda row: row["payload"].update(cost_bps=0), lambda row: row["payload"].update(champion_id="wrong")]:
        rows = observations()
        for row in rows:
            change(row)
        assert registry.forward_gate(rows, "candidate:new", "old")[0] is False
    assert not registry.forward_gate([r for r in observations() if r["strategy"] != "champion"], "candidate:new", "old")[0]


def test_forward_gate_rejects_underperforming_candidate():
    assert registry.forward_gate(observations(candidate_gain=-.001), "candidate:new", "old")[0] is False


def test_switching_broker_accounts_cannot_combine_short_evidence_windows():
    rows = observations()
    for row in rows:
        row["broker_account_id"] = "first" if int(row["execution_request_id"]) < 12 else "second"
    assert not registry.forward_gate(rows, "candidate:new", "old")[0]


def test_forward_gate_accepts_only_completed_positive_improvement():
    assert registry.forward_gate(observations(), "candidate:new", "old")[0] is True
    rows = observations(25)
    assert registry.forward_gate(rows, "candidate:new", "old", now=datetime(2026, 1, 15, tzinfo=timezone.utc))[0] is False


def test_release_hash_changes_with_weights_and_norm(tmp_path):
    (tmp_path / "final_model.zip").write_bytes(b"weights")
    (tmp_path / "vec_normalize.pkl").write_bytes(b"norm")
    identity = registry.artifact_id(tmp_path)
    (tmp_path / "vec_normalize.pkl").write_bytes(b"different")
    assert registry.artifact_id(tmp_path) != identity


def test_incomplete_ensemble_cannot_silently_become_a_single_model(tmp_path, monkeypatch):
    for name in registry.PAIR:
        (tmp_path / name).write_bytes(b"fallback")
    (tmp_path / "ensemble_0.zip").write_bytes(b"partial")
    with pytest.raises(ValueError, match="Incomplete ensemble"):
        registry.artifact_id(tmp_path)
    import ensemble_agent
    monkeypatch.setattr(ensemble_agent, "MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="Incomplete ensemble"):
        ensemble_agent.load_ensemble()


def test_staging_does_not_change_production_and_keeps_snapshot(tmp_path):
    for name in registry.PAIR:
        (tmp_path / name).write_bytes(b"old")
    registry.copy_release(tmp_path, tmp_path / "snapshot")
    for name in registry.PAIR:
        (tmp_path / name).write_bytes(b"new")
    manifest = registry.stage(tmp_path, tmp_path, {"approved": True})
    registry.copy_release(tmp_path / "snapshot", tmp_path)
    assert registry.artifact_id(tmp_path) == manifest["champion_id"]
    assert registry.artifact_id(tmp_path / "challenger") == manifest["candidate_id"]
    assert not registry.promote(tmp_path, [])["promoted"]


def test_release_refuses_changed_artifacts(tmp_path):
    for name in registry.PAIR:
        (tmp_path / name).write_bytes(b"old")
    registry.copy_release(tmp_path, tmp_path / "snapshot")
    registry.stage(tmp_path, tmp_path, {})
    (tmp_path / "challenger/final_model.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="no longer match"):
        registry.promote(tmp_path, [])


def test_successful_release_and_automatic_rollback_preserve_exact_artifacts(tmp_path):
    for name in registry.PAIR:
        (tmp_path / name).write_bytes(b"production")
    registry.copy_release(tmp_path, tmp_path / "snapshot")
    for name in registry.PAIR:
        (tmp_path / name).write_bytes(b"candidate")
    manifest = registry.stage(tmp_path, tmp_path, {})
    registry.copy_release(tmp_path / "snapshot", tmp_path)
    rows = observations()
    for row in rows:
        row["payload"]["champion_id"] = manifest["champion_id"]
        if row["strategy"] == "candidate:new": row["strategy"] = "candidate:" + manifest["candidate_id"]
    assert registry.promote(tmp_path, rows)["promoted"]
    assert registry.artifact_id(tmp_path) == manifest["candidate_id"]
    rows = observations(candidate_gain=.008)
    for row in rows:
        row["payload"]["champion_id"] = manifest["candidate_id"]
        if row["strategy"] == "candidate:new": row["strategy"] = "rollback:" + manifest["champion_id"]
    assert registry.rollback_if_degraded(tmp_path, rows)["rolled_back"]
    assert registry.artifact_id(tmp_path) == manifest["champion_id"]
