from pathlib import Path

import control_plane as cp


def test_default_control_state_allows_trading(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "LOCAL_CONTROL_STATE_FILE", tmp_path / "control_state.local.json")
    monkeypatch.setenv("GITHUB_REPOSITORY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("ATZMA_ALLOW_LOCAL_CONTROL_FALLBACK", "1")
    monkeypatch.setenv("ATZMA_FAIL_CLOSED_CONTROL", "0")

    state = cp.load_control_state(prefer_remote=False)

    assert state["status"] == "running"
    assert state["trading_enabled"] is True
    assert cp.can_trade(state) == (True, None)


def test_pause_resume_stop_cycle_uses_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "LOCAL_CONTROL_STATE_FILE", tmp_path / "control_state.local.json")
    monkeypatch.setenv("GITHUB_REPOSITORY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("ATZMA_ALLOW_LOCAL_CONTROL_FALLBACK", "1")
    monkeypatch.setenv("ATZMA_FAIL_CLOSED_CONTROL", "0")

    paused = cp.apply_control_action("pause", actor="test", prefer_remote=False)
    assert paused["status"] == "paused"
    assert cp.can_trade(paused) == (False, "paused")

    resumed = cp.apply_control_action("resume", actor="test", prefer_remote=False)
    assert resumed["status"] == "running"
    assert cp.can_trade(resumed) == (True, None)

    stopped = cp.apply_control_action("stop", actor="test", prefer_remote=False)
    assert stopped["status"] == "stopped"
    assert stopped["emergency_stop"] is True
    assert cp.can_trade(stopped) == (False, "emergency_stop")


def test_local_control_state_persists(monkeypatch, tmp_path):
    state_file = tmp_path / "control_state.local.json"
    monkeypatch.setattr(cp, "LOCAL_CONTROL_STATE_FILE", state_file)
    monkeypatch.setenv("GITHUB_REPOSITORY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("ATZMA_ALLOW_LOCAL_CONTROL_FALLBACK", "1")
    monkeypatch.setenv("ATZMA_FAIL_CLOSED_CONTROL", "0")

    cp.apply_control_action("pause", actor="persist", prefer_remote=False)

    assert state_file.exists()
    loaded = cp.load_control_state(prefer_remote=False)
    assert loaded["updated_by"] == "persist"
    assert loaded["status"] == "paused"
