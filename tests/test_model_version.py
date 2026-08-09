"""
test_model_version.py
======================
בדיקות ל-live_trader._resolve_model_version():
- ATZMA_MODEL_VERSION מנצח תמיד אם מוגדר
- אחרת נשלף trained_at מ-training_meta.pkl
- ואם שניהם חסרים — "unknown", לא קריסה
"""

from __future__ import annotations

import pickle

import live_trader


class TestResolveModelVersion:

    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATZMA_MODEL_VERSION", "pinned-v3")
        monkeypatch.setattr(live_trader, "MODEL_DIR", str(tmp_path))
        assert live_trader._resolve_model_version() == "pinned-v3"

    def test_falls_back_to_training_meta(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ATZMA_MODEL_VERSION", raising=False)
        monkeypatch.setattr(live_trader, "MODEL_DIR", str(tmp_path))
        meta = {"trained_at": "2026-08-01T06:00:00+00:00"}
        with open(tmp_path / "training_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        assert live_trader._resolve_model_version() == "2026-08-01T06:00:00+00:00"

    def test_unknown_when_nothing_available(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ATZMA_MODEL_VERSION", raising=False)
        monkeypatch.setattr(live_trader, "MODEL_DIR", str(tmp_path))
        assert live_trader._resolve_model_version() == "unknown"

    def test_unknown_when_meta_missing_trained_at(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ATZMA_MODEL_VERSION", raising=False)
        monkeypatch.setattr(live_trader, "MODEL_DIR", str(tmp_path))
        with open(tmp_path / "training_meta.pkl", "wb") as f:
            pickle.dump({"best_params": {}}, f)
        assert live_trader._resolve_model_version() == "unknown"
