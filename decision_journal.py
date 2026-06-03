"""
Persistent decision snapshots for explainability and dashboard inspection.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _runtime_dir() -> Path:
    return Path(os.getenv("ATZMA_RUNTIME_DIR", Path(__file__).resolve().parent / "runtime"))


def _last_decision_file() -> Path:
    return _runtime_dir() / "last_decision.json"


def write_last_decision(payload: dict[str, Any]) -> None:
    runtime_dir = _runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    enriched = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    _last_decision_file().write_text(
        json.dumps(enriched, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def read_last_decision() -> dict[str, Any] | None:
    last_decision_file = _last_decision_file()
    if not last_decision_file.exists():
        return None
    try:
        return json.loads(last_decision_file.read_text(encoding="utf-8"))
    except Exception:
        return None
