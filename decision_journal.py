"""
Persistent decision snapshots for explainability and dashboard inspection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
LAST_DECISION_FILE = RUNTIME_DIR / "last_decision.json"


def write_last_decision(payload: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    enriched = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    LAST_DECISION_FILE.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def read_last_decision() -> dict[str, Any] | None:
    if not LAST_DECISION_FILE.exists():
        return None
    try:
        return json.loads(LAST_DECISION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
