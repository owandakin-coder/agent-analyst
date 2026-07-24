"""
Cloud-safe entrypoint for the ATZMA persistent execution worker.

Used by Render or any always-on worker host. It validates critical runtime
dependencies before entering the queue loop so deployment failures are loud and
obvious instead of failing later in the execution path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from config_loader import CFG
from logging_setup import configure_logging
from user_execution_worker import worker_loop


def _required_env() -> list[str]:
    return [
        "ATZMA_REMOTE_API_BASE",
        "ATZMA_WORKER_SHARED_TOKEN",
    ]


def _missing_env() -> list[str]:
    missing: list[str] = []
    for key in _required_env():
        if not os.getenv(key, "").strip():
            missing.append(key)
    return missing


def _required_model_paths() -> list[Path]:
    model_dir = Path(CFG.model_dir)
    return [
        model_dir / "final_model.zip",
        model_dir / "vec_normalize.pkl",
    ]


def _missing_model_paths() -> list[str]:
    return [str(path) for path in _required_model_paths() if not path.exists()]


def main() -> int:
    configure_logging()

    missing_env = _missing_env()
    if missing_env:
        print(
            f"[RenderWorker] Missing required environment variables: {', '.join(missing_env)}",
            file=sys.stderr,
        )
        return 2

    missing_models = _missing_model_paths()
    if missing_models:
        print(
            f"[RenderWorker] Missing required model artifacts: {', '.join(missing_models)}",
            file=sys.stderr,
        )
        return 3

    worker_id = os.getenv("ATZMA_WORKER_ID", "").strip() or None
    poll_seconds = int(os.getenv("ATZMA_WORKER_POLL_SECONDS", "10"))
    lease_seconds = int(os.getenv("ATZMA_WORKER_LEASE_SECONDS", "90"))

    return worker_loop(
        worker_id=worker_id,
        poll_seconds=max(poll_seconds, 1),
        lease_seconds=max(lease_seconds, 30),
    )


if __name__ == "__main__":
    raise SystemExit(main())
