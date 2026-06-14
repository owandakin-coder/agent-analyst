"""
Shared logging setup for ATZMA launch environments.
"""

from __future__ import annotations

import logging
import json
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

try:
    from config_loader import CFG
    LOGS_DIR = Path(CFG.logs_dir)
except Exception:  # pragma: no cover
    LOGS_DIR = Path("logs")


def _rotating_handler(path: Path, level: int = logging.INFO) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())
    return handler


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _build_formatter() -> logging.Formatter:
    if str(getattr(configure_logging, "_json_enabled", os.getenv("ATZMA_JSON_LOGS", "1"))).lower() not in {"0", "false", "no"}:
        return JsonLogFormatter()
    return logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")


def configure_logging() -> None:
    if getattr(configure_logging, "_configured", False):
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(_rotating_handler(LOGS_DIR / "agent_analyst.log"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_build_formatter())
    root.addHandler(stream_handler)

    logger_files = {
      "BrokerAPI": "trading.log",
      "LiveTrader": "trading.log",
      "RiskManager": "risk.log",
      "Auth": "auth.log",
      "Errors": "errors.log",
    }
    for name, filename in logger_files.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = True
        logger.addHandler(_rotating_handler(LOGS_DIR / filename))

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.WARNING)
    configure_logging._configured = True
