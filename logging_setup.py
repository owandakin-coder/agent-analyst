"""
Shared logging setup for ATZMA launch environments.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from config_loader import CFG
    LOGS_DIR = Path(CFG.logs_dir)
except Exception:  # pragma: no cover
    LOGS_DIR = Path("logs")


def _rotating_handler(path: Path, level: int = logging.INFO) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    return handler


def configure_logging() -> None:
    if getattr(configure_logging, "_configured", False):
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(_rotating_handler(LOGS_DIR / "agent_analyst.log"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
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
