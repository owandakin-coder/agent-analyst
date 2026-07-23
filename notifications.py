"""
Operator notifications for launch-critical events.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request

log = logging.getLogger("Notifications")


def send_operator_alert(message: str, markdown: bool = True) -> bool:
    """Send a launch-critical alert to every configured channel.

    Returns True only if at least one channel is configured AND accepted the
    message, so callers (and health_check.py) can detect a fully-silent
    alerting pipeline instead of assuming delivery.
    """
    telegram_ok = _send_telegram(message, markdown=markdown)
    discord_ok = _send_discord(message)
    if not telegram_ok and not discord_ok:
        log.warning("Operator alert not delivered on any channel: %s", message)
    return telegram_ok or discord_ok


def _send_telegram(message: str, markdown: bool = True) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown" if markdown else "",
        }).encode()
        urllib.request.urlopen(url, data, timeout=10)
        return True
    except Exception as exc:
        log.warning("Telegram alert failed to send: %s", exc)
        return False


def _send_discord(message: str) -> bool:
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return False
    try:
        payload = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as exc:
        log.warning("Discord alert failed to send: %s", exc)
        return False
