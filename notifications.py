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


def send_operator_alert(message: str, markdown: bool = True) -> None:
    _send_telegram(message, markdown=markdown)
    _send_discord(message)


def _send_telegram(message: str, markdown: bool = True) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown" if markdown else "",
        }).encode()
        urllib.request.urlopen(url, data, timeout=10)
    except Exception as exc:
        log.debug("Telegram send failed: %s", exc)


def _send_discord(message: str) -> None:
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    try:
        payload = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        log.debug("Discord webhook send failed: %s", exc)
