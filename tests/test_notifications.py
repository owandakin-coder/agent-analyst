"""
test_notifications.py
======================
בדיקות ל-notifications.py:
- אין נתיב שקט לחלוטין: אם אף ערוץ לא הצליח, מוחזר False ונרשמת אזהרה
- הצלחה בערוץ אחד מספיקה כדי להחזיר True
- היעדר קונפיגורציה (טוקן/webhook ריקים) לא זורק חריגה
"""

from __future__ import annotations

from unittest.mock import patch

import notifications


class TestNoChannelsConfigured:

    def test_returns_false_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        assert notifications.send_operator_alert("halt triggered") is False

    def test_warns_when_nothing_configured(self, monkeypatch, caplog):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        with caplog.at_level("WARNING"):
            notifications.send_operator_alert("halt triggered")

        assert any("not delivered" in record.message for record in caplog.records)


class TestTelegramChannel:

    def test_success_returns_true(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        with patch("notifications.urllib.request.urlopen") as mock_urlopen:
            assert notifications.send_operator_alert("hello") is True
            mock_urlopen.assert_called_once()

    def test_network_failure_does_not_raise(self, monkeypatch, caplog):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

        with patch("notifications.urllib.request.urlopen", side_effect=OSError("boom")):
            with caplog.at_level("WARNING"):
                result = notifications.send_operator_alert("hello")

        assert result is False
        assert any("Telegram alert failed" in record.message for record in caplog.records)


class TestDiscordFallback:

    def test_discord_success_when_telegram_unconfigured(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

        with patch("notifications.urllib.request.urlopen") as mock_urlopen:
            assert notifications.send_operator_alert("hello") is True
            mock_urlopen.assert_called_once()

    def test_true_if_at_least_one_channel_succeeds(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")

        def flaky_urlopen(request, *args, **kwargs):
            url = request.full_url if hasattr(request, "full_url") else request
            if "discord.example" in url:
                raise OSError("discord down")
            return None

        with patch("notifications.urllib.request.urlopen", side_effect=flaky_urlopen):
            assert notifications.send_operator_alert("hello") is True
