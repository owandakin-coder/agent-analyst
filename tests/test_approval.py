"""
test_approval.py
================
בדיקות לזרימת האישור:
- auto_approve=False → input() נקרא, טלגרם נשלח (אם מוגדר)
- auto_approve=True  → ללא input(), הפקודה מבוצעת אוטומטית
- דחייה → submit_order לא נקרא
"""

from unittest.mock import MagicMock, patch, call
import pytest


class TestApprovalFlow:

    # ── auto_approve=False ────────────────────────────────────────────────────

    def test_approve_false_calls_input(self, broker, monkeypatch):
        """כשauto_approve=False, הפונקציה קוראת ל-input()."""
        broker.auto_approve = False
        calls = []
        monkeypatch.setattr("builtins.input", lambda _: calls.append(True) or "n")

        broker._request_approval(
            {"side": "BUY", "ticker": "AAPL", "shares": 5, "price": 150.0, "time": "now"}
        )
        assert len(calls) == 1, "input() חייב להיקרא בדיוק פעם אחת"

    def test_approve_false_yes_returns_true(self, broker, monkeypatch):
        """'y' → מאושר."""
        broker.auto_approve = False
        monkeypatch.setattr("builtins.input", lambda _: "y")
        result = broker._request_approval(
            {"side": "BUY", "ticker": "AAPL", "shares": 5, "price": 150.0, "time": "now"}
        )
        assert result is True

    def test_approve_false_no_returns_false(self, broker, monkeypatch):
        """'n' → נדחה."""
        broker.auto_approve = False
        monkeypatch.setattr("builtins.input", lambda _: "n")
        result = broker._request_approval(
            {"side": "BUY", "ticker": "AAPL", "shares": 5, "price": 150.0, "time": "now"}
        )
        assert result is False

    def test_approve_false_empty_string_rejected(self, broker, monkeypatch):
        """Enter ריק → נדחה (ברירת מחדל N)."""
        broker.auto_approve = False
        monkeypatch.setattr("builtins.input", lambda _: "")
        result = broker._request_approval(
            {"side": "BUY", "ticker": "AAPL", "shares": 5, "price": 150.0, "time": "now"}
        )
        assert result is False

    def test_approve_false_eoferror_rejected(self, broker, monkeypatch):
        """EOFError (ללא טרמינל, כגון CI) → נדחה בשקט."""
        broker.auto_approve = False

        def raise_eof(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        result = broker._request_approval(
            {"side": "BUY", "ticker": "AAPL", "shares": 5, "price": 150.0, "time": "now"}
        )
        assert result is False

    # ── auto_approve=True ─────────────────────────────────────────────────────

    def test_approve_true_skips_input(self, broker, monkeypatch):
        """כשauto_approve=True, input() לא נקרא כלל."""
        broker.auto_approve = True
        called = []
        monkeypatch.setattr("builtins.input", lambda _: called.append(1) or "n")

        result = broker._request_approval(
            {"side": "BUY", "ticker": "AAPL", "shares": 5, "price": 150.0, "time": "now"}
        )
        assert result is True
        assert len(called) == 0, "input() לא אמור להיקרא כשauto_approve=True"

    def test_approve_true_order_submitted(self, broker):
        """auto_approve=True → submit_order נקרא."""
        broker.auto_approve = True
        broker.buy("AAPL", shares=3, price=150.0)
        broker._trading.submit_order.assert_called_once()

    def test_approve_false_no_submission(self, broker, monkeypatch):
        """auto_approve=False + 'n' → submit_order לא נקרא."""
        broker.auto_approve = False
        monkeypatch.setattr("builtins.input", lambda _: "n")
        broker.buy("AAPL", shares=3, price=150.0)
        broker._trading.submit_order.assert_not_called()

    # ── Telegram ──────────────────────────────────────────────────────────────

    def test_telegram_sent_when_configured(self, broker, monkeypatch):
        """כשTELEGRAM_BOT_TOKEN ו-CHAT_ID מוגדרים, נשלחת בקשת HTTP."""
        broker.auto_approve = False
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID",   "123456")
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch("urllib.request.urlopen") as mock_url:
            broker._request_approval(
                {"side": "BUY", "ticker": "AAPL", "shares": 5, "price": 150.0, "time": "now"}
            )
            mock_url.assert_called_once()
            url_called = mock_url.call_args[0][0]
            assert "fake_token" in url_called

    def test_telegram_not_sent_when_not_configured(self, broker, monkeypatch):
        """ללא פרטי טלגרם, urlopen לא נקרא."""
        broker.auto_approve = False
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID",   raising=False)
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with patch("urllib.request.urlopen") as mock_url:
            broker._request_approval(
                {"side": "SELL", "ticker": "MSFT", "shares": 2, "price": 300.0, "time": "now"}
            )
            mock_url.assert_not_called()

    def test_telegram_failure_does_not_crash(self, broker, monkeypatch):
        """כישלון בשליחת טלגרם לא קורס את תהליך האישור."""
        broker.auto_approve = False
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID",   "123")
        monkeypatch.setattr("builtins.input", lambda _: "y")

        with patch("urllib.request.urlopen", side_effect=Exception("network fail")):
            result = broker._request_approval(
                {"side": "BUY", "ticker": "AAPL", "shares": 1, "price": 100.0, "time": "now"}
            )
        # למרות כישלון טלגרם, תהליך האישור ממשיך
        assert result is True
