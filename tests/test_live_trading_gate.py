"""
test_live_trading_gate.py
==========================
בדיקות לשער ATZMA_LIVE_TRADING_ENABLED ב-main.py.step_live_real:
- ללא המשתנה — יוצא מיד (SystemExit(1)), בלי להגיע בכלל לבקשת האישור
- עם המשתנה — עובר את השער ומגיע לשלב הבא (בקשת 'I UNDERSTAND')
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import main


class TestLiveTradingGate:

    def test_blocked_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("ATZMA_LIVE_TRADING_ENABLED", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            main.step_live_real(model=None, vec_norm=None, auto_approve=True)

        assert exc_info.value.code == 1

    def test_blocked_when_env_var_falsy(self, monkeypatch):
        monkeypatch.setenv("ATZMA_LIVE_TRADING_ENABLED", "0")

        with pytest.raises(SystemExit) as exc_info:
            main.step_live_real(model=None, vec_norm=None, auto_approve=True)

        assert exc_info.value.code == 1

    def test_passes_gate_and_reaches_confirmation_when_enabled(self, monkeypatch):
        monkeypatch.setenv("ATZMA_LIVE_TRADING_ENABLED", "1")

        # If the gate is bypassed correctly, execution reaches the
        # "Type 'I UNDERSTAND'" prompt next; declining it exits with code 0
        # (not 1), which proves we got past the gate.
        with patch("builtins.input", return_value="no thanks"):
            with pytest.raises(SystemExit) as exc_info:
                main.step_live_real(model=None, vec_norm=None, auto_approve=True)

        assert exc_info.value.code == 0
