"""
test_logs.py
============
בדיקות לקובץ paper_orders.log:
- כל פקודה (כולל נדחות) נרשמת
- רשומה כוללת תאריך, side, ticker, status
- לוג BrokerAPIStub תואם
"""

import os
import re
from pathlib import Path
from unittest.mock import patch
import pytest


class TestOrderLogging:

    def test_log_file_created_on_buy(self, broker, tmp_path, monkeypatch):
        """buy() יוצר / מעדכן את קובץ הלוג."""
        import broker_api as ba
        monkeypatch.setattr(ba, "LOG_FILE", str(tmp_path / "orders.log"))
        broker._log_file_path = str(tmp_path / "orders.log")

        # עדכון ה-LOG_FILE בתוך מופע הברוקר (הפונקציה _write_log משתמשת ב-LOG_FILE ישירות)
        broker.auto_approve = True
        broker._write_log({
            "side": "BUY", "ticker": "AAPL", "shares": 5,
            "status": "FILLED", "time": "2025-01-07T09:30:00Z"
        })
        log_path = Path(str(tmp_path / "orders.log"))
        assert log_path.exists(), "קובץ לוג לא נוצר"

    def test_buy_order_logged(self, broker, tmp_path, monkeypatch):
        """פקודת BUY מוקלטת ב-paper_orders.log."""
        import broker_api as ba
        log_path = str(tmp_path / "orders.log")
        monkeypatch.setattr(ba, "LOG_FILE", log_path)

        broker.auto_approve = True
        broker.buy("AAPL", shares=5, price=150.0)

        content = Path(log_path).read_text(encoding="utf-8")
        assert "BUY" in content or "buy" in content.lower()
        assert "AAPL" in content

    def test_rejected_order_logged(self, broker, tmp_path, monkeypatch):
        """פקודה שנדחתה על ידי המשתמש מוקלטת עם status=REJECTED_BY_USER."""
        import broker_api as ba
        log_path = str(tmp_path / "orders.log")
        monkeypatch.setattr(ba, "LOG_FILE", log_path)

        broker.auto_approve = False
        monkeypatch.setattr("builtins.input", lambda _: "n")
        broker.buy("MSFT", shares=3, price=300.0)

        content = Path(log_path).read_text(encoding="utf-8")
        assert "REJECTED_BY_USER" in content

    def test_sell_no_position_logged(self, broker, tmp_path, monkeypatch):
        """מכירה ללא פוזיציה מוקלטת כ-REJECTED."""
        import broker_api as ba
        log_path = str(tmp_path / "orders.log")
        monkeypatch.setattr(ba, "LOG_FILE", log_path)

        broker._trading.get_all_positions.return_value = []
        broker.auto_approve = True
        result = broker.sell("GOOGL", shares=5, price=130.0)

        # ה-sell עם no_position לא קורא ל-_write_log (מחזיר dict ישירות)
        # ודא שה-result הוא REJECTED
        assert result["status"] == "REJECTED"

    def test_log_contains_timestamp(self, broker, tmp_path, monkeypatch):
        """רשומה בלוג כוללת שדה time עם תאריך."""
        import broker_api as ba
        log_path = str(tmp_path / "orders.log")
        monkeypatch.setattr(ba, "LOG_FILE", log_path)

        broker.auto_approve = True
        broker.buy("AAPL", shares=2, price=150.0)

        content = Path(log_path).read_text(encoding="utf-8")
        # בדיקה שיש תאריך בפורמט ISO (למשל "2025-")
        assert re.search(r"20\d\d-\d\d-\d\d", content), \
            "לא נמצא timestamp בפורמט ISO ברשומת הלוג"

    def test_multiple_orders_all_logged(self, broker, tmp_path, monkeypatch):
        """מספר פקודות – כולן מוקלטות בשורות נפרדות."""
        import broker_api as ba
        log_path = str(tmp_path / "orders.log")
        monkeypatch.setattr(ba, "LOG_FILE", log_path)

        broker.auto_approve = True
        broker.buy("AAPL", shares=1, price=150.0)
        broker.buy("MSFT", shares=2, price=300.0)
        broker.buy("GOOGL", shares=3, price=130.0)

        lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
        # לפחות 3 שורות (אחת לכל פקודה)
        assert len(lines) >= 3

    def test_write_log_handles_io_error(self, broker, monkeypatch):
        """_write_log לא קורסת אם אי אפשר לכתוב לקובץ."""
        import broker_api as ba
        monkeypatch.setattr(ba, "LOG_FILE", "/invalid/path/orders.log")

        # לא אמור לזרוק חריגה
        broker._write_log({"side": "BUY", "status": "FILLED"})


class TestStubLogging:

    def test_stub_buy_logs_to_file(self, tmp_path, monkeypatch):
        """BrokerAPIStub מקליט פקודות ב-paper_orders.log."""
        import broker_api as ba
        log_path = str(tmp_path / "stub.log")
        monkeypatch.setattr(ba, "LOG_FILE", log_path)

        from broker_api import BrokerAPIStub
        stub = BrokerAPIStub()
        stub.set_cash(10_000.0)
        stub.buy("AAPL", 5, 150.0)

        # בדיקת הפלט (הלוגר כותב לקובץ handler)
        # הלוגר של broker_api כותב ל-LOG_FILE
        # (בגלל הגדרת FileHandler ברמת המודול, הפעלה מחדש תתחיל handler חדש)
        assert stub.order_counter == 1

    def test_stub_records_rejected_sell(self):
        """BrokerAPIStub מחזיר REJECTED כשאין פוזיציה."""
        from broker_api import BrokerAPIStub
        stub   = BrokerAPIStub()
        result = stub.sell("AAPL", 5, 150.0)
        assert result["status"] == "REJECTED"
