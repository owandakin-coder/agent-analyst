from __future__ import annotations

from pathlib import Path


def test_root_and_dashboard_html_are_identical():
    root_html = Path("index.html").read_bytes()
    app_html = Path("dashboard_app/index.html").read_bytes()
    assert root_html == app_html
