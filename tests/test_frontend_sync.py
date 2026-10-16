from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.parametrize("filename", ["index.html", "workspace.css", "workspace-navigation.js"])
def test_root_and_dashboard_html_are_identical(filename):
    root_html = Path(filename).read_bytes()
    app_html = (Path("dashboard_app") / filename).read_bytes()
    assert root_html == app_html
