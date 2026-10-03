"""forward_eval: pure row building and the one-row-per-day CSV log (no network)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from forward_eval import FIELDS, append_row, build_row


def _row(run_date="2026-10-03", alpha_model=1.2, **kw):
    model = np.linspace(100_000, 100_000 * alpha_model, 60)
    spy = np.linspace(100_000, 110_000, 60)
    return build_row(model, spy, run_date=run_date, data_last_date="2026-10-02",
                     trading_days=59, members=1, git_sha="abc1234", **kw)


def test_build_row_computes_alpha_as_model_minus_spy():
    row = _row(alpha_model=1.2)
    assert row["model_return"] == pytest.approx(0.20, abs=1e-4)
    assert row["spy_return"] == pytest.approx(0.10, abs=1e-4)
    assert row["alpha"] == pytest.approx(0.10, abs=1e-4)
    assert set(row) == set(FIELDS)


def test_append_row_creates_file_with_header(tmp_path):
    path = tmp_path / "forward_log" / "perf.csv"
    append_row(path, _row())
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_date"] for r in rows] == ["2026-10-03"]


def test_append_row_keeps_history_and_replaces_same_day(tmp_path):
    path = tmp_path / "perf.csv"
    append_row(path, _row(run_date="2026-10-03", alpha_model=1.2))
    append_row(path, _row(run_date="2026-10-04", alpha_model=1.1))
    append_row(path, _row(run_date="2026-10-04", alpha_model=1.3))
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_date"] for r in rows] == ["2026-10-03", "2026-10-04"]
    assert float(rows[1]["model_return"]) == pytest.approx(0.30, abs=1e-4)
