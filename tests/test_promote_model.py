"""
test_promote_model.py
======================
בדיקות לשער הקידום (promote_model.py):
- should_promote היא פונקציה טהורה, נבדקת בלי אימון/רשת
- מודל ראשון (אין production קיים) תמיד מקודם
- רגרסיה ב-Sharpe / drawdown / return מעבר לטולרנס נדחית
- שיפור או שינוי זניח מתקבל
- אינטגרציה: backtest_equity_curve/evaluate_model_metrics רצים על מודל אמיתי
  (tiny_model_and_norm) בלי לקרוס ומחזירים את כל המפתחות הנדרשים
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promote_model import (
    should_promote,
    backtest_equity_curve,
    metrics_from_equity,
    compute_spy_baseline,
    _sub_period_bounds,
    sub_period_report,
)


def _metrics(sharpe=1.0, max_drawdown=0.10, annualized_return=0.15):
    return {"sharpe": sharpe, "max_drawdown": max_drawdown, "annualized_return": annualized_return}


class TestShouldPromote:

    def test_first_model_always_promoted(self):
        promote, reason = should_promote(_metrics(), None)
        assert promote is True
        assert "no existing production model" in reason

    def test_equal_metrics_promoted(self):
        current = _metrics()
        candidate = _metrics()
        promote, _ = should_promote(candidate, current)
        assert promote is True

    def test_better_candidate_promoted(self):
        current = _metrics(sharpe=1.0, max_drawdown=0.10, annualized_return=0.15)
        candidate = _metrics(sharpe=1.5, max_drawdown=0.05, annualized_return=0.25)
        promote, _ = should_promote(candidate, current)
        assert promote is True

    def test_sharpe_regression_beyond_tolerance_rejected(self):
        current = _metrics(sharpe=1.0)
        candidate = _metrics(sharpe=0.5)
        promote, reason = should_promote(candidate, current)
        assert promote is False
        assert "sharpe" in reason

    def test_sharpe_within_tolerance_accepted(self):
        current = _metrics(sharpe=1.0)
        candidate = _metrics(sharpe=0.95)
        promote, _ = should_promote(candidate, current)
        assert promote is True

    def test_drawdown_regression_beyond_tolerance_rejected(self):
        current = _metrics(max_drawdown=0.10)
        candidate = _metrics(max_drawdown=0.20)
        promote, reason = should_promote(candidate, current)
        assert promote is False
        assert "max_drawdown" in reason

    def test_return_regression_beyond_tolerance_rejected(self):
        current = _metrics(annualized_return=0.20)
        candidate = _metrics(annualized_return=0.05)
        promote, reason = should_promote(candidate, current)
        assert promote is False
        assert "annualized_return" in reason

    def test_multiple_regressions_all_reported(self):
        current = _metrics(sharpe=1.0, max_drawdown=0.10, annualized_return=0.20)
        candidate = _metrics(sharpe=0.3, max_drawdown=0.30, annualized_return=0.0)
        promote, reason = should_promote(candidate, current)
        assert promote is False
        assert "sharpe" in reason and "max_drawdown" in reason and "annualized_return" in reason


class TestSpyBaselineGate:

    def test_badly_lagging_spy_rejected_even_if_better_than_previous(self):
        """A model that beats a bad previous model but loses money while
        the market does fine must still be rejected — this is exactly the
        'previous model was already mediocre' loophole the baseline closes."""
        previous = _metrics(sharpe=-1.0, max_drawdown=0.30, annualized_return=-0.30)
        candidate = _metrics(sharpe=-0.5, max_drawdown=0.20, annualized_return=-0.20)
        spy = _metrics(sharpe=1.0, max_drawdown=0.10, annualized_return=0.15)

        # Beats the (bad) previous model on every metric...
        promote_vs_previous_only, _ = should_promote(candidate, previous)
        assert promote_vs_previous_only is True

        # ...but still loses badly against SPY over the same window.
        promote, reason = should_promote(candidate, previous, spy)
        assert promote is False
        assert "SPY" in reason

    def test_within_spy_tolerance_accepted(self):
        spy = _metrics(annualized_return=0.15)
        candidate = _metrics(annualized_return=0.00)  # 15pp behind, within the 20pp tolerance
        promote, _ = should_promote(candidate, None, spy)
        assert promote is True

    def test_first_deployment_still_checked_against_spy(self):
        """No previous model to compare against must not mean 'anything goes' —
        a terrible first model should still fail the SPY sanity check."""
        spy = _metrics(annualized_return=0.15)
        candidate = _metrics(annualized_return=-0.20)
        promote, reason = should_promote(candidate, None, spy)
        assert promote is False
        assert "SPY" in reason

    def test_no_spy_data_available_skips_the_check(self):
        candidate = _metrics(annualized_return=-0.50)
        promote, _ = should_promote(candidate, None, spy=None)
        assert promote is True


class TestBacktestIntegration:

    def test_backtest_and_metrics_run_end_to_end(self, tiny_model_and_norm):
        model, vec_norm, raw_data, _ = tiny_model_and_norm

        equity = backtest_equity_curve(model, vec_norm, raw_data)
        assert isinstance(equity, np.ndarray)
        assert len(equity) > 1
        assert equity[0] == pytest.approx(100_000.0)

        metrics = metrics_from_equity(equity)
        for key in ("sharpe", "max_drawdown", "annualized_return", "total_return", "final_value"):
            assert key in metrics
        assert np.isfinite(metrics["sharpe"])

    def test_compute_spy_baseline_runs_end_to_end(self, tiny_model_and_norm):
        _, _, raw_data, _ = tiny_model_and_norm
        spy_data = {"SPY": raw_data["AAPL"]}

        metrics = compute_spy_baseline(spy_data)

        assert metrics is not None
        for key in ("sharpe", "max_drawdown", "annualized_return"):
            assert key in metrics

    def test_compute_spy_baseline_returns_none_without_spy(self, tiny_model_and_norm):
        _, _, raw_data, _ = tiny_model_and_norm
        assert compute_spy_baseline(raw_data) is None


class TestSubPeriodDiagnostic:
    """The sub-period breakdown (added 2026-09-17 after a walk-forward run
    showed a candidate that passed the single-window gate had no edge across
    independent windows) is diagnostic only — it must never change what
    should_promote() decides, only add visibility into whether a pass is
    consistent across the test window or driven by one stretch of it."""

    def test_sub_period_bounds_covers_the_full_range_with_no_gaps_or_overlap(self):
        bounds = _sub_period_bounds(n=3, start="2022-01-01", end="2022-12-31")
        assert len(bounds) == 3
        assert bounds[0][0] == pd.Timestamp("2022-01-01")
        assert bounds[-1][1] == pd.Timestamp("2022-12-31")
        for i in range(len(bounds) - 1):
            assert bounds[i][1] == bounds[i + 1][0]

    def test_sub_period_report_runs_end_to_end_on_synthetic_data(self, tiny_model_and_norm):
        model, vec_norm, raw_data, tmp = tiny_model_and_norm
        candidate_model = tmp / "test_model.zip"
        candidate_norm = tmp / "vec_norm.pkl"
        assert candidate_model.exists() and candidate_norm.exists()

        rows = sub_period_report(
            candidate_model, candidate_norm,
            previous_model=Path("no_such_file.zip"), previous_norm=Path("no_such_file.pkl"),
            n=2, test_data=raw_data,
        )

        assert len(rows) >= 1
        for row in rows:
            assert "start" in row and "end" in row
            assert "candidate" in row
            assert "previous" not in row  # no previous model on disk
            for key in ("sharpe", "max_drawdown", "annualized_return"):
                assert key in row["candidate"]

    def test_sub_period_report_skips_periods_with_too_little_data(self, tiny_model_and_norm):
        """Asking for far more slices than the data supports should skip
        the too-thin ones rather than crash or report on noise."""
        _, _, raw_data, tmp = tiny_model_and_norm
        candidate_model = tmp / "test_model.zip"
        candidate_norm = tmp / "vec_norm.pkl"

        rows = sub_period_report(
            candidate_model, candidate_norm,
            previous_model=Path("no_such_file.zip"), previous_norm=Path("no_such_file.pkl"),
            n=50, test_data=raw_data,
        )

        # 50 slices of a ~9-month synthetic series leaves most slices under
        # the 30-row floor — those must be dropped, not returned as noise.
        assert len(rows) < 50
