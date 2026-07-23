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

import numpy as np
import pytest

from promote_model import should_promote, backtest_equity_curve, metrics_from_equity


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
