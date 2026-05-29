"""
test_config.py
==============
בדיקות ל-config_loader — ודא שהקונפיגורציה נטענת ותקינה.
"""

import pytest
from config_loader import CFG


class TestConfigValues:

    def test_tickers_not_empty(self):
        assert len(CFG.tickers) > 0

    def test_benchmark_in_tickers(self):
        assert CFG.benchmark in CFG.tickers, \
            f"Benchmark '{CFG.benchmark}' must be in tickers list"

    def test_capital_positive(self):
        assert CFG.initial_capital > 0

    def test_kelly_fraction_valid(self):
        assert 0 < CFG.kelly_fraction <= 1.0

    def test_drawdown_hierarchy(self):
        """drawdown_halt חייב להיות גדול מ-drawdown_reduce."""
        assert CFG.drawdown_reduce < CFG.drawdown_halt, \
            "drawdown_reduce must be < drawdown_halt"

    def test_train_before_val(self):
        assert CFG.train_end < CFG.val_start, \
            "train_end must be before val_start"

    def test_val_before_test(self):
        assert CFG.val_end < CFG.test_start, \
            "val_end must be before test_start"

    def test_no_train_test_overlap(self):
        assert CFG.train_end < CFG.test_start, \
            "Train and test periods must not overlap"

    def test_window_size_positive(self):
        assert CFG.window_size > 0

    def test_commission_reasonable(self):
        """עמלה בין 0% ל-1%."""
        assert 0 <= CFG.commission_pct <= 0.01

    def test_slippage_less_than_commission(self):
        """slippage בד"כ קטן מהעמלה."""
        assert CFG.slippage_pct <= CFG.commission_pct

    def test_ensemble_seeds_unique(self):
        seeds = CFG.ensemble_seeds
        assert len(seeds) == len(set(seeds)), "Ensemble seeds must be unique"

    def test_wf_windows_positive(self):
        assert CFG.wf_n_windows >= 1

    def test_wf_train_longer_than_test(self):
        assert CFG.wf_train_months > CFG.wf_test_months


class TestConfigGet:

    def test_get_nested_value(self):
        val = CFG.get("risk", "kelly_fraction")
        assert val == CFG.kelly_fraction

    def test_get_missing_returns_default(self):
        val = CFG.get("nonexistent", "key", default=42)
        assert val == 42

    def test_get_deep_missing(self):
        val = CFG.get("risk", "nonexistent_key", default="fallback")
        assert val == "fallback"


class TestConfigConsistency:
    """בדיקות עקביות בין הפרמטרים השונים."""

    def test_risk_manager_uses_config(self):
        """RiskManager טוען את הערכים מ-CFG."""
        from risk_manager import DRAWDOWN_REDUCE, DRAWDOWN_HALT, KELLY_FRACTION
        assert DRAWDOWN_REDUCE == pytest.approx(CFG.drawdown_reduce)
        assert DRAWDOWN_HALT   == pytest.approx(CFG.drawdown_halt)
        assert KELLY_FRACTION  == pytest.approx(CFG.kelly_fraction)

    def test_trading_env_uses_config(self):
        """TradingEnvironment טוען את הערכים מ-CFG."""
        from trading_env import COMMISSION_RATE, SLIPPAGE_RATE, WINDOW_SIZE
        assert COMMISSION_RATE == pytest.approx(CFG.commission_pct)
        assert SLIPPAGE_RATE   == pytest.approx(CFG.slippage_pct)
        assert WINDOW_SIZE     == CFG.window_size
