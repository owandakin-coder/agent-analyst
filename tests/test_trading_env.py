"""
test_trading_env.py
====================
בדיקות ל-TradingEnvironment._execute_actions():

הממצא המרכזי: הנוסחה הישנה נירמלה תקציב לפי סכום כל האותות החיוביים —
מה שגרם למכפיל אחיד (regime multiplier, RiskManager.REDUCED) *להתבטל
מתמטית* ולא להשפיע על גודל הפוזיציה בפועל. מתועד ומאומת ב-
tests/test_walk_forward_eval.py + בדיקה ידנית: walk-forward עם regime
detection אמיתי הפיק תוצאות זהות בייט לבייט לריצה בלי regime detection.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestBudgetRespondsToUniformScaling:

    def test_scaling_all_buy_signals_down_reduces_capital_deployed(self, multi_featured):
        """This is the regression test for the bug: scale the SAME action
        vector by a uniform factor (as RiskManager.REDUCED / a regime
        multiplier would) and confirm less cash actually gets deployed —
        not identical spending due to normalization cancellation."""
        from trading_env import TradingEnvironment

        tickers = list(multi_featured.keys())
        n = len(tickers)

        env_full = TradingEnvironment(multi_featured)
        env_full.reset()
        prices = env_full._get_current_prices()
        full_action = np.full(n, 0.8, dtype=np.float32)
        env_full._execute_actions(full_action, prices)
        cash_spent_full = env_full.initial_cash - env_full.cash

        env_scaled = TradingEnvironment(multi_featured)
        env_scaled.reset()
        scaled_action = full_action * 0.5  # e.g. RiskManager.REDUCED (50%)
        env_scaled._execute_actions(scaled_action, prices)
        cash_spent_scaled = env_scaled.initial_cash - env_scaled.cash

        assert cash_spent_scaled < cash_spent_full * 0.75, (
            f"Scaling all buy signals by 0.5x should meaningfully reduce cash "
            f"deployed (full={cash_spent_full:.2f}, scaled={cash_spent_scaled:.2f}) "
            f"— if these are close, the cancellation bug is back."
        )

    def test_budget_is_bounded_by_available_cash(self, multi_featured):
        """Even at max conviction on every ticker, total spend can't exceed cash."""
        from trading_env import TradingEnvironment

        tickers = list(multi_featured.keys())
        env = TradingEnvironment(multi_featured)
        env.reset()
        prices = env._get_current_prices()
        max_action = np.full(len(tickers), 1.0, dtype=np.float32)
        env._execute_actions(max_action, prices)
        assert env.cash >= -1e-6  # never goes negative

    def test_full_conviction_deploys_most_of_the_cash(self, multi_featured):
        """Sanity check the fix didn't overcorrect into starving normal,
        strong-signal trading of capital."""
        from trading_env import TradingEnvironment

        tickers = list(multi_featured.keys())
        env = TradingEnvironment(multi_featured)
        env.reset()
        prices = env._get_current_prices()
        max_action = np.full(len(tickers), 1.0, dtype=np.float32)
        env._execute_actions(max_action, prices)
        cash_spent = env.initial_cash - env.cash
        assert cash_spent > env.initial_cash * 0.5
