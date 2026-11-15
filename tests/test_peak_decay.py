from datetime import date
from types import SimpleNamespace

from risk_manager import RiskLevel, RiskManager
import risk_state


def _manager(half_life):
    rm = RiskManager(100_000.0)
    rm.peak_half_life_days = half_life
    return rm


def test_without_decay_a_cash_account_stays_locked_below_the_peak():
    rm = _manager(0)
    rm.decay_peak(89_000.0, date(2020, 3, 2))
    for day in range(3, 28):
        rm.decay_peak(89_000.0, date(2020, 3, day))
    assert rm.update(89_000.0) == RiskLevel.REDUCED


def test_peak_fades_toward_equity_so_buying_can_resume():
    rm = _manager(60)
    rm.decay_peak(89_000.0, date(2020, 3, 2))
    rm.decay_peak(89_000.0, date(2020, 4, 30))     # about 42 business days later
    assert 89_000.0 < rm.peak_value < 100_000.0
    assert rm.update(89_000.0) == RiskLevel.NORMAL


def test_decay_runs_once_per_new_business_day_and_never_lifts_the_peak():
    rm = _manager(60)
    rm.decay_peak(95_000.0, date(2020, 3, 2))
    rm.decay_peak(95_000.0, date(2020, 3, 2))      # same day, many 15-minute cycles
    rm.decay_peak(95_000.0, date(2020, 3, 1))      # a weekend date adds no business day
    assert rm.peak_value == 100_000.0
    rm.decay_peak(120_000.0, date(2020, 3, 10))    # equity above the peak: nothing to fade
    assert rm.peak_value == 100_000.0


def test_peak_date_survives_the_persisted_risk_state():
    rm = _manager(60)
    rm.peak_date = date(2020, 3, 2)
    trader = SimpleNamespace(
        _peak_net_worth=100_000.0, risk_manager=rm, _entry_prices={}, _trailing_highs={}, _shadow_portfolios={})
    state = risk_state.snapshot(trader)
    assert state["peak_date"] == "2020-03-02"
    fresh = SimpleNamespace(
        _peak_net_worth=1.0, risk_manager=_manager(60), _entry_prices={}, _trailing_highs={}, _shadow_portfolios={})
    risk_state.restore(fresh, state)
    assert fresh.risk_manager.peak_date == date(2020, 3, 2)
