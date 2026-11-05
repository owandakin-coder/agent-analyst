import numpy as np
import pandas as pd
import pytest

from research_agents import ResearchAgentPanel


def test_thousands_are_distinct_bounded_variants_with_opposing_families():
    panel = ResearchAgentPanel(3000)
    specs = set(zip(panel._families, panel._windows, panel._thresholds))
    assert len(specs) == 3000
    result = panel.evaluate({"AAPL": pd.DataFrame({"close": np.linspace(100, 200, 200)})})
    assert result["execution_authority"] is False
    ticker = result["tickers"]["AAPL"]
    assert ticker["evaluated"] == 3000
    assert sum(sum(v.values()) for v in ticker["families"].values()) == 3000
    assert ticker["families"]["momentum"]["buy"] > 0
    assert ticker["families"]["mean_reversion"]["sell"] > 0


@pytest.mark.parametrize("count", [-1, 10001, True, 1.5, "3000"])
def test_invalid_resource_limits_are_rejected(count):
    with pytest.raises(ValueError):
        ResearchAgentPanel(count)


@pytest.mark.parametrize("prices", [[100.0] * 199, [100.0] * 199 + [np.nan],
                                  [100.0] * 199 + [0], [100.0] * 199 + [np.inf]])
def test_missing_or_invalid_history_produces_no_votes(prices):
    result = ResearchAgentPanel(3000).evaluate({"AAPL": pd.DataFrame({"close": prices})})
    assert result["tickers"]["AAPL"]["evaluated"] == 0


def test_disabled_panel_and_flat_history():
    frame = {"AAPL": pd.DataFrame({"close": [100.0] * 200})}
    assert ResearchAgentPanel(0).evaluate(frame)["tickers"] == {}
    families = ResearchAgentPanel(10000).evaluate(frame)["tickers"]["AAPL"]["families"]
    assert sum(v["hold"] for v in families.values()) == 10000
