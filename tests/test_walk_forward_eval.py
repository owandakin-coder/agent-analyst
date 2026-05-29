from walk_forward_eval import _annualized_return, _build_summary


def test_annualized_return_positive():
    result = _annualized_return(0.10, 252)
    assert 0.09 < result < 0.11


def test_build_summary_counts_positive_windows():
    summary = _build_summary(
        [
            {
                "total_return": 0.10,
                "annualized_return": 0.10,
                "alpha": 0.02,
                "sharpe": 1.2,
                "calmar": 0.8,
                "max_drawdown": 0.05,
            },
            {
                "total_return": -0.02,
                "annualized_return": -0.02,
                "alpha": -0.01,
                "sharpe": 0.4,
                "calmar": -0.2,
                "max_drawdown": 0.07,
            },
        ]
    )
    assert summary["windows_evaluated"] == 2
    assert summary["positive_alpha_windows"] == 1
    assert summary["positive_return_windows"] == 1
