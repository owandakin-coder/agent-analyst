import numpy as np
import pandas as pd

from walk_forward_eval import _annualized_return, _build_summary, evaluate_window


def _make_ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


def test_evaluate_window_runs_with_transformer_policy():
    from data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    raw = {t: dm._compute_features(_make_ohlcv(220, seed=ord(t[0])), t) for t in ("AAPL", "MSFT")}
    idx = raw["AAPL"].index
    split = int(len(idx) * 0.7)

    window = {
        "window": 1,
        "train_start": idx[0].strftime("%Y-%m-%d"),
        "train_end": idx[split].strftime("%Y-%m-%d"),
        "test_start": idx[split + 1].strftime("%Y-%m-%d"),
        "test_end": idx[-1].strftime("%Y-%m-%d"),
    }

    result = evaluate_window(
        window, raw, timesteps=256, seed=0,
        model_params={"n_steps": 64, "batch_size": 32, "n_epochs": 2},
        use_transformer_policy=True,
    )

    assert result["skipped"] is False
    assert "sharpe" in result and "total_return" in result


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
