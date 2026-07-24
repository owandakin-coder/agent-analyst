import numpy as np
import pandas as pd

from training_pipeline import TrainingPipeline, score_validation_metrics


def test_score_validation_metrics_rewards_better_risk_return_profile():
    strong = {
        "annualized_return": 0.22,
        "sharpe": 1.8,
        "max_drawdown": 0.08,
        "calmar": 2.4,
    }
    weak = {
        "annualized_return": 0.11,
        "sharpe": 0.9,
        "max_drawdown": 0.19,
        "calmar": 0.7,
    }
    assert score_validation_metrics(strong) > score_validation_metrics(weak)


def test_build_validation_slices_uses_offsets_and_fallback():
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.window_size = 30

    dates = pd.date_range("2024-01-01", periods=180, freq="B")
    frame = pd.DataFrame({"close": np.linspace(100, 120, len(dates))}, index=dates)
    data = {"AAPL": frame, "MSFT": frame.copy()}

    slices = pipeline._build_validation_slices(data)
    labels = [label for label, _ in slices]
    assert "offset_0" in labels
    assert len(slices) >= 2

    short_dates = pd.date_range("2024-01-01", periods=80, freq="B")
    short_frame = pd.DataFrame({"close": np.linspace(100, 105, len(short_dates))}, index=short_dates)
    fallback = pipeline._build_validation_slices({"AAPL": short_frame})
    assert fallback == [("full_validation", {"AAPL": short_frame})]
