"""Generate training-only rolling windows ending before today's session.

Does not change the committed runtime configuration. CI uses this temporary
configuration to train on recent data, reserving separate validation and test years.
"""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import yaml


def periods(today):
    test_start = today - timedelta(days=365)
    val_start = test_start - timedelta(days=365)
    return {"data_start": "2015-01-01", "data_end": today.isoformat(),
            "train_start": "2015-01-01", "train_end": (val_start - timedelta(days=1)).isoformat(),
            "val_start": val_start.isoformat(), "val_end": (test_start - timedelta(days=1)).isoformat(),
            "test_start": test_start.isoformat(), "test_end": (today - timedelta(days=1)).isoformat()}


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["periods"] = periods(datetime.now(ZoneInfo("America/New_York")).date())
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("Temporary training windows:", config["periods"])
