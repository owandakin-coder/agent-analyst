"""
config_loader.py
================
טוען את config.yaml ומספק גישה נוחה לכל הפרמטרים.
שימוש: from config_loader import CFG
"""

from __future__ import annotations
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load() -> dict:
    try:
        import yaml
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("Install PyYAML:  pip install pyyaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}")


class Config:
    """Wrapper נוח סביב dict הקונפיגורציה."""

    def __init__(self, data: dict):
        self._data = data

    # ── Universe ──────────────────────────────────────────────────
    @property
    def tickers(self) -> list[str]:
        return self._data["universe"]["tickers"]

    @property
    def benchmark(self) -> str:
        return self._data["universe"]["benchmark"]

    # ── Periods ───────────────────────────────────────────────────
    @property
    def data_start(self) -> str:
        return self._data["periods"]["data_start"]

    @property
    def data_end(self) -> str:
        return self._data["periods"]["data_end"]

    @property
    def train_start(self) -> str:
        return self._data["periods"]["train_start"]

    @property
    def train_end(self) -> str:
        return self._data["periods"]["train_end"]

    @property
    def val_start(self) -> str:
        return self._data["periods"]["val_start"]

    @property
    def val_end(self) -> str:
        return self._data["periods"]["val_end"]

    @property
    def test_start(self) -> str:
        return self._data["periods"]["test_start"]

    @property
    def test_end(self) -> str:
        return self._data["periods"]["test_end"]

    # ── Capital ───────────────────────────────────────────────────
    @property
    def initial_capital(self) -> float:
        return float(self._data["capital"]["initial"])

    # ── Training ──────────────────────────────────────────────────
    @property
    def timesteps(self) -> int:
        return int(self._data["training"]["timesteps"])

    @property
    def optuna_trials(self) -> int:
        return int(self._data["training"]["optuna_trials"])

    @property
    def ensemble_seeds(self) -> list[int]:
        return self._data["training"]["ensemble_seeds"]

    @property
    def window_size(self) -> int:
        return int(self._data["training"]["window_size"])

    # ── Risk ──────────────────────────────────────────────────────
    @property
    def drawdown_reduce(self) -> float:
        return float(self._data["risk"]["drawdown_reduce"])

    @property
    def drawdown_halt(self) -> float:
        return float(self._data["risk"]["drawdown_halt"])

    @property
    def kelly_fraction(self) -> float:
        return float(self._data["risk"]["kelly_fraction"])

    @property
    def kelly_min(self) -> float:
        return float(self._data["risk"]["kelly_min"])

    @property
    def kelly_max(self) -> float:
        return float(self._data["risk"]["kelly_max"])

    @property
    def corr_threshold(self) -> float:
        return float(self._data["risk"]["corr_threshold"])

    @property
    def corr_window(self) -> int:
        return int(self._data["risk"]["corr_window"])

    # ── Costs ─────────────────────────────────────────────────────
    @property
    def commission_pct(self) -> float:
        return float(self._data["costs"]["commission_pct"])

    @property
    def slippage_pct(self) -> float:
        return float(self._data["costs"]["slippage_pct"])

    # ── Paths ─────────────────────────────────────────────────────
    @property
    def model_dir(self) -> str:
        return self._data["paths"]["models"]

    @property
    def results_dir(self) -> str:
        return self._data["paths"]["results"]

    @property
    def logs_dir(self) -> str:
        return self._data["paths"]["logs"]

    @property
    def plots_dir(self) -> str:
        return self._data["paths"]["plots"]

    # ── Walk-Forward ──────────────────────────────────────────────
    @property
    def wf_n_windows(self) -> int:
        return int(self._data["walk_forward"]["n_windows"])

    @property
    def wf_train_months(self) -> int:
        return int(self._data["walk_forward"]["train_months"])

    @property
    def wf_test_months(self) -> int:
        return int(self._data["walk_forward"]["test_months"])

    # ── Raw access ────────────────────────────────────────────────
    def get(self, *keys, default=None):
        """גישה עמוקה: cfg.get('risk','kelly_fraction')"""
        d = self._data
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
        return d


# Singleton
CFG = Config(_load())


if __name__ == "__main__":
    print("Config loaded successfully:")
    print(f"  Tickers    : {CFG.tickers}")
    print(f"  Train      : {CFG.train_start} → {CFG.train_end}")
    print(f"  Test       : {CFG.test_start}  → {CFG.test_end}")
    print(f"  Capital    : ${CFG.initial_capital:,.0f}")
    print(f"  Kelly      : {CFG.kelly_fraction}×")
    print(f"  DD Halt    : {CFG.drawdown_halt:.0%}")
