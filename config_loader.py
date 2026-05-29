"""
Load config.yaml and expose a typed config wrapper.
"""

from __future__ import annotations

from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load() -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML: pip install pyyaml") from exc

    try:
        with open(_CONFIG_PATH, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}") from exc

    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping")
    return data


class Config:
    def __init__(self, data: dict):
        self._data = data

    def get(self, *keys, default=None):
        current = self._data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
        return current

    @property
    def tickers(self) -> list[str]:
        return list(self._data["universe"]["tickers"])

    @property
    def benchmark(self) -> str:
        return str(self._data["universe"]["benchmark"])

    @property
    def data_start(self) -> str:
        return str(self._data["periods"]["data_start"])

    @property
    def data_end(self) -> str:
        return str(self._data["periods"]["data_end"])

    @property
    def train_start(self) -> str:
        return str(self._data["periods"]["train_start"])

    @property
    def train_end(self) -> str:
        return str(self._data["periods"]["train_end"])

    @property
    def val_start(self) -> str:
        return str(self._data["periods"]["val_start"])

    @property
    def val_end(self) -> str:
        return str(self._data["periods"]["val_end"])

    @property
    def test_start(self) -> str:
        return str(self._data["periods"]["test_start"])

    @property
    def test_end(self) -> str:
        return str(self._data["periods"]["test_end"])

    @property
    def initial_capital(self) -> float:
        return float(self._data["capital"]["initial"])

    @property
    def min_trade_value(self) -> float:
        return float(self._data["capital"]["min_trade_value"])

    @property
    def timesteps(self) -> int:
        return int(self._data["training"]["timesteps"])

    @property
    def trial_timesteps(self) -> int:
        return int(self._data["training"]["trial_timesteps"])

    @property
    def optuna_trials(self) -> int:
        return int(self._data["training"]["optuna_trials"])

    @property
    def ensemble_seeds(self) -> list[int]:
        return list(self._data["training"]["ensemble_seeds"])

    @property
    def ensemble_timesteps(self) -> int:
        return int(self._data["training"]["ensemble_timesteps"])

    @property
    def window_size(self) -> int:
        return int(self._data["training"]["window_size"])

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

    @property
    def commission_pct(self) -> float:
        return float(self._data["costs"]["commission_pct"])

    @property
    def slippage_pct(self) -> float:
        return float(self._data["costs"]["slippage_pct"])

    @property
    def live_min_data_days(self) -> int:
        return int(self._data["live"]["min_data_days"])

    @property
    def live_buy_threshold(self) -> float:
        return float(self._data["live"]["buy_threshold"])

    @property
    def live_sell_threshold(self) -> float:
        return float(self._data["live"]["sell_threshold"])

    @property
    def live_max_concentration(self) -> float:
        return float(self._data["live"]["max_concentration"])

    @property
    def live_cash_buffer_pct(self) -> float:
        return float(self._data["live"]["cash_buffer_pct"])

    @property
    def live_stop_loss_pct(self) -> float:
        return float(self._data["live"]["stop_loss_pct"])

    @property
    def live_price_min(self) -> float:
        return float(self._data["live"]["price_min"])

    @property
    def live_price_max(self) -> float:
        return float(self._data["live"]["price_max"])

    @property
    def live_poll_seconds(self) -> int:
        return int(self._data["live"]["poll_seconds"])

    @property
    def live_market_open_buffer_minutes(self) -> int:
        return int(self._data["live"]["market_open_buffer_minutes"])

    @property
    def broker_duplicate_window_days(self) -> int:
        return int(self._data["broker"]["duplicate_window_days"])

    @property
    def broker_recent_orders_limit(self) -> int:
        return int(self._data["broker"]["recent_orders_limit"])

    @property
    def broker_submitted_orders_file(self) -> str:
        return str(self._data["broker"]["submitted_orders_file"])

    @property
    def model_dir(self) -> str:
        return str(self._data["paths"]["models"])

    @property
    def results_dir(self) -> str:
        return str(self._data["paths"]["results"])

    @property
    def logs_dir(self) -> str:
        return str(self._data["paths"]["logs"])

    @property
    def plots_dir(self) -> str:
        return str(self._data["paths"]["plots"])

    @property
    def wf_n_windows(self) -> int:
        return int(self._data["walk_forward"]["n_windows"])

    @property
    def wf_train_months(self) -> int:
        return int(self._data["walk_forward"]["train_months"])

    @property
    def wf_test_months(self) -> int:
        return int(self._data["walk_forward"]["test_months"])

    @property
    def github_repo(self) -> str:
        return str(self.get("github", "repo", default=""))

    @property
    def github_branch(self) -> str:
        return str(self.get("github", "branch", default="main"))

    @property
    def github_control_state_path(self) -> str:
        return str(self.get("github", "control_state_path", default="runtime/control_state.json"))

    @property
    def github_trade_workflow(self) -> str:
        return str(self.get("github", "trade_workflow", default="trade.yml"))


CFG = Config(_load())


if __name__ == "__main__":
    print("Config loaded successfully:")
    print(f"  Tickers : {CFG.tickers}")
    print(f"  Train   : {CFG.train_start} -> {CFG.train_end}")
    print(f"  Test    : {CFG.test_start} -> {CFG.test_end}")
    print(f"  Capital : ${CFG.initial_capital:,.0f}")
