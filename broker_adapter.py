"""
Normalized broker adapter interfaces for ATZMA execution workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BrokerSessionConfig:
    broker_name: str
    trading_mode: str
    base_url: str
    api_key: str
    secret_key: str


class BrokerAdapter(Protocol):
    broker_name: str

    def get_account(self) -> dict:
        ...

    def reconcile_account_state(self) -> dict:
        ...

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        ...

    def get_positions(self) -> dict[str, float]:
        ...

    def is_market_open(self) -> bool:
        ...

    def next_market_open(self):
        ...

    def buy(self, ticker: str, shares: float, price: float | None = None) -> dict:
        ...

    def sell(self, ticker: str, shares: float, price: float | None = None) -> dict:
        ...

    def hold(self, ticker: str):
        ...
