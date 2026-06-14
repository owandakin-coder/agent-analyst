"""
Factory helpers for broker adapters.
"""

from __future__ import annotations

import os

from broker_adapter import BrokerSessionConfig
from broker_api import AlpacaBrokerAPI


def build_broker_from_session(session: BrokerSessionConfig, *, auto_approve: bool = True):
    broker_name = session.broker_name.strip().lower()
    if broker_name != "alpaca":
        raise ValueError(f"Unsupported broker adapter: {session.broker_name}")

    os.environ["ALPACA_API_KEY"] = session.api_key
    os.environ["ALPACA_SECRET_KEY"] = session.secret_key
    os.environ["ALPACA_BASE_URL"] = session.base_url
    return AlpacaBrokerAPI(paper=session.trading_mode != "live", auto_approve=auto_approve)
