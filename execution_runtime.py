"""
Lightweight runtime hooks for durable execution persistence.
"""

from __future__ import annotations

from typing import Any, Callable


ExecutionHook = Callable[[str, dict[str, Any]], None]
RiskHook = Callable[[str, dict[str, Any]], None]
OrderPrepareHook = Callable[[dict[str, Any]], dict[str, Any] | None]
OrderUpdateHook = Callable[[dict[str, Any]], None]


_execution_hook: ExecutionHook | None = None
_risk_hook: RiskHook | None = None
_order_prepare_hook: OrderPrepareHook | None = None
_order_update_hook: OrderUpdateHook | None = None
_state_load_hook: Callable[[], dict] | None = None
_state_save_hook: Callable[[dict], None] | None = None


def set_execution_runtime_hooks(
    *,
    execution_hook: ExecutionHook | None = None,
    risk_hook: RiskHook | None = None,
    order_prepare_hook: OrderPrepareHook | None = None,
    order_update_hook: OrderUpdateHook | None = None,
    state_load_hook: Callable[[], dict] | None = None,
    state_save_hook: Callable[[dict], None] | None = None,
) -> None:
    global _execution_hook, _risk_hook, _order_prepare_hook, _order_update_hook
    _execution_hook = execution_hook
    _risk_hook = risk_hook
    _order_prepare_hook = order_prepare_hook
    _order_update_hook = order_update_hook
    global _state_load_hook, _state_save_hook
    _state_load_hook, _state_save_hook = state_load_hook, state_save_hook


def load_execution_state() -> dict:
    return _state_load_hook() if _state_load_hook else {}


def save_execution_state(state: dict) -> None:
    if _state_save_hook:
        _state_save_hook(state)


def clear_execution_runtime_hooks() -> None:
    set_execution_runtime_hooks()


def emit_execution_event(stage: str, payload: dict[str, Any]) -> None:
    if _execution_hook is not None:
        _execution_hook(stage, payload)


def emit_risk_event(stage: str, payload: dict[str, Any]) -> None:
    if _risk_hook is not None:
        _risk_hook(stage, payload)


def prepare_order(payload: dict[str, Any]) -> dict[str, Any] | None:
    if _order_prepare_hook is None:
        return None
    return _order_prepare_hook(payload)


def update_order(payload: dict[str, Any]) -> None:
    if _order_update_hook is not None:
        _order_update_hook(payload)
