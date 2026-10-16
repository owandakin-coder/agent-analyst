"""Interpret broker receipts without treating acceptance as execution."""
from __future__ import annotations

import math


def order_status(receipt: dict) -> str:
    return str(receipt.get("status", "")).split(".")[-1].lower()


def confirmed_fill(receipt: dict) -> tuple[float, float]:
    """Only explicit, finite filled quantity and execution price are evidence."""
    try:
        qty = float(receipt.get("filled_qty", 0) or 0)
        price = float(receipt.get("filled_avg_price", 0) or 0)
        requested = float(receipt.get("shares", 0) or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not all(math.isfinite(v) and v > 0 for v in (qty, price, requested)) or qty > requested:
        return 0.0, 0.0
    return qty, price


def requires_reconciliation(receipt: dict) -> bool:
    status = order_status(receipt)
    if status in {"rejected", "rejected_by_user", "duplicate_blocked", "canceled", "expired"}:
        return False
    qty, _ = confirmed_fill(receipt)
    return not (status in {"filled", "filled_stub"} and qty == float(receipt.get("shares", 0)))
