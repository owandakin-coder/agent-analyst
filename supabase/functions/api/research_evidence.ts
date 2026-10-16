export function summarizeFills(orders: Record<string, unknown>[]) {
  let notional = 0, reference = 0, adverse = 0, filled = 0;
  for (const order of orders) {
    const payload = (order.payload ?? {}) as Record<string, unknown>;
    const quantity = Number(payload.filled_qty ?? 0);
    const price = Number(payload.filled_avg_price ?? 0);
    if (!Number.isFinite(quantity) || !Number.isFinite(price) || quantity <= 0 || price <= 0) continue;
    filled += 1;
    notional += quantity * price;
    const requested = Number(order.requested_price ?? 0);
    const side = String(order.side ?? "").toLowerCase();
    if (Number.isFinite(requested) && requested > 0 && ["buy", "sell"].includes(side)) {
      reference += quantity * requested;
      adverse += quantity * (price - requested) * (side === "buy" ? 1 : -1);
    }
  }
  return { filled_orders: filled, filled_notional: notional,
    adverse_fill_vs_decision_bps: reference > 0 ? adverse / reference * 10000 : null,
    fees: null, fees_verified: false, source: "broker_confirmed_fills" };
}
