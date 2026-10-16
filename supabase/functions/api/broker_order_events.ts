type Scope = {user_id: unknown; broker_connection_id: unknown; broker_account_id?: unknown};
export function brokerOrderEventPolicy(request: Scope, order: Scope, eventType: unknown): "event_only" | "transition" {
  if (String(request.user_id) !== String(order.user_id)
    || String(request.broker_connection_id) !== String(order.broker_connection_id)
    || String(request.broker_account_id ?? "") !== String(order.broker_account_id ?? "")) {
    throw new Error("Broker order scope mismatch");
  }
  // Blocking an attempted duplicate does not reject the previously filled order.
  return eventType === "duplicate_blocked" ? "event_only" : "transition";
}
