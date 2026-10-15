// Keep market-clock validation independent so it can be tested without secrets.
export function isFreshOpenClock(clock: Record<string, unknown>, now = Date.now()): boolean {
  const timestamp = Date.parse(String(clock.timestamp ?? ""));
  const close = Date.parse(String(clock.next_close ?? ""));
  return clock.is_open === true && Number.isFinite(timestamp) && Number.isFinite(close)
    && Math.abs(now - timestamp) <= 120_000 && close > now;
}

export function schedulingAllowed(control: Record<string, unknown>): boolean {
  return control.trading_enabled === true && control.emergency_stop === false && control.mode === "paper";
}
