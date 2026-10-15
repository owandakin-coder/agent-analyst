import { isFreshOpenClock, schedulingAllowed } from "./autonomous_scheduler.ts";

function assert(value: boolean) { if (!value) throw new Error("Assertion failed"); }
const now = Date.parse("2026-10-15T18:00:00Z");
const clock = { is_open: true, timestamp: "2026-10-15T18:00:00Z", next_close: "2026-10-15T20:00:00Z" };
Deno.test("only an open, recent broker clock permits scheduling", () => {
  assert(isFreshOpenClock(clock, now));
  assert(!isFreshOpenClock({ ...clock, is_open: false }, now));
  assert(!isFreshOpenClock({ ...clock, timestamp: "2026-10-15T17:57:00Z" }, now));
  assert(!isFreshOpenClock({ ...clock, timestamp: "2026-10-15T18:03:00Z" }, now));
  assert(!isFreshOpenClock({ ...clock, next_close: "2026-10-15T17:59:00Z" }, now));
  assert(!isFreshOpenClock({}, now));
});
Deno.test("pause, emergency stop, unknown control and live mode block scheduling", () => {
  const control = { trading_enabled: true, emergency_stop: false, mode: "paper" };
  assert(schedulingAllowed(control));
  assert(!schedulingAllowed({ ...control, trading_enabled: false }));
  assert(!schedulingAllowed({ ...control, emergency_stop: true }));
  assert(!schedulingAllowed({ ...control, mode: "live" }));
  assert(!schedulingAllowed({}));
});
