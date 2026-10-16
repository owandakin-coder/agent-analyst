import { ownsLease } from "./execution_lease.ts";
function assert(value: boolean) { if (!value) throw new Error("Assertion failed"); }
Deno.test("only the current Paper lease owner may mutate runtime state or submit orders", () => {
  const now = Date.parse("2026-10-15T15:00:00Z");
  const request = { worker_id:"worker", status:"running", requested_mode:"paper", lease_expires_at:"2026-10-15T15:01:00Z" };
  assert(ownsLease(request,"worker","running",now));
  assert(!ownsLease(request,"other","running",now));
  assert(!ownsLease(request,undefined,"running",now));
  assert(!ownsLease({...request,lease_expires_at:"2026-10-15T14:59:00Z"},"worker","running",now));
  assert(!ownsLease({...request,lease_expires_at:"invalid"},"worker","running",now));
  assert(!ownsLease({...request,status:"succeeded"},"worker","running",now));
  assert(!ownsLease({...request,requested_mode:"live"},"worker","running",now));
});
