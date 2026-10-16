import { summarizeFills } from "./research_evidence.ts";
function assert(value: boolean) { if (!value) throw new Error("Assertion failed"); }
Deno.test("accepted orders do not count as fills or generate invented fees", () => {
  const result = summarizeFills([{ status: "accepted", quantity: 10, requested_price: 100, payload: {} }]);
  assert(result.filled_orders === 0 && result.filled_notional === 0 && result.fees === null);
});
Deno.test("partial quantities and side-correct adverse price changes are weighted", () => {
  const result = summarizeFills([
    {side:"BUY", requested_price:100, payload:{filled_qty:"2",filled_avg_price:"101"}},
    {side:"SELL", requested_price:100, payload:{filled_qty:"1",filled_avg_price:"99"}},
    {side:"BUY", requested_price:100, payload:{filled_qty:"0",filled_avg_price:"100"}},
  ]);
  assert(result.filled_orders === 2 && result.filled_notional === 301);
  assert(result.adverse_fill_vs_decision_bps === 100);
});
