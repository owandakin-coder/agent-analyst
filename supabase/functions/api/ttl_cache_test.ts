import { ttlCached, type TtlEntry } from "./ttl_cache.ts";

function assertEquals(a: unknown, b: unknown) {
  if (a !== b) throw new Error(`expected ${String(b)}, got ${String(a)}`);
}

Deno.test("ttlCached shares one in-flight load and reuses the value until it expires", async () => {
  const store = new Map<string, TtlEntry<number>>();
  let loads = 0;
  const load = async () => { loads += 1; await new Promise((r) => setTimeout(r, 5)); return loads; };
  const [a, b] = await Promise.all([ttlCached(store, "k", 1000, load), ttlCached(store, "k", 1000, load)]);
  assertEquals(a, 1);
  assertEquals(b, 1);
  assertEquals(await ttlCached(store, "k", 1000, load), 1);
  assertEquals(loads, 1);
  assertEquals(await ttlCached(store, "other", 1000, load), 2);
});

Deno.test("ttlCached never caches a failure and reloads after the ttl", async () => {
  const store = new Map<string, TtlEntry<number>>();
  let calls = 0;
  const flaky = async () => { calls += 1; if (calls === 1) throw new Error("boom"); return calls; };
  let failed = false;
  try { await ttlCached(store, "k", 1000, flaky); } catch { failed = true; }
  assertEquals(failed, true);
  assertEquals(await ttlCached(store, "k", 5, flaky), 2);
  await new Promise((r) => setTimeout(r, 15));
  assertEquals(await ttlCached(store, "k", 5, flaky), 3);
});
