const MAX_ENTRIES = 200;

export type TtlEntry<T> = { expires: number; value?: T; pending?: Promise<T> };

export function ttlCached<T>(store: Map<string, TtlEntry<T>>, key: string, ttlMs: number, load: () => Promise<T>): Promise<T> {
  const now = Date.now();
  const hit = store.get(key);
  if (hit && hit.expires > now) return hit.pending ?? Promise.resolve(hit.value as T);
  if (store.size >= MAX_ENTRIES) {
    for (const [k, v] of store) if (v.expires <= now || store.size >= MAX_ENTRIES) store.delete(k);
  }
  const pending = load().then(
    (value) => { store.set(key, { expires: Date.now() + ttlMs, value }); return value; },
    (error) => { store.delete(key); throw error; },
  );
  store.set(key, { expires: now + ttlMs, pending });
  return pending;
}
