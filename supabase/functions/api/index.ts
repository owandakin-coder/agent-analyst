/**
 * ATZMA - Supabase Edge API
 * Public market data + authenticated user profile/preferences + trading control plane.
 */

const PROJECT_REF = "sofowpweliticltlbxrj";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? Deno.env.get("ATZMA_SUPABASE_URL") ?? `https://${PROJECT_REF}.supabase.co`;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? Deno.env.get("ATZMA_SUPABASE_ANON_KEY") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? Deno.env.get("ATZMA_SUPABASE_SERVICE_ROLE_KEY") ?? "";

const ALPACA_KEY = Deno.env.get("ALPACA_API_KEY") ?? "";
const ALPACA_SECRET = Deno.env.get("ALPACA_SECRET_KEY") ?? "";
const ALPACA_BASE = "https://paper-api.alpaca.markets/v2";
const BROKER_CREDENTIAL_KEY = Deno.env.get("ATZMA_BROKER_CREDENTIAL_KEY") ?? "";

const GITHUB_REPO = Deno.env.get("GITHUB_REPOSITORY") ?? "owandakin-coder/agent-analyst";
const WORKER_SHARED_TOKEN = Deno.env.get("ATZMA_WORKER_SHARED_TOKEN") ?? "";
const ALLOW_LEGACY_CONTROL_FALLBACK = (Deno.env.get("ATZMA_ALLOW_LEGACY_CONTROL_FALLBACK") ?? "").toLowerCase() === "1";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, apikey",
};

const SECURITY_HEADERS = {
  "Cache-Control": "no-store",
  "Pragma": "no-cache",
  "Referrer-Policy": "same-origin",
  "X-Content-Type-Options": "nosniff",
};

const AUTH_RATE_LIMIT_WINDOW_MS = 60_000;
const AUTH_RATE_LIMIT_MAX = 5;
const NETWORK_RETRY_ATTEMPTS = 3;
const JOB_DEDUPE_WINDOW_MINUTES = 10;
const authRateLimit = new Map<string, { count: number; resetAt: number }>();

type JsonMap = Record<string, unknown>;

type BrokerConnectionRow = {
  id: string;
  user_id: string;
  broker_name: string;
  account_label: string | null;
  trading_mode: string;
  base_url: string;
  api_key_encrypted: string | null;
  secret_key_encrypted: string | null;
  enabled: boolean;
  last_verified_at: string | null;
  last_verified_status: string;
  last_error: string | null;
  metadata: JsonMap | null;
  key_version?: number | null;
  kms_key_id?: string | null;
  encrypted_data_key?: string | null;
  rotated_at?: string | null;
};

type ExecutionRequestRow = {
  id: string;
  user_id: string;
  broker_connection_id: string | null;
  strategy_id: string;
  trigger_type: string;
  requested_mode: string;
  idempotency_key: string;
  priority: number;
  status: string;
  actor: string | null;
  payload: JsonMap | null;
  result: JsonMap | null;
  error_text: string | null;
  run_after: string;
  attempt_count: number;
  max_attempts: number;
  lease_expires_at: string | null;
  worker_id: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
};

type BrokerOrderRow = {
  id: string;
  user_id: string;
  execution_request_id: string | null;
  broker_connection_id: string | null;
  broker_name: string;
  broker_order_id: string | null;
  client_order_id: string | null;
  symbol: string | null;
  side: string | null;
  quantity: number | null;
  requested_price: number | null;
  status: string;
  idempotency_key?: string | null;
  payload: JsonMap | null;
  created_at: string;
  updated_at: string;
};

type TransitionResult = BrokerOrderRow[];

type AlpacaSession = {
  apiKey: string;
  secretKey: string;
  baseUrl: string;
  scoped: boolean;
  mode: string;
};

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS, ...SECURITY_HEADERS },
  });
}

function getClientIp(req: Request): string {
  const forwarded = req.headers.get("x-forwarded-for") ?? "";
  const realIp = req.headers.get("x-real-ip") ?? "";
  return (forwarded.split(",")[0] || realIp || "unknown").trim();
}

function enforceAuthRateLimit(req: Request, action: "signup" | "signin" | "resend"): void {
  const now = Date.now();
  const ip = getClientIp(req);
  const key = `${action}:${ip}`;
  const current = authRateLimit.get(key);
  if (!current || current.resetAt <= now) {
    authRateLimit.set(key, { count: 1, resetAt: now + AUTH_RATE_LIMIT_WINDOW_MS });
    return;
  }
  current.count += 1;
  authRateLimit.set(key, current);
  if (current.count > AUTH_RATE_LIMIT_MAX) {
    throw new Error("Rate limit exceeded");
  }
}

function getBearerToken(req: Request): string | null {
  const auth = req.headers.get("Authorization") ?? "";
  return auth.startsWith("Bearer ") ? auth.slice(7).trim() : null;
}

function base64Encode(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}

function base64Decode(value: string): Uint8Array {
  return Uint8Array.from(atob(value), (c) => c.charCodeAt(0));
}

async function getBrokerCryptoKey() {
  if (!BROKER_CREDENTIAL_KEY) throw new Error("ATZMA_BROKER_CREDENTIAL_KEY secret is missing");
  const source = new TextEncoder().encode(BROKER_CREDENTIAL_KEY);
  const digest = await crypto.subtle.digest("SHA-256", source);
  return crypto.subtle.importKey("raw", digest, { name: "AES-GCM" }, false, ["encrypt", "decrypt"]);
}

async function encryptSecret(value: string): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await getBrokerCryptoKey();
  const cipher = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    new TextEncoder().encode(value),
  );
  return `${base64Encode(iv)}.${base64Encode(new Uint8Array(cipher))}`;
}

async function decryptSecret(payload: string | null | undefined): Promise<string> {
  if (!payload) return "";
  const [ivPart, cipherPart] = payload.split(".", 2);
  if (!ivPart || !cipherPart) throw new Error("Invalid encrypted secret format");
  const key = await getBrokerCryptoKey();
  const plain = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: base64Decode(ivPart) },
    key,
    base64Decode(cipherPart),
  );
  return new TextDecoder().decode(plain);
}

function maskCredential(value: string | null | undefined): string | null {
  if (!value) return null;
  if (value.length <= 8) return value;
  return `${value.slice(0, 4)}••••${value.slice(-4)}`;
}

function createJobClaimToken(): string {
  return crypto.randomUUID().replace(/-/g, "");
}

function sanitizeJob(job: JsonMap | null | undefined): JsonMap | null {
  if (!job) return null;
  const payload = typeof job.payload === "object" && job.payload ? { ...(job.payload as JsonMap) } : {};
  delete payload.claim_token;
  return { ...job, payload };
}

function buildAlpacaUrl(baseUrl: string, path: string, params = ""): string {
  const root = baseUrl.endsWith("/v2") ? baseUrl : `${baseUrl.replace(/\/$/, "")}/v2`;
  let url = `${root}${path}`;
  if (params) url += `?${params}`;
  return url;
}

async function alpacaGetWithSession(session: AlpacaSession, path: string, params = ""): Promise<unknown> {
  const url = buildAlpacaUrl(session.baseUrl, path, params);
  const res = await fetchWithRetry(url, {
    headers: {
      "APCA-API-KEY-ID": session.apiKey,
      "APCA-API-SECRET-KEY": session.secretKey,
      "Accept": "application/json",
    },
  });
  if (!res.ok) throw new Error(`Alpaca ${path} returned ${res.status}`);
  return res.json();
}

async function alpacaGet(path: string, params = ""): Promise<unknown> {
  return alpacaGetWithSession({
    apiKey: ALPACA_KEY,
    secretKey: ALPACA_SECRET,
    baseUrl: ALPACA_BASE.replace(/\/v2$/, ""),
    scoped: false,
    mode: "paper",
  }, path, params);
}

async function fetchYF(symbol: string): Promise<JsonMap> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=1d&interval=1d`;
  try {
    const res = await fetchWithRetry(url, {
      headers: {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://finance.yahoo.com/",
      },
    });
    const data = await res.json() as { chart: { result: Array<{ meta: Record<string, number> }> } };
    const meta = data.chart.result[0].meta;
    const prev = meta.regularMarketPreviousClose ?? meta.previousClose ?? meta.regularMarketPrice ?? 0;
    return { symbol, price: meta.regularMarketPrice ?? 0, prevClose: prev };
  } catch (e) {
    return { symbol, error: String(e) };
  }
}

async function wait(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function shouldRetryStatus(status: number): boolean {
  return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;
}

async function fetchWithRetry(url: string, init: RequestInit = {}, attempts = NETWORK_RETRY_ATTEMPTS): Promise<Response> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const res = await fetch(url, init);
      if (res.ok || !shouldRetryStatus(res.status) || attempt === attempts) {
        return res;
      }
      await wait(250 * (2 ** (attempt - 1)));
    } catch (error) {
      lastError = error;
      if (attempt === attempts) break;
      await wait(250 * (2 ** (attempt - 1)));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Network request failed");
}

function defaultControlState() {
  const now = new Date().toISOString();
  return {
    mode: "paper",
    trading_enabled: true,
    emergency_stop: false,
    status: "running",
    executor: "worker_pool",
    executor_label: "ATZMA Worker Pool",
    last_command: "bootstrap",
    last_command_at: now,
    updated_at: now,
    updated_by: "system",
    note: "Paper engine is allowed to execute.",
    command_version: 1,
  };
}

function normalizeControlState(input: JsonMap | null | undefined) {
  const state = { ...defaultControlState(), ...(input ?? {}) } as JsonMap;
  if (state.emergency_stop) {
    state.status = "stopped";
    state.trading_enabled = false;
  } else if (!state.trading_enabled) {
    state.status = "paused";
  } else {
    state.status = "running";
  }
  return state;
}

async function loadControlState(): Promise<JsonMap> {
  try {
    const stateRes = await restAdmin("/control_state?id=eq.global&select=*");
    if (stateRes.ok) {
      const rows = await stateRes.json() as Array<{ state?: JsonMap; updated_by?: string; command_version?: number }>;
      const row = rows[0];
      if (row?.state) {
        return {
          ...normalizeControlState(row.state),
          _source: "control_state",
          updated_by: row.updated_by ?? null,
          command_version: row.command_version ?? 1,
          can_dispatch: false,
        };
      }
    }
  } catch {
    if (!ALLOW_LEGACY_CONTROL_FALLBACK) throw new Error("Control state unavailable");
  }
  if (!ALLOW_LEGACY_CONTROL_FALLBACK) throw new Error("Authoritative control state missing");
  try {
    const dbRes = await restAdmin("/audit_events?event_type=eq.control_state&select=event_payload,created_at&order=created_at.desc&limit=1");
    if (dbRes.ok) {
      const rows = await dbRes.json() as Array<{ event_payload?: JsonMap }>;
      const payload = rows[0]?.event_payload ?? {};
      return { ...normalizeControlState(payload), _source: "supabase_db", can_dispatch: false };
    }
  } catch {
    // Fall through to default control state for older deployments.
  }
  return { ...defaultControlState(), _source: "default", can_dispatch: false };
}

async function saveControlState(state: JsonMap, actor: string): Promise<JsonMap> {
  const normalized = normalizeControlState(state);
  try {
    const controlRes = await restAdmin("/control_state?id=eq.global", {
      method: "PATCH",
      headers: { "Prefer": "return=representation" },
      body: JSON.stringify({
        state: normalized,
        updated_at: new Date().toISOString(),
        updated_by: actor,
        command_version: Number(normalized.command_version ?? 1),
      }),
    });
    if (controlRes.ok) {
      const saved = { ...normalized, _source: "control_state", can_dispatch: false };
      await appendControlEvent("control_state_updated", saved, null, null, Number(normalized.command_version ?? 1));
      return saved;
    }
  } catch {
    if (!ALLOW_LEGACY_CONTROL_FALLBACK) throw new Error("Could not persist authoritative control state");
  }
  if (!ALLOW_LEGACY_CONTROL_FALLBACK) throw new Error("Authoritative control state write failed");
  try {
    const res = await restAdmin("/audit_events", {
      method: "POST",
      headers: { "Prefer": "return=minimal" },
      body: JSON.stringify({
        user_id: null,
        event_type: "control_state",
        event_payload: normalized,
      }),
    });
    if (res.ok) {
      return { ...normalized, _source: "supabase_db", can_dispatch: false };
    }
  } catch {
    // Fall through to explicit failure for older deployments.
  }
  throw new Error("Authoritative control state write failed");
}

async function applyControlAction(action: string, actor: string): Promise<JsonMap> {
  const state = await loadControlState();
  const next = { ...state } as JsonMap;
  const now = new Date().toISOString();
  const normalizedAction = action.trim().toLowerCase();

  if (normalizedAction === "pause") {
    next.trading_enabled = false;
    next.emergency_stop = false;
    next.note = "Trading paused by operator.";
  } else if (normalizedAction === "resume") {
    next.trading_enabled = true;
    next.emergency_stop = false;
    next.note = "Trading resumed by operator.";
  } else if (normalizedAction === "stop" || normalizedAction === "emergency_stop") {
    next.trading_enabled = false;
    next.emergency_stop = true;
    next.note = "Emergency stop is active. No new execution is allowed.";
  } else if (normalizedAction === "paper") {
    next.mode = "paper";
    next.note = "Paper mode selected from control surface.";
  } else if (normalizedAction === "live") {
    next.mode = "live";
    next.note = "Live mode selected from control surface.";
  } else {
    throw new Error(`Unsupported control action: ${action}`);
  }

  next.updated_by = actor;
  next.updated_at = now;
  next.last_command = normalizedAction;
  next.last_command_at = now;
  next.command_version = Number(next.command_version ?? 1) + 1;
  return saveControlState(next, actor);
}

async function authRequest(path: string, init: RequestInit = {}, useServiceRole = false): Promise<Response> {
  const headers = new Headers(init.headers);
  const apiKey = useServiceRole ? SUPABASE_SERVICE_ROLE_KEY : SUPABASE_ANON_KEY;
  if (!apiKey) throw new Error("Supabase auth secret is missing");
  headers.set("apikey", apiKey);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  return fetchWithRetry(`${SUPABASE_URL}${path}`, { ...init, headers });
}

async function restAdmin(path: string, init: RequestInit = {}): Promise<Response> {
  if (!SUPABASE_SERVICE_ROLE_KEY) throw new Error("SUPABASE_SERVICE_ROLE_KEY secret is missing");
  const headers = new Headers(init.headers);
  headers.set("apikey", SUPABASE_SERVICE_ROLE_KEY);
  headers.set("Authorization", `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  return fetchWithRetry(`${SUPABASE_URL}/rest/v1${path}`, { ...init, headers });
}

async function restRpc<T>(fn: string, payload: JsonMap = {}): Promise<T> {
  if (!SUPABASE_SERVICE_ROLE_KEY) throw new Error("SUPABASE_SERVICE_ROLE_KEY secret is missing");
  const res = await fetchWithRetry(`${SUPABASE_URL}/rest/v1/rpc/${fn}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "apikey": SUPABASE_SERVICE_ROLE_KEY,
      "Authorization": `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
    },
    body: JSON.stringify(payload),
  });
  const body = await readJsonSafe<T | JsonMap>(res);
  if (!res.ok) {
    throw new Error(extractServiceError(body, `RPC ${fn} failed`));
  }
  return body as T;
}

async function readJsonSafe<T>(res: Response): Promise<T | null> {
  try {
    return await res.json() as T;
  } catch {
    return null;
  }
}

function extractServiceError(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    if (typeof record.message === "string" && record.message.trim()) return record.message.trim();
    if (typeof record.error === "string" && record.error.trim()) return record.error.trim();
    if (typeof record.hint === "string" && record.hint.trim()) return record.hint.trim();
  }
  return fallback;
}

async function requireUser(req: Request): Promise<JsonMap> {
  const token = getBearerToken(req);
  if (!token) throw new Error("Unauthorized");
  const res = await authRequest("/auth/v1/user", {
    headers: { "Authorization": `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Unauthorized");
  return await res.json() as JsonMap;
}

async function ensureUserRows(user: JsonMap): Promise<void> {
  const id = String(user.id);
  const email = String(user.email ?? "");
  const rawMeta = (user.user_metadata ?? {}) as JsonMap;
  const displayName = String(rawMeta.display_name ?? email.split("@")[0] ?? "Member");

  await restAdmin("/profiles?id=eq." + encodeURIComponent(id), {
    method: "PATCH",
    headers: { "Prefer": "resolution=merge-duplicates,return=minimal" },
    body: JSON.stringify({ email, display_name: displayName, active_executor: "worker_pool" }),
  });

  const profileRes = await restAdmin(`/profiles?id=eq.${encodeURIComponent(id)}&select=id`);
  const profileRows = await profileRes.json() as JsonMap[];
  if (!profileRows.length) {
    await restAdmin("/profiles", {
      method: "POST",
      headers: { "Prefer": "return=minimal" },
      body: JSON.stringify({ id, email, display_name: displayName, active_executor: "worker_pool" }),
    });
  }

  const prefsRes = await restAdmin(`/user_preferences?user_id=eq.${encodeURIComponent(id)}&select=user_id`);
  const prefRows = await prefsRes.json() as JsonMap[];
  if (!prefRows.length) {
    await restAdmin("/user_preferences", {
      method: "POST",
      headers: { "Prefer": "return=minimal" },
      body: JSON.stringify({ user_id: id }),
    });
  }
}

async function fetchUserBundle(userId: string) {
  const [profileRes, prefsRes] = await Promise.all([
    restAdmin(`/profiles?id=eq.${encodeURIComponent(userId)}&select=*`),
    restAdmin(`/user_preferences?user_id=eq.${encodeURIComponent(userId)}&select=*`),
  ]);
  const profile = ((await profileRes.json()) as JsonMap[])[0] ?? null;
  const preferences = ((await prefsRes.json()) as JsonMap[])[0] ?? null;
  if (profile) {
    profile.active_executor = "worker_pool";
  }
  return { profile, preferences };
}

async function audit(userId: string, eventType: string, payload: JsonMap = {}) {
  try {
    await restAdmin("/audit_events", {
      method: "POST",
      headers: { "Prefer": "return=minimal" },
      body: JSON.stringify({ user_id: userId, event_type: eventType, event_payload: payload }),
    });
  } catch {
    // Best effort only.
  }
}

async function appendExecutionEvent(userId: string, executionRequestId: string, stage: string, payload: JsonMap = {}) {
  await restAdmin("/execution_events", {
    method: "POST",
    headers: { "Prefer": "return=minimal" },
    body: JSON.stringify({
      user_id: userId,
      execution_request_id: executionRequestId,
      stage,
      payload,
    }),
  });
}

async function appendControlEvent(eventType: string, payload: JsonMap = {}, executionRequestId?: string | null, userId?: string | null, commandVersion?: number | null) {
  await restAdmin("/control_events", {
    method: "POST",
    headers: { "Prefer": "return=minimal" },
    body: JSON.stringify({
      user_id: userId ?? null,
      execution_request_id: executionRequestId ?? null,
      command_version: commandVersion ?? null,
      event_type: eventType,
      payload,
    }),
  });
}

async function appendRiskEvent(userId: string, executionRequestId: string, eventType: string, payload: JsonMap = {}) {
  await restAdmin("/risk_events", {
    method: "POST",
    headers: { "Prefer": "return=minimal" },
    body: JSON.stringify({
      user_id: userId,
      execution_request_id: executionRequestId,
      event_type: eventType,
      risk_level: typeof payload.risk_level === "string" ? payload.risk_level : null,
      drawdown: typeof payload.drawdown === "number" ? payload.drawdown : null,
      payload,
    }),
  });
  if (eventType === "daily_loss_state" || eventType === "daily_loss_breached") {
    const tradingDay = typeof payload.trading_day === "string" ? payload.trading_day : new Date().toISOString().slice(0, 10);
    await restAdmin("/daily_risk_state", {
      method: "POST",
      headers: { "Prefer": "resolution=merge-duplicates,return=minimal" },
      body: JSON.stringify({
        user_id: userId,
        trading_day: tradingDay,
        baseline_equity: Number(payload.baseline_equity ?? 0),
        current_equity: Number(payload.current_equity ?? 0),
        realized_pnl: Number(payload.realized_pnl ?? 0),
        unrealized_pnl: Number(payload.unrealized_pnl ?? 0),
        realized_loss_limit: Number(payload.realized_loss_limit ?? 0),
        unrealized_loss_limit: Number(payload.unrealized_loss_limit ?? 0),
        breached: eventType === "daily_loss_breached",
        breached_at: eventType === "daily_loss_breached" ? new Date().toISOString() : null,
        reset_required: eventType === "daily_loss_breached",
        metadata: payload,
      }),
    });
  }
}

async function appendPositionSnapshot(userId: string, executionRequestId: string, snapshotType: string, payload: JsonMap = {}) {
  await restAdmin("/position_snapshots", {
    method: "POST",
    headers: { "Prefer": "return=minimal" },
    body: JSON.stringify({
      user_id: userId,
      execution_request_id: executionRequestId,
      snapshot_type: snapshotType,
      snapshot_hash: typeof payload.snapshot_hash === "string" ? payload.snapshot_hash : null,
      payload,
    }),
  });
}

async function upsertBrokerOrder(request: ExecutionRequestRow, payload: JsonMap): Promise<BrokerOrderRow | null> {
  const userId = String(request.user_id);
  const clientOrderId = typeof payload.client_order_id === "string" ? payload.client_order_id : null;
  if (!clientOrderId) return null;
  const existingRes = await restAdmin(`/broker_orders?client_order_id=eq.${encodeURIComponent(clientOrderId)}&select=*`);
  const existingRows = await readJsonSafe<BrokerOrderRow[] | JsonMap>(existingRes);
  if (existingRes.ok && Array.isArray(existingRows) && existingRows[0]) {
    return existingRows[0];
  }
  const res = await restAdmin("/broker_orders", {
    method: "POST",
    headers: { "Prefer": "return=representation" },
    body: JSON.stringify({
      user_id: userId,
      execution_request_id: request.id,
      broker_connection_id: request.broker_connection_id,
      broker_name: typeof payload.broker_name === "string" ? payload.broker_name : "alpaca",
      broker_order_id: typeof payload.order_id === "string" ? payload.order_id : null,
      client_order_id: clientOrderId,
      symbol: typeof payload.ticker === "string" ? payload.ticker : null,
      side: typeof payload.side === "string" ? payload.side : null,
      quantity: typeof payload.shares === "number" ? payload.shares : null,
      requested_price: typeof payload.requested_price === "number"
        ? payload.requested_price
        : (typeof payload.price === "number" ? payload.price : null),
      status: typeof payload.status === "string" ? payload.status.toLowerCase() : "created",
      idempotency_key: typeof payload.idempotency_key === "string" ? payload.idempotency_key : null,
      submitted_at: typeof payload.status === "string" && ["submit_requested", "submit_acknowledged"].includes(payload.status) ? new Date().toISOString() : null,
      reconciled_at: typeof payload.status === "string" && ["reconciled", "filled", "cancelled", "rejected"].includes(payload.status) ? new Date().toISOString() : null,
      payload,
    }),
  });
  const rows = await readJsonSafe<BrokerOrderRow[] | JsonMap>(res);
  if (!res.ok) {
    throw new Error(extractServiceError(rows, "Could not persist broker order"));
  }
  return Array.isArray(rows) ? rows[0] ?? null : null;
}

async function patchBrokerOrderByClientOrderId(clientOrderId: string, patch: JsonMap): Promise<BrokerOrderRow | null> {
  const nextStatus = typeof patch.status === "string" ? patch.status : null;
  if (!nextStatus) {
    throw new Error("Missing broker order status transition");
  }
  const rows = await restRpc<TransitionResult>("transition_broker_order_state", {
    p_client_order_id: clientOrderId,
    p_next_status: nextStatus,
    p_payload: typeof patch.payload === "object" && patch.payload ? patch.payload : {},
    p_broker_order_id: typeof patch.broker_order_id === "string" ? patch.broker_order_id : null,
    p_event_type: typeof patch.event_type === "string" ? patch.event_type : null,
  });
  return rows[0] ?? null;
}

async function appendBrokerOrderEvent(userId: string, executionRequestId: string, brokerOrderId: string | null, eventType: string, payload: JsonMap = {}) {
  await restAdmin("/broker_order_events", {
    method: "POST",
    headers: { "Prefer": "return=minimal" },
    body: JSON.stringify({
      user_id: userId,
      broker_order_id: brokerOrderId,
      execution_request_id: executionRequestId,
      event_type: eventType,
      payload,
    }),
  });
}

async function reconcileBrokerOrdersForRequest(request: ExecutionRequestRow, brokerRow: BrokerConnectionRow, clientOrderId?: string | null): Promise<JsonMap[]> {
  const session = {
    apiKey: await decryptSecret(brokerRow.api_key_encrypted),
    secretKey: await decryptSecret(brokerRow.secret_key_encrypted),
    baseUrl: brokerRow.base_url,
    scoped: true,
    mode: brokerRow.trading_mode,
  } satisfies AlpacaSession;
  const filter = clientOrderId
    ? `client_order_id=eq.${encodeURIComponent(clientOrderId)}`
    : `execution_request_id=eq.${encodeURIComponent(request.id)}&status=in.(submit_requested,submit_acknowledged,partial_fill,reconciliation_pending,cancel_requested,reconciled)`;
  const orderRes = await restAdmin(`/broker_orders?${filter}&select=*`);
  const orders = await orderRes.json() as BrokerOrderRow[];
  const reconciled: JsonMap[] = [];
  for (const order of orders) {
    const currentClientOrderId = String(order.client_order_id ?? "").trim();
    if (!currentClientOrderId) continue;
    try {
      const brokerOrder = await alpacaGetWithSession(session, "/orders:by_client_order_id", `client_order_id=${encodeURIComponent(currentClientOrderId)}`) as JsonMap;
      const brokerStatus = String(brokerOrder.status ?? "").toLowerCase();
      let mappedStatus = "reconciled";
      if (brokerStatus === "partially_filled") mappedStatus = "partial_fill";
      else if (brokerStatus === "filled") mappedStatus = "filled";
      else if (brokerStatus === "canceled") mappedStatus = "cancelled";
      else if (brokerStatus === "rejected") mappedStatus = "rejected";
      else if (brokerStatus === "accepted" || brokerStatus === "new" || brokerStatus === "pending_new") mappedStatus = "submit_acknowledged";
      const patched = await patchBrokerOrderByClientOrderId(currentClientOrderId, {
        broker_order_id: String(brokerOrder.id ?? order.broker_order_id ?? ""),
        status: mappedStatus,
        payload: brokerOrder,
        event_type: "reconciled_poll",
      });
      await appendBrokerOrderEvent(String(request.user_id), request.id, patched?.id ?? order.id, mappedStatus, brokerOrder);
      reconciled.push({ client_order_id: currentClientOrderId, status: mappedStatus });
    } catch (error) {
      await appendBrokerOrderEvent(String(request.user_id), request.id, order.id, "reconciliation_mismatch", {
        client_order_id: currentClientOrderId,
        error: String(error),
      });
      reconciled.push({ client_order_id: currentClientOrderId, status: "mismatch", error: String(error) });
    }
  }
  await appendExecutionEvent(String(request.user_id), request.id, "reconciliation_completed", { orders: reconciled });
  return reconciled;
}

async function fetchBrokerConnection(userId: string): Promise<BrokerConnectionRow | null> {
  const res = await restAdmin(`/broker_connections?user_id=eq.${encodeURIComponent(userId)}&select=*`);
  const rows = await res.json() as BrokerConnectionRow[];
  return rows[0] ?? null;
}

async function fetchExecutionRequest(requestId: string): Promise<ExecutionRequestRow | null> {
  const res = await restAdmin(`/execution_requests?id=eq.${encodeURIComponent(requestId)}&select=*`);
  const rows = await res.json() as ExecutionRequestRow[];
  return rows[0] ?? null;
}

async function resolveScopedBrokerSession(req: Request): Promise<AlpacaSession | null> {
  if (!getBearerToken(req)) return null;
  try {
    const user = await requireUser(req);
    const row = await fetchBrokerConnection(String(user.id));
    if (!row?.enabled || row.last_verified_status !== "verified" || !row.api_key_encrypted || !row.secret_key_encrypted) {
      return null;
    }
    return {
      apiKey: await decryptSecret(row.api_key_encrypted),
      secretKey: await decryptSecret(row.secret_key_encrypted),
      baseUrl: row.base_url,
      scoped: true,
      mode: row.trading_mode,
    };
  } catch {
    return null;
  }
}

async function requireScopedBrokerSession(req: Request): Promise<AlpacaSession> {
  const user = await requireUser(req);
  const row = await fetchBrokerConnection(String(user.id));
  if (!row?.enabled || row.last_verified_status !== "verified" || !row.api_key_encrypted || !row.secret_key_encrypted) {
    throw new Error("Verified broker connection required");
  }
  return {
    apiKey: await decryptSecret(row.api_key_encrypted),
    secretKey: await decryptSecret(row.secret_key_encrypted),
    baseUrl: row.base_url,
    scoped: true,
    mode: row.trading_mode,
  };
}

function serializeBrokerConnection(row: BrokerConnectionRow | null, decryptedApiKey = "", decryptedSecretKey = "") {
  if (!row) {
    return {
      connected: false,
      broker_name: "alpaca",
      trading_mode: "paper",
      base_url: "https://paper-api.alpaca.markets",
      enabled: false,
      last_verified_status: "pending",
      last_verified_at: null,
      last_error: null,
      account_label: null,
      masked_api_key: null,
      masked_secret_key: null,
    };
  }
  return {
    id: row.id,
    connected: !!row.api_key_encrypted && !!row.secret_key_encrypted,
    broker_name: row.broker_name,
    trading_mode: row.trading_mode,
    base_url: row.base_url,
    enabled: row.enabled,
    last_verified_status: row.last_verified_status,
    last_verified_at: row.last_verified_at,
    last_error: row.last_error,
    account_label: row.account_label,
    masked_api_key: maskCredential(decryptedApiKey),
    masked_secret_key: maskCredential(decryptedSecretKey),
    metadata: row.metadata ?? {},
  };
}

function sanitizeBrokerInput(body: JsonMap, existing?: BrokerConnectionRow | null) {
  const tradingMode = typeof body.trading_mode === "string" && body.trading_mode.toLowerCase() === "live" ? "live" : "paper";
  const fallbackBaseUrl = existing?.base_url ?? (tradingMode === "live" ? "https://api.alpaca.markets" : "https://paper-api.alpaca.markets");
  let baseUrl = fallbackBaseUrl;
  if (typeof body.base_url === "string" && body.base_url.trim()) {
    try {
      const parsed = new URL(body.base_url.trim());
      const host = parsed.hostname.toLowerCase();
      if (host === "app.alpaca.markets") {
        baseUrl = tradingMode === "live" ? "https://api.alpaca.markets" : "https://paper-api.alpaca.markets";
      } else if (host === "paper-api.alpaca.markets") {
        baseUrl = "https://paper-api.alpaca.markets";
      } else if (host === "api.alpaca.markets") {
        baseUrl = tradingMode === "live" ? "https://api.alpaca.markets" : "https://paper-api.alpaca.markets";
      } else {
        baseUrl = `${parsed.protocol}//${parsed.host}`;
      }
    } catch {
      baseUrl = fallbackBaseUrl;
    }
  }
  const accountLabel = typeof body.account_label === "string"
    ? body.account_label.replace(/[<>{}]/g, "").trim().slice(0, 80)
    : existing?.account_label ?? null;
  return { tradingMode, baseUrl, accountLabel };
}

function sanitizeBrokerError(error: unknown): string {
  const text = String(error || "").toLowerCase();
  if (text.includes("unauthorized") || text.includes("forbidden") || text.includes("revoked")) {
    return "Alpaca rejected these keys. Confirm Paper vs Live mode and use a matching API Key + Secret Key pair.";
  }
  if (text.includes("timeout") || text.includes("network") || text.includes("fetch")) {
    return "Alpaca could not be reached right now. Try again in a minute.";
  }
  return "Broker verification failed. Review your Alpaca credentials and try again.";
}

function sanitizeExecutionRequest(request: ExecutionRequestRow | JsonMap | null | undefined): JsonMap | null {
  if (!request) return null;
  const payload = typeof request.payload === "object" && request.payload ? { ...(request.payload as JsonMap) } : {};
  delete payload.claim_token;
  return { ...request, payload };
}

function buildExecutionIdempotencyKey(userId: string, brokerConnectionId: string, requestedMode: string, triggerType: string): string {
  const bucket = Math.floor(Date.now() / (JOB_DEDUPE_WINDOW_MINUTES * 60_000));
  return `${triggerType}:${userId}:${brokerConnectionId}:${requestedMode}:${bucket}`;
}

async function findOpenTradeRequest(userId: string): Promise<ExecutionRequestRow | null> {
  const since = new Date(Date.now() - (JOB_DEDUPE_WINDOW_MINUTES * 60_000)).toISOString();
  const path = `/execution_requests?user_id=eq.${encodeURIComponent(userId)}&strategy_id=eq.default&status=in.(queued,leased,running,retrying,reconcile_pending)&created_at=gte.${encodeURIComponent(since)}&order=created_at.desc&limit=1`;
  const res = await restAdmin(path);
  const rows = await res.json() as ExecutionRequestRow[];
  return rows[0] ?? null;
}

async function findOpenTradeJob(userId: string): Promise<JsonMap | null> {
  const since = new Date(Date.now() - (JOB_DEDUPE_WINDOW_MINUTES * 60_000)).toISOString();
  const path = `/execution_jobs?user_id=eq.${encodeURIComponent(userId)}&job_type=eq.trade_once&status=in.(queued,running)&requested_at=gte.${encodeURIComponent(since)}&order=requested_at.desc&limit=1`;
  const res = await restAdmin(path);
  const rows = await res.json() as JsonMap[];
  return rows[0] ?? null;
}

async function verifyAlpacaCredentials(apiKey: string, secretKey: string, baseUrl: string): Promise<JsonMap> {
  const endpoint = `${baseUrl.replace(/\/$/, "")}/v2/account`;
  const res = await fetchWithRetry(endpoint, {
    headers: {
      "APCA-API-KEY-ID": apiKey,
      "APCA-API-SECRET-KEY": secretKey,
      "Accept": "application/json",
    },
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof payload?.message === "string" ? payload.message : `Broker verification returned ${res.status}`;
    throw new Error(detail);
  }
  return payload as JsonMap;
}

function workerAuthorized(req: Request, expectedToken = ""): boolean {
  const token = req.headers.get("x-atzma-worker-token") ?? "";
  return !!expectedToken && token === expectedToken;
}

function sharedWorkerAuthorized(req: Request): boolean {
  const token = req.headers.get("x-atzma-worker-token") ?? "";
  return !!WORKER_SHARED_TOKEN && token === WORKER_SHARED_TOKEN;
}

async function enqueueExecutionRequest(user: JsonMap, broker: BrokerConnectionRow, triggerType: "manual" | "control_plane", sourcePayload: JsonMap = {}): Promise<ExecutionRequestRow> {
  const userId = String(user.id);
  const requestedMode = broker.trading_mode === "live" ? "live" : "paper";
  const idempotencyKey = typeof sourcePayload.idempotency_key === "string" && sourcePayload.idempotency_key.trim()
    ? sourcePayload.idempotency_key.trim()
    : buildExecutionIdempotencyKey(userId, broker.id, requestedMode, triggerType);
  const actor = String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id);
  const claimToken = createJobClaimToken();
  const res = await restAdmin("/execution_requests", {
    method: "POST",
    headers: { "Prefer": "return=representation,resolution=merge-duplicates" },
    body: JSON.stringify({
      user_id: userId,
      broker_connection_id: broker.id,
      strategy_id: "default",
      trigger_type: triggerType,
      requested_mode: requestedMode,
      idempotency_key: idempotencyKey,
      priority: triggerType === "control_plane" ? 50 : 100,
      status: "queued",
      actor,
      payload: { ...sourcePayload, claim_token: claimToken },
    }),
  });
  const rowsPayload = await readJsonSafe<ExecutionRequestRow[] | JsonMap>(res);
  if (!res.ok) {
    throw new Error(extractServiceError(rowsPayload, "Could not queue execution request"));
  }
  const rows = Array.isArray(rowsPayload) ? rowsPayload : [];
  return rows[0];
}

async function persistExecutionArtifacts(request: ExecutionRequestRow, result: JsonMap = {}): Promise<void> {
  const userId = String(request.user_id);
  await restAdmin("/decision_events", {
    method: "POST",
    headers: { "Prefer": "return=minimal" },
    body: JSON.stringify({
      user_id: userId,
      execution_request_id: request.id,
      model_version: typeof result.model_version === "string" ? result.model_version : null,
      feature_snapshot_hash: typeof result.feature_snapshot_hash === "string" ? result.feature_snapshot_hash : null,
      market_data_source: typeof result.market_data_source === "string" ? result.market_data_source : "alpaca+yfinance",
      regime: typeof result.regime === "string" ? result.regime : null,
      strategy_mode: typeof result.strategy_mode === "string" ? result.strategy_mode : null,
      summary: typeof result.decision_summary === "string" ? result.decision_summary : null,
      raw_action: Array.isArray(result.raw_action) ? result.raw_action : [],
      scaled_action: Array.isArray(result.scaled_action) ? result.scaled_action : [],
      decisions: Array.isArray(result.decisions) ? result.decisions : [],
      payload: typeof result.payload === "object" && result.payload ? result.payload : result,
    }),
  });
  await appendExecutionEvent(userId, request.id, "execution_result_persisted", {
    model_version: typeof result.model_version === "string" ? result.model_version : null,
    strategy_version: typeof result.strategy_version === "string" ? result.strategy_version : null,
    market_snapshot_hash: typeof result.market_snapshot_hash === "string" ? result.market_snapshot_hash : null,
    broker_snapshot_hash: typeof result.broker_snapshot_hash === "string" ? result.broker_snapshot_hash : null,
    feature_snapshot_hash: typeof result.feature_snapshot_hash === "string" ? result.feature_snapshot_hash : null,
  });

  if (typeof result.broker_snapshot_hash === "string") {
    await appendPositionSnapshot(userId, request.id, "post_execution", {
      snapshot_hash: result.broker_snapshot_hash,
      snapshot: typeof result.broker_snapshot === "object" && result.broker_snapshot ? result.broker_snapshot : result,
    });
  }
  if (typeof result.market_snapshot_hash === "string" && typeof result.market_snapshot === "object" && result.market_snapshot) {
    await appendPositionSnapshot(userId, request.id, "market_snapshot", {
      snapshot_hash: result.market_snapshot_hash,
      snapshot: result.market_snapshot,
    });
  }
  if (typeof result.feature_snapshot_hash === "string" && typeof result.feature_snapshot === "object" && result.feature_snapshot) {
    await appendPositionSnapshot(userId, request.id, "feature_snapshot", {
      snapshot_hash: result.feature_snapshot_hash,
      snapshot: result.feature_snapshot,
    });
  }

  const brokerOrders = Array.isArray(result.broker_orders) ? result.broker_orders : [];
  for (const order of brokerOrders) {
    const orderPayload = typeof order === "object" && order ? order as JsonMap : {};
    const brokerOrder = await upsertBrokerOrder(request, orderPayload);
    if (brokerOrder?.id) {
      await appendBrokerOrderEvent(
        userId,
        request.id,
        brokerOrder.id,
        typeof orderPayload.event_type === "string" ? orderPayload.event_type : "submitted",
        orderPayload,
      );
    }
  }

  if (typeof result.risk_level === "string" || typeof result.drawdown === "number") {
    await appendRiskEvent(userId, request.id, "execution_cycle", {
      risk_level: result.risk_level ?? null,
      drawdown: result.drawdown ?? null,
    });
  }
}

Deno.serve(async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") return new Response(null, { headers: CORS_HEADERS });

  const url = new URL(req.url);
  const rawPath = url.pathname;
  const funcIdx = rawPath.indexOf("/api");
  const path = funcIdx >= 0 ? rawPath.slice(funcIdx + 4) || "/" : rawPath;
  const qs = url.searchParams;

  try {
    if (path === "/health") {
      return json({
        ok: true,
        system: "ATZMA",
        timestamp: new Date().toISOString(),
        github_repo: GITHUB_REPO,
        control_plane: true,
        executor: "worker_pool",
        can_dispatch: false,
        broker_credentials_encryption: !!BROKER_CREDENTIAL_KEY,
      });
    }

    if (path === "/account") {
      const session = await requireScopedBrokerSession(req);
      const data = await alpacaGetWithSession(session, "/account") as Record<string, string>;
      return json({
        equity: parseFloat(data.equity ?? "0"),
        last_equity: parseFloat(data.last_equity ?? "0"),
        cash: parseFloat(data.cash ?? "0"),
        buying_power: parseFloat(data.buying_power ?? "0"),
        portfolio_value: parseFloat(data.portfolio_value ?? "0"),
        account_type: data.account_blocked ? "restricted" : session.mode,
        id: data.account_number ?? data.id ?? "",
        status: data.status ?? "active",
        scoped_user: true,
      });
    }

    if (path === "/positions") {
      const session = await requireScopedBrokerSession(req);
      const positions = await alpacaGetWithSession(session, "/positions") as Array<Record<string, string>>;
      return json(positions.map((p) => ({
        symbol: p.symbol,
        qty: parseFloat(p.qty ?? "0"),
        avg_entry: parseFloat(p.avg_entry_price ?? "0"),
        current_price: parseFloat(p.current_price ?? "0"),
        market_value: parseFloat(p.market_value ?? "0"),
        unrealized_pl: parseFloat(p.unrealized_pl ?? "0"),
        unrealized_plpc: parseFloat(p.unrealized_plpc ?? "0"),
        change_today: parseFloat(p.change_today ?? "0"),
      })));
    }

    if (path === "/orders") {
      const session = await requireScopedBrokerSession(req);
      const orders = await alpacaGetWithSession(session, "/orders", "status=closed&limit=50&direction=desc") as Array<Record<string, string>>;
      return json(orders
        .filter((o) => o.filled_at && o.filled_avg_price)
        .map((o) => ({
          id: o.id,
          symbol: o.symbol,
          side: o.side,
          qty: parseFloat(o.filled_qty ?? o.qty ?? "0"),
          price: parseFloat(o.filled_avg_price ?? "0"),
          time: o.filled_at ?? "",
          type: o.type ?? "",
        })));
    }

    if (path === "/history") {
      const period = qs.get("period") ?? "1M";
      const timeframe = qs.get("timeframe") ?? "1D";
      const session = await requireScopedBrokerSession(req);
      const data = await alpacaGetWithSession(
        session,
        "/account/portfolio/history",
        `period=${period}&timeframe=${timeframe}&intraday_reporting=market_hours`,
      ) as { timestamp: number[]; equity: number[] };
      return json({ timestamps: data.timestamp ?? [], equity: data.equity ?? [] });
    }

    if (path === "/quotes") {
      const symbolsStr = qs.get("symbols") ?? "AAPL,MSFT,GOOGL,NVDA,AMZN,META,TSLA,JPM,V,BAC,JNJ,UNH,XOM,WMT,SPY";
      const symbols = symbolsStr.split(",").map((s) => s.trim()).filter(Boolean);
      const quotes = await Promise.all(symbols.map(fetchYF));
      return json({ quotes });
    }

    if (path === "/auth/signup" && req.method === "POST") {
      enforceAuthRateLimit(req, "signup");
      const body = await req.json().catch(() => ({})) as { email?: string; password?: string; display_name?: string };
      if (!body.email || !body.password) return json({ error: "email and password are required" }, 400);
      const res = await authRequest("/auth/v1/signup", {
        method: "POST",
        body: JSON.stringify({
          email: body.email,
          password: body.password,
          data: { display_name: body.display_name ?? body.email.split("@")[0] },
        }),
      });
      const payload = await res.json();
      if (!res.ok) return json(payload, res.status);
      const user = payload.user as JsonMap | undefined;
      if (user?.id) {
        await ensureUserRows(user);
        await audit(String(user.id), "auth_signup", { email: body.email });
      }
      return json(payload, 201);
    }

    if (path === "/auth/signin" && req.method === "POST") {
      enforceAuthRateLimit(req, "signin");
      const body = await req.json().catch(() => ({})) as { email?: string; password?: string };
      if (!body.email || !body.password) return json({ error: "email and password are required" }, 400);
      const res = await authRequest("/auth/v1/token?grant_type=password", {
        method: "POST",
        body: JSON.stringify({ email: body.email, password: body.password }),
      });
      const payload = await res.json();
      if (!res.ok) return json(payload, res.status);
      const user = payload.user as JsonMap | undefined;
      if (user?.id) {
        await ensureUserRows(user);
        await audit(String(user.id), "auth_signin", { email: body.email });
      }
      return json(payload);
    }

    if (path === "/auth/resend" && req.method === "POST") {
      enforceAuthRateLimit(req, "resend");
      const body = await req.json().catch(() => ({})) as { email?: string };
      if (!body.email) return json({ error: "email is required" }, 400);
      const res = await authRequest("/auth/v1/resend", {
        method: "POST",
        body: JSON.stringify({
          type: "signup",
          email: body.email,
        }),
      });
      const payload = await res.json();
      if (!res.ok) return json(payload, res.status);
      return json({ ok: true, ...payload });
    }

    if (path === "/auth/verify" && req.method === "POST") {
      const body = await req.json().catch(() => ({})) as { token_hash?: string; type?: string; email?: string; token?: string };
      if (!body.type) return json({ error: "type is required" }, 400);
      if (!body.token_hash && !body.token) return json({ error: "token_hash or token is required" }, 400);
      const res = await authRequest("/auth/v1/verify", {
        method: "POST",
        body: JSON.stringify({
          type: body.type,
          token_hash: body.token_hash,
          email: body.email,
          token: body.token,
        }),
      });
      const payload = await res.json();
      if (!res.ok) return json(payload, res.status);
      return json(payload);
    }

    if (path === "/auth/me" && req.method === "GET") {
      const user = await requireUser(req);
      await ensureUserRows(user);
      return json({ user });
    }

    if (path === "/me" && req.method === "GET") {
      const user = await requireUser(req);
      await ensureUserRows(user);
      const bundle = await fetchUserBundle(String(user.id));
      return json({ user, ...bundle });
    }

    if (path === "/me/profile" && req.method === "PUT") {
      const user = await requireUser(req);
      const body = await req.json().catch(() => ({})) as JsonMap;
      const payload = {
        email: typeof body.email === "string" ? body.email : user.email,
        display_name: typeof body.display_name === "string" ? body.display_name : null,
        phone: typeof body.phone === "string" ? body.phone : null,
        timezone: typeof body.timezone === "string" ? body.timezone : "Asia/Jerusalem",
        account_tier: typeof body.account_tier === "string" ? body.account_tier : "paper",
        trading_mode: typeof body.trading_mode === "string" ? body.trading_mode : "paper",
        active_executor: typeof body.active_executor === "string" ? body.active_executor : "worker_pool",
      };
      const res = await restAdmin(`/profiles?id=eq.${encodeURIComponent(String(user.id))}`, {
        method: "PATCH",
        headers: { "Prefer": "return=representation" },
        body: JSON.stringify(payload),
      });
      const rows = await res.json() as JsonMap[];
      await audit(String(user.id), "profile_update", payload);
      return json({ profile: rows[0] ?? payload });
    }

    if (path === "/me/preferences" && req.method === "PUT") {
      const user = await requireUser(req);
      const body = await req.json().catch(() => ({})) as JsonMap;
      const payload = {
        daily_email: typeof body.daily_email === "string" ? body.daily_email : null,
        risk_level: typeof body.risk_level === "string" ? body.risk_level : "Normal",
        risk_note: typeof body.risk_note === "string" ? body.risk_note : "Max drawdown threshold: 15%",
        auto_trade: body.auto_trade === undefined ? true : !!body.auto_trade,
        stop_loss: body.stop_loss === undefined ? true : !!body.stop_loss,
        kelly: body.kelly === undefined ? true : !!body.kelly,
        push_alerts: body.push_alerts === undefined ? false : !!body.push_alerts,
        watchlist: Array.isArray(body.watchlist) ? body.watchlist : [],
        ui: typeof body.ui === "object" && body.ui ? body.ui : {},
        notifications: typeof body.notifications === "object" && body.notifications ? body.notifications : {},
      };
      const res = await restAdmin(`/user_preferences?user_id=eq.${encodeURIComponent(String(user.id))}`, {
        method: "PATCH",
        headers: { "Prefer": "return=representation" },
        body: JSON.stringify(payload),
      });
      const rows = await res.json() as JsonMap[];
      await audit(String(user.id), "preferences_update", payload);
      return json({ preferences: rows[0] ?? payload });
    }

    if (path === "/me/audit" && req.method === "GET") {
      const user = await requireUser(req);
      const res = await restAdmin(`/audit_events?user_id=eq.${encodeURIComponent(String(user.id))}&select=*&order=created_at.desc&limit=30`);
      return json({ events: await res.json() });
    }

    if (path === "/me/broker" && req.method === "GET") {
      const user = await requireUser(req);
      const row = await fetchBrokerConnection(String(user.id));
      let apiKey = "";
      let secretKey = "";
      if (row) {
        apiKey = await decryptSecret(row.api_key_encrypted);
        secretKey = await decryptSecret(row.secret_key_encrypted);
      }
      return json({ connection: serializeBrokerConnection(row, apiKey, secretKey) });
    }

    if (path === "/me/broker" && req.method === "PUT") {
      const user = await requireUser(req);
      const body = await req.json().catch(() => ({})) as JsonMap;
      const userId = String(user.id);
      const existing = await fetchBrokerConnection(userId);
      const sanitized = sanitizeBrokerInput(body, existing);
      const apiKeyPlain = typeof body.api_key === "string" && body.api_key.trim()
        ? body.api_key.trim()
        : existing ? await decryptSecret(existing.api_key_encrypted) : "";
      const secretKeyPlain = typeof body.secret_key === "string" && body.secret_key.trim()
        ? body.secret_key.trim()
        : existing ? await decryptSecret(existing.secret_key_encrypted) : "";
      const payload = {
        user_id: userId,
        broker_name: "alpaca",
        account_label: sanitized.accountLabel,
        trading_mode: sanitized.tradingMode,
        base_url: sanitized.baseUrl,
        enabled: body.enabled === undefined ? (existing?.enabled ?? true) : !!body.enabled,
        api_key_encrypted: apiKeyPlain ? await encryptSecret(apiKeyPlain) : null,
        secret_key_encrypted: secretKeyPlain ? await encryptSecret(secretKeyPlain) : null,
        last_error: existing?.last_error ?? null,
        last_verified_status: existing?.last_verified_status ?? "pending",
      };
      const method = existing ? "PATCH" : "POST";
      const pathSuffix = existing ? `/broker_connections?id=eq.${encodeURIComponent(existing.id)}` : "/broker_connections";
      const res = await restAdmin(pathSuffix, {
        method,
        headers: { "Prefer": "return=representation" },
        body: JSON.stringify(payload),
      });
      const rowsPayload = await readJsonSafe<BrokerConnectionRow[] | JsonMap>(res);
      if (!res.ok) {
        throw new Error(extractServiceError(rowsPayload, "Could not save broker connection"));
      }
      const rows = Array.isArray(rowsPayload) ? rowsPayload : [];
      await audit(userId, "broker_connection_saved", {
        trading_mode: sanitized.tradingMode,
        base_url: sanitized.baseUrl,
        enabled: payload.enabled,
      });
      return json({ connection: serializeBrokerConnection(rows[0] ?? existing ?? null, apiKeyPlain, secretKeyPlain) });
    }

    if (path === "/me/broker/verify" && req.method === "POST") {
      const user = await requireUser(req);
      const body = await req.json().catch(() => ({})) as JsonMap;
      const userId = String(user.id);
      let row = await fetchBrokerConnection(userId);
      const sanitized = sanitizeBrokerInput(body, row);
      const apiKey = typeof body.api_key === "string" && body.api_key.trim()
        ? body.api_key.trim()
        : row ? await decryptSecret(row.api_key_encrypted) : "";
      const secretKey = typeof body.secret_key === "string" && body.secret_key.trim()
        ? body.secret_key.trim()
        : row ? await decryptSecret(row.secret_key_encrypted) : "";

      if ((!row || body.api_key || body.secret_key || body.base_url || body.account_label || body.trading_mode) && apiKey && secretKey) {
        const payload = {
          user_id: userId,
          broker_name: "alpaca",
          account_label: sanitized.accountLabel,
          trading_mode: sanitized.tradingMode,
          base_url: sanitized.baseUrl,
          enabled: body.enabled === undefined ? (row?.enabled ?? true) : !!body.enabled,
          api_key_encrypted: await encryptSecret(apiKey),
          secret_key_encrypted: await encryptSecret(secretKey),
          last_error: row?.last_error ?? null,
          last_verified_status: row?.last_verified_status ?? "pending",
        };
        const method = row ? "PATCH" : "POST";
        const pathSuffix = row ? `/broker_connections?id=eq.${encodeURIComponent(row.id)}` : "/broker_connections";
        const saveRes = await restAdmin(pathSuffix, {
          method,
          headers: { "Prefer": "return=representation" },
          body: JSON.stringify(payload),
        });
        const savePayload = await readJsonSafe<BrokerConnectionRow[] | JsonMap>(saveRes);
        if (!saveRes.ok) {
          throw new Error(extractServiceError(savePayload, "Could not save broker connection"));
        }
        const saveRows = Array.isArray(savePayload) ? savePayload : [];
        row = saveRows[0] ?? row;
      }

      if (!row) return json({ error: "Save your broker connection first, then verify it." }, 404);
      if (!apiKey || !secretKey) return json({ error: "Enter both the Alpaca API Key and Secret Key before verification." }, 400);
      try {
        const account = await verifyAlpacaCredentials(apiKey, secretKey, row.base_url);
        const res = await restAdmin(`/broker_connections?id=eq.${encodeURIComponent(row.id)}`, {
          method: "PATCH",
          headers: { "Prefer": "return=representation" },
          body: JSON.stringify({
            enabled: true,
            last_verified_status: "verified",
            last_verified_at: new Date().toISOString(),
            last_error: null,
            metadata: { account_number: account.account_number ?? account.id ?? null },
          }),
        });
        const rows = await res.json() as BrokerConnectionRow[];
        await audit(String(user.id), "broker_connection_verified", {});
        return json({
          connection: serializeBrokerConnection(rows[0] ?? row, apiKey, secretKey),
          account: {
            id: account.id ?? null,
            account_number: account.account_number ?? null,
            status: account.status ?? null,
            buying_power: account.buying_power ?? null,
          },
        });
      } catch (e) {
        const safeError = sanitizeBrokerError(e);
        await restAdmin(`/broker_connections?id=eq.${encodeURIComponent(row.id)}`, {
          method: "PATCH",
          headers: { "Prefer": "return=minimal" },
          body: JSON.stringify({
            last_verified_status: "failed",
            last_verified_at: new Date().toISOString(),
            last_error: safeError,
          }),
        });
        return json({ error: safeError }, 400);
      }
    }

    if (path === "/me/execution/jobs" && req.method === "GET") {
      const user = await requireUser(req);
      const res = await restAdmin(`/execution_requests?user_id=eq.${encodeURIComponent(String(user.id))}&select=*&order=created_at.desc&limit=30`);
      const jobs = await res.json() as ExecutionRequestRow[];
      return json({ jobs: jobs.map((job) => sanitizeExecutionRequest(job)) });
    }

    if (path === "/me/execution/run" && req.method === "POST") {
      const user = await requireUser(req);
      const broker = await fetchBrokerConnection(String(user.id));
      if (!broker?.enabled || !broker.api_key_encrypted || !broker.secret_key_encrypted) {
        return json({ error: "Verify your Alpaca broker connection before requesting a run." }, 400);
      }
      const existingJob = await findOpenTradeRequest(String(user.id));
      if (existingJob) {
        return json({ job: sanitizeExecutionRequest(existingJob), duplicate: true }, 202);
      }
      const body = await req.json().catch(() => ({})) as JsonMap;
      const sourcePayload = typeof body.payload === "object" && body.payload ? body.payload as JsonMap : {};
      const job = await enqueueExecutionRequest(user, broker, "manual", sourcePayload);
      const dispatch: JsonMap = { status: "queued", executor: "worker_pool" };
      await audit(String(user.id), "execution_request_queued", { request_id: job?.id ?? null });
      return json({ job: sanitizeExecutionRequest(job), dispatch }, 202);
    }

    if (path === "/control" && req.method === "GET") {
      return json(await loadControlState());
    }

    if (path === "/control" && req.method === "POST") {
      const user = await requireUser(req);
      const body = await req.json().catch(() => ({})) as { action?: string };
      if (!body.action) return json({ error: "action is required" }, 400);
      const state = await applyControlAction(body.action, String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id));
      await audit(String(user.id), "control_action", { action: body.action });
      return json(state);
    }

    if (path === "/control/reset_daily_loss" && req.method === "POST") {
      const user = await requireUser(req);
      const today = new Date().toISOString().slice(0, 10);
      await restAdmin(`/daily_risk_state?user_id=eq.${encodeURIComponent(String(user.id))}&trading_day=eq.${today}`, {
        method: "PATCH",
        headers: { "Prefer": "return=minimal" },
        body: JSON.stringify({
          breached: false,
          breached_at: null,
          reset_required: false,
          metadata: { reset_by: String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id), reset_at: new Date().toISOString() },
        }),
      });
      await appendControlEvent("daily_loss_reset", { trading_day: today }, null, String(user.id));
      return json({ ok: true, trading_day: today });
    }

    if (path === "/control/run_once" && req.method === "POST") {
      const user = await requireUser(req);
      const broker = await fetchBrokerConnection(String(user.id));
      if (broker?.enabled && broker.api_key_encrypted && broker.secret_key_encrypted) {
        const existingJob = await findOpenTradeRequest(String(user.id));
        if (existingJob) {
          return json({ job: sanitizeExecutionRequest(existingJob), duplicate: true }, 202);
        }
        const actor = String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id);
        const job = await enqueueExecutionRequest(user, broker, "control_plane");
        const result = { status: "queued", executor: "worker_pool", actor };
        await audit(String(user.id), "run_once_dispatch", { request_id: job?.id ?? null });
        return json({ ...result, job: sanitizeExecutionRequest(job) }, 202);
      }
      return json({ error: "Verify your Alpaca broker connection before changing live controls." }, 400);
    }

    if (path === "/worker/execution/claim-next" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { worker_id?: string; lease_seconds?: number; capacity?: number };
      const workerId = String(body.worker_id ?? "").trim();
      if (!workerId) return json({ error: "worker_id is required" }, 400);

      await restRpc<number>("requeue_expired_execution_requests");
      const claimed = await restRpc<ExecutionRequestRow[]>("claim_execution_request", {
        p_worker_id: workerId,
        p_lease_seconds: Math.max(15, Math.min(Number(body.lease_seconds ?? 90), 600)),
      });
      const request = claimed[0] ?? null;
      await restAdmin("/worker_heartbeats", {
        method: "POST",
        headers: { "Prefer": "resolution=merge-duplicates,return=minimal" },
        body: JSON.stringify({
          worker_id: workerId,
          version: "v2",
          capacity: Math.max(1, Number(body.capacity ?? 1)),
          active_jobs: request ? 1 : 0,
          metadata: { source: "edge_api" },
          last_seen_at: new Date().toISOString(),
        }),
      });
      if (!request) {
        return json({ request: null, worker_id: workerId });
      }
      await appendExecutionEvent(String(request.user_id), request.id, "execution_leased", {
        worker_id: workerId,
        lease_expires_at: request.lease_expires_at,
      });

      const brokerRow = request.broker_connection_id ? await fetchBrokerConnection(String(request.user_id)) : null;
      const profileRes = await restAdmin(`/profiles?id=eq.${encodeURIComponent(String(request.user_id))}&select=*`);
      const profile = ((await profileRes.json()) as JsonMap[])[0] ?? null;
      if (brokerRow) {
        await audit(String(request.user_id), "broker_secret_decrypt", { request_id: request.id, worker_id: workerId });
      }
      return json({
        request: sanitizeExecutionRequest(request),
        profile,
        broker_connection: brokerRow ? {
          id: brokerRow.id,
          broker_name: brokerRow.broker_name,
          trading_mode: brokerRow.trading_mode,
          base_url: brokerRow.base_url,
          account_label: brokerRow.account_label,
          api_key: await decryptSecret(brokerRow.api_key_encrypted),
          secret_key: await decryptSecret(brokerRow.secret_key_encrypted),
        } : null,
      });
    }

    if (path === "/worker/execution/start" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { request_id?: string; worker_id?: string };
      if (!body.request_id || !body.worker_id) return json({ error: "request_id and worker_id are required" }, 400);
      const request = await fetchExecutionRequest(body.request_id);
      if (!request) return json({ error: "Request not found" }, 404);
      if (request.worker_id !== body.worker_id || request.status !== "leased") {
        return json({ error: "Request is not leased by this worker" }, 409);
      }
      const control = await loadControlState();
      const allowed = !!control.trading_enabled && !control.emergency_stop;
      const today = new Date().toISOString().slice(0, 10);
      const dailyRiskRes = await restAdmin(`/daily_risk_state?user_id=eq.${encodeURIComponent(String(request.user_id))}&trading_day=eq.${today}&select=*`);
      const dailyRiskRows = await dailyRiskRes.json() as JsonMap[];
      const dailyRisk = dailyRiskRows[0] ?? null;
      const dailyRiskBlocked = !!dailyRisk && !!dailyRisk.reset_required;
      await appendControlEvent("execution_control_check", {
        request_id: request.id,
        worker_id: body.worker_id,
        allowed,
        daily_risk_blocked: dailyRiskBlocked,
        control,
      }, request.id, String(request.user_id), Number(control.command_version ?? 1));
      if (!allowed || dailyRiskBlocked) {
        await restRpc<ExecutionRequestRow[]>("transition_execution_request", {
          p_request_id: request.id,
          p_worker_id: body.worker_id,
          p_from_status: "leased",
          p_to_status: "cancelled",
          p_error_text: dailyRiskBlocked ? "daily_loss_reset_required" : "control_plane_blocked",
        });
        return json({ error: dailyRiskBlocked ? "Daily loss reset required" : "Control plane blocked execution" }, 409);
      }
      const transitioned = await restRpc<ExecutionRequestRow[]>("transition_execution_request", {
        p_request_id: request.id,
        p_worker_id: body.worker_id,
        p_from_status: "leased",
        p_to_status: "running",
      });
      const row = transitioned[0] ?? null;
      if (!row) return json({ error: "Could not transition request to running" }, 409);
      await appendExecutionEvent(String(request.user_id), request.id, "execution_running", {
        worker_id: body.worker_id,
        control_version: control.command_version ?? 1,
      });
      return json({ ok: true, request: sanitizeExecutionRequest(row), control });
    }

    if (path === "/worker/execution/event" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { request_id?: string; stage?: string; payload?: JsonMap };
      if (!body.request_id || !body.stage) return json({ error: "request_id and stage are required" }, 400);
      const request = await fetchExecutionRequest(body.request_id);
      if (!request) return json({ error: "Request not found" }, 404);
      await appendExecutionEvent(String(request.user_id), request.id, body.stage, body.payload ?? {});
      if (body.stage === "execution_aborted" || body.stage === "execution_blocked") {
        await appendControlEvent(body.stage, body.payload ?? {}, request.id, String(request.user_id));
      }
      return json({ ok: true });
    }

    if (path === "/worker/risk/event" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { request_id?: string; event_type?: string; payload?: JsonMap };
      if (!body.request_id || !body.event_type) return json({ error: "request_id and event_type are required" }, 400);
      const request = await fetchExecutionRequest(body.request_id);
      if (!request) return json({ error: "Request not found" }, 404);
      await appendRiskEvent(String(request.user_id), request.id, body.event_type, body.payload ?? {});
      return json({ ok: true });
    }

    if (path === "/worker/orders/prepare" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { request_id?: string; order?: JsonMap };
      if (!body.request_id || !body.order) return json({ error: "request_id and order are required" }, 400);
      const request = await fetchExecutionRequest(body.request_id);
      if (!request) return json({ error: "Request not found" }, 404);
      const order = await upsertBrokerOrder(request, body.order);
      if (!order) return json({ error: "client_order_id is required" }, 400);
      await appendBrokerOrderEvent(String(request.user_id), request.id, order.id, "created", body.order);
      return json({ ok: true, order_id: order.id, client_order_id: order.client_order_id });
    }

    if (path === "/worker/orders/update" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { request_id?: string; order?: JsonMap };
      if (!body.request_id || !body.order) return json({ error: "request_id and order are required" }, 400);
      const request = await fetchExecutionRequest(body.request_id);
      if (!request) return json({ error: "Request not found" }, 404);
      const orderPayload = body.order;
      const clientOrderId = String(orderPayload.client_order_id ?? "").trim();
      if (!clientOrderId) return json({ error: "client_order_id is required" }, 400);
      const patched = await patchBrokerOrderByClientOrderId(clientOrderId, {
        broker_order_id: typeof orderPayload.order_id === "string" ? orderPayload.order_id : null,
        status: typeof orderPayload.status === "string" ? orderPayload.status.toLowerCase() : "created",
        payload: orderPayload,
        submitted_at: typeof orderPayload.status === "string" && ["submit_requested", "submit_acknowledged"].includes(orderPayload.status) ? new Date().toISOString() : undefined,
        reconciled_at: typeof orderPayload.status === "string" && ["reconciled", "filled", "cancelled", "rejected", "partial_fill"].includes(orderPayload.status) ? new Date().toISOString() : undefined,
      });
      if (!patched) return json({ error: "Broker order not found" }, 404);
      const eventType = typeof orderPayload.event_type === "string" ? orderPayload.event_type : String(orderPayload.status ?? "updated");
      await appendBrokerOrderEvent(String(request.user_id), request.id, patched.id, eventType, orderPayload);
      return json({ ok: true, broker_order_id: patched.id });
    }

    if (path === "/worker/execution/reconcile" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { request_id?: string; client_order_id?: string };
      if (!body.request_id) return json({ error: "request_id is required" }, 400);
      const request = await fetchExecutionRequest(body.request_id);
      if (!request) return json({ error: "Request not found" }, 404);
      const brokerRow = await fetchBrokerConnection(String(request.user_id));
      if (!brokerRow) return json({ error: "Broker connection missing" }, 404);
      const reconciled = await reconcileBrokerOrdersForRequest(request, brokerRow, body.client_order_id ?? null);
      return json({ ok: true, orders: reconciled });
    }

    if (path === "/worker/reconcile/open" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { limit?: number };
      const limit = Math.max(1, Math.min(Number(body.limit ?? 20), 100));
      const ordersRes = await restAdmin(`/broker_orders?status=in.(submit_requested,submit_acknowledged,partial_fill,reconciliation_pending,cancel_requested,reconciled)&order=updated_at.asc&limit=${limit}&select=execution_request_id`);
      const rows = await ordersRes.json() as Array<{ execution_request_id?: string | null }>;
      const requestIds = [...new Set(rows.map((row) => String(row.execution_request_id ?? "")).filter(Boolean))];
      const results: JsonMap[] = [];
      for (const requestId of requestIds) {
        const request = await fetchExecutionRequest(requestId);
        if (!request) continue;
        const brokerRow = await fetchBrokerConnection(String(request.user_id));
        if (!brokerRow) continue;
        const reconciled = await reconcileBrokerOrdersForRequest(request, brokerRow, null);
        results.push({ request_id: request.id, orders: reconciled });
      }
      return json({ ok: true, reconciled: results });
    }

    if (path === "/worker/execution/complete" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { request_id?: string; status?: string; result?: JsonMap; error?: string; worker_id?: string };
      if (!body.request_id || !body.status) return json({ error: "request_id and status are required" }, 400);
      const request = await fetchExecutionRequest(body.request_id);
      if (!request) return json({ error: "Request not found" }, 404);
      if (request.worker_id && body.worker_id && request.worker_id !== body.worker_id) {
        return json({ error: "Worker ownership mismatch" }, 409);
      }
      const normalizedStatus = ["succeeded", "failed", "skipped", "cancelled", "dead_letter", "reconcile_pending"].includes(body.status)
        ? body.status
        : "failed";
      await restAdmin(`/execution_requests?id=eq.${encodeURIComponent(body.request_id)}`, {
        method: "PATCH",
        headers: { "Prefer": "return=representation" },
        body: JSON.stringify({
          status: normalizedStatus,
          completed_at: new Date().toISOString(),
          result: body.result ?? {},
          error_text: body.error ?? null,
          worker_id: body.worker_id ?? request.worker_id ?? null,
          lease_expires_at: null,
        }),
      });
      if (body.result && typeof body.result === "object") {
        await persistExecutionArtifacts(request, body.result);
      }
      await appendExecutionEvent(String(request.user_id), request.id, "execution_completed", {
        status: normalizedStatus,
        worker_id: body.worker_id ?? request.worker_id ?? null,
      });
      await audit(String(request.user_id), "execution_request_completed", {
        request_id: request.id,
        status: normalizedStatus,
        worker_id: body.worker_id ?? request.worker_id ?? null,
      });
      return json({ ok: true, request_id: body.request_id, status: normalizedStatus });
    }

    if (path === "/worker/execution/heartbeat" && req.method === "POST") {
      if (!sharedWorkerAuthorized(req)) return json({ error: "Unauthorized" }, 401);
      const body = await req.json().catch(() => ({})) as { worker_id?: string; request_id?: string; lease_seconds?: number; active_jobs?: number; capacity?: number };
      const workerId = String(body.worker_id ?? "").trim();
      if (!workerId) return json({ error: "worker_id is required" }, 400);
      await restAdmin("/worker_heartbeats", {
        method: "POST",
        headers: { "Prefer": "resolution=merge-duplicates,return=minimal" },
        body: JSON.stringify({
          worker_id: workerId,
          version: "v2",
          capacity: Math.max(1, Number(body.capacity ?? 1)),
          active_jobs: Math.max(0, Number(body.active_jobs ?? 0)),
          metadata: { source: "edge_api" },
          last_seen_at: new Date().toISOString(),
        }),
      });
      let renewed = true;
      if (body.request_id) {
        renewed = await restRpc<boolean>("renew_execution_request_lease", {
          p_request_id: body.request_id,
          p_worker_id: workerId,
          p_lease_seconds: Math.max(15, Math.min(Number(body.lease_seconds ?? 90), 600)),
        });
      }
      return json({ ok: true, worker_id: workerId, renewed });
    }

    if (path === "/worker/jobs/claim" && req.method === "POST") {
      const body = await req.json().catch(() => ({})) as { job_id?: string };
      if (!body.job_id) return json({ error: "job_id is required" }, 400);
      const jobRes = await restAdmin(`/execution_jobs?id=eq.${encodeURIComponent(body.job_id)}&select=*`);
      const jobRows = await jobRes.json() as JsonMap[];
      const job = jobRows[0];
      if (!job) return json({ error: "Job not found" }, 404);
      const expectedToken = String(((job.payload as JsonMap | undefined)?.claim_token ?? ""));
      if (!workerAuthorized(req, expectedToken)) return json({ error: "Unauthorized" }, 401);
      if (!["queued", "claimed"].includes(String(job.status ?? ""))) {
        return json({ error: "Job is not claimable", job: sanitizeJob(job) }, 409);
      }
      await restAdmin(`/execution_jobs?id=eq.${encodeURIComponent(body.job_id)}`, {
        method: "PATCH",
        headers: { "Prefer": "return=representation" },
        body: JSON.stringify({
          status: "running",
          started_at: new Date().toISOString(),
        }),
      });
      const broker = job.broker_connection_id
        ? await restAdmin(`/broker_connections?id=eq.${encodeURIComponent(String(job.broker_connection_id))}&select=*`)
        : null;
      const brokerRow = broker ? ((await broker.json()) as BrokerConnectionRow[])[0] ?? null : null;
      const profileRes = await restAdmin(`/profiles?id=eq.${encodeURIComponent(String(job.user_id))}&select=*`);
      const profile = ((await profileRes.json()) as JsonMap[])[0] ?? null;
      return json({
        job: sanitizeJob({
          ...job,
          status: "running",
        }),
        profile,
        broker_connection: brokerRow ? {
          id: brokerRow.id,
          broker_name: brokerRow.broker_name,
          trading_mode: brokerRow.trading_mode,
          base_url: brokerRow.base_url,
          account_label: brokerRow.account_label,
          api_key: await decryptSecret(brokerRow.api_key_encrypted),
          secret_key: await decryptSecret(brokerRow.secret_key_encrypted),
        } : null,
      });
    }

    if (path === "/worker/jobs/complete" && req.method === "POST") {
      const body = await req.json().catch(() => ({})) as { job_id?: string; status?: string; result?: JsonMap; error?: string };
      if (!body.job_id || !body.status) return json({ error: "job_id and status are required" }, 400);
      const jobRes = await restAdmin(`/execution_jobs?id=eq.${encodeURIComponent(body.job_id)}&select=*`);
      const jobRows = await jobRes.json() as JsonMap[];
      const job = jobRows[0];
      if (!job) return json({ error: "Job not found" }, 404);
      const expectedToken = String(((job.payload as JsonMap | undefined)?.claim_token ?? ""));
      if (!workerAuthorized(req, expectedToken)) return json({ error: "Unauthorized" }, 401);
      const normalizedStatus = ["succeeded", "failed", "skipped", "cancelled"].includes(body.status) ? body.status : "failed";
      await restAdmin(`/execution_jobs?id=eq.${encodeURIComponent(body.job_id)}`, {
        method: "PATCH",
        headers: { "Prefer": "return=representation" },
        body: JSON.stringify({
          status: normalizedStatus,
          completed_at: new Date().toISOString(),
          result: body.result ?? {},
          error_text: body.error ?? null,
        }),
      });
      return json({ ok: true, job_id: body.job_id, status: normalizedStatus });
    }

    return json({ error: "Not found" }, 404);
  } catch (e) {
    const message = String(e);
    if (message === "Unauthorized" || message.endsWith("Unauthorized")) return json({ error: "Unauthorized" }, 401);
    if (message.includes("Rate limit exceeded")) return json({ error: "Too many attempts. Please wait a minute and try again." }, 429);
    if (message === "Verified broker connection required") return json({ error: message }, 412);
    return json({ error: message }, 500);
  }
});
