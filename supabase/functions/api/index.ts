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

const GITHUB_TOKEN = Deno.env.get("GITHUB_TOKEN") ?? "";
const GITHUB_REPO = Deno.env.get("GITHUB_REPOSITORY") ?? "owandakin-coder/agent-analyst";
const GITHUB_BRANCH = Deno.env.get("ATZMA_GITHUB_BRANCH") ?? "main";
const CONTROL_PATH = Deno.env.get("ATZMA_CONTROL_STATE_PATH") ?? "runtime/control_state.json";
const TRADE_WORKFLOW = Deno.env.get("ATZMA_TRADE_WORKFLOW") ?? "trade.yml";

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
};

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

async function githubRequest(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/vnd.github+json");
  headers.set("User-Agent", "ATZMA-ControlPlane/1.0");
  if (GITHUB_TOKEN) headers.set("Authorization", `Bearer ${GITHUB_TOKEN}`);
  return fetchWithRetry(`https://api.github.com${path}`, { ...init, headers });
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
    executor: "github_actions",
    executor_label: "GitHub Actions",
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
  const res = await githubRequest(`/repos/${GITHUB_REPO}/contents/${CONTROL_PATH}`);
  if (res.status === 404) return { ...defaultControlState(), _source: "default", can_dispatch: !!GITHUB_TOKEN };
  if (!res.ok) throw new Error(`GitHub control state returned ${res.status}`);
  const payload = await res.json() as { content?: string };
  const encoded = payload.content ?? "";
  const decoded = encoded ? atob(encoded.replace(/\n/g, "")) : "{}";
  const parsed = JSON.parse(decoded) as JsonMap;
  return { ...normalizeControlState(parsed), _source: "github_repo", can_dispatch: !!GITHUB_TOKEN };
}

async function saveControlState(state: JsonMap, actor: string): Promise<JsonMap> {
  if (!GITHUB_TOKEN) throw new Error("GITHUB_TOKEN secret is missing");
  const normalized = normalizeControlState(state);
  const urlPath = `/repos/${GITHUB_REPO}/contents/${CONTROL_PATH}`;
  const existing = await githubRequest(urlPath);
  let sha: string | undefined;
  if (existing.ok) {
    const payload = await existing.json() as { sha?: string };
    sha = payload.sha;
  } else if (existing.status !== 404) {
    throw new Error(`GitHub control state returned ${existing.status}`);
  }

  const body: JsonMap = {
    message: `ATZMA control update by ${actor}`,
    content: btoa(JSON.stringify(normalized, null, 2)),
    branch: GITHUB_BRANCH,
  };
  if (sha) body.sha = sha;

  const res = await githubRequest(urlPath, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`GitHub control save returned ${res.status}`);
  return { ...normalized, _source: "github_repo", can_dispatch: true };
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

async function dispatchTradeWorkflow(actor: string, inputs: JsonMap = {}) {
  if (!GITHUB_TOKEN) throw new Error("GITHUB_TOKEN secret is missing");
  const res = await githubRequest(`/repos/${GITHUB_REPO}/actions/workflows/${TRADE_WORKFLOW}/dispatches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref: GITHUB_BRANCH, inputs }),
  });
  if (!res.ok) throw new Error(`GitHub workflow dispatch returned ${res.status}`);
  return {
    status: "dispatched",
    workflow: TRADE_WORKFLOW,
    branch: GITHUB_BRANCH,
    repo: GITHUB_REPO,
    actor,
    inputs,
    dispatched_at: new Date().toISOString(),
  };
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
    body: JSON.stringify({ email, display_name: displayName }),
  });

  const profileRes = await restAdmin(`/profiles?id=eq.${encodeURIComponent(id)}&select=id`);
  const profileRows = await profileRes.json() as JsonMap[];
  if (!profileRows.length) {
    await restAdmin("/profiles", {
      method: "POST",
      headers: { "Prefer": "return=minimal" },
      body: JSON.stringify({ id, email, display_name: displayName }),
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

async function fetchBrokerConnection(userId: string): Promise<BrokerConnectionRow | null> {
  const res = await restAdmin(`/broker_connections?user_id=eq.${encodeURIComponent(userId)}&select=*`);
  const rows = await res.json() as BrokerConnectionRow[];
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
  const baseUrl = typeof body.base_url === "string" && /^https:\/\/[A-Za-z0-9.-]+$/.test(body.base_url.trim())
    ? body.base_url.trim()
    : existing?.base_url ?? (tradingMode === "live" ? "https://api.alpaca.markets" : "https://paper-api.alpaca.markets");
  const accountLabel = typeof body.account_label === "string"
    ? body.account_label.replace(/[<>{}]/g, "").trim().slice(0, 80)
    : existing?.account_label ?? null;
  return { tradingMode, baseUrl, accountLabel };
}

function sanitizeBrokerError(error: unknown): string {
  const text = String(error || "").toLowerCase();
  if (text.includes("unauthorized") || text.includes("forbidden") || text.includes("revoked")) {
    return "Broker disconnected";
  }
  return "Broker disconnected";
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
        control_plane: !!GITHUB_TOKEN,
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
      const symbolsStr = qs.get("symbols") ?? "AAPL,NVDA,MSFT,JPM,META";
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
        active_executor: typeof body.active_executor === "string" ? body.active_executor : "github_actions",
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
      const rows = await res.json() as BrokerConnectionRow[];
      await audit(userId, "broker_connection_saved", {
        trading_mode: sanitized.tradingMode,
        base_url: sanitized.baseUrl,
        enabled: payload.enabled,
      });
      return json({ connection: serializeBrokerConnection(rows[0] ?? existing ?? null, apiKeyPlain, secretKeyPlain) });
    }

    if (path === "/me/broker/verify" && req.method === "POST") {
      const user = await requireUser(req);
      const row = await fetchBrokerConnection(String(user.id));
      if (!row) return json({ error: "Broker connection not found" }, 404);
      const apiKey = await decryptSecret(row.api_key_encrypted);
      const secretKey = await decryptSecret(row.secret_key_encrypted);
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
      const res = await restAdmin(`/execution_jobs?user_id=eq.${encodeURIComponent(String(user.id))}&select=*&order=requested_at.desc&limit=30`);
      const jobs = await res.json() as JsonMap[];
      return json({ jobs: jobs.map((job) => sanitizeJob(job)) });
    }

    if (path === "/me/execution/run" && req.method === "POST") {
      const user = await requireUser(req);
      const broker = await fetchBrokerConnection(String(user.id));
      if (!broker?.enabled || !broker.api_key_encrypted || !broker.secret_key_encrypted) {
        return json({ error: "Verified broker connection is required" }, 400);
      }
      const existingJob = await findOpenTradeJob(String(user.id));
      if (existingJob) {
        return json({ job: sanitizeJob(existingJob), duplicate: true }, 202);
      }
      const body = await req.json().catch(() => ({})) as JsonMap;
      const actor = String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id);
      const requestedMode = broker.trading_mode === "live" ? "live" : "paper";
      const claimToken = createJobClaimToken();
      const sourcePayload = typeof body.payload === "object" && body.payload ? body.payload as JsonMap : {};
      const res = await restAdmin("/execution_jobs", {
        method: "POST",
        headers: { "Prefer": "return=representation" },
        body: JSON.stringify({
          user_id: String(user.id),
          broker_connection_id: broker.id,
          job_type: "trade_once",
          status: "queued",
          actor,
          requested_mode: requestedMode,
          payload: { ...sourcePayload, claim_token: claimToken },
        }),
      });
      const rows = await res.json() as JsonMap[];
      const job = rows[0] ?? null;
      const dispatch = await dispatchTradeWorkflow(actor, {
        job_id: String(job?.id ?? ""),
        user_id: String(user.id),
        claim_token: claimToken,
      });
      await restAdmin(`/execution_jobs?id=eq.${encodeURIComponent(String(job?.id ?? ""))}`, {
        method: "PATCH",
        headers: { "Prefer": "return=minimal" },
        body: JSON.stringify({ workflow_run_id: String(dispatch.dispatched_at ?? "") }),
      });
      await audit(String(user.id), "execution_job_requested", { job_id: job?.id ?? null });
      return json({ job: sanitizeJob(job), dispatch }, 202);
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

    if (path === "/control/run_once" && req.method === "POST") {
      const user = await requireUser(req);
      const broker = await fetchBrokerConnection(String(user.id));
      if (broker?.enabled && broker.api_key_encrypted && broker.secret_key_encrypted) {
        const existingJob = await findOpenTradeJob(String(user.id));
        if (existingJob) {
          return json({ job: sanitizeJob(existingJob), duplicate: true }, 202);
        }
        const actor = String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id);
        const claimToken = createJobClaimToken();
        const res = await restAdmin("/execution_jobs", {
          method: "POST",
          headers: { "Prefer": "return=representation" },
          body: JSON.stringify({
            user_id: String(user.id),
            broker_connection_id: broker.id,
            job_type: "trade_once",
            status: "queued",
            actor,
            requested_mode: broker.trading_mode === "live" ? "live" : "paper",
            payload: { claim_token: claimToken },
          }),
        });
        const rows = await res.json() as JsonMap[];
        const job = rows[0] ?? null;
        const result = await dispatchTradeWorkflow(actor, {
          job_id: String(job?.id ?? ""),
          user_id: String(user.id),
          claim_token: claimToken,
        });
        await audit(String(user.id), "run_once_dispatch", { job_id: job?.id ?? null });
        return json({ ...result, job: sanitizeJob(job) }, 202);
      }
      const result = await dispatchTradeWorkflow(String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id));
      await audit(String(user.id), "run_once_dispatch", {});
      return json(result, 202);
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
