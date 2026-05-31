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

type JsonMap = Record<string, unknown>;

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

function getBearerToken(req: Request): string | null {
  const auth = req.headers.get("Authorization") ?? "";
  return auth.startsWith("Bearer ") ? auth.slice(7).trim() : null;
}

async function alpacaGet(path: string, params = ""): Promise<unknown> {
  let url = `${ALPACA_BASE}${path}`;
  if (params) url += `?${params}`;
  const res = await fetch(url, {
    headers: {
      "APCA-API-KEY-ID": ALPACA_KEY,
      "APCA-API-SECRET-KEY": ALPACA_SECRET,
      "Accept": "application/json",
    },
  });
  if (!res.ok) throw new Error(`Alpaca ${path} returned ${res.status}`);
  return res.json();
}

async function fetchYF(symbol: string): Promise<JsonMap> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=1d&interval=1d`;
  try {
    const res = await fetch(url, {
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
  return fetch(`https://api.github.com${path}`, { ...init, headers });
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

async function dispatchTradeWorkflow(actor: string) {
  if (!GITHUB_TOKEN) throw new Error("GITHUB_TOKEN secret is missing");
  const res = await githubRequest(`/repos/${GITHUB_REPO}/actions/workflows/${TRADE_WORKFLOW}/dispatches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref: GITHUB_BRANCH }),
  });
  if (!res.ok) throw new Error(`GitHub workflow dispatch returned ${res.status}`);
  return {
    status: "dispatched",
    workflow: TRADE_WORKFLOW,
    branch: GITHUB_BRANCH,
    repo: GITHUB_REPO,
    actor,
    dispatched_at: new Date().toISOString(),
  };
}

async function authRequest(path: string, init: RequestInit = {}, useServiceRole = false): Promise<Response> {
  const headers = new Headers(init.headers);
  const apiKey = useServiceRole ? SUPABASE_SERVICE_ROLE_KEY : SUPABASE_ANON_KEY;
  if (!apiKey) throw new Error("Supabase auth secret is missing");
  headers.set("apikey", apiKey);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  return fetch(`${SUPABASE_URL}${path}`, { ...init, headers });
}

async function restAdmin(path: string, init: RequestInit = {}): Promise<Response> {
  if (!SUPABASE_SERVICE_ROLE_KEY) throw new Error("SUPABASE_SERVICE_ROLE_KEY secret is missing");
  const headers = new Headers(init.headers);
  headers.set("apikey", SUPABASE_SERVICE_ROLE_KEY);
  headers.set("Authorization", `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  return fetch(`${SUPABASE_URL}/rest/v1${path}`, { ...init, headers });
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

Deno.serve(async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") return new Response(null, { headers: CORS_HEADERS });

  const url = new URL(req.url);
  const rawPath = url.pathname;
  const funcIdx = rawPath.indexOf("/api");
  const path = funcIdx >= 0 ? rawPath.slice(funcIdx + 4) || "/" : rawPath;
  const qs = url.searchParams;

  try {
    if (path === "/account") {
      const data = await alpacaGet("/account") as Record<string, string>;
      return json({
        equity: parseFloat(data.equity ?? "0"),
        last_equity: parseFloat(data.last_equity ?? "0"),
        cash: parseFloat(data.cash ?? "0"),
        buying_power: parseFloat(data.buying_power ?? "0"),
        portfolio_value: parseFloat(data.portfolio_value ?? "0"),
        account_type: data.account_blocked ? "restricted" : "paper",
        id: data.account_number ?? data.id ?? "",
        status: data.status ?? "active",
      });
    }

    if (path === "/positions") {
      const positions = await alpacaGet("/positions") as Array<Record<string, string>>;
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
      const orders = await alpacaGet("/orders", "status=closed&limit=50&direction=desc") as Array<Record<string, string>>;
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
      const data = await alpacaGet(
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
      const result = await dispatchTradeWorkflow(String((user.user_metadata as JsonMap | undefined)?.display_name ?? user.email ?? user.id));
      await audit(String(user.id), "run_once_dispatch", {});
      return json(result, 202);
    }

    return json({ error: "Not found" }, 404);
  } catch (e) {
    const message = String(e);
    if (message === "Unauthorized" || message.endsWith("Unauthorized")) return json({ error: "Unauthorized" }, 401);
    return json({ error: message }, 500);
  }
});
