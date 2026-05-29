/**
 * QuantPulse — Supabase Edge Function
 * Proxies Alpaca Paper Trading API & Yahoo Finance.
 * Keys are stored as Supabase secrets — never exposed to the browser.
 *
 * Deploy:
 *   supabase secrets set ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
 *   supabase functions deploy api --no-verify-jwt
 */

const ALPACA_KEY    = Deno.env.get("ALPACA_API_KEY")    ?? "";
const ALPACA_SECRET = Deno.env.get("ALPACA_SECRET_KEY") ?? "";
const ALPACA_BASE   = "https://paper-api.alpaca.markets/v2";
const GITHUB_TOKEN  = Deno.env.get("GITHUB_TOKEN")      ?? "";
const GITHUB_REPO   = Deno.env.get("GITHUB_REPOSITORY") ?? "owandakin-coder/agent-analyst";
const GITHUB_BRANCH = Deno.env.get("ATZMA_GITHUB_BRANCH") ?? "main";
const CONTROL_PATH  = Deno.env.get("ATZMA_CONTROL_STATE_PATH") ?? "runtime/control_state.json";
const TRADE_WORKFLOW = Deno.env.get("ATZMA_TRADE_WORKFLOW") ?? "trade.yml";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin":  "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, apikey",
};

// ── Alpaca helper ─────────────────────────────────────────────────────────────
async function alpacaGet(path: string, params = ""): Promise<unknown> {
  let url = `${ALPACA_BASE}${path}`;
  if (params) url += `?${params}`;
  const res = await fetch(url, {
    headers: {
      "APCA-API-KEY-ID":     ALPACA_KEY,
      "APCA-API-SECRET-KEY": ALPACA_SECRET,
      "Accept":              "application/json",
    },
  });
  if (!res.ok) throw new Error(`Alpaca ${path} returned ${res.status}`);
  return res.json();
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

function normalizeControlState(input: Record<string, unknown> | null | undefined) {
  const state = { ...defaultControlState(), ...(input ?? {}) } as Record<string, unknown>;
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

async function githubRequest(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/vnd.github+json");
  headers.set("User-Agent", "ATZMA-ControlPlane/1.0");
  if (GITHUB_TOKEN) headers.set("Authorization", `Bearer ${GITHUB_TOKEN}`);
  return fetch(`https://api.github.com${path}`, { ...init, headers });
}

async function loadControlState(): Promise<Record<string, unknown>> {
  const res = await githubRequest(`/repos/${GITHUB_REPO}/contents/${CONTROL_PATH}`);
  if (res.status === 404) {
    return { ...defaultControlState(), _source: "default" };
  }
  if (!res.ok) {
    throw new Error(`GitHub control state returned ${res.status}`);
  }
  const payload = await res.json() as { content?: string };
  const encoded = payload.content ?? "";
  const decoded = encoded ? atob(encoded.replace(/\n/g, "")) : "{}";
  const parsed = JSON.parse(decoded) as Record<string, unknown>;
  return { ...normalizeControlState(parsed), _source: "github_repo" };
}

async function saveControlState(state: Record<string, unknown>, actor: string): Promise<Record<string, unknown>> {
  if (!GITHUB_TOKEN) {
    throw new Error("GITHUB_TOKEN secret is missing");
  }
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

  const content = btoa(JSON.stringify(normalized, null, 2));
  const body: Record<string, unknown> = {
    message: `ATZMA control update by ${actor}`,
    content,
    branch: GITHUB_BRANCH,
  };
  if (sha) body.sha = sha;

  const res = await githubRequest(urlPath, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`GitHub control save returned ${res.status}`);
  }
  return { ...normalized, _source: "github_repo" };
}

async function applyControlAction(action: string, actor: string): Promise<Record<string, unknown>> {
  const state = await loadControlState();
  const next = { ...state } as Record<string, unknown>;
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
  if (!GITHUB_TOKEN) {
    throw new Error("GITHUB_TOKEN secret is missing");
  }
  const res = await githubRequest(`/repos/${GITHUB_REPO}/actions/workflows/${TRADE_WORKFLOW}/dispatches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref: GITHUB_BRANCH }),
  });
  if (!res.ok) {
    throw new Error(`GitHub workflow dispatch returned ${res.status}`);
  }
  return {
    status: "dispatched",
    workflow: TRADE_WORKFLOW,
    branch: GITHUB_BRANCH,
    repo: GITHUB_REPO,
    actor,
    dispatched_at: new Date().toISOString(),
  };
}

// ── Yahoo Finance helper ──────────────────────────────────────────────────────
async function fetchYF(symbol: string): Promise<Record<string, unknown>> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=1d&interval=1d`;
  try {
    const res = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json",
        "Referer":    "https://finance.yahoo.com/",
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

// ── JSON response helper ──────────────────────────────────────────────────────
function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

// ── Main handler ──────────────────────────────────────────────────────────────
Deno.serve(async (req: Request): Promise<Response> => {
  // CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: CORS_HEADERS });
  }

  const url  = new URL(req.url);
  // Strip everything up to and including the function name "api"
  // e.g. /functions/v1/api/account  →  /account
  //      /account                   →  /account  (local dev)
  const rawPath = url.pathname;
  const funcIdx = rawPath.indexOf("/api");
  const path = funcIdx >= 0 ? rawPath.slice(funcIdx + 4) || "/" : rawPath;
  const qs   = url.searchParams;

  try {
    // ── GET /account ──────────────────────────────────────────────────────────
    if (path === "/account") {
      const data = await alpacaGet("/account") as Record<string, string>;
      return json({
        equity:          parseFloat(data.equity          ?? "0"),
        last_equity:     parseFloat(data.last_equity     ?? "0"),
        cash:            parseFloat(data.cash            ?? "0"),
        buying_power:    parseFloat(data.buying_power    ?? "0"),
        portfolio_value: parseFloat(data.portfolio_value ?? "0"),
      });
    }

    // ── GET /positions ────────────────────────────────────────────────────────
    if (path === "/positions") {
      const positions = await alpacaGet("/positions") as Array<Record<string, string>>;
      return json(positions.map(p => ({
        symbol:          p.symbol,
        qty:             parseFloat(p.qty            ?? "0"),
        avg_entry:       parseFloat(p.avg_entry_price ?? "0"),
        current_price:   parseFloat(p.current_price  ?? "0"),
        market_value:    parseFloat(p.market_value   ?? "0"),
        unrealized_pl:   parseFloat(p.unrealized_pl  ?? "0"),
        unrealized_plpc: parseFloat(p.unrealized_plpc ?? "0"),
        change_today:    parseFloat(p.change_today   ?? "0"),
      })));
    }

    // ── GET /orders ───────────────────────────────────────────────────────────
    if (path === "/orders") {
      const orders = await alpacaGet("/orders", "status=closed&limit=50&direction=desc") as Array<Record<string, string>>;
      return json(
        orders
          .filter(o => o.filled_at && o.filled_avg_price)
          .map(o => ({
            id:     o.id,
            symbol: o.symbol,
            side:   o.side,
            qty:    parseFloat(o.filled_qty ?? o.qty ?? "0"),
            price:  parseFloat(o.filled_avg_price ?? "0"),
            time:   o.filled_at ?? "",
            type:   o.type ?? "",
          }))
      );
    }

    // ── GET /history ──────────────────────────────────────────────────────────
    if (path === "/history") {
      const period    = qs.get("period")    ?? "1M";
      const timeframe = qs.get("timeframe") ?? "1D";
      const data = await alpacaGet(
        "/account/portfolio/history",
        `period=${period}&timeframe=${timeframe}&intraday_reporting=market_hours`
      ) as { timestamp: number[]; equity: number[] };
      return json({ timestamps: data.timestamp ?? [], equity: data.equity ?? [] });
    }

    // ── GET /quotes ───────────────────────────────────────────────────────────
    if (path === "/quotes") {
      const symbolsStr = qs.get("symbols") ?? "AAPL,NVDA,MSFT,JPM,META";
      const symbols    = symbolsStr.split(",").map(s => s.trim()).filter(Boolean);
      const quotes     = await Promise.all(symbols.map(fetchYF));
      return json({ quotes });
    }

    if (path === "/control" && req.method === "GET") {
      return json(await loadControlState());
    }

    if (path === "/control" && req.method === "POST") {
      const body = await req.json().catch(() => ({})) as { action?: string; actor?: string };
      if (!body.action) return json({ error: "action is required" }, 400);
      return json(await applyControlAction(body.action, body.actor ?? "dashboard_remote"));
    }

    if (path === "/control/run_once" && req.method === "POST") {
      const body = await req.json().catch(() => ({})) as { actor?: string };
      return json(await dispatchTradeWorkflow(body.actor ?? "dashboard_remote"), 202);
    }

    return json({ error: "Not found" }, 404);

  } catch (e) {
    return json({ error: String(e) }, 500);
  }
});
