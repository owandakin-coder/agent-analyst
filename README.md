# ATZMA

ATZMA is an autonomous trading platform built around an RL decision engine, a protected web app, and a paper/live execution pipeline.

## What It Includes

- Static frontend on GitHub Pages
- Backend API on Supabase Edge Functions
- Control plane backed by GitHub Actions
- Alpaca broker integration
- Multi-agent decision engine with unanimous execution voting
- Market regime detection with adaptive strategy modes
- Explainable execution summaries stored per run
- Per-user broker connections and isolated execution jobs
- Guest `Locked` state with no live portfolio access
- Health checks, launch checklist, and automated tests

## Core Architecture

- Frontend: [https://owandakin-coder.github.io/agent-analyst/](https://owandakin-coder.github.io/agent-analyst/)
- Backend API: `supabase/functions/api/index.ts`
- Local dashboard server: `dashboard_app/server.py`
- Trading runtime: `main.py`, `live_trader.py`, `broker_api.py`, `risk_manager.py`
- Decision layer: `multi_agent.py`, `regime_detector.py`, `decision_journal.py`
- Per-user worker: `user_execution_worker.py`

## Decision Stack

ATZMA no longer relies on a single opaque model output.

Each live trading cycle now flows through:

1. `RegimeDetector`
   - Classifies the market as `TRENDING_UP`, `TRENDING_DOWN`, `RANGE_BOUND`, `HIGH_VOLATILITY`, or `CRASH_CORRECTION`
   - Sets adaptive exposure through the risk manager

2. `RL proposal`
   - The trained model still proposes a raw action vector

3. `MultiAgentDecisionEngine`
   - `Trend Agent` validates direction with MA / MACD structure
   - `Entry Agent` validates timing with RSI / Bollinger context
   - `Defense Agent` can veto new risk or force defense
   - Orders only proceed on unanimous agreement

4. `Explainability`
   - The final decision bundle is persisted
   - Execution jobs carry a human-readable decision summary
   - Dashboard surfaces the latest decision context

## Environment Setup

1. Copy `.env.example` to `.env`
2. Fill only your own secrets
3. Never commit `.env`

Required secrets:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `GITHUB_TOKEN`
- `GITHUB_REPOSITORY`
- `ATZMA_BROKER_CREDENTIAL_KEY`

Optional:

- `SUPABASE_ACCESS_TOKEN`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `DISCORD_WEBHOOK_URL`

## Local Development

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the local dashboard:

```powershell
python dashboard_app/server.py
```

Run a fast health check:

```powershell
python health_check.py --fast
```

Run the full health check:

```powershell
python health_check.py
```

## Trading Modes

Download data:

```powershell
python main.py --mode download
```

Train:

```powershell
python main.py --mode train
```

Backtest:

```powershell
python main.py --mode simulate
```

Paper live loop:

```powershell
python main.py --mode live_paper --auto-approve
```

Single live cycle:

```powershell
python main.py --mode live_once --auto-approve
```

## Tests

Run the launch-critical suite:

```powershell
python -m pytest -q tests/test_multi_agent.py tests/test_user_execution_worker.py tests/test_control_plane.py tests/test_config.py tests/test_idempotency.py tests/test_broker.py tests/test_market_sync.py tests/test_end_to_end.py
```

Run everything:

```powershell
python -m pytest -q
```

## Production Notes

- Keep GitHub Actions as the single automated executor
- Keep Supabase auth email confirmation enabled
- Use Paper mode first
- Only verified broker connections may access portfolio endpoints
- Broker secrets are encrypted server-side and never returned to the client

## Contributing

1. Create a branch
2. Make small, reviewable changes
3. Run tests and `health_check.py`
4. Open a pull request

## Launch Readiness

See `LAUNCH_CHECKLIST.md` before every production launch.
