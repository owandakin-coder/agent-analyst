# ATZMA Launch Checklist

This is the minimum release checklist for launching ATZMA tomorrow without guessing.

## 1. Core user flow

- Open [https://owandakin-coder.github.io/agent-analyst/](https://owandakin-coder.github.io/agent-analyst/)
- Confirm the app opens in `Locked` state
- Confirm no live portfolio numbers are shown before login
- Create a new account
- Confirm the success message explains email confirmation
- Confirm the email arrives
- Click the confirmation link from the email
- Confirm the site either signs the user in automatically or shows a clear `Email confirmed` message
- Sign in manually if auto-session is not returned

## 2. Broker onboarding

- Open `More -> Broker Connection`
- Confirm the help copy explains where `API Key` and `Secret Key` come from
- Confirm the default Paper URL is `https://paper-api.alpaca.markets`
- Save a Paper Alpaca connection
- Click `Verify`
- Confirm the connection status becomes `Verified`
- Confirm `account / positions / orders` appear only after verification
- Confirm invalid or mismatched keys return a clear user-facing message

## 3. Execution flow

- Open `Execution Queue`
- Click `Run Now`
- Confirm a job is created
- Confirm job status moves from `queued` to `running` to `succeeded` or `skipped`
- Confirm no duplicate run is created from the same manual action
- Confirm the latest job contains a readable decision summary
- Confirm the summary mentions regime or agent reasoning, not only raw status
- Confirm the persistent worker is online and claiming jobs

## 3.5. Multi-agent and regime checks

- Confirm the decision layer blocks trades when the vote is not unanimous
- Confirm high-volatility or crash regimes reduce or fully block long exposure
- Confirm at least one successful run writes a decision explanation for the dashboard

## 4. Control plane

- Confirm `Pause`, `Resume`, and `Emergency Stop` update correctly
- Confirm remote control state shows `executor = worker_pool`
- Confirm `can_dispatch = false`

## 5. Environment and secrets

Required for launch:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `SUPABASE_ACCESS_TOKEN`
- `ATZMA_BROKER_CREDENTIAL_KEY`
- `ATZMA_WORKER_SHARED_TOKEN`

Recommended:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `DISCORD_WEBHOOK_URL`

## 5.5. Legal and trust surface

- Confirm `terms.html` opens from the auth screen
- Confirm `privacy.html` opens from the auth screen
- Confirm the product clearly states that users should start in Paper mode

## 6. Validation commands

Run before launch:

```powershell
python health_check.py
python -m pytest -q tests/test_multi_agent.py tests/test_user_execution_worker.py tests/test_control_plane.py tests/test_config.py tests/test_idempotency.py tests/test_broker.py tests/test_market_sync.py tests/test_end_to_end.py
```

## 7. Go / no-go

Go only if all are true:

- Auth flow works end-to-end
- Broker verify works with a real Paper account
- Execution queue can create and finish a job
- Persistent worker is running on Render
- Multi-agent decisions produce readable explanations
- Health check passes
- Test suite passes
- Guest users see only `Locked` state
- `/health` returns `200`
- `/control` returns `executor=worker_pool`
- Duplicate `Run Now` requests do not create extra queued jobs

No-go if any of these fail:

- Confirmation emails do not arrive
- Signed-in users still cannot verify Alpaca
- Guest users still see live portfolio data
- Jobs stay stuck in `queued`
