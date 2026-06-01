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
- Save a Paper Alpaca connection
- Click `Verify`
- Confirm the connection status becomes `Verified`
- Confirm `account / positions / orders` appear only after verification

## 3. Execution flow

- Open `Execution Queue`
- Click `Run Now`
- Confirm a job is created
- Confirm job status moves from `queued` to `running` to `succeeded` or `skipped`
- Confirm no duplicate run is created from the same manual action

## 4. Control plane

- Confirm `Pause`, `Resume`, and `Emergency Stop` update correctly
- Confirm `Run Once` is disabled when control dispatch is unavailable
- Confirm GitHub Actions still runs from the shared control state

## 5. Environment and secrets

Required for launch:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `GITHUB_TOKEN`
- `SUPABASE_ACCESS_TOKEN`
- `ATZMA_BROKER_CREDENTIAL_KEY`

Recommended:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## 6. Validation commands

Run before launch:

```powershell
python health_check.py
python -m pytest -q tests/test_user_execution_worker.py tests/test_control_plane.py tests/test_config.py tests/test_idempotency.py tests/test_broker.py tests/test_market_sync.py tests/test_end_to_end.py
```

## 7. Go / no-go

Go only if all are true:

- Auth flow works end-to-end
- Broker verify works with a real Paper account
- Execution queue can create and finish a job
- Health check passes
- Test suite passes
- Guest users see only `Locked` state

No-go if any of these fail:

- Confirmation emails do not arrive
- Signed-in users still cannot verify Alpaca
- Guest users still see live portfolio data
- Jobs stay stuck in `queued`
