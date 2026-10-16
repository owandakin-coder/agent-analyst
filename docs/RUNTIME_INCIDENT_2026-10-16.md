# Open-market Paper runtime repair

At 18:09 UTC monitoring found 38 historical dead letters and 15 failed requests.
The worker heartbeat and scheduler were current; restarting would not fix this.
Database event stages, Postgres logs and completed Actions traces established:

* The daily risk observation used a generic merge against a surrogate primary
  key instead of the unique user/day key. Later cycles raised
  `daily_risk_state_user_day_uniq`. Replaced it with an atomic SQL function that
  retains the original baseline and never clears breach/reset latches.
* Two stored Paper connections failed credential decryption after claim. Their
  leases previously expired three times without a useful error. The claim path
  now fails the owned request explicitly with a safe reverification reason and
  invalidates verification only if the connection was not edited concurrently.
  Existing naturally observed failures were used to invalidate stale verification;
  enabled flags, user pauses, risk stops and failed request history were retained.
* The next natural 18:15 cycle confirmed the risk write returned HTTP 204, then
  exposed a second error: `invalid_broker_order_transition:filled->rejected`.
  The duplicate-order guard was trying to reject the original filled order.
  Duplicate attempts now append an event without changing the original order or
  its fill payload. Existing order references must match user, connection and
  broker account before reuse or mutation. Duplicate prevention remains active.

GET broker settings can now render an unreadable connection without exposing
partial decrypted data. Saving a complete replacement key pair works even when
the old pair cannot be decrypted; incomplete replacements are rejected and new
credentials still require verification. No credentials were printed or replaced
by the monitor. No historical failed request was replayed and no trade was forced.

Validation: ten Deno tests and API type checking; database transaction/rollback
checks for repeated risk observations, unchanged baseline, sticky breach/reset
flags and service-role-only privileges. `verify-daily-risk` makes that integration
check repeatable. Deploy database function before the Edge API. A later scheduled
cycle is required to confirm the combined repair end to end. Historical dead
letters remain visible in health for their normal 24-hour window.
