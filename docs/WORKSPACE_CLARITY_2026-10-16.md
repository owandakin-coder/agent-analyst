# Workspace clarity and evidence

The overview now answers three questions in Hebrew before the account totals:
whether this account is connected, what its latest recorded cycle did, and the
next action. A disclosure explains each destination, Paper trading, empty data,
and the difference between a completed cycle and a broker-confirmed fill.
The main connected-account action opens the execution journal; it sends no run.

Health, control state, broker data and execution history are separate sources.
Market state requires fresh `/health` and scheduler timestamps. Worker status
requires a heartbeat within ten minutes. A successful account fetch never
implies an open market or successful execution. Old or missing control data is
explicitly unverified. Personal history errors are distinguished from an empty
history. Journal rows expose recorded source, reason and time without inventing
an event timestamp.

Removed the position-count-derived risk caps and progress bars, hardcoded win
rate, inferred market regime from portfolio daily P&L, and unconditional
ENFORCED/MODEL LIVE badges. Actual engine risk limits are not exposed by the
current API, so the view says not reported. Risk, stop-loss, Kelly and auto-trade
preferences are labeled as saved preferences that do not modify engine policy.
Global controls retain their existing disabled behavior and move behind an
explicit read-only disclosure. This is not an authorization change to the API.

Broker account and fill responses are discarded if the user, broker connection
or session token changes while a request is pending. Invalid account/position
shapes do not become fresh broker data. Root and dashboard assets are mirrored;
the new status module is included in the Pages deployment.

Validation: 296 Python tests passed. Sixteen Node tests cover routes, swipe
guards, setup/closed/open/paused/stopped/stale/failed/succeeded status states and
six delayed private-response scenarios. Browser QA used network-isolated local
fixtures for guest, closed-market, paused and completed-without-fill states on
mobile, and the deployed guest view on desktop. Fixtures are not trading evidence.

Operational evidence at 13:31 UTC: the database audit recorded three scheduled
requests and one succeeded Paper request, with no missing account marks, durable
risk checkpoints, independent research or scope violations. No broker-confirmed
orders were recorded. This verifies the first forward execution cycle, not
investment performance or a completed forward model-release window.
