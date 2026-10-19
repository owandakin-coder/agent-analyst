# Daily market discovery

Adds an account-scoped, read-only Alpaca scanner and a Hebrew research list in
the execution journal. The model's existing 15-symbol configuration is unchanged.

Sources: Alpaca Assets (`/v2/assets`) and batched snapshots
(`https://docs.alpaca.markets/us/reference/stocksnapshots-1`). SIP is attempted;
403 falls back to IEX, with explicit venue-only volume disclosure. Other errors
fail the scan, rather than silently substituting data or showing partial success.

Coverage: all active, tradable, syntactically valid US equities on listed
supported exchanges, including ETFs; OTC excluded. Missing/stale/non-finite
bars, prices below $5, feed dollar volume below $1m are excluded. Top 30 ranked
by liquidity and capped absolute daily movement. Source bar timestamps are shown.
This first discovery stage does not verify long history, sector diversification
or corporate-action-adjusted momentum and is not consumed by the trading model.

Safety: server-side broker keys only, existing shared-worker authentication,
RLS/private table, per-user/connection/account/credential scope, atomic daily
claim and attempt fence, no member-triggered scan/order endpoint. Browser drops
late responses after account/session changes. Data retrieval errors remain visible.

Validation: 13 Deno tests and API type check; 20 Node UI/session tests; SQL
migration plus scanner fixture passed in a transaction that rolled back. Browser
fixture inspected populated Hebrew/RTL rendering; fixtures are not market evidence.
