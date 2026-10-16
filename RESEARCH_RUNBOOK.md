# ATZMA evidence and release contract

The deployment runs Paper execution. Shadow portfolios and broker results have
different provenance and are never treated as interchangeable returns.

## Evidence captured each open-market cycle

- The broker's observed equity/cash and SPY quote, scoped to the broker connection.
- Content hash of the active weights and normalization files; hash of actual
  observation values rather than just their shape.
- Model actions, risk-scaled actions, regime, veto reasons and decision summary.
- Independent momentum, mean-reversion and defensive allocations, plus champion,
  cash and 50%-SPY/50%-cash shadow portfolios.
- When staged: candidate predictions through the same veto/risk controls; after
  release: previous-model shadow predictions for rollback comparisons.
- Broker-confirmed filled quantities/prices, including partial fills. Accepted
  orders do not count as fills. Fill-versus-decision-price basis points are
  weighted by reference notional with buy/sell signs applied correctly.

Shadow portfolios mark existing holdings before choosing the next allocation.
They use a 50% exposure ceiling and 10 bps per turnover (5 bps fee allowance plus
5 bps slippage assumption). This is an estimate, not a broker fee measurement.
Missing holding prices abort the calculation. Missing SPY prices produce no SPY
observation. Actual fees remain unverified, and raw account equity is explicitly
not a cashflow-adjusted or strategy-only return: transfers/manual activity can
change it. No advantage claim can be made from the initial empty ledger.

## Durable risk state

State is keyed by user, broker connection and strategy. It preserves account and
risk peaks, entry prices, trailing highs, Kelly outcomes and shadow holdings.
Every write requires an active owned Paper lease and matching state version.
State and research observations are committed in one database transaction before
broker submission. Duplicate checkpoint writes do not duplicate observations.
The worker persists state on aborted/partial cycles as well. A missing state
backend, unavailable broker positions or unavailable duplicate check blocks new
execution. Account snapshots are not silently replaced by stale cached balances.

## Candidate lifecycle

1. Monthly training uses rolling, non-overlapping training/validation/test
   periods ending before the current New York date. Runtime config is unchanged.
2. The existing backtest gate still screens return/Sharpe/drawdown and rejects
   missing or non-finite metrics. It stages a candidate only; the production
   snapshot is restored even if the gate crashes.
3. A pending candidate's forward window is preserved for at least 45 calendar
   days across retrains, unless its comparison base has been superseded.
4. Daily governance uses paired observations from the same user and execution
   request, with matching champion hash, schema and cost assumptions. It keeps
   the last paired mark per completed New York day. Today is excluded.
   Schema 2 observations must fall inside an XNYS regular trading session;
   weekends, holidays and observations after an early close do not count.
   Paired strategies require the same timestamp and account/request scope.
   Evidence predating this candidate's staging or this release is excluded.
5. At least 21 daily marks (20 completed return intervals) are required per
   eligible account. Repeating intraday samples or adding accounts cannot
   substitute for elapsed sessions. Candidate and champion are normalized to
   the first paired mark, not compared from different inception dates.
6. Median candidate return must beat the champion by 0.5 percentage points,
   meet the SPY shadow return, have drawdown no more than 1 percentage point worse,
   and Sharpe at least the champion's and at least zero. These are empirical
   release rules, not a statistical guarantee of future profitability.
7. On release, exact prior artifacts and a content-addressed release report
   are retained in `models/rollback`. Only committed releases reach workers.
8. After at least 5 completed paired return intervals, the previous model can
   trigger rollback if it beats the active model by 2 percentage points while
   clearing the same drawdown/Sharpe constraints. Hash mismatches stop releases.

The monthly retrain and daily release workflows share a concurrency group.
Git history retains older releases. A single new candidate cannot replace a
champion solely because a historical backtest looked good.

## Verification

```powershell
python -m pytest -q
npx --yes deno test supabase/functions/api/*_test.ts
npx --yes deno check supabase/functions/api/index.ts
python scripts/autonomous_ops.py verify-scheduler
python scripts/autonomous_ops.py verify-research
python scripts/autonomous_ops.py verify-daily-risk
python scripts/autonomous_ops.py verify-audit
python scripts/autonomous_ops.py audit-research
```

The SQL integration checks create isolated fixtures inside a rolled-back
transaction, never decrypt broker credentials, and never submit orders. The
browser fixture verifies populated/empty/signed-out views without using a real
account. Research API reads require authentication; private tables/RPCs are not
accessible to anonymous or authenticated database roles directly.

Deploy database changes before the Edge API, then publish the worker/UI commit.
The GitHub watchdog and the additional chat monitor check execution freshness.
Forward validation requires real elapsed market sessions and cannot be replaced
by unit tests, a synthetic fixture, or an empty successful poll.

Daily risk observations use an atomic user/day upsert. The original baseline,
breach timestamp and reset-required latch survive subsequent normal observations;
only the separate authorized reset flow can clear an existing stop. Unreadable
broker secrets terminate an owned lease with a safe reverification error rather
than repeatedly expiring it. Never replay failed requests or clear dead letters
to make health green. `recent_dead_letters` includes historical failures for 24h.
Duplicate-order attempts append evidence without changing the original broker
fill. Existing ledger references must match user, connection and broker account.
Unreadable connections require a complete replacement Paper key pair and fresh
verification; the monitor must not copy credentials between accounts.

Each monitoring pass should also run `audit-research`. It inspects the last
24 hours (from the research deployment on 2026-10-16) and exits nonzero for a
successful Paper cycle missing account marks, scoped durable state or the five
core shadow strategies, or for cross-account evidence. Zero scheduled cycles
means forward execution remains unverified. `verify-audit` proves detection of
missing, complete and incorrectly scoped evidence in a rolled-back transaction.

Broker acceptance is never a fill. Execution state uses explicit finite
`filled_qty` and `filled_avg_price`; partial or ambiguous receipts stop the
batch until reconciliation. Existing open broker orders block new submissions.
Exchange quote timestamps, control state and market openness are checked again
before each submission. Download timestamps cannot make old quotes fresh.
Whole-share orders must fit the calculated budget. Shadow allocations target
post-cost equity so the 50% exposure ceiling holds after costs (schema 2).
