# ATZMA execution and research audit — 2026-10-16

## Outcome

The highest-value changes were execution correctness and evidence integrity.
The deployed system already had a scheduler, a worker watchdog, independent
shadow strategies, durable scoped risk state, and model release gates. Adding
more agents or another data subscription would not repair the defects below.

The capability review inspected the declared purpose/description of all 75
locally installed SKILL.md files, representing 61 distinct skill names. The
[inventory](SKILLS_REVIEW_2026-10-16.md) lists every source, including cached
versions. No trading-specific skill provides a validated predictive edge.
Plugin-management guidance was applied to assess integration needs; existing
GitHub, Supabase and Python capabilities were sufficient.

## Findings and implemented changes

| Finding | Consequence before the change | Implemented behavior |
|---|---|---|
| Order acceptance treated as execution | A pending or duplicate sell could clear stops, reduce holdings or create a fictional Kelly outcome; a pending buy could invent an entry. | Use only explicit finite broker-filled quantity and execution price. Preserve protection for remaining holdings. Partial/unknown receipts stop the batch. |
| Outstanding broker instructions ignored by sizing | Another cycle could size new exposure before earlier orders settled. | Query broker open orders before execution; stop new submissions while any remain; query failure blocks execution. |
| Download time accepted as quote freshness | Fetching an old exchange quote again made it appear fresh. | Exchange timestamp determines age; quote price and timestamp come from the same receipt; cached receipts are cleared on refresh. |
| Controls checked only before inference | A pause, quote expiry or closing bell during a slow cycle could be missed. | Check control state, quote age and broker market clock immediately before each order. |
| Whole-share minimum could exceed budget | With a $4,000 order limit and a $9,000 share, `max(1, ...)` could buy over the limit. | Skip an order if its budget cannot buy one whole share. Reject malformed, nonfinite or out-of-range action vectors before orders. |
| Calendar dates counted as sessions | Weekend/holiday observations or marks after an early close could manufacture evidence. | Use the XNYS session calendar, regular hours and completed days; require paired timestamps, explicit account/request scope and consistent duplicates. |
| Reused model hashes could reuse old evidence | Historical rows from an earlier incarnation of identical weights could satisfy a new release. | Candidate evidence starts at staging; rollback evidence starts at release. Rollback can operate without an unrelated benchmark mark. |
| Fees increased shadow exposure above its ceiling | Allocating 50% of pre-cost equity exceeded 50% after fees. | Solve turnover costs against post-cost equity. Schema 2 prevents comparisons with observations calculated under the old convention. |
| Poller health did not demonstrate complete forward evidence | A healthy idle worker could be mistaken for an executed research cycle. | Add a read-only database audit with aggregate counts and a nonzero exit on missing marks, checkpoints, independent research, or cross-account evidence. |

## Evidence and release validation

Regression coverage includes accepted buys, accepted stops, partial stops at a
different execution price, invalid fill quantities/prices, unsettled orders,
expensive shares, malformed actions, mid-cycle pauses and quote expiry.
Research tests cover holidays, weekends, early closes, premarket observations,
malformed timestamps, inconsistent pairs, missing scopes, old evidence and
cost-adjusted exposure ceilings.

The SQL audit integration uses isolated fixtures inside a rolled-back
transaction. It proves that missing evidence is detected, a complete scoped
cycle is recognized, and a wrong-account mark invalidates that recognition.
No broker order is submitted by these tests.

Local validation: 293 Python tests passed, five Deno API tests passed, Edge API
type checking passed, and workflow YAML parsed. The durable-state and new
research-audit SQL integration checks passed against the cloud database with
all fixture changes rolled back.

Actual cloud inspection during this audit found no overdue requests, dead
letters or account-scope violations. There were zero new scheduled successful
cycles in the audit window, so `forward_execution_verified` remained false.
The completed retrain was rejected by its existing historical gate and
production artifacts were preserved; this patch does not replace trained weights.

## Practical limits and next research priorities

There is still no demonstrated forward trading edge. Broker Paper fills are
simulation evidence, distinct from shadow returns and live execution. Raw
account equity is not a cashflow-adjusted, strategy-only return. A code test
cannot replace elapsed market sessions.

The next decisive evidence is the first regular-session scheduled cycle with
scoped durable state, actual broker marks and independent shadow marks, followed
by broker-confirmed fills only when a strategy elects to trade. The existing
heartbeat follows the runbook and now has an explicit research-audit command.

Later research should compare sufficiently long paired net-return histories,
test sensitivity to cost assumptions and multiple candidate selection, and
measure actual fill quality. An always-on hosting migration would require
separate evidence of Actions latency affecting execution. None of these
uncertainties justifies loosening promotion or risk gates.

## Primary sources consulted

- [Alpaca: Working with orders](https://docs.alpaca.markets/us/docs/working-with-orders): order submission, client order IDs and execution lifecycle.
- [Alpaca: Paper Trading](https://docs.alpaca.markets/us/v1.4.2/docs/paper-trading): simulated execution and partial fills; Paper is not live performance evidence.
- [NYSE: Holidays and trading hours](https://www.nyse.com/trade/hours-calendars): regular sessions, holidays and early closes.
- [exchange_calendars project](https://github.com/gerrymanoim/exchange_calendars): XNYS schedule queries used in the release gate.
- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): chronological validation and exclusion gaps, considered for subsequent research design.
