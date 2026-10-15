create table if not exists public.autonomous_runtime (
  id text primary key check (id = 'paper'),
  last_tick_at timestamptz not null,
  market_open boolean not null,
  last_queued_count integer not null default 0
);
alter table public.autonomous_runtime enable row level security;
revoke all on public.autonomous_runtime from anon, authenticated;

-- Enforce serialization even when a manual click races the scheduler or a
-- second worker. A time-window check alone cannot prevent that race.
create unique index if not exists execution_requests_one_active_strategy
  on public.execution_requests(user_id, strategy_id)
  where status in ('queued', 'leased', 'running', 'retrying', 'reconcile_pending');

-- Only the trusted worker API supplies the broker's authoritative market clock.
-- One transaction serializes scheduling and inserts each user/slot at most once.
create or replace function public.schedule_paper_executions(p_market_open boolean)
returns integer
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  inserted_count integer := 0;
  slot bigint := floor(extract(epoch from now()) / 900);
begin
  if not pg_try_advisory_xact_lock(681752104) then
    return 0;
  end if;
  insert into public.autonomous_runtime(id, last_tick_at, market_open, last_queued_count)
  values ('paper', now(), p_market_open, 0)
  on conflict (id) do update set last_tick_at = excluded.last_tick_at,
    market_open = excluded.market_open, last_queued_count = 0;
  if not p_market_open then return 0; end if;
  if not exists (
    select 1 from public.control_state where id = 'global'
      and state->>'trading_enabled' = 'true'
      and coalesce(state->>'emergency_stop', 'false') = 'false'
      and coalesce(state->>'mode', 'paper') = 'paper'
  ) then return 0; end if;

  insert into public.execution_requests (
    user_id, broker_connection_id, strategy_id, trigger_type,
    requested_mode, idempotency_key, priority, actor, payload
  )
  select b.user_id, b.id, 'default', 'schedule', 'paper',
    'schedule:' || b.id::text || ':' || slot::text, 100,
    'autonomous-paper-scheduler', jsonb_build_object('schedule_slot', slot)
  from public.broker_connections b
  where b.enabled and b.last_verified_status = 'verified'
    and b.trading_mode = 'paper'
    and b.base_url = 'https://paper-api.alpaca.markets'
    and b.api_key_encrypted is not null and b.secret_key_encrypted is not null
    and coalesce(b.metadata->>'auto_execution_enabled', 'true') = 'true'
    and not exists (
      select 1 from public.execution_requests r where r.user_id = b.user_id
        and r.strategy_id = 'default'
        and r.status in ('queued', 'leased', 'running', 'retrying', 'reconcile_pending')
    )
    and not exists (
      select 1 from public.daily_risk_state d where d.user_id = b.user_id
        and d.trading_day = (now() at time zone 'America/New_York')::date
        and (d.reset_required or d.breached)
    )
  on conflict do nothing;
  get diagnostics inserted_count = row_count;
  update public.autonomous_runtime set last_queued_count = inserted_count where id = 'paper';
  return inserted_count;
end;
$$;

revoke all on function public.schedule_paper_executions(boolean) from public, anon, authenticated;
grant execute on function public.schedule_paper_executions(boolean) to service_role;

create or replace function public.autonomous_runtime_status()
returns jsonb language sql security definer set search_path = public, pg_temp as $$
  select jsonb_build_object(
    'worker_last_seen_at', (select max(last_seen_at) from public.worker_heartbeats),
    'worker_online', exists (select 1 from public.worker_heartbeats where last_seen_at > now() - interval '10 minutes'),
    'scheduler_last_tick_at', (select last_tick_at from public.autonomous_runtime where id = 'paper'),
    'market_open', (select market_open from public.autonomous_runtime where id = 'paper'),
    'overdue_requests', (select count(*) from public.execution_requests where status in ('queued', 'retrying') and run_after < now() - interval '15 minutes'),
    'recent_dead_letters', (select count(*) from public.execution_requests where status = 'dead_letter' and updated_at > now() - interval '1 day')
  );
$$;
revoke all on function public.autonomous_runtime_status() from public, anon, authenticated;
grant execute on function public.autonomous_runtime_status() to service_role;
