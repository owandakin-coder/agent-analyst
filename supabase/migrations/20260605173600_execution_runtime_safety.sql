create table if not exists public.execution_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  execution_request_id uuid references public.execution_requests(id) on delete cascade,
  stage text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists execution_events_user_created_at_idx
  on public.execution_events (user_id, created_at desc);

create table if not exists public.control_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  execution_request_id uuid references public.execution_requests(id) on delete set null,
  command_version integer,
  event_type text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists control_events_created_at_idx
  on public.control_events (created_at desc);

create table if not exists public.daily_risk_state (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  trading_day date not null,
  baseline_equity numeric not null default 0,
  current_equity numeric not null default 0,
  realized_pnl numeric not null default 0,
  unrealized_pnl numeric not null default 0,
  realized_loss_limit numeric,
  unrealized_loss_limit numeric,
  breached boolean not null default false,
  breached_at timestamptz,
  reset_required boolean not null default false,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint daily_risk_state_user_day_uniq unique (user_id, trading_day)
);

create index if not exists daily_risk_state_user_day_idx
  on public.daily_risk_state (user_id, trading_day desc);

create table if not exists public.position_snapshots (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  execution_request_id uuid references public.execution_requests(id) on delete cascade,
  snapshot_type text not null,
  snapshot_hash text,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists position_snapshots_user_created_at_idx
  on public.position_snapshots (user_id, created_at desc);

alter table public.broker_orders
  add column if not exists idempotency_key text,
  add column if not exists submitted_at timestamptz,
  add column if not exists reconciled_at timestamptz;

create unique index if not exists broker_orders_client_order_id_uniq
  on public.broker_orders (client_order_id)
  where client_order_id is not null;

create unique index if not exists broker_orders_idempotency_key_uniq
  on public.broker_orders (user_id, idempotency_key)
  where idempotency_key is not null;

alter table public.execution_events enable row level security;
alter table public.control_events enable row level security;
alter table public.daily_risk_state enable row level security;
alter table public.position_snapshots enable row level security;

drop policy if exists "execution_events_select_own" on public.execution_events;
create policy "execution_events_select_own"
on public.execution_events
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "daily_risk_state_select_own" on public.daily_risk_state;
create policy "daily_risk_state_select_own"
on public.daily_risk_state
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "position_snapshots_select_own" on public.position_snapshots;
create policy "position_snapshots_select_own"
on public.position_snapshots
for select
to authenticated
using (auth.uid() = user_id);

drop trigger if exists daily_risk_state_set_updated_at on public.daily_risk_state;
create trigger daily_risk_state_set_updated_at
before update on public.daily_risk_state
for each row execute function public.set_updated_at();

create or replace function public.claim_execution_request(
  p_worker_id text,
  p_lease_seconds integer default 90
)
returns setof public.execution_requests
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  with candidate as (
    select id
    from public.execution_requests
    where status in ('queued', 'retrying')
      and run_after <= timezone('utc', now())
    order by priority asc, created_at asc
    for update skip locked
    limit 1
  )
  update public.execution_requests r
  set status = 'leased',
      worker_id = p_worker_id,
      lease_expires_at = timezone('utc', now()) + make_interval(secs => greatest(p_lease_seconds, 15)),
      attempt_count = r.attempt_count + 1,
      updated_at = timezone('utc', now())
  from candidate
  where r.id = candidate.id
  returning r.*;
end;
$$;

create or replace function public.renew_execution_request_lease(
  p_request_id uuid,
  p_worker_id text,
  p_lease_seconds integer default 90
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  updated_count integer;
begin
  update public.execution_requests
  set lease_expires_at = timezone('utc', now()) + make_interval(secs => greatest(p_lease_seconds, 15)),
      updated_at = timezone('utc', now())
  where id = p_request_id
    and worker_id = p_worker_id
    and status in ('leased', 'running', 'reconcile_pending');

  get diagnostics updated_count = row_count;
  return updated_count > 0;
end;
$$;

create or replace function public.transition_execution_request(
  p_request_id uuid,
  p_worker_id text,
  p_from_status text,
  p_to_status text,
  p_error_text text default null
)
returns setof public.execution_requests
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  update public.execution_requests
  set status = p_to_status,
      started_at = case when p_to_status = 'running' then coalesce(started_at, timezone('utc', now())) else started_at end,
      completed_at = case when p_to_status in ('succeeded', 'failed', 'skipped', 'cancelled', 'dead_letter') then timezone('utc', now()) else completed_at end,
      error_text = coalesce(p_error_text, error_text),
      updated_at = timezone('utc', now())
  where id = p_request_id
    and worker_id = p_worker_id
    and status = p_from_status
  returning *;
end;
$$;

create or replace function public.requeue_expired_execution_requests()
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  updated_count integer;
begin
  update public.execution_requests
  set status = case when attempt_count >= max_attempts then 'dead_letter' else 'retrying' end,
      worker_id = null,
      lease_expires_at = null,
      updated_at = timezone('utc', now())
  where status in ('leased', 'running', 'reconcile_pending')
    and lease_expires_at is not null
    and lease_expires_at < timezone('utc', now());

  get diagnostics updated_count = row_count;
  return updated_count;
end;
$$;
