alter table public.broker_connections
  add column if not exists key_version integer not null default 1,
  add column if not exists kms_key_id text,
  add column if not exists encrypted_data_key text,
  add column if not exists rotated_at timestamptz;

create table if not exists public.control_state (
  id text primary key default 'global',
  state jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default timezone('utc', now()),
  updated_by text not null default 'system',
  command_version integer not null default 1
);

insert into public.control_state (id, state, updated_by, command_version)
values (
  'global',
  jsonb_build_object(
    'mode', 'paper',
    'trading_enabled', true,
    'emergency_stop', false,
    'status', 'running',
    'executor', 'worker_pool',
    'executor_label', 'ATZMA Worker Pool',
    'last_command', 'bootstrap',
    'last_command_at', timezone('utc', now()),
    'updated_at', timezone('utc', now()),
    'updated_by', 'system',
    'note', 'Paper engine is allowed to execute.'
  ),
  'system',
  1
)
on conflict (id) do nothing;

create table if not exists public.execution_requests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  broker_connection_id uuid references public.broker_connections(id) on delete set null,
  strategy_id text not null default 'default',
  trigger_type text not null default 'manual',
  requested_mode text not null default 'paper',
  idempotency_key text not null,
  priority integer not null default 100,
  status text not null default 'queued',
  actor text,
  payload jsonb not null default '{}'::jsonb,
  result jsonb not null default '{}'::jsonb,
  error_text text,
  run_after timestamptz not null default timezone('utc', now()),
  attempt_count integer not null default 0,
  max_attempts integer not null default 3,
  lease_expires_at timestamptz,
  worker_id text,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint execution_requests_trigger_type_check check (trigger_type in ('manual', 'schedule', 'recovery', 'control_plane')),
  constraint execution_requests_status_check check (status in ('queued', 'leased', 'running', 'retrying', 'reconcile_pending', 'succeeded', 'failed', 'dead_letter', 'cancelled', 'skipped')),
  constraint execution_requests_mode_check check (requested_mode in ('paper', 'live'))
);

create unique index if not exists execution_requests_user_idempotency_key_uniq
  on public.execution_requests (user_id, idempotency_key);

create index if not exists execution_requests_status_run_after_idx
  on public.execution_requests (status, run_after asc, priority asc, created_at asc);

create index if not exists execution_requests_user_created_at_idx
  on public.execution_requests (user_id, created_at desc);

create table if not exists public.worker_heartbeats (
  worker_id text primary key,
  version text,
  capacity integer not null default 1,
  active_jobs integer not null default 0,
  metadata jsonb not null default '{}'::jsonb,
  last_seen_at timestamptz not null default timezone('utc', now()),
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.decision_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  execution_request_id uuid references public.execution_requests(id) on delete cascade,
  model_version text,
  feature_snapshot_hash text,
  market_data_source text,
  regime text,
  strategy_mode text,
  summary text,
  raw_action jsonb not null default '[]'::jsonb,
  scaled_action jsonb not null default '[]'::jsonb,
  decisions jsonb not null default '[]'::jsonb,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists decision_events_user_created_at_idx
  on public.decision_events (user_id, created_at desc);

create table if not exists public.broker_orders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  execution_request_id uuid references public.execution_requests(id) on delete set null,
  broker_connection_id uuid references public.broker_connections(id) on delete set null,
  broker_name text not null default 'alpaca',
  broker_order_id text,
  client_order_id text,
  symbol text,
  side text,
  quantity numeric,
  requested_price numeric,
  status text not null default 'submitted',
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists broker_orders_user_created_at_idx
  on public.broker_orders (user_id, created_at desc);

create index if not exists broker_orders_client_order_id_idx
  on public.broker_orders (client_order_id);

create table if not exists public.broker_order_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  broker_order_id uuid references public.broker_orders(id) on delete cascade,
  execution_request_id uuid references public.execution_requests(id) on delete set null,
  event_type text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists broker_order_events_user_created_at_idx
  on public.broker_order_events (user_id, created_at desc);

create table if not exists public.risk_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  execution_request_id uuid references public.execution_requests(id) on delete cascade,
  event_type text not null,
  risk_level text,
  drawdown numeric,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists risk_events_user_created_at_idx
  on public.risk_events (user_id, created_at desc);

drop trigger if exists control_state_set_updated_at on public.control_state;
create trigger control_state_set_updated_at
before update on public.control_state
for each row execute function public.set_updated_at();

drop trigger if exists execution_requests_set_updated_at on public.execution_requests;
create trigger execution_requests_set_updated_at
before update on public.execution_requests
for each row execute function public.set_updated_at();

drop trigger if exists worker_heartbeats_set_updated_at on public.worker_heartbeats;
create trigger worker_heartbeats_set_updated_at
before update on public.worker_heartbeats
for each row execute function public.set_updated_at();

drop trigger if exists broker_orders_set_updated_at on public.broker_orders;
create trigger broker_orders_set_updated_at
before update on public.broker_orders
for each row execute function public.set_updated_at();

alter table public.control_state enable row level security;
alter table public.execution_requests enable row level security;
alter table public.worker_heartbeats enable row level security;
alter table public.decision_events enable row level security;
alter table public.broker_orders enable row level security;
alter table public.broker_order_events enable row level security;
alter table public.risk_events enable row level security;

drop policy if exists "control_state_select_authenticated" on public.control_state;
create policy "control_state_select_authenticated"
on public.control_state
for select
to authenticated
using (true);

drop policy if exists "execution_requests_select_own" on public.execution_requests;
create policy "execution_requests_select_own"
on public.execution_requests
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "execution_requests_insert_own" on public.execution_requests;
create policy "execution_requests_insert_own"
on public.execution_requests
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "decision_events_select_own" on public.decision_events;
create policy "decision_events_select_own"
on public.decision_events
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "broker_orders_select_own" on public.broker_orders;
create policy "broker_orders_select_own"
on public.broker_orders
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "broker_order_events_select_own" on public.broker_order_events;
create policy "broker_order_events_select_own"
on public.broker_order_events
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "risk_events_select_own" on public.risk_events;
create policy "risk_events_select_own"
on public.risk_events
for select
to authenticated
using (auth.uid() = user_id);

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
  set status = 'running',
      worker_id = p_worker_id,
      started_at = coalesce(r.started_at, timezone('utc', now())),
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
    and status = 'running';

  get diagnostics updated_count = row_count;
  return updated_count > 0;
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
      run_after = timezone('utc', now()) + interval '15 seconds',
      error_text = coalesce(error_text, 'lease_expired'),
      updated_at = timezone('utc', now())
  where status = 'running'
    and lease_expires_at is not null
    and lease_expires_at < timezone('utc', now());

  get diagnostics updated_count = row_count;
  return updated_count;
end;
$$;
