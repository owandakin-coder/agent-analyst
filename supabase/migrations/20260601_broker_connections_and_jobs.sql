create table if not exists public.broker_connections (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null unique references auth.users(id) on delete cascade,
  broker_name text not null default 'alpaca',
  account_label text,
  trading_mode text not null default 'paper',
  base_url text not null default 'https://paper-api.alpaca.markets',
  api_key_encrypted text,
  secret_key_encrypted text,
  enabled boolean not null default false,
  last_verified_at timestamptz,
  last_verified_status text not null default 'pending',
  last_error text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint broker_connections_broker_name_check check (broker_name in ('alpaca')),
  constraint broker_connections_trading_mode_check check (trading_mode in ('paper', 'live')),
  constraint broker_connections_verify_status_check check (last_verified_status in ('pending', 'verified', 'failed'))
);

create table if not exists public.execution_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  broker_connection_id uuid references public.broker_connections(id) on delete set null,
  job_type text not null default 'trade_once',
  status text not null default 'queued',
  actor text,
  requested_mode text not null default 'paper',
  payload jsonb not null default '{}'::jsonb,
  result jsonb not null default '{}'::jsonb,
  error_text text,
  workflow_run_id text,
  requested_at timestamptz not null default timezone('utc', now()),
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint execution_jobs_type_check check (job_type in ('trade_once')),
  constraint execution_jobs_status_check check (status in ('queued', 'claimed', 'running', 'succeeded', 'failed', 'skipped', 'cancelled')),
  constraint execution_jobs_mode_check check (requested_mode in ('paper', 'live'))
);

create index if not exists execution_jobs_user_id_requested_at_idx
  on public.execution_jobs (user_id, requested_at desc);

create index if not exists execution_jobs_status_requested_at_idx
  on public.execution_jobs (status, requested_at asc);

drop trigger if exists broker_connections_set_updated_at on public.broker_connections;
create trigger broker_connections_set_updated_at
before update on public.broker_connections
for each row execute function public.set_updated_at();

drop trigger if exists execution_jobs_set_updated_at on public.execution_jobs;
create trigger execution_jobs_set_updated_at
before update on public.execution_jobs
for each row execute function public.set_updated_at();

alter table public.broker_connections enable row level security;
alter table public.execution_jobs enable row level security;

drop policy if exists "broker_connections_select_own" on public.broker_connections;
create policy "broker_connections_select_own"
on public.broker_connections
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "broker_connections_insert_own" on public.broker_connections;
create policy "broker_connections_insert_own"
on public.broker_connections
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "broker_connections_update_own" on public.broker_connections;
create policy "broker_connections_update_own"
on public.broker_connections
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "execution_jobs_select_own" on public.execution_jobs;
create policy "execution_jobs_select_own"
on public.execution_jobs
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "execution_jobs_insert_own" on public.execution_jobs;
create policy "execution_jobs_insert_own"
on public.execution_jobs
for insert
to authenticated
with check (auth.uid() = user_id);
