alter table public.execution_requests add column if not exists broker_account_id text;
alter table public.broker_orders add column if not exists broker_account_id text not null default '';
alter table public.decision_events add column if not exists broker_account_id text not null default '';

create table if not exists public.strategy_runtime_state (
  user_id uuid not null references auth.users(id) on delete cascade,
  broker_connection_id uuid not null references public.broker_connections(id) on delete cascade,
  broker_account_id text not null default '',
  strategy_id text not null,
  version bigint not null default 0,
  state jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  primary key(user_id, broker_connection_id, strategy_id, broker_account_id)
);
alter table public.strategy_runtime_state add column if not exists broker_account_id text not null default '';
do $$ begin
  if exists(select 1 from pg_constraint where conrelid='public.strategy_runtime_state'::regclass and contype='p' and array_length(conkey,1)=3) then
    alter table public.strategy_runtime_state drop constraint strategy_runtime_state_pkey;
    alter table public.strategy_runtime_state add primary key(user_id,broker_connection_id,strategy_id,broker_account_id);
  end if;
end $$;
create table if not exists public.research_observations (
  execution_request_id uuid not null references public.execution_requests(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  strategy text not null,
  observed_at timestamptz not null,
  equity numeric not null check (equity > 0),
  payload jsonb not null,
  broker_account_id text not null default '',
  primary key(execution_request_id, strategy)
);
alter table public.research_observations add column if not exists broker_account_id text not null default '';
create index if not exists research_user_time on public.research_observations(user_id, observed_at desc);
create table if not exists public.paper_account_marks (
  execution_request_id uuid primary key references public.execution_requests(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  observed_at timestamptz not null,
  equity numeric not null,
  payload jsonb not null
);
alter table public.paper_account_marks drop constraint if exists paper_account_marks_equity_check;
alter table public.paper_account_marks add column if not exists broker_account_id text not null default '';
alter table public.paper_account_marks enable row level security;
revoke all on public.paper_account_marks from anon, authenticated;
alter table public.strategy_runtime_state enable row level security;
alter table public.research_observations enable row level security;
revoke all on public.strategy_runtime_state from anon, authenticated;
revoke all on public.research_observations from anon, authenticated;

create or replace function public.save_strategy_runtime_state(p_request_id uuid, p_worker_id text, p_expected_version bigint, p_state jsonb)
returns bigint language plpgsql security definer set search_path = public, pg_temp as $$
declare r public.execution_requests; current_version bigint; next_version bigint; account_id text;
begin
  select * into r from public.execution_requests where id = p_request_id for update;
  if r.id is null or r.status <> 'running' or r.worker_id is distinct from p_worker_id
    or r.lease_expires_at is null or r.lease_expires_at <= now() or r.requested_mode <> 'paper' then
    raise exception 'Runtime state write requires an owned, active Paper lease';
  end if;
  if p_state->>'schema_version' is distinct from '1' or jsonb_typeof(p_state) <> 'object' then
    raise exception 'Unsupported runtime state schema';
  end if;
  select coalesce(metadata->>'account_number','') into account_id from public.broker_connections where id=r.broker_connection_id;
  if coalesce(r.broker_account_id,'') is distinct from account_id then raise exception 'Broker account changed during execution'; end if;
  insert into public.strategy_runtime_state(user_id, broker_connection_id, strategy_id,broker_account_id)
    values(r.user_id, r.broker_connection_id, r.strategy_id,account_id) on conflict do nothing;
  select version into current_version from public.strategy_runtime_state
    where user_id=r.user_id and broker_connection_id=r.broker_connection_id and strategy_id=r.strategy_id and broker_account_id=account_id for update;
  if current_version is distinct from p_expected_version then raise exception 'Runtime state version conflict'; end if;
  update public.strategy_runtime_state set state=p_state, version=version+1, updated_at=now()
    where user_id=r.user_id and broker_connection_id=r.broker_connection_id and strategy_id=r.strategy_id and broker_account_id=account_id
    returning version into next_version;
  insert into public.research_observations(execution_request_id,user_id,strategy,observed_at,equity,payload,broker_account_id)
    select r.id, r.user_id, sample->>'strategy', (sample->>'timestamp')::timestamptz,
      (sample->>'equity')::numeric, sample,account_id
    from jsonb_array_elements(coalesce(p_state->'research_samples','[]'::jsonb)) sample
    on conflict do nothing;
  if jsonb_typeof(p_state->'account_sample') = 'object' then
    insert into public.paper_account_marks(execution_request_id,user_id,observed_at,equity,payload,broker_account_id)
      values(r.id,r.user_id,(p_state->'account_sample'->>'observed_at')::timestamptz,
        (p_state->'account_sample'->>'equity')::numeric,p_state->'account_sample',account_id)
      on conflict do nothing;
  end if;
  return next_version;
end;
$$;
revoke all on function public.save_strategy_runtime_state(uuid,text,bigint,jsonb) from public, anon, authenticated;
grant execute on function public.save_strategy_runtime_state(uuid,text,bigint,jsonb) to service_role;

create or replace function public.research_summary(p_user_id uuid)
returns jsonb language sql security definer set search_path = public, pg_temp as $$
  select coalesce(jsonb_agg(jsonb_build_object('strategy', stats.strategy,
    'completed_sessions', stats.completed_sessions, 'samples', stats.samples,
    'first_observed_at', stats.first_observed_at, 'latest', latest.payload)), '[]'::jsonb)
  from (
    select strategy, count(*) as samples, min(observed_at) as first_observed_at,
      count(distinct (observed_at at time zone 'America/New_York')::date)
        filter(where (observed_at at time zone 'America/New_York')::date < (now() at time zone 'America/New_York')::date) as completed_sessions
    from public.research_observations where user_id=p_user_id
      and broker_account_id=(select coalesce(metadata->>'account_number','') from public.broker_connections where user_id=p_user_id limit 1)
      group by strategy
  ) stats
  cross join lateral (
    select payload from public.research_observations
      where user_id=p_user_id and strategy=stats.strategy
        and broker_account_id=(select coalesce(metadata->>'account_number','') from public.broker_connections where user_id=p_user_id limit 1)
      order by observed_at desc limit 1
  ) latest;
$$;
revoke all on function public.research_summary(uuid) from public, anon, authenticated;
grant execute on function public.research_summary(uuid) to service_role;
