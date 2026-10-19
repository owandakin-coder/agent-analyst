create table if not exists public.market_scans (
  broker_connection_id uuid primary key references public.broker_connections(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  broker_account_id text not null,
  credential_version text not null,
  scan_day date not null,
  attempt_id uuid not null,
  attempted_at timestamptz not null default now(),
  status text not null check (status in ('running','completed','failed')),
  completed_at timestamptz,
  report jsonb,
  error_code text
);
alter table public.market_scans enable row level security;
revoke all on public.market_scans from anon, authenticated;
grant all on public.market_scans to service_role;

create or replace function public.claim_market_scan(p_connection_id uuid, p_attempt_id uuid)
returns boolean language plpgsql security definer set search_path = public as $$
declare n integer;
begin
  insert into market_scans (broker_connection_id,user_id,broker_account_id,credential_version,scan_day,attempt_id,status)
  select id,user_id,coalesce(metadata->>'account_number',''),updated_at::text,
    (now() at time zone 'America/New_York')::date,p_attempt_id,'running'
  from broker_connections where id=p_connection_id and enabled and last_verified_status='verified'
    and trading_mode='paper' and base_url='https://paper-api.alpaca.markets'
  on conflict (broker_connection_id) do update set
    user_id=excluded.user_id,broker_account_id=excluded.broker_account_id,
    credential_version=excluded.credential_version,scan_day=excluded.scan_day,
    attempt_id=excluded.attempt_id,attempted_at=now(),status='running',error_code=null,report=null,completed_at=null
  where (market_scans.status <> 'running' or market_scans.attempted_at < now()-interval '10 minutes')
    and (market_scans.scan_day <> excluded.scan_day or market_scans.credential_version <> excluded.credential_version
      or (market_scans.status <> 'completed' and market_scans.attempted_at < now()-interval '30 minutes'));
  get diagnostics n = row_count;
  return n=1;
end $$;
revoke all on function public.claim_market_scan(uuid,uuid) from public,anon,authenticated;
grant execute on function public.claim_market_scan(uuid,uuid) to service_role;
