create extension if not exists "pgcrypto";

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  display_name text,
  phone text,
  timezone text not null default 'Asia/Jerusalem',
  account_tier text not null default 'paper',
  account_role text not null default 'member',
  trading_mode text not null default 'paper',
  active_executor text not null default 'github_actions',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint profiles_account_tier_check check (account_tier in ('paper', 'live', 'enterprise')),
  constraint profiles_account_role_check check (account_role in ('member', 'operator', 'admin')),
  constraint profiles_trading_mode_check check (trading_mode in ('paper', 'live'))
);

create table if not exists public.user_preferences (
  user_id uuid primary key references auth.users(id) on delete cascade,
  daily_email text,
  risk_level text not null default 'Normal',
  risk_note text not null default 'Max drawdown threshold: 15%',
  auto_trade boolean not null default true,
  stop_loss boolean not null default true,
  kelly boolean not null default true,
  push_alerts boolean not null default false,
  watchlist text[] not null default '{}',
  ui jsonb not null default '{}'::jsonb,
  notifications jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint user_preferences_risk_level_check check (risk_level in ('Conservative', 'Normal', 'Aggressive'))
);

create table if not exists public.audit_events (
  id bigint generated always as identity primary key,
  user_id uuid references auth.users(id) on delete cascade,
  event_type text not null,
  event_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists audit_events_user_id_created_at_idx
  on public.audit_events (user_id, created_at desc);

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

drop trigger if exists user_preferences_set_updated_at on public.user_preferences;
create trigger user_preferences_set_updated_at
before update on public.user_preferences
for each row execute function public.set_updated_at();

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (
    id,
    email,
    display_name
  )
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1))
  )
  on conflict (id) do update
  set email = excluded.email,
      display_name = coalesce(public.profiles.display_name, excluded.display_name);

  insert into public.user_preferences (user_id)
  values (new.id)
  on conflict (user_id) do nothing;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

alter table public.profiles enable row level security;
alter table public.user_preferences enable row level security;
alter table public.audit_events enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own"
on public.profiles
for select
to authenticated
using (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own"
on public.profiles
for update
to authenticated
using (auth.uid() = id)
with check (auth.uid() = id);

drop policy if exists "profiles_insert_own" on public.profiles;
create policy "profiles_insert_own"
on public.profiles
for insert
to authenticated
with check (auth.uid() = id);

drop policy if exists "prefs_select_own" on public.user_preferences;
create policy "prefs_select_own"
on public.user_preferences
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "prefs_update_own" on public.user_preferences;
create policy "prefs_update_own"
on public.user_preferences
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "prefs_insert_own" on public.user_preferences;
create policy "prefs_insert_own"
on public.user_preferences
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "audit_select_own" on public.audit_events;
create policy "audit_select_own"
on public.audit_events
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "audit_insert_own" on public.audit_events;
create policy "audit_insert_own"
on public.audit_events
for insert
to authenticated
with check (auth.uid() = user_id);
