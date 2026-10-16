-- One account-day row is updated atomically. A routine observation must never
-- clear a breach/reset latch or replace the original day's baseline.
create or replace function public.record_daily_risk_event(
  p_user_id uuid, p_trading_day date, p_breached boolean, p_payload jsonb
) returns void
language sql security definer set search_path = public
as $$
  insert into public.daily_risk_state as existing
    (user_id,trading_day,baseline_equity,current_equity,realized_pnl,unrealized_pnl,
     realized_loss_limit,unrealized_loss_limit,breached,breached_at,reset_required,metadata)
  values (p_user_id,p_trading_day,
    coalesce((p_payload->>'baseline_equity')::numeric,0),
    coalesce((p_payload->>'current_equity')::numeric,0),
    coalesce((p_payload->>'realized_pnl')::numeric,0),
    coalesce((p_payload->>'unrealized_pnl')::numeric,0),
    (p_payload->>'realized_loss_limit')::numeric,
    (p_payload->>'unrealized_loss_limit')::numeric,
    p_breached,case when p_breached then now() end,p_breached,p_payload)
  on conflict (user_id,trading_day) do update set
    current_equity=excluded.current_equity,
    realized_pnl=excluded.realized_pnl, unrealized_pnl=excluded.unrealized_pnl,
    realized_loss_limit=excluded.realized_loss_limit,
    unrealized_loss_limit=excluded.unrealized_loss_limit,
    breached=existing.breached or excluded.breached,
    breached_at=coalesce(existing.breached_at,excluded.breached_at),
    reset_required=existing.reset_required or excluded.reset_required,
    metadata=excluded.metadata;
$$;
revoke all on function public.record_daily_risk_event(uuid,date,boolean,jsonb) from public,anon,authenticated;
grant execute on function public.record_daily_risk_event(uuid,date,boolean,jsonb) to service_role;
