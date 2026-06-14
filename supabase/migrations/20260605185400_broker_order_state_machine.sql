create or replace function public.transition_broker_order_state(
  p_client_order_id text,
  p_next_status text,
  p_payload jsonb default '{}'::jsonb,
  p_broker_order_id text default null,
  p_event_type text default null
)
returns setof public.broker_orders
language plpgsql
security definer
set search_path = public
as $$
declare
  current_status text;
  allowed boolean := false;
begin
  select status
  into current_status
  from public.broker_orders
  where client_order_id = p_client_order_id
  for update;

  if current_status is null then
    raise exception 'broker_order_not_found';
  end if;

  allowed :=
    (current_status = 'created' and p_next_status in ('submit_requested', 'rejected', 'cancel_requested')) or
    (current_status = 'submit_requested' and p_next_status in ('submit_acknowledged', 'reconciliation_pending', 'rejected', 'cancel_requested')) or
    (current_status = 'submit_acknowledged' and p_next_status in ('partial_fill', 'filled', 'cancel_requested', 'cancelled', 'rejected', 'reconciliation_pending', 'reconciled')) or
    (current_status = 'partial_fill' and p_next_status in ('partial_fill', 'filled', 'cancel_requested', 'cancelled', 'reconciliation_pending', 'reconciled')) or
    (current_status = 'reconciliation_pending' and p_next_status in ('submit_acknowledged', 'partial_fill', 'filled', 'cancelled', 'rejected', 'reconciled')) or
    (current_status = 'cancel_requested' and p_next_status in ('cancelled', 'reconciliation_pending')) or
    (current_status = p_next_status);

  if not allowed then
    raise exception 'invalid_broker_order_transition:%->%', current_status, p_next_status;
  end if;

  return query
  update public.broker_orders
  set status = p_next_status,
      broker_order_id = coalesce(p_broker_order_id, broker_order_id),
      payload = coalesce(p_payload, '{}'::jsonb),
      submitted_at = case when p_next_status in ('submit_requested', 'submit_acknowledged') then coalesce(submitted_at, timezone('utc', now())) else submitted_at end,
      reconciled_at = case when p_next_status in ('partial_fill', 'filled', 'cancelled', 'rejected', 'reconciled') then timezone('utc', now()) else reconciled_at end,
      updated_at = timezone('utc', now())
  where client_order_id = p_client_order_id
  returning *;
end;
$$;
