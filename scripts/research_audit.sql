-- Read-only, aggregate-only inspection. No credentials, identities or orders.
with recent as (
  select * from public.execution_requests
  where created_at >= greatest(now() - interval '24 hours', timestamptz '2026-10-16 00:00:00Z')
    and requested_mode = 'paper'
), completed as (
  select * from recent where status = 'succeeded'
), scoped_evidence as (
  select r.id,
    exists(select 1 from public.paper_account_marks m where m.execution_request_id=r.id
      and m.user_id=r.user_id and m.broker_account_id=r.broker_account_id
      and m.payload->>'source'='paper_broker_snapshot') as has_mark,
    exists(select 1 from public.strategy_runtime_state s where s.user_id=r.user_id
      and s.broker_connection_id=r.broker_connection_id and s.broker_account_id=r.broker_account_id
      and s.strategy_id=r.strategy_id and s.version>0 and s.updated_at>=r.created_at) as has_checkpoint,
    (select count(distinct o.strategy) from public.research_observations o where o.execution_request_id=r.id
      and o.user_id=r.user_id and o.broker_account_id=r.broker_account_id
      and o.strategy in ('champion','cash','momentum','mean_reversion','defensive')
      and o.payload->>'source'='shadow_simulation') as research_strategies
  from completed r
), scope_violations as (
  select count(*) as count from (
    select o.execution_request_id from public.research_observations o join recent r on r.id=o.execution_request_id
      where o.user_id is distinct from r.user_id or o.broker_account_id is distinct from r.broker_account_id
    union all
    select m.execution_request_id from public.paper_account_marks m join recent r on r.id=m.execution_request_id
      where m.user_id is distinct from r.user_id or m.broker_account_id is distinct from r.broker_account_id
  ) mismatches
)
select jsonb_build_object(
  'scheduled_requests', (select count(*) from recent where trigger_type='schedule'),
  'succeeded', (select count(*) from completed),
  'skipped', (select count(*) from recent where status='skipped'),
  'missing_account_marks', (select count(*) from scoped_evidence where not has_mark),
  'missing_risk_checkpoints', (select count(*) from scoped_evidence where not has_checkpoint),
  'missing_independent_research', (select count(*) from scoped_evidence where research_strategies<5),
  'scope_violations', (select count from scope_violations),
  'broker_confirmed_orders', (select count(*) from public.broker_orders o join recent r on r.id=o.execution_request_id
    where o.user_id=r.user_id and o.broker_account_id=r.broker_account_id
      and o.payload->>'filled_qty' ~ '^[0-9]+([.][0-9]+)?$'
      and o.payload->>'filled_avg_price' ~ '^[0-9]+([.][0-9]+)?$'
      and (o.payload->>'filled_qty')::numeric>0 and (o.payload->>'filled_avg_price')::numeric>0),
  'forward_execution_verified', exists(select 1 from scoped_evidence e join recent r on e.id=r.id
    where r.trigger_type='schedule' and e.has_mark and e.has_checkpoint and e.research_strategies=5)
) as research_audit;
