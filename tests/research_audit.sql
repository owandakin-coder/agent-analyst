begin;
create temporary table audit_fixture_scope as select gen_random_uuid() as user_id, gen_random_uuid() as broker_id, gen_random_uuid() as request_id;
insert into auth.users(id,email) select user_id,user_id::text || '@audit.invalid' from audit_fixture_scope;
insert into public.broker_connections(id,user_id,metadata)
  select broker_id,user_id,'{"account_number":"audit-paper"}'::jsonb from audit_fixture_scope;
insert into public.execution_requests(id,user_id,broker_connection_id,broker_account_id,strategy_id,trigger_type,requested_mode,idempotency_key,status)
  select request_id,user_id,broker_id,'audit-paper','default','schedule','paper','audit:'||request_id::text,'succeeded' from audit_fixture_scope;
create temporary table audit_before as
-- AUDIT QUERY
do $$ begin
  if not exists(select 1 from audit_before where (research_audit->>'missing_account_marks')::int=1
    and (research_audit->>'missing_risk_checkpoints')::int=1 and (research_audit->>'missing_independent_research')::int=1
    and (research_audit->>'forward_execution_verified')::boolean=false) then raise exception 'Missing evidence was not detected'; end if;
end $$;
insert into public.strategy_runtime_state(user_id,broker_connection_id,broker_account_id,strategy_id,version,state)
  select user_id,broker_id,'audit-paper','default',1,'{"schema_version":1}'::jsonb from audit_fixture_scope;
insert into public.paper_account_marks(execution_request_id,user_id,broker_account_id,observed_at,equity,payload)
  select request_id,user_id,'audit-paper',now(),100000,'{"source":"paper_broker_snapshot"}'::jsonb from audit_fixture_scope;
insert into public.research_observations(execution_request_id,user_id,broker_account_id,strategy,observed_at,equity,payload)
  select request_id,user_id,'audit-paper',s,now(),100000,'{"source":"shadow_simulation"}'::jsonb
  from audit_fixture_scope cross join unnest(array['champion','cash','momentum','mean_reversion','defensive']) s;
create temporary table audit_after as
-- AUDIT QUERY
do $$ begin
  if not exists(select 1 from audit_after where (research_audit->>'missing_account_marks')::int=0
    and (research_audit->>'missing_risk_checkpoints')::int=0 and (research_audit->>'missing_independent_research')::int=0
    and (research_audit->>'scope_violations')::int=0 and (research_audit->>'forward_execution_verified')::boolean=true)
    then raise exception 'Complete evidence was not recognized'; end if;
end $$;
update public.paper_account_marks set broker_account_id='wrong-account' where execution_request_id=(select request_id from audit_fixture_scope);
create temporary table audit_wrong_scope as
-- AUDIT QUERY
do $$ begin
  if not exists(select 1 from audit_wrong_scope where (research_audit->>'scope_violations')::int=1
    and (research_audit->>'missing_account_marks')::int=1 and (research_audit->>'forward_execution_verified')::boolean=false)
    then raise exception 'Cross-account evidence was accepted'; end if;
end $$;
select 'research audit detected incomplete, complete and cross-account evidence (fixtures rolled back)' as result;
rollback;
