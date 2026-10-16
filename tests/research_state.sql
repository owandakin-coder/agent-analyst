begin;
do $$
declare
  test_user uuid := gen_random_uuid();
  test_broker uuid := gen_random_uuid();
  test_request uuid := gen_random_uuid();
  saved_version bigint;
  saved_state jsonb := '{"schema_version":1,"peak_net_worth":150000,"risk_peak":150000,"trailing_highs":{"AAPL":130},"entry_prices":{"AAPL":100},"trade_outcomes":{},"shadow_portfolios":{}}';
begin
  insert into auth.users(id,email) values(test_user,test_user::text || '@risk-state-test.invalid');
  insert into public.broker_connections(id,user_id) values(test_broker,test_user);
  insert into public.execution_requests(id,user_id,broker_connection_id,idempotency_key,status,worker_id,lease_expires_at)
    values(test_request,test_user,test_broker,'state-fixture','running','owned-worker',now()+interval '5 minutes');
  saved_state := saved_state || jsonb_build_object('research_samples', jsonb_build_array(jsonb_build_object(
    'strategy','momentum','timestamp',now(),'equity',100000,'source','shadow_simulation')));
  saved_version := public.save_strategy_runtime_state(test_request,'owned-worker',0,saved_state);
  if saved_version <> 1 then raise exception 'Wrong initial state version'; end if;
  if not exists(select 1 from public.strategy_runtime_state where user_id=test_user and state->'trailing_highs'->>'AAPL' = '130') then
    raise exception 'Trailing high did not survive persistence';
  end if;
  begin
    perform public.save_strategy_runtime_state(test_request,'owned-worker',0,saved_state);
    raise exception using errcode='ZX001', message='Stale state overwrite accepted';
  exception when raise_exception then null; end;
  update public.execution_requests set lease_expires_at=now()+interval '5 minutes' where id=test_request;
  update public.broker_connections set metadata='{"account_number":"other-account"}'::jsonb where id=test_broker;
  begin
    perform public.save_strategy_runtime_state(test_request,'owned-worker',2,saved_state);
    raise exception using errcode='ZX001', message='Account change reused old risk scope';
  exception when raise_exception then
    if sqlerrm not like 'Broker account changed%' then raise; end if;
  end;
  update public.execution_requests set status='succeeded' where id=test_request;
  insert into public.execution_requests(id,user_id,broker_connection_id,broker_account_id,idempotency_key,status,worker_id,lease_expires_at)
    values(gen_random_uuid(),test_user,test_broker,'other-account','new-account-fixture','running','owned-worker',now()+interval '5 minutes')
    returning id into test_request;
  perform public.save_strategy_runtime_state(test_request,'owned-worker',0,saved_state || '{"peak_net_worth":100000,"risk_peak":100000}'::jsonb);
  if (select count(*) from public.strategy_runtime_state where user_id=test_user) <> 2 then
    raise exception 'Broker accounts shared a risk state';
  end if;
  begin
    perform public.save_strategy_runtime_state(test_request,'other-worker',1,saved_state);
    raise exception using errcode='ZX001', message='Wrong worker accepted';
  exception when raise_exception then null; end;
  perform public.save_strategy_runtime_state(test_request,'owned-worker',1,saved_state);
  if (select count(*) from public.research_observations where execution_request_id=test_request) <> 1 then
    raise exception 'Checkpoint retries duplicated research evidence';
  end if;
  update public.execution_requests set lease_expires_at=now()-interval '1 second' where id=test_request;
  begin
    perform public.save_strategy_runtime_state(test_request,'owned-worker',2,saved_state);
    raise exception using errcode='ZX001', message='Expired lease accepted';
  exception when raise_exception then null; end;
  if has_function_privilege('authenticated','public.save_strategy_runtime_state(uuid,text,bigint,jsonb)','execute')
    or has_table_privilege('anon','public.strategy_runtime_state','select') then raise exception 'Private risk state exposed'; end if;
end;
$$;
select 'risk state, lease fencing, version conflict, evidence idempotency and permissions passed (rolled back)' as result;
rollback;
