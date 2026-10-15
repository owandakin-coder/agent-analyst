begin;
-- Isolated fixtures and all control changes are invisible to workers and
-- rolled back. This test never decrypts credentials or contacts a broker.
do $$
declare
  test_user uuid := gen_random_uuid();
  test_broker uuid := gen_random_uuid();
begin
  update public.broker_connections set enabled = false;
  insert into auth.users(id, email) values (test_user, test_user::text || '@scheduler-test.invalid');
  insert into public.broker_connections(id, user_id, enabled, last_verified_status, api_key_encrypted, secret_key_encrypted)
    values (test_broker, test_user, true, 'verified', 'test-only', 'test-only');
  update public.control_state set state = state || '{"trading_enabled":true,"emergency_stop":false,"mode":"paper"}'::jsonb where id = 'global';
  if public.schedule_paper_executions(false) <> 0 then raise exception 'Closed market scheduled jobs'; end if;
  update public.control_state set state = state || '{"emergency_stop":true}'::jsonb where id = 'global';
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Emergency stop ignored'; end if;
  update public.control_state set state = state || '{"emergency_stop":false,"trading_enabled":false}'::jsonb where id = 'global';
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Pause ignored'; end if;
  update public.control_state set state = state || '{"trading_enabled":true}'::jsonb where id = 'global';
  update public.broker_connections set metadata = '{"auto_execution_enabled":false}' where id = test_broker;
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Opt-out ignored'; end if;
  update public.broker_connections set metadata = '{}', trading_mode = 'live' where id = test_broker;
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Live account scheduled'; end if;
  update public.broker_connections set trading_mode = 'paper', last_verified_status = 'failed' where id = test_broker;
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Unverified account scheduled'; end if;
  update public.broker_connections set last_verified_status = 'verified' where id = test_broker;
  insert into public.daily_risk_state(user_id, trading_day, breached, reset_required)
    values (test_user, (now() at time zone 'America/New_York')::date, true, true);
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Risk stop ignored'; end if;
  delete from public.daily_risk_state where user_id = test_user;
  if public.schedule_paper_executions(true) <> 1 then raise exception 'Eligible account did not schedule'; end if;
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Duplicate slot scheduled'; end if;
  update public.execution_requests set status = 'succeeded' where user_id = test_user;
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Completed slot scheduled twice'; end if;
  delete from public.execution_requests where user_id = test_user;
  insert into public.execution_requests(user_id, broker_connection_id, idempotency_key, created_at)
    values (test_user, test_broker, 'old-pending-job', now() - interval '2 days');
  if public.schedule_paper_executions(true) <> 0 then raise exception 'Old outstanding job ignored'; end if;
  begin
    insert into public.execution_requests(user_id, broker_connection_id, idempotency_key)
      values (test_user, test_broker, 'racing-manual-request');
    raise exception 'Database accepted concurrent active requests';
  exception when unique_violation then null;
  end;
  if has_function_privilege('anon', 'public.schedule_paper_executions(boolean)', 'execute')
    or has_function_privilege('authenticated', 'public.schedule_paper_executions(boolean)', 'execute') then
    raise exception 'Untrusted role can schedule trades';
  end if;
end;
$$;
select 'scheduler integration checks passed (rolled back)' as result;
rollback;
