begin;
do $$
declare u uuid:=gen_random_uuid(); b uuid:=gen_random_uuid(); a uuid:=gen_random_uuid();
begin
  insert into auth.users(id,email) values(u,u::text||'@scanner-fixture.invalid');
  insert into public.broker_connections(id,user_id,enabled,last_verified_status,metadata)
    values(b,u,true,'verified','{"account_number":"fixture"}');
  if not public.claim_market_scan(b,a) then raise exception 'Initial scan not claimed'; end if;
  if public.claim_market_scan(b,gen_random_uuid()) then raise exception 'Concurrent scan allowed'; end if;
  update public.market_scans set status='completed',report='{"candidates":[]}' where broker_connection_id=b;
  if public.claim_market_scan(b,gen_random_uuid()) then raise exception 'Same-day duplicate allowed'; end if;
  update public.market_scans set scan_day=scan_day-1 where broker_connection_id=b;
  if not public.claim_market_scan(b,gen_random_uuid()) then raise exception 'Next-day scan blocked'; end if;
  update public.market_scans set status='failed',attempted_at=now()-interval '31 minutes' where broker_connection_id=b;
  if not public.claim_market_scan(b,gen_random_uuid()) then raise exception 'Failure not retryable'; end if;
  update public.broker_connections set enabled=false where id=b;
  update public.market_scans set scan_day=scan_day-1 where broker_connection_id=b;
  if public.claim_market_scan(b,gen_random_uuid()) then raise exception 'Disabled connection claimed'; end if;
  if has_table_privilege('authenticated','public.market_scans','SELECT') or has_table_privilege('anon','public.market_scans','SELECT')
    then raise exception 'Private scanner readable directly'; end if;
  if has_function_privilege('authenticated','public.claim_market_scan(uuid,uuid)','EXECUTE')
    then raise exception 'Member can claim scanner work'; end if;
end $$;
rollback;
