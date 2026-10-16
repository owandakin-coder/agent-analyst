begin;
do $$
declare u uuid := gen_random_uuid(); first_breach timestamptz;
begin
  insert into auth.users(id,email) values(u,u::text || '@daily-risk-fixture.invalid');
  perform public.record_daily_risk_event(u,current_date,false,'{"baseline_equity":100,"current_equity":99}');
  perform public.record_daily_risk_event(u,current_date,false,'{"baseline_equity":99,"current_equity":98}');
  if (select count(*) from public.daily_risk_state where user_id=u) <> 1 then raise exception 'Duplicate risk rows'; end if;
  if not exists(select 1 from public.daily_risk_state where user_id=u and baseline_equity=100 and current_equity=98) then raise exception 'Baseline replaced or observation not updated'; end if;
  perform public.record_daily_risk_event(u,current_date,true,'{"current_equity":70}');
  select breached_at into first_breach from public.daily_risk_state where user_id=u;
  perform public.record_daily_risk_event(u,current_date,false,'{"baseline_equity":110,"current_equity":110}');
  if not exists(select 1 from public.daily_risk_state where user_id=u and breached and reset_required and breached_at=first_breach and baseline_equity=100) then raise exception 'Breach latch was reset'; end if;
  if has_function_privilege('authenticated','public.record_daily_risk_event(uuid,date,boolean,jsonb)','EXECUTE') then raise exception 'Member can mutate risk latch'; end if;
end $$;
rollback;
