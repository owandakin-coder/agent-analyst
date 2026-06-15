alter table public.profiles
  alter column active_executor set default 'worker_pool';

update public.profiles
set active_executor = 'worker_pool'
where active_executor = 'github_actions';
