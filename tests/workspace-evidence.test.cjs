const test = require('node:test');
const assert = require('node:assert/strict');
const {summarize} = require('../workspace-evidence.js');
const {markdown} = require('../scripts/evidence_report.cjs');
const names = ['champion','benchmark_spy','cash','momentum','mean_reversion','defensive'];
const now = '2026-10-23T10:00:00Z';
function cycle(day, value=100, overrides={}) {
  return names.map(strategy => ({execution_request_id:day, strategy, equity:strategy==='cash'?100:value,
    observed_at:day+'T19:45:00Z', user_id:'u',broker_account_id:'a',
    payload:{source:'shadow_simulation',schema_version:2,cost_bps:10,champion_id:'model'},...overrides}));
}
test('same-period daily normalization, drawdown and intervals are computed without claiming proof',()=>{
  const rows = [...cycle('2026-10-19',100),...cycle('2026-10-20',110),...cycle('2026-10-21',99)];
  const r=summarize(rows,{now});
  assert.equal(r.daily_marks,3); assert.equal(r.return_intervals,2);
  assert.ok(Math.abs(r.comparisons[0].return+0.01)<1e-10);
  assert.ok(Math.abs(r.comparisons[0].max_drawdown-0.1)<1e-10);
  assert.equal(r.profitability_proven,false); assert.equal(r.release_eligible,false);
});
test('incomplete, cross-account, mismatched-model and duplicate pairs cannot inflate evidence',()=>{
  const split=cycle('2026-10-20'); split[0].broker_account_id='another';
  const model=cycle('2026-10-21'); model[0].payload={...model[0].payload,champion_id:'other'};
  const duplicate=cycle('2026-10-22'); duplicate.push({...duplicate[0]});
  const r=summarize([...cycle('2026-10-19'),...split,...model,...duplicate],{now});
  assert.equal(r.daily_marks,1); assert.equal(r.comparisons.length,0);
});
test('today, weekends, after-hours, invalid prices and incompatible assumptions are excluded',()=>{
  const bad=cycle('2026-10-20'); bad[0].equity=NaN;
  const costs=cycle('2026-10-21'); costs[0].payload={...costs[0].payload,cost_bps:0};
  const after=cycle('2026-10-22',100,{observed_at:'2026-10-22T21:00:00Z'});
  const r=summarize([...cycle('2026-10-19'),...bad,...costs,...after,...cycle('2026-10-23'),...cycle('2026-10-18')],{now});
  assert.equal(r.daily_marks,1);
});
test('only the last paired observation of each prior day counts and models never stitch',()=>{
  const later=cycle('2026-10-20',105,{execution_request_id:'later',observed_at:'2026-10-20T19:50:00Z'});
  const changed=cycle('2026-10-22',120).map(r=>({...r,payload:{...r.payload,champion_id:'new'}}));
  const rows=[...cycle('2026-10-19'),...cycle('2026-10-20'),...later];
  assert.equal(summarize(rows.reverse(),{now}).daily_marks,2);
  assert.equal(summarize([...rows,...changed],{now}).daily_marks,1);
  assert.equal(summarize([],{now}).daily_marks,0);
});
test('time threshold alone never certifies profitability, and truncation is explicit',()=>{
  const rows=[];
  for(let day=1;day<=30;day++) rows.push(...cycle('2026-09-'+String(day).padStart(2,'0')));
  const r=summarize(rows,{now,truncated:true});
  assert.ok(r.return_intervals>=20); assert.equal(r.remaining_intervals,0);
  assert.equal(r.profitability_proven,false); assert.equal(r.release_eligible,false); assert.equal(r.truncated,true);
});
test('numeric API values are accepted but null and blank values never become zero equity; cash-only behavior is explicit',()=>{
  const rows=[...cycle('2026-10-19'),...cycle('2026-10-20')].map(r=>({...r,equity:String(r.equity),payload:{...r.payload,exposure:0}}));
  const r=summarize(rows,{now}); assert.equal(r.daily_marks,2);
  assert.equal(r.champion_exposure_observations,2); assert.equal(r.champion_invested_cycles,0);
  for (const invalid of [null,'', ' ',Infinity]) {
    const broken=cycle('2026-10-21');broken[0].equity=invalid;
    assert.equal(summarize(broken,{now}).daily_marks,0);
  }
});
test('a report contains same-period returns and limitations, without raw account/request identifiers',()=>{
  const r=summarize([...cycle('2026-10-19'),...cycle('2026-10-20',101)],{now});
  const text=markdown(r);
  assert.ok(text.includes('1.0000%')); assert.ok(text.includes('50% SPY'));
  assert.ok(text.includes('XNYS')); assert.ok(text.includes('אין עדיין ראיות מספיקות'));
  assert.ok(!text.includes('broker_account_id')); assert.ok(!text.includes('execution_request_id'));
});
