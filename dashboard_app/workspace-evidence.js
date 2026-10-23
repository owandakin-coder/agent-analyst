/* Informational comparisons only. Model releases use model_registry.py. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.WorkspaceEvidence = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const strategies = ['champion', 'benchmark_spy', 'cash', 'momentum', 'mean_reversion', 'defensive'];
  const ny = new Intl.DateTimeFormat('en-CA', {timeZone:'America/New_York', year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hourCycle:'h23', weekday:'short'});
  function sessionParts(value) {
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return null;
    const p = Object.fromEntries(ny.formatToParts(date).map(p => [p.type, p.value]));
    return {day:`${p.year}-${p.month}-${p.day}`, minute:Number(p.hour)*60+Number(p.minute), weekday:p.weekday};
  }
  function metrics(values) {
    let peak = values[0], drawdown = 0;
    for (const value of values) { peak = Math.max(peak, value); drawdown = Math.max(drawdown, 1-value/peak); }
    return {return:values.at(-1)/values[0]-1, max_drawdown:drawdown};
  }
  function summarize(rows, options = {}) {
    const today = sessionParts(options.now || new Date()).day;
    const groups = new Map();
    let excluded = 0;
    for (const row of rows || []) {
      const p = row.payload || {}, time = Date.parse(row.observed_at), day = sessionParts(row.observed_at);
      const equity = (typeof row.equity === 'number' || (typeof row.equity === 'string' && row.equity.trim())) ? Number(row.equity) : NaN;
      if (!strategies.includes(row.strategy) || !Number.isFinite(equity) || equity <= 0 ||
          !row.execution_request_id || !day || day.day >= today || ['Sat','Sun'].includes(day.weekday) || day.minute < 570 || day.minute >= 960 ||
          p.source !== 'shadow_simulation' || p.schema_version !== 2 || p.cost_bps !== 10 || !p.champion_id) { excluded++; continue; }
      // Include account and user in the key even when the caller already scoped the response.
      const key = JSON.stringify([row.user_id || '', row.broker_account_id || '', row.execution_request_id, time]);
      if (!groups.has(key)) groups.set(key, {time, day:day.day, model:p.champion_id, scope:JSON.stringify([row.user_id || '',row.broker_account_id || '']), rows:new Map(), exposures:new Map(), invalid:false});
      const group = groups.get(key);
      if (group.rows.has(row.strategy) || group.model !== p.champion_id) group.invalid = true;
      group.rows.set(row.strategy, equity);
      if (typeof p.exposure === 'number' && Number.isFinite(p.exposure) && p.exposure >= 0) group.exposures.set(row.strategy,p.exposure);
    }
    const complete = [...groups.values()].filter(g => !g.invalid && strategies.every(s => g.rows.has(s))).sort((a,b) => a.time-b.time);
    const latest = complete.at(-1);
    const daily = new Map();
    // A model/account boundary starts a fresh comparison window. Never stitch accounts or models.
    let cohort = [];
    for (const group of complete) {
      if (group.model !== latest.model || group.scope !== latest.scope) { cohort = []; continue; }
      cohort.push(group);
    }
    for (const group of cohort) daily.set(group.day, group);
    const points = [...daily.values()];
    const intervals = Math.max(0, points.length-1);
    const comparisons = points.length >= 2 ? strategies.map(strategy => ({strategy, ...metrics(points.map(g => g.rows.get(strategy)))})) : [];
    const champion = comparisons.find(c => c.strategy === 'champion');
    const benchmark = comparisons.find(c => c.strategy === 'benchmark_spy');
    return {verdict:'insufficient_evidence', profitability_proven:false, release_eligible:false,
      daily_marks:points.length, return_intervals:intervals, minimum_intervals:20,
      remaining_intervals:Math.max(0,20-intervals), model_id:latest?.model || null,
      first_day:points[0]?.day || null, last_day:points.at(-1)?.day || null,
      paired_cycles:cohort.length, excluded_rows:excluded, incomplete_groups:groups.size-complete.length,
      champion_exposure_observations:cohort.filter(g => g.exposures.has('champion')).length,
      champion_invested_cycles:cohort.filter(g => (g.exposures.get('champion') || 0) > 1e-9).length,
      truncated:!!options.truncated, comparisons,
      excess_vs_spy:champion && benchmark ? champion.return-benchmark.return : null,
      daily_points:points.map(g => ({day:g.day, values:Object.fromEntries(strategies.map(s => [s,g.rows.get(s)/points[0].rows.get(s)*100]))})),
      limitations:['Shadow simulations with estimated costs; not broker returns.',
        'Daily marks are the last paired observations before today, not official closing prices.',
        'Exchange holidays, early closes and candidate staging are checked separately by the release gate.',
        'Twenty intervals are a minimum release rule, not proof of profitability.']};
  }
  return {summarize, metrics};
});
