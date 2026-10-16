const test = require('node:test');
const assert = require('node:assert/strict');
const {describe, fresh} = require('../workspace-status.js');
const now = Date.parse('2026-10-16T13:31:00Z');
const iso = new Date(now).toISOString();
const ready = {signedIn:true, verified:true, controlAt:iso, accountAt:iso, jobsLoaded:true,
 health:{timestamp:iso, scheduler_last_tick_at:iso, worker_last_seen_at:iso, worker_online:true, market_open:true}};
test('guest and unverified accounts get setup instructions without claiming execution', () => {
 assert.match(describe({},now).title,/הדמו/);
 assert.match(describe({signedIn:true},now).title,/נדרש חיבור/);
 assert.match(describe({},now).activity,/התחבר/);
});
test('market state requires fresh health and scheduler evidence', () => {
 assert.equal(describe(ready,now).market,'השוק פתוח');
 assert.equal(describe({...ready,health:{...ready.health,market_open:false}},now).market,'השוק סגור');
 assert.equal(describe({...ready,health:{...ready.health,scheduler_last_tick_at:'2026-10-16T13:00:00Z'}},now).market,'מצב השוק לא אומת');
 assert.match(describe({...ready,health:null},now).title,/אינה מאומתת/);
});
test('pause and stop override healthy runtime and never request resume', () => {
 assert.match(describe({...ready,paused:true},now).title,/מושהה/);
 assert.match(describe({...ready,emergency:true},now).title,/נעצר/);
 assert.match(describe({...ready,paused:true},now).next,/אין חידוש אוטומטי/);
});
test('successful cycle is not presented as a broker fill or proof of return', () => {
 const result=describe({...ready,jobs:[{status:'succeeded',finished_at:iso}]},now);
 assert.equal(result.activity,'מחזור הושלם');
 assert.match(result.next,/אינה בהכרח עסקה/);
 assert.equal(result.jobTime,iso);
});
test('missing, failed and empty history are different states', () => {
 assert.match(describe({...ready,jobsLoaded:false},now).activity,/טרם נטענה/);
 assert.match(describe(ready,now).activity,/אין עדיין מחזור/);
 assert.match(describe({...ready,jobs:[{status:'failed'}]},now).title,/דורש בדיקה/);
 assert.match(describe({...ready,jobs:[{status:'skipped'}]},now).activity,/דולג/);
});
test('worker liveness and account freshness are independent, with expired and future stamps rejected', () => {
 assert.match(describe({...ready,health:{...ready.health,worker_last_seen_at:'2026-10-16T13:00:00Z'}},now).title,/אינה מאומתת/);
 assert.match(describe({...ready,accountAt:null},now).account,/אין נתוני/);
 assert.equal(fresh('2026-10-17T00:00:00Z',now),false);
 assert.match(describe({...ready,controlAt:null},now).title,/אינו עדכני/);
});
