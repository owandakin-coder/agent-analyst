const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const html=fs.readFileSync('index.html','utf8');

for (const change of ['user','connection','token','version']) {
 test(`research evidence discards delayed results after ${change} changes`,async()=>{
  let release;
  const pending=new Promise(resolve=>release=resolve);
  const context={currentUser:{id:'first'},currentBrokerConnection:{id:'broker-first'},authSession:{access_token:'first'},
   hasVerifiedBrokerConnection:()=>true,getAuthHeaders:()=>({}),apiJSON:()=>pending,
   renderResearchEvidence:()=>{throw Error('Stale research rendered');},document:{getElementById:()=>null}};
  vm.createContext(context);
  vm.runInContext(html.slice(html.indexOf('async function refreshResearchEvidence()'),html.indexOf('function hasOpenExecutionJob()')),context);
  const request=context.refreshResearchEvidence();
  if(change==='user')context.currentUser={id:'second'};
  if(change==='connection')context.currentBrokerConnection={id:'broker-second'};
  if(change==='token')context.authSession={access_token:'second'};
  if(change==='version')context.currentBrokerConnection.updated_at='changed';
  release({observations:[]}); await request;
 });
 if (change==='version') continue;
 test(`market scanner discards delayed results after ${change} changes`,async()=>{
  let release;
  const pending=new Promise(resolve=>release=resolve);
  const context={currentUser:{id:'first'},currentBrokerConnection:{id:'broker-first'},authSession:{access_token:'first'},
   hasVerifiedBrokerConnection:()=>true,getAuthHeaders:()=>({}),apiJSON:()=>pending,
   renderMarketScan:()=>{throw Error('Stale scanner data rendered');},setText:()=>{throw Error('Stale error rendered');}};
  vm.createContext(context);
  vm.runInContext(html.slice(html.indexOf('async function refreshMarketScan()'),html.indexOf('async function refreshResearchEvidence()')),context);
  const request=context.refreshMarketScan();
  if(change==='user')context.currentUser={id:'second'};
  if(change==='connection')context.currentBrokerConnection={id:'broker-second'};
  if(change==='token')context.authSession={access_token:'second'};
  release({scan:{status:'completed'}});
  await request;
 });
}
test('manual Paper request needs fresh open-market evidence and cannot bypass a pause or active job',()=>{
 const now=new Date().toISOString();
 const base=()=>({authSession:{access_token:'fixture'}, hasVerifiedBrokerConnection:()=>true,
  controlState:{mode:'Paper',tradingStatus:'Running',emergency:false},hasOpenExecutionJob:()=>false,
  controlObservedAt:now,WorkspaceStatus:require('../workspace-status.js'),
  runtimeHealth:{timestamp:now,market_open:true,worker_online:true,scheduler_last_tick_at:now,worker_last_seen_at:now}});
 const source=html.slice(html.indexOf('function canRequestPersonalRun()'),html.indexOf('async function requestPersonalRun()'));
 function allowed(change){const c=base();change(c);vm.createContext(c);vm.runInContext(source,c);return c.canRequestPersonalRun();}
 assert.equal(allowed(()=>{}),true);
 for(const change of [c=>c.controlState.emergency=true,c=>c.controlState.tradingStatus='Paused',c=>c.controlState.mode='Live',
  c=>c.hasOpenExecutionJob=()=>true,c=>c.hasVerifiedBrokerConnection=()=>false,c=>c.authSession=null,
  c=>c.runtimeHealth.market_open=false,c=>c.runtimeHealth.worker_online=false,c=>c.runtimeHealth=null,
  c=>c.controlObservedAt=null,c=>c.runtimeHealth.worker_last_seen_at='2020-01-01T00:00:00Z'])assert.equal(allowed(change),false);
});
for (const [fn,end] of [['fetchAlpacaData','// Fetch every 30s'],['loadAlpacaTrades','function fmtOrderTime']]) {
 for (const change of ['user','connection','token']) {
  test(`${fn} discards a delayed private response after ${change} changes`,async()=>{
   let release;
   const pending=new Promise(resolve=>{release=resolve;});
   const context={authSession:{access_token:'fixture'},currentUser:{id:'first'},currentBrokerConnection:{id:'broker-first'},
    hasVerifiedBrokerConnection:()=>true,API_BASE:'https://fixture.invalid',getAuthHeaders:()=>({}),
    fetch:()=>pending,console:{warn:()=>{}},alpacaAccount:null,ALL_TRADES:[],
    renderHomeAudit:()=>{throw new Error('A stale response rendered private data');}};
   vm.createContext(context);
   vm.runInContext(html.slice(html.indexOf(`async function ${fn}()`),html.indexOf(end,html.indexOf(`async function ${fn}()`))),context);
   const request=context[fn]();
   if(change==='user')context.currentUser={id:'second'};
   if(change==='connection')context.currentBrokerConnection={id:'broker-second'};
   if(change==='token')context.authSession={access_token:'new-session'};
   release({ok:true,json:async()=>({equity:123,last_equity:120,buying_power:100})});
   await request;
   assert.equal(context.alpacaAccount,null);
   assert.equal(context.ALL_TRADES.length,0);
  });
 }
}
