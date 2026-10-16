const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const html=fs.readFileSync('index.html','utf8');
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
