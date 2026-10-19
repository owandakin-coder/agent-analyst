import {eligibleAssets,screenSnapshots} from "./market_scanner.ts";
const now=Date.parse("2026-10-19T00:00:00Z");
const asset=(symbol:string)=>({symbol,name:symbol,status:"active",tradable:true,class:"us_equity",exchange:"NASDAQ"});
const snapshot=(change=1)=>({dailyBar:{c:20*change,v:1000000,vw:20,t:"2026-10-16T04:00:00Z"},prevDailyBar:{c:20,t:"2026-10-15T04:00:00Z"}});
Deno.test("scanner excludes inactive, OTC, untradable and invalid symbols",()=>{
  const rows=eligibleAssets([asset("AAPL"),{...asset("OTC"),exchange:"OTC"},{...asset("NO"),tradable:false},asset("<script>"),{...asset("OLD"),status:"inactive"}]);
  if(rows.length!==1 || rows[0].symbol!=="AAPL")throw Error("Invalid universe");
});
Deno.test("scanner rejects stale, missing, malformed and low liquidity observations",()=>{
  const names=["OK","STALE","MISSING","LOW","PENNY","NAN","FUTURE"];
  const data={OK:snapshot(),STALE:{...snapshot(),dailyBar:{...snapshot().dailyBar,t:"2026-09-01T04:00:00Z"}},
    LOW:{...snapshot(),dailyBar:{...snapshot().dailyBar,v:1}},PENNY:{...snapshot(),dailyBar:{...snapshot().dailyBar,c:1}},
    NAN:{...snapshot(),dailyBar:{...snapshot().dailyBar,c:NaN}},FUTURE:{...snapshot(),dailyBar:{...snapshot().dailyBar,t:"2027-01-01T00:00:00Z"}}};
  const result=screenSnapshots(names.map(asset),data,now);
  if(result.candidates.length!==1 || result.candidates[0].symbol!=="OK")throw Error("Bad evidence ranked");
  if(Object.values(result.excluded).reduce((a,b)=>a+b,0)!==6)throw Error("Missing exclusion accounting");
});
Deno.test("attention ranking is deterministic and includes declining stocks",()=>{
  const result=screenSnapshots([asset("UP"),asset("DOWN"),asset("FLAT")],{UP:snapshot(1.05),DOWN:snapshot(.95),FLAT:snapshot()},now);
  if(result.candidates[0].symbol!=="DOWN" || result.candidates[2].symbol!=="FLAT")throw Error("Ranking inconsistent");
});
