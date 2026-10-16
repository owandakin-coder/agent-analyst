import {brokerOrderEventPolicy} from "./broker_order_events.ts";
const scope={user_id:"fixture-user",broker_connection_id:"fixture-broker",broker_account_id:"fixture-account"};
Deno.test("duplicate attempts preserve the original broker order instead of rejecting its fill",()=>{
  if(brokerOrderEventPolicy(scope,scope,"duplicate_blocked")!=="event_only")throw new Error("Fill would be overwritten");
  if(brokerOrderEventPolicy(scope,scope,"submitted")!=="transition")throw new Error("Normal transition missing");
});
Deno.test("order events cannot mutate or attach to another user, connection or account",()=>{
  for(const key of ["user_id","broker_connection_id","broker_account_id"])for(const event of ["duplicate_blocked","submitted"]){
    let rejected=false;try{brokerOrderEventPolicy(scope,{...scope,[key]:"different"},event);}catch{rejected=true;}
    if(!rejected)throw new Error("Cross-scope order event accepted");
  }
});
