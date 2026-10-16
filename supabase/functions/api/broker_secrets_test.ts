import {readStoredBrokerSecrets, replacementBrokerSecrets} from "./broker_secrets.ts";
function assert(value: boolean) { if (!value) throw new Error("Assertion failed"); }
Deno.test("unreadable stored credentials expose neither partial plaintext nor crypto errors",async()=>{
  const result=await readStoredBrokerSecrets({api_key_encrypted:"fixture-a",secret_key_encrypted:"fixture-b"},async value=>{
    if(value==="fixture-b") throw new Error("sensitive crypto detail"); return "fixture-plaintext";
  });
  assert(!result.readable && result.apiKey==="" && result.secretKey==="");
});
Deno.test("corrupt credentials can be replaced only with a complete new pair",()=>{
  const stored={apiKey:"",secretKey:"",readable:false};
  for(const pair of [["",""],["fixture-new",""],["","fixture-new"]]) {
    let rejected=false;try{replacementBrokerSecrets(stored,...pair as [string,string]);}catch{rejected=true;}assert(rejected);
  }
  const result=replacementBrokerSecrets(stored,"fixture-key","fixture-secret");
  assert(result.apiKey==="fixture-key" && result.secretKey==="fixture-secret");
});
Deno.test("readable stored credentials survive a label-only edit",async()=>{
  const stored=await readStoredBrokerSecrets({api_key_encrypted:"a",secret_key_encrypted:"b"},async value=>String(value));
  const result=replacementBrokerSecrets(stored,"","");assert(result.apiKey==="a" && result.secretKey==="b");
});
