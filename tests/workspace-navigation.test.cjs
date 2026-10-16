const test=require('node:test');
const assert=require('node:assert/strict');
const {validRoute,nextRoute,isBackSwipe}=require('../workspace-navigation.js');

test('opening another detail replaces the visible destination without nesting panels',()=>{
  const first=nextRoute({tab:'settings',panel:null},{type:'panel',panel:'profile-sheet'});
  const second=nextRoute(first,{type:'panel',panel:'broker-sheet'});
  assert.deepEqual(second,{tab:'settings',panel:'broker-sheet'});
  assert.deepEqual(nextRoute(second,{type:'tab',tab:'home'}),{tab:'home',panel:null});
});
test('unknown and inherited object keys cannot become application routes',()=>{
  for(const panel of ['__proto__','constructor','not-a-panel']){
    assert.equal(validRoute({tab:'home',panel}),false);
    assert.throws(()=>nextRoute({tab:'home',panel:null},{type:'panel',panel}));
  }
  assert.equal(validRoute({tab:'unknown',panel:null}),false);
  assert.equal(validRoute(null),false);
});
test('intentional left swipe on a mobile detail can go back',()=>{
  assert.equal(isBackSwipe({dx:-150,dy:15,startX:280,width:390,duration:250}),true);
});
test('scrolling, controls, browser edges, pinch and desktop gestures do not navigate',()=>{
  const good={dx:-150,dy:15,startX:280,width:390,duration:250};
  for(const override of [{dy:130},{dx:-40},{dx:150},{startX:15},{startX:380},{interactive:true},{touches:2},{width:1440},{duration:1000}]){
    assert.equal(isBackSwipe({...good,...override}),false,JSON.stringify(override));
  }
});
