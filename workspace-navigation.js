/* One in-app detail surface, real browser history, and accessible navigation. */
(function (root) {
  'use strict';
  const tabs = ['home', 'alerts', 'charts', 'trades', 'settings'];
  const panels = ['profile-sheet','broker-sheet','execution-sheet','risk-sheet','email-sheet','model-sheet','retrain-sheet','export-sheet','regime-sheet','modal-overlay'];
  function validRoute(route) {
    return !!route && tabs.includes(route.tab) && (route.panel === null || panels.includes(route.panel));
  }
  function nextRoute(current, action) {
    if (!validRoute(current)) throw new Error('Invalid current route');
    if (action.type === 'tab' && tabs.includes(action.tab)) return {tab:action.tab, panel:null};
    if (action.type === 'panel' && panels.includes(action.panel)) return {tab:current.tab, panel:action.panel};
    throw new Error('Invalid workspace destination');
  }
  function isBackSwipe({dx,dy,startX,width,duration,touches=1,interactive=false}) {
    return !interactive && touches===1 && width<=900 && startX>28 && startX<width-28
      && duration>0 && duration<800 && dx < -90 && Math.abs(dx)>Math.abs(dy)*1.8;
  }
  // Small pure helpers are exercised without a browser or a broker connection.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports={validRoute,nextRoute,isBackSwipe};
    return;
  }
  const d=root.document;
  const titles={'profile-sheet':'Your profile','broker-sheet':'Broker connection','execution-sheet':'Activity & research','risk-sheet':'Risk preferences','email-sheet':'Daily summary','model-sheet':'Model details','retrain-sheet':'Training history','export-sheet':'Export your data','regime-sheet':'Market context','modal-overlay':'Create an alert'};
  const personalPanels=new Set(['profile-sheet','broker-sheet','execution-sheet']);
  const originalSwitch=root.switchTab, originalShowAuth=root.showAuthOverlay, originalHideAuth=root.hideAuthOverlay;
  const detail=d.getElementById('workspace-detail'), detailContent=d.getElementById('detail-content');
  const activePages=[...d.querySelectorAll('.page')];
  const auth=d.getElementById('auth-overlay');
  let route={tab:'home',panel:null}, depth=0, gesture=null, authReturn=null;
  const focusReturns=new Map();
  function authVisible(){return !auth.classList.contains('hidden');}
  function focusable(scope){return [...scope.querySelectorAll('button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]')].filter(el=>!el.closest('[hidden],[inert]') && el.getClientRects().length);}
  function safeState(){return {...(root.history.state || {}),atzmaUI:{...route,depth}};}
  function locationFor(value){
    // Email confirmation tokens belong to the existing auth flow, never to UI routes.
    if (/access_token=|refresh_token=/.test(root.location.hash)) return root.location.href;
    return root.location.pathname+root.location.search+'#view/'+value.tab+(value.panel?'/'+value.panel:'');
  }
  function paint(value, {restore=false}={}) {
    // Returning through history after signing out must never reveal the
    // previous member's profile or connection form.
    if(personalPanels.has(value.panel) && !currentUser) value={tab:value.tab,panel:null};
    route={...value};
    originalSwitch(value.tab);
    for(const page of activePages){
      const active=page.id==='pg-'+value.tab && !value.panel;
      page.inert=!active;
      page.setAttribute('aria-hidden',String(!active));
    }
    for(const id of panels){
      const el=d.getElementById(id);
      if(!el) continue;
      const on=id===value.panel;
      el.hidden=!on; el.inert=!on;
      el.classList.toggle('open',on);
      el.setAttribute('aria-hidden',String(!on));
    }
    detail.hidden=!value.panel;
    d.body.classList.toggle('has-detail',!!value.panel);
    d.getElementById('detail-title').textContent=titles[value.panel] || '';
    d.getElementById('detail-context').textContent=HDR[value.tab][0]+' / Workspace';
    if(value.panel) detail.scrollTop=0;
    for(const tab of d.querySelectorAll('.tab')){
      const current=tab.id==='tab-'+value.tab;
      tab.setAttribute('aria-current',current?'page':'false');
    }
    const restored=restore?focusReturns.get(depth+1):null;
    const target=restored?.isConnected && !restored.closest('[inert]') ? restored : d.getElementById(value.panel?'detail-title':'hdr-title');
    if(!authVisible()) target?.focus({preventScroll:true});
    d.title=(value.panel?titles[value.panel]:HDR[value.tab][0])+' · ATZMA';
  }
  function navigate(value){
    if(!validRoute(value) || (route.tab===value.tab && route.panel===value.panel)) return;
    focusReturns.set(depth+1,d.activeElement);
    depth+=1; paint(value);
    root.history.pushState(safeState(),'',locationFor(route));
  }
  function back(){
    if(authVisible()) {root.hideAuthOverlay();return;}
    if(depth>0) root.history.back();
    else if(route.panel){paint({tab:route.tab,panel:null},{restore:true});root.history.replaceState(safeState(),'',locationFor(route));}
  }
  for(const id of panels){
    const el=d.getElementById(id);
    if(!el) continue;
    el.removeAttribute('onclick');
    el.hidden=true; el.inert=true;
    el.setAttribute('role','region'); el.setAttribute('aria-label',titles[id]);
    detailContent.append(el);
  }
  root.switchTab=name=>navigate(nextRoute(route,{type:'tab',tab:name}));
  root.openSheet=id=>navigate(nextRoute(route,{type:'panel',panel:id}));
  root.closeSheet=id=>{if(route.panel===id)back();};
  root.openModal=()=>root.openSheet('modal-overlay');
  root.closeModal=event=>{if(!event || event.target===d.getElementById('modal-overlay'))back();};
  d.getElementById('workspace-back').addEventListener('click',back);
  root.addEventListener('popstate',event=>{
    const saved=event.state?.atzmaUI;
    if(authVisible()) root.hideAuthOverlay();
    depth=Number.isInteger(saved?.depth)?saved.depth:0;
    paint(validRoute(saved)?saved:{tab:'home',panel:null},{restore:true});
  });
  root.showAuthOverlay=()=>{
    authReturn=d.activeElement;
    if(!currentUser && personalPanels.has(route.panel)){
      depth=0;paint({tab:'home',panel:null});
      root.history.replaceState(safeState(),'',locationFor(route));
      for(const id of ['broker-api-key','broker-secret-key'])d.getElementById(id).value='';
      d.getElementById('broker-sheet').dataset.dirty='false';
    }
    originalShowAuth();
    d.getElementById('app').inert=true;
    auth.setAttribute('aria-hidden','false');
    // Auth is the only modal transaction; it never stacks another dialog.
    (d.getElementById('auth-email') || auth).focus({preventScroll:true});
  };
  // Successful authentication must be able to dismiss the form even while
  // submitAuth still owns the busy flag. Only user dismissal is blocked.
  root.hideAuthOverlay=()=>{
    originalHideAuth();
    d.getElementById('app').inert=false;
    auth.setAttribute('aria-hidden','true');
    if(authReturn?.isConnected && !authReturn.closest('[inert]')) authReturn.focus({preventScroll:true});
  };
  const dismissAuth=()=>{if(!authBusy)root.hideAuthOverlay();};
  d.getElementById('auth-dismiss').addEventListener('click',dismissAuth);
  d.querySelector('.skip-link').addEventListener('click',event=>{
    event.preventDefault();d.getElementById('workspace-main').focus({preventScroll:true});
  });
  d.addEventListener('keydown',event=>{
    if(authVisible()){
      if(event.key==='Escape'){event.preventDefault();dismissAuth();}
      if(event.key==='Tab'){
        const items=focusable(auth),first=items[0],last=items[items.length-1];
        if(event.shiftKey && (d.activeElement===first || !auth.contains(d.activeElement))){event.preventDefault();last?.focus();}
        else if(!event.shiftKey && (d.activeElement===last || !auth.contains(d.activeElement))){event.preventDefault();first?.focus();}
      }
      return;
    }
    if(event.key==='Escape' && route.panel){event.preventDefault();back();}
    // Native keyboard access for legacy interactive rows, without turning
    // a row containing a switch into a second competing control.
    if((event.key==='Enter' || event.key===' ') && event.target.matches('[role=button]:not(button)')){
      event.preventDefault();event.target.click();
    }
  });
  function decorateControls(scope){
    const selector='[onclick]:not(button):not(input):not(select)';
    const candidates=[...(scope.matches?.(selector)?[scope]:[]),...scope.querySelectorAll(selector)];
    for(const el of candidates){
      if(el.matches('.sheet-box,.modal,a[href]')) continue;
      if(el.querySelector('button,input,select,a')){
        const textTarget=el.matches('.s-item')?el.querySelector('.s-l'):null;
        if(textTarget){textTarget.setAttribute('role','button');textTarget.tabIndex=0;}
        continue;
      }
      el.setAttribute('role','button');el.tabIndex=0;
    }
    for(const input of scope.querySelectorAll('input,select')){
      const rowLabel=input.closest('.s-item')?.querySelector('.s-txt-main')?.textContent;
      if(rowLabel && !input.hasAttribute('aria-label')) input.setAttribute('aria-label',rowLabel);
      if(input.getAttribute('aria-label') || input.labels?.length) continue;
      const prior=input.previousElementSibling;
      if(prior?.tagName==='LABEL' && input.id) prior.htmlFor=input.id;
      else if(input.placeholder) input.setAttribute('aria-label',input.placeholder);
    }
  }
  decorateControls(d);
  for(const toggle of d.querySelectorAll('.s-item .tog'))toggle.addEventListener('click',event=>event.stopPropagation());
  const brokerForm=d.getElementById('broker-sheet');
  for(const event of ['input','change'])brokerForm.addEventListener(event,()=>{brokerForm.dataset.dirty='true';});
  const nav=d.querySelector('.tabbar'),footer=nav.querySelector('.nav-footer');
  for(const name of tabs)nav.insertBefore(d.getElementById('tab-'+name),footer);
  const positionsIcon=d.querySelector('#tab-alerts svg');
  positionsIcon.innerHTML='<rect x="3" y="6" width="18" height="14" rx="3" stroke="currentColor" stroke-width="1.7"/><path d="M8 6V4h8v2M3 12h18M10 12v3h4v-3" stroke="currentColor" stroke-width="1.7"/>';
  const settingIcons={
    '👤':'M20 21v-2a6 6 0 0 0-12 0v2M14 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8',
    '🔗':'m10 13 4-4M8 15l-2 2a3 3 0 0 1-4-4l5-5a3 3 0 0 1 4 0M13 11a3 3 0 0 1 0-4l5-5a3 3 0 0 1 4 4l-2 2',
    '🛡️':'m12 3 8 4v5c0 5-8 9-8 9s-8-4-8-9V7l8-4Zm-4 9 3 3 5-6',
    '⚖️':'M3 17a9 9 0 0 1 18 0M12 17l4-6M6 20h12',
    '🤖':'M6 5h12v14H6V5Zm3 4v2m6-2v2m-6 4h6M12 2v3M3 9v6m18-6v6',
    '🛑':'M6 3h12l3 3v12l-3 3H6l-3-3V6l3-3Zm2 5h8v8H8V8Z',
    '🎲':'M4 20V4m0 16h16M8 16v-5m5 5V6m5 10v-8',
    '📡':'M3 12h4l3-8 4 16 3-8h4',
    '🚀':'m9 5 10 7-10 7V5Z',
    '🔔':'M6 9a6 6 0 0 1 12 0c0 6 3 8 3 8H3s3-2 3-8Zm4 12h4',
    '📧':'M3 5h18v14H3V5Zm0 1 9 7 9-7',
    '🧠':'M7 3h10v18H7V3ZM3 7h4m-4 5h4m-4 5h4m10-10h4m-4 5h4m-4 5h4M10 8h4v8h-4V8Z',
    '📅':'M3 6h18v15H3V6Zm4-3v6m10-6v6M3 11h18',
    '🗑️':'M4 6h16M9 3h6M6 6l1 15h10l1-15M10 10v7m4-7v7',
    '📤':'M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6'
  };
  for(const icon of d.querySelectorAll('.s-ico')){
    const path=settingIcons[icon.textContent.trim()];
    if(path)icon.innerHTML='<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="'+path+'" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  }
  const observer=new MutationObserver(records=>{
    for(const record of records)for(const node of record.addedNodes)if(node.nodeType===1)decorateControls(node);
  });
  observer.observe(d.getElementById('app'),{childList:true,subtree:true});
  detail.addEventListener('touchstart',event=>{
    const t=event.touches[0];
    gesture=t?{x:t.clientX,y:t.clientY,at:performance.now(),touches:event.touches.length,
      interactive:!!event.target.closest('input,select,textarea,button,a,canvas,svg,summary,[role=button],[data-no-swipe]')}:null;
  },{passive:true});
  detail.addEventListener('touchmove',event=>{if(gesture && event.touches.length!==1)gesture.interactive=true;},{passive:true});
  detail.addEventListener('touchcancel',()=>{gesture=null;},{passive:true});
  detail.addEventListener('touchend',event=>{
    const start=gesture,t=event.changedTouches[0];gesture=null;
    if(!start || !t || !route.panel || authVisible())return;
    if(isBackSwipe({dx:t.clientX-start.x,dy:t.clientY-start.y,startX:start.x,width:root.innerWidth,
      duration:performance.now()-start.at,touches:start.touches,interactive:start.interactive}))back();
  },{passive:true});
  const segments=[...d.querySelectorAll('[data-execution-section]')];
  function setExecutionSection(name){
    for(const segment of segments)segment.setAttribute('aria-pressed',String(segment.dataset.executionSection===name));
    d.getElementById('execution-evidence-section').hidden=name!=='evidence';
    d.getElementById('execution-jobs-section').hidden=name!=='queue';
  }
  for(const segment of segments)segment.addEventListener('click',()=>setExecutionSection(segment.dataset.executionSection));
  setExecutionSection('evidence');
  d.getElementById('workspace-refresh').addEventListener('click',()=>{
    fetchControlState();
    fetchRuntimeHealth();
    if(currentUser)refreshMemberWorkspace();
    else toast('Sign in to refresh your personal account.');
  });
  // Move secondary connection details behind a disclosure without removing fields.
  const base=d.getElementById('broker-base-url'),mode=d.getElementById('broker-trading-mode');
  const options=d.createElement('details');options.className='connection-options';
  options.innerHTML='<summary>Connection options · Paper environment</summary>';
  const modeLabel=mode.previousElementSibling,baseLabel=base.previousElementSibling,modeHint=mode.nextElementSibling;
  base.parentNode.insertBefore(options,modeLabel);
  for(const node of [modeLabel,mode,modeHint,baseLabel,base])options.append(node);
  // Keep documentation available without making every connection form long.
  const secret=d.getElementById('broker-secret-key'),guide=secret.nextElementSibling;
  if(guide?.tagName==='DIV'){
    const help=d.createElement('details');help.className='connection-options';
    help.innerHTML='<summary>Where to find your Alpaca API keys</summary>';
    guide.before(help);help.append(guide);
  }
  root.WorkspaceUI={back,setExecutionSection};
  const saved=root.history.state?.atzmaUI;
  const hash=root.location.hash.match(/^#view\/(home|alerts|charts|trades|settings)(?:\/([a-z-]+))?$/);
  // Direct URLs may select a top-level destination, never open personal
  // panels before the normal authentication/verification flow has run.
  const initial=validRoute(saved)?{tab:saved.tab,panel:null}:{tab:hash?.[1]||'home',panel:null};
  depth=0;paint(initial);
  root.history.replaceState(safeState(),'',locationFor(route));
  if(authVisible()) {d.getElementById('app').inert=true;auth.setAttribute('aria-hidden','false');}
})(typeof window!=='undefined'?window:globalThis);
