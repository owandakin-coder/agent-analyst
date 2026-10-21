const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const html = fs.readFileSync('index.html', 'utf8');
const source = html.slice(html.indexOf('let hydrationPromise = null;'), html.indexOf('function mergeRemotePreferences()'));

function fixture() {
  let release;
  const workspace = new Promise(resolve => { release = resolve; });
  const calls = { me: 0, ready: 0 };
  const context = {
    authSession: { access_token: 'first' }, currentUser: null, currentPreferences: null,
    refreshAuthSession: async () => {}, getAuthHeaders: () => ({}),
    apiJSON: async () => { calls.me++; return { profile: { id: 'member' } }; },
    mergeRemotePreferences() {}, loadSettings() {}, updateIdentityUI() {},
    finishBootstrap() { calls.ready++; }, refreshMemberWorkspace: () => workspace,
    isAuthFailure: () => false, console,
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return { context, calls, release };
}

test('a slow broker workspace does not keep the authenticated screen blocked; concurrent restores share work', async () => {
  const { context, calls, release } = fixture();
  const first = context.hydrateCurrentUser();
  const second = context.hydrateCurrentUser();
  assert.equal(first, second);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(calls.me, 1);
  assert.equal(calls.ready, 1);
  release();
  await first;
});

test('a delayed identity response cannot restore an account after sign-out', async () => {
  const { context, calls } = fixture();
  let release;
  context.apiJSON = () => new Promise(resolve => { release = resolve; });
  const pending = context.hydrateCurrentUser();
  await new Promise(resolve => setImmediate(resolve));
  context.authSession = null;
  release({ profile: { id: 'old-member' } });
  await pending;
  assert.equal(context.currentUser, null);
  assert.equal(calls.ready, 0);
});
