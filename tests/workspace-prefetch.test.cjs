const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const html = fs.readFileSync('index.html', 'utf8');
const source = html.slice(html.indexOf('let memberPrefetch = null;'), html.indexOf('async function refreshMemberWorkspace()'));

function context(fetchImpl) {
  const calls = [];
  const c = {
    authSession: { access_token: 'tok' }, API_BASE: 'https://fixture.invalid', Date,
    getAuthHeaders: () => ({ Authorization: 'Bearer tok' }),
    apiJSON: async path => { calls.push(path); return {}; },
    fetch: (url, init) => { calls.push(url.replace('https://fixture.invalid', '')); return fetchImpl(url, init); },
  };
  vm.createContext(c);
  vm.runInContext(source + '; this.prefetchMemberWorkspace = prefetchMemberWorkspace; this.state = () => ({ memberPrefetch, portfolioPrefetch });', c);
  return { c, calls };
}

test('sign-in starts broker, jobs, account and positions together, keyed to the token', async () => {
  const { c, calls } = context(async () => ({ ok: true, json: async () => ({}) }));
  c.prefetchMemberWorkspace();
  const s = c.state();
  assert.equal(s.memberPrefetch.token, 'tok');
  assert.equal(s.portfolioPrefetch.token, 'tok');
  assert.deepEqual(calls.sort(), ['/account', '/me/broker', '/me/execution/jobs', '/positions']);
  const [a, p] = await Promise.all([s.portfolioPrefetch.account, s.portfolioPrefetch.positions]);
  assert.equal(a.res.ok, true);
  assert.equal(p.bodyError, null);
});

test('a failed prefetch settles as a value instead of an unhandled rejection', async () => {
  const { c } = context(async () => { throw new Error('offline'); });
  c.prefetchMemberWorkspace();
  const s = c.state();
  assert.equal((await s.portfolioPrefetch.account).error.message, 'offline');
  assert.equal((await s.portfolioPrefetch.positions).error.message, 'offline');
});
