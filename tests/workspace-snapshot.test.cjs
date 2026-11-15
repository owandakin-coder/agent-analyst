const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(require('node:path').join(__dirname, '..', 'index.html'), 'utf8');
const start = html.indexOf('function paintPortfolioSnapshot(');
const source = html.slice(start, html.indexOf('\nfunction ', start + 10) > 0 ? html.indexOf('\n}\n', start) + 3 : undefined);

function fixture(snap) {
  const store = { atzma_portfolio_snap: snap === undefined ? undefined : JSON.stringify(snap) };
  const calls = [];
  const ctx = {
    PORTFOLIO_SNAPSHOT_KEY: 'atzma_portfolio_snap',
    localStorage: { getItem: k => store[k] ?? null },
    STOCKS: [], NAMES: {}, portBase: 0, portOpen: 0, lastPort: 0, Date,
    renderStocks: () => calls.push('stocks'), renderPositionsPage: () => calls.push('page'),
    updateHeroUI: () => calls.push('hero'), setText: (id, v) => calls.push(['text', id, v]),
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  return { ctx, calls };
}

const snap = (scope, at = Date.now()) => ({
  scope, at,
  acc: { equity: 1000, last_equity: 990 },
  positions: [{ symbol: 'AAPL', qty: 2, current_price: 100, market_value: 200, unrealized_pl: 5, change_today: 0.01 }],
});

test('the saved snapshot paints for the same user', () => {
  const { ctx, calls } = fixture(snap('u1'));
  ctx.paintPortfolioSnapshot('u1');
  assert.equal(ctx.portBase, 1000);
  assert.equal(ctx.STOCKS[0].t, 'AAPL');
  assert.ok(calls.includes('hero'));
});

test('another user, a stale snapshot or a corrupt one paints nothing', () => {
  for (const [stored, user] of [[snap('u1'), 'u2'], [snap('u1', Date.now() - 25 * 3600 * 1000), 'u1'],
    [{ scope: 'u1', at: Date.now(), acc: {}, positions: 'x' }, 'u1'], [undefined, 'u1']]) {
    const { ctx, calls } = fixture(stored);
    ctx.paintPortfolioSnapshot(user);
    assert.equal(ctx.portBase, 0);
    assert.equal(calls.length, 0);
  }
});
