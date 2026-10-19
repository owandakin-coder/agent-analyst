// Research screening only. This module cannot submit orders or change a model universe.
export type Row = Record<string, any>;
export function eligibleAssets(assets: Row[]): Row[] {
  return assets.filter(a => a.status === "active" && a.tradable === true && a.class === "us_equity"
    && ["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "NYSEARCA"].includes(a.exchange)
    && typeof a.symbol === "string" && /^[A-Z][A-Z0-9.-]{0,14}$/.test(a.symbol));
}
export function screenSnapshots(assets: Row[], snapshots: Row, now = Date.now()) {
  const candidates: Row[] = [];
  const excluded = {missing: 0, stale: 0, price: 0, liquidity: 0};
  for (const a of assets) {
    const s = snapshots[a.symbol];
    const b = s?.dailyBar, p = s?.prevDailyBar;
    const values = [b?.c, b?.v, b?.vw, p?.c];
    if (!values.every(v => typeof v === "number" && Number.isFinite(v) && v > 0)) { excluded.missing++; continue; }
    const observed = Date.parse(b.t), previous = Date.parse(p.t);
    // Weekends/holidays are allowed, but old/future or out-of-order bars never rank.
    if (!Number.isFinite(observed) || !Number.isFinite(previous) || observed > now || previous >= observed
      || now - observed > 4 * 86400000 || observed - previous > 5 * 86400000) { excluded.stale++; continue; }
    if (b.c < 5) { excluded.price++; continue; }
    const dollarVolume = b.v * b.vw;
    if (dollarVolume < 1000000) { excluded.liquidity++; continue; }
    const change = (b.c / p.c - 1) * 100;
    // Attention ranking: liquidity first, capped price movement second. Not a buy score.
    const score = Math.log10(dollarVolume) + Math.min(Math.abs(change), 10) / 10;
    candidates.push({symbol:a.symbol, name:String(a.name ?? a.symbol), exchange:a.exchange,
      price:b.c, change_pct:change, dollar_volume:dollarVolume, observed_at:b.t,
      score:Math.round(score * 1000) / 1000, reason:"liquidity_and_absolute_daily_move"});
  }
  candidates.sort((a,b) => b.score-a.score || a.symbol.localeCompare(b.symbol));
  return {candidates, excluded};
}
