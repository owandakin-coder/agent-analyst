"""Do these features carry out-of-sample information about next-period returns?

Cross-sectional supervised probe (minutes, not hours): fit on <=2019 (20-day
embargo), choose hyperparameters on 2020-21, report untouched 2022-24 and 2025-.
If rank IC stays ~0, retraining the PPO on the same inputs cannot help.
"""
import sys, numpy as np, pandas as pd
from datetime import date
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from config_loader import CFG
from data_manager import DataManager
from live_trader import FEATURE_COLS

dm = DataManager(tickers=CFG.tickers, start=CFG.data_start, end=date.today().isoformat()); dm.load_all(force_download=False)
full = dm.get_aligned_data(); tick = [t for t in full if t != "SPY"] if False else list(full)
close = pd.DataFrame({t: d["close"] for t, d in full.items()})
H = int(sys.argv[1]) if len(sys.argv) > 1 else 20

def build():
    rows = []
    for t in tick:
        d = full[t]; c = d["close"]
        f = d[[x for x in FEATURE_COLS if x in d.columns]].copy()
        for k in (5, 20, 60, 126):
            f[f"mom{k}"] = c / c.shift(k) - 1
        f["dist200"] = c / c.rolling(200).mean() - 1
        f["vol60"] = c.pct_change().rolling(60).std()
        f["ticker"] = t
        f["fwd"] = c.shift(-H) / c - 1
        rows.append(f)
    df = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    df = df.rename(columns={df.columns[0]: "date"})
    return df
df = build().dropna()
feat = [c for c in df.columns if c not in ("date", "ticker", "fwd")]
g = df.groupby("date")
X = g[feat].rank(pct=True) - 0.5
y = g["fwd"].rank(pct=True) - 0.5
df = df.assign(**{f"x_{c}": X[c] for c in feat}, y=y)
xc = [f"x_{c}" for c in feat]
D = pd.to_datetime(df["date"])
tr = D <= "2019-12-31"
va = (D >= "2020-01-31") & (D <= "2021-12-31")
te1 = (D >= "2022-01-01") & (D <= "2024-12-31")
te2 = D >= "2025-01-01"

def ic(sub, pred):
    s = sub.assign(p=pred)
    keep = sorted(s["date"].unique())[::5]  # every 5th date, for the t-stat
    s = s[s["date"].isin(keep)]
    vals = [spearmanr(x["p"], x["fwd"])[0] for _, x in s.groupby("date") if len(x) > 8]
    v = np.array(vals); return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), len(v)

def spread(sub, pred, k=3):
    s = sub.assign(p=pred); out = []
    for _, x in s.iloc[::1].groupby("date"):
        if len(x) < 2 * k + 2: continue
        x = x.sort_values("p"); out.append(x["fwd"].iloc[-k:].mean() - x["fwd"].iloc[:k].mean())
    return float(np.mean(out))

def report(name, model):
    model.fit(df.loc[tr, xc], df.loc[tr, "y"])
    res = []
    for label, m in (("val20-21", va), ("test22-24", te1), ("test25-", te2)):
        p = model.predict(df.loc[m, xc]); a, t, n = ic(df.loc[m], p)
        res.append(f"{label}: IC={a:+.3f} t={t:+.1f} top3-bot3 {H}d={spread(df.loc[m], p):+.2%}")
    print(f"{name:28s} | " + " | ".join(res), flush=True)

print(f"horizon={H}d  features={len(feat)}  train rows={tr.sum()}")
for a in (10, 100, 1000): report(f"ridge a={a}", Ridge(alpha=a))
for d, lr in ((2, 0.05), (3, 0.05)):
    report(f"hgb depth={d}", HistGradientBoostingRegressor(max_depth=d, learning_rate=lr, max_iter=150, min_samples_leaf=200, random_state=0))
# single-feature references
for c in ("mom126", "mom60", "dist200", "rsi", "vol60"):
    res=[]
    for label,m in (("val20-21",va),("test22-24",te1),("test25-",te2)):
        a,t,n=ic(df.loc[m], df.loc[m, f"x_{c}"].values); res.append(f"{label}: IC={a:+.3f} t={t:+.1f}")
    print(f"single {c:20s} | "+" | ".join(res), flush=True)
