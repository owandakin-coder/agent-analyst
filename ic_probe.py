import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
import live_faithful_replay as L
from config_loader import CFG
from data_manager import DataManager
from datetime import date
wins={"V 2020-21":("2020-01-01","2021-12-31"),"T 2022-24":("2022-01-01","2024-12-31"),"F 2025-":("2025-01-01",None)}
dm=DataManager(tickers=CFG.tickers,start=CFG.data_start,end=date.today().isoformat()); dm.load_all(force_download=False)
full=dm.get_aligned_data(); close=pd.DataFrame({t:df["close"] for t,df in full.items()})
for name,(s,e) in wins.items():
    r=L.replay(start=s,end=e)
    raw=pd.DataFrame(r["raw"],index=pd.DatetimeIndex(r["dates"]),columns=r["tickers"])
    out=[name, f"ret={r['live']['total_return']:+.1%} sh={r['live']['sharpe']:.2f} inv={r['avg_gross']:.0%}"]
    for h in (1,5,20):
        fwd=close.shift(-h)/close-1
        ics=[]; 
        for d in raw.index:
            f=fwd.loc[d].dropna()
            a=raw.loc[d].reindex(f.index)
            if len(f)>5 and a.std()>0: ics.append(spearmanr(a,f)[0])
        ics=np.array(ics); out.append(f"IC{h}d={ics.mean():+.3f} (t={ics.mean()/ (ics.std()/np.sqrt(len(ics))):+.1f}, n={len(ics)})")
    # raw sign accuracy for 5d
    fwd5=(close.shift(-5)/close-1).reindex(raw.index)
    pos=(raw>0.08)
    out.append(f"avg 5d fwd ret when buy-signal={fwd5[pos].stack().mean():+.2%} vs all={fwd5.stack().mean():+.2%}")
    print(" | ".join(out), flush=True)
