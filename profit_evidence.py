import sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
import live_faithful_replay as L

WINS = {"P":("2016-07-01","2019-12-31"),"V":("2020-01-01","2021-12-31"),"T":("2022-01-01","2024-12-31"),"F":("2025-01-01",None)}

def job(args):
    label, win, ov = args
    s,e = WINS[win]
    r = L.replay(ov, start=s, end=e)
    eq = np.array(r["equity"]); 
    return label, win, (eq[1:]/eq[:-1]-1).tolist()

def stats(rets, name, seed=0):
    r = np.array(rets); n=len(r)
    ann = r.mean()*252; vol = r.std(ddof=1)*np.sqrt(252)
    t = r.mean()/(r.std(ddof=1)/np.sqrt(n))
    rng = np.random.default_rng(seed); B=10000; bl=20; nb=int(np.ceil(n/bl))
    means=[]
    for _ in range(B):
        idx = (rng.integers(0,n-bl,size=nb)[:,None]+np.arange(bl)).ravel()[:n]
        means.append(r[idx].mean()*252)
    means=np.array(means)
    lo,hi=np.percentile(means,[2.5,97.5])
    print(f"{name:28s} days={n} annRet={ann:+.1%} vol={vol:.1%} sharpe={ann/vol:.2f} t={t:+.2f} 95%CI[{lo:+.1%},{hi:+.1%}] P(ret<=0)={(means<=0).mean():.3f}")

if __name__=="__main__":
    variants = {"trend": {"SIGNAL_SOURCE":"trend"}, "model(prod)": {}}
    jobs=[(k,w,dict(v)) for k,v in variants.items() for w in WINS]
    out={}
    with ProcessPoolExecutor(4) as ex:
        for label,win,rets in ex.map(job, jobs):
            out.setdefault(label,{})[win]=rets
    for label,d in out.items():
        for w in WINS: stats(d[w], f"{label} {w}")
        allr=[x for w in WINS for x in d[w]]
        stats(allr, f"{label} ALL (4 windows)")
        oos=[x for w in ("V","T","F") for x in d[w]]
        stats(oos, f"{label} V+T+F")
