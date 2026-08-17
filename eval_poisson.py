"""Ablation + holdout evaluation of the ZIP Dixon-Coles scoreline model.

Fits per-league on data before the holdout cutoff (same 80/20 split as
all prior evaluations) and scores on the untouched final 20%.
Run: python eval_poisson.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from model.poisson import fit_league_models

df = pd.read_parquet("data/processed_features.parquet")
cutoff_idx = int(len(df) * 0.8)
cutoff_date = df["Date"].iloc[cutoff_idx]
train = df[df["Date"] < cutoff_date]
hold = df[df["Date"] >= cutoff_date].copy()
print(f"train: {len(train)} matches (< {cutoff_date.date()}), "
      f"holdout: {len(hold)}")

y = hold["Result"].values          # 0=A, 1=D, 2=H
variants = {
    "plain Poisson":  {"zero_inflation": False, "dc_adjust": False},
    "+ZIP":           {"zero_inflation": True,  "dc_adjust": False},
    "+DC":            {"zero_inflation": False, "dc_adjust": True},
    "+ZIP +DC":       {"zero_inflation": True,  "dc_adjust": True},
}

results = {}
for name, kw in variants.items():
    models = fit_league_models(train, **kw)
    probs, ok = [], []
    for _, r in hold.iterrows():
        mdl = models.get(r["League"])
        if mdl is None:
            ok.append(False); probs.append([1/3]*3); continue
        p = mdl.predict_1x2(r["HomeTeam"], r["AwayTeam"])
        probs.append([p["away"], p["draw"], p["home"]])
        ok.append(True)
    probs = np.array(probs)
    ll = log_loss(y, probs, labels=[0, 1, 2])
    acc = (probs.argmax(axis=1) == y).mean()
    draw_pred = probs[:, 1].mean()
    results[name] = (ll, acc, draw_pred, models)
    print(f"{name:15s} log_loss={ll:.4f}  acc={acc:.4f}  "
          f"mean P(draw)={draw_pred:.4f}")

draw_actual = (y == 1).mean()
print(f"{'actual draw rate':15s} {draw_actual:.4f}")

# fitted parameters of the full model
print("\nfitted parameters (+ZIP +DC):")
for lg, mdl in results["+ZIP +DC"][3].items():
    p = mdl.params
    print(f"  {lg:15s} pi={p['pi']:.4f}  rho={p['rho']:+.4f}  "
          f"home_adv={np.exp(p['home_adv']):.3f}x  converged={p['converged']}")

# market reference on same holdout
mkt = hold[["norm_prob_A", "norm_prob_D", "norm_prob_H"]].values
mkt = mkt / mkt.sum(axis=1, keepdims=True)
print(f"\nmarket (devigged Bet365): log_loss={log_loss(y, mkt, labels=[0,1,2]):.4f}  "
      f"acc={(mkt.argmax(axis=1)==y).mean():.4f}  mean P(draw)={mkt[:,1].mean():.4f}")

# scoreline + totals quality of the full model
models = results["+ZIP +DC"][3]
sl_ll, tot_hits, n_sl = 0.0, 0, 0
exact_hits = 0
for _, r in hold.iterrows():
    mdl = models.get(r["League"])
    if mdl is None or pd.isna(r["FTHG"]):
        continue
    m = mdl.score_matrix(r["HomeTeam"], r["AwayTeam"])
    h, a = int(r["FTHG"]), int(r["FTAG"])
    if h <= mdl.max_goals and a <= mdl.max_goals:
        sl_ll += -np.log(max(m[h, a], 1e-10))
        top = mdl.top_scorelines(r["HomeTeam"], r["AwayTeam"], k=1)[0][0]
        exact_hits += (top == f"{h}-{a}")
        n_sl += 1
    t = mdl.predict_totals(r["HomeTeam"], r["AwayTeam"])
    tot_hits += ((t["over_2.5"] > 0.5) == (h + a > 2.5))

print(f"\nscoreline log-lik per match: {sl_ll/n_sl:.4f}")
print(f"exact-score hit rate (top scoreline): {exact_hits/n_sl:.4f} "
      f"(naive most-common-score baseline ~0.09-0.11)")
print(f"over/under 2.5 accuracy: {tot_hits/n_sl:.4f} (base rate "
      f"{max((hold['FTHG']+hold['FTAG']>2.5).mean(), 1-(hold['FTHG']+hold['FTAG']>2.5).mean()):.4f})")

# demo output
lg_models = models["Premier League"]
print("\ndemo - Arsenal vs Chelsea (Premier League model):")
print(f"  rates: lambda={lg_models.rates('Arsenal','Chelsea')[0]:.2f}, "
      f"mu={lg_models.rates('Arsenal','Chelsea')[1]:.2f}")
print(f"  1X2: {({k: round(v,3) for k,v in lg_models.predict_1x2('Arsenal','Chelsea').items()})}")
print(f"  totals: {({k: round(v,3) for k,v in lg_models.predict_totals('Arsenal','Chelsea').items()})}")
print(f"  top scorelines: {[(s, round(p,3)) for s,p in lg_models.top_scorelines('Arsenal','Chelsea')]}")
