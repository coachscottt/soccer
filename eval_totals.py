"""Independent TOTALS vs the market's closing over/under - big-3 2025/26.

The engine's lambda/mu (results + XI, no market inputs) -> P(over 2.5),
walk-forward by month, judged against devigged closing totals prices
(Pinnacle closing PC>2.5 preferred, consensus AvgC fallback). Paper bets
struck at REAL closing prices when |model - market| >= 5 points.

Run: python eval_totals.py
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from model.independent import IndependentEngine, load_league, xi_quality, DB
from statsdb.features import build_team_map

LEAGUES = {39: ("Premier League", "E0"), 140: ("La Liga", "SP1"),
           78: ("Bundesliga", "D1")}
RAW = "data/raw/{code}_2526.csv"

conn = sqlite3.connect(DB)

# --- closing totals odds from cached full season files ---
odds_frames = []
for lid, (lname, code) in LEAGUES.items():
    o = pd.read_csv(RAW.format(code=code), encoding="utf-8-sig",
                    on_bad_lines="skip")
    o.columns = [str(c).strip() for c in o.columns]
    o["Date"] = pd.to_datetime(o["Date"], dayfirst=True, errors="coerce")
    over = o.get("PC>2.5")
    under = o.get("PC<2.5")
    src = "Pinnacle closing"
    if over is None or over.isna().mean() > 0.5:
        over, under, src = o.get("AvgC>2.5"), o.get("AvgC<2.5"), "Avg closing"
    o["over_odds"], o["under_odds"] = over, under
    o["League"] = lname
    odds_frames.append(o[["Date", "League", "HomeTeam", "AwayTeam",
                          "FTHG", "FTAG", "over_odds", "under_odds"]])
    print(f"{lname}: totals source = {src}, "
          f"coverage {o['over_odds'].notna().mean()*100:.0f}%")
odds = pd.concat(odds_frames).dropna(subset=["over_odds", "under_odds"])
odds["date"] = odds["Date"].dt.normalize()

fd_teams = {lg: sorted(set(odds[odds.League == lg].HomeTeam))
            for lg, _ in LEAGUES.values()}
tmap = build_team_map(conn, fd_teams)

# --- walk-forward independent totals predictions ---
rows = []
for lid, (lname, code) in LEAGUES.items():
    matches = load_league(conn, lid)
    xi = xi_quality(conn, lid)
    xi_lookup = xi.set_index(["fixture_id", "team_id"])["xi_rating"]
    season = matches[matches["Date"] >= "2025-08-01"]
    for month in sorted(season["Date"].dt.to_period("M").unique()):
        m_start = month.to_timestamp()
        train = matches[matches["Date"] < m_start]
        if len(train) < 500:
            continue
        eng = IndependentEngine().fit(
            train, xi[xi["fixture_id"].isin(train["fixture_id"])])
        block = season[season["Date"].dt.to_period("M") == month]
        for _, r in block.iterrows():
            xh = xi_lookup.get((r["fixture_id"], r["home_id"]), np.nan)
            xa = xi_lookup.get((r["fixture_id"], r["away_id"]), np.nan)
            try:
                lam, mu = eng.rates(
                    r["HomeTeam"], r["AwayTeam"],
                    None if np.isnan(xh) else xh,
                    None if np.isnan(xa) else xa)
            except KeyError:
                continue
            mtx = eng._matrix(lam, mu)
            size = mtx.shape[0]
            p_over = float(sum(mtx[i, j] for i in range(size)
                               for j in range(size) if i + j >= 3))
            rows.append({"date": r["Date"].normalize(),
                         "home_fd": tmap.get(r["home_id"]),
                         "away_fd": tmap.get(r["away_id"]),
                         "p_over": p_over, "x_total": lam + mu})

pred = pd.DataFrame(rows).dropna(subset=["home_fd", "away_fd"])
merged = pred.merge(odds, left_on=["date", "home_fd", "away_fd"],
                    right_on=["date", "HomeTeam", "AwayTeam"], how="inner")
print(f"\njoined: {len(merged)} fixtures with independent totals + closing "
      "over/under")

y = (merged["FTHG"] + merged["FTAG"] > 2.5).astype(int).values
inv_o, inv_u = 1/merged["over_odds"], 1/merged["under_odds"]
mkt_over = (inv_o / (inv_o + inv_u)).values
mod_over = merged["p_over"].values

def bll(p):
    return log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])

print(f"\n=== independent totals vs closing market (n={len(merged)}) ===")
print(f"  base rate over 2.5: {y.mean():.4f}")
print(f"  independent model: log_loss={bll(mod_over):.4f}  "
      f"acc={((mod_over>0.5)==y).mean():.4f}")
print(f"  closing market:    log_loss={bll(mkt_over):.4f}  "
      f"acc={((mkt_over>0.5)==y).mean():.4f}")
print(f"  always-base-rate:  log_loss={bll(np.full_like(mod_over, y.mean())):.4f}")

print("\n=== divergence test (>=5 pts) ===")
for side, mask, out in [("model OVER  vs mkt", mod_over - mkt_over >= 0.05, 1),
                        ("model UNDER vs mkt", mkt_over - mod_over >= 0.05, 0)]:
    if mask.sum() < 20:
        print(f"  {side}: n={mask.sum()} (too few)")
        continue
    rate = y[mask].mean() if out else 1 - y[mask].mean()
    mp = (mod_over[mask] if out else 1-mod_over[mask]).mean()
    kp = (mkt_over[mask] if out else 1-mkt_over[mask]).mean()
    verdict = "MODEL closer" if abs(rate-mp) < abs(rate-kp) else "CLOSE closer"
    print(f"  {side} n={mask.sum():4d}  actual {rate:.3f} | "
          f"model {mp:.3f} close {kp:.3f} -> {verdict}")

print("\n=== paper bets at REAL closing prices ===")
pnl, nb, wins = 0.0, 0, 0
mask_o = mod_over - mkt_over >= 0.05
won_o = y[mask_o] == 1
pnl += (won_o * (merged.loc[mask_o, "over_odds"].values - 1)
        - (~won_o) * 1.0).sum()
nb += int(mask_o.sum()); wins += int(won_o.sum())
mask_u = mkt_over - mod_over >= 0.05
won_u = y[mask_u] == 0
pnl += (won_u * (merged.loc[mask_u, "under_odds"].values - 1)
        - (~won_u) * 1.0).sum()
nb += int(mask_u.sum()); wins += int(won_u.sum())
print(f"  {nb} bets, {wins} won, P/L {pnl:+.1f}u ({pnl/max(nb,1)*100:+.1f}% ROI)")
