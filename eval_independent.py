"""Judge the independent engine against the closing line - 2025/26
seasons of EPL / La Liga / Bundesliga, walk-forward by month.

For each eval month M: fit on all warehouse data before M (team DC +
XI elasticity), predict every fixture in M using pre-match XI quality,
then compare with the devigged closing odds (football-data CSVs).

Metrics: log loss overall (model vs market), and the divergence test -
in matches where model and close disagree by >= 5 points on an outcome,
whose side of the line was right (Brier on those matches)?

Run: python eval_independent.py
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ingest import FootballDataLoader
from model.independent import IndependentEngine, load_league, xi_quality, DB
from processing import DataCleaner
from statsdb.features import build_team_map

LEAGUES = {39: ("Premier League", "E0"), 140: ("La Liga", "SP1"),
           78: ("Bundesliga", "D1")}

conn = sqlite3.connect(DB)

# odds for 2025/26 from football-data (join target, never an input)
loader = FootballDataLoader(seasons=["2526"], leagues=[c for _, c in LEAGUES.values()])
odds = DataCleaner.clean(loader.load_all())
inv = np.array([1/odds["B365H"], 1/odds["B365D"], 1/odds["B365A"]]).T
odds[["mkt_H", "mkt_D", "mkt_A"]] = inv / inv.sum(axis=1, keepdims=True)

fd_teams = {lg: sorted(set(odds[odds.League == lg].HomeTeam))
            for lg, _ in LEAGUES.values()}
tmap = build_team_map(conn, fd_teams)

rows = []
for lid, (lname, code) in LEAGUES.items():
    matches = load_league(conn, lid)
    xi = xi_quality(conn, lid)
    xi_lookup = xi.set_index(["fixture_id", "team_id"])["xi_rating"]
    season = matches[matches["Date"] >= "2025-08-01"]
    print(f"{lname}: {len(season)} eval fixtures")

    for month in sorted(season["Date"].dt.to_period("M").unique()):
        m_start = month.to_timestamp()
        train = matches[matches["Date"] < m_start]
        xi_train = xi[xi["fixture_id"].isin(train["fixture_id"])]
        if len(train) < 500:
            continue
        eng = IndependentEngine().fit(train, xi_train)
        block = season[season["Date"].dt.to_period("M") == month]
        for _, r in block.iterrows():
            xh = xi_lookup.get((r["fixture_id"], r["home_id"]), np.nan)
            xa = xi_lookup.get((r["fixture_id"], r["away_id"]), np.nan)
            try:
                p = eng.predict_1x2(r["HomeTeam"], r["AwayTeam"],
                                    None if np.isnan(xh) else xh,
                                    None if np.isnan(xa) else xa)
            except KeyError:
                continue
            rows.append({"league": lname, "fixture_id": r["fixture_id"],
                         "date": r["Date"].normalize(),
                         "home_af": r["HomeTeam"], "away_af": r["AwayTeam"],
                         "home_id": r["home_id"], "away_id": r["away_id"],
                         "p_home": p["home"], "p_draw": p["draw"],
                         "p_away": p["away"],
                         "hg": int(r["FTHG"]), "ag": int(r["FTAG"])})

pred = pd.DataFrame(rows)
pred["home_fd"] = pred["home_id"].map(tmap)
pred["away_fd"] = pred["away_id"].map(tmap)
odds["date"] = odds["Date"].dt.normalize()
merged = pred.merge(odds[["date", "HomeTeam", "AwayTeam",
                          "mkt_H", "mkt_D", "mkt_A"]],
                    left_on=["date", "home_fd", "away_fd"],
                    right_on=["date", "HomeTeam", "AwayTeam"], how="inner")
print(f"\njoined with closing odds: {len(merged)} of {len(pred)} predictions")

y = np.select([merged.hg > merged.ag, merged.hg == merged.ag], [2, 1], 0)
model_p = merged[["p_away", "p_draw", "p_home"]].values
mkt_p = merged[["mkt_A", "mkt_D", "mkt_H"]].values

print(f"\n=== overall (n={len(merged)}) ===")
print(f"  independent model: log_loss={log_loss(y, model_p, labels=[0,1,2]):.4f}"
      f"  acc={(model_p.argmax(1)==y).mean():.4f}")
print(f"  closing market:    log_loss={log_loss(y, mkt_p, labels=[0,1,2]):.4f}"
      f"  acc={(mkt_p.argmax(1)==y).mean():.4f}")

print("\n=== divergence test: model vs close disagree >= 5 points ===")
for i, name in [(2, "home"), (0, "away")]:
    dv = model_p[:, i] - mkt_p[:, i]
    for side, mask in [("model HIGHER", dv >= 0.05), ("model LOWER", dv <= -0.05)]:
        if mask.sum() < 10:
            continue
        hit = (y[mask] == i).mean()
        m_avg, k_avg = model_p[mask, i].mean(), mkt_p[mask, i].mean()
        print(f"  {name:4s} {side:12s} n={mask.sum():4d}  outcome rate {hit:.3f} "
              f"| model said {m_avg:.3f}, close said {k_avg:.3f} "
              f"-> {'MODEL closer' if abs(hit-m_avg) < abs(hit-k_avg) else 'CLOSE closer'}")

# flat-stake EV test: back the outcome where model >= close + 5pts at
# closing odds (1/mkt devig ~ closing price w/o vig -> conservative)
print("\n=== paper bets at devigged closing prices (no vig!) ===")
pnl, nb = 0.0, 0
for i, col in [(2, "mkt_H"), (1, "mkt_D"), (0, "mkt_A")]:
    mask = (model_p[:, i] - mkt_p[:, i]) >= 0.05
    price = 1 / merged.loc[mask, col].values
    won = (y[mask] == i)
    pnl += (won * (price - 1) - (~won) * 1).sum()
    nb += mask.sum()
print(f"  {nb} bets, P/L {pnl:+.1f}u ({pnl/max(nb,1)*100:+.1f}% ROI)"
      f"  [vig-free prices; real prices would be worse]")
