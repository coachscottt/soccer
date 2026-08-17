"""v3 judgment: xG-DC baseline + style-interaction residuals vs closes.

Walk-forward big-3 2025/26, both markets, v2 (baseline) vs v3
(baseline + StyleResidual) fit on identical training windows.
Run: python eval_v3.py
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ingest import FootballDataLoader
from model.independent import (IndependentEngine, load_league_xg,
                               xi_quality, DB)
from model.depth import team_style_features, StyleResidual, STYLE_COLS
from processing import DataCleaner
from statsdb.features import build_team_map

LEAGUES = {39: ("Premier League", "E0"), 140: ("La Liga", "SP1"),
           78: ("Bundesliga", "D1")}

conn = sqlite3.connect(DB)

loader = FootballDataLoader(seasons=["2526"],
                            leagues=[c for _, c in LEAGUES.values()])
odds = DataCleaner.clean(loader.load_all())
inv = np.array([1/odds["B365H"], 1/odds["B365D"], 1/odds["B365A"]]).T
odds[["mkt_H", "mkt_D", "mkt_A"]] = inv / inv.sum(axis=1, keepdims=True)
odds["date"] = odds["Date"].dt.normalize()

tot_frames = []
for lid, (lname, code) in LEAGUES.items():
    o = pd.read_csv(f"data/raw/{code}_2526.csv", encoding="utf-8-sig",
                    on_bad_lines="skip")
    o.columns = [str(c).strip() for c in o.columns]
    o["Date"] = pd.to_datetime(o["Date"], dayfirst=True, errors="coerce")
    ov = o.get("PC>2.5"); un = o.get("PC<2.5")
    if ov is None or ov.isna().mean() > 0.5:
        ov, un = o.get("AvgC>2.5"), o.get("AvgC<2.5")
    o["over_odds"], o["under_odds"] = ov, un
    o["date"] = o["Date"].dt.normalize()
    tot_frames.append(o[["date", "HomeTeam", "AwayTeam",
                         "over_odds", "under_odds"]])
totals_odds = pd.concat(tot_frames).dropna()

fd_teams = {lg: sorted(set(odds[odds.League == lg].HomeTeam))
            for lg, _ in LEAGUES.values()}
tmap = build_team_map(conn, fd_teams)

variants = {"v2 (xG-DC + XI)": False, "v3 (+ style residuals)": True}
results = {k: [] for k in variants}
for lid, (lname, code) in LEAGUES.items():
    matches = load_league_xg(conn, lid)
    xi = xi_quality(conn, lid)
    xi_lookup = xi.set_index(["fixture_id", "team_id"])["xi_rating"]
    styles = team_style_features(conn, lid)
    sty_lookup = styles.set_index(["fixture_id", "team_id"])[STYLE_COLS]
    season = matches[matches["Date"] >= "2025-08-01"]
    print(f"{lname}: styles for {styles['fixture_id'].nunique()} fixtures")

    for month in sorted(season["Date"].dt.to_period("M").unique()):
        m_start = month.to_timestamp()
        train = matches[matches["Date"] < m_start]
        if len(train) < 500:
            continue
        eng = IndependentEngine().fit(
            train, xi[xi["fixture_id"].isin(train["fixture_id"])])

        # style residual fit on training window
        tr = train.dropna(subset=["FTHG", "FTAG"])
        lam_b, mu_b, hstyles, astyles = [], [], [], []
        for _, r in tr.iterrows():
            l, m_ = eng.dc.rates(r["HomeTeam"], r["AwayTeam"])
            lam_b.append(l); mu_b.append(m_)
            hstyles.append(sty_lookup.reindex(
                [(r["fixture_id"], r["home_id"])]).values[0])
            astyles.append(sty_lookup.reindex(
                [(r["fixture_id"], r["away_id"])]).values[0])
        resid = StyleResidual(l2=25.0, inter_only=True).fit(
            np.array(lam_b), np.array(mu_b),
            tr["FTHG"].values.astype(float), tr["FTAG"].values.astype(float),
            np.array(hstyles), np.array(astyles))

        block = season[season["Date"].dt.to_period("M") == month]
        for _, r in block.iterrows():
            xh = xi_lookup.get((r["fixture_id"], r["home_id"]), np.nan)
            xa = xi_lookup.get((r["fixture_id"], r["away_id"]), np.nan)
            hs = sty_lookup.reindex(
                [(r["fixture_id"], r["home_id"])]).values[0]
            as_ = sty_lookup.reindex(
                [(r["fixture_id"], r["away_id"])]).values[0]
            try:
                lam, mu = eng.rates(r["HomeTeam"], r["AwayTeam"],
                                    None if np.isnan(xh) else xh,
                                    None if np.isnan(xa) else xa)
            except KeyError:
                continue
            for name, use_style in variants.items():
                l2, m2 = (resid.adjust(lam, mu, hs, as_)
                          if use_style else (lam, mu))
                mtx = eng._matrix(l2, m2)
                size = mtx.shape[0]
                results[name].append({
                    "date": r["Date"].normalize(),
                    "home_fd": tmap.get(r["home_id"]),
                    "away_fd": tmap.get(r["away_id"]),
                    "p_home": float(np.tril(mtx, -1).sum()),
                    "p_draw": float(np.trace(mtx)),
                    "p_away": float(np.triu(mtx, 1).sum()),
                    "p_over": float(sum(mtx[i, j] for i in range(size)
                                        for j in range(size)
                                        if i + j >= 3)),
                    "hg": int(r["FTHG"]), "ag": int(r["FTAG"])})

for name, rows in results.items():
    pred = pd.DataFrame(rows).dropna(subset=["home_fd", "away_fd"])
    m1 = pred.merge(odds[["date", "HomeTeam", "AwayTeam",
                          "mkt_H", "mkt_D", "mkt_A"]],
                    left_on=["date", "home_fd", "away_fd"],
                    right_on=["date", "HomeTeam", "AwayTeam"], how="inner")
    y = np.select([m1.hg > m1.ag, m1.hg == m1.ag], [2, 1], 0)
    mp = m1[["p_away", "p_draw", "p_home"]].values
    kp = m1[["mkt_A", "mkt_D", "mkt_H"]].values
    m2 = pred.merge(totals_odds, left_on=["date", "home_fd", "away_fd"],
                    right_on=["date", "HomeTeam", "AwayTeam"], how="inner")
    y2 = ((m2.hg + m2.ag) > 2.5).astype(int).values
    io, iu = 1/m2["over_odds"], 1/m2["under_odds"]
    mk2 = (io / (io + iu)).values
    mo2 = m2["p_over"].values

    def b(p):
        return log_loss(y2, np.column_stack([1-p, p]), labels=[0, 1])

    print(f"\n=== {name} ===")
    print(f"  1X2  (n={len(m1)}): model {log_loss(y, mp, labels=[0,1,2]):.4f}"
          f" vs close {log_loss(y, kp, labels=[0,1,2]):.4f}"
          f"  gap {log_loss(y, mp, labels=[0,1,2]) - log_loss(y, kp, labels=[0,1,2]):+.4f}")
    print(f"  O/U2.5 (n={len(m2)}): model {b(mo2):.4f} vs close {b(mk2):.4f}"
          f"  gap {b(mo2) - b(mk2):+.4f}")
