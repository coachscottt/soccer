"""Experiment: do lineup-strength / absence features add predictive
value beyond the existing 53-feature set (which includes the market)?

Joins stats.db lineup features onto the processed odds frame for the
overlap window (2023/24-2024/25, EPL + La Liga + Bundesliga), then
trains the ensemble with and without the new features on a
chronological 80/20 split of the overlap.
Run: python exp_lineup.py
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from model import prepare_model_data, build_ensemble
from statsdb.features import compute_lineup_features, build_team_map, DB

print("[1/4] lineup features from stats.db...")
lf = compute_lineup_features()
print(f"  {lf['fixture_id'].nunique()} fixtures with lineup features")

print("[2/4] mapping + join onto odds frame...")
fd = pd.read_parquet("data/processed_features.parquet")
fd_teams = {lg: sorted(set(g["HomeTeam"])) for lg, g in fd.groupby("League")}
conn = sqlite3.connect(DB)
tmap = build_team_map(conn, fd_teams)
conn.close()

lf["fd_team"] = lf["team_id"].map(tmap)
lf["is_home"] = (lf["team_id"] == lf["home_id"]).astype(int)
home = lf[lf.is_home == 1][["fixture_id", "date", "fd_team", "xi_rating",
                            "xi_rated", "missing_regulars"]]
away = lf[lf.is_home == 0][["fixture_id", "fd_team", "xi_rating",
                            "xi_rated", "missing_regulars"]]
j = home.merge(away, on="fixture_id", suffixes=("_h", "_a")).dropna(
    subset=["fd_team_h", "fd_team_a"])
j["match_date"] = j["date"].dt.normalize()

fd["match_date"] = fd["Date"].dt.normalize()
merged = fd.merge(
    j, left_on=["match_date", "HomeTeam", "AwayTeam"],
    right_on=["match_date", "fd_team_h", "fd_team_a"], how="inner")
# kickoff dates can straddle midnight UTC vs local: retry +/- 1 day for misses
for delta in (1, -1):
    missed = fd[~fd.index.isin(merged.index)]
    j2 = j.copy()
    j2["match_date"] = j2["match_date"] + pd.Timedelta(days=delta)
    m2 = missed.merge(j2, left_on=["match_date", "HomeTeam", "AwayTeam"],
                      right_on=["match_date", "fd_team_h", "fd_team_a"],
                      how="inner")
    merged = pd.concat([merged, m2])
merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
merged = merged.sort_values("Date").reset_index(drop=True)
print(f"  joined: {len(merged)} matches with odds + lineup features")

merged["xi_rating_diff"] = merged["xi_rating_h"] - merged["xi_rating_a"]
merged["missing_diff"] = (merged["missing_regulars_h"].fillna(0)
                          - merged["missing_regulars_a"].fillna(0))
LINEUP_COLS = ["xi_rating_h", "xi_rating_a", "xi_rating_diff",
               "missing_regulars_h", "missing_regulars_a", "missing_diff"]

print("[3/4] train/evaluate with vs without lineup features...")
X_base, y, base_cols = prepare_model_data(merged)
X_plus = pd.concat([X_base,
                    merged[LINEUP_COLS].fillna(merged[LINEUP_COLS].median())],
                   axis=1)
split = int(len(merged) * 0.8)
yt = y.iloc[split:].values

results = {}
for name, X in [("base (53 feats)", X_base), ("base + lineup", X_plus)]:
    ens, sc = build_ensemble(X, y, verbose=False)
    p = ens.predict_proba(sc.transform(X.iloc[split:]))
    results[name] = p
    print(f"  {name:18s} log_loss={log_loss(yt, p, labels=[0,1,2]):.4f}  "
          f"acc={accuracy_score(yt, p.argmax(axis=1)):.4f}")

mkt = merged[["norm_prob_A", "norm_prob_D", "norm_prob_H"]].values[split:]
mkt = mkt / mkt.sum(axis=1, keepdims=True)
print(f"  {'market':18s} log_loss={log_loss(yt, mkt, labels=[0,1,2]):.4f}  "
      f"acc={accuracy_score(yt, mkt.argmax(axis=1)):.4f}")

print("[4/4] lineup-feature importance (RF within ensemble refit)...")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                            min_samples_leaf=10, random_state=42)
rf.fit(X_plus.iloc[:split], y.iloc[:split])
imp = pd.Series(rf.feature_importances_, index=X_plus.columns)
print("  lineup features rank (of", len(X_plus.columns), "features):")
ranks = imp.rank(ascending=False)
for c in LINEUP_COLS:
    print(f"    {c:22s} importance={imp[c]:.4f}  rank={int(ranks[c])}")
print(f"  sanity: xi_rating_diff corr with result: "
      f"{merged['xi_rating_diff'].corr(merged['Result']):.3f}")
