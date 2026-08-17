"""Arena test: the independent engine vs SOFT closing lines.

Same walk-forward judgment as eval_independent.py, but against the five
smaller leagues (Norway, Sweden, Korea, Poland, Denmark) whose closes
are less sharp than the big-3. Benchmarks: devigged Pinnacle closing
(fallback: average closing). Paper bets are struck at REAL closing
prices (vig included) when the model's edge vs devig >= 5 points.

Run: python eval_arena.py
"""
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
import requests
from io import StringIO
from sklearn.metrics import log_loss

from model.independent import (IndependentEngine, load_league_xg as
                               load_league, xi_quality, DB)

ARENA = [
    # (label, stats.db league_id, csv code, eval_start)
    ("Norway",  103, "NOR", "2025-01-01"),
    ("Sweden",  113, "SWE", "2025-01-01"),
    ("Korea",   292, "KOR", "2025-01-01"),
    ("Poland",  106, "POL", "2025-08-01"),
    ("Denmark", 119, "DNK", "2025-08-01"),
]

CLUB_TOKENS = {"fc", "sc", "cf", "if", "is", "ff", "aif", "bk", "sk", "ifk",
               "ik", "fk", "afc", "ac", "cd", "as", "sv", "1995", "08", "1909"}
ALIASES = {  # af name (normalized) -> csv name, where matching fails
    "bodo/glimt": "Bodo/Glimt",
    "djurgardens": "Djurgarden",
    "gais goteborg": "GAIS",
    "jeonbuk motors": "Jeonbuk Hyundai Motors",
    "gwangju": "Gwangju FC",
    "pogon szczecin": "Pogon Szczecin",
    "gornik zabrze": "Gornik Zabrze",
    "fc kobenhavn": "FC Copenhagen",
    "nordsjaelland": "Nordsjaelland",
    "vejle boldklub": "Vejle",
    "agf aarhus": "Aarhus",
    "sonderjyske": "Sonderjyske",
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("-", " ").replace("/", "/").strip()
    toks = [t for t in s.split() if t not in CLUB_TOKENS]
    return " ".join(toks) or s


def map_teams(af_names, csv_names):
    out, unmatched = {}, []
    csv_norm = {norm(c): c for c in csv_names}
    for a in af_names:
        na = norm(a)
        if na in ALIASES and ALIASES[na] in csv_names:
            out[a] = ALIASES[na]; continue
        if na in csv_norm:
            out[a] = csv_norm[na]; continue
        cands = [c for nc, c in csv_norm.items()
                 if na in nc or nc in na
                 or set(na.split()) <= set(nc.split())
                 or set(nc.split()) <= set(na.split())]
        if len(set(cands)) == 1:
            out[a] = cands[0]
        else:
            unmatched.append((a, cands))
    return out, unmatched


def load_csv(code):
    r = requests.get(f"https://www.football-data.co.uk/new/{code}.csv",
                     timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), on_bad_lines="skip",
                     encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in df.columns:
        if c not in ("Country", "League", "Date", "Time", "Home", "Away", "Res"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date", "HG", "AG"])


conn = sqlite3.connect(DB)
all_rows = []
for label, lid, code, eval_start in ARENA:
    matches = load_league(conn, lid)
    xi = xi_quality(conn, lid)
    xi_lookup = xi.set_index(["fixture_id", "team_id"])["xi_rating"]
    csv = load_csv(code)

    af_names = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    csv_names = sorted(set(csv["Home"]) | set(csv["Away"]))
    tmap, unmatched = map_teams(af_names, csv_names)
    if unmatched:
        print(f"[{label}] unmatched: {unmatched}")

    season = matches[matches["Date"] >= eval_start]
    preds = []
    for month in sorted(season["Date"].dt.to_period("M").unique()):
        m_start = month.to_timestamp()
        train = matches[matches["Date"] < m_start]
        if len(train) < 300:
            continue
        eng = IndependentEngine().fit(
            train, xi[xi["fixture_id"].isin(train["fixture_id"])])
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
            preds.append({"date": r["Date"].normalize(),
                          "h": tmap.get(r["HomeTeam"]),
                          "a": tmap.get(r["AwayTeam"]),
                          "p_home": p["home"], "p_draw": p["draw"],
                          "p_away": p["away"]})
    pred = pd.DataFrame(preds).dropna(subset=["h", "a"])

    csv["date"] = csv["Date"].dt.normalize()
    oc = ["PSCH", "PSCD", "PSCA"]
    fallback = ["AvgCH", "AvgCD", "AvgCA"]
    for main_c, fb in zip(oc, fallback):
        csv[main_c] = csv[main_c].fillna(csv.get(fb))
    merged = pred.merge(
        csv[["date", "Home", "Away", "HG", "AG"] + oc],
        left_on=["date", "h", "a"], right_on=["date", "Home", "Away"],
        how="inner").dropna(subset=oc)
    merged["league"] = label
    all_rows.append(merged)
    print(f"[{label}] predictions {len(pred)}, joined with closing odds "
          f"{len(merged)}")

df = pd.concat(all_rows, ignore_index=True)
inv = np.array([1/df["PSCA"], 1/df["PSCD"], 1/df["PSCH"]]).T
mkt = inv / inv.sum(axis=1, keepdims=True)     # A, D, H devigged
model = df[["p_away", "p_draw", "p_home"]].values
y = np.select([df.HG > df.AG, df.HG == df.AG], [2, 1], 0)

print(f"\n=== ARENA overall (n={len(df)}) ===")
print(f"  independent model: log_loss={log_loss(y, model, labels=[0,1,2]):.4f}"
      f"  acc={(model.argmax(1)==y).mean():.4f}")
print(f"  soft closing:      log_loss={log_loss(y, mkt, labels=[0,1,2]):.4f}"
      f"  acc={(mkt.argmax(1)==y).mean():.4f}")
print("\nper league:")
for lg in df["league"].unique():
    m = df["league"] == lg
    print(f"  {lg:8s} n={m.sum():4d}  model {log_loss(y[m], model[m], labels=[0,1,2]):.4f}"
          f"  close {log_loss(y[m], mkt[m], labels=[0,1,2]):.4f}"
          f"  gap {log_loss(y[m], model[m], labels=[0,1,2]) - log_loss(y[m], mkt[m], labels=[0,1,2]):+.4f}")

print("\n=== divergence test (>=5 pts vs devig close) ===")
for i, name in [(2, "home"), (0, "away")]:
    for side, mask in [("HIGHER", model[:, i] - mkt[:, i] >= 0.05),
                       ("LOWER", model[:, i] - mkt[:, i] <= -0.05)]:
        if mask.sum() < 20:
            continue
        hit = (y[mask] == i).mean()
        ma, ka = model[mask, i].mean(), mkt[mask, i].mean()
        verdict = "MODEL closer" if abs(hit-ma) < abs(hit-ka) else "CLOSE closer"
        print(f"  {name:4s} model {side:6s} n={mask.sum():4d}  actual {hit:.3f} "
              f"| model {ma:.3f} close {ka:.3f} -> {verdict}")

print("\n=== paper bets at REAL closing prices (vig included) ===")
price_cols = {2: "PSCH", 1: "PSCD", 0: "PSCA"}
pnl, nb, wins = 0.0, 0, 0
for i, col in price_cols.items():
    mask = (model[:, i] - mkt[:, i]) >= 0.05
    prices = df.loc[mask, col].values
    won = (y[mask] == i)
    pnl += (won * (prices - 1) - (~won) * 1.0).sum()
    nb += int(mask.sum()); wins += int(won.sum())
print(f"  {nb} bets, {wins} won, P/L {pnl:+.1f}u ({pnl/max(nb,1)*100:+.1f}% ROI)")
