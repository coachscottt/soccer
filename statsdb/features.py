"""Lineup-strength and absence features from stats.db.

All features are leak-safe pre-match quantities:
  xi_rating          - mean of the starting XI's PRIOR per-match ratings
                       (each player's expanding mean over earlier apps)
  xi_rated           - how many starters had >=3 prior rated apps
  missing_regulars   - how many of the team's top-5 players (by minutes
                       over the last 10 team fixtures) are absent from
                       the entire matchday squad
Lineups publish ~1h before kickoff, so these are legitimate pre-match
inputs (the closing line also knows them; opening lines do not).
"""
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path(__file__).resolve().parents[1] / "data" / "stats.db"

LEAGUE_NAMES = {39: "Premier League", 140: "La Liga", 78: "Bundesliga"}

# api-football name -> football-data name, where auto-matching fails
ALIASES = {
    "West Ham": "West Ham",
    "Manchester United": "Man United",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Athletic Club": "Ath Bilbao",
    "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "FC Augsburg": "Augsburg",
    "FC St. Pauli": "St Pauli",
    "Espanyol": "Espanol",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    for kill in [" fc", " cf", " sc", " ac", " club", "1. ", " 04", " 05", " 09"]:
        s = s.replace(kill, " ")
    return " ".join(s.split())


def build_team_map(conn, fd_teams_by_league: dict) -> dict:
    """api-football team_id -> football-data team name (or None)."""
    af = pd.read_sql(
        """SELECT DISTINCT t.team_id, t.name, f.league_id
           FROM teams t JOIN fixtures f ON t.team_id IN (f.home_id, f.away_id)
           WHERE f.league_id IN (39, 140, 78)""", conn)
    out = {}
    for _, r in af.iterrows():
        fd_teams = fd_teams_by_league.get(LEAGUE_NAMES[r.league_id], [])
        if r["name"] in ALIASES:
            out[r.team_id] = ALIASES[r["name"]] if ALIASES[r["name"]] in fd_teams else None
            continue
        cands = [t for t in fd_teams
                 if _norm(t) == _norm(r["name"])
                 or _norm(t) in _norm(r["name"]) or _norm(r["name"]) in _norm(t)]
        out[r.team_id] = cands[0] if len(cands) == 1 else None
    return out


def compute_lineup_features() -> pd.DataFrame:
    """One row per (fixture, side) with xi_rating / xi_rated /
    missing_regulars, plus fixture metadata for joining."""
    conn = sqlite3.connect(DB)
    fx = pd.read_sql(
        """SELECT fixture_id, league_id, kickoff_utc, home_id, away_id
           FROM fixtures WHERE league_id IN (39,140,78) AND status='FT'""",
        conn)
    fx["date"] = pd.to_datetime(fx["kickoff_utc"], utc=True,
                                format="ISO8601").dt.tz_convert(None)
    pms = pd.read_sql(
        """SELECT s.fixture_id, s.team_id, s.player_id, s.minutes, s.rating
           FROM player_match_stats s
           JOIN fixtures f USING(fixture_id)
           WHERE f.league_id IN (39,140,78) AND f.status='FT'""", conn)
    lineups = pd.read_sql(
        """SELECT l.fixture_id, l.team_id, l.player_id, l.is_starter
           FROM lineups l JOIN fixtures f USING(fixture_id)
           WHERE f.league_id IN (39,140,78) AND f.status='FT'""", conn)

    pms = pms.merge(fx[["fixture_id", "date"]], on="fixture_id")
    pms = pms.sort_values("date")

    # expanding prior mean rating per player (exclude current match)
    pms["prior_n"] = pms.groupby("player_id").cumcount()
    pms["prior_rating"] = (pms.groupby("player_id")["rating"]
                           .transform(lambda s: s.shift(1).expanding().mean()))
    pms.loc[pms["prior_n"] < 3, "prior_rating"] = np.nan
    prior = pms.set_index(["fixture_id", "player_id"])["prior_rating"]

    starters = lineups[lineups["is_starter"] == 1]
    xi = starters.join(prior, on=["fixture_id", "player_id"])
    xi_agg = (xi.groupby(["fixture_id", "team_id"])["prior_rating"]
              .agg(xi_rating="mean", xi_rated="count").reset_index())

    # missing regulars: top-5 by minutes over the team's last 10 fixtures
    minutes = pms.groupby(["fixture_id", "team_id", "player_id"])["minutes"] \
                 .sum().reset_index()
    squads = lineups.groupby(["fixture_id", "team_id"])["player_id"] \
                    .apply(set).to_dict()
    team_fixtures = {}
    for _, r in fx.iterrows():
        for tid in (r.home_id, r.away_id):
            team_fixtures.setdefault(tid, []).append((r.date, r.fixture_id))
    minutes_by_fx = {(r.fixture_id, r.team_id): (r.player_id, r.minutes)
                     for r in minutes.itertuples()}
    min_lookup = minutes.groupby(["fixture_id", "team_id"]) \
                        .apply(lambda g: dict(zip(g.player_id, g.minutes)),
                               include_groups=False).to_dict()

    rows = []
    for tid, fixlist in team_fixtures.items():
        fixlist.sort()
        history = []                     # list of {player: minutes}
        for date, fid in fixlist:
            if len(history) >= 3:        # need some history
                agg = {}
                for h in history[-10:]:
                    for p, m in h.items():
                        agg[p] = agg.get(p, 0) + (m or 0)
                regulars = sorted(agg, key=agg.get, reverse=True)[:5]
                squad = squads.get((fid, tid), set())
                missing = sum(1 for p in regulars if p not in squad)
            else:
                missing = np.nan
            rows.append({"fixture_id": fid, "team_id": tid,
                         "missing_regulars": missing})
            history.append(min_lookup.get((fid, tid), {}))
    miss = pd.DataFrame(rows)

    out = xi_agg.merge(miss, on=["fixture_id", "team_id"], how="left")
    out = out.merge(fx[["fixture_id", "league_id", "date",
                        "home_id", "away_id"]], on="fixture_id")
    conn.close()
    return out
