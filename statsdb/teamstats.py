"""Per-fixture team statistics from API-Football (incl. expected goals).

The v2 information layer: match xG distills ~25 shots of quality signal
vs ~2.7 noisy goal events - the documented gap-closer for independent
models. One request per fixture, cached like everything else.
"""
import sqlite3

from .apifootball import ApiFootballClient
from .schema import init_db

DDL = """
CREATE TABLE IF NOT EXISTS team_match_stats (
    fixture_id INTEGER, team_id INTEGER,
    xg REAL, shots INTEGER, shots_on INTEGER,
    possession REAL, corners INTEGER, passes_pct REAL,
    PRIMARY KEY (fixture_id, team_id) ON CONFLICT REPLACE
);"""

FIELDS = {"expected_goals": "xg", "Total Shots": "shots",
          "Shots on Goal": "shots_on", "Ball Possession": "possession",
          "Corner Kicks": "corners", "Passes %": "passes_pct"}


def _num(v):
    if v is None:
        return None
    s = str(v).rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def sync_team_stats(league_id: int, season: int,
                    max_requests: int = 2000,
                    min_interval: float = 0.15) -> dict:
    api = ApiFootballClient(max_requests=max_requests,
                            min_interval=min_interval)
    conn = init_db()
    conn.executescript(DDL)
    fids = [r[0] for r in conn.execute(
        "SELECT fixture_id FROM fixtures WHERE league_id=? AND season=? "
        "AND status='FT'", (league_id, season))]
    done = {r[0] for r in conn.execute(
        "SELECT DISTINCT fixture_id FROM team_match_stats")}
    counts = {"fixtures": len(fids), "synced": 0, "pending": 0,
              "with_xg": 0, "requests": 0}

    for fid in fids:
        if fid in done:
            continue
        resp = api.get("fixtures/statistics", fixture=fid)
        if resp is None:
            counts["pending"] += 1
            continue
        for team in resp:
            row = {v: None for v in FIELDS.values()}
            for st in team.get("statistics", []):
                key = FIELDS.get(st.get("type"))
                if key:
                    row[key] = _num(st.get("value"))
            conn.execute(
                "INSERT OR REPLACE INTO team_match_stats VALUES "
                "(?,?,?,?,?,?,?,?)",
                (fid, team["team"]["id"], row["xg"], row["shots"],
                 row["shots_on"], row["possession"], row["corners"],
                 row["passes_pct"]))
            if row["xg"] is not None:
                counts["with_xg"] += 1
        counts["synced"] += 1
        if counts["synced"] % 200 == 0:
            conn.commit()
    conn.commit()
    counts["requests"] = api.spent
    conn.close()
    return counts
