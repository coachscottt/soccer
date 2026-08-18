"""API-Football (v3.football.api-sports.io) client + ingestion.

Free tier: 100 requests/day, 10/minute. Every raw response is cached to
disk so a request is never spent twice; re-running a partially-synced
season resumes where the quota ran out.

League ids: EPL=39, La Liga=140, Bundesliga=78, Serie A=135,
Ligue 1=61, MLS=253, Champions League=2.
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from .schema import init_db

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

HOST = "https://v3.football.api-sports.io"
CACHE = Path(__file__).resolve().parents[1] / "data" / "apifootball_cache"

LEAGUE_IDS = {"EPL": 39, "LaLiga": 140, "Bundesliga": 78,
              "SerieA": 135, "Ligue1": 61, "MLS": 253, "UCL": 2,
              "Championship": 40, "Eredivisie": 88, "PrimeiraLiga": 94,
              "JupilerPro": 144, "SuperLig": 203, "ScotPrem": 179,
              "BrazilSerieA": 71, "LigaMX": 262,
              "Ekstraklasa": 106, "CzechLiga": 345, "GreeceSL": 197,
              "DKSuperliga": 119, "Eliteserien": 103, "CyprusD1": 318,
              "SwissSL": 207,
              "HungaryNBI": 271, "Allsvenskan": 113, "RomaniaLigaI": 283,
              "AustriaBL": 218, "CroatiaHNL": 210,
              "KLeague1": 292, "J1League": 98,
              "SerbiaSL": 286, "ChinaSL": 169,
              "ArgentinaPrimera": 128}


class ApiFootballClient:

    def __init__(self, max_requests: int = 95, min_interval: float = 6.5):
        self.key = os.environ.get("API_FOOTBALL_KEY", "")
        self.max_requests = max_requests      # per-run budget
        self.min_interval = min_interval      # free tier: 10 req/min
        self.spent = 0
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers["x-apisports-key"] = self.key
        CACHE.mkdir(parents=True, exist_ok=True)

    def get(self, endpoint: str, force: bool = False, **params) -> dict | None:
        """Cached GET. Returns parsed json 'response' or None when the
        run budget is exhausted. force=True bypasses the cache (used to
        refresh in-season fixture lists)."""
        stamp = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        cache_file = CACHE / f"{endpoint.replace('/', '_')}__{stamp}.json"
        if cache_file.exists() and not force:
            return json.loads(cache_file.read_text(encoding="utf-8"))

        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY not set in football/.env")
        if self.spent >= self.max_requests:
            return None

        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.session.get(f"{HOST}/{endpoint}", params=params, timeout=30)
        self._last = time.time()
        self.spent += 1
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"api-football error: {body['errors']}")
        data = body.get("response", [])
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data


def sync_league_season(league_id: int, season: int,
                       max_requests: int = 95,
                       min_interval: float = 6.5,
                       refresh_fixtures: bool = False) -> dict:
    """Sync fixtures + events + lineups + player stats for one season.
    Resumable: cached fixtures are skipped; stops cleanly on budget.
    Free tier: defaults are right. Paid tiers: raise max_requests and
    drop min_interval (Ultra allows ~450 req/min -> 0.15-0.2s)."""
    api = ApiFootballClient(max_requests=max_requests,
                            min_interval=min_interval)
    conn = init_db()
    counts = {"fixtures": 0, "events": 0, "lineups": 0, "player_stats": 0,
              "requests_spent": 0, "fixtures_pending": 0}

    fixtures = api.get("fixtures", force=refresh_fixtures,
                       league=league_id, season=season)
    if fixtures is None:
        return counts

    for fx in fixtures:
        f = fx["fixture"]
        teams, goals = fx["teams"], fx["goals"]
        conn.execute(
            "INSERT OR REPLACE INTO fixtures VALUES (?,?,?,?,?,?,?,?,?)",
            (f["id"], league_id, season, f["date"],
             teams["home"]["id"], teams["away"]["id"],
             goals["home"], goals["away"], f["status"]["short"]))
        for side in ("home", "away"):
            t = teams[side]
            conn.execute("INSERT OR REPLACE INTO teams "
                         "(team_id, name) VALUES (?,?)", (t["id"], t["name"]))
        counts["fixtures"] += 1
    conn.commit()

    finished = [fx for fx in fixtures
                if fx["fixture"]["status"]["short"] == "FT"]
    for fx in finished:
        fid = fx["fixture"]["id"]
        done = conn.execute(
            "SELECT COUNT(*) FROM events WHERE fixture_id=?", (fid,)
        ).fetchone()[0]
        stats_done = conn.execute(
            "SELECT COUNT(*) FROM player_match_stats WHERE fixture_id=?",
            (fid,)).fetchone()[0]
        if done and stats_done:
            continue

        events = api.get("fixtures/events", fixture=fid)
        players = api.get("fixtures/players", fixture=fid)
        lineups = api.get("fixtures/lineups", fixture=fid)
        if events is None or players is None or lineups is None:
            counts["fixtures_pending"] += 1
            continue

        conn.execute("DELETE FROM events WHERE fixture_id=?", (fid,))
        for e in events:
            conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                (fid, e["time"]["elapsed"], e["time"].get("extra"),
                 e["team"]["id"], (e.get("player") or {}).get("id"),
                 (e.get("assist") or {}).get("id"),
                 e.get("type"), e.get("detail"), e.get("comments")))
            counts["events"] += 1

        conn.execute("DELETE FROM lineups WHERE fixture_id=?", (fid,))
        for team in lineups:
            tid = team["team"]["id"]
            for p in team.get("startXI", []):
                pl = p["player"]
                conn.execute("INSERT INTO lineups VALUES (?,?,?,?,?,1)",
                             (fid, tid, pl["id"], pl.get("pos"),
                              pl.get("grid")))
                counts["lineups"] += 1
            for p in team.get("substitutes", []):
                pl = p["player"]
                conn.execute("INSERT INTO lineups VALUES (?,?,?,?,?,0)",
                             (fid, tid, pl["id"], pl.get("pos"),
                              pl.get("grid")))
                counts["lineups"] += 1

        for team in players:
            tid = team["team"]["id"]
            for p in team.get("players", []):
                pid = p["player"]["id"]
                conn.execute("INSERT OR REPLACE INTO players "
                             "(player_id, name) VALUES (?,?)",
                             (pid, p["player"]["name"]))
                s = p["statistics"][0]
                g, sh, pa, ta, du, dr, fo, ca = (
                    s.get("goals") or {}, s.get("shots") or {},
                    s.get("passes") or {}, s.get("tackles") or {},
                    s.get("duels") or {}, s.get("dribbles") or {},
                    s.get("fouls") or {}, s.get("cards") or {})
                gm = s.get("games") or {}
                conn.execute(
                    "INSERT OR REPLACE INTO player_match_stats VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (fid, tid, pid, gm.get("minutes"),
                     float(gm["rating"]) if gm.get("rating") else None,
                     g.get("total") or 0, g.get("assists") or 0,
                     sh.get("total"), sh.get("on"),
                     pa.get("total"), pa.get("key"),
                     float(str(pa.get("accuracy") or "0").rstrip("%") or 0),
                     ta.get("total"), ta.get("interceptions"),
                     du.get("won"), dr.get("success"),
                     fo.get("drawn"), fo.get("committed"),
                     ca.get("yellow") or 0, ca.get("red") or 0))
                counts["player_stats"] += 1
        conn.commit()

    counts["requests_spent"] = api.spent
    conn.close()
    return counts
