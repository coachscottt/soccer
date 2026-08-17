"""StatsBomb open-data loader: true event-level play-by-play.

Licensed open data (github.com/statsbomb/open-data). We ingest match
metadata and every shot (with xG, coordinates, body part, pressure).
Competitions overlapping our odds data:
  La Liga 2020/21    (comp 11, season 90)   - 380 matches
  Bundesliga 2023/24 (comp 9,  season 281)  - 306 matches
"""
import json

import requests

from .schema import init_db

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

COMPETITIONS = {
    "laliga-2021": (11, 90, "La Liga", "2020/2021"),
    "bundesliga-2324": (9, 281, "Bundesliga", "2023/2024"),
}


def ingest_competition(key: str, verbose: bool = True) -> dict:
    comp_id, season_id, comp_name, season_name = COMPETITIONS[key]
    conn = init_db()

    matches = requests.get(
        f"{RAW}/matches/{comp_id}/{season_id}.json", timeout=30).json()
    done = {r[0] for r in conn.execute(
        "SELECT DISTINCT sb_match_id FROM sb_shots")}
    counts = {"matches": len(matches), "new": 0, "shots": 0, "skipped": 0}

    for i, m in enumerate(matches):
        mid = m["match_id"]
        conn.execute(
            "INSERT OR REPLACE INTO sb_matches VALUES (?,?,?,?,?,?,?,?)",
            (mid, comp_name, season_name, m["match_date"],
             m["home_team"]["home_team_name"],
             m["away_team"]["away_team_name"],
             m["home_score"], m["away_score"]))
        if mid in done:
            counts["skipped"] += 1
            continue

        try:
            events = requests.get(f"{RAW}/events/{mid}.json",
                                  timeout=60).json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if verbose:
                print(f"  [WARN] match {mid}: {e}")
            continue

        for e in events:
            if e.get("type", {}).get("name") != "Shot":
                continue
            shot = e.get("shot", {})
            loc = e.get("location") or [None, None]
            conn.execute(
                "INSERT INTO sb_shots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, e.get("minute"), e.get("second"), e.get("period"),
                 e.get("team", {}).get("name"),
                 e.get("player", {}).get("name"),
                 loc[0], loc[1],
                 shot.get("statsbomb_xg"),
                 shot.get("outcome", {}).get("name"),
                 shot.get("body_part", {}).get("name"),
                 e.get("play_pattern", {}).get("name"),
                 int(bool(e.get("under_pressure")))))
            counts["shots"] += 1
        counts["new"] += 1
        conn.commit()
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(matches)} matches "
                  f"({counts['shots']} shots so far)")

    conn.close()
    return counts
