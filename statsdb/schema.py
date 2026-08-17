"""SQLite schema for the team/player stats database.

One database, two feeds:
  - API-Football (v3.football.api-sports.io): teams, players, fixtures,
    lineups, per-player match stats, event timeline (goals/cards/subs)
  - StatsBomb open data: true event-level play-by-play (shots with xG
    and coordinates) for its covered competitions
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "stats.db"

DDL = """
CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY,        -- api-football id
    name        TEXT NOT NULL,
    country     TEXT,
    founded     INTEGER,
    venue       TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,        -- api-football id
    name        TEXT NOT NULL,
    birth_date  TEXT,
    nationality TEXT,
    height_cm   INTEGER,
    position    TEXT
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id  INTEGER PRIMARY KEY,        -- api-football id
    league_id   INTEGER,
    season      INTEGER,
    kickoff_utc TEXT,
    home_id     INTEGER REFERENCES teams(team_id),
    away_id     INTEGER REFERENCES teams(team_id),
    home_goals  INTEGER,
    away_goals  INTEGER,
    status      TEXT
);

CREATE TABLE IF NOT EXISTS events (
    fixture_id  INTEGER REFERENCES fixtures(fixture_id),
    minute      INTEGER,
    extra_min   INTEGER,
    team_id     INTEGER,
    player_id   INTEGER,
    assist_id   INTEGER,
    type        TEXT,       -- Goal / Card / subst / Var
    detail      TEXT,       -- Normal Goal / Yellow Card / ...
    comments    TEXT
);

CREATE TABLE IF NOT EXISTS lineups (
    fixture_id  INTEGER REFERENCES fixtures(fixture_id),
    team_id     INTEGER,
    player_id   INTEGER,
    position    TEXT,
    grid        TEXT,
    is_starter  INTEGER
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    fixture_id  INTEGER REFERENCES fixtures(fixture_id),
    team_id     INTEGER,
    player_id   INTEGER,
    minutes     INTEGER,
    rating      REAL,
    goals       INTEGER, assists INTEGER,
    shots_total INTEGER, shots_on INTEGER,
    passes      INTEGER, key_passes INTEGER, pass_pct REAL,
    tackles     INTEGER, interceptions INTEGER, duels_won INTEGER,
    dribbles_won INTEGER, fouls_drawn INTEGER, fouls_committed INTEGER,
    yellow      INTEGER, red INTEGER,
    PRIMARY KEY (fixture_id, player_id)
);

-- StatsBomb play-by-play (shots carry xG; all events carry coordinates)
CREATE TABLE IF NOT EXISTS sb_matches (
    sb_match_id INTEGER PRIMARY KEY,
    competition TEXT, season TEXT,
    match_date  TEXT,
    home_team   TEXT, away_team TEXT,
    home_score  INTEGER, away_score INTEGER
);

CREATE TABLE IF NOT EXISTS sb_shots (
    sb_match_id INTEGER REFERENCES sb_matches(sb_match_id),
    minute      INTEGER, second INTEGER, period INTEGER,
    team        TEXT, player TEXT,
    x           REAL, y REAL,
    xg          REAL,
    outcome     TEXT,       -- Goal / Saved / Off T / ...
    body_part   TEXT,
    play_pattern TEXT,
    under_pressure INTEGER
);

CREATE INDEX IF NOT EXISTS ix_events_fixture ON events(fixture_id);
CREATE INDEX IF NOT EXISTS ix_pms_player ON player_match_stats(player_id);
CREATE INDEX IF NOT EXISTS ix_sb_shots_match ON sb_shots(sb_match_id);
CREATE INDEX IF NOT EXISTS ix_fixtures_league ON fixtures(league_id, season);
"""


def get_conn(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = get_conn(db_path)
    conn.executescript(DDL)
    conn.commit()
    return conn
