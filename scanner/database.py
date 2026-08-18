"""Scanner storage: schema, migrations, and the prediction_audit view.

prediction_audit is a VIEW (not a copy of the data): one row per match
joining latest anchored fairs, latest independent v2 fairs, the result,
and the Pinnacle closing 1X2. Bet-level fields (edge, CLV, P/L, market)
stay in value_spots and join to it on event_id, e.g.:

    -- when v2 disagrees with the anchor by >8%, how often is it right?
    SELECT AVG((ind_p_home > p_home + 0.08) = (home_goals > away_goals))
    FROM prediction_audit
    WHERE home_goals IS NOT NULL AND ind_p_home > p_home + 0.08;

    -- do +CLV bets only outperform when the calibration fit was tight?
    SELECT a.fit_error < 0.002 AS tight, AVG(s.pnl), AVG(s.clv)
    FROM value_spots s JOIN prediction_audit a USING(event_id)
    WHERE s.won IS NOT NULL GROUP BY tight;
"""
import sqlite3
from pathlib import Path

from statsdb.schema import DB_PATH

DDL = """
CREATE TABLE IF NOT EXISTS match_fairs (
    logged_at TEXT, league TEXT, event_id TEXT, kickoff_utc TEXT,
    home TEXT, away TEXT, anchor TEXT,
    p_home REAL, p_draw REAL, p_away REAL,
    xg_home REAL, xg_away REAL, x_total REAL, rho REAL,
    tot_line REAL, p_over_line REAL, btts_yes REAL, fit_error REAL,
    PRIMARY KEY (event_id, logged_at) ON CONFLICT REPLACE
);
CREATE TABLE IF NOT EXISTS odds_snapshots (
    logged_at TEXT, event_id TEXT, market TEXT, selection TEXT,
    line REAL, book TEXT, price REAL,
    PRIMARY KEY (event_id, market, selection, line, book, logged_at)
    ON CONFLICT REPLACE
);
CREATE TABLE IF NOT EXISTS match_results (
    event_id TEXT PRIMARY KEY, league TEXT, kickoff_utc TEXT,
    home TEXT, away TEXT, home_goals INTEGER, away_goals INTEGER,
    settled_at TEXT
);
CREATE TABLE IF NOT EXISTS indep_fairs (
    logged_at TEXT, league TEXT, event_id TEXT, kickoff_utc TEXT,
    home TEXT, away TEXT,
    p_home REAL, p_draw REAL, p_away REAL,
    xg_home REAL, xg_away REAL,
    tot_line REAL, p_over_line REAL, btts_yes REAL,
    PRIMARY KEY (event_id, logged_at) ON CONFLICT REPLACE
);
CREATE TABLE IF NOT EXISTS value_spots (
    logged_at TEXT, league TEXT, event_id TEXT, kickoff_utc TEXT,
    home TEXT, away TEXT, market TEXT, selection TEXT, line REAL,
    book TEXT, price REAL, fair_prob REAL, fair_price REAL, ev REAL,
    closing_price REAL, clv REAL, won INTEGER, pnl REAL,
    void INTEGER DEFAULT 0,
    PRIMARY KEY (event_id, market, selection, line, book)
    ON CONFLICT REPLACE
);"""

# recreated on every connect so definition changes deploy automatically.
# close_* is NULL where no Pinnacle feed exists (manual-anchor leagues).
AUDIT_VIEW = """
DROP VIEW IF EXISTS prediction_audit;
CREATE VIEW prediction_audit AS
SELECT f.event_id, f.league, f.kickoff_utc, f.home, f.away, f.anchor,
       f.p_home, f.p_draw, f.p_away, f.xg_home, f.xg_away,
       f.tot_line, f.p_over_line, f.btts_yes, f.fit_error,
       i.p_home  AS ind_p_home, i.p_draw AS ind_p_draw,
       i.p_away  AS ind_p_away,
       i.xg_home AS ind_xg_home, i.xg_away AS ind_xg_away,
       r.home_goals, r.away_goals,
       (SELECT price FROM odds_snapshots o
        WHERE o.event_id = f.event_id AND o.market = 'h2h'
          AND o.selection = 'home' AND o.book = 'pinnacle'
          AND o.logged_at < f.kickoff_utc
        ORDER BY o.logged_at DESC LIMIT 1) AS close_home,
       (SELECT price FROM odds_snapshots o
        WHERE o.event_id = f.event_id AND o.market = 'h2h'
          AND o.selection = 'draw' AND o.book = 'pinnacle'
          AND o.logged_at < f.kickoff_utc
        ORDER BY o.logged_at DESC LIMIT 1) AS close_draw,
       (SELECT price FROM odds_snapshots o
        WHERE o.event_id = f.event_id AND o.market = 'h2h'
          AND o.selection = 'away' AND o.book = 'pinnacle'
          AND o.logged_at < f.kickoff_utc
        ORDER BY o.logged_at DESC LIMIT 1) AS close_away
FROM match_fairs f
LEFT JOIN indep_fairs i ON i.event_id = f.event_id
  AND i.logged_at = (SELECT MAX(logged_at) FROM indep_fairs
                     WHERE event_id = f.event_id)
LEFT JOIN match_results r ON r.event_id = f.event_id
WHERE f.logged_at = (SELECT MAX(logged_at) FROM match_fairs
                     WHERE event_id = f.event_id);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(match_fairs)").fetchall()]
    if "fit_error" not in cols:   # pre-2026-07-24 databases
        conn.execute("ALTER TABLE match_fairs ADD COLUMN fit_error REAL")
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(value_spots)").fetchall()]
    if "void" not in cols:        # pre-2026-08-18 databases
        conn.execute("ALTER TABLE value_spots "
                     "ADD COLUMN void INTEGER DEFAULT 0")
    conn.executescript(AUDIT_VIEW)


def get_conn() -> sqlite3.Connection:
    db = Path(DB_PATH)
    if not db.exists() or db.stat().st_size < 50_000_000:
        raise SystemExit(
            "stats.db missing or suspiciously small at " + str(DB_PATH)
            + " -> copy the real warehouse (~265MB) here before running; "
            "a fresh empty DB would silently lose the ledger + priors.")
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    return conn
