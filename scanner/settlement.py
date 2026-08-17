"""Settle mode: results ingestion, bet grading, CLV, ledger reports.

Grades every open spot against final scores (Asian totals payoff-exact,
incl. pushes and half-wins), captures the closing price (last pre-kickoff
snapshot at the same book), and prints the per-market distinct-bet
ledger, the two-sided totals experiment, and fairs-vs-results Brier.
"""
import os
from datetime import datetime, timezone

import numpy as np
import requests

from .database import get_conn
from .asian_totals import totals_result_pnl
from .asian_handicap import hcp_result_pnl

BET_COLS = ("event_id, league, kickoff_utc, home, away, market, "
            "selection, line, book, price, ev, won, pnl, clv")


def collapse_bets(rows):
    """One bet per (match, market, side): the best-EV line at its best
    price. A ladder of alternate lines (spread +0.75..+2.0, totals
    2.25..3.5) is ONE opinion and must not count as many bets (owner
    call 2026-07-26). Opposite sides (over vs under, home vs draw)
    remain separate bets. rows: dicts with BET_COLS keys; returns bet
    dicts plus books (quotes on the chosen line) and avg clv."""
    lines = {}
    for r in rows:
        k = (r["event_id"], r["market"], r["selection"], r["line"])
        e = lines.get(k)
        if e is None:
            e = lines[k] = {**r, "books": 0, "_clvs": []}
        e["books"] += 1
        if r.get("clv") is not None:
            e["_clvs"].append(r["clv"])
        if r["price"] > e["price"]:
            b, c = e["books"], e["_clvs"]
            e.update(r)
            e["books"], e["_clvs"] = b, c
        e["ev"] = max(e["ev"], r["ev"])
    best = {}
    for e in lines.values():
        k = (e["event_id"], e["market"], e["selection"])
        if k not in best or e["ev"] > best[k]["ev"]:
            best[k] = e
    out = []
    for e in best.values():
        clvs = e.pop("_clvs")
        e["clv"] = sum(clvs) / len(clvs) if clvs else None
        out.append(e)
    return out


def settle(args, cfg):
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_res = 0

    if cfg.get("af_league"):
        # results come from API-Football fixtures (refresh, then read)
        from statsdb.apifootball import sync_league_season
        sync_league_season(cfg["af_league"], cfg["af_season"],
                           max_requests=500, min_interval=0.2,
                           refresh_fixtures=True)
        names = dict(conn.execute(
            "SELECT team_id, name FROM teams").fetchall())
        for fid, h, a, hg, ag, ko in conn.execute(
                """SELECT fixture_id, home_id, away_id, home_goals,
                          away_goals, kickoff_utc
                   FROM fixtures WHERE league_id=? AND season=?
                   AND status='FT'""",
                (cfg["af_league"], cfg["af_season"])):
            conn.execute("INSERT OR REPLACE INTO match_results VALUES "
                         "(?,?,?,?,?,?,?,?)",
                         (f"af_{fid}", cfg["label"], ko,
                          names.get(h), names.get(a), hg, ag, now))
            n_res += 1
    else:
        key = os.environ.get("THE_ODDS_API_KEY")
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{cfg['sport']}/scores",
            params={"apiKey": key, "daysFrom": 3}, timeout=30)
        r.raise_for_status()
        for ev in r.json():
            if not ev.get("completed") or not ev.get("scores"):
                continue
            sc = {s["name"]: int(s["score"]) for s in ev["scores"]}
            hg, ag = sc.get(ev["home_team"]), sc.get(ev["away_team"])
            if hg is None or ag is None:
                continue
            conn.execute("INSERT OR REPLACE INTO match_results VALUES "
                         "(?,?,?,?,?,?,?,?)",
                         (ev["id"], cfg["label"], ev["commence_time"],
                          ev["home_team"], ev["away_team"], hg, ag, now))
            n_res += 1

    graded = clvd = 0
    spots = conn.execute(
        """SELECT s.rowid, s.event_id, s.market, s.selection, s.line,
                  s.book, s.price, s.kickoff_utc, r.home_goals, r.away_goals
           FROM value_spots s JOIN match_results r USING(event_id)
           WHERE s.won IS NULL""").fetchall()
    for (rid, eid, mk, sel, line, book, price, ko, hg, ag) in spots:
        if mk == "h2h":
            won = int(sel == ("home" if hg > ag else
                              "away" if ag > hg else "draw"))
        elif mk == "totals":
            pnl = round(totals_result_pnl(hg + ag, sel, line, price), 3)
            won = int(pnl > 0)
        elif mk == "spread":
            d = (hg - ag) if sel == "home" else (ag - hg)
            pnl = round(hcp_result_pnl(d, line, price), 3)
            won = int(pnl > 0)
        else:
            won = int((hg > 0 and ag > 0) == (sel == "yes"))
        if mk not in ("totals", "spread"):
            pnl = round(price - 1, 3) if won else -1.0
        close = conn.execute(
            """SELECT price FROM odds_snapshots
               WHERE event_id=? AND market=? AND selection=? AND line=?
                 AND book=? AND logged_at < ?
               ORDER BY logged_at DESC LIMIT 1""",
            (eid, mk, sel, line, book, ko)).fetchone()
        clv = round(price / close[0] - 1, 4) if close else None
        if clv is not None:
            clvd += 1
        conn.execute("UPDATE value_spots SET won=?, pnl=?, closing_price=?, "
                     "clv=? WHERE rowid=?",
                     (won, pnl, close[0] if close else None, clv, rid))
        graded += 1
    conn.commit()

    keys = BET_COLS.split(", ")
    bets = collapse_bets([dict(zip(keys, r)) for r in conn.execute(
        f"""SELECT {BET_COLS} FROM value_spots
            WHERE won IS NOT NULL AND league = ?""", (cfg["label"],))])
    print(f"results ingested: {n_res} | spots graded this run: {graded} "
          f"(with CLV: {clvd})")
    if bets:
        from collections import defaultdict
        agg = defaultdict(lambda: [0, 0, 0.0, 0.0, []])
        for b in bets:
            a = agg[b["market"]]
            a[0] += 1
            a[1] += b["won"]
            a[2] += b["pnl"]
            a[3] += b["books"]
            if b["clv"] is not None:
                a[4].append(b["clv"])
        print(f"\n{cfg['label']} - bets (best play per match/market/side, "
              f"1u at best price):")
        print(f"{'market':8s} {'bets':>5s} {'quotes':>7s} {'hit%':>6s} "
              f"{'P/L(u)':>8s} {'avgCLV':>8s}")
        for mk in sorted(agg):
            n, w, pnl, nq, cl = agg[mk]
            ac = sum(cl) / len(cl) * 100 if cl else 0.0
            print(f"{mk:8s} {n:5d} {int(nq):7d} {w/n*100:5.1f}% "
                  f"{pnl:+8.2f} {ac:+7.2f}%")
        n_q, n_l = conn.execute(
            """SELECT COUNT(*),
                      (SELECT COUNT(*) FROM
                       (SELECT 1 FROM value_spots
                        WHERE won IS NOT NULL AND league = ?
                        GROUP BY event_id, market, selection, line))
               FROM value_spots WHERE won IS NOT NULL AND league = ?""",
            (cfg["label"], cfg["label"])).fetchone()
        print(f"research log: {n_q} graded quotes / {n_l} line-level edges "
              f"in value_spots (official ledger = best play only)")

    # two-sided totals: over AND under flagged on the same match.
    # Cross-book dispersion around the sharp anchor; tracking each side
    # separately answers "where is the true signal" (owner watch-list).
    # Best play per side (ladder collapsed) since 2026-07-26.
    tb = collapse_bets([dict(zip(keys, r)) for r in conn.execute(
        f"""SELECT {BET_COLS} FROM value_spots
            WHERE won IS NOT NULL AND market='totals'""")])
    per = {}
    for b in tb:
        per.setdefault(b["event_id"], {})[b["selection"]] = b
    pairs = [d for d in per.values() if "over" in d and "under" in d]
    if pairs:
        n = len(pairs)
        o_pnl = sum(d["over"]["pnl"] for d in pairs)
        u_pnl = sum(d["under"]["pnl"] for d in pairs)
        o_clv = np.mean([d["over"]["clv"] for d in pairs
                         if d["over"]["clv"] is not None] or [0])
        u_clv = np.mean([d["under"]["clv"] for d in pairs
                         if d["under"]["clv"] is not None] or [0])
        print(f"\ntwo-sided totals matches (all leagues, cumulative): {n}")
        print(f"  over legs:  P/L {o_pnl:+.2f}u  avgCLV {o_clv*100:+.2f}%")
        print(f"  under legs: P/L {u_pnl:+.2f}u  avgCLV {u_clv*100:+.2f}%")
        print(f"  both-legs middle: {o_pnl+u_pnl:+.2f}u over {n} matches")

    # fairs calibration vs results
    fb = conn.execute(
        """SELECT f.p_home, f.p_draw, f.p_away, r.home_goals, r.away_goals
           FROM match_fairs f JOIN match_results r USING(event_id)
           WHERE f.logged_at = (SELECT MAX(logged_at) FROM match_fairs f2
                                WHERE f2.event_id = f.event_id)""").fetchall()
    if fb:
        briers = []
        for ph, pd_, pa, hg, ag in fb:
            y = [hg > ag, hg == ag, hg < ag]
            briers.append(sum((p - int(o)) ** 2
                              for p, o in zip([ph, pd_, pa], y)))
        print(f"\nfairs vs results: {len(fb)} matches, "
              f"mean 3-way Brier {np.mean(briers):.4f} "
              f"(sharp-market typical ~0.58-0.62)")
    conn.close()
