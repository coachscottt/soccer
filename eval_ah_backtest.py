"""Retrospective Asian handicap review (2026-07-26).

Question: had the AH market been in the scanner from the start, what
would it have flagged and how would those bets have graded?

Leak-free replay: for every settled Odds-API match we re-price AH from
the FAIRS WE LOGGED PRE-MATCH (lam/mu/rho in match_fairs, first row),
pull the bookmaker AH quotes from The Odds API historical archive at
that same logging timestamp, flag exactly as the scanner would
(EV >= 3%, price <= 6.0), then grade against the real final score and
measure CLV vs the archived close (kickoff - 5 min, same book/line).
"""
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

from mls_report import LEAGUES
from scanner.calibration import dc_matrix
from scanner.asian_handicap import margin_dist, hcp_ev, hcp_result_pnl
from statsdb.schema import DB_PATH

MIN_EV, MAX_ODDS = 0.03, 6.0

for line in (Path(__file__).resolve().parents[1] / ".env").read_text().splitlines():
    if line.startswith("THE_ODDS_API_KEY"):
        os.environ.setdefault("THE_ODDS_API_KEY", line.split("=", 1)[1].strip())
KEY = os.environ["THE_ODDS_API_KEY"]

SPORT = {cfg["label"]: cfg["sport"] for cfg in LEAGUES.values()
         if cfg.get("sport")}


def iso_z(ts):
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dt


def fetch_spreads(sport, eid, ts):
    r = requests.get(
        f"https://api.the-odds-api.com/v4/historical/sports/{sport}"
        f"/events/{eid}/odds",
        params={"apiKey": KEY, "regions": "eu,uk,us", "markets": "spreads",
                "oddsFormat": "decimal", "date": ts}, timeout=30)
    if r.status_code != 200:
        return None
    data = (r.json() or {}).get("data") or {}
    home, away = data.get("home_team"), data.get("away_team")
    quotes = {}
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] != "spreads":
                continue
            for o in m.get("outcomes", []):
                if o.get("point") is None:
                    continue
                side = ("home" if o["name"] == home else
                        "away" if o["name"] == away else None)
                if side:
                    quotes[(side, float(o["point"]), bk["key"])] = o["price"]
    return quotes


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = conn.execute("""
        SELECT r.event_id, f.league, r.kickoff_utc, r.home, r.away,
               r.home_goals, r.away_goals, f.xg_home, f.xg_away, f.rho,
               f.logged_at
        FROM match_results r
        JOIN match_fairs f ON f.event_id = r.event_id
          AND f.logged_at = (SELECT MIN(logged_at) FROM match_fairs
                             WHERE event_id = r.event_id)
        WHERE r.event_id NOT LIKE 'af_%'""").fetchall()
    conn.close()

    bets, skipped = [], 0
    for (eid, lg, ko, home, away, hg, ag, lam, mu, rho, flagged_at) in matches:
        sport = SPORT.get(lg)
        if not sport or lam is None:
            skipped += 1
            continue
        flag_ts, _ = iso_z(flagged_at)
        _, ko_dt = iso_z(ko)
        close_ts = (ko_dt - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        open_q = fetch_spreads(sport, eid, flag_ts)
        time.sleep(0.1)
        close_q = fetch_spreads(sport, eid, close_ts) if open_q else None
        time.sleep(0.1)
        if not open_q:
            skipped += 1
            continue
        mdist = margin_dist(dc_matrix(lam, mu, rho or 0.0))
        best = {}
        for (side, line_, book), price in open_q.items():
            if price > MAX_ODDS:
                continue
            pdist = mdist if side == "home" else mdist[::-1]
            ev = hcp_ev(pdist, line_, price)
            if ev < MIN_EV:
                continue
            k = (side, line_)
            if k not in best or price > best[k]["price"]:
                cl = (close_q or {}).get((side, line_, book))
                best[k] = {"league": lg, "home": home, "away": away,
                           "side": side, "line": line_, "book": book,
                           "price": price, "ev": ev,
                           "clv": price / cl - 1 if cl else None,
                           "pnl": hcp_result_pnl(
                               (hg - ag) if side == "home" else (ag - hg),
                               line_, price)}
        bets.extend(best.values())

    print(f"matches replayed: {len(matches) - skipped} "
          f"(skipped {skipped}: no archive/af)")
    if not bets:
        print("no AH bets would have been flagged")
        return
    n = len(bets)
    wins = sum(1 for b in bets if b["pnl"] > 0)
    pnl = sum(b["pnl"] for b in bets)
    cl = [b["clv"] for b in bets if b["clv"] is not None]
    aclv = sum(cl) / len(cl) * 100 if cl else 0.0
    print(f"\nAH flags (distinct, 1u at best price): {n}")
    print(f"  hit {wins/n*100:.1f}% | P/L {pnl:+.2f}u ({pnl/n*100:+.1f}% ROI)"
          f" | avgCLV {aclv:+.2f}% ({len(cl)} with close)")

    by = defaultdict(lambda: [0, 0.0, []])
    for b in bets:
        a = by[b["league"]]
        a[0] += 1
        a[1] += b["pnl"]
        if b["clv"] is not None:
            a[2].append(b["clv"])
    print(f"\n{'league':22s} {'bets':>5s} {'P/L(u)':>8s} {'avgCLV':>8s}")
    for lg, (nn, pp, cc) in sorted(by.items()):
        ac = sum(cc) / len(cc) * 100 if cc else 0.0
        print(f"{lg:22s} {nn:5d} {pp:+8.2f} {ac:+7.2f}%")

    print("\ntop flags by EV (would-have-been):")
    for b in sorted(bets, key=lambda x: -x["ev"])[:12]:
        clv = f"{b['clv']*100:+.1f}%" if b["clv"] is not None else "  n/a"
        print(f"  {b['league'][:12]:12s} {b['home'][:14]:14s} v "
              f"{b['away'][:14]:14s} {b['side']} {b['line']:+.2f} "
              f"@{b['price']:.2f} ({b['book'][:10]}) EV {b['ev']*100:+.1f}% "
              f"CLV {clv} pnl {b['pnl']:+.2f}")


if __name__ == "__main__":
    main()
