"""Rebuild the Lavish board payload (const P = ... in .lavish/scanner.html)
from stats.db. Run after scans/settles, then republish the artifact.

    python build_board.py
"""
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scanner.database import get_conn
from scanner.settlement import collapse_bets, BET_COLS

HTML = Path(__file__).parent / ".lavish" / "scanner.html"
ET = ZoneInfo("America/New_York")


def to_et(iso_utc: str) -> str:
    """UTC ISO -> 'YYYY-MM-DDTHH:MM' in US Eastern (board display TZ)."""
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return dt.astimezone(ET).isoformat(timespec="minutes")[:16]

LEAGUE_ORDER = ["Brazil Serie A", "Argentina Primera", "MLS", "Liga MX",
                "Eliteserien", "Allsvenskan", "K League 1", "J1 League",
                "Chinese Super League", "Danish Superliga",
                "Czech First League", "Scottish Premiership",
                "Austrian Bundesliga", "Primeira Liga", "Eredivisie",
                "Jupiler Pro League", "La Liga",
                "English Championship", "Premier League",
                "Serie A", "Ligue 1"]


def build_payload():
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    boards = {}
    for lg in LEAGUE_ORDER:
        rows = []
        for (eid, ko, home, away, anchor, ph, pd_, pa, xh, xa, xt,
             line, pov, btts) in conn.execute(
                """SELECT f.event_id, f.kickoff_utc, f.home, f.away,
                          f.anchor, f.p_home, f.p_draw, f.p_away,
                          f.xg_home, f.xg_away, f.x_total,
                          f.tot_line, f.p_over_line, f.btts_yes
                   FROM match_fairs f
                   WHERE f.league = ? AND f.kickoff_utc > ?
                     AND f.logged_at = (SELECT MAX(logged_at)
                                        FROM match_fairs
                                        WHERE event_id = f.event_id)
                   ORDER BY f.kickoff_utc""", (lg, now)):
            row = {"kickoff_utc": ko, "home": home, "away": away,
                   "anchor": anchor, "fair_home": ph, "fair_draw": pd_,
                   "fair_away": pa, "xg_home": xh, "xg_away": xa,
                   "x_total": xt, "tot_line": line, "fair_over": pov,
                   "btts_yes": btts}
            ind = conn.execute(
                """SELECT p_home, p_draw, p_away FROM indep_fairs
                   WHERE event_id = ? ORDER BY logged_at DESC LIMIT 1""",
                (eid,)).fetchone()
            if ind:
                row.update(ip_home=ind[0], ip_draw=ind[1], ip_away=ind[2])
            rows.append(row)
        if rows:
            boards[lg] = rows

    # open bets: best play per (match, market, side) — ladders collapse
    keys = BET_COLS.split(", ") + ["fair_price"]
    open_rows = [dict(zip(keys, r)) for r in conn.execute(
        f"""SELECT {BET_COLS}, fair_price FROM value_spots
            WHERE won IS NULL AND void = 0 AND kickoff_utc > ?""", (now,))]
    suggestions = [
        {"kick": to_et(b["kickoff_utc"])[5:], "ko": b["kickoff_utc"],
         "league": b["league"], "home": b["home"], "away": b["away"],
         "market": b["market"], "selection": b["selection"],
         "line": b["line"], "price": b["price"], "book": b["book"],
         "fair": b["fair_price"], "ev": b["ev"]}
        for b in sorted(collapse_bets(open_rows),
                        key=lambda x: x["kickoff_utc"])]

    by_market = defaultdict(int)
    for s in suggestions:
        by_market[s["market"]] += 1

    # settled ledger: best play per (match, market, side), shown in $
    # at the risk/win $100 convention (dogs risk 100, favourites risk
    # 100/(price-1) to win 100). DB stays 1u flat for research; pnl here
    # is DOLLARS = unit pnl x stake.
    def _stake(price):
        return 100.0 if price >= 2 else round(100 / (price - 1), 2)

    graded_rows = [dict(zip(BET_COLS.split(", "), r)) for r in conn.execute(
        f"SELECT {BET_COLS} FROM value_spots WHERE won IS NOT NULL")]
    settled = [
        {"date": to_et(b["kickoff_utc"])[5:10], "league": b["league"],
         "home": b["home"], "away": b["away"], "market": b["market"],
         "selection": b["selection"], "line": b["line"],
         "price": b["price"], "books": b["books"], "won": float(b["won"]),
         "stake": _stake(b["price"]),
         "pnl": round(b["pnl"] * _stake(b["price"])),
         # board JS renders clv with toFixed(2)+"%": PERCENT units
         "clv": round(b["clv"] * 100, 2) if b["clv"] is not None else None}
        for b in sorted(collapse_bets(graded_rows),
                        key=lambda x: x["kickoff_utc"], reverse=True)]

    graded = []
    agg = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
    for e in settled:
        a = agg[e["market"]]
        a["bets"] += 1
        a["wins"] += e["won"]
        a["pnl"] += e["pnl"]
    for mk in sorted(agg):
        a = agg[mk]
        graded.append({"market": mk, "bets": a["bets"],
                       "hit": a["wins"] / a["bets"],
                       "pnl": round(a["pnl"], 2)})

    conn.close()
    return {"boards": boards,
            "meta": {"spots_open": len(suggestions),
                     "by_market": dict(by_market), "graded": graded},
            "settled": settled, "suggestions": suggestions,
            "slate_date": to_et(now)[:10]}


def main():
    p = build_payload()
    html = HTML.read_text(encoding="utf-8")
    html, n = re.subn(r"^const P = .*;$",
                      "const P = " + json.dumps(p, ensure_ascii=False) + ";",
                      html, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("const P line not found in scanner.html")
    HTML.write_text(html, encoding="utf-8")
    n_up = sum(len(v) for v in p["boards"].values())
    print(f"payload rebuilt: {n_up} upcoming matches / "
          f"{p['meta']['spots_open']} open bets / "
          f"{len(p['settled'])} settled")


if __name__ == "__main__":
    main()
