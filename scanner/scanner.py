"""Scan mode: odds feeds, anchors, fair boards, value flagging.

Per run, for every match in a league: fair 1X2 + team xG (calibrated
matrix), independent v2 fairs (xG engine, no market inputs), every book
quote snapshotted, +EV spots logged (first flagged price is immutable -
it is the CLV reference).
"""
import csv
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from model.poisson import ZIPDixonColes
from model.independent import (IndependentEngine, load_league_xg,
                               load_league, xi_quality)
from mls_report import load_history, _strip_club, select_decay
from statsdb.schema import DB_PATH

from .database import get_conn
from .calibration import calibrated_matrix, fair_from_matrix, devig
from .asian_totals import total_dist, totals_ev, totals_fair_price
from .asian_handicap import margin_dist, hcp_ev, hcp_fair_price

MAX_ODDS = 6.0
# a "fair-value edge" above this is a data artifact, not a bet (owner
# call 2026-08-02): every real graded edge sits in 3-15%; the 20%+ ones
# were stale feeds, in-play leaks, or phantom fixtures. Skipped flags
# are printed as warnings - they usually mean a feed problem.
MAX_EV = 0.20

# stats.db league ids for the independent (v2, xG-based) engine
INDEP_LEAGUE_IDS = {"brazil": 71, "mls": 253, "ligamx": 262,
                    "norway": 103, "sweden": 113, "korea": 292,
                    "japan": 98, "china": 169, "denmark": 119,
                    "czech": 345, "serbia": 286, "argentina": 128,
                    "scotland": 179, "austria": 218, "portugal": 94,
                    "netherlands": 88, "belgium": 144,
                    "laliga": 140, "championship": 40,
                    "epl": 39, "seriea": 135, "ligue1": 61}

# Odds-API name -> warehouse (API-Football) name for the independent
# engine, for the cases normalization cannot bridge. Only names the
# resolver reports as unresolved or ambiguous belong here; everything
# else should keep matching on its own. Keys are normalized before
# lookup, so case/accents/punctuation in them do not matter.
INDEP_ALIASES = {
    "argentina": {
        "Gimnasia La Plata": "Gimnasia L.P.",
        "Argentinos Juniors": "Argentinos JRS",
        # The feed spells 'Estudiantes de Rio Cuarto' out in full on the
        # same slate, so a bare 'Estudiantes' is the La Plata club.
        # Without this the two are an ambiguous pair and BOTH matches
        # lose their independent fair.
        "Estudiantes": "Estudiantes L.P.",
    },
}

AF_BET_MAP = {"Match Winner": "h2h", "Goals Over/Under": "totals",
              "Both Teams Score": "btts", "Asian Handicap": "spread"}

MANUAL_ANCHORS_PATH = Path(__file__).resolve().parents[1] / "data" / \
    "manual_anchors.json"


# --------------------------------------------------------------- feeds

def fetch_events(cfg, key):
    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{cfg['sport']}/odds",
        params={"apiKey": key, "regions": "us,eu,uk",
                "markets": "h2h,totals,spreads",
                "oddsFormat": "decimal"}, timeout=30)
    r.raise_for_status()
    events = r.json()
    for ev in events:
        try:
            r2 = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{cfg['sport']}"
                f"/events/{ev['id']}/odds",
                params={"apiKey": key, "regions": "us,eu,uk",
                        "markets": "btts", "oddsFormat": "decimal"},
                timeout=30)
            if r2.status_code == 200:
                by_key = {b["key"]: b for b in ev.setdefault("bookmakers", [])}
                for b in r2.json().get("bookmakers", []):
                    if b["key"] in by_key:
                        by_key[b["key"]]["markets"].extend(b.get("markets", []))
                    else:
                        ev["bookmakers"].append(b)
        except requests.RequestException:
            pass
    return events


def fetch_events_af(cfg):
    """Odds via API-Football for leagues The Odds API doesn't carry.
    Returns events shaped like the Odds API feed."""
    from statsdb.apifootball import ApiFootballClient
    conn = sqlite3.connect(DB_PATH)
    names = dict(conn.execute("SELECT team_id, name FROM teams").fetchall())
    fixmap = {fid: (names.get(h), names.get(a), ko) for fid, h, a, ko in
              conn.execute("SELECT fixture_id, home_id, away_id, kickoff_utc "
                           "FROM fixtures WHERE league_id=? AND season=?",
                           (cfg["af_league"], cfg["af_season"]))}
    conn.close()

    api = ApiFootballClient(max_requests=300, min_interval=0.2)
    events, page = {}, 1
    while True:
        resp = api.get("odds", force=True, league=cfg["af_league"],
                       season=cfg["af_season"], page=page)
        if not resp:
            break
        for item in resp:
            fid = item["fixture"]["id"]
            if fid not in fixmap:
                continue
            home, away, ko = fixmap[fid]
            if not home or not away:
                continue
            ev = events.setdefault(fid, {
                "id": f"af_{fid}", "home_team": home, "away_team": away,
                "commence_time": ko, "bookmakers": []})
            for bk in item.get("bookmakers", []):
                markets = []
                for bet in bk.get("bets", []):
                    mk = AF_BET_MAP.get(bet.get("name"))
                    if mk is None:
                        continue
                    outs = []
                    for v in bet.get("values", []):
                        val, odd = str(v.get("value")), float(v.get("odd", 0))
                        if odd <= 1:
                            continue
                        if mk == "h2h":
                            nm = {"Home": home, "Draw": "Draw",
                                  "Away": away}.get(val)
                            if nm:
                                outs.append({"name": nm, "price": odd})
                        elif mk == "totals" and " " in val:
                            side, pt = val.split(" ", 1)
                            try:
                                outs.append({"name": side, "price": odd,
                                             "point": float(pt)})
                            except ValueError:
                                pass
                        elif mk == "spread" and " " in val:
                            # AF quotes the HOME-perspective line for both
                            # sides ("Away +1" = home +1 = away -1); the
                            # away side's own start is the negation.
                            # 2026-07-26: unnegated parse made fake +400%
                            # EV flags (Partizan "+3.75" at 6.00).
                            side, pt = val.split(" ", 1)
                            try:
                                p = float(pt)
                            except ValueError:
                                continue
                            if side == "Home":
                                outs.append({"name": home, "price": odd,
                                             "point": p})
                            elif side == "Away":
                                outs.append({"name": away, "price": odd,
                                             "point": -p})
                        elif mk == "btts":
                            outs.append({"name": val, "price": odd})
                    if outs:
                        markets.append({"key": mk, "outcomes": outs})
                if markets:
                    ev["bookmakers"].append(
                        {"key": bk.get("name", "book").lower(),
                         "markets": markets})
        if len(resp) < 10:
            break
        page += 1
    return [e for e in events.values() if e["bookmakers"]]


def collect_offers(ev):
    home, away = ev["home_team"], ev["away_team"]
    offers, h2h_by_book, pinnacle = defaultdict(list), {}, None
    for bk in ev.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                try:
                    o = {x["name"]: x["price"] for x in mkt["outcomes"]}
                    trio = (o[home], o["Draw"], o[away])
                except KeyError:
                    continue
                h2h_by_book[bk["key"]] = trio
                if bk["key"] == "pinnacle":
                    pinnacle = trio
                for sel, price in [("home", o[home]), ("draw", o["Draw"]),
                                   ("away", o[away])]:
                    offers[("h2h", sel, 0.0)].append((bk["key"], price))
            elif mkt["key"] == "totals":
                for x in mkt["outcomes"]:
                    if x.get("point") is not None and x["point"] % 1 != 0:
                        offers[("totals", x["name"].lower(),
                                float(x["point"]))].append(
                            (bk["key"], x["price"]))
            elif mkt["key"] in ("spreads", "spread"):
                for x in mkt["outcomes"]:
                    if x.get("point") is None:
                        continue
                    side = ("home" if x["name"] == home else
                            "away" if x["name"] == away else None)
                    if side:
                        offers[("spread", side, float(x["point"]))].append(
                            (bk["key"], x["price"]))
            elif mkt["key"] == "btts":
                for x in mkt["outcomes"]:
                    offers[("btts", x["name"].lower(), 0.0)].append(
                        (bk["key"], x["price"]))
    return offers, h2h_by_book, pinnacle


# ------------------------------------------------------------- anchors

def totals_anchor(offers):
    tq = defaultdict(lambda: {"over": [], "under": [], "pinn": {}})
    for (mk, sel, line), quotes in offers.items():
        if mk != "totals":
            continue
        for book, price in quotes:
            tq[line][sel].append(price)
            if book == "pinnacle":
                tq[line]["pinn"][sel] = price
    pinn = {l: d["pinn"] for l, d in tq.items()
            if {"over", "under"} <= d["pinn"].keys()}
    if pinn:
        line = min(pinn, key=lambda l: abs(pinn[l]["over"] - pinn[l]["under"]))
        return line, devig([pinn[line]["over"], pinn[line]["under"]])[0]
    cands = {l: d for l, d in tq.items()
             if len(d["over"]) >= 3 and len(d["under"]) >= 3}
    if not cands:
        return None, None
    line = max(cands, key=lambda l: len(cands[l]["over"]))
    return line, devig([float(np.median(cands[line]["over"])),
                        float(np.median(cands[line]["under"]))])[0]


def load_manual_anchors(league_key):
    if not MANUAL_ANCHORS_PATH.exists():
        return []
    try:
        return json.loads(MANUAL_ANCHORS_PATH.read_text(
            encoding="utf-8")).get(league_key, [])
    except (json.JSONDecodeError, OSError):
        return []


def _mtok(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode(
        "ascii", "ignore").decode().lower()
    return set(s.replace("-", " ").split())


def match_manual(manual, home, away):
    th, ta = _mtok(home), _mtok(away)
    for m in manual:
        mh, mt = _mtok(m["home"]), _mtok(m["away"])
        if (mh & th) and (mt & ta):
            return m
    return None


# ------------------------------------------------- independent engine

def _build_indep(league_key, conn):
    """v2 independent engine + odds-api-name resolver for this league.
    Returns (engine, resolve_fn, xi_fn) or None if unavailable."""
    lid = INDEP_LEAGUE_IDS.get(league_key)
    if lid is None:
        return None
    try:
        matches = load_league_xg(conn, lid)
        if len(matches) < 300:
            return None
        decay = select_decay(matches)   # per-league, walk-forward tuned
        print(f"  v2 decay_xi={decay} "
              f"(half-life ~{int(np.log(2) / decay)} days)")
        xi = xi_quality(conn, lid)
        eng = IndependentEngine(decay_xi=decay).fit(matches, xi)
        # xg_alpha=1.0 default (swept optimum)
        name_to_id = dict(zip(matches["HomeTeam"], matches["home_id"]))
        name_to_id.update(zip(matches["AwayTeam"], matches["away_id"]))
        import re
        import unicodedata

        def n(s):
            s = unicodedata.normalize("NFKD", str(s)).encode(
                "ascii", "ignore").decode().lower()
            # ALL punctuation -> space. 'Yokohama F. Marinos' (warehouse)
            # and 'Yokohama F Marinos' (odds API) must normalize alike:
            # when they did not, the exact branch missed and the fuzzy
            # fallback matched club-suffix-stripped 'Yokohama FC' ->
            # 'yokohama' as the ONLY candidate, so the wrong club's
            # ratings priced the match with no warning (2026-08-18).
            s = re.sub(r"[^a-z0-9]+", " ", s)
            return _strip_club(" ".join(s.split()))

        norm_map = {}
        for k in name_to_id:
            nk = n(k)
            if nk in norm_map and norm_map[nk] != k:
                print(f"  [WARN] '{k}' and '{norm_map[nk]}' normalize "
                      f"alike ('{nk}') - independent fairs for both are "
                      f"unreliable, add a distinguishing alias")
            norm_map[nk] = k

        aliases = {}
        for raw, target in INDEP_ALIASES.get(league_key, {}).items():
            if target not in name_to_id:
                print(f"  [WARN] alias '{raw}' -> '{target}' not found in "
                      f"{league_key} history - alias ignored (renamed "
                      f"upstream?)")
                continue
            aliases[n(raw)] = target

        def resolve(name):
            nn = n(name)
            if nn in aliases:      # explicit owner mapping wins
                return aliases[nn]
            if nn in norm_map:
                return norm_map[nn]
            nt = set(nn.split())
            cands = {v for k, v in norm_map.items()
                     if nn in k or k in nn
                     or nt <= set(k.split()) or set(k.split()) <= nt}
            if len(cands) == 1:
                return next(iter(cands))
            # ambiguous or unknown: skip rather than guess. Silent before
            # this print, which is how the Yokohama mismatch survived.
            print(f"  [WARN] no unique history match for '{name}'"
                  + (f" (candidates: {sorted(cands)})" if cands else "")
                  + " - independent fair skipped")
            return None

        return eng, resolve, (lambda af: eng.latest_xi(name_to_id[af]))
    except Exception as e:
        print(f"  [WARN] independent engine unavailable: {e}")
        return None


# ----------------------------------------------------------- scan mode

def scan(args, cfg):
    print(f"[1/3] league rho prior from {cfg['label']} history...")
    hist_id = cfg.get("af_league") or cfg.get("hist_league")
    if hist_id:
        c0 = sqlite3.connect(DB_PATH)
        hist = load_league(c0, hist_id)
        c0.close()
    else:
        hist = load_history(cfg["csv"])
    rho0 = ZIPDixonColes(zero_inflation=False,
                         dc_adjust=True).fit(hist).params["rho"]

    print("[2/3] per-book odds...")
    key = os.environ.get("THE_ODDS_API_KEY")
    events = (fetch_events_af(cfg) if cfg.get("af_league")
              else fetch_events(cfg, key))
    print(f"  {len(events)} events")

    print("[3/3] fairs board + scan...\n")
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    board, n_spots = [], 0
    indep = _build_indep(args.league, conn)
    n_indep = 0
    manual_anchors = load_manual_anchors(args.league)
    if manual_anchors:
        print(f"  {len(manual_anchors)} manual sharp anchors loaded")

    for ev in events:
        # skip matches already kicked off: both feeds can return live
        # events, and in-play prices vs pre-match fairs are fake edges
        # (2026-07-25: 23 such flags purged, incl. a graded ledger bet)
        if ev.get("commence_time", "") <= now.replace("+00:00", "Z"):
            continue
        home, away = ev["home_team"], ev["away_team"]
        offers, h2h_by_book, pinnacle = collect_offers(ev)
        if len(h2h_by_book) < 3:
            continue
        for (mk, sel, line), quotes in offers.items():
            for book, price in quotes:
                conn.execute("INSERT OR REPLACE INTO odds_snapshots VALUES "
                             "(?,?,?,?,?,?,?)",
                             (now, ev["id"], mk, sel, line, book, price))

        anchor = pinnacle or tuple(np.median(
            [list(t) for t in h2h_by_book.values()], axis=0))
        p = devig(anchor)
        line, p_over = totals_anchor(offers)
        src = "pinnacle" if pinnacle else "median"

        ma = match_manual(manual_anchors, home, away)
        if ma:   # owner-supplied sharp anchor (e.g. Pinnacle site odds)
            p = devig(ma["h2h"])
            line = float(ma["line"])
            p_over = devig([ma["over"], ma["under"]])[0]
            src = "pinnacle-manual"
        if line is None:
            continue
        m, lam, mu, rho, fit_err = calibrated_matrix(p[0], p[2], p_over,
                                                     line, rho0)
        if m is None:
            continue
        fair = fair_from_matrix(m)
        dist = total_dist(m)
        mdist = margin_dist(m)          # for Asian handicap pricing
        fp_over = totals_fair_price(dist, "over", line)
        eff_over = 1 / fp_over if fp_over else 0.5   # break-even prob

        row = {"kickoff_utc": ev["commence_time"], "home": home,
               "away": away, "anchor": src,
               "fair_home": round(p[0], 4), "fair_draw": round(p[1], 4),
               "fair_away": round(p[2], 4),
               "xg_home": round(lam, 2), "xg_away": round(mu, 2),
               "x_total": round(lam + mu, 2),
               "tot_line": line, "fair_over": round(eff_over, 4),
               "btts_yes": round(fair["btts_yes"], 4)}
        board.append(row)
        conn.execute("INSERT OR REPLACE INTO match_fairs VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (now, cfg["label"], ev["id"], ev["commence_time"],
                      home, away, src, *p.round(4), round(lam, 3),
                      round(mu, 3), round(lam + mu, 3), round(rho, 3),
                      line, round(eff_over, 4),
                      round(fair["btts_yes"], 4), fit_err))

        # independent v2 fairs (xG engine, no market inputs) - the
        # prospective model-vs-opener-vs-close record
        if indep:
            eng, resolve, latest_xi = indep
            afh, afa = resolve(home), resolve(away)
            if afh and afa:
                try:
                    il, im = eng.rates(afh, afa, latest_xi(afh),
                                       latest_xi(afa))
                    mtx = eng._matrix(il, im)
                    size = mtx.shape[0]
                    lo = int(np.ceil(line))
                    ip_over = float(sum(mtx[i, j] for i in range(size)
                                        for j in range(size)
                                        if i + j >= lo))
                    ibtts = float(1 - mtx[0, :].sum() - mtx[:, 0].sum()
                                  + mtx[0, 0])
                    conn.execute(
                        "INSERT OR REPLACE INTO indep_fairs VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now, cfg["label"], ev["id"], ev["commence_time"],
                         home, away,
                         round(float(np.tril(mtx, -1).sum()), 4),
                         round(float(np.trace(mtx)), 4),
                         round(float(np.triu(mtx, 1).sum()), 4),
                         round(il, 3), round(im, 3),
                         line, round(ip_over, 4), round(ibtts, 4)))
                    n_indep += 1
                except (KeyError, ValueError):
                    pass

        # af-league feeds carry no live sharp book: value flags need an
        # owner-supplied Pinnacle anchor. Median-of-soft-books still
        # prices the board but must not call edges (2026-07-27: Serbia
        # MW3 median anchors flagged BOTH sides of the same btts market)
        can_flag = ((not cfg.get("af_league")) or src == "pinnacle-manual") \
            and not cfg.get("no_flag")
        for (mk, sel, oline), quotes in offers.items():
            if mk == "h2h":
                fp = {"home": p[0], "draw": p[1], "away": p[2]}[sel]
            elif mk == "totals":
                d = totals_fair_price(dist, sel, oline)
                if d is None:
                    continue
                fp = 1 / d          # payoff-aware break-even prob
            elif mk == "spread":
                pdist = mdist if sel == "home" else mdist[::-1]
                d = hcp_fair_price(pdist, oline)
                if d is None:
                    continue
                fp = 1 / d
            else:
                fp = fair["btts_yes"] if sel == "yes" else 1 - fair["btts_yes"]
            if not 0 < fp < 1:
                continue
            for book, price in quotes:
                if price > MAX_ODDS:
                    continue
                if mk == "totals":
                    ev_val = totals_ev(dist, sel, oline, price)
                elif mk == "spread":
                    ev_val = hcp_ev(mdist if sel == "home" else mdist[::-1],
                                    oline, price)
                else:
                    ev_val = fp * price - 1
                if ev_val > MAX_EV and can_flag:
                    print(f"  [WARN] implausible edge skipped: {home} v "
                          f"{away} {mk} {sel} {oline or ''} @{price} "
                          f"({book}) EV {ev_val*100:+.0f}% - check feed")
                    continue
                if ev_val >= args.min_ev and can_flag:
                    # OR IGNORE: the FIRST flagged price is the CLV
                    # reference and must never be overwritten by re-scans
                    conn.execute(
                        "INSERT OR IGNORE INTO value_spots (logged_at, "
                        "league, event_id, kickoff_utc, home, away, "
                        "market, selection, line, book, price, fair_prob, "
                        "fair_price, ev) VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now, cfg["label"], ev["id"], ev["commence_time"],
                         home, away, mk, sel, oline, book, price,
                         round(fp, 4), round(1 / fp, 3), round(ev_val, 4)))
                    n_spots += 1
    conn.commit()

    print(f"{'match':42s} {'fair H/D/A':>17s} {'xG':>11s} "
          f"{'xTot':>5s} {'line':>5s} {'P(ov)':>6s} {'BTTS':>6s}")
    for r in sorted(board, key=lambda x: x["kickoff_utc"]):
        print(f"{r['home'][:19]:20s} v {r['away'][:19]:20s} "
              f"{r['fair_home']:.2f}/{r['fair_draw']:.2f}/{r['fair_away']:.2f} "
              f"{r['xg_home']:.2f}-{r['xg_away']:.2f} "
              f"{r['x_total']:5.2f} {r['tot_line']:5.2f} "
              f"{r['fair_over']:6.2f} {r['btts_yes']:6.2f}")

    total = conn.execute("SELECT COUNT(*) FROM value_spots").fetchone()[0]
    conn.close()
    if not board:   # e.g. every remaining match already kicked off
        print("\nboard: 0 matches (nothing upcoming to price)")
        print(f"spots: {n_spots} flagged this run; {total} total in log")
        return
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = Path("reports")
    out.mkdir(exist_ok=True)
    jpath = out / f"fairs_{args.league}_{stamp}.json"
    jpath.write_text(json.dumps(board, indent=2), encoding="utf-8")
    cpath = out / f"fairs_{args.league}_{stamp}.csv"
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(board[0].keys()))
        w.writeheader()
        w.writerows(board)
    print(f"\nboard: {len(board)} matches -> {cpath.name} / {jpath.name}")
    print(f"spots: {n_spots} flagged this run; {total} total in log")
    if indep:
        print(f"independent v2 fairs logged: {n_indep}")
