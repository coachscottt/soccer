"""
MLS report: Dixon-Coles scoreline model vs live market odds.

    python mls_report.py           (manual runs only - no scheduler)

Pipeline per run:
  1. Fresh MLS history from football-data.co.uk (results + closing odds)
  2. Fit per-run Dixon-Coles model (time-decay weighted)
  3. Rolling honesty check: model vs Pinnacle closing on the last 100
     matches - printed every run so the model-lags-market fact stays visible
  4. Live h2h + totals odds from The Odds API (consensus = median across books)
  5. Per fixture: model 1X2 vs devigged market, divergences, totals
     probabilities vs the real over/under line, model-EV on totals,
     top scorelines
  6. JSON report to reports/, optional Telegram summary

The 1X2 divergences are context, not edges (the model has never beaten a
closing line in any of our validations). The totals comparison is the
component with demonstrated skill on European data - still unproven on MLS.
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import log_loss

from model.poisson import ZIPDixonColes
from output import write_json_report, send_telegram_message

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")   # root: THE_ODDS_API_KEY

LEAGUES = {
    "mls": {"label": "MLS",
            "csv": "https://www.football-data.co.uk/new/USA.csv",
            "sport": "soccer_usa_mls"},
    "brazil": {"label": "Brazil Serie A",
               "csv": "https://www.football-data.co.uk/new/BRA.csv",
               "sport": "soccer_brazil_campeonato"},
    "ligamx": {"label": "Liga MX",
               "csv": "https://www.football-data.co.uk/new/MEX.csv",
               "sport": "soccer_mexico_ligamx"},
    "norway": {"label": "Eliteserien",
               "csv": "https://www.football-data.co.uk/new/NOR.csv",
               "sport": "soccer_norway_eliteserien"},
    "sweden": {"label": "Allsvenskan",
               "csv": "https://www.football-data.co.uk/new/SWE.csv",
               "sport": "soccer_sweden_allsvenskan"},
    "korea":  {"label": "K League 1",
               "csv": "https://www.football-data.co.uk/new/KOR.csv",
               "sport": "soccer_korea_kleague1"},
    "japan":  {"label": "J1 League",   # activates Aug 2026 (new season)
               "csv": "https://www.football-data.co.uk/new/JPN.csv",
               "sport": "soccer_japan_j_league"},
    "china":  {"label": "Chinese Super League",
               "csv": "https://www.football-data.co.uk/new/CHN.csv",
               "sport": "soccer_china_superleague"},
    "denmark": {"label": "Danish Superliga",
                "csv": "https://www.football-data.co.uk/new/DNK.csv",
                "sport": "soccer_denmark_superliga"},
    "argentina": {"label": "Argentina Primera",
                  "csv": "https://www.football-data.co.uk/new/ARG.csv",
                  "sport": "soccer_argentina_primera_division"},
    # odds from The Odds API, history from the stats.db warehouse
    # (football-data only has per-season files for Scotland)
    "scotland": {"label": "Scottish Premiership", "csv": None,
                 "sport": "soccer_spl", "hist_league": 179},
    "portugal": {"label": "Primeira Liga", "csv": None,
                 "sport": "soccer_portugal_primeira_liga",
                 "hist_league": 94},
    "netherlands": {"label": "Eredivisie", "csv": None,
                    "sport": "soccer_netherlands_eredivisie",
                    "hist_league": 88},
    "belgium": {"label": "Jupiler Pro League", "csv": None,
                "sport": "soccer_belgium_first_div",
                "hist_league": 144},
    "laliga": {"label": "La Liga", "csv": None,
               "sport": "soccer_spain_la_liga", "hist_league": 140},
    "championship": {"label": "English Championship", "csv": None,
                     "sport": "soccer_efl_champ", "hist_league": 40},
    "epl": {"label": "Premier League", "csv": None,
            "sport": "soccer_epl", "hist_league": 39},
    "seriea": {"label": "Serie A", "csv": None,
               "sport": "soccer_italy_serie_a", "hist_league": 135},
    "ligue1": {"label": "Ligue 1", "csv": None,
               "sport": "soccer_france_ligue_one", "hist_league": 61},
    "austria": {"label": "Austrian Bundesliga", "csv": None,
                "sport": "soccer_austria_bundesliga", "hist_league": 218},
    # The Odds API carries neither league -> odds via API-Football
    # (af_league/af_season), history/results via stats.db
    "czech":  {"label": "Czech First League", "csv": None, "sport": None,
               "af_league": 345, "af_season": 2026},
    # retired from tracking 2026-08-02 (owner call): API-Football book
    # odds unverifiably stale -> fake EV (e.g. Vojvodina -2377 cluster).
    # Board/fairs only if ever scanned; no_flag blocks value_spots.
    "serbia": {"no_flag": True,"label": "Serbia SuperLiga", "csv": None, "sport": None,
               "af_league": 286, "af_season": 2026},
}
STALE_DAYS = 21

# The Odds API name -> football-data name (fallback: suffix-stripped match)
ALIASES = {
    "st. louis city sc": "St. Louis City",
    "vancouver whitecaps fc": "Vancouver Whitecaps",
    "los angeles fc": "Los Angeles FC",
    "la galaxy": "Los Angeles Galaxy",
    "atlanta united fc": "Atlanta Utd",
    "atlanta united": "Atlanta Utd",
    "inter miami cf": "Inter Miami",
    "minnesota united fc": "Minnesota United",
    "new england revolution": "New England Revolution",
    "columbus crew sc": "Columbus Crew",
    "d.c. united": "DC United",
    "sporting kansas city": "Sporting Kansas City",
    # Brazil (football-data suffixes state codes)
    "atletico mineiro": "Atletico-MG",
    "atletico paranaense": "Athletico-PR",
    "athletico paranaense": "Athletico-PR",
    "chapecoense": "Chapecoense-SC",
    "flamengo": "Flamengo RJ",
    "botafogo": "Botafogo RJ",
    "red bull bragantino": "Bragantino",
    "bragantino-sp": "Bragantino",
    "vasco da gama": "Vasco",
    "sport club do recife": "Sport Recife",
    # Liga MX
    "atletico san luis": "Atl. San Luis",
    "america": "Club America",
    "club america": "Club America",
    "leon": "Club Leon",
    "tijuana": "Club Tijuana",
    "chivas": "Guadalajara Chivas",
    "guadalajara": "Guadalajara Chivas",
    "fc juarez": "Juarez",
    "mazatlan": "Mazatlan FC",
    "tigres uanl": "Tigres UANL",
    "tigres": "Tigres UANL",
    "pumas unam": "UNAM Pumas",
    "pumas": "UNAM Pumas",
    # Eliteserien (Odds API adds FK/BK/IK/SK suffixes football-data omits)
    "bodo/glimt": "Bodo/Glimt",
    "bodo glimt": "Bodo/Glimt",
    "viking fk": "Viking",
    "kristiansund bk": "Kristiansund",
    "ik start": "Start",
    "sarpsborg 08 ff": "Sarpsborg 08",
    "valerenga fotball": "Valerenga",
    "sandefjord fotball": "Sandefjord",
    "fk haugesund": "Haugesund",
    "odds bk": "Odd",
    # Allsvenskan (names that differ beyond club-token suffixes)
    "djurgardens if": "Djurgarden",
    "djurgarden": "Djurgarden",
    "malmo ff": "Malmo FF",
    "ifk goteborg": "Goteborg",
    "ifk norrkoping": "Norrkoping",
    "ifk varnamo": "Varnamo",
    "osters if": "Oster",
}


def load_history(csv_url: str) -> pd.DataFrame:
    cache = (Path(__file__).resolve().parent / "data" / "history_cache" /
             csv_url.rsplit("/", 1)[-1])
    try:
        r = requests.get(csv_url, timeout=30)
        r.raise_for_status()
        text = r.text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    except requests.RequestException as e:
        if not cache.exists():
            raise
        print(f"  [WARN] {csv_url.rsplit('/',1)[-1]} fetch failed ({e}); "
              "using cached history")
        text = cache.read_text(encoding="utf-8")
    df = pd.read_csv(StringIO(text), on_bad_lines="skip", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                            "HG": "FTHG", "AG": "FTAG"})
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    text_cols = {"Country", "League", "Date", "Time",
                 "HomeTeam", "AwayTeam", "Res"}
    for c in df.columns:
        if c not in text_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date", "FTHG", "FTAG"]).sort_values("Date")


def recalibrate_environment(mdl: ZIPDixonColes, hist: pd.DataFrame,
                            n_recent: int = 300) -> float:
    """Scale the base scoring rate so model-implied totals match the
    recent observed scoring environment (MLS 2026 runs ~10% hotter
    than the decay-weighted history). Relative strengths untouched."""
    recent = hist.tail(n_recent)
    implied = np.mean([sum(mdl.rates(m["HomeTeam"], m["AwayTeam"]))
                       for _, m in recent.iterrows()])
    observed = (recent["FTHG"] + recent["FTAG"]).mean()
    r = observed / implied
    if abs(r - 1) > 0.03:
        mdl.params["base"] += np.log(r)
    return r


def select_decay(hist: pd.DataFrame, n_eval: int = 200) -> float:
    """Pick the time-decay rate that best predicts the most recent
    matches (guards against scoring-environment drift - MLS 2026 is
    scoring ~10% above the multi-season average)."""
    cutoff = hist["Date"].iloc[-n_eval]
    train = hist[hist["Date"] < cutoff]
    recent = hist[hist["Date"] >= cutoff]
    best_xi, best_ll = None, np.inf
    for xi in (0.0018, 0.004, 0.008):
        mdl = ZIPDixonColes(zero_inflation=False, dc_adjust=True,
                            decay_xi=xi).fit(train)
        ll = 0.0
        for _, m in recent.iterrows():
            mat = mdl.score_matrix(m["HomeTeam"], m["AwayTeam"])
            h, a = int(m["FTHG"]), int(m["FTAG"])
            if h <= mdl.max_goals and a <= mdl.max_goals:
                ll -= np.log(max(mat[h, a], 1e-10))
        if ll < best_ll:
            best_xi, best_ll = xi, ll
    return best_xi


def _closing_probs(m) -> np.ndarray | None:
    for pre in ("PSC", "AvgC"):
        vals = [m.get(f"{pre}A"), m.get(f"{pre}D"), m.get(f"{pre}H")]
        if all(pd.notna(v) and v > 1 for v in vals):
            inv = np.array([1/v for v in vals])
            return inv / inv.sum()
    return None


def honesty_check(hist: pd.DataFrame, decay_xi: float,
                  n_last: int = 100) -> str:
    cutoff = hist["Date"].iloc[-n_last]
    mdl = ZIPDixonColes(zero_inflation=False, dc_adjust=True,
                        decay_xi=decay_xi).fit(hist[hist["Date"] < cutoff])
    probs, mkts, ys = [], [], []
    for _, m in hist[hist["Date"] >= cutoff].iterrows():
        mk = _closing_probs(m)
        if mk is None:
            continue
        p = mdl.predict_1x2(m["HomeTeam"], m["AwayTeam"])
        probs.append([p["away"], p["draw"], p["home"]])
        mkts.append(mk)
        ys.append({"A": 0, "D": 1, "H": 2}[m["Res"]])
    if len(ys) < 20:
        return "honesty check: insufficient recent matches with closing odds"
    ll_m = log_loss(ys, np.array(probs), labels=[0, 1, 2])
    ll_k = log_loss(ys, np.array(mkts), labels=[0, 1, 2])
    verdict = "model LAGS market" if ll_m > ll_k else "model AHEAD of market (!)"
    return (f"honesty check (last {len(ys)}): model log_loss {ll_m:.4f} vs "
            f"Pinnacle closing {ll_k:.4f} -> {verdict}")


CLUB_TOKENS = {"fc", "sc", "cf", "if", "is", "ff", "aif", "bk", "sk",
               "ifk", "ik", "fk", "afc"}


def _strip_club(low: str) -> str:
    kept = [t for t in low.split() if t not in CLUB_TOKENS]
    return " ".join(kept) or low


def resolve_team(name: str, known: set[str]) -> str | None:
    import unicodedata
    name = unicodedata.normalize("NFKD", name).encode(
        "ascii", "ignore").decode()
    low = name.lower().strip()
    if ALIASES.get(low) in known:
        return ALIASES[low]
    stripped = _strip_club(low)
    for team in known:
        if _strip_club(team.lower()) == stripped:
            return team
    # token-subset fallback
    tokens = set(low.split())
    cands = [team for team in known
             if tokens <= set(team.lower().split()) or
                set(team.lower().split()) <= tokens]
    return cands[0] if len(cands) == 1 else None


def fetch_live_odds(sport: str) -> list[dict]:
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        print("  [WARN] THE_ODDS_API_KEY not set - skipping live odds")
        return []
    r = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                     params={"apiKey": key, "regions": "us",
                             "markets": "h2h,totals",
                             "oddsFormat": "decimal"}, timeout=30)
    r.raise_for_status()
    print(f"  odds api quota remaining: {r.headers.get('x-requests-remaining')}")
    return r.json()


def consensus(event: dict) -> dict:
    """Median odds across bookmakers for h2h and the modal totals line."""
    h2h = defaultdict(list)
    totals = defaultdict(lambda: defaultdict(list))   # line -> side -> prices
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                for o in mkt["outcomes"]:
                    h2h[o["name"]].append(o["price"])
            elif mkt["key"] == "totals":
                for o in mkt["outcomes"]:
                    if o.get("point") is not None:
                        totals[o["point"]][o["name"]].append(o["price"])
    out = {"h2h": {k: float(np.median(v)) for k, v in h2h.items()}}
    if totals:
        line = max(totals, key=lambda l: len(totals[l].get("Over", [])))
        if {"Over", "Under"} <= totals[line].keys() and line % 1 != 0:
            out["totals_line"] = float(line)
            out["over"] = float(np.median(totals[line]["Over"]))
            out["under"] = float(np.median(totals[line]["Under"]))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(LEAGUES), default="mls")
    cfg = LEAGUES[ap.parse_args().league]

    print(f"[1/4] {cfg['label']} history...")
    hist = load_history(cfg["csv"])
    last = hist["Date"].max()
    age = (pd.Timestamp.now() - last).days
    print(f"  {len(hist)} matches, latest {last.date()} ({age} days ago)")
    if age > STALE_DAYS:
        print(f"  [WARN] model data is {age} days stale (league break?) - "
              "market has priced news the model cannot see; treat 1X2 "
              "divergences as lag, not edge")

    print("[2/4] Fitting Dixon-Coles + honesty check...")
    xi = select_decay(hist)
    print(f"  selected decay_xi={xi} (half-life ~{int(np.log(2)/xi)} days)")
    mdl = ZIPDixonColes(zero_inflation=False, dc_adjust=True,
                        decay_xi=xi).fit(hist)
    r = recalibrate_environment(mdl, hist)
    if abs(r - 1) > 0.03:
        print(f"  scoring-environment recalibration applied: x{r:.3f} "
              "(recent scoring vs decay-weighted history)")
    honesty = honesty_check(hist, decay_xi=xi)
    print(f"  {honesty}")

    print("[3/4] Live odds...")
    events = fetch_live_odds(cfg["sport"])
    known = set(mdl.params["attack"])

    print("[4/4] Building report...\n")
    reports, lines = [], []
    for ev in events:
        home = resolve_team(ev["home_team"], known)
        away = resolve_team(ev["away_team"], known)
        if not home or not away:
            print(f"  [skip] {ev['home_team']} vs {ev['away_team']} "
                  "(unmapped team)")
            continue
        cons = consensus(ev)
        if len(cons["h2h"]) < 3:
            continue
        p = mdl.predict_1x2(home, away)
        inv = np.array([1/cons["h2h"][ev["home_team"]],
                        1/cons["h2h"].get("Draw", np.inf),
                        1/cons["h2h"][ev["away_team"]]])
        mkt = inv / inv.sum()   # devigged H, D, A

        entry = {
            "match": f"{home} vs {away}",
            "kickoff_utc": ev["commence_time"],
            "model_1x2": {k: round(v, 4) for k, v in p.items()},
            "market_1x2": {"home": round(mkt[0], 4), "draw": round(mkt[1], 4),
                           "away": round(mkt[2], 4)},
            "divergence": {"home": round(p["home"] - mkt[0], 4),
                           "draw": round(p["draw"] - mkt[1], 4),
                           "away": round(p["away"] - mkt[2], 4)},
            "top_scorelines": [{"score": s, "prob": round(q, 4)}
                               for s, q in mdl.top_scorelines(home, away, 3)],
        }
        line_txt = (f"{home} vs {away}\n"
                    f"  model H/D/A {p['home']:.3f}/{p['draw']:.3f}/{p['away']:.3f}"
                    f" | market {mkt[0]:.3f}/{mkt[1]:.3f}/{mkt[2]:.3f}")

        if "totals_line" in cons:
            L = cons["totals_line"]
            t = mdl.predict_totals(home, away, lines=(L,))
            p_over = t[f"over_{L}"]
            m_over = (1/cons["over"]) / (1/cons["over"] + 1/cons["under"])
            ev_over = p_over * cons["over"] - 1
            ev_under = (1 - p_over) * cons["under"] - 1
            entry["totals"] = {
                "line": L, "model_over": round(p_over, 4),
                "market_over": round(m_over, 4),
                "over_odds": cons["over"], "under_odds": cons["under"],
                "model_ev_over": round(ev_over, 4),
                "model_ev_under": round(ev_under, 4),
                "expected_total": round(t["expected_total"], 3),
            }
            best = ("OVER", ev_over) if ev_over > ev_under else ("UNDER", ev_under)
            line_txt += (f"\n  totals {L}: model over {p_over:.3f} vs market "
                         f"{m_over:.3f} | best model-EV: {best[0]} {best[1]:+.3f}")
        reports.append(entry)
        lines.append(line_txt)
        print(line_txt)

    if not reports:
        print("No fixtures with usable odds right now.")
        return

    stamp = datetime.now().strftime("%Y-%m-%d")
    path = Path("reports") / f"{ap.parse_args().league}_report_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "history_through": str(last.date()),
        "staleness_days": age,
        "honesty_check": honesty,
        "matches": reports,
    }, indent=2), encoding="utf-8")
    print(f"\nJSON report: {path}")

    send_telegram_message(f"{cfg['label']} report {stamp}\n{honesty}\n\n"
                          + "\n\n".join(lines))


if __name__ == "__main__":
    main()
