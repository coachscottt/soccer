"""
Daily driver: the article's "triple hybrid" system end-to-end.

    ML ensemble  +  Bookmaker odds  +  Polymarket  ->  divergences
                 -> Claude synthesis -> JSON report / Telegram

Run manually:  python predict_matchday.py  [--refresh] [--demo]
(No scheduler by design - runs are manual only.)

Upcoming fixtures come from football-data.co.uk/fixtures.csv, which
lists forthcoming matches for the main leagues WITH Bet365 odds. If
no fixtures are listed (off-season), --demo predicts the most recent
played matchweek instead so the pipeline can be exercised.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ingest import FootballDataLoader, PolymarketClient
from processing import (DataCleaner, FeatureEngineer, add_odds_features,
                        FootballELO, compute_xg_proxy,
                        compute_fatigue_features, compute_h2h_features,
                        TripleLayerFeatures)
from model import prepare_model_data, build_ensemble, fit_league_models
from output import build_match_report, write_json_report, send_telegram_message

SEASONS = ["2526", "2425", "2324", "2223", "2122", "2021"]
LEAGUES = ["E0", "SP1", "D1"]
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
WINDOW = 5

# football-data team name -> tokens that identify it in Polymarket titles
TEAM_ALIASES = {
    "Man United": ["manchester united", "man united"],
    "Man City": ["manchester city", "man city"],
    "Tottenham": ["tottenham", "spurs"],
    "Nott'm Forest": ["nottingham forest"],
    "Wolves": ["wolverhampton", "wolves"],
    "Ath Madrid": ["atletico madrid", "atletico"],
    "Ath Bilbao": ["athletic bilbao", "athletic club"],
    "Betis": ["real betis", "betis"],
    "Sociedad": ["real sociedad", "sociedad"],
    "Leverkusen": ["bayer leverkusen", "leverkusen"],
    "Dortmund": ["borussia dortmund", "dortmund"],
    "M'gladbach": ["monchengladbach", "gladbach"],
    "Ein Frankfurt": ["eintracht frankfurt", "frankfurt"],
    "Bayern Munich": ["bayern munich", "bayern"],
}


def team_tokens(team: str) -> list[str]:
    return TEAM_ALIASES.get(team, [team.lower()])


# ----------------------------------------------------------------- history

def load_and_process(refresh: bool = False):
    """Full processing chain; returns featured frame + live context
    (fitted ELO, per-team records, cleaned history)."""
    loader = FootballDataLoader(seasons=SEASONS, leagues=LEAGUES,
                                refresh=refresh)
    clean = DataCleaner.clean(loader.load_all())

    # Sequential features (ELO / fatigue / H2H) MUST be computed on the
    # full match history. build_match_features inner-joins and drops rows
    # (e.g. promoted teams' first matches); computing sequential features
    # after that join runs them on a history with holes and creates
    # train/serve skew against the driver's full-history feature builder.
    elo = FootballELO(k=32, home_advantage=65)
    full = elo.compute_elo_features(clean)          # elo now holds current ratings
    full = compute_xg_proxy(full)
    full = compute_fatigue_features(full)
    full = compute_h2h_features(full)

    engineer = FeatureEngineer(window=WINDOW)
    featured = engineer.build_match_features(full)  # row-dropping join LAST
    featured = add_odds_features(featured)

    team_stats = engineer.compute_team_stats(clean)
    return featured, elo, team_stats, clean


# ------------------------------------------------- fixture feature assembly

def latest_team_form(team_stats: pd.DataFrame, team: str,
                     window: int = WINDOW) -> dict | None:
    """Rolling stats over the team's last `window` PLAYED matches.
    (For a future fixture no shift is needed - the fixture itself
    is not in the history.)"""
    rows = team_stats[team_stats["Team"] == team].tail(window)
    if len(rows) < 3:
        return None
    stats_cols = ["GF", "GA", "Shots", "ShotsAgainst", "SoT", "SoTAgainst",
                  "Corners", "CornersAgainst", "Fouls", "FoulsAgainst"]
    out = {f"avg_{c}": rows[c].mean() for c in stats_cols}
    out["Form"] = rows["Points"].mean() if "Points" in rows else np.nan
    return out


def build_fixture_row(fixture: pd.Series, ctx: dict) -> dict | None:
    """Assemble the model feature dict for one upcoming fixture."""
    home, away = fixture["HomeTeam"], fixture["AwayTeam"]
    fix_date = fixture["Date"]
    hist = ctx["history"]

    hf = latest_team_form(ctx["team_stats"], home)
    af = latest_team_form(ctx["team_stats"], away)
    if hf is None or af is None:
        return None

    row = {}
    for feat, val in hf.items():
        row[f"home_{feat}"] = val
    for feat, val in af.items():
        row[f"away_{feat}"] = val
    for feat in list(hf.keys()):
        row[f"diff_{feat}"] = hf[feat] - af[feat]

    # odds features (devig)
    try:
        inv = [1 / fixture["B365H"], 1 / fixture["B365D"], 1 / fixture["B365A"]]
    except (KeyError, ZeroDivisionError, TypeError):
        return None
    total = sum(inv)
    row["odds_prob_H"], row["odds_prob_D"], row["odds_prob_A"] = inv
    row["norm_prob_H"] = inv[0] / total
    row["norm_prob_D"] = inv[1] / total
    row["norm_prob_A"] = inv[2] / total
    row["odds_spread"] = row["norm_prob_H"] - row["norm_prob_A"]

    # ELO (current ratings after full history)
    elo = ctx["elo"]
    r_home, r_away = elo.get_rating(home), elo.get_rating(away)
    e_home = elo.expected_score(r_home + elo.home_advantage, r_away)
    row.update({"elo_home": r_home, "elo_away": r_away,
                "elo_diff": r_home - r_away,
                "elo_expected_home": e_home,
                "elo_expected_away": 1 - e_home})

    # H2H (same math as processing.h2h, >=2 prior meetings)
    prev = hist[((hist["HomeTeam"] == home) & (hist["AwayTeam"] == away)) |
                ((hist["HomeTeam"] == away) & (hist["AwayTeam"] == home))].tail(5)
    if len(prev) >= 2:
        wins = draws = goals = 0
        for _, p in prev.iterrows():
            was_home = p["HomeTeam"] == home
            if p["FTR"] == ("H" if was_home else "A"):
                wins += 1
            elif p["FTR"] == "D":
                draws += 1
            goals += p["FTHG"] + p["FTAG"]
        row["h2h_home_wins"] = wins / len(prev)
        row["h2h_draws"] = draws / len(prev)
        row["h2h_total_goals_avg"] = goals / len(prev)
    else:
        row["h2h_home_wins"] = row["h2h_draws"] = np.nan
        row["h2h_total_goals_avg"] = np.nan

    # fatigue
    for side, team in (("home", home), ("away", away)):
        played = hist[(hist["HomeTeam"] == team) | (hist["AwayTeam"] == team)]
        last = played["Date"].max()
        rest = min((fix_date - last).days, 30) if pd.notna(last) else 14
        row[f"{side}_rest_days"] = rest
        recent = played[(fix_date - played["Date"]).dt.days <= 14]
        row[f"{side}_matches_14d"] = len(recent)
        row[f"{side}_fatigued"] = int(rest <= 3)
    row["rest_advantage"] = row["home_rest_days"] - row["away_rest_days"]
    row["is_midweek"] = int(fix_date.dayofweek in (1, 2))

    return row


# ----------------------------------------------------------- data sources

def fetch_fixtures() -> pd.DataFrame:
    """Upcoming fixtures with odds from football-data.co.uk."""
    try:
        resp = requests.get(FIXTURES_URL, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), on_bad_lines="skip")
    except requests.RequestException as e:
        print(f"  [WARN] fixtures.csv fetch failed: {e}")
        return pd.DataFrame()
    if df.empty or "Div" not in df.columns:
        return pd.DataFrame()
    df = df[df["Div"].isin(LEAGUES)].dropna(subset=["HomeTeam", "AwayTeam"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"])


def match_polymarket(home: str, away: str, markets: list[dict],
                     client: PolymarketClient) -> dict | None:
    """Find a Polymarket 1X2 for the fixture by token-matching the
    event title, then classifying each market in the event."""
    h_tokens, a_tokens = team_tokens(home), team_tokens(away)
    events: dict[str, list[dict]] = {}
    for m in markets:
        title = (m.get("event_title") or "").lower()
        if any(t in title for t in h_tokens) and any(t in title for t in a_tokens):
            events.setdefault(m.get("event_slug", ""), []).append(m)
    if not events:
        return None

    # take the event with the most markets (main match event)
    event_markets = max(events.values(), key=len)
    probs, liquidity, volume = {}, 0.0, 0.0
    for m in event_markets:
        odds = client.extract_match_odds(m)
        if odds is None:
            continue
        q = (m.get("question") or "").lower()
        yes = odds.home_win  # binary market: prices[0] = Yes
        if "draw" in q or "tie" in q:
            probs["draw"] = yes
        elif any(t in q for t in h_tokens):
            probs["home"] = yes
        elif any(t in q for t in a_tokens):
            probs["away"] = yes
        liquidity = max(liquidity, odds.liquidity)
        volume = max(volume, odds.volume_24h)

    if {"home", "draw", "away"} <= probs.keys():
        total = sum(probs.values())
        if total > 0:
            return {"probs": {k: v / total for k, v in probs.items()},
                    "liquidity": liquidity, "volume_24h": volume}
    return None


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-download season CSVs (needed in-season)")
    ap.add_argument("--demo", action="store_true",
                    help="if no upcoming fixtures, predict the last played matchweek")
    args = ap.parse_args()

    print("[1/6] Loading history and building features...")
    featured, elo, team_stats, history = load_and_process(refresh=args.refresh)
    X, y, feature_names = prepare_model_data(featured)

    print("[2/6] Training ensemble on full history...")
    ensemble, scaler = build_ensemble(X, y, verbose=False)
    ensemble.fit(scaler.fit_transform(X), y)          # refit on ALL data
    medians = X.median()

    print("[2b/6] Fitting scoreline models (Dixon-Coles, per league)...")
    scoreline_models = fit_league_models(featured, zero_inflation=False,
                                         dc_adjust=True)

    print("[3/6] Fetching upcoming fixtures...")
    fixtures = fetch_fixtures()
    demo_mode = False
    if fixtures.empty:
        if not args.demo:
            print("  No upcoming fixtures listed (off-season?). "
                  "Use --demo to run on the last played matchweek.")
            return
        demo_mode = True
        last_date = featured["Date"].max()
        recent = featured[featured["Date"] >= last_date - pd.Timedelta(days=4)]
        fixtures = recent[["Date", "HomeTeam", "AwayTeam",
                           "B365H", "B365D", "B365A", "League"]].copy()
        print(f"  [demo] predicting last played matchweek: "
              f"{len(fixtures)} matches around {last_date.date()}")

    print("[4/6] Fetching Polymarket football markets...")
    poly_client = PolymarketClient()
    try:
        poly_markets = poly_client.search_football_markets(limit=300)
    except Exception as e:
        print(f"  [WARN] Polymarket unavailable: {e}")
        poly_markets = []

    print("[5/6] Building predictions...")
    ctx = {"team_stats": team_stats, "elo": elo, "history": history}
    reports, matchday_rows = [], []
    for _, fx in fixtures.iterrows():
        row = build_fixture_row(fx, ctx)
        if row is None:
            continue
        feats = pd.DataFrame([row]).reindex(columns=feature_names)
        feats = feats.fillna(medians)
        proba = ensemble.predict_proba(scaler.transform(feats))[0]
        ml = {"home": float(proba[2]), "draw": float(proba[1]),
              "away": float(proba[0])}
        bk = {"home": row["norm_prob_H"], "draw": row["norm_prob_D"],
              "away": row["norm_prob_A"]}

        league_name = FootballDataLoader.LEAGUES.get(
            fx.get("Div"), fx.get("League"))
        scorelines = None
        sl_model = scoreline_models.get(league_name)
        if sl_model is not None:
            scorelines = {
                "poisson_1x2": {k: round(v, 4) for k, v in
                                sl_model.predict_1x2(fx["HomeTeam"],
                                                     fx["AwayTeam"]).items()},
                "totals": {k: round(v, 4) for k, v in
                           sl_model.predict_totals(fx["HomeTeam"],
                                                   fx["AwayTeam"]).items()},
                "top_scorelines": [
                    {"score": s, "prob": round(p, 4)} for s, p in
                    sl_model.top_scorelines(fx["HomeTeam"], fx["AwayTeam"])],
            }

        poly = match_polymarket(fx["HomeTeam"], fx["AwayTeam"],
                                poly_markets, poly_client)
        divergence = None
        if poly:
            divergence = TripleLayerFeatures.compute_divergence_features(
                bk, poly["probs"], ml)

        reports.append(build_match_report(
            home_team=fx["HomeTeam"], away_team=fx["AwayTeam"],
            league=fx.get("League", fx.get("Div", "")),
            kickoff=str(fx["Date"].date()),
            ml_probs={"home_win": ml["home"], "draw": ml["draw"],
                      "away_win": ml["away"]},
            bookmaker_probs={"home_win": bk["home"], "draw": bk["draw"],
                             "away_win": bk["away"]},
            polymarket_probs=(
                {"home_win": poly["probs"]["home"],
                 "draw": poly["probs"]["draw"],
                 "away_win": poly["probs"]["away"]} if poly else None),
            divergence_features=(
                {k: round(float(v), 4) for k, v in divergence.items()}
                if divergence else None),
            scorelines=scorelines,
        ))
        matchday_rows.append({
            "home": fx["HomeTeam"], "away": fx["AwayTeam"],
            "prob_H": ml["home"], "prob_D": ml["draw"], "prob_A": ml["away"],
            "home_form": row.get("home_Form", float("nan")),
            "away_form": row.get("away_Form", float("nan")),
        })

    if not reports:
        print("  No predictable fixtures (teams lack history).")
        return

    # Claude matchday synthesis - only if a key is configured
    claude_summary = None
    if os.environ.get("ANTHROPIC_API_KEY") or _env_has_key():
        try:
            from interpretation import analyze_matchday
            print("  Generating Claude matchday analysis...")
            claude_summary = analyze_matchday(matchday_rows)
        except Exception as e:
            print(f"  [WARN] Claude analysis failed: {e}")

    print("[6/6] Writing outputs...")
    path = write_json_report(reports)
    print(f"  JSON report: {path}")

    lines = [f"Football predictions - {datetime.now():%Y-%m-%d}"
             + (" (DEMO: last played matchweek)" if demo_mode else "")]
    for r in reports:
        p = r["probabilities"]["ml_model"]
        lines.append(f"{r['match']}: H {p['home_win']:.0%} / "
                     f"D {p['draw']:.0%} / A {p['away_win']:.0%}"
                     + (f"  [max div {r['divergence']['max_divergence']:.2f}]"
                        if r.get("divergence") else ""))
    if claude_summary:
        lines.append("\n" + claude_summary)
    summary = "\n".join(lines)
    send_telegram_message(summary)

    poly_n = sum(1 for r in reports if r["probabilities"]["polymarket"])
    print(f"\nDone: {len(reports)} predictions "
          f"({poly_n} with Polymarket coverage). ")


def _env_has_key() -> bool:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


if __name__ == "__main__":
    main()
