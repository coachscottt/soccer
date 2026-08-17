import pandas as pd


def compute_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fatigue features: rest days between matches.

    Less than 3 rest days → significant performance decline.
    Midweek matches (Tue/Wed) after weekends → fatigue.

    Leak-safe: uses only each team's PRIOR match dates. Note that
    the source data is league matches only, so midweek cup/European
    fixtures are invisible to these features.
    """
    df = df.sort_values("Date").copy()

    rest_days_home = []
    rest_days_away = []
    matches_14d_home = []
    matches_14d_away = []
    last_match: dict[str, pd.Timestamp] = {}
    recent: dict[str, list] = {}

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]

        # Rest days for home team
        if home in last_match:
            delta = (date - last_match[home]).days
            rest_days_home.append(min(delta, 30))  # cap at 30
        else:
            rest_days_home.append(14)  # first match → default

        # Rest days for away team
        if away in last_match:
            delta = (date - last_match[away]).days
            rest_days_away.append(min(delta, 30))
        else:
            rest_days_away.append(14)

        # Fixture congestion: league matches in the previous 14 days
        for team, out in ((home, matches_14d_home), (away, matches_14d_away)):
            history = [d for d in recent.get(team, [])
                       if (date - d).days <= 14]
            recent[team] = history
            out.append(len(history))

        last_match[home] = date
        last_match[away] = date
        recent.setdefault(home, []).append(date)
        recent.setdefault(away, []).append(date)

    df["home_rest_days"] = rest_days_home
    df["away_rest_days"] = rest_days_away
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_matches_14d"] = matches_14d_home
    df["away_matches_14d"] = matches_14d_away

    # Binary flags
    df["home_fatigued"] = (df["home_rest_days"] <= 3).astype(int)
    df["away_fatigued"] = (df["away_rest_days"] <= 3).astype(int)

    # Day of week (Tuesday/Wednesday = midweek fixture)
    df["is_midweek"] = df["Date"].dt.dayofweek.isin([1, 2]).astype(int)

    return df
