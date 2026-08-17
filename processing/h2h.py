import numpy as np
import pandas as pd


def compute_h2h_features(df: pd.DataFrame, n_last: int = 5) -> pd.DataFrame:
    """
    Head-to-head statistics between teams.
    Some pairs have persistent patterns
    (e.g., one team historically dominates).
    """
    df = df.sort_values("Date").copy()
    h2h_features = []

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]

        # All previous meetings between these teams
        prev = df[
            (df["Date"] < date)
            & (
                ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
                | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
            )
        ].tail(n_last)

        if len(prev) < 2:
            h2h_features.append({
                "h2h_home_wins": np.nan,
                "h2h_draws": np.nan,
                "h2h_total_goals_avg": np.nan,
            })
            continue

        # Count results from home team's perspective
        home_wins = 0
        draws = 0
        total_goals = 0

        for _, p in prev.iterrows():
            if p["HomeTeam"] == home:
                if p["FTR"] == "H": home_wins += 1
                elif p["FTR"] == "D": draws += 1
                total_goals += p["FTHG"] + p["FTAG"]
            else:  # home team played away
                if p["FTR"] == "A": home_wins += 1
                elif p["FTR"] == "D": draws += 1
                total_goals += p["FTHG"] + p["FTAG"]

        n = len(prev)
        h2h_features.append({
            "h2h_home_wins": home_wins / n,
            "h2h_draws": draws / n,
            "h2h_total_goals_avg": total_goals / n,
        })

    h2h_df = pd.DataFrame(h2h_features, index=df.index)
    return pd.concat([df, h2h_df], axis=1)
