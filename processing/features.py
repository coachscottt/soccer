import pandas as pd


class FeatureEngineer:
    """
    Feature generation based on historical team statistics.
    Key idea: for each match we use ONLY data
    available BEFORE the match starts.
    """

    def __init__(self, window: int = 5):
        self.window = window

    def compute_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling averages for each team
        over the last N matches.
        """
        df = df.sort_values("Date").copy()

        # Create separate records for home and away teams
        home_records = df[["Date", "HomeTeam", "FTHG", "FTAG",
                           "HS", "AS", "HST", "AST",
                           "HC", "AC", "HF", "AF"]].copy()
        home_records.columns = ["Date", "Team", "GF", "GA",
                                "Shots", "ShotsAgainst",
                                "SoT", "SoTAgainst",
                                "Corners", "CornersAgainst",
                                "Fouls", "FoulsAgainst"]
        home_records["IsHome"] = 1

        away_records = df[["Date", "AwayTeam", "FTAG", "FTHG",
                           "AS", "HS", "AST", "HST",
                           "AC", "HC", "AF", "HF"]].copy()
        away_records.columns = ["Date", "Team", "GF", "GA",
                                "Shots", "ShotsAgainst",
                                "SoT", "SoTAgainst",
                                "Corners", "CornersAgainst",
                                "Fouls", "FoulsAgainst"]
        away_records["IsHome"] = 0

        all_records = pd.concat([home_records, away_records])
        all_records = all_records.sort_values("Date")

        # Calculate rolling averages per team
        stats_cols = ["GF", "GA", "Shots", "ShotsAgainst",
                      "SoT", "SoTAgainst", "Corners",
                      "CornersAgainst", "Fouls", "FoulsAgainst"]

        rolling_stats = {}
        for team in all_records["Team"].unique():
            team_data = all_records[all_records["Team"] == team].copy()
            for col in stats_cols:
                # shift(1) — to exclude the current match
                team_data[f"avg_{col}"] = (
                    team_data[col]
                    .shift(1)
                    .rolling(window=self.window, min_periods=3)
                    .mean()
                )
            # Form: average points over last N matches
            team_data["Points"] = team_data.apply(
                lambda r: 3 if r["GF"] > r["GA"]
                         else (1 if r["GF"] == r["GA"] else 0),
                axis=1,
            )
            team_data["Form"] = (
                team_data["Points"]
                .shift(1)
                .rolling(window=self.window, min_periods=3)
                .mean()
            )
            rolling_stats[team] = team_data

        return pd.concat(rolling_stats.values())

    def build_match_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Join home and away team statistics
        for each match.
        """
        team_stats = self.compute_team_stats(df)

        stat_features = [c for c in team_stats.columns if c.startswith("avg_")]
        stat_features.append("Form")

        features_list = []

        for idx, match in df.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]
            date = match["Date"]

            home_stats = team_stats[
                (team_stats["Team"] == home) &
                (team_stats["Date"] == date) &
                (team_stats["IsHome"] == 1)
            ]
            away_stats = team_stats[
                (team_stats["Team"] == away) &
                (team_stats["Date"] == date) &
                (team_stats["IsHome"] == 0)
            ]

            if home_stats.empty or away_stats.empty:
                continue

            row = {"match_idx": idx}
            for feat in stat_features:
                h_val = home_stats[feat].values[0]
                a_val = away_stats[feat].values[0]
                row[f"home_{feat}"] = h_val
                row[f"away_{feat}"] = a_val
                # Difference — one of the strongest features
                row[f"diff_{feat}"] = h_val - a_val

            features_list.append(row)

        features_df = pd.DataFrame(features_list).set_index("match_idx")
        result = df.join(features_df, how="inner")
        return result.dropna(subset=[c for c in features_df.columns])
