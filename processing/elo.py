import numpy as np
import pandas as pd


class FootballELO:
    """
    ELO ratings for football teams.

    FIFA formula: R_new = R_old + K * M * (S - E)
    where:
      K — match significance coefficient
      M — goal difference multiplier
      S — actual result (1 / 0.5 / 0)
      E — expected result by ELO
    """

    def __init__(self, k: int = 32, home_advantage: int = 65):
        self.k = k
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.setdefault(team, 1500.0)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability of A beating B using the ELO formula."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def margin_multiplier(self, goal_diff: int) -> float:
        """
        Goal difference multiplier.
        A 5-0 win should have more impact than 1-0.
        Formula from FiveThirtyEight: log(|diff| + 1) * 2.2 / (elo_diff * 0.001 + 2.2)
        Simplified version, without the elo_diff correction.
        A draw (diff 0) counts like a 1-goal margin — log(1) = 0 would
        otherwise zero out the S-E update on every draw.
        """
        return np.log(max(abs(goal_diff), 1) + 1)

    def update(self, home: str, away: str,
               home_goals: int, away_goals: int) -> tuple[float, float]:
        """
        Update ratings after a match.
        Returns (new_home_elo, new_away_elo).
        """
        r_home = self.get_rating(home) + self.home_advantage
        r_away = self.get_rating(away)

        e_home = self.expected_score(r_home, r_away)
        e_away = 1.0 - e_home

        # Actual result
        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals < away_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # Goal difference multiplier
        m = self.margin_multiplier(home_goals - away_goals)

        # Update (without home_advantage in stored rating)
        self.ratings[home] += self.k * m * (s_home - e_home)
        self.ratings[away] += self.k * m * (s_away - e_away)

        return self.ratings[home], self.ratings[away]

    def compute_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through all matches chronologically,
        updating ELO after each. For each match we save
        ratings BEFORE it starts (to prevent leakage).
        """
        df = df.sort_values("Date").copy()
        elo_features = []

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            r_home = self.get_rating(home)
            r_away = self.get_rating(away)

            e_home = self.expected_score(
                r_home + self.home_advantage, r_away
            )

            elo_features.append({
                "elo_home": r_home,
                "elo_away": r_away,
                "elo_diff": r_home - r_away,
                "elo_expected_home": e_home,
                "elo_expected_away": 1 - e_home,
            })

            # Update AFTER saving pre-match ratings
            if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
                self.update(home, away,
                            int(row["FTHG"]), int(row["FTAG"]))

        return pd.concat(
            [df.reset_index(drop=True),
             pd.DataFrame(elo_features)],
            axis=1,
        )
