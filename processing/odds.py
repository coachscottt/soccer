import pandas as pd


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bookmaker odds to probabilities
    and add as features.
    """
    df = df.copy()

    if all(col in df.columns for col in ["B365H", "B365D", "B365A"]):
        # Raw implied probabilities
        df["odds_prob_H"] = 1 / df["B365H"]
        df["odds_prob_D"] = 1 / df["B365D"]
        df["odds_prob_A"] = 1 / df["B365A"]

        # Normalization (removing bookmaker margin)
        total = df["odds_prob_H"] + df["odds_prob_D"] + df["odds_prob_A"]
        df["norm_prob_H"] = df["odds_prob_H"] / total
        df["norm_prob_D"] = df["odds_prob_D"] / total
        df["norm_prob_A"] = df["odds_prob_A"] / total

        # Probability spread (favorite vs underdog)
        df["odds_spread"] = df["norm_prob_H"] - df["norm_prob_A"]

    return df
