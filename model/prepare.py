import pandas as pd

# Post-match columns that the prefix scan would otherwise pick up.
# These are computed FROM the match being predicted (shots, goals) —
# including them is target leakage and produces fake accuracy.
LEAKY_COLS = [
    "home_xG_proxy", "away_xG_proxy",
    "home_xG_overperf", "away_xG_overperf",
]

# Pre-match features the article's prefix scan misses.
EXTRA_FEATURES = [
    "elo_home", "elo_away", "elo_diff",
    "elo_expected_home", "elo_expected_away",
    "h2h_home_wins", "h2h_draws", "h2h_total_goals_avg",
    "rest_advantage", "is_midweek",
]


def prepare_model_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data: extract features and target variable.
    We use ONLY features available before match start.
    """
    feature_cols = [c for c in df.columns
                    if c.startswith(("home_", "away_", "diff_",
                                     "norm_prob_", "odds_spread"))
                    and c not in LEAKY_COLS]
    feature_cols += [c for c in EXTRA_FEATURES if c in df.columns]

    X = df[feature_cols].copy()
    y = df["Result"].copy()

    # Fill missing values with median
    X = X.fillna(X.median())

    print(f"Features: {X.shape[1]}")
    print(f"Matches: {X.shape[0]}")
    print(f"Class balance: {y.value_counts().to_dict()}")

    return X, y, feature_cols
