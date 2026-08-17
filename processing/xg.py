import pandas as pd


def compute_xg_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    xG proxy: expected goals approximation from basic statistics.

    Formula: xG ≈ SoT * conversion_rate + (Shots - SoT) * low_conversion
    where conversion_rate ~ 0.30 for shots on target,
          low_conversion ~ 0.03 for shots off target.

    This is a rough approximation, but it captures
    key information: attacking play quality.

    LEAKAGE WARNING: these columns are computed from the CURRENT
    match's shot statistics — they are post-match quantities.
    Never feed them to a model predicting this same match; only
    lagged/rolling aggregates of them are legal features.
    """
    df = df.copy()

    SOT_CONVERSION = 0.30   # ~30% shots on target = goal (EPL average)
    SHOT_CONVERSION = 0.03  # ~3% shots off target = goal

    if "HST" in df.columns and "HS" in df.columns:
        df["home_xG_proxy"] = (
            df["HST"] * SOT_CONVERSION
            + (df["HS"] - df["HST"]).clip(lower=0) * SHOT_CONVERSION
        )
        df["away_xG_proxy"] = (
            df["AST"] * SOT_CONVERSION
            + (df["AS"] - df["AST"]).clip(lower=0) * SHOT_CONVERSION
        )

        # xG overperformance: actual goals minus expected
        # Positive value = team scores more than it "should"
        # Usually regresses to mean → correction signal
        df["home_xG_overperf"] = df["FTHG"] - df["home_xG_proxy"]
        df["away_xG_overperf"] = df["FTAG"] - df["away_xG_proxy"]

    return df
