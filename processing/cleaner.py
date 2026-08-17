import pandas as pd


class DataCleaner:
    """Data cleaning and standardization."""

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Standardize date format
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Numeric columns
        numeric_cols = [
            "FTHG", "FTAG", "HTHG", "HTAG",
            "HS", "AS", "HST", "AST",
            "HF", "AF", "HC", "AC",
            "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Encoding result: H=2, D=1, A=0
        result_map = {"H": 2, "D": 1, "A": 0}
        df["Result"] = df["FTR"].map(result_map)
        df = df.dropna(subset=["Result"])
        df["Result"] = df["Result"].astype(int)

        return df
