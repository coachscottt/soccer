import pandas as pd
from pathlib import Path


class FootballDataLoader:
    """
    Historical football match data loader.
    Source: football-data.co.uk
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    LEAGUES = {
        "E0": "Premier League",
        "SP1": "La Liga",
        "D1": "Bundesliga",
        "I1": "Serie A",
        "F1": "Ligue 1",
    }

    COLUMNS_TO_KEEP = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",       # Final score and result
        "HTHG", "HTAG", "HTR",       # Half-time score
        "HS", "AS",                   # Shots
        "HST", "AST",                 # Shots on target
        "HF", "AF",                   # Fouls
        "HC", "AC",                   # Corners
        "HY", "AY",                   # Yellow cards
        "HR", "AR",                   # Red cards
        "B365H", "B365D", "B365A",   # Bet365 odds
    ]

    def __init__(self, seasons: list[str], leagues: list[str] = None,
                 cache_dir: str | Path = None, refresh: bool = False):
        self.seasons = seasons  # format: ["2324", "2223", "2122"]
        self.leagues = leagues or list(self.LEAGUES.keys())
        # Snapshot every download so backtests stay reproducible even if
        # the source CSVs change. refresh=True forces a re-download
        # (needed for the in-progress season, which updates weekly).
        self.cache_dir = Path(cache_dir) if cache_dir else \
            Path(__file__).resolve().parents[1] / "data" / "raw"
        self.refresh = refresh

    def load_season(self, league: str, season: str) -> pd.DataFrame:
        """Load data for a single season and league."""
        url = f"{self.BASE_URL}/{season}/{league}.csv"
        cache_file = self.cache_dir / f"{league}_{season}.csv"
        try:
            if cache_file.exists() and not self.refresh:
                df = pd.read_csv(cache_file, encoding="utf-8", on_bad_lines="skip")
            else:
                df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_file, index=False)
            available_cols = [c for c in self.COLUMNS_TO_KEEP if c in df.columns]
            df = df[available_cols].dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            # football-data.co.uk dates are day-first (dd/mm/yy or dd/mm/yyyy)
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
            df["League"] = self.LEAGUES.get(league, league)
            df["Season"] = season
            return df
        except Exception as e:
            print(f"Error loading {league}/{season}: {e}")
            return pd.DataFrame()

    def load_all(self) -> pd.DataFrame:
        """Load all data for specified leagues and seasons."""
        frames = []
        for league in self.leagues:
            for season in self.seasons:
                df = self.load_season(league, season)
                if not df.empty:
                    frames.append(df)
                    print(f"  [OK] {self.LEAGUES.get(league)}, season {season}: "
                          f"{len(df)} matches")
        result = pd.concat(frames, ignore_index=True)
        print(f"\nTotal loaded: {len(result)} matches")
        return result
