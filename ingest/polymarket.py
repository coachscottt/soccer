import requests
import json
import re
import time
from dataclasses import dataclass

import pandas as pd

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class PolymarketOdds:
    """Structure for storing Polymarket probabilities."""
    home_win: float
    draw: float | None  # Some markets are binary (no draw)
    away_win: float
    liquidity: float
    volume_24h: float
    market_slug: str
    last_updated: str


class PolymarketClient:
    """
    Client for fetching sports markets from Polymarket.

    Polymarket Gamma API — public REST API requiring
    no authorization. Limits: ~50 results per request,
    recommended rate limit ~1 req/sec.
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/118.0.0.0 Safari/537.36"
        )
    }

    # Keywords for filtering football markets
    FOOTBALL_KEYWORDS = [
        "soccer", "premier league", "la liga", "bundesliga",
        "serie a", "ligue 1", "champions league", "uefa",
        "manchester", "liverpool", "arsenal", "chelsea",
        "barcelona", "real madrid", "bayern", "psg",
        "epl", "football match",
    ]

    # Gamma tag IDs (GET /tags/slug/<slug>) — the authoritative way
    # to find sports markets; the global /markets feed is dominated
    # by politics and misses most football markets entirely.
    SOCCER_TAG_ID = 100350
    LEAGUE_TAG_IDS = {
        "epl": 306,
        "premier-league": 82,
        "la-liga": 780,
        "bundesliga": 1494,
        "serie-a": 100618,
        "ligue-1": 102070,
        "champions-league": 1234,
        "fifa-world-cup": 102232,
    }

    def search_football_markets(
        self, limit: int = 100, tag_id: int | None = None
    ) -> list[dict]:
        """
        Find active football markets on Polymarket via the soccer
        event tag, returning a flat list of market dicts.
        Falls back to keyword scanning if the tag query fails.
        """
        all_markets = []
        offset = 0
        tag = tag_id or self.SOCCER_TAG_ID

        while offset < limit:
            try:
                resp = requests.get(
                    f"{GAMMA_API}/events",
                    params={
                        "tag_id": tag,
                        "closed": "false",
                        "limit": 50,
                        "offset": offset,
                    },
                    headers=self.HEADERS,
                    timeout=15,
                )
                resp.raise_for_status()
                events = resp.json()

                if not events:
                    break

                for event in events:
                    for market in event.get("markets", []):
                        market["event_title"] = event.get("title", "")
                        market["event_slug"] = event.get("slug", "")
                        all_markets.append(market)

                offset += 50
                time.sleep(0.5)  # Polite rate limiting

            except requests.RequestException as e:
                print(f"  [WARN] Tag query failed ({e}); "
                      "falling back to keyword scan")
                return self.search_football_markets_by_keywords(limit)

        print(f"  [OK] Found {len(all_markets)} football markets")
        return all_markets

    def search_football_markets_by_keywords(
        self, limit: int = 100
    ) -> list[dict]:
        """
        Fallback: scan the global market feed and filter by
        keywords in the market question (the article's original
        approach — prone to missing markets and false positives).
        """
        all_markets = []
        offset = 0

        while offset < limit:
            try:
                resp = requests.get(
                    f"{GAMMA_API}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 50,
                        "offset": offset,
                    },
                    headers=self.HEADERS,
                    timeout=15,
                )
                resp.raise_for_status()
                markets = resp.json()

                if not markets:
                    break

                # Filter by football keywords (word-boundary match:
                # bare "in" matching lets "epl" hit "replaced" etc.)
                for market in markets:
                    question = market.get("question", "").lower()
                    description = market.get("description", "").lower()
                    text = question + " " + description

                    if any(re.search(rf"\b{re.escape(kw)}\b", text)
                           for kw in self.FOOTBALL_KEYWORDS):
                        all_markets.append(market)

                offset += 50
                time.sleep(0.5)  # Polite rate limiting

            except requests.RequestException as e:
                print(f"  [WARN] Request error: {e}")
                break

        print(f"  [OK] Found {len(all_markets)} football markets")
        return all_markets

    def get_event_markets(self, event_slug: str) -> list[dict]:
        """
        Get all markets for a specific event (e.g., a match).

        Polymarket organizes data hierarchically:
        Event -> Markets -> Outcomes
        """
        try:
            resp = requests.get(
                f"{GAMMA_API}/events",
                params={
                    "slug": event_slug,
                    "closed": "false",
                },
                headers=self.HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            events = resp.json()

            if events:
                return events[0].get("markets", [])
            return []

        except requests.RequestException as e:
            print(f"  [WARN] Error: {e}")
            return []

    def extract_match_odds(self, market: dict) -> PolymarketOdds | None:
        """
        Extract probabilities from market data.

        On Polymarket, contract price = implied probability.
        "Yes" price = $0.65 -> 65% probability.
        """
        try:
            outcomes_raw = market.get("outcomes", [])
            prices_raw = market.get("outcomePrices", "[]")

            # The Gamma API JSON-encodes both fields as strings
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw

            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw

            if len(prices) < 2:
                return None

            prices = [float(p) for p in prices]
            outcomes_lower = [str(o).lower() for o in outcomes]

            # Determine market type
            # Option 1: binary market "Team A wins?"
            if len(prices) == 2:
                return PolymarketOdds(
                    home_win=prices[0],
                    draw=None,
                    away_win=prices[1],
                    liquidity=float(market.get("liquidity", 0) or 0),
                    volume_24h=float(market.get("volume24hr", 0) or 0),
                    market_slug=market.get("slug", ""),
                    last_updated=market.get("updatedAt", ""),
                )

            # Option 2: 3-way market (Home / Draw / Away)
            if len(prices) >= 3:
                home_idx = next(
                    (i for i, o in enumerate(outcomes_lower)
                     if "home" in o or "win" in o),
                    0,
                )
                draw_idx = next(
                    (i for i, o in enumerate(outcomes_lower)
                     if "draw" in o or "tie" in o),
                    1,
                )
                away_idx = next(
                    (i for i, o in enumerate(outcomes_lower)
                     if "away" in o or "lose" in o),
                    2,
                )

                return PolymarketOdds(
                    home_win=prices[home_idx],
                    draw=prices[draw_idx],
                    away_win=prices[away_idx],
                    liquidity=float(market.get("liquidity", 0) or 0),
                    volume_24h=float(market.get("volume24hr", 0) or 0),
                    market_slug=market.get("slug", ""),
                    last_updated=market.get("updatedAt", ""),
                )

        except (ValueError, IndexError, KeyError) as e:
            print(f"  [WARN] Failed to extract prices: {e}")

        return None


class PolymarketHistorical:
    """
    Fetching historical prices from Polymarket CLOB API
    for use in backtesting.
    """

    CLOB_API = "https://clob.polymarket.com"

    def get_price_history(
        self, token_id: str, interval: str = "1d",
        fidelity: int = 60,
    ) -> pd.DataFrame:
        """
        Get price history for a specific outcome token.

        Args:
            token_id: Token ID from market data
            interval: time interval ('1d', '1w', '1m', 'all')
            fidelity: granularity in minutes
        """
        try:
            resp = requests.get(
                f"{self.CLOB_API}/prices-history",
                params={
                    "market": token_id,
                    "interval": interval,
                    "fidelity": fidelity,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data or "history" not in data:
                return pd.DataFrame()

            df = pd.DataFrame(data["history"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="s")
            df["price"] = df["p"].astype(float)
            df = df[["timestamp", "price"]].sort_values("timestamp")

            return df

        except requests.RequestException as e:
            print(f"  [WARN] Error fetching history: {e}")
            return pd.DataFrame()

    def get_orderbook_snapshot(self, token_id: str) -> dict:
        """
        Order book snapshot — shows liquidity depth.
        Thin order book = unreliable signal.
        Deep order book = strong market consensus.
        """
        try:
            resp = requests.get(
                f"{self.CLOB_API}/book",
                params={"token_id": token_id},
                timeout=15,
            )
            resp.raise_for_status()
            book = resp.json()

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            total_bid_depth = sum(
                float(b.get("size", 0)) for b in bids
            )
            total_ask_depth = sum(
                float(a.get("size", 0)) for a in asks
            )

            # CLOB sorts away from the touch: best bid is the HIGHEST
            # price, best ask the LOWEST, regardless of array order
            best_bid = max((float(b["price"]) for b in bids), default=0)
            best_ask = min((float(a["price"]) for a in asks), default=1)
            spread = best_ask - best_bid
            midpoint = (best_bid + best_ask) / 2

            return {
                "midpoint": midpoint,
                "spread": spread,
                "spread_pct": spread / midpoint if midpoint > 0 else 0,
                "bid_depth_usd": total_bid_depth,
                "ask_depth_usd": total_ask_depth,
                "total_depth": total_bid_depth + total_ask_depth,
                "imbalance": (
                    (total_bid_depth - total_ask_depth)
                    / (total_bid_depth + total_ask_depth)
                    if (total_bid_depth + total_ask_depth) > 0
                    else 0
                ),
            }

        except (requests.RequestException, IndexError, ValueError):
            return {}
