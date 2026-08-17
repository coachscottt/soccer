import numpy as np


class TripleLayerFeatures:
    """
    Combining three probability layers:
    1. Bookmaker (Bet365) — margined odds
    2. Polymarket — blockchain crowd intelligence
    3. ML model — our own estimate

    Divergences between layers are among the most valuable features.
    """

    @staticmethod
    def compute_divergence_features(
        bookmaker_probs: dict,
        polymarket_probs: dict,
        ml_probs: dict | None = None,
    ) -> dict:
        """
        Compute features based on divergences
        between probability sources.

        High divergence may indicate:
        - Insider information on one of the markets
        - One source lagging behind
        - Value bet opportunity
        """
        features = {}

        # === Raw probabilities from each source ===
        for prefix, probs in [("bk", bookmaker_probs),
                               ("poly", polymarket_probs)]:
            features[f"{prefix}_prob_H"] = probs.get("home", 0)
            features[f"{prefix}_prob_D"] = probs.get("draw", 0)
            features[f"{prefix}_prob_A"] = probs.get("away", 0)

        # === KL-divergence between bookmaker and Polymarket ===
        # Higher KL-divergence = stronger disagreement
        epsilon = 1e-6
        kl_div = 0
        for key in ["home", "draw", "away"]:
            p = bookmaker_probs.get(key, epsilon)
            q = polymarket_probs.get(key, epsilon)
            p = max(p, epsilon)
            q = max(q, epsilon)
            kl_div += p * np.log(p / q)
        features["kl_div_bk_poly"] = kl_div

        # === Absolute divergences ===
        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            bk = bookmaker_probs.get(key, 0)
            poly = polymarket_probs.get(key, 0)
            features[f"divergence_{label}"] = bk - poly
            features[f"abs_divergence_{label}"] = abs(bk - poly)

        # === Maximum divergence (across any outcome) ===
        features["max_divergence"] = max(
            features["abs_divergence_H"],
            features["abs_divergence_D"],
            features["abs_divergence_A"],
        )

        # === Who is favorite by each source ===
        bk_favorite = max(bookmaker_probs, key=bookmaker_probs.get)
        poly_favorite = max(polymarket_probs, key=polymarket_probs.get)
        features["sources_agree"] = int(bk_favorite == poly_favorite)

        # === Weighted average probabilities ===
        # Polymarket with higher liquidity → higher weight
        for key, label in [("home", "H"), ("draw", "D"), ("away", "A")]:
            bk = bookmaker_probs.get(key, 0)
            poly = polymarket_probs.get(key, 0)
            # 50/50 by default, adjustable
            features[f"blended_prob_{label}"] = 0.5 * bk + 0.5 * poly

        # === If ML probabilities available — triple system ===
        if ml_probs:
            for key, label in [("home", "H"), ("draw", "D"),
                                ("away", "A")]:
                ml = ml_probs.get(key, 0)
                bk = bookmaker_probs.get(key, 0)
                poly = polymarket_probs.get(key, 0)

                features[f"ml_prob_{label}"] = ml
                features[f"ml_vs_bk_{label}"] = ml - bk
                features[f"ml_vs_poly_{label}"] = ml - poly

                # Triple blend: ML=40%, Polymarket=35%, Bookmaker=25%
                features[f"triple_blend_{label}"] = (
                    0.40 * ml + 0.35 * poly + 0.25 * bk
                )

            # All three sources agree?
            ml_favorite = max(ml_probs, key=ml_probs.get)
            features["all_three_agree"] = int(
                bk_favorite == poly_favorite == ml_favorite
            )

        return features

    @staticmethod
    def compute_liquidity_features(orderbook: dict) -> dict:
        """
        Features based on Polymarket liquidity.

        Liquidity depth is a market confidence indicator.
        Narrow spread + deep order book = strong consensus.
        """
        return {
            "poly_spread": orderbook.get("spread", 0),
            "poly_spread_pct": orderbook.get("spread_pct", 0),
            "poly_depth_total": orderbook.get("total_depth", 0),
            "poly_depth_log": np.log1p(
                orderbook.get("total_depth", 0)
            ),
            "poly_imbalance": orderbook.get("imbalance", 0),
            # Binary: liquidity above threshold?
            "poly_liquid_market": int(
                orderbook.get("total_depth", 0) > 5000
            ),
        }
