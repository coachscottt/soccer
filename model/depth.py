"""v3 depth layer: team style profiles + style-interaction residuals.

Style profiles (rolling decay-weighted, strictly pre-match) from the
warehouse:
  sot90_for / sot90_ag   shots on target for/against
  sot_pct                finishing territory quality (SoT / shots)
  poss                   possession %
  passes90 / pass_acc    volume + accuracy of build-up
  direct                 directness: shots per 100 passes
  ppda                   pressing proxy: opp passes / (our tackles+
                         interceptions+fouls)   (lower = higher press)
  corners90              territory pressure

Residual model: log-rate adjustment on top of the xG-DC baseline,
  log lam = log lam_base + beta . phi(home_style, away_style)
fit by L2-regularized Poisson MLE on actual goals, walk-forward safe.
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HALF_LIFE_DAYS = 240
STYLE_COLS = ["sot90_for", "sot90_ag", "sot_pct", "poss", "passes90",
              "pass_acc", "direct", "ppda", "corners90"]


def team_style_features(conn, league_id: int) -> pd.DataFrame:
    """Per (fixture, team): decay-weighted PRIOR style profile."""
    fx = pd.read_sql(
        """SELECT fixture_id, kickoff_utc, home_id, away_id
           FROM fixtures WHERE league_id=? AND status='FT'""",
        conn, params=(league_id,))
    fx["Date"] = pd.to_datetime(fx["kickoff_utc"], utc=True,
                                format="ISO8601").dt.tz_convert(None)
    ts = pd.read_sql(
        """SELECT t.fixture_id, t.team_id, t.shots, t.shots_on,
                  t.possession, t.corners
           FROM team_match_stats t JOIN fixtures f USING(fixture_id)
           WHERE f.league_id=? AND f.status='FT'""",
        conn, params=(league_id,))
    pl = pd.read_sql(
        """SELECT s.fixture_id, s.team_id,
                  SUM(s.passes) passes, SUM(s.passes*s.pass_pct/100.0) cmp,
                  SUM(COALESCE(s.tackles,0) + COALESCE(s.interceptions,0)
                      + COALESCE(s.fouls_committed,0)) defact
           FROM player_match_stats s JOIN fixtures f USING(fixture_id)
           WHERE f.league_id=? AND f.status='FT'
           GROUP BY s.fixture_id, s.team_id""",
        conn, params=(league_id,))

    m = ts.merge(pl, on=["fixture_id", "team_id"], how="left")
    # attach opponent's per-match numbers
    opp = m.rename(columns={c: f"opp_{c}" for c in m.columns
                            if c not in ("fixture_id",)})
    both = m.merge(opp, on="fixture_id")
    both = both[both["team_id"] != both["opp_team_id"]]
    both = both.merge(fx[["fixture_id", "Date"]], on="fixture_id")
    both = both.sort_values("Date")

    # per-match raw style observations
    both["o_sot_for"] = both["shots_on"]
    both["o_sot_ag"] = both["opp_shots_on"]
    both["o_sot_pct"] = both["shots_on"] / both["shots"].clip(lower=1)
    both["o_poss"] = both["possession"]
    both["o_passes"] = both["passes"]
    both["o_pass_acc"] = both["cmp"] / both["passes"].clip(lower=1)
    both["o_direct"] = 100 * both["shots"] / both["passes"].clip(lower=1)
    both["o_ppda"] = both["opp_passes"] / both["defact"].clip(lower=1)
    both["o_corners"] = both["corners"]

    lam = np.log(2) / HALF_LIFE_DAYS
    out_rows = []
    obs_cols = ["o_sot_for", "o_sot_ag", "o_sot_pct", "o_poss",
                "o_passes", "o_pass_acc", "o_direct", "o_ppda",
                "o_corners"]
    for tid, g in both.groupby("team_id"):
        g = g.sort_values("Date")
        vals = g[obs_cols].values
        dates = g["Date"].values.astype("datetime64[D]").astype(float)
        n = len(g)
        prior = np.full((n, len(obs_cols)), np.nan)
        for i in range(n):
            if i < 5:
                continue
            w = np.exp(-lam * (dates[i] - dates[:i]))
            v = vals[:i]
            ok = ~np.isnan(v)
            ww = np.where(ok, w[:, None], 0.0)
            with np.errstate(invalid="ignore"):
                prior[i] = np.nansum(ww * np.nan_to_num(v), axis=0) / \
                    ww.sum(axis=0).clip(min=1e-9)
        block = pd.DataFrame(prior, columns=STYLE_COLS, index=g.index)
        block["fixture_id"] = g["fixture_id"].values
        block["team_id"] = tid
        out_rows.append(block)
    return pd.concat(out_rows, ignore_index=True)


def build_phi(hs: np.ndarray, as_: np.ndarray, mean, std) -> np.ndarray:
    """Feature map: z-scored own/opp styles + key interactions."""
    hz = (hs - mean) / std
    az = (as_ - mean) / std
    # interactions chosen for football logic:
    #  press vs build-up:  home ppda(z, low=press) x away pass_acc
    #  directness vs possession clash, territory pressure vs leakiness
    inter = np.column_stack([
        hz[:, 7] * az[:, 5],        # home press x away pass accuracy
        az[:, 7] * hz[:, 5],        # away press x home pass accuracy
        hz[:, 6] * az[:, 1],        # home directness x away SoT conceded
        az[:, 6] * hz[:, 1],
        hz[:, 3] - az[:, 3],        # possession clash
    ])
    return np.column_stack([hz, az, inter])


class StyleResidual:
    """L2 Poisson residual on top of baseline rates.
    inter_only=True restricts to the 5 matchup-interaction terms -
    the full 23-dim map overfit ~700-match windows (judged 2026-07-23)."""

    def __init__(self, l2: float = 3.0, inter_only: bool = False):
        self.l2 = l2
        self.inter_only = inter_only
        self.beta_h = None
        self.beta_a = None
        self.mean = None
        self.std = None

    def fit(self, lam_base, mu_base, hg, ag, home_style, away_style):
        ok = ~(np.isnan(home_style).any(1) | np.isnan(away_style).any(1))
        lam_base, mu_base = lam_base[ok], mu_base[ok]
        hg, ag = hg[ok], ag[ok]
        hs, as_ = home_style[ok], away_style[ok]
        allv = np.vstack([hs, as_])
        self.mean, self.std = allv.mean(0), allv.std(0).clip(min=1e-6)
        X_h = build_phi(hs, as_, self.mean, self.std)
        X_a = build_phi(as_, hs, self.mean, self.std)
        if self.inter_only:
            X_h, X_a = X_h[:, 18:], X_a[:, 18:]

        def make_nll(X, base, goals):
            def nll(b):
                eta = np.clip(X @ b, -1.5, 1.5)
                rate = base * np.exp(eta)
                return -(goals * np.log(rate) - rate).sum() \
                    + self.l2 * (b @ b)
            return nll

        k = X_h.shape[1]
        self.beta_h = minimize(make_nll(X_h, lam_base, hg), np.zeros(k),
                               method="L-BFGS-B").x
        self.beta_a = minimize(make_nll(X_a, mu_base, ag), np.zeros(k),
                               method="L-BFGS-B").x
        return self

    def adjust(self, lam, mu, home_style, away_style):
        if (self.beta_h is None or np.isnan(home_style).any()
                or np.isnan(away_style).any()):
            return lam, mu
        hs = home_style.reshape(1, -1)
        as_ = away_style.reshape(1, -1)
        Xh = build_phi(hs, as_, self.mean, self.std)
        Xa = build_phi(as_, hs, self.mean, self.std)
        if self.inter_only:
            Xh, Xa = Xh[:, 18:], Xa[:, 18:]
        eh = np.clip(Xh @ self.beta_h, -1.5, 1.5)[0]
        ea = np.clip(Xa @ self.beta_a, -1.5, 1.5)[0]
        return lam * np.exp(eh), mu * np.exp(ea)
