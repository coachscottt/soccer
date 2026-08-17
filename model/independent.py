"""Independent fair-value engine v1 - NO market inputs.

Learns from the stats.db warehouse only:
  Team layer:   decay-weighted Dixon-Coles attack/defence per league,
                fit on results.
  Player layer: starting-XI quality index per fixture (each starter's
                prior expanding mean rating; leak-safe), converted to
                lambda/mu adjustments via elasticities fit by Poisson MLE.

Output per fixture: independent lambda/mu -> scoreline matrix -> own
fair 1X2 / totals / BTTS. The market appears nowhere in the inputs;
it is only the judge (see eval_independent.py).
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from model.poisson import ZIPDixonColes

DB = Path(__file__).resolve().parents[1] / "data" / "stats.db"


def load_league_xg(conn, league_id: int) -> pd.DataFrame:
    """load_league + per-team match xG (team_match_stats) as
    xg_home / xg_away columns (NaN where not synced)."""
    fx = load_league(conn, league_id)
    ts = pd.read_sql("SELECT fixture_id, team_id, xg FROM team_match_stats",
                     conn)
    fx = fx.merge(ts.rename(columns={"team_id": "home_id", "xg": "xg_home"}),
                  on=["fixture_id", "home_id"], how="left")
    fx = fx.merge(ts.rename(columns={"team_id": "away_id", "xg": "xg_away"}),
                  on=["fixture_id", "away_id"], how="left")
    return fx


def load_league(conn, league_id: int) -> pd.DataFrame:
    fx = pd.read_sql(
        """SELECT fixture_id, kickoff_utc, home_id, away_id,
                  home_goals, away_goals
           FROM fixtures WHERE league_id=? AND status='FT'""",
        conn, params=(league_id,))
    fx["Date"] = pd.to_datetime(fx["kickoff_utc"], utc=True,
                                format="ISO8601").dt.tz_convert(None)
    names = dict(conn.execute("SELECT team_id, name FROM teams").fetchall())
    fx["HomeTeam"] = fx["home_id"].map(names)
    fx["AwayTeam"] = fx["away_id"].map(names)
    fx = fx.rename(columns={"home_goals": "FTHG", "away_goals": "FTAG"})
    return fx.dropna(subset=["FTHG", "FTAG"]).sort_values("Date")


def xi_quality(conn, league_id: int) -> pd.DataFrame:
    """Per (fixture, team): starting-XI mean prior rating, leak-safe."""
    pms = pd.read_sql(
        """SELECT s.fixture_id, s.team_id, s.player_id, s.rating,
                  f.kickoff_utc
           FROM player_match_stats s JOIN fixtures f USING(fixture_id)
           WHERE f.league_id=? AND f.status='FT'""",
        conn, params=(league_id,))
    lineups = pd.read_sql(
        """SELECT l.fixture_id, l.team_id, l.player_id
           FROM lineups l JOIN fixtures f USING(fixture_id)
           WHERE f.league_id=? AND f.status='FT' AND l.is_starter=1""",
        conn, params=(league_id,))
    pms["Date"] = pd.to_datetime(pms["kickoff_utc"], utc=True,
                                 format="ISO8601").dt.tz_convert(None)
    pms = pms.sort_values("Date")
    pms["prior_n"] = pms.groupby("player_id").cumcount()
    pms["prior_rating"] = (pms.groupby("player_id")["rating"]
                           .transform(lambda s: s.shift(1).expanding().mean()))
    pms.loc[pms["prior_n"] < 3, "prior_rating"] = np.nan
    prior = pms.set_index(["fixture_id", "player_id"])["prior_rating"]
    xi = lineups.join(prior, on=["fixture_id", "player_id"])
    agg = (xi.groupby(["fixture_id", "team_id"])["prior_rating"]
             .agg(xi_rating="mean", xi_rated="count").reset_index())
    # centre within league so the index is a *relative* quality signal
    agg.loc[agg["xi_rated"] < 7, "xi_rating"] = np.nan
    agg["xi_rating"] = agg["xi_rating"] - agg["xi_rating"].mean()
    return agg


class IndependentEngine:
    """One league. fit() on history strictly before predictions."""

    def __init__(self, decay_xi: float = 0.0018, xg_alpha: float = 1.0):
        # xg_alpha=1.0 is the swept optimum (big-3 2025/26): strengths
        # from xG only; real goals enter via the DC rho refit alone.
        self.decay_xi = decay_xi
        self.xg_alpha = xg_alpha     # 0 = goals-only (v1); >0 = xG blend (v2)
        self.dc: ZIPDixonColes | None = None
        self.elasticity = 0.0        # goals multiplier per XI-rating point
        self.xi_table: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame, xi: pd.DataFrame):
        a = self.xg_alpha
        if a > 0 and "xg_home" in matches.columns:
            # strengths on xG-blended pseudo-goals (dc_adjust off: the
            # tau cells need integers); rho refit on real goals below
            blend = matches.copy()
            blend["FTHG"] = np.where(blend["xg_home"].notna(),
                                     a * blend["xg_home"]
                                     + (1 - a) * blend["FTHG"],
                                     blend["FTHG"])
            blend["FTAG"] = np.where(blend["xg_away"].notna(),
                                     a * blend["xg_away"]
                                     + (1 - a) * blend["FTAG"],
                                     blend["FTAG"])
            self.dc = ZIPDixonColes(zero_inflation=False, dc_adjust=False,
                                    decay_xi=self.decay_xi).fit(blend)
            self.dc.params["rho"] = self._fit_rho(matches)
        else:
            self.dc = ZIPDixonColes(zero_inflation=False, dc_adjust=True,
                                    decay_xi=self.decay_xi).fit(matches)
        self.xi_table = xi

        # elasticity: how much does XI quality shift scoring, beyond the
        # team's baseline strength? Poisson MLE on a single parameter.
        m = matches.merge(
            xi.rename(columns={"team_id": "home_id",
                               "xi_rating": "xi_home"})[
                ["fixture_id", "home_id", "xi_home"]],
            on=["fixture_id", "home_id"], how="left").merge(
            xi.rename(columns={"team_id": "away_id",
                               "xi_rating": "xi_away"})[
                ["fixture_id", "away_id", "xi_away"]],
            on=["fixture_id", "away_id"], how="left")
        m = m.dropna(subset=["xi_home", "xi_away"])
        if len(m) < 200:
            return self
        lam = np.array([self.dc.rates(h, a)[0]
                        for h, a in zip(m["HomeTeam"], m["AwayTeam"])])
        mu = np.array([self.dc.rates(h, a)[1]
                       for h, a in zip(m["HomeTeam"], m["AwayTeam"])])
        hg, ag = m["FTHG"].values, m["FTAG"].values
        dxh = (m["xi_home"] - m["xi_away"]).values

        def nll(k):
            lh = lam * np.exp(k * dxh)
            la = mu * np.exp(-k * dxh)
            return -(hg * np.log(lh) - lh + ag * np.log(la) - la).sum()

        self.elasticity = float(minimize_scalar(
            nll, bounds=(-2.0, 2.0), method="bounded").x)
        return self

    def _fit_rho(self, matches: pd.DataFrame) -> float:
        """DC dependence refit on REAL (integer) goals with strengths
        held fixed from the blended fit."""
        hg = matches["FTHG"].astype(int).values
        ag = matches["FTAG"].astype(int).values
        lam = np.array([self.dc.rates(h, a)[0] for h, a in
                        zip(matches["HomeTeam"], matches["AwayTeam"])])
        mu = np.array([self.dc.rates(h, a)[1] for h, a in
                       zip(matches["HomeTeam"], matches["AwayTeam"])])

        def nll(rho):
            tau = np.ones(len(hg))
            m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            return -np.log(np.clip(tau, 1e-10, None)).sum()

        return float(minimize_scalar(nll, bounds=(-0.2, 0.2),
                                     method="bounded").x)

    def latest_xi(self, team_id: int) -> float:
        rows = self.xi_table[self.xi_table["team_id"] == team_id]
        return float(rows["xi_rating"].iloc[-1]) if len(rows) else 0.0

    def rates(self, home: str, away: str,
              xi_home: float | None = None,
              xi_away: float | None = None) -> tuple[float, float]:
        lam, mu = self.dc.rates(home, away)
        dx = (xi_home or 0.0) - (xi_away or 0.0)
        return lam * np.exp(self.elasticity * dx), \
            mu * np.exp(-self.elasticity * dx)

    def predict_1x2(self, home, away, xi_home=None, xi_away=None) -> dict:
        lam, mu = self.rates(home, away, xi_home, xi_away)
        saved = self.dc.params
        m = self._matrix(lam, mu)
        return {"home": float(np.tril(m, -1).sum()),
                "draw": float(np.trace(m)),
                "away": float(np.triu(m, 1).sum())}

    def _matrix(self, lam, mu):
        from scipy.special import gammaln
        rho = self.dc.params["rho"]
        k = np.arange(self.dc.max_goals + 1)
        ph = np.exp(-lam + k * np.log(lam) - gammaln(k + 1))
        pa = np.exp(-mu + k * np.log(mu) - gammaln(k + 1))
        mtx = np.outer(ph, pa)
        mtx[0, 0] *= max(1 - lam * mu * rho, 1e-10)
        mtx[0, 1] *= max(1 + lam * rho, 1e-10)
        mtx[1, 0] *= max(1 + mu * rho, 1e-10)
        mtx[1, 1] *= max(1 - rho, 1e-10)
        return mtx / mtx.sum()
