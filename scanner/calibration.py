"""Market-calibrated Dixon-Coles matrix.

Per match: solve (lambda, mu, rho) so the scoreline matrix reproduces
the devigged sharp 1X2 exactly AND prices the anchor totals line to
zero EV (payoff-exact via asian_totals - handles int/quarter lines).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln

from .asian_totals import total_dist, totals_ev

MAX_GOALS = 10


def dc_matrix(lam, mu, rho):
    k = np.arange(MAX_GOALS + 1)
    ph = np.exp(-lam + k * np.log(lam) - gammaln(k + 1))
    pa = np.exp(-mu + k * np.log(mu) - gammaln(k + 1))
    m = np.outer(ph, pa)
    m[0, 0] *= max(1 - lam * mu * rho, 1e-10)
    m[0, 1] *= max(1 + lam * rho, 1e-10)
    m[1, 0] *= max(1 + mu * rho, 1e-10)
    m[1, 1] *= max(1 - rho, 1e-10)
    return m / m.sum()


def calibrated_matrix(p_home, p_away, p_over, over_line, rho0):
    """(matrix, lam, mu, rho, fit_error) or (None,)*5 when the solve
    fails. fit_error = max abs residual — how exactly the DC family
    could reproduce the anchor; logged per match so the ledger can ask
    whether flags from tight fits outperform flags from strained ones."""
    d_over = 1 / p_over

    def resid(theta):
        m = dc_matrix(np.exp(theta[0]), np.exp(theta[1]), theta[2])
        dist = total_dist(m)
        return [np.tril(m, -1).sum() - p_home,
                np.triu(m, 1).sum() - p_away,
                totals_ev(dist, "over", over_line, d_over)]

    sol = least_squares(resid, [np.log(1.4), np.log(1.1), rho0],
                        bounds=([-3, -3, -0.9], [3, 3, 0.9]))
    fit_err = float(max(abs(np.array(resid(sol.x)))))
    if fit_err > 5e-3:
        return None, None, None, None, None
    lam, mu = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
    return dc_matrix(lam, mu, sol.x[2]), lam, mu, float(sol.x[2]), fit_err


def fair_from_matrix(m):
    size = MAX_GOALS + 1
    total = np.zeros(2 * MAX_GOALS + 1)
    for i in range(size):
        for j in range(size):
            total[i + j] += m[i, j]
    return {"over": lambda line: float(total[int(np.ceil(line)):].sum()),
            "btts_yes": float(1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0])}


def devig(prices):
    inv = np.array([1 / p for p in prices])
    return inv / inv.sum()
