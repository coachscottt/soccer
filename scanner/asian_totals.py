"""Payoff-exact Asian totals math.

Lines come in four flavours with different payoff regions:
  x.5     win / loss
  integer win / push (stake back at exactly L) / loss
  x.25    half stake on L (int), half on L+0.5 -> half-loss at L
  x.75    half on L+0.5, half on L+1 -> half-win at L+1
Treating every line as x.5 mispriced integer anchors by ~0.5 goals
(found 2026-07-24: Brno-Sparta line 3.0 -> fake +43% EV unders).
"""
import numpy as np


def total_dist(m):
    """Total-goals distribution from a scoreline matrix."""
    size = m.shape[0]
    dist = np.zeros(2 * size - 1)
    for i in range(size):
        for j in range(size):
            dist[i + j] += m[i, j]
    return dist


def totals_outcome_probs(dist, side, line):
    """(win, half_win, push, half_loss, loss) probabilities."""
    q = round(line * 4) % 4          # 0=int, 1=.25, 2=.5, 3=.75
    L = int(np.floor(line + 1e-9))
    n = len(dist)
    Pge = lambda a: float(dist[min(max(a, 0), n):].sum())
    Ple = lambda b: float(dist[:min(max(b + 1, 0), n)].sum())
    at = lambda k: float(dist[k]) if 0 <= k < n else 0.0
    if q == 2:      # x.5
        w, hw, pu, hl, lo = Pge(L + 1), 0, 0, 0, Ple(L)
    elif q == 0:    # integer: push at L
        w, hw, pu, hl, lo = Pge(L + 1), 0, at(L), 0, Ple(L - 1)
    elif q == 1:    # x.25: half-loss at L
        w, hw, pu, hl, lo = Pge(L + 1), 0, 0, at(L), Ple(L - 1)
    else:           # x.75: half-win at L+1
        w, hw, pu, hl, lo = Pge(L + 2), at(L + 1), 0, 0, Ple(L)
    if side == "under":              # mirror the payoff regions
        w, hw, pu, hl, lo = lo, hl, pu, hw, w
    return w, hw, pu, hl, lo


def totals_ev(dist, side, line, price):
    w, hw, _, hl, lo = totals_outcome_probs(dist, side, line)
    return w * (price - 1) + hw * (price - 1) / 2 - hl / 2 - lo


def totals_fair_price(dist, side, line):
    w, hw, _, hl, lo = totals_outcome_probs(dist, side, line)
    denom = w + hw / 2
    return 1 + (hl / 2 + lo) / denom if denom > 1e-9 else None


def totals_result_pnl(total, side, line, price):
    """Exact 1u P/L for an Asian-style totals bet given the final total."""
    q = round(line * 4) % 4
    L = int(np.floor(line + 1e-9))
    if q == 2:
        res = "w" if total >= L + 1 else "l"
    elif q == 0:
        res = "w" if total >= L + 1 else ("p" if total == L else "l")
    elif q == 1:
        res = "w" if total >= L + 1 else ("hl" if total == L else "l")
    else:
        res = ("w" if total >= L + 2 else
               "hw" if total == L + 1 else "l")
    if side == "under":
        res = {"w": "l", "hw": "hl", "p": "p", "hl": "hw", "l": "w"}[res]
    return {"w": price - 1, "hw": (price - 1) / 2, "p": 0.0,
            "hl": -0.5, "l": -1.0}[res]
