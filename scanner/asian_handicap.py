"""Payoff-exact Asian handicap math (sibling of asian_totals).

A handicap `line` is the goal start quoted FOR the selected side; the
bet settles on adjusted margin d + line where d is the perspective
margin (home: hg-ag, away: ag-hg). Quarter lines (x.25 / x.75, either
sign) split the stake onto the two neighbouring half/integer lines
(line -/+ 0.25), giving half-wins and half-losses; integer lines push
at exactly d == -line. Same discipline as asian_totals: never treat a
line as x.5 when it isn't.
"""
import numpy as np


def margin_dist(m):
    """Home-margin (hg - ag) distribution from a scoreline matrix.
    Index d + G for d in [-G, +G]. Away perspective = dist[::-1]."""
    size = m.shape[0]
    dist = np.zeros(2 * size - 1)
    for i in range(size):
        for j in range(size):
            dist[i - j + size - 1] += m[i, j]
    return dist


def _cat(adj):
    return "w" if adj > 1e-9 else ("p" if abs(adj) < 1e-9 else "l")


def _lines(line):
    q4 = int(round(line * 4))
    if q4 % 2:                       # x.25 / x.75: two half-stakes
        return [(q4 - 1) / 4.0, (q4 + 1) / 4.0]
    return [line]


def hcp_outcome_probs(dist, line):
    """(win, half_win, push, half_loss, loss) probabilities for the
    perspective side receiving `line`; dist = perspective margin dist."""
    G = (len(dist) - 1) // 2
    halves = _lines(line)
    out = {"w": 0.0, "hw": 0.0, "p": 0.0, "hl": 0.0, "l": 0.0}
    for i, pr in enumerate(dist):
        if pr <= 0:
            continue
        cats = {_cat((i - G) + h) for h in halves}
        if cats == {"w"}:
            c = "w"
        elif cats == {"l"}:
            c = "l"
        elif cats == {"p"}:
            c = "p"
        elif cats == {"w", "p"}:
            c = "hw"
        else:                        # {"p", "l"}; {"w","l"} impossible
            c = "hl"
        out[c] += float(pr)
    return out["w"], out["hw"], out["p"], out["hl"], out["l"]


def hcp_ev(dist, line, price):
    w, hw, _, hl, lo = hcp_outcome_probs(dist, line)
    return w * (price - 1) + hw * (price - 1) / 2 - hl / 2 - lo


def hcp_fair_price(dist, line):
    w, hw, _, hl, lo = hcp_outcome_probs(dist, line)
    denom = w + hw / 2
    return 1 + (hl / 2 + lo) / denom if denom > 1e-9 else None


def hcp_result_pnl(d, line, price):
    """Exact 1u P/L given the final perspective margin d."""
    total = 0.0
    halves = _lines(line)
    for h in halves:
        c = _cat(d + h)
        total += {"w": price - 1, "p": 0.0, "l": -1.0}[c] / len(halves)
    return total
