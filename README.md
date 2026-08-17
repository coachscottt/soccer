# Football Fair-Value System

Fair values + team xG for matches across 10+ live leagues; flags value
vs per-book prices; logs everything and grades it against closing lines
and results. Paper-tracked until the CLV ledger proves categories real.
All runs are manual. Shares the workspace `.venv`.

**Away from the PC?** `SETUP_CHROMEBOOK.md` (one-time Linux setup) and `RUNBOOK_CHROMEBOOK.md` (day-to-day commands, no Linux experience needed).

## Daily-use scripts (run from this directory, venv python)

| Script | Purpose |
|---|---|
| `value_scanner.py --league X` | THE main tool: fairs board + team xG + value flags + full odds snapshot. Leagues: brazil, argentina, mls, ligamx, norway, sweden, korea, china, denmark, czech, scotland, austria, portugal, netherlands, belgium, laliga, championship, epl, seriea, ligue1 (japan from Aug; serbia retired) |
| `value_scanner.py --league X --settle` | After matchdays: ingest scores, grade bets (P/L, CLV incl. Asian push/half lines), two-sided totals report |
| Matchday rhythm | Morning scan flags the slate; a second scan 30-60 min before the first kickoff captures the close (same-day flags grade CLV 0.0 without it); settle after the games |
| `mls_report.py --league X` | Standalone Dixon-Coles model vs market report |
| `predict_matchday.py [--demo]` | Original article pipeline (big-5 European leagues) |
| `sync_all.py` | Refresh the stats warehouse (resumable, cached) |

## Key data & config

| Path | What |
|---|---|
| `data/stats.db` | The warehouse: 28+ leagues — fixtures, events, lineups, player match stats, team xG, plus all scanner tables (`match_fairs`, `indep_fairs`, `odds_snapshots`, `value_spots`, `match_results`) and the `prediction_audit` view (one row per match: anchored + independent fairs, result, Pinnacle close, calibration `fit_error` — example queries in `scanner/database.py`) |
| `data/manual_anchors.json` | Owner-supplied Pinnacle odds for leagues The Odds API lacks (Czech, Serbia) — update each matchweek from screenshots |
| `.env` | API keys (Anthropic, API-Football Ultra; Telegram optional). Never commit/share |
| `reports/` | Per-scan fairs CSVs/JSONs, prediction reports, charts |
| `.lavish/scanner.html` | The visual board (open via lavish-axi) |

## Packages

| Package | Contents |
|---|---|
| `scanner/` | The scanner implementation: `scanner.py` (feeds, anchors, flagging) · `settlement.py` (grading, CLV, ledger) · `calibration.py` (market-calibrated DC matrix) · `asian_totals.py` / `asian_handicap.py` (payoff-exact totals + AH math) · `database.py` (schema). `value_scanner.py` is a thin CLI shim over this |
| `model/` | `poisson.py` Dixon-Coles scoreline model · `independent.py` v2 xG engine (no market inputs) · `ensemble.py`/`train.py`/`backtest.py` article ML · `calibrate.py`, `depth.py` (tested, shelved) |
| `statsdb/` | Warehouse: `schema.py`, `apifootball.py` (client+sync), `teamstats.py` (xG), `statsbomb.py`, `features.py` |
| `ingest/` | football-data.co.uk loader, Polymarket client |
| `processing/` | Article feature pipeline (cleaner, rolling, ELO, odds, fatigue, h2h, triple-layer) |
| `interpretation/` | Claude narrative layer (matchup, divergence, reports, matchday) |
| `output/` | Charts, JSON reports, Telegram |

## Evaluation scripts (the receipts)

`eval_independent.py`, `eval_arena.py`, `eval_totals.py`, `eval_v2.py`,
`eval_v3.py`, `eval_poisson.py`, `exp_lineup.py` — every model claim in
this project was judged walk-forward vs closing lines. Verdicts logged
in project memory and MODEL_AUDIT_2026-07-07.md.

## Method summary

- Anchored fairs: per-match Dixon-Coles calibrated to devigged sharp
  1X2 + totals (payoff-exact for integer/quarter Asian lines). Team xG
  = calibrated rates. Flags = book prices beating fair by >=3% EV.
- Independent fairs: v2 xG-trained engine, zero market inputs, logged
  alongside for the prospective model-vs-market record.
- Discipline: paper only; every flag graded vs close (CLV) and result;
  h2h line-shopping is the best-evidenced category so far.
