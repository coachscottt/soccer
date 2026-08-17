# Football Prediction System — Model Audit

**Date:** 2026-07-07
**Scope:** full system as built from the zostaff article (data → features → models → fusion → interpretation → outputs), audited on the 5-season snapshot (5,178 matches, EPL/La Liga/Bundesliga 2020–2025).

## Verdict summary

| Dimension | Verdict | Notes |
|---|---|---|
| Data integrity | **PASS** | 0 duplicates, 15/15 league-seasons complete, 0 impossible odds, overround 2.8–16.7% (sane) |
| Target leakage | **PASS** (after build-time fixes) | 0 post-match columns in features; max feature-target corr 0.431 (market-level) |
| Train/serve consistency | **FAIL → FIXED** | Pipeline-order bug: sequential features computed post-join on holed history. Fixed; skew now 0.000000 |
| Temporal validation | **PASS** | TimeSeriesSplit + walk-forward, both date-sorted, verified monotonic |
| Statistical honesty | **PASS** | Model is significantly *worse* than market (paired t, p=6e-4) — correctly not claiming an edge |
| Calibration | **FLAG** | Overconfident on home favorites (55%→46% actual); ECE 3.7–5.5% per class; isotonic fix recommended |
| Seed stability | **PASS** | 3 seeds: acc 54.2–54.3%, log loss 0.9895–0.9905 |
| Draw prediction | **KNOWN LIMIT** | P(draw) carries ~no discriminative signal (distributions overlap); inherent to the problem |
| Operational | **FLAG** | See ops findings below |

## Audit finding: train/serve skew (fixed this audit)

**Mechanism.** `FeatureEngineer.build_match_features` inner-joins rolling stats and drops rows
where either team lacks 3 prior matches — including every newly promoted team's first
matches *each season* (e.g. Holstein Kiel 2024). The article's pipeline then computed
ELO, fatigue, and H2H **on the filtered frame**, i.e. on a history with holes:
- ELO ratings skipped updates for dropped matches (drift up to ~1%)
- rest days inflated (verified case: Bayern shown 20 days rest, true value 7)
- 14-day congestion undercounted; H2H could miss meetings

The live driver (`predict_matchday.py`) computes from *full* history, so the deployed
model received different features than it trained on, concentrated on fixtures
involving promoted teams.

**Fix.** Sequential features are now computed on full cleaned data first; the
row-dropping join runs last. Verified: rebuilt driver feature rows for 5 holdout
matches match training features exactly (max relative diff 0.000000). Snapshot
regenerated; holdout metrics after fix: acc 53.1%, log loss 0.991 (previous
54.2%/0.991 partly reflected fitting mismeasured features).

## Key measurements (post-fix)

| Metric | Model (ensemble) | Market (devigged Bet365) |
|---|---|---|
| Holdout accuracy | 53.1% | 53.7% |
| Holdout log loss | 0.991 | 0.971 |
| Walk-forward accuracy (4,674 preds) | 51.9% (XGB) | 54.4% |
| Walk-forward log loss | 1.026 (XGB) | 0.966 |
| Per-class Brier (A/D/H) | .195/.189/.206 | .191/.184/.203 |
| Optimal ML weight in ML+market blend | **0.00** (log loss monotonically worsens with ML weight) | — |

Significance: paired t-test on per-match log loss, t=3.45, p=6e-4 — the market is
genuinely better, not noise. The model's value is as an independent divergence
signal, not an edge over the closing line.

## Bug ledger (article code, found during build + audit)

1. `FeatureEngineer`: away-records column assignment crashes (13 vs 12 columns) — fixed.
2. ELO margin multiplier zeroes out all draw updates (log(0+1)=0) — fixed (draw = 1-goal margin).
3. Feature selector included 4 post-match xG columns (target leakage) and silently
   omitted ELO/H2H/rest/midweek features (6 of the model's top-9 by importance) — fixed.
4. Polymarket keyword search: substring match ("epl" ⊂ "replaced") returned politics
   markets; scanning the global feed finds ~0 real football markets — rewritten on
   Polymarket's tag system (soccer tag 100350 + league tags).
5. CLOB order book: `bids[0]` is the *worst* bid; article's spread/midpoint math was
   garbage on live data — fixed with max/min.
6. Two `print(f"...{classification_report(...)}")` calls split an expression across
   f-string literals — syntax errors as published — fixed.
7. `multi_class="multinomial"` removed in sklearn 1.8 — removed (now default).
8. Deprecated model `claude-sonnet-4-20250514` (retires 2026-06) → `claude-sonnet-5`;
   unguarded `message.content[0].text` → refusal-safe extraction; module-level client
   crash without API key → lazy client; `'N/A':.2f` format crash → `_fmt` helper.
9. Pipeline order (this audit): sequential features after row-dropping join → skew. Fixed.
10. Cosmetics/robustness: `✓`/`⭐` crash cp1252 console; `outcomes` JSON-string not
    parsed; radar red/green CVD pair; 4-color list cycling on 5 bars; misleading
    "Correct/Incorrect" calibration labels; dead `all_dates`; wrong return annotation.

## Known limitations (accepted, documented)

- **Draws are unpredictable** from these features — P(draw) ≈ base rate regardless of
  outcome. Consume probabilities, never 1X2 argmax picks.
- **League-only congestion**: cup/European midweek matches invisible to fatigue features.
- **Global median imputation**: whisper of future data in imputation (minor; per-fold
  imputation would be strictly correct).
- **ELO home advantage 65 slightly hot** for these leagues (expected 0.580 vs actual 0.563).
- **Triple-blend weights (40/35/25 ML/Poly/BK) empirically inverted** — optimal ML weight
  vs market alone is 0.00 by log loss. Present as features only; do not act on the blend.
- **Polymarket divergence signal unproven**: plumbing works live, but no backtest evidence
  divergences predict outcomes (historical Polymarket coverage too thin).
- **Claude layer untested live** (no API key on machine yet); xG proxy is a weighted shot
  count, not true xG (corr 0.57 with goals, ~10% hot).

## Operational findings

- **Secrets in OneDrive**: `football/.env` will sync API keys to cloud storage once
  populated. Acceptable for personal use; consider machine-level env vars if uncomfortable.
- **No version control**: the workspace is not a git repo; this build has no history/rollback.
- **Unpinned dependencies**: requirements.txt uses `>=`; a future sklearn/xgboost major
  could break or silently change behavior. Recommend freezing a lockfile.
- **No automated test suite**: all verification is ad-hoc. The audit's leak scan and
  train/serve skew check are natural pytest regression tests.
- **Manual runs only** (by design): `schedule` installed but unwired.
- **Cache discipline**: raw CSVs snapshotted; current-season file needs `--refresh` in-season.

## Recommendations (ranked)

1. ~~Isotonic calibration~~ — **tested 2026-07-07, rejected.** With ~1,000
   calibration matches (260 draws), isotonic overfits (log loss 0.994 → 1.056)
   and Platt scaling is a wash (−0.0005). The ensemble's ~5% ECE is real but
   too small to correct from this much data. Revisit if the dataset grows
   substantially (more leagues/seasons). `model/calibrate.py` retained as the
   measured implementation.
2. **Pin dependencies + git init** — reproducibility floor.
3. **Convert audit checks to pytest** (leak scan, skew check, devig sanity) so future
   changes can't silently reintroduce the fixed bugs.
4. **Extend Polymarket alias map / add fuzzy matching** before next season (14 aliases now).
5. **If pursuing the divergence hypothesis**: start snapshotting Polymarket prices daily
   *now* to accumulate the history a divergence backtest needs.
6. Retune `home_advantage` (~50) and ensemble weights (2,2,1) at next retrain — both
   measured, both minor.
