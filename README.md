# Tennis Betting Model — v1

A working pipeline for tennis match probability estimation, built around point-level Markov chain simulation with regression-based skill estimation.

## What's here

```
markov.py            Closed-form / DP probabilities: point -> game -> set -> match
data_loader.py       Downloads & parses Sackmann ATP/WTA match data
synthetic_data.py    Generates realistic fake data for offline development
skill_estimation.py  Joint regression for serve/return skills (ridge, recency-weighted)
predict.py           Predict P(A wins) + value/Kelly utilities
backtest.py          Walk-forward backtest with calibration metrics
run_pipeline.py      Entry point: data -> backtest -> summary
```

## Quick start

```bash
pip install pandas numpy scipy
python3 run_pipeline.py
```

By default this runs on synthetic data (no network needed). To use real ATP data, edit `run_pipeline.py` and uncomment the `data_loader` lines.

## Current results (synthetic data, 3763 matches)

| Metric | Value |
|---|---|
| Accuracy | 74.0% |
| Log-loss (model) | 0.510 |
| Log-loss (baseline) | 0.693 |
| Improvement | 0.183 nats |
| Calibration | Within ±5% in nearly every bucket |

These numbers come from synthetic data where the data-generating process roughly matches our model assumptions, so they're optimistic. **Real-data numbers will be worse, especially log-loss vs bookmaker closing lines.** That comparison is the real benchmark.

## Architecture

### 1. Markov chain (`markov.py`)
Closed-form game probability + DP for tiebreak/set + analytic match prob. Single point-win prob `p` -> hold prob ≈ `p^4 * (...) + deuce stuff`. The win-by-2 tiebreak rule is handled via a closed-form "extension phase" once both players reach `target-1`.

### 2. Skill estimation (`skill_estimation.py`)
A weighted ridge regression on match-level data:

```
logit(spw_i) = serve_skill[player_i] - return_skill[opp_i] + intercept[surface]
```

- One model per surface
- Weights = `service_points × exp(-ln(2) × age_days / half_life)`
- Ridge penalty shrinks unknown players toward the mean
- Recovers true latent skills with Spearman ρ = 0.97 on synthetic data

### 3. Prediction (`predict.py`)
Skill table + two player IDs -> point probs -> match prob via Markov. Plus utilities for converting odds to fair probabilities, finding value, and Kelly sizing.

### 4. Backtest (`backtest.py`)
Walk-forward with weekly refits. Critical: skill table for date `t` only uses matches strictly before `t`. No look-ahead.

## Path forward — what to do next

1. **Get real data.** `data_loader.py` is ready. You'll need to run from a network where `raw.githubusercontent.com` is reachable (this sandbox blocks it). Sackmann's ATP data covers 1968-present.

2. **Add bookmaker odds.** `tennis-data.co.uk` has free historical odds. Get Pinnacle closing odds and merge them in. Then `evaluate()` will show `logloss_bookmaker` and you can see whether your model beats or loses to the closing line.

3. **Tune hyperparameters.** Three knobs matter most:
   - `half_life_days` (try 90, 180, 365, 730)
   - `ridge` (try 0.5, 1, 2, 5)
   - `min_obs_per_player` (try 5, 10, 20)
   Use a held-out validation period (NOT the test period) to pick.

4. **Cross-surface borrowing.** Currently each surface fits independently, which is wasteful. Better: one big regression with `serve_skill[player] + surface_effect_serve[player, surface]`, allowing partial pooling.

5. **Find the inefficient markets.** WTA, lower tiers (Challenger, ITF), live odds, and prop markets (set scores, totals, handicaps) are where bookmakers are weaker. Start there before betting moneyline on ATP main draw.

6. **Walk-forward CLV check.** For 200+ matches, compare your prob to closing prob. If your model consistently moves toward the closing line, you have edge. If not, you don't — and no amount of P&L variance will tell you otherwise in <1000 bets.

## Known limitations

- **i.i.d. point assumption.** Real points have momentum, pressure-point effects, and serve-streak correlation. Beyond v1 this is a known frontier.
- **No fatigue / scheduling.** Players going deep into the previous week aren't penalized.
- **No head-to-head.** Some matchups have stylistic edges not captured by overall skill (heavy topspin vs flat hitter, etc.).
- **No injury / retirement modeling.** Players coming back from layoffs are systematically over-rated.
- **Skills are static within a refit window.** A player who suddenly improves won't be picked up for up to `refit_freq_days` days.
