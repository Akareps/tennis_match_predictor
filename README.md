# Tennis Match Prediction — Final Project Report

A multi-iteration empirical investigation into whether amateur quantitative
modeling can beat sharp tennis betting markets, with a focus on ATP and WTA
main-tour singles using publicly available match data.

**Final verdict:** No, with a clear empirical reason why.
The model's log-loss gap to Pinnacle's closing prices is structurally
~+0.04 nats on both ATP and WTA, stable across eleven architectural variants.
This is the value of the proprietary information sharp markets price in.

---

## Table of contents

1. [Quick start](#quick-start)
2. [What's in the repo](#whats-in-the-repo)
3. [The headline results](#the-headline-results)
4. [What we tried, in order](#what-we-tried-in-order)
5. [What we learned](#what-we-learned)
6. [Where edge might still exist](#where-edge-might-still-exist)
7. [Limitations and known issues](#limitations-and-known-issues)

---

## Quick start

```bash
pip install pandas numpy scipy scikit-learn openpyxl

# The current best ATP model. Auto-downloads Sackmann match data and
# tennis-data.co.uk Pinnacle odds on first run, caches both locally.
python3 rank_warmstart.py

# The same architecture on WTA.
python3 wta_warmstart.py
```

Each script outputs a `*_predictions.csv` and prints a comparison vs the
Pinnacle closing line, broken down by surface, tier, cold-start status, and
disagreement magnitude.

Both runs take ~5–10 minutes on a modern laptop.

---

## What's in the repo

### Production model

| File | Role |
|---|---|
| `rank_warmstart.py` | **Current best ATP model.** Surface-blended Elo with rank-derived warmstart for new players. Auto-fits rank→rating mapping on pre-test data. Walks forward strictly with no leakage. |
| `wta_warmstart.py` | Same architecture for WTA, with adjusted tier-code dictionary. |

### Core infrastructure

| File | Role |
|---|---|
| `data_loader.py` | Auto-downloads & parses Sackmann ATP/WTA CSVs from GitHub. Long-format conversion (one row per player-match perspective). |
| `odds_loader.py` | Auto-downloads tennis-data.co.uk Excel files. Sackmann↔tennis-data name fuzzy-matching (Djokovic N. ↔ Novak Djokovic). Vig removal. |
| `markov.py` | Point-by-point probability math: closed-form game prob, DP for tiebreak/set, analytic match. Used by score-market modules. |
| `predict.py` | Match prediction utilities: skill→point→match, Kelly sizing, value detection. |
| `backtest.py` | Walk-forward harness with weekly skill-table refits, log-loss + calibration metrics. |
| `calibration.py` | Symmetric isotonic regression for post-hoc probability calibration. |
| `skill_estimation.py` | Joint regression for serve/return skills (used by older score-market path). |

### Diagnostics & analysis

| File | Role |
|---|---|
| `disagreement_audit.py` | Slices model-vs-Pinnacle gap by tier, surface, favorite-strength, and disagreement magnitude. Bootstrap CIs. |
| `score_parser.py` | Parses Sackmann score strings (`"6-4 3-6 7-5"`) into structured set data. |
| `score_markets.py` | Derives P(2-0), P(2-1), P(over X.5 games), set-count distributions from per-point probs. |
| `score_backtest.py` | Backtest variant that retains spw_a, spw_b, and actual scores. |
| `score_market_eval.py` | Evaluates derived markets: log-loss vs base-rate baseline. |

### Removed / deprecated experiments (negative results documented in [What we tried](#what-we-tried-in-order))

`enhanced_elo.py`, `glicko_warmstart.py`, `underdog_fix.py`,
`empirical_calibration.py`, `style_diagnostic.py`, `sweep.py`,
`shrinkage_diagnostic.py`, `elo_baseline.py`, `ensemble.py`,
`run_pipeline_v3.py`, `synthetic_data.py`.

These were superseded or empirically rejected and have been removed.

---

## The headline results

All numbers are out-of-sample, walk-forward, against Pinnacle closing odds for
2024 ATP and WTA matches. Pinnacle vig: ~3% on both tours.

```
Architecture                          ATP gap to Pinnacle    WTA gap to Pinnacle
-------------------------------------------------------------------------------------
Random guess (uniform 0.5)              +0.1100               +0.1130
Base regression + Markov chain          +0.0438               (not measured)
Surface-blended Elo                     +0.0438               +0.0420
Rank-warmstart Elo (production)         +0.0417               +0.0420
Glicko-1 with warmstart                 +0.0795               (not measured)
Pinnacle closing                        0.0000                0.0000
```

The gap stayed at ~+0.04 nats across every variant we tried, on both tours.
Pinnacle is reliably ~0.04 nats better than our best model.

### What that gap means in practice

- Pinnacle's vig is ~3%; matching Pinnacle's accuracy still loses 3% per bet.
- A +0.04 nat gap to Pinnacle plus 3% vig means our model loses ~5–7% per bet
  across thousands of matches.
- This is consistent with what the value-bet simulations showed throughout
  the project: predicted +10% edges that resolved to actual −10 to −20% ROI.

### What we did beat

Our model is well-calibrated against actual outcomes within the matches we
predict (calibration table mostly within ±5pp). It significantly beats the
uniform 0.5 baseline (improvement ~0.05 nats). It correctly identifies
favorites in close matches at rates competitive with Pinnacle. It's just not
*better* than Pinnacle anywhere.

---

## What we tried, in order

A condensed log of architectures explored, and what we learned from each.

### v1 — Regression-based skill model + Markov chain
Joint regression for serve & return skills per surface, recency-weighted, with
ridge shrinkage. Per-point probs feed a Markov chain to derive match probs.
**Result on synthetic data:** 74% accuracy, well-calibrated.
**On real ATP data:** 63% accuracy, log-loss 0.64, systematically overconfident
on favorites.

### v2 — Match-level shrinkage (α=0.20)
Shrunk match probs toward 0.5 by 20%. Improved aggregate log-loss to 0.63.
Looked like a fix; later turned out to mask a deeper miscalibration.

### v3 — Ridge sweep + Pinnacle calibration
Tried `ridge ∈ {1, 2, 3, 5, 8, 12}`. Log-loss flat across all values
(measurement bias from self-defined buckets). Calibration to Pinnacle made
things worse than the α=0.20 fix because the calibrator was learning on
already-shrunken data.

### v4 — Elo ensemble
Plain Elo and surface-blended Elo, ensembled with the regression model.
Best ensemble weights: 35% regression / 65% Elo. Confirmed that **a 50-line Elo
beats the 600-line regression model** (0.62 vs 0.64 log-loss). The regression
architecture was hurting more than helping.

### v5 — Score market evaluation
Tested whether the model has signal in derived markets (P(over 21.5 games),
P(match goes 3 sets), exact set scores). **Result:** dominance markets (≥1 set,
2-0) work; match-length markets show essentially no signal beyond base rate.
The i.i.d.-points Markov chain produces match-length distributions that don't
match reality at all — it predicted 84% of matches go over 21.5 games when the
true rate is 53%. Empirical recalibration via logistic regression couldn't
recover signal because the underlying features don't carry it.

### v6 — Loaded Pinnacle odds; head-to-head
First clean evaluation against bookmaker closing prices. Gap +0.034 nats.
Value-bet simulation showed the "fake CLV" pattern: model thinks +10% edge per
bet, actual ROI −13%. The model was confidently disagreeing with Pinnacle in
matches where Pinnacle was right ~100% of the time.

### v7 — Disagreement audit
Sliced the model-vs-Pinnacle gap by every dimension we had. Findings:
- Pinnacle picked the winner in **20/20** of our biggest disagreements
- Model is competitive (CI crossing zero) only in close matches at Masters tier
- Heavy favorite slice (Pinnacle 75%+) had the worst gap
- Pattern: Pinnacle has information advantage specifically on lopsided matches

### v8 — Tournament + H2H + variance features
Added tournament-specific Elo adjustments (the "Tsitsipas at AO" idea) and
head-to-head residual adjustments. **Both made things worse** (combined effect
−0.014 nats). Tournament shrinkage was insufficient; H2H was active on only
5% of matches but still added noise on the other 95%.

### v9 — Style clustering diagnostic
Clustered players on aggregate match stats. **Result:** 36.8% cluster-stability
across train/test windows (1.5x random baseline, but two-thirds of players
moved clusters). Aggregate stats aren't fine-grained enough for stable style
identification. Would need shot-level data (Match Charting Project) for this
to work.

### v10 — Rank warmstart
Targeted the cold-start failure mode identified in v7. New players get an
initial rating derived from their ATP/WTA rank (empirically fit rank→rating
mapping); blend fades over 30 matches.
**Result:** Cold-start log-loss improved by 0.014 nats. Big-disagreement count
dropped from 43 to 29 (−33%). Upper-bucket calibration improved. **This became
the production model.**

### v11 — WTA pivot
Same architecture on WTA. **Result:** gap to Pinnacle +0.0420, essentially
identical to ATP's +0.0417. The "WTA is softer" hypothesis was not supported.

### v12 — Glicko-1 (final attempt)
Replaced Elo with Glicko-1 for proper uncertainty quantification. **Result:**
gap nearly doubled to +0.080. Glicko's "shrink predictions toward 0.5 when
opponent has high RD" logic backfires when Pinnacle has information we don't.
Worse, not better. Confirmed we'd hit the structural ceiling.

---

## What we learned

Six findings, listed roughly in order of how much they reshaped my thinking
during the project.

### 1. CLV without calibration is misleading

The most important finding for anyone planning to do this. Closing-line value
is the standard metric for "do I have edge," but it's only valid if your
model is well-calibrated. An overconfident model produces "positive CLV" by
disagreeing with sharp prices — but those disagreements are model error, not
edge. We saw this repeatedly: every value-bet threshold showed positive CLV,
every threshold lost money.

### 2. The information-set gap is structural, not architectural

The +0.04 nat gap to Pinnacle didn't move meaningfully across eleven
architectures spanning from pure regression to pure Elo to Glicko, with
ensembling, calibration, feature engineering, and rank warmstart layered in.
Pinnacle has access to lower-tier results, current form, injury info, and
sharp-money flow that public match data doesn't capture.

### 3. Simpler models beat complex ones on this data

50 lines of Elo beat 600 lines of skill-regression + Markov-chain. Adding
features (tournament adjustments, H2H, Glicko uncertainty) consistently made
results worse. The data is too noisy and the sample sizes per feature cell
are too small for sophisticated architectures to pay off.

### 4. Score-derivative markets aren't actually exploitable from public data

A common claim in the betting literature is that bookmakers use top-down
formulas for derivative markets, creating exploitable mispricing. We tested
this directly. The match-winner model has signal; the match-length and
totals models have essentially zero signal beyond the base rate. The
i.i.d.-points assumption fails badly enough that even empirical recalibration
can't recover edge.

### 5. Cold-start matches are where you bleed out

35% of matches involve at least one player with <30 prior observations. These
matches account for a disproportionate share of the worst model errors. Rank
warmstart helps but doesn't close the gap — Pinnacle has signal beyond rank.

### 6. The "Tennis is unpredictable" framing is wrong

Tennis isn't unpredictable. Pinnacle predicts it at log-loss ~0.58, and
Pinnacle is only one prediction system among many sharp ones. The market is
*efficient*, which is different. Tennis predictability is high; the
*marginal* predictability beyond what's already priced into the closing line
is essentially zero for a public-data model.

---

## Where edge might still exist

Genuinely worth exploring; we didn't.

- **Live betting.** Books update slowly during play. A model with accurate
  per-state win probabilities could find edge in moments where the line lags
  reality. Different infrastructure (live odds streams, low-latency execution).
- **Soft retail books** (Bet365, DraftKings, FanDuel). They post wider lines
  than Pinnacle and update slower. Even at parity with sharp prices, you
  might beat retail. Requires multi-book odds collection.
- **Lower-tier tours** (Challenger, ITF). Sackmann publishes Challenger data;
  Pinnacle's Challenger lines are thin or absent. Soft books carry them
  inconsistently. Possibly less efficient, but data is also worse.
- **Specific prop markets** that bookmakers thin out (live first-set winner,
  next-game winner, individual game total points). Less liquid, less heavily
  shaped by sharp money.

What's *not* worth more time on this exact architecture:

- More features on the existing model
- More calibration techniques
- More ensemble variants
- Different tree-based methods (XGBoost, LightGBM)
- Different Elo K-factor schedules

We've tested enough variants in this family to know the ceiling.

---

## Limitations and known issues

- Test window: 2024 only (~1400 matches per tour after odds matching).
  Statistical noise is meaningful. A gap of +0.04 nats has roughly ±0.005
  bootstrap CI on this sample. The qualitative findings (no edge, cold-start
  is worst slice) are robust; specific numbers should be treated as estimates.
- Name-matching between Sackmann and tennis-data.co.uk fails on ~15-20% of
  matches. Unmatched matches are dropped. There's no evidence of selection
  bias from this drop, but it could exist.
- The i.i.d.-points assumption underlying the Markov chain is empirically
  wrong (within-match correlation, momentum, pressure-point effects exist).
  Score-market predictions in particular are unreliable because of this.
- Skill estimates use only main-tour data. Players coming up from Challengers
  or returning from injury are systematically misestimated.
- The H2H and tournament-adjustment features that didn't work might work with
  proper hierarchical Bayesian shrinkage (we used simple empirical shrinkage).
  Out of scope for this project.

---

## Final note

This project is paused indefinitely. The negative result is the result.

If you come back to it: the architecture in `rank_warmstart.py` is sound and
calibrated. Point it at a less efficient market (live, soft books, lower
tiers) and the same model that fails on ATP/WTA Pinnacle moneyline might
succeed there. The infrastructure is reusable; the data sources need to
change.
