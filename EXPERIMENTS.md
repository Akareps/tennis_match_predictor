# Rebuilt experiment sequence

The original top-level scripts remain as a research archive. New work lives in
the `tennis_experiments` package and is run one immutable stage at a time. Each
stage has its own output directory and refuses to replace an existing run
unless `--overwrite` is supplied deliberately.

## Run and test

Use Python 3.10 or newer from the project directory:

```powershell
python -m unittest discover -s tests -v
python run_experiments.py linkage --tour both
python run_experiments.py validation --tour both
python run_experiments.py holdout --tour both
python run_experiments.py price-bound --tour both
python run_experiments.py market-snapshot --tour both
python run_experiments.py cross-tier-validation --tour both
python run_experiments.py cross-tier-holdout --tour both
python run_experiments.py workload-validation --tour both
python run_experiments.py workload-holdout --tour both
python run_experiments.py event-conditions-validation --tour both
python run_experiments.py event-conditions-holdout --tour both
```

## Stages and information boundaries

1. `00_linkage` links unordered player pairs using identity, tournament,
   round, surface, and date context. Outcomes, rankings, scores, and prices do
   not choose a link. It records an actual match date and a reason code for
   every unlinked row.
2. `01_validation` evaluates 39 Elo configurations on expanding 2022 and 2023
   folds. The final 2024 period is never passed to selection.
3. `02_holdout` evaluates the saved choices once on 2024, excludes retirements
   from rating updates and scoring, and compares every model on the same
   complete-Pinnacle cohort.
4. `03_price_bound` holds each positive-EV selection fixed and settles that
   same side at Pinnacle, Bet365, and the ex-post `MaxW/MaxL` quote. The maximum
   quote is an optimistic diagnostic, not a claim about executable returns.
5. `04_market_snapshot` audits a frozen, free seven-day Odds Gap export. It
   verifies timestamp, player orientation, changing scheduled starts, and
   fixed 24h/6h/1h snapshots. The export has no outcomes, so this stage makes
   no accuracy, edge, or profitability claim.
6. `05_cross_tier_validation` selects fractional evidence weights for
   qualifying, Challenger/WTA125, and Futures/ITF histories using only complete
   pre-2024 folds. The primary selection score is cold-start main-tour log
   loss.
7. `06_cross_tier_holdout` loads only 2021–2023 lower-tier results, keeps the
   2024 main-tour cohort unchanged, and evaluates cumulative lower-tier
   ablations once.
8. `07_workload_validation` builds causal recovery, recent-load, same-event
   load, and prior-availability features. Models train on out-of-sample 2022
   predictions, select shrinkage on 2023, and do not load 2024.
9. `08_workload_holdout` refits the locked residual models on the combined
   2022–2023 out-of-sample panels and evaluates the unchanged 2024 Pinnacle
   cohort once.
10. `09_event_conditions_validation` uses 2021 only to warm up player/surface
    profiles, trains direct score models on 2022, and selects on 2023. It loads
    no 2024 file. Conditions are calculated from completed conventional matches
    in strictly earlier rounds of the same event, leaving out matches involving
    either target player.
11. `10_event_conditions_holdout` refits the locked specification on 2022–2023
    and runs a forward 2024 benchmark. The condition residual has no intercept:
    where fewer than eight earlier point-stat matches are available, its
    prediction is exactly the baseline prediction.

Every stage writes a `manifest.json` containing input SHA-256 hashes, the code
digest at execution time, and the exact configuration. Row-level predictions,
metrics, fitted parameters, audits, and selections are separate artifacts.

## Free data added

### Lower-tier match histories

Twelve qualifying, Challenger/WTA125, Futures, and ITF files for 2021–2024
were frozen from the pinned
[Aneeshers Sackmann archive](https://github.com/Aneeshers/tennis-sackmann-archive/tree/83733587353df8a41f2fd4f516147d5aa83f5a8d).
They contain 233,451 raw rows. Exact hashes and attribution are in
`data/lower_tier_sources.json`.

The source license is CC BY-NC-SA 4.0: this material is for non-commercial
research, attribution is required, and redistributed derivatives are
ShareAlike. Cleaning removes 428 duplicate WTA rows, two duplicate ATP rows,
and one impossible WTA self-match while retaining a row-level audit.

### Contemporary market snapshot

The frozen [Odds Gap research export](https://theoddsgap.com/data) contains
216,408 rows, 231 canonical matches, 442 snapshot times, and 16 books over
2026-08-28 through 2026-09-04. Its SHA-256 is
`3275fff3a9849e13d4285807a00c30fcf1d358935af120768d1de2df9a453783`.
The source permits research, modeling, and writing with attribution, but not
resale, feed repackaging, raw redistribution, or sustained scraping.

Anonymous retention is only seven days and this snapshot has no outcomes,
stable source match ID, exchange depth, or traded volume. It is useful for
ingestion and timing tests, not a defensible performance study.

### Source deliberately skipped

Polymarket documents free public event and trade APIs requiring no account or
API key. Both official data hosts timed out from this environment. No proxy,
account-gated route, paid source, or unverified bulk mirror was substituted.
The opening-to-close experiment remains pending until a legitimate public
source is directly reachable.

## Results

### Repaired benchmark

| Tour | Holdout matches | Best model/horizon | Model log loss | Pinnacle log loss | Gap (model - market) |
|---|---:|---|---:|---:|---:|
| ATP | 2,442 | rank warm-start, round | 0.61124 | 0.57776 | +0.03348 |
| WTA | 2,191 | rank warm-start, round | 0.61746 | 0.58443 | +0.03303 |

Tournament-block bootstrap intervals for both gaps stay above zero. The
optimistic maximum-price replay also remains negative for every model and
horizon.

### Cross-tier history

| Tour | Locked addition | Base log loss | Added-history log loss | Delta | 95% block-bootstrap interval |
|---|---|---:|---:|---:|---:|
| ATP | qualifying at 0.5 evidence | 0.61124 | 0.61012 | -0.00112 | [-0.00273, +0.00045] |
| WTA | qualifying + WTA125 at 1.0 | 0.61746 | 0.61593 | -0.00153 | [-0.00745, +0.00449] |

Futures/ITF evidence received zero validation weight on both tours. Challenger
evidence received zero ATP weight. The selected changes point in a useful
direction, but neither interval excludes zero and both models remain clearly
behind Pinnacle.

### Workload and recent availability

The residual uses the selected cross-tier probability as a fixed log-odds
offset. Cumulative steps add recovery, 14-day load, load from completed earlier
rounds in the same event, and prior retirement/default/walkover history.

| Tour | Locked 2024 model | Base log loss | Workload log loss | Delta | 95% block-bootstrap interval | Pinnacle |
|---|---|---:|---:|---:|---:|---:|
| ATP | all feature groups | 0.61012 | 0.60784 | -0.00228 | [-0.00768, +0.00299] | 0.57776 |
| WTA | all feature groups | 0.61593 | 0.61298 | -0.00295 | [-0.00900, +0.00323] | 0.58443 |

The direction repeated on both tours. Same-event load supplied the largest
marginal ATP step, while recovery supplied the largest WTA step; the final
changes are still not statistically decisive. ATP
cold-start matches became worse (`+0.00508` log loss versus the cross-tier
base), while established ATP matches improved (`-0.00495`); the corresponding
intervals still include zero. This is evidence for further refinement, not a
trading edge.

Flat-stake positive-EV replay also remains negative: -7.13% ATP and -9.34% WTA
for the locked workload models. Those figures do not model historical spread,
liquidity, commission, or order fills.

Detailed results are in:

- `results/05_cross_tier_validation/{atp,wta}`
- `results/06_cross_tier_holdout/{atp,wta}`
- `results/07_workload_validation/{atp,wta}`
- `results/08_workload_holdout/{atp,wta}`

### Dynamic event conditions and score outcomes

This experiment uses only the already-frozen free main-tour match files. It
targets completed conventional best-of-three matches and excludes team ties,
Olympic events, NextGen/Laver formats, retirements, malformed scores, and every
event containing bracketed match tiebreaks. The primary target is over 22.5
games; the secondary target is whether a deciding set was played.

Player/surface serve, return, ace, hold, and score expectations are frozen for
an entire published tournament start date. Same-event residuals are revealed
one whole round at a time and exclude earlier matches involving either target
player. This produced 2,146 ATP and 2,445 WTA benchmark targets in 2024. Event
conditions were established from at least eight eligible earlier matches for
1,207 ATP and 1,299 WTA targets.

Pre-2024 validation locked the player/context baseline for both ATP targets.
WTA validation locked the full serve-plus-score condition model for both
targets:

| Tour | Target | Context baseline | Locked condition model | Validation delta |
|---|---|---:|---:|---:|
| ATP | Over 22.5 | 0.68636 | baseline retained | 0.00000 |
| ATP | Deciding set | 0.65673 | baseline retained | 0.00000 |
| WTA | Over 22.5 | 0.66413 | 0.66356 | -0.00057 |
| WTA | Deciding set | 0.63991 | 0.63979 | -0.00012 |

The locked WTA changes did not provide a clear repeat in the 2024 forward
benchmark:

| Target | Baseline | Locked model | Delta | 95% event-bootstrap interval |
|---|---:|---:|---:|---:|
| Over 22.5 | 0.66718 | 0.66758 | +0.00040 | [-0.00066, +0.00144] |
| Deciding set | 0.64739 | 0.64718 | -0.00021 | [-0.00085, +0.00035] |

Neither interval excludes zero. On the established-evidence subset, the locked
WTA over-22.5 model was also worse by `+0.00075`; deciding set improved by
`-0.00040`, again with intervals spanning zero. Because ATP validation retained
the baseline, its condition variants in 2024 are diagnostics only and are not
promoted as results.

The mechanism check is more encouraging than the final forecasts. Earlier ATP
event residuals improved later-match binomial log loss for ace rate
(`-0.00064`), service-points-won rate (`-0.00023`), and hold rate (`-0.00118`),
with all three tournament-bootstrap intervals below zero. WTA ace conditions
also persisted (`-0.00038`, interval below zero), while WTA service-points-won
and hold intervals included zero. In short: event conditions are measurable,
but this direct totals/deciding-set specification does not turn them into a
reliable forecasting gain.

Detailed results are in:

- `results/09_event_conditions_validation/{atp,wta}`
- `results/10_event_conditions_holdout/{atp,wta}`

## Important limitations

- Sackmann `tourney_date` is an event start date, not a match timestamp.
  Workload chronology therefore treats a whole same-date round as
  simultaneous and only uses completed earlier rounds from the same event.
- The 14-day windows use prior event-start dates. They cannot distinguish a
  Sunday final from a Monday first round exactly.
- Duration is incomplete. Missing-minute counts are explicit model features;
  score-derived games and match counts retain partial load information.
- Retirement history identifies the recorded losing player and is only a
  structured availability proxy, not a diagnosis of injury.
- The workload validation diagnostics reuse the cross-tier configuration
  selected on all pre-2024 folds. Only the 2024 result is a fully untouched
  downstream test.
- The paired bootstrap is by tournament/event block. All uncertainty intervals
  should be read together with the predeclared feature sequence and unchanged
  common cohort.
- The 2024 period is now a reused forward benchmark. It is still excluded from
  fitting and selection inside Stage 09, but earlier experiments and hypothesis
  generation have examined it, so it is not a pristine confirmatory holdout.
- No historical totals or set-market prices are available in the free files.
  Stages 09–10 therefore assess forecast accuracy only and make no edge, ROI,
  or profitability claim.

## Guardrails covered by tests

- outcome-neutral player ordering and price orientation;
- unique match identities and audited source deduplication;
- whole-round simultaneous prediction and update;
- strict draw-time, round-time, and linked-date boundaries;
- retirement handling and loser-only availability events;
- causal workload snapshots with no within-batch visibility;
- one-to-one, outcome-neutral odds linkage;
- complete common cohorts and paired tournament bootstraps;
- expanding folds that cannot touch the final test period;
- finite, regularized offset-logistic fits that retain base Elo log-odds;
- strict conventional score parsing and event-level alternative-format
  exclusion;
- same-date frozen player profiles and same-round frozen event state;
- leave-target-players-out condition aggregates and exact neutral predictions
  below the established-evidence threshold.

The next defensible no-cost direction is to test whether measured event speed
interacts with pre-event player styles in the moneyline model. A raw symmetric
court-speed value cannot enter a player-A win model by itself; it must interact
with an A-minus-B serve/return style difference. Weather remains a later option
once tournament locations and usable match timestamps are mapped reliably.
