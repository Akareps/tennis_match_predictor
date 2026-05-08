"""
Walk-forward backtest framework.

For each match in the test period:
  1. Refit skill table using ONLY data before the match date.
  2. Predict P(winner wins).
  3. Compare to (a) the actual outcome and (b) the bookmaker's implied prob.

Key metrics:
  - Log loss of model vs uniform 0.5 baseline (lower is better).
  - Log loss vs bookmaker (the real benchmark).
  - Calibration: when model says 60%, does it win ~60%?
  - Closing-line value (CLV): does our model agree with where prices closed?
    This is the only reliable signal of edge in small samples.

Refitting the skill table for every match is expensive. We refit weekly
(every 7 days) and reuse for matches in that week. This is standard practice
and reflects how a real betting model would operate.
"""

import numpy as np
import pandas as pd
from typing import Optional
from skill_estimation import fit_skills, SkillTable, _sigmoid
from predict import predict_match


def backtest(
    long_df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    surfaces: tuple = ("Hard", "Clay", "Grass"),
    refit_freq_days: int = 7,
    half_life_days: float = 365.0,
    ridge: float = 1.0,
    min_obs_per_player: int = 10,
    bookmaker_probs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per backtested match containing:
      date, surface, player_a_id, player_b_id, actual (1 if A won),
      model_prob_a, [implied_prob_a if bookmaker_probs supplied]
    """
    # Reduce to one row per match for prediction (the long format has 2 rows
    # per match, one per perspective). Use the player_id < opp_id row to
    # canonicalize "A".
    df = long_df[long_df["player_id"] < long_df["opp_id"]].copy()
    df = df[(df["date"] >= test_start) & (df["date"] < test_end)]
    df = df.sort_values("date").reset_index(drop=True)

    # Build refit dates: every Monday in the test window
    refit_dates = pd.date_range(
        start=test_start - pd.Timedelta(days=refit_freq_days),
        end=test_end,
        freq=f"{refit_freq_days}D",
    )

    # Cache: surface -> {refit_date: SkillTable}
    skill_cache: dict[tuple, SkillTable] = {}

    def get_skill_table(surface, match_date):
        # Find the most recent refit_date <= match_date
        rd = max([d for d in refit_dates if d <= match_date], default=None)
        if rd is None:
            return None
        key = (surface, rd)
        if key not in skill_cache:
            try:
                skill_cache[key] = fit_skills(
                    long_df, surface=surface, as_of=rd,
                    half_life_days=half_life_days, ridge=ridge,
                )
            except ValueError:
                skill_cache[key] = None
        return skill_cache[key]

    results = []
    for i, r in df.iterrows():
        if r["surface"] not in surfaces:
            continue
        st = get_skill_table(r["surface"], r["date"])
        if st is None:
            continue
        # Skip if either player has too few obs in our training window
        if (st.n_obs.get(r["player_id"], 0) < min_obs_per_player or
                st.n_obs.get(r["opp_id"], 0) < min_obs_per_player):
            continue
        pred = predict_match(st, r["player_id"], r["opp_id"], best_of=int(r["best_of"]))
        results.append({
            "date": r["date"],
            "surface": r["surface"],
            "best_of": r["best_of"],
            "player_a_id": r["player_id"],
            "player_b_id": r["opp_id"],
            "actual_a_won": int(r["won"]),
            "model_prob_a": pred["p_match"],
        })

    res_df = pd.DataFrame(results)
    if bookmaker_probs is not None:
        res_df = res_df.merge(bookmaker_probs, on=["date", "player_a_id", "player_b_id"], how="left")
    return res_df


def evaluate(results: pd.DataFrame) -> dict:
    """Compute log-loss, accuracy, calibration metrics."""
    if len(results) == 0:
        return {"n": 0}
    p = np.clip(results["model_prob_a"].values, 1e-6, 1 - 1e-6)
    y = results["actual_a_won"].values
    logloss_model = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    logloss_baseline = -np.log(0.5)  # always predict 0.5

    metrics = {
        "n_matches": len(results),
        "logloss_model": logloss_model,
        "logloss_baseline": logloss_baseline,
        "logloss_improvement": logloss_baseline - logloss_model,
        "accuracy": np.mean((p > 0.5) == (y == 1)),
    }

    if "implied_prob_a" in results.columns and results["implied_prob_a"].notna().any():
        valid = results["implied_prob_a"].notna()
        bp = np.clip(results.loc[valid, "implied_prob_a"].values, 1e-6, 1 - 1e-6)
        by = results.loc[valid, "actual_a_won"].values
        bm_p = np.clip(results.loc[valid, "model_prob_a"].values, 1e-6, 1 - 1e-6)
        metrics["logloss_bookmaker"] = -np.mean(by * np.log(bp) + (1 - by) * np.log(1 - bp))
        metrics["logloss_model_on_bm_subset"] = -np.mean(by * np.log(bm_p) + (1 - by) * np.log(1 - bm_p))
        # CLV signal: model's prob in same direction as bookmaker (correlation)
        metrics["model_bm_corr"] = float(np.corrcoef(bm_p, bp)[0, 1])

    # Calibration buckets
    bins = np.linspace(0, 1, 11)
    bucket = np.digitize(p, bins) - 1
    cal = []
    for b in range(10):
        mask = (bucket == b)
        if mask.sum() >= 20:
            cal.append({
                "bucket": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "actual": float(y[mask].mean()),
            })
    metrics["calibration"] = cal

    return metrics


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_matches

    print("Generating synthetic data (10,000 matches over 3 years)...")
    long, truth = generate_synthetic_matches(n_players=120, n_matches=10000, seed=42)

    # Backtest the last 6 months
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2024-06-30")
    print(f"Backtesting from {test_start.date()} to {test_end.date()}...")
    print(f"  Refitting every 14 days, half-life 365 days, ridge=1.0")

    results = backtest(
        long, test_start, test_end,
        refit_freq_days=14,
        half_life_days=365,
        ridge=1.0,
        min_obs_per_player=8,
    )
    print(f"\nBacktest produced {len(results)} predictions.")

    metrics = evaluate(results)
    print(f"\n=== METRICS ===")
    print(f"N matches:        {metrics['n_matches']}")
    print(f"Accuracy:         {metrics['accuracy']:.3f}")
    print(f"Log-loss model:   {metrics['logloss_model']:.4f}")
    print(f"Log-loss base:    {metrics['logloss_baseline']:.4f}  (uniform 0.5)")
    print(f"Improvement:      {metrics['logloss_improvement']:.4f}")
    print(f"\nCalibration (well-calibrated => predicted ≈ actual):")
    for c in metrics["calibration"]:
        print(f"  {c['bucket']}: n={c['n']:4d}, predicted={c['predicted']:.3f}, actual={c['actual']:.3f}")
