"""
Main pipeline with configurable hyperparameters and optional calibration layer.

After running sweep.py, plug the best (ridge, half_life) here.
The optional calibration step trains an isotonic on a held-out warm-up period
and applies it to the test predictions.
"""

import pandas as pd
import numpy as np
from data_loader import load_years, to_long_format
from backtest import backtest, evaluate
from calibration import IsotonicCalibrator, evaluate_calibration


# --- Configurable from sweep results ---
RIDGE = 3.0          # bumped from 1.0 — adjust based on sweep_results.csv
HALF_LIFE_DAYS = 365 # adjust based on sweep
USE_CALIBRATION = True
CAL_WARMUP_DAYS = 90  # train calibrator on first 90 days of test window,
                      # evaluate on rest. Trade-off: more warmup = better
                      # calibrator, fewer test matches.


def main():
    print("=" * 60)
    print("TENNIS BETTING MODEL — v2 Pipeline")
    print(f"  ridge={RIDGE}  half_life={HALF_LIFE_DAYS}d  cal={USE_CALIBRATION}")
    print("=" * 60)

    print("\n[1/4] Loading ATP data 2021-2024...")
    raw = load_years([2021, 2022, 2023, 2024], tour="atp")
    long = to_long_format(raw)
    print(f"      {len(long)//2} matches.")

    print("\n[2/4] Walk-forward backtest...")
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2024-09-30")
    results = backtest(
        long, test_start, test_end,
        refit_freq_days=14,
        half_life_days=HALF_LIFE_DAYS,
        ridge=RIDGE,
        min_obs_per_player=10,
    )
    print(f"      {len(results)} predictions.")

    if USE_CALIBRATION:
        print(f"\n[3/4] Calibration (train on first {CAL_WARMUP_DAYS} days of test)...")
        warmup_end = test_start + pd.Timedelta(days=CAL_WARMUP_DAYS)
        warmup = results[results["date"] < warmup_end]
        eval_set = results[results["date"] >= warmup_end].copy()
        print(f"      Calibrator train: {len(warmup)} matches "
              f"({warmup['date'].min().date()} to {warmup['date'].max().date()})")
        print(f"      Evaluation set:   {len(eval_set)} matches "
              f"({eval_set['date'].min().date()} to {eval_set['date'].max().date()})")

        # Show before
        print("\n      Before calibration:")
        before = evaluate_calibration(
            eval_set["model_prob_a"].values, eval_set["actual_a_won"].values
        )
        print(f"        log-loss = {before['logloss']:.4f}")

        cal = IsotonicCalibrator(symmetric=True)
        cal.fit(warmup["model_prob_a"].values, warmup["actual_a_won"].values)
        eval_set["model_prob_calibrated"] = cal.transform(eval_set["model_prob_a"].values)

        print("      After calibration:")
        after = evaluate_calibration(
            eval_set["model_prob_calibrated"].values, eval_set["actual_a_won"].values
        )
        print(f"        log-loss = {after['logloss']:.4f}  "
              f"(delta {before['logloss'] - after['logloss']:+.4f})")

        # Use calibrated probs for final reporting
        final_probs = eval_set["model_prob_calibrated"].values
        final_actuals = eval_set["actual_a_won"].values
        final_df = eval_set
        prob_col = "model_prob_calibrated"
    else:
        final_probs = results["model_prob_a"].values
        final_actuals = results["actual_a_won"].values
        final_df = results
        prob_col = "model_prob_a"

    print("\n[4/4] Final metrics...")
    metrics = evaluate_calibration(final_probs, final_actuals)
    print(f"\n  N matches:    {len(final_df)}")
    print(f"  Log-loss:     {metrics['logloss']:.4f}")
    print(f"  Accuracy:     {(np.round(final_probs) == final_actuals).mean():.1%}")

    # By surface
    print(f"\n  By surface:")
    for s in ("Hard", "Clay", "Grass"):
        sub = final_df[final_df["surface"] == s]
        if len(sub) >= 50:
            sm = evaluate_calibration(sub[prob_col].values, sub["actual_a_won"].values)
            acc = (np.round(sub[prob_col].values) == sub["actual_a_won"].values).mean()
            print(f"    {s:6s}: n={len(sub):4d}  acc={acc:.1%}  logloss={sm['logloss']:.4f}")

    print("\n  Calibration:")
    print(f"  {'bucket':<12}{'n':>6}{'pred':>8}{'actual':>9}{'gap':>9}")
    for c in metrics["calibration"]:
        flag = " *" if abs(c["gap"]) > 0.04 else "  "
        print(f"  {flag}{c['bucket']:<10}{c['n']:>6}{c['predicted']:>8.3f}"
              f"{c['actual']:>9.3f}{c['gap']:>+9.3f}")

    out = "predictions_v2.csv"
    final_df.to_csv(out, index=False)
    print(f"\n  Saved to {out}")
    print("Done.")


if __name__ == "__main__":
    main()
