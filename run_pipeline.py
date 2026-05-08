"""
Main entry point. Runs the full pipeline:
  1. Load data (synthetic for offline dev; switch to data_loader for real).
  2. Walk-forward backtest.
  3. Print summary.

Run: python3 run_pipeline.py
"""

import pandas as pd

from backtest import backtest, evaluate
from data_loader import load_years, to_long_format  # Use these for real Sackmann data


def main():
    print("=" * 60)
    print("TENNIS BETTING MODEL — v1 Pipeline")
    print("=" * 60)

    # --- Option A: synthetic data (always available, deterministic) ---
    print("\n[1/3] Loading data...")
    # long, _truth = generate_synthetic_matches(n_players=150, n_matches=15000, seed=2024)
    # print(f"      {len(long)//2} matches, {long['date'].min().date()} to {long['date'].max().date()}")

    # --- Option B: real Sackmann data (uncomment when network allows) ---
    raw = load_years([2021, 2022, 2023, 2024], tour="atp")
    long = to_long_format(raw)
    print(f"      {len(long)//2} matches loaded from Sackmann")

    print("\n[2/3] Running walk-forward backtest...")
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2024-09-30")
    results = backtest(
        long, test_start, test_end,
        refit_freq_days=14,
        half_life_days=365,
        ridge=1.0,
        min_obs_per_player=10,
    )
    print(f"      {len(results)} predictions made.")

    print("\n[3/3] Evaluating...")
    metrics = evaluate(results)
    print()
    print(f"  N predicted matches: {metrics['n_matches']}")
    print(f"  Accuracy:            {metrics['accuracy']:.1%}")
    print(f"  Log-loss (model):    {metrics['logloss_model']:.4f}")
    print(f"  Log-loss (baseline): {metrics['logloss_baseline']:.4f}")
    print(f"  Improvement:         {metrics['logloss_improvement']:.4f} nats")
    print(f"\n  By surface:")
    for s in ("Hard", "Clay", "Grass"):
        sub = results[results["surface"] == s]
        if len(sub) >= 50:
            sub_metrics = evaluate(sub)
            print(f"    {s:6s}: n={sub_metrics['n_matches']:4d}  "
                  f"acc={sub_metrics['accuracy']:.1%}  "
                  f"logloss={sub_metrics['logloss_model']:.4f}")

    print("\n  Calibration (model says X% -> actually wins Y%):")
    for c in metrics["calibration"]:
        gap = c["actual"] - c["predicted"]
        flag = "  " if abs(gap) < 0.05 else " *"
        print(f"    {flag}{c['bucket']}: n={c['n']:4d}  pred={c['predicted']:.3f}  actual={c['actual']:.3f}  gap={gap:+.3f}")

    # Save predictions for further analysis
    out_path = "predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  Predictions saved to: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
