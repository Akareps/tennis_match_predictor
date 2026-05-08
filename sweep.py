"""
Hyperparameter sweep for the tennis model.

Grid over (ridge, half_life_days). For each combo, runs a walk-forward backtest
and reports log-loss, accuracy, and worst calibration gap.

Key efficiency: reuses the data loading and just re-runs the backtest for each
config. On 4 years of ATP data this should take ~5-15 minutes.

Usage:
    python3 sweep.py

Outputs:
    sweep_results.csv  — full grid with metrics
    Prints best config and recommends next steps.
"""

import itertools
import time
import pandas as pd
import numpy as np
from data_loader import load_years, to_long_format
from backtest import backtest, evaluate


# --- Grid definition ---
# Ridge: how aggressively to shrink skills toward the mean.
#   1.0  = current (overconfident on favorites)
#   3-5  = moderate shrinkage, expected sweet spot
#   10+  = heavy shrinkage, should hurt strong players
RIDGE_GRID = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

# Half-life: how fast to forget old matches.
#   180 = aggressive recency, 6 month effective window
#   365 = current, 1 year half-life
#   730 = slow, uses 2+ years of history
HALF_LIFE_GRID = [180, 365, 730]

# Held-out test period
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-09-30")


def worst_gap(calibration_list):
    """Largest absolute pred-vs-actual gap, only counting buckets with n>=30."""
    if not calibration_list:
        return None
    return max(
        (abs(c["actual"] - c["predicted"]) for c in calibration_list if c["n"] >= 30),
        default=None,
    )


def signed_upper_gap(calibration_list):
    """Mean signed gap in upper buckets (>=0.6); negative = overconfident on favs."""
    upper = [c for c in calibration_list if c["predicted"] >= 0.6 and c["n"] >= 30]
    if not upper:
        return None
    return float(np.mean([c["actual"] - c["predicted"] for c in upper]))


def main():
    print("Loading ATP data 2021-2024...")
    raw = load_years([2021, 2022, 2023, 2024], tour="atp")
    long = to_long_format(raw)
    print(f"  {len(long)//2} matches loaded.\n")

    rows = []
    configs = list(itertools.product(RIDGE_GRID, HALF_LIFE_GRID))
    print(f"Sweeping {len(configs)} configs over {(TEST_END - TEST_START).days} test days...\n")

    for i, (ridge, half_life) in enumerate(configs):
        t0 = time.time()
        try:
            results = backtest(
                long, TEST_START, TEST_END,
                refit_freq_days=14,
                half_life_days=half_life,
                ridge=ridge,
                min_obs_per_player=10,
            )
            metrics = evaluate(results)
            row = {
                "ridge": ridge,
                "half_life": half_life,
                "n": metrics["n_matches"],
                "logloss": metrics["logloss_model"],
                "accuracy": metrics["accuracy"],
                "improvement": metrics["logloss_improvement"],
                "worst_gap": worst_gap(metrics["calibration"]),
                "upper_gap": signed_upper_gap(metrics["calibration"]),
                "elapsed_s": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"  [{i+1:2d}/{len(configs)}] ridge={ridge:>5}  hl={half_life:>4}  "
                  f"logloss={row['logloss']:.4f}  "
                  f"acc={row['accuracy']:.3f}  "
                  f"upper_gap={row['upper_gap']:+.3f}  "
                  f"({row['elapsed_s']}s)")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(configs)}] ridge={ridge}  hl={half_life}  FAILED: {e}")

    df = pd.DataFrame(rows).sort_values("logloss")
    df.to_csv("sweep_results.csv", index=False)

    print("\n" + "=" * 70)
    print("RESULTS (sorted by log-loss, best first)")
    print("=" * 70)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest config: ridge={best['ridge']}, half_life={best['half_life']}")
    print(f"  log-loss = {best['logloss']:.4f}")
    print(f"  accuracy = {best['accuracy']:.3f}")
    print(f"  upper-bucket gap = {best['upper_gap']:+.3f}  (closer to 0 = better calibrated)")

    # Diagnostic: if upper_gap is still very negative across all configs,
    # ridge alone won't fix it.
    if df["upper_gap"].abs().min() > 0.04:
        print("\n  WARNING: even the best config has upper-bucket gap > 4%.")
        print("  Ridge alone is not enough — next step is calibration on top,")
        print("  or per-surface tuning.")
    else:
        print("\n  Upper-bucket calibration is reasonable for the best config.")

    print("\nResults saved to sweep_results.csv")


if __name__ == "__main__":
    main()
