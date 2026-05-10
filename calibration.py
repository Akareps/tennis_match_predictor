"""
Probability calibration via isotonic regression.

The raw model is systematically overconfident on favorites (real-data
calibration showed -8% gap in upper buckets). Isotonic regression learns
a monotonic mapping from raw_prob -> calibrated_prob using a held-out set.

Usage:
    cal = IsotonicCalibrator()
    cal.fit(raw_probs_train, outcomes_train)
    calibrated_probs = cal.transform(raw_probs_test)

It's monotonic (preserves ordering — favorites stay favorites) but bends
the curve to match observed frequencies. Standard technique for binary
classifiers; works well with a few hundred+ samples.

Includes a "symmetric" variant that enforces f(p) + f(1-p) = 1, which
matters for tennis because we want P(A wins) and P(B wins) to sum to 1.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self, symmetric: bool = True):
        """
        symmetric=True: enforces f(p) + f(1-p) = 1 by training on doubled
        data. Strongly recommended for two-way markets.
        """
        self.symmetric = symmetric
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray):
        raw_probs = np.asarray(raw_probs)
        outcomes = np.asarray(outcomes).astype(float)
        if self.symmetric:
            # For each (p, y), also include (1-p, 1-y)
            p = np.concatenate([raw_probs, 1 - raw_probs])
            y = np.concatenate([outcomes, 1 - outcomes])
            self.iso.fit(p, y)
        else:
            self.iso.fit(raw_probs, outcomes)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self.iso.transform(np.asarray(raw_probs))


def evaluate_calibration(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> dict:
    """Return calibration table + log-loss."""
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    outcomes = np.asarray(outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    bucket = np.digitize(probs, bins) - 1
    bucket = np.clip(bucket, 0, n_bins - 1)
    cal = []
    for b in range(n_bins):
        mask = bucket == b
        if mask.sum() >= 20:
            cal.append({
                "bucket": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(probs[mask].mean()),
                "actual": float(outcomes[mask].mean()),
                "gap": float(outcomes[mask].mean() - probs[mask].mean()),
            })
    logloss = -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    return {"logloss": logloss, "calibration": cal}


def time_split(df: pd.DataFrame, split_frac: float = 0.5) -> tuple:
    """Earlier matches -> calibrator training; later -> evaluation."""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df_sorted) * split_frac)
    return df_sorted.iloc[:cut], df_sorted.iloc[cut:]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/predictions.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df)} predictions from {path}")

    train, test = time_split(df, split_frac=0.5)
    print(f"Train (early): {len(train)} matches, "
          f"{train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Test  (late):  {len(test)} matches, "
          f"{test['date'].min().date()} to {test['date'].max().date()}")

    # --- Before calibration ---
    print("\n=== BEFORE CALIBRATION (test set) ===")
    raw = evaluate_calibration(test["model_prob_a"].values, test["actual_a_won"].values)
    print(f"Log-loss: {raw['logloss']:.4f}")
    print(f"{'bucket':<12}{'n':>6}{'pred':>8}{'actual':>9}{'gap':>9}")
    for c in raw["calibration"]:
        flag = " *" if abs(c["gap"]) > 0.04 else "  "
        print(f"{flag}{c['bucket']:<10}{c['n']:>6}{c['predicted']:>8.3f}"
              f"{c['actual']:>9.3f}{c['gap']:>+9.3f}")

    # --- Fit calibrator on train, apply to test ---
    cal = IsotonicCalibrator(symmetric=True)
    cal.fit(train["model_prob_a"].values, train["actual_a_won"].values)
    test_cal = cal.transform(test["model_prob_a"].values)

    print("\n=== AFTER CALIBRATION (test set) ===")
    after = evaluate_calibration(test_cal, test["actual_a_won"].values)
    print(f"Log-loss: {after['logloss']:.4f}  (was {raw['logloss']:.4f}, "
          f"delta {raw['logloss'] - after['logloss']:+.4f})")
    print(f"{'bucket':<12}{'n':>6}{'pred':>8}{'actual':>9}{'gap':>9}")
    for c in after["calibration"]:
        flag = " *" if abs(c["gap"]) > 0.04 else "  "
        print(f"{flag}{c['bucket']:<10}{c['n']:>6}{c['predicted']:>8.3f}"
              f"{c['actual']:>9.3f}{c['gap']:>+9.3f}")

    # --- Show the learned mapping ---
    print("\n=== LEARNED MAPPING (raw_prob -> calibrated) ===")
    test_pts = np.linspace(0.05, 0.95, 19)
    mapped = cal.transform(test_pts)
    print(f"{'raw':>6}{'cal':>8}{'shrink':>9}")
    for r, m in zip(test_pts, mapped):
        print(f"{r:>6.2f}{m:>8.3f}{m - r:>+9.3f}")
