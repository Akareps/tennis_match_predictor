"""
Disagreement audit.

Goal: understand WHERE our model and Pinnacle disagree most, and whether
the pattern is systematic enough to identify a sub-segment of ATP where
our model has edge.

Procedure:
  1. For every match, compute |p_model - p_pinnacle|
  2. Slice by tournament tier, surface, time of year, prediction confidence
  3. Compare model vs Pinnacle log-loss in each slice
  4. The KEY question: is there ANY slice where model_logloss < pinnacle_logloss
     AND the gap is statistically robust (large enough sample, consistent pattern)?

Uses enhanced_predictions.csv but only the BASE Elo column (the enhanced
features made things worse).

Outputs:
  audit_report.csv with per-slice metrics
  prints findings with explicit verdicts
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats


def evaluate(p, y):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    y = np.asarray(y).astype(float)
    if len(p) == 0:
        return float("nan")
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_logloss_diff(p_model, p_pinn, y, n_boot=1000, seed=42):
    """
    Estimate confidence interval for (model_logloss - pinnacle_logloss).
    Negative = model better. Returns (mean_diff, ci_low, ci_high, p_value_better).

    p_value_better = fraction of bootstrap samples where model_logloss < pinnacle.
    """
    n = len(p_model)
    if n < 30:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = evaluate(p_model[idx], y[idx]) - evaluate(p_pinn[idx], y[idx])
        diffs.append(d)
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), (diffs < 0).mean()


def slice_metrics(df, slice_name, slice_value, p_model_col="p_a_base_elo"):
    """Compute metrics for a single slice. Returns dict."""
    p = df[p_model_col].values
    pp = df["fair_prob_a"].values
    y = df["actual_a_won"].values
    if len(df) < 20:
        return None
    ll_m = evaluate(p, y)
    ll_p = evaluate(pp, y)
    diff_mean, ci_lo, ci_hi, p_better = bootstrap_logloss_diff(p, pp, y)
    return {
        "slice": slice_name,
        "value": slice_value,
        "n": len(df),
        "model_ll": ll_m,
        "pinn_ll": ll_p,
        "gap": ll_m - ll_p,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "p_better": p_better,    # bootstrap probability that model beats pinn
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/enhanced_predictions.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df)} predictions.")
    df = df[df["fair_prob_a"].notna()].copy()
    print(f"With odds: {len(df)}")

    # We use BASE Elo (the enhanced features made things worse)
    p_col = "p_a_base_elo"
    p_model = df[p_col].values
    p_pinn = df["fair_prob_a"].values
    y = df["actual_a_won"].values

    # Add disagreement column
    df["disagreement"] = np.abs(p_model - p_pinn)
    df["model_minus_pinn"] = p_model - p_pinn

    print("\n" + "=" * 75)
    print("OVERALL STATS")
    print("=" * 75)
    ll_m_all = evaluate(p_model, y)
    ll_p_all = evaluate(p_pinn, y)
    print(f"  N: {len(df)}")
    print(f"  Model log-loss:    {ll_m_all:.4f}")
    print(f"  Pinnacle log-loss: {ll_p_all:.4f}")
    print(f"  Gap (overall):     {ll_m_all - ll_p_all:+.4f}")
    print(f"  Mean |disagreement|: {df['disagreement'].mean():.4f}")
    print(f"  P(disagreement > 10pp): {(df['disagreement'] > 0.10).mean():.3f}")
    print(f"  P(disagreement > 20pp): {(df['disagreement'] > 0.20).mean():.3f}")

    # ============ SLICES ============
    print("\n" + "=" * 75)
    print("SLICE 1: BY TOURNAMENT TIER")
    print("=" * 75)
    print(f"  {'tier':<10}{'n':>5}{'model_ll':>11}{'pinn_ll':>11}{'gap':>10}{'ci_lo':>9}{'ci_hi':>9}{'p_better':>10}")
    print("  " + "-" * 73)
    rows = []
    tier_names = {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP 250/500"}
    for tier in ["G", "M", "A"]:
        sub = df[df["tourney_level"] == tier]
        m = slice_metrics(sub, "tourney_level", tier)
        if m:
            rows.append(m)
            label = f"{tier} ({tier_names.get(tier, tier)})"
            print(f"  {tier:<10}{m['n']:>5}{m['model_ll']:>11.4f}{m['pinn_ll']:>11.4f}"
                  f"{m['gap']:>+10.4f}{m['ci_low']:>+9.4f}{m['ci_high']:>+9.4f}"
                  f"{m['p_better']:>10.3f}")

    print("\n" + "=" * 75)
    print("SLICE 2: BY SURFACE")
    print("=" * 75)
    print(f"  {'surface':<10}{'n':>5}{'model_ll':>11}{'pinn_ll':>11}{'gap':>10}{'ci_lo':>9}{'ci_hi':>9}{'p_better':>10}")
    print("  " + "-" * 73)
    for surf in ["Hard", "Clay", "Grass"]:
        sub = df[df["surface"] == surf]
        m = slice_metrics(sub, "surface", surf)
        if m:
            rows.append(m)
            print(f"  {surf:<10}{m['n']:>5}{m['model_ll']:>11.4f}{m['pinn_ll']:>11.4f}"
                  f"{m['gap']:>+10.4f}{m['ci_low']:>+9.4f}{m['ci_high']:>+9.4f}"
                  f"{m['p_better']:>10.3f}")

    print("\n" + "=" * 75)
    print("SLICE 3: BY TIER × SURFACE")
    print("=" * 75)
    print(f"  {'cell':<22}{'n':>5}{'model_ll':>11}{'pinn_ll':>11}{'gap':>10}{'p_better':>10}")
    print("  " + "-" * 69)
    for tier in ["G", "M", "A"]:
        for surf in ["Hard", "Clay", "Grass"]:
            sub = df[(df["tourney_level"] == tier) & (df["surface"] == surf)]
            m = slice_metrics(sub, f"{tier}/{surf}", "")
            if m and m["n"] >= 30:
                rows.append(m)
                label = f"{tier}/{surf}"
                print(f"  {label:<22}{m['n']:>5}{m['model_ll']:>11.4f}{m['pinn_ll']:>11.4f}"
                      f"{m['gap']:>+10.4f}{m['p_better']:>10.3f}")

    print("\n" + "=" * 75)
    print("SLICE 4: BY PINNACLE FAVORITE STRENGTH")
    print("=" * 75)
    print("  (How strong is Pinnacle's call? Coin-flips harder for everyone.)")
    print(f"  {'fav_strength':<22}{'n':>5}{'model_ll':>11}{'pinn_ll':>11}{'gap':>10}{'p_better':>10}")
    print("  " + "-" * 69)
    # Strength of fav = max(p_pinn, 1-p_pinn)
    df["fav_strength"] = np.maximum(p_pinn, 1 - p_pinn)
    for label, lo, hi in [
        ("toss-up (.50-.55)", 0.50, 0.55),
        ("light fav (.55-.65)", 0.55, 0.65),
        ("medium fav (.65-.75)", 0.65, 0.75),
        ("heavy fav (.75-.85)", 0.75, 0.85),
        ("vy heavy (.85-.95)", 0.85, 0.95),
        ("locks (.95+)", 0.95, 1.01),
    ]:
        sub = df[(df["fav_strength"] >= lo) & (df["fav_strength"] < hi)]
        m = slice_metrics(sub, "fav_strength", label)
        if m and m["n"] >= 30:
            rows.append(m)
            print(f"  {label:<22}{m['n']:>5}{m['model_ll']:>11.4f}{m['pinn_ll']:>11.4f}"
                  f"{m['gap']:>+10.4f}{m['p_better']:>10.3f}")

    print("\n" + "=" * 75)
    print("SLICE 5: BY MODEL/PINNACLE DISAGREEMENT MAGNITUDE")
    print("=" * 75)
    print("  (When we disagree most, who's right more often?)")
    print(f"  {'disagreement':<22}{'n':>5}{'model_ll':>11}{'pinn_ll':>11}{'gap':>10}{'p_better':>10}")
    print("  " + "-" * 69)
    for label, lo, hi in [
        ("agree (<5pp)", 0.0, 0.05),
        ("mild (5-10pp)", 0.05, 0.10),
        ("medium (10-15pp)", 0.10, 0.15),
        ("large (15-25pp)", 0.15, 0.25),
        ("huge (25%+)", 0.25, 1.0),
    ]:
        sub = df[(df["disagreement"] >= lo) & (df["disagreement"] < hi)]
        m = slice_metrics(sub, "disagreement", label)
        if m and m["n"] >= 30:
            rows.append(m)
            print(f"  {label:<22}{m['n']:>5}{m['model_ll']:>11.4f}{m['pinn_ll']:>11.4f}"
                  f"{m['gap']:>+10.4f}{m['p_better']:>10.3f}")

    # ============ DIRECTIONAL DISAGREEMENT ============
    print("\n" + "=" * 75)
    print("SLICE 6: WHEN MODEL DISAGREES, WHICH DIRECTION IS WRONG?")
    print("=" * 75)
    print("  When model > Pinnacle by >10pp, model thinks A is more likely than market does.")
    print("  When model < Pinnacle by >10pp, model thinks A is less likely.")
    print("  Compare to actual win rates.")
    print(f"  {'condition':<28}{'n':>5}{'model_pred':>12}{'pinn_pred':>12}{'actual':>9}")
    print("  " + "-" * 66)

    cond1 = df["model_minus_pinn"] > 0.10
    cond2 = df["model_minus_pinn"] < -0.10
    print(f"  model > pinn by >10pp       {cond1.sum():>5}"
          f"{p_model[cond1].mean():>12.4f}{p_pinn[cond1].mean():>12.4f}"
          f"{y[cond1].mean():>9.4f}")
    print(f"  model < pinn by >10pp       {cond2.sum():>5}"
          f"{p_model[cond2].mean():>12.4f}{p_pinn[cond2].mean():>12.4f}"
          f"{y[cond2].mean():>9.4f}")

    # ============ THE BIG QUESTION ============
    print("\n" + "=" * 75)
    print("BOTTOM LINE: ANY SLICE WHERE MODEL BEATS PINNACLE?")
    print("=" * 75)
    res_df = pd.DataFrame(rows)
    res_df.to_csv("audit_report.csv", index=False)

    # Find slices where p_better > 0.5 (model beats pinn in >50% of bootstrap samples)
    candidates = res_df[res_df["p_better"] > 0.5].sort_values("p_better", ascending=False)
    print(f"\n  Slices where bootstrap suggests model >= Pinnacle (p_better > 0.5):")
    if len(candidates) == 0:
        print("    NONE. Pinnacle wins everywhere.")
    else:
        for _, r in candidates.iterrows():
            print(f"    {r['slice']:<18} {str(r['value']):<22} n={int(r['n']):>4}  "
                  f"gap={r['gap']:+.4f}  p_better={r['p_better']:.3f}")

    # Find slices where gap is small enough that model might beat with calibration
    print(f"\n  Slices with smallest gap (closest to beating Pinnacle):")
    sorted_slices = res_df.sort_values("gap").head(10)
    for _, r in sorted_slices.iterrows():
        print(f"    {r['slice']:<18} {str(r['value']):<22} n={int(r['n']):>4}  "
              f"gap={r['gap']:+.4f}  ci=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    print("\n  Saved full table to audit_report.csv")

    # ============ TOP DISAGREEMENTS ============
    print("\n" + "=" * 75)
    print("TOP 20 LARGEST DISAGREEMENTS — FORENSIC LOOK")
    print("=" * 75)
    big = df.sort_values("disagreement", ascending=False).head(20)
    print(f"  {'date':<12}{'tier':>6}{'surf':>8}{'p_model':>10}{'p_pinn':>10}{'diff':>9}{'won':>6}")
    print("  " + "-" * 61)
    for _, r in big.iterrows():
        won = "YES" if r["actual_a_won"] == 1 else "no"
        print(f"  {str(r['date'].date()):<12}{r['tourney_level']:>6}{r['surface']:>8}"
              f"{r['p_a_base_elo']:>10.3f}{r['fair_prob_a']:>10.3f}"
              f"{r['model_minus_pinn']:>+9.3f}{won:>6}")
    n_big_model_correct = (big["actual_a_won"] == (big["p_a_base_elo"] > 0.5)).sum()
    n_big_pinn_correct = (big["actual_a_won"] == (big["fair_prob_a"] > 0.5)).sum()
    print(f"\n  Of top 20 disagreements:")
    print(f"    Model picked winner: {n_big_model_correct}/20")
    print(f"    Pinnacle picked winner: {n_big_pinn_correct}/20")


if __name__ == "__main__":
    main()
