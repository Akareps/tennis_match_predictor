"""
Score market evaluation.

For each match in predictions_score.csv:
  - Apply alpha shrinkage to spw probs (we found alpha=0.20 helps at match level;
    apply equivalent at point level via Sackmann transform)
  - Derive probabilities for each score market
  - Compare against actual outcomes
  - Evaluate calibration & log-loss per market

Markets evaluated (Bo3 only for now — most ATP matches):
  1. Match winner             P(A wins)
  2. A wins ≥1 set            P(A wins set | not 0-2)
  3. B wins ≥1 set            P(B wins set | not 0-2)  [a.k.a. "no straight set sweep"]
  4. A wins 2-0
  5. A wins 2-1
  6. B wins 2-0
  7. B wins 2-1
  8. Match goes 3 sets
  9. Total games over 21.5
 10. Total games over 22.5

For each market we compute:
  - log-loss (lower is better; baseline = log-loss using market prior)
  - calibration table
  - "edge over prior" — does the model do better than just guessing the average rate?

The "edge over prior" metric is more honest than log-loss-vs-uniform here,
because some markets have low/high base rates (e.g. P(2-0) ~ 0.35).
"""

import sys
import numpy as np
import pandas as pd
from score_markets import match_market_probs, p_total_games_over


def shrink_match_prob(p, alpha=0.20):
    """Same shrinkage we found at match level."""
    return p * (1 - alpha) + 0.5 * alpha


def shrink_point_prob(p, alpha=0.10):
    """
    Equivalent point-level shrinkage. We use a smaller alpha because point
    probs compound through the Markov chain — a 10% shrinkage at point level
    produces a much larger effect at match level.
    
    Empirically, alpha_point ~ 0.5 * alpha_match gives roughly equivalent
    match-level shrinkage when point probs are in 0.55-0.70 range.
    """
    return p * (1 - alpha) + 0.5 * alpha


def evaluate_market(probs, outcomes, name=""):
    """
    Returns log-loss, accuracy (for binary markets), calibration table.
    """
    probs = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)
    outcomes = np.asarray(outcomes).astype(float)
    n = len(probs)
    if n == 0:
        return None
    logloss = -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    base_rate = float(outcomes.mean())
    # Baseline log-loss = always predict base rate
    base_logloss = -(base_rate * np.log(max(base_rate, 1e-6))
                     + (1 - base_rate) * np.log(max(1 - base_rate, 1e-6)))
    # Calibration table
    bins = np.linspace(0, 1, 11)
    bucket = np.clip(np.digitize(probs, bins) - 1, 0, 9)
    cal = []
    for b in range(10):
        mask = bucket == b
        if mask.sum() >= 30:
            cal.append({
                "bucket": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "n": int(mask.sum()),
                "pred": float(probs[mask].mean()),
                "actual": float(outcomes[mask].mean()),
            })
    return {
        "name": name,
        "n": n,
        "logloss": logloss,
        "base_logloss": base_logloss,
        "improvement": base_logloss - logloss,
        "base_rate": base_rate,
        "mean_pred": float(probs.mean()),
        "calibration": cal,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "predictions_score.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df)} predictions from {path}")

    # Bo3 only for primary analysis (most ATP matches)
    bo3 = df[df["best_of"] == 3].copy()
    print(f"Bo3 matches: {len(bo3)}")

    # Compute derived market probabilities for each match
    print("\nDeriving market probabilities for each match...")
    # Apply point-level shrinkage to spw_a, spw_b
    SHRINK = 0.10
    bo3["spw_a_shrunk"] = shrink_point_prob(bo3["spw_a"].values, SHRINK)
    bo3["spw_b_shrunk"] = shrink_point_prob(bo3["spw_b"].values, SHRINK)

    market_cols = []
    rows = []
    for _, r in bo3.iterrows():
        m = match_market_probs(r["spw_a_shrunk"], r["spw_b_shrunk"], best_of=3)
        # Total games markets — use unshrunk for now; over/under is symmetric
        # under shrinkage so it doesn't matter much
        p_o215 = p_total_games_over(r["spw_a_shrunk"], r["spw_b_shrunk"], 21.5, 3)
        p_o225 = p_total_games_over(r["spw_a_shrunk"], r["spw_b_shrunk"], 22.5, 3)
        rows.append({
            "p_match_a": m["p_match_a"],
            "p_a_2_0": m["p_a_2_0"],
            "p_a_2_1": m["p_a_2_1"],
            "p_b_2_0": m["p_b_2_0"],
            "p_b_2_1": m["p_b_2_1"],
            "p_a_wins_at_least_one_set": m["p_a_wins_at_least_one_set"],
            "p_b_wins_at_least_one_set": m["p_b_wins_at_least_one_set"],
            "p_match_3_sets": m["p_match_total_sets_3"],
            "p_total_over_21_5": p_o215,
            "p_total_over_22_5": p_o225,
        })
    market_df = pd.DataFrame(rows, index=bo3.index)
    bo3 = pd.concat([bo3, market_df], axis=1)

    # Evaluate each market
    print("\n" + "=" * 75)
    print("MARKET-BY-MARKET EVALUATION (Bo3, n={})".format(len(bo3)))
    print("=" * 75)
    print(f"{'market':<32}{'n':>5}{'base':>9}{'pred':>9}{'logloss':>10}{'improv':>9}")
    print("-" * 75)

    # Outcomes for each market
    actual_a_2_0 = ((bo3["actual_a_sets"] == 2) & (bo3["actual_b_sets"] == 0)).astype(int)
    actual_a_2_1 = ((bo3["actual_a_sets"] == 2) & (bo3["actual_b_sets"] == 1)).astype(int)
    actual_b_2_0 = ((bo3["actual_a_sets"] == 0) & (bo3["actual_b_sets"] == 2)).astype(int)
    actual_b_2_1 = ((bo3["actual_a_sets"] == 1) & (bo3["actual_b_sets"] == 2)).astype(int)
    actual_match_3_sets = (bo3["actual_a_sets"] + bo3["actual_b_sets"] == 3).astype(int)
    actual_total_over_21_5 = (bo3["actual_total_games"] > 21.5).astype(int)
    actual_total_over_22_5 = (bo3["actual_total_games"] > 22.5).astype(int)

    markets = [
        ("Match winner (A)", bo3["p_match_a"], bo3["actual_a_won"]),
        ("A wins ≥1 set", bo3["p_a_wins_at_least_one_set"], bo3["actual_a_won_at_least_one_set"]),
        ("B wins ≥1 set", bo3["p_b_wins_at_least_one_set"], bo3["actual_b_won_at_least_one_set"]),
        ("A wins 2-0", bo3["p_a_2_0"], actual_a_2_0),
        ("A wins 2-1", bo3["p_a_2_1"], actual_a_2_1),
        ("B wins 2-0", bo3["p_b_2_0"], actual_b_2_0),
        ("B wins 2-1", bo3["p_b_2_1"], actual_b_2_1),
        ("Match goes 3 sets", bo3["p_match_3_sets"], actual_match_3_sets),
        ("Total games over 21.5", bo3["p_total_over_21_5"], actual_total_over_21_5),
        ("Total games over 22.5", bo3["p_total_over_22_5"], actual_total_over_22_5),
    ]

    eval_results = []
    for name, probs, y in markets:
        m = evaluate_market(probs, y, name)
        eval_results.append(m)
        improv_str = f"{m['improvement']:+.4f}"
        flag = " *" if m['improvement'] < 0.001 else "  "
        print(f"{flag}{name:<30}{m['n']:>5}{m['base_rate']:>9.3f}{m['mean_pred']:>9.3f}"
              f"{m['logloss']:>10.4f}{improv_str:>9}")

    print("\n  * = market where model does NOT meaningfully beat the base-rate prior.")

    # Detailed calibration for the most interesting markets
    print("\n" + "=" * 75)
    print("CALIBRATION TABLES — KEY MARKETS")
    print("=" * 75)
    for r in eval_results:
        if r["name"] in ("Match winner (A)", "A wins ≥1 set", "B wins ≥1 set",
                        "Match goes 3 sets", "Total games over 22.5"):
            print(f"\n{r['name']}  (n={r['n']}, base_rate={r['base_rate']:.3f})")
            print(f"  {'bucket':<12}{'n':>6}{'pred':>8}{'actual':>9}{'gap':>9}")
            for c in r["calibration"]:
                gap = c["actual"] - c["pred"]
                flag = " *" if abs(gap) > 0.04 else "  "
                print(f"  {flag}{c['bucket']:<10}{c['n']:>6}{c['pred']:>8.3f}"
                      f"{c['actual']:>9.3f}{gap:>+9.3f}")

    # Save full predictions for later analysis
    out_path = "predictions_with_markets.csv"
    bo3.to_csv(out_path, index=False)
    print(f"\nSaved full predictions with all market probs to {out_path}")

    # Bottom line
    print("\n" + "=" * 75)
    print("BOTTOM LINE")
    print("=" * 75)
    print(f"Markets ranked by improvement over base rate (best edge first):")
    eval_results.sort(key=lambda r: r["improvement"], reverse=True)
    for r in eval_results:
        marker = "  ✓" if r["improvement"] > 0.005 else "  ?" if r["improvement"] > 0 else "  ✗"
        print(f"  {marker} {r['name']:<32}  improvement = {r['improvement']:+.4f}")
    print()
    print("  ✓ = strong model signal in this market (>0.005 nats over base rate)")
    print("  ? = weak signal (0 to 0.005 nats)")
    print("  ✗ = no signal — model isn't useful in this market")


if __name__ == "__main__":
    main()
