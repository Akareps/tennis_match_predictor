"""
Elo baseline for tennis matches, with surface-specific variant.

Tests the null hypothesis: does a 50-line Elo beat our 600-line serve/return
decomposition pipeline?

Two flavors:
  1. Vanilla Elo: one rating per player, K-factor by tournament tier.
  2. Surface Elo: separate rating per surface, weighted blend of overall +
     surface-specific (Sackmann-style).

Run on the SAME train/test split as the main pipeline so log-loss is
directly comparable.

Usage:
    python3 elo_baseline.py
"""

import pandas as pd
import numpy as np
from data_loader import load_years, to_long_format


# Tournament-tier K-factors (rough guide):
# Slams matter most, then Masters, then 500s, then 250s, then quals.
K_BY_LEVEL = {
    "G": 32,   # Grand Slam
    "M": 28,   # Masters 1000
    "A": 24,   # ATP 500/250 (Sackmann lumps these as 'A')
    "F": 28,   # Tour Finals
    "D": 20,   # Davis Cup
    "C": 18,   # Challenger
    "S": 14,   # Satellite/ITF
    "O": 20,   # Olympics
}
DEFAULT_K = 24

INITIAL_RATING = 1500
SCALE = 400.0  # standard Elo scale


def elo_expected(r_a: float, r_b: float) -> float:
    """Standard Elo: P(A beats B) = 1 / (1 + 10^((r_b - r_a) / 400))."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / SCALE))


def run_vanilla_elo(matches: pd.DataFrame, test_start: pd.Timestamp,
                    test_end: pd.Timestamp) -> pd.DataFrame:
    """
    matches: ONE row per match (winner / loser format), sorted by date.
    Walks chronologically. For matches in test window, records prediction
    BEFORE updating rating. For all matches, updates after.
    """
    ratings = {}  # player_id -> rating
    predictions = []

    for _, m in matches.iterrows():
        w = m["winner_id"]
        l = m["loser_id"]
        rw = ratings.get(w, INITIAL_RATING)
        rl = ratings.get(l, INITIAL_RATING)
        p_w = elo_expected(rw, rl)

        if test_start <= m["date"] < test_end:
            # Canonicalize: predict for player_a = min(w, l)
            if w < l:
                a, b, p_a, actual = w, l, p_w, 1
            else:
                a, b, p_a, actual = l, w, 1 - p_w, 0
            predictions.append({
                "date": m["date"],
                "surface": m["surface"],
                "player_a_id": a,
                "player_b_id": b,
                "actual_a_won": actual,
                "elo_prob_a": p_a,
            })

        # Update ratings
        K = K_BY_LEVEL.get(m["tourney_level"], DEFAULT_K)
        ratings[w] = rw + K * (1 - p_w)
        ratings[l] = rl + K * (0 - (1 - p_w))

    return pd.DataFrame(predictions)


def run_surface_elo(matches: pd.DataFrame, test_start: pd.Timestamp,
                    test_end: pd.Timestamp, surface_weight: float = 0.5) -> pd.DataFrame:
    """
    Sackmann-style: maintain BOTH overall rating AND surface-specific rating.
    Prediction blends them: p = w * surface_elo + (1-w) * overall_elo.
    Update both after each match.
    """
    overall = {}
    surface_ratings = {"Hard": {}, "Clay": {}, "Grass": {}, "Carpet": {}}
    predictions = []

    for _, m in matches.iterrows():
        w = m["winner_id"]
        l = m["loser_id"]
        s = m["surface"]
        if s not in surface_ratings:
            surface_ratings[s] = {}

        ro_w = overall.get(w, INITIAL_RATING)
        ro_l = overall.get(l, INITIAL_RATING)
        rs_w = surface_ratings[s].get(w, INITIAL_RATING)
        rs_l = surface_ratings[s].get(l, INITIAL_RATING)

        p_overall = elo_expected(ro_w, ro_l)
        p_surface = elo_expected(rs_w, rs_l)
        p_w = surface_weight * p_surface + (1 - surface_weight) * p_overall

        if test_start <= m["date"] < test_end:
            if w < l:
                a, b, p_a, actual = w, l, p_w, 1
            else:
                a, b, p_a, actual = l, w, 1 - p_w, 0
            predictions.append({
                "date": m["date"],
                "surface": s,
                "player_a_id": a,
                "player_b_id": b,
                "actual_a_won": actual,
                "elo_prob_a": p_a,
            })

        K = K_BY_LEVEL.get(m["tourney_level"], DEFAULT_K)
        # Use the BLENDED probability for the update (so surface and overall
        # both nudge in same direction, just with different magnitudes).
        # But standard practice is to update each rating using its OWN
        # expected prob, which is what we'll do — it's cleaner.
        overall[w] = ro_w + K * (1 - p_overall)
        overall[l] = ro_l + K * (0 - (1 - p_overall))
        surface_ratings[s][w] = rs_w + K * (1 - p_surface)
        surface_ratings[s][l] = rs_l + K * (0 - (1 - p_surface))

    return pd.DataFrame(predictions)


def evaluate(probs, y, n_bins=10):
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    logloss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    bins = np.linspace(0, 1, n_bins + 1)
    bucket = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    table = []
    for b in range(n_bins):
        mask = bucket == b
        if mask.sum() >= 30:
            pred = float(probs[mask].mean())
            actual = float(y[mask].mean())
            table.append((f"{bins[b]:.1f}-{bins[b+1]:.1f}", int(mask.sum()),
                          pred, actual, actual - pred))
    acc = ((probs > 0.5) == (y == 1)).mean()
    return logloss, acc, table


def main():
    print("=" * 65)
    print("ELO BASELINE COMPARISON")
    print("=" * 65)

    print("\nLoading 2021-2024 ATP data...")
    raw = load_years([2021, 2022, 2023, 2024], tour="atp")
    # Build match-level DF (not long format — Elo wants winner/loser)
    matches = raw[raw["w_svpt"].notna() & raw["l_svpt"].notna()].copy()
    matches["date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")
    matches = matches.dropna(subset=["date", "surface"])
    matches = matches.sort_values("date").reset_index(drop=True)
    print(f"  {len(matches)} matches.")

    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2024-09-30")

    # --- Vanilla Elo ---
    print(f"\nRunning vanilla Elo...")
    vanilla = run_vanilla_elo(matches, test_start, test_end)
    print(f"  {len(vanilla)} predictions in test window.")

    # --- Surface Elo ---
    print(f"\nRunning surface Elo (blend weight=0.5)...")
    surface = run_surface_elo(matches, test_start, test_end, surface_weight=0.5)

    # Try a few blend weights
    print(f"\nTrying different surface blend weights:")
    for w in [0.3, 0.5, 0.7, 1.0]:
        surf_w = run_surface_elo(matches, test_start, test_end, surface_weight=w)
        ll, acc, _ = evaluate(surf_w["elo_prob_a"].values, surf_w["actual_a_won"].values)
        print(f"  w={w}: log-loss={ll:.4f}, acc={acc:.1%}")

    # --- Compare ---
    print("\n" + "=" * 65)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 65)

    ll_v, acc_v, tab_v = evaluate(vanilla["elo_prob_a"].values, vanilla["actual_a_won"].values)
    ll_s, acc_s, tab_s = evaluate(surface["elo_prob_a"].values, surface["actual_a_won"].values)

    print(f"\nVanilla Elo:  log-loss={ll_v:.4f}, acc={acc_v:.1%}, n={len(vanilla)}")
    print(f"Surface Elo:  log-loss={ll_s:.4f}, acc={acc_s:.1%}, n={len(surface)}")

    # Compare to v3 pipeline
    print(f"\nFor reference, current v3 pipeline (post-warmup, n=1185):")
    print(f"  Raw:        log-loss=0.6404")
    print(f"  + Shrunk:   log-loss=0.6337  <-- our best so far")
    print(f"  + Cal'd:    log-loss=0.6460")

    # Restrict Elo to same n=1185 sample for apples-to-apples
    # (after warmup_days=90 from test_start)
    cal_warmup = test_start + pd.Timedelta(days=90)
    vanilla_post = vanilla[vanilla["date"] >= cal_warmup]
    surface_post = surface[surface["date"] >= cal_warmup]
    if len(vanilla_post) > 0:
        ll_vp, acc_vp, _ = evaluate(
            vanilla_post["elo_prob_a"].values, vanilla_post["actual_a_won"].values
        )
        ll_sp, acc_sp, _ = evaluate(
            surface_post["elo_prob_a"].values, surface_post["actual_a_won"].values
        )
        print(f"\nElo on the SAME post-warmup window for direct comparison:")
        print(f"  Vanilla Elo (post-warmup): log-loss={ll_vp:.4f}, acc={acc_vp:.1%}, n={len(vanilla_post)}")
        print(f"  Surface Elo (post-warmup): log-loss={ll_sp:.4f}, acc={acc_sp:.1%}, n={len(surface_post)}")

    print("\n" + "=" * 65)
    print("VANILLA ELO CALIBRATION TABLE")
    print("=" * 65)
    print(f"{'bucket':<12}{'n':>6}{'pred':>8}{'actual':>9}{'gap':>9}")
    for bucket, n, pred, actual, gap in tab_v:
        flag = " *" if abs(gap) > 0.04 else "  "
        print(f"{flag}{bucket:<10}{n:>6}{pred:>8.3f}{actual:>9.3f}{gap:>+9.3f}")

    print("\n" + "=" * 65)
    print("SURFACE ELO CALIBRATION TABLE")
    print("=" * 65)
    print(f"{'bucket':<12}{'n':>6}{'pred':>8}{'actual':>9}{'gap':>9}")
    for bucket, n, pred, actual, gap in tab_s:
        flag = " *" if abs(gap) > 0.04 else "  "
        print(f"{flag}{bucket:<10}{n:>6}{pred:>8.3f}{actual:>9.3f}{gap:>+9.3f}")

    # Save Elo predictions for later analysis
    vanilla.to_csv("elo_vanilla_predictions.csv", index=False)
    surface.to_csv("elo_surface_predictions.csv", index=False)
    print("\nElo predictions saved to elo_vanilla_predictions.csv and elo_surface_predictions.csv")


if __name__ == "__main__":
    main()
