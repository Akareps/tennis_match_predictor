"""
Rank-warm-start Elo.

Diagnoses cold-start failures (the first-rounders we lose against Pinnacle)
by giving new players a rating derived from their ATP ranking at that
moment, then fading that signal as they accumulate match observations.

Logic:
  At each match prediction:
    For each player, get their current rank (from Sackmann's winner_rank/loser_rank).
    Compute rank_rating = empirical_mapping(rank).
    Compute n = matches observed for this player so far.
    Blend: effective_rating = w * rank_rating + (1-w) * elo_rating
           where w = max(0, 1 - n / WARMUP_THRESHOLD)
    n = 0:    fully rank-derived
    n = WARMUP_THRESHOLD/2:  50/50 blend
    n >= WARMUP_THRESHOLD:   pure Elo

Step 1: empirically fit rank → Elo mapping using established players (n>=30).
Step 2: walk forward, applying the blend for cold-start.
Step 3: evaluate vs Pinnacle, especially in the cold-start slice.

Output: rank_warmstart_predictions.csv + comparison vs base Elo.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from data_loader import load_years
from odds_loader import load_odds_years, merge_predictions_with_odds


INITIAL_RATING = 1500
SCALE = 400.0
K_BY_LEVEL = {
    "G": 32, "M": 28, "A": 24, "F": 28,
    "D": 20, "C": 18, "S": 14, "O": 20,
}
DEFAULT_K = 24

WARMUP_THRESHOLD = 30   # after this many matches, fully Elo
SURFACE_BLEND = 0.5

TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-09-30")


def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / SCALE))


def fit_rank_to_rating(matches, established_threshold=30):
    """
    Walk forward, accumulate Elo for each player. Then for established players
    (>= established_threshold matches), pair their (rank_at_match, elo_at_match)
    observations and fit a regression of elo vs log(rank).

    Returns: dict with 'a', 'b' coefficients for elo = a + b * log(rank).
    """
    overall = defaultdict(lambda: INITIAL_RATING)
    n_obs = defaultdict(int)
    pairs = []  # (rank, elo) tuples for established players

    for _, m in matches.iterrows():
        w = m["winner_id"]
        l = m["loser_id"]
        rw = overall[w]
        rl = overall[l]
        p_w = elo_expected(rw, rl)

        # Record (rank, elo) BEFORE this match for established players
        if pd.notna(m.get("winner_rank")) and n_obs[w] >= established_threshold:
            try:
                rank_w = int(m["winner_rank"])
                if 1 <= rank_w <= 1500:
                    pairs.append((rank_w, rw))
            except (ValueError, TypeError):
                pass
        if pd.notna(m.get("loser_rank")) and n_obs[l] >= established_threshold:
            try:
                rank_l = int(m["loser_rank"])
                if 1 <= rank_l <= 1500:
                    pairs.append((rank_l, rl))
            except (ValueError, TypeError):
                pass

        # Update Elo
        K = K_BY_LEVEL.get(m["tourney_level"], DEFAULT_K)
        overall[w] = rw + K * (1 - p_w)
        overall[l] = rl + K * (0 - (1 - p_w))
        n_obs[w] += 1
        n_obs[l] += 1

    # Fit elo = a + b * log(rank)
    pairs = np.array(pairs)
    log_ranks = np.log(pairs[:, 0])
    elos = pairs[:, 1]
    # Simple least squares
    n = len(pairs)
    mean_lr = log_ranks.mean()
    mean_elo = elos.mean()
    b = np.sum((log_ranks - mean_lr) * (elos - mean_elo)) / np.sum((log_ranks - mean_lr) ** 2)
    a = mean_elo - b * mean_lr
    # R-squared
    pred = a + b * log_ranks
    ss_res = np.sum((elos - pred) ** 2)
    ss_tot = np.sum((elos - mean_elo) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return {
        "a": float(a),
        "b": float(b),
        "n_pairs": int(n),
        "r_squared": float(r_squared),
        "rank_to_rating": lambda rank: a + b * np.log(np.clip(rank, 1, 1500)),
    }


def fit_warmstart_model(matches, rank_map, test_start, test_end):
    """
    Walk forward with rank-warmstart blending.
    rank_map: function rank -> initial rating estimate
    """
    overall = defaultdict(lambda: None)         # None = uninitialized
    surface_ratings = defaultdict(lambda: defaultdict(lambda: None))
    n_obs_overall = defaultdict(int)
    n_obs_surface = defaultdict(lambda: defaultdict(int))

    predictions = []

    def get_blended_rating(player_id, rank, base_rating, n_obs_count):
        """
        Blend rank-derived rating with current Elo based on n_obs.
        Uses INITIAL_RATING fallback if rank is missing.
        """
        if base_rating is None:
            # First time seeing this player
            if rank is not None and 1 <= rank <= 1500:
                return rank_map(rank), True
            return INITIAL_RATING, True
        # Have an Elo. Blend with rank if we still have warmup weight.
        w_rank = max(0.0, 1.0 - n_obs_count / WARMUP_THRESHOLD)
        if w_rank == 0 or rank is None or rank < 1 or rank > 1500:
            return base_rating, False
        rank_rating = rank_map(rank)
        return w_rank * rank_rating + (1 - w_rank) * base_rating, False

    for _, m in matches.iterrows():
        w = m["winner_id"]
        l = m["loser_id"]
        s = m["surface"]

        rank_w = m.get("winner_rank") if pd.notna(m.get("winner_rank")) else None
        rank_l = m.get("loser_rank") if pd.notna(m.get("loser_rank")) else None
        try:
            rank_w = int(rank_w) if rank_w is not None else None
        except (ValueError, TypeError):
            rank_w = None
        try:
            rank_l = int(rank_l) if rank_l is not None else None
        except (ValueError, TypeError):
            rank_l = None

        # Get blended ratings for prediction
        ro_w, w_was_new = get_blended_rating(w, rank_w, overall[w], n_obs_overall[w])
        ro_l, l_was_new = get_blended_rating(l, rank_l, overall[l], n_obs_overall[l])
        rs_w_base = surface_ratings[s][w]
        rs_l_base = surface_ratings[s][l]
        rs_w, _ = get_blended_rating(w, rank_w, rs_w_base, n_obs_surface[s][w])
        rs_l, _ = get_blended_rating(l, rank_l, rs_l_base, n_obs_surface[s][l])

        p_overall = elo_expected(ro_w, ro_l)
        p_surface = elo_expected(rs_w, rs_l)
        p_w = SURFACE_BLEND * p_surface + (1 - SURFACE_BLEND) * p_overall

        # Capture diagnostic info
        warmup_weight_w = max(0.0, 1.0 - n_obs_overall[w] / WARMUP_THRESHOLD) if not w_was_new else 1.0
        warmup_weight_l = max(0.0, 1.0 - n_obs_overall[l] / WARMUP_THRESHOLD) if not l_was_new else 1.0
        is_cold_start = (n_obs_overall[w] < WARMUP_THRESHOLD) or (n_obs_overall[l] < WARMUP_THRESHOLD)

        if test_start <= m["date"] < test_end:
            if w < l:
                a_id, b_id = w, l
                p_a = p_w
                actual_a = 1
                a_n = n_obs_overall[w]
                b_n = n_obs_overall[l]
                a_rank = rank_w
                b_rank = rank_l
            else:
                a_id, b_id = l, w
                p_a = 1 - p_w
                actual_a = 0
                a_n = n_obs_overall[l]
                b_n = n_obs_overall[w]
                a_rank = rank_l
                b_rank = rank_w
            predictions.append({
                "date": m["date"],
                "surface": s,
                "tourney_name": m.get("tourney_name", ""),
                "tourney_level": m["tourney_level"],
                "player_a_id": a_id,
                "player_b_id": b_id,
                "actual_a_won": actual_a,
                "p_a_warmstart": p_a,
                "a_n_obs": a_n,
                "b_n_obs": b_n,
                "a_rank": a_rank,
                "b_rank": b_rank,
                "is_cold_start": is_cold_start,
            })

        # Update ratings
        K = K_BY_LEVEL.get(m["tourney_level"], DEFAULT_K)
        overall[w] = ro_w + K * (1 - p_overall)
        overall[l] = ro_l + K * (0 - (1 - p_overall))
        surface_ratings[s][w] = rs_w + K * (1 - p_surface)
        surface_ratings[s][l] = rs_l + K * (0 - (1 - p_surface))
        n_obs_overall[w] += 1
        n_obs_overall[l] += 1
        n_obs_surface[s][w] += 1
        n_obs_surface[s][l] += 1

    return pd.DataFrame(predictions)


def evaluate_logloss(probs, y):
    probs = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)
    y = np.asarray(y).astype(float)
    if len(probs) == 0:
        return float("nan")
    return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))


def main():
    print("=" * 75)
    print("RANK-WARMSTART ELO")
    print("=" * 75)

    print("\n[1/5] Loading ATP data 2021-2024...")
    raw = load_years([2021, 2022, 2023, 2024], tour="atp")
    matches = raw[raw["w_svpt"].notna() & raw["l_svpt"].notna()].copy()
    matches["date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")
    matches = matches.dropna(subset=["date", "surface", "tourney_name"])
    matches["surface"] = matches["surface"].str.strip()
    matches = matches.sort_values("date").reset_index(drop=True)
    print(f"      {len(matches)} matches.")

    name_map = {}
    for _, r in raw.iterrows():
        if pd.notna(r.get("winner_id")) and pd.notna(r.get("winner_name")):
            name_map[int(r["winner_id"])] = r["winner_name"]
        if pd.notna(r.get("loser_id")) and pd.notna(r.get("loser_name")):
            name_map[int(r["loser_id"])] = r["loser_name"]

    print(f"\n[2/5] Fitting rank → Elo mapping on established players (pre-test window)...")
    # Avoid leakage: fit only on matches BEFORE the test window
    pretest_matches = matches[matches["date"] < TEST_START]
    print(f"      Using {len(pretest_matches)} pre-test matches for the fit.")
    fit = fit_rank_to_rating(pretest_matches)
    print(f"      Elo ≈ {fit['a']:.1f} + {fit['b']:.1f} * log(rank)")
    print(f"      n_pairs: {fit['n_pairs']}, R²: {fit['r_squared']:.3f}")
    print(f"      Examples:")
    for r in [1, 5, 10, 25, 50, 100, 200, 500]:
        rating = fit["rank_to_rating"](r)
        print(f"        rank {r:>4}  →  rating {rating:.0f}")

    print(f"\n[3/5] Walk-forward with warmstart (threshold={WARMUP_THRESHOLD})...")
    pred = fit_warmstart_model(matches, fit["rank_to_rating"], TEST_START, TEST_END)
    print(f"      {len(pred)} predictions.")
    print(f"      Cold-start matches (either player < {WARMUP_THRESHOLD} obs): "
          f"{pred['is_cold_start'].sum()} ({pred['is_cold_start'].mean():.1%})")

    print("\n[4/5] Loading Pinnacle odds and merging...")
    odds = load_odds_years([2024])
    merged = merge_predictions_with_odds(pred, odds, name_map)
    print(f"      Matched {len(merged)} / {len(pred)}")

    if len(merged) == 0:
        print("No matches — aborting.")
        return

    # Also load base Elo predictions for comparison
    print("\n[5/5] Comparing to base Elo (from enhanced_predictions.csv)...")
    base = pd.read_csv("enhanced_predictions.csv", parse_dates=["date"])
    # Match on (date, player_a_id, player_b_id)
    keys = ["date", "player_a_id", "player_b_id"]
    cmp_df = merged.merge(
        base[keys + ["p_a_base_elo"]],
        on=keys, how="inner",
    )
    print(f"      Common matches: {len(cmp_df)}")

    y = cmp_df["actual_a_won"].values
    p_warm = cmp_df["p_a_warmstart"].values
    p_base = cmp_df["p_a_base_elo"].values
    p_pinn = cmp_df["fair_prob_a"].values

    ll_warm = evaluate_logloss(p_warm, y)
    ll_base = evaluate_logloss(p_base, y)
    ll_pinn = evaluate_logloss(p_pinn, y)

    print("\n" + "=" * 75)
    print("HEAD-TO-HEAD")
    print("=" * 75)
    print(f"  Base Elo:       {ll_base:.4f}")
    print(f"  Warmstart Elo:  {ll_warm:.4f}  (delta {ll_base - ll_warm:+.4f})")
    print(f"  Pinnacle:       {ll_pinn:.4f}")
    print(f"\n  Gap base vs Pinnacle:      {ll_base - ll_pinn:+.4f}")
    print(f"  Gap warmstart vs Pinnacle: {ll_warm - ll_pinn:+.4f}")

    # The crucial slice: cold-start matches
    print("\n" + "=" * 75)
    print("COLD-START SLICE: matches where at least one player has < 30 prior obs")
    print("=" * 75)
    cold = cmp_df[cmp_df["is_cold_start"] == True]
    warm_only = cmp_df[cmp_df["is_cold_start"] == False]
    print(f"\n  Cold-start matches: {len(cold)}")
    if len(cold) >= 30:
        ll_w_cold = evaluate_logloss(cold["p_a_warmstart"].values, cold["actual_a_won"].values)
        ll_b_cold = evaluate_logloss(cold["p_a_base_elo"].values, cold["actual_a_won"].values)
        ll_p_cold = evaluate_logloss(cold["fair_prob_a"].values, cold["actual_a_won"].values)
        print(f"    Base:      {ll_b_cold:.4f}")
        print(f"    Warmstart: {ll_w_cold:.4f}  (delta {ll_b_cold - ll_w_cold:+.4f})")
        print(f"    Pinnacle:  {ll_p_cold:.4f}")
        print(f"    Gap base vs Pinn:      {ll_b_cold - ll_p_cold:+.4f}")
        print(f"    Gap warmstart vs Pinn: {ll_w_cold - ll_p_cold:+.4f}")
    print(f"\n  Warm matches: {len(warm_only)}")
    if len(warm_only) >= 30:
        ll_w_warm = evaluate_logloss(warm_only["p_a_warmstart"].values, warm_only["actual_a_won"].values)
        ll_b_warm = evaluate_logloss(warm_only["p_a_base_elo"].values, warm_only["actual_a_won"].values)
        ll_p_warm = evaluate_logloss(warm_only["fair_prob_a"].values, warm_only["actual_a_won"].values)
        print(f"    Base:      {ll_b_warm:.4f}")
        print(f"    Warmstart: {ll_w_warm:.4f}  (delta {ll_b_warm - ll_w_warm:+.4f})")
        print(f"    Pinnacle:  {ll_p_warm:.4f}")
        print(f"    Gap base vs Pinn:      {ll_b_warm - ll_p_warm:+.4f}")
        print(f"    Gap warmstart vs Pinn: {ll_w_warm - ll_p_warm:+.4f}")

    # By tier (Slams have most cold-start since players come up from quals)
    print("\n" + "=" * 75)
    print("BY TIER (where does the warmstart help?)")
    print("=" * 75)
    print(f"  {'tier':<10}{'n':>5}{'base':>10}{'warm':>10}{'pinn':>10}{'cold%':>8}")
    print("  " + "-" * 55)
    for tier in ["G", "M", "A"]:
        sub = cmp_df[cmp_df["tourney_level"] == tier]
        if len(sub) >= 30:
            llb = evaluate_logloss(sub["p_a_base_elo"].values, sub["actual_a_won"].values)
            llw = evaluate_logloss(sub["p_a_warmstart"].values, sub["actual_a_won"].values)
            llp = evaluate_logloss(sub["fair_prob_a"].values, sub["actual_a_won"].values)
            cold_pct = sub["is_cold_start"].mean()
            print(f"  {tier:<10}{len(sub):>5}{llb:>10.4f}{llw:>10.4f}{llp:>10.4f}{cold_pct:>8.1%}")

    # By disagreement bucket — does warmstart fix the huge-disagreement cases?
    print("\n" + "=" * 75)
    print("BY MODEL/PINNACLE DISAGREEMENT (does warmstart fix our blowups?)")
    print("=" * 75)
    cmp_df_with_diff = cmp_df.copy()
    cmp_df_with_diff["base_disagree"] = (cmp_df["p_a_base_elo"] - cmp_df["fair_prob_a"]).abs()
    cmp_df_with_diff["warm_disagree"] = (cmp_df["p_a_warmstart"] - cmp_df["fair_prob_a"]).abs()
    print(f"  Mean |base - pinn|:       {cmp_df_with_diff['base_disagree'].mean():.4f}")
    print(f"  Mean |warmstart - pinn|:  {cmp_df_with_diff['warm_disagree'].mean():.4f}")
    print(f"  P(base disagrees > 25pp):      {(cmp_df_with_diff['base_disagree'] > 0.25).mean():.3f}")
    print(f"  P(warmstart disagrees > 25pp): {(cmp_df_with_diff['warm_disagree'] > 0.25).mean():.3f}")

    # Among the matches where base had huge disagreement (>20pp), did warmstart shrink the gap?
    big = cmp_df_with_diff[cmp_df_with_diff["base_disagree"] > 0.20]
    if len(big) >= 30:
        print(f"\n  Among matches where BASE disagreed with Pinnacle by >20pp (n={len(big)}):")
        avg_base_disagree = big["base_disagree"].mean()
        avg_warm_disagree = big["warm_disagree"].mean()
        print(f"    Avg |base-pinn|:      {avg_base_disagree:.4f}")
        print(f"    Avg |warmstart-pinn|: {avg_warm_disagree:.4f}")
        print(f"    Reduction: {(avg_base_disagree - avg_warm_disagree):.4f} "
              f"({100*(avg_base_disagree - avg_warm_disagree) / avg_base_disagree:.1f}%)")

    cmp_df.to_csv("rank_warmstart_predictions.csv", index=False)
    print(f"\n  Saved to rank_warmstart_predictions.csv")


if __name__ == "__main__":
    main()
