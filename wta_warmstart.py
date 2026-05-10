"""
WTA pivot of the rank-warmstart Elo model.

Same architecture as rank_warmstart.py, but loads WTA data and WTA odds.

The hypothesis: WTA is documented to be a less efficient market than ATP.
If our model loses to ATP Pinnacle by ~0.04 nats but only loses to WTA Pinnacle
by ~0.01-0.02 nats (or breaks even), WTA is the more profitable target.

Outputs:
  wta_warmstart_predictions.csv
  Side-by-side comparison: ATP gap vs WTA gap to Pinnacle.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from data_loader import load_years
from odds_loader import load_odds_years, merge_predictions_with_odds


# Same hyperparameters as ATP model
INITIAL_RATING = 1500
SCALE = 400.0
K_BY_LEVEL = {
    "G": 32, "PM": 28, "P": 24, "I": 20, "F": 28,
    "D": 20, "C": 18, "S": 14, "O": 20,
    # WTA tier codes are slightly different from ATP. Common WTA codes:
    #   G  = Grand Slam
    #   PM = Premier Mandatory (now WTA 1000)
    #   P  = Premier
    #   I  = International
    # Add ATP fallback codes too in case Sackmann uses them:
    "M": 28, "A": 24,
}
DEFAULT_K = 24

WARMUP_THRESHOLD = 30
SURFACE_BLEND = 0.5

TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-09-30")

TOUR = "wta"


def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / SCALE))


def fit_rank_to_rating(matches, established_threshold=30):
    overall = defaultdict(lambda: INITIAL_RATING)
    n_obs = defaultdict(int)
    pairs = []

    for _, m in matches.iterrows():
        w = m["winner_id"]
        l = m["loser_id"]
        rw = overall[w]
        rl = overall[l]
        p_w = elo_expected(rw, rl)

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

        K = K_BY_LEVEL.get(m["tourney_level"], DEFAULT_K)
        overall[w] = rw + K * (1 - p_w)
        overall[l] = rl + K * (0 - (1 - p_w))
        n_obs[w] += 1
        n_obs[l] += 1

    if len(pairs) < 100:
        raise RuntimeError(f"Only {len(pairs)} (rank, elo) pairs collected — too few to fit")

    pairs = np.array(pairs)
    log_ranks = np.log(pairs[:, 0])
    elos = pairs[:, 1]
    n = len(pairs)
    mean_lr = log_ranks.mean()
    mean_elo = elos.mean()
    b = np.sum((log_ranks - mean_lr) * (elos - mean_elo)) / np.sum((log_ranks - mean_lr) ** 2)
    a = mean_elo - b * mean_lr
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
    overall = defaultdict(lambda: None)
    surface_ratings = defaultdict(lambda: defaultdict(lambda: None))
    n_obs_overall = defaultdict(int)
    n_obs_surface = defaultdict(lambda: defaultdict(int))

    predictions = []

    def get_blended_rating(player_id, rank, base_rating, n_obs_count):
        if base_rating is None:
            if rank is not None and 1 <= rank <= 1500:
                return rank_map(rank), True
            return INITIAL_RATING, True
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

        ro_w, w_was_new = get_blended_rating(w, rank_w, overall[w], n_obs_overall[w])
        ro_l, l_was_new = get_blended_rating(l, rank_l, overall[l], n_obs_overall[l])
        rs_w_base = surface_ratings[s][w]
        rs_l_base = surface_ratings[s][l]
        rs_w, _ = get_blended_rating(w, rank_w, rs_w_base, n_obs_surface[s][w])
        rs_l, _ = get_blended_rating(l, rank_l, rs_l_base, n_obs_surface[s][l])

        p_overall = elo_expected(ro_w, ro_l)
        p_surface = elo_expected(rs_w, rs_l)
        p_w = SURFACE_BLEND * p_surface + (1 - SURFACE_BLEND) * p_overall

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
    print(f"{TOUR.upper()} RANK-WARMSTART ELO")
    print("=" * 75)

    print(f"\n[1/5] Loading {TOUR.upper()} data 2021-2024...")
    raw = load_years([2021, 2022, 2023, 2024], tour=TOUR)
    matches = raw[raw["w_svpt"].notna() & raw["l_svpt"].notna()].copy()
    matches["date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")
    matches = matches.dropna(subset=["date", "surface", "tourney_name"])
    matches["surface"] = matches["surface"].str.strip()
    matches = matches.sort_values("date").reset_index(drop=True)
    print(f"      {len(matches)} matches.")

    # Diagnostic on tier codes (since WTA may differ from ATP)
    tiers = matches["tourney_level"].value_counts()
    print(f"      Tier distribution: {tiers.to_dict()}")
    unknown_tiers = set(tiers.index) - set(K_BY_LEVEL.keys())
    if unknown_tiers:
        print(f"      WARNING: unknown tier codes will use default K={DEFAULT_K}: {unknown_tiers}")

    name_map = {}
    for _, r in raw.iterrows():
        if pd.notna(r.get("winner_id")) and pd.notna(r.get("winner_name")):
            name_map[int(r["winner_id"])] = r["winner_name"]
        if pd.notna(r.get("loser_id")) and pd.notna(r.get("loser_name")):
            name_map[int(r["loser_id"])] = r["loser_name"]

    print(f"\n[2/5] Fitting rank → Elo mapping (pre-test window only)...")
    pretest_matches = matches[matches["date"] < TEST_START]
    print(f"      Using {len(pretest_matches)} pre-test matches.")
    fit = fit_rank_to_rating(pretest_matches)
    print(f"      Elo ≈ {fit['a']:.1f} + {fit['b']:.1f} * log(rank)")
    print(f"      n_pairs: {fit['n_pairs']}, R²: {fit['r_squared']:.3f}")
    print(f"      Examples:")
    for r in [1, 5, 10, 25, 50, 100, 200, 500]:
        rating = fit["rank_to_rating"](r)
        print(f"        rank {r:>4}  →  rating {rating:.0f}")

    print(f"\n[3/5] Walk-forward warmstart model...")
    pred = fit_warmstart_model(matches, fit["rank_to_rating"], TEST_START, TEST_END)
    print(f"      {len(pred)} predictions in test window.")
    print(f"      Cold-start matches: {pred['is_cold_start'].sum()} ({pred['is_cold_start'].mean():.1%})")

    print(f"\n[4/5] Loading {TOUR.upper()} Pinnacle odds...")
    odds = load_odds_years([2024], tour=TOUR)
    print(f"      Loaded {len(odds)} odds rows.")

    merged = merge_predictions_with_odds(pred, odds, name_map)
    print(f"      Matched {len(merged)} / {len(pred)} predictions ({len(merged)/len(pred):.1%})")

    if len(merged) < 100:
        print("Too few merged matches — stopping. Investigate name-matching.")
        merged.to_csv(f"{TOUR}_warmstart_predictions.csv", index=False)
        return

    print(f"\n[5/5] Evaluation...")
    y = merged["actual_a_won"].values
    p_warm = merged["p_a_warmstart"].values
    p_pinn = merged["fair_prob_a"].values

    ll_warm = evaluate_logloss(p_warm, y)
    ll_pinn = evaluate_logloss(p_pinn, y)

    print("\n" + "=" * 75)
    print(f"{TOUR.upper()} HEAD-TO-HEAD")
    print("=" * 75)
    print(f"  N matches:      {len(merged)}")
    print(f"  Mean vig:       {merged['vig'].mean():.4f}")
    print(f"  Warmstart Elo:  {ll_warm:.4f}")
    print(f"  Pinnacle:       {ll_pinn:.4f}")
    print(f"  Gap:            {ll_warm - ll_pinn:+.4f} nats")
    print(f"\n  ATP reference (from rank_warmstart.py): gap was +0.0417")

    # Cold-start split
    cold = merged[merged["is_cold_start"] == True]
    warm = merged[merged["is_cold_start"] == False]
    print(f"\n  Cold-start (n={len(cold)}):")
    if len(cold) >= 30:
        ll_w_c = evaluate_logloss(cold["p_a_warmstart"].values, cold["actual_a_won"].values)
        ll_p_c = evaluate_logloss(cold["fair_prob_a"].values, cold["actual_a_won"].values)
        print(f"    Warmstart: {ll_w_c:.4f}, Pinnacle: {ll_p_c:.4f}, gap: {ll_w_c - ll_p_c:+.4f}")
    print(f"  Warm matches (n={len(warm)}):")
    if len(warm) >= 30:
        ll_w_w = evaluate_logloss(warm["p_a_warmstart"].values, warm["actual_a_won"].values)
        ll_p_w = evaluate_logloss(warm["fair_prob_a"].values, warm["actual_a_won"].values)
        print(f"    Warmstart: {ll_w_w:.4f}, Pinnacle: {ll_p_w:.4f}, gap: {ll_w_w - ll_p_w:+.4f}")

    # By tier
    print(f"\n  By tier:")
    print(f"    {'tier':<6}{'n':>5}{'warm_ll':>10}{'pinn_ll':>10}{'gap':>9}")
    for tier in matches["tourney_level"].unique():
        sub = merged[merged["tourney_level"] == tier]
        if len(sub) >= 30:
            llw = evaluate_logloss(sub["p_a_warmstart"].values, sub["actual_a_won"].values)
            llp = evaluate_logloss(sub["fair_prob_a"].values, sub["actual_a_won"].values)
            print(f"    {tier:<6}{len(sub):>5}{llw:>10.4f}{llp:>10.4f}{llw-llp:>+9.4f}")

    # By disagreement bucket — the audit's most important slice
    print(f"\n  By disagreement (where is the model competitive?)")
    diff = np.abs(p_warm - p_pinn)
    print(f"    {'disagreement':<22}{'n':>5}{'warm_ll':>10}{'pinn_ll':>10}{'gap':>9}")
    for label, lo, hi in [
        ("agree (<5pp)", 0.0, 0.05),
        ("mild (5-10pp)", 0.05, 0.10),
        ("medium (10-15pp)", 0.10, 0.15),
        ("large (15-25pp)", 0.15, 0.25),
        ("huge (25%+)", 0.25, 1.01),
    ]:
        mask = (diff >= lo) & (diff < hi)
        if mask.sum() >= 30:
            llw = evaluate_logloss(p_warm[mask], y[mask])
            llp = evaluate_logloss(p_pinn[mask], y[mask])
            print(f"    {label:<22}{mask.sum():>5}{llw:>10.4f}{llp:>10.4f}{llw-llp:>+9.4f}")

    # By Pinnacle favorite strength
    print(f"\n  By favorite strength:")
    fav_strength = np.maximum(p_pinn, 1 - p_pinn)
    print(f"    {'fav_strength':<22}{'n':>5}{'warm_ll':>10}{'pinn_ll':>10}{'gap':>9}")
    for label, lo, hi in [
        ("toss-up (.50-.55)", 0.50, 0.55),
        ("light fav (.55-.65)", 0.55, 0.65),
        ("medium fav (.65-.75)", 0.65, 0.75),
        ("heavy fav (.75-.85)", 0.75, 0.85),
        ("vy heavy (.85-.95)", 0.85, 0.95),
        ("locks (.95+)", 0.95, 1.01),
    ]:
        mask = (fav_strength >= lo) & (fav_strength < hi)
        if mask.sum() >= 30:
            llw = evaluate_logloss(p_warm[mask], y[mask])
            llp = evaluate_logloss(p_pinn[mask], y[mask])
            print(f"    {label:<22}{mask.sum():>5}{llw:>10.4f}{llp:>10.4f}{llw-llp:>+9.4f}")

    # Calibration table
    print(f"\n  Calibration (bucketed by Pinnacle):")
    print(f"    {'bucket':<14}{'n':>5}{'warm':>9}{'pinn':>9}{'actual':>9}{'gap_warm':>10}")
    bins = np.linspace(0, 1, 11)
    bucket = np.clip(np.digitize(p_pinn, bins) - 1, 0, 9)
    for b in range(10):
        mask = bucket == b
        if mask.sum() >= 20:
            wp = p_warm[mask].mean()
            pp = p_pinn[mask].mean()
            ac = y[mask].mean()
            print(f"    {bins[b]:.1f}-{bins[b+1]:.1f}        {mask.sum():>5}{wp:>9.3f}"
                  f"{pp:>9.3f}{ac:>9.3f}{ac-wp:>+10.3f}")

    merged.to_csv(f"{TOUR}_warmstart_predictions.csv", index=False)
    print(f"\n  Saved to {TOUR}_warmstart_predictions.csv")


if __name__ == "__main__":
    main()
