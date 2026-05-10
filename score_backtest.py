"""
Score-market backtest.

Like backtest.py but ALSO saves:
  - spw_a, spw_b           (per-point win probs — needed to derive markets)
  - actual_score_string    (Sackmann score field for the actual match)
  - parsed score data:
      a_sets_won, b_sets_won, total_games, a_won_at_least_one_set, etc.

Output: predictions_score.csv

This is the bridge between the model and score-market evaluation.
"""

import pandas as pd
import numpy as np
from data_loader import load_years
from skill_estimation import fit_skills, _sigmoid
from score_parser import parse_score


# Best hyperparams from sweep
RIDGE = 12.0
HALF_LIFE_DAYS = 365
MIN_OBS = 10
REFIT_FREQ_DAYS = 14
SHRINK_ALPHA = 0.20  # match-level shrinkage from earlier finding


def fit_long_format_with_scores(df_raw):
    """
    Like to_long_format but ALSO retains the score string per match.
    Returns long format + a separate match_id mapping.
    """
    keep = ["tourney_date", "surface", "tourney_level", "best_of", "year",
            "winner_id", "winner_name", "loser_id", "loser_name",
            "score",
            "w_svpt", "w_1stWon", "w_2ndWon", "w_SvGms",
            "l_svpt", "l_1stWon", "l_2ndWon", "l_SvGms"]
    df = df_raw[[c for c in keep if c in df_raw.columns]].copy()
    df = df.dropna(subset=["w_svpt", "l_svpt", "surface"])
    df = df[df["w_svpt"] > 0]
    df = df[df["l_svpt"] > 0]
    df["surface"] = df["surface"].str.strip()

    # Winner-perspective rows
    w = pd.DataFrame({
        "date": pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce"),
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "best_of": df["best_of"],
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"],
        "opp_name": df["loser_name"],
        "won": True,
        "svpt": df["w_svpt"],
        "sv_won": df["w_1stWon"] + df["w_2ndWon"],
        "rpt": df["l_svpt"],
        "rpt_won": df["l_svpt"] - (df["l_1stWon"] + df["l_2ndWon"]),
        "score": df["score"],
    })
    l = pd.DataFrame({
        "date": pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce"),
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "best_of": df["best_of"],
        "player_id": df["loser_id"],
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"],
        "opp_name": df["winner_name"],
        "won": False,
        "svpt": df["l_svpt"],
        "sv_won": df["l_1stWon"] + df["l_2ndWon"],
        "rpt": df["w_svpt"],
        "rpt_won": df["w_svpt"] - (df["w_1stWon"] + df["w_2ndWon"]),
        "score": df["score"],
    })
    long = pd.concat([w, l], ignore_index=True)
    long = long.dropna(subset=["date"])
    long["spw"] = long["sv_won"] / long["svpt"]
    long["rpw"] = long["rpt_won"] / long["rpt"]
    return long.sort_values("date").reset_index(drop=True)


def main():
    print("Loading 2021-2024 ATP data...")
    raw = load_years([2021, 2022, 2023, 2024], tour="atp")
    long = fit_long_format_with_scores(raw)
    print(f"  {len(long)//2} matches in long format.")

    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2024-09-30")

    # Build canonical (one row per match) DF for the test window
    df = long[long["player_id"] < long["opp_id"]].copy()
    df = df[(df["date"] >= test_start) & (df["date"] < test_end)]
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} matches in test window (canonicalized).")

    refit_dates = pd.date_range(
        start=test_start - pd.Timedelta(days=REFIT_FREQ_DAYS),
        end=test_end,
        freq=f"{REFIT_FREQ_DAYS}D",
    )
    skill_cache = {}

    def get_skill_table(surface, match_date):
        rd = max([d for d in refit_dates if d <= match_date], default=None)
        if rd is None:
            return None
        key = (surface, rd)
        if key not in skill_cache:
            try:
                skill_cache[key] = fit_skills(
                    long, surface=surface, as_of=rd,
                    half_life_days=HALF_LIFE_DAYS, ridge=RIDGE,
                )
            except ValueError:
                skill_cache[key] = None
        return skill_cache[key]

    out_rows = []
    skipped_score = 0
    for i, r in df.iterrows():
        if r["surface"] not in ("Hard", "Clay", "Grass"):
            continue
        st = get_skill_table(r["surface"], r["date"])
        if st is None:
            continue
        if (st.n_obs.get(r["player_id"], 0) < MIN_OBS or
                st.n_obs.get(r["opp_id"], 0) < MIN_OBS):
            continue

        # Compute per-point probs
        s_a = st.get_serve(r["player_id"])
        rt_a = st.get_return(r["player_id"])
        s_b = st.get_serve(r["opp_id"])
        rt_b = st.get_return(r["opp_id"])
        spw_a = float(_sigmoid(st.intercept + s_a - rt_b))
        spw_b = float(_sigmoid(st.intercept + s_b - rt_a))

        # Parse score
        # NOTE: r["won"] tells us if player_a (=r.player_id since canonical) won
        # the score string is from WINNER's perspective always
        parsed = parse_score(r["score"])
        if parsed is None:
            skipped_score += 1
            continue

        # Convert to player_a's perspective
        a_won_match = bool(r["won"])
        if a_won_match:
            sets_a_perspective = parsed
        else:
            # winner is B, so flip each set
            sets_a_perspective = [(b, a) for (a, b) in parsed]

        a_sets_won = sum(1 for (a, b) in sets_a_perspective if a > b)
        b_sets_won = sum(1 for (a, b) in sets_a_perspective if b > a)
        total_games_actual = sum(a + b for (a, b) in sets_a_perspective)

        out_rows.append({
            "date": r["date"],
            "surface": r["surface"],
            "best_of": int(r["best_of"]),
            "player_a_id": r["player_id"],
            "player_b_id": r["opp_id"],
            "spw_a": spw_a,
            "spw_b": spw_b,
            "actual_score": r["score"],
            "actual_a_won": int(a_won_match),
            "actual_a_sets": a_sets_won,
            "actual_b_sets": b_sets_won,
            "actual_total_games": total_games_actual,
            "actual_a_won_at_least_one_set": int(a_sets_won >= 1),
            "actual_b_won_at_least_one_set": int(b_sets_won >= 1),
        })

    out_df = pd.DataFrame(out_rows)
    print(f"\n  Built {len(out_df)} predictions with scores.")
    print(f"  Skipped {skipped_score} due to unparseable scores (W/O, RET, etc.)")

    out_path = "predictions_score.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    # Quick sanity: distribution of score outcomes
    print(f"\n  Distribution of actual outcomes:")
    print(f"    Bo3 matches: {(out_df['best_of'] == 3).sum()}")
    print(f"    Bo5 matches: {(out_df['best_of'] == 5).sum()}")
    print(f"    A won 2-0:   {((out_df['best_of']==3) & (out_df['actual_a_sets']==2) & (out_df['actual_b_sets']==0)).sum()}")
    print(f"    A won 2-1:   {((out_df['best_of']==3) & (out_df['actual_a_sets']==2) & (out_df['actual_b_sets']==1)).sum()}")
    print(f"    B won 2-0:   {((out_df['best_of']==3) & (out_df['actual_a_sets']==0) & (out_df['actual_b_sets']==2)).sum()}")
    print(f"    B won 2-1:   {((out_df['best_of']==3) & (out_df['actual_a_sets']==1) & (out_df['actual_b_sets']==2)).sum()}")
    print(f"  Mean total games: {out_df['actual_total_games'].mean():.1f}")


if __name__ == "__main__":
    main()
