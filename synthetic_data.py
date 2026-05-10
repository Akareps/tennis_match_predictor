"""
Synthetic match data generator for testing the pipeline offline.

Produces a long-format DataFrame matching the schema of data_loader.to_long_format,
so it's a drop-in for development. Real Sackmann data should be used in production.
"""

import numpy as np
import pandas as pd


def generate_synthetic_matches(
    n_players: int = 100,
    n_matches: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate matches between synthetic players. Each player has a true latent
    serve skill and return skill on each surface; we sample matches and
    simulate the aggregate stats (svpt, sv_won, etc.) consistent with those.
    """
    rng = np.random.default_rng(seed)

    surfaces = ["Hard", "Clay", "Grass"]
    surface_weights = [0.55, 0.35, 0.10]

    # Latent skills: serve and return % per surface for each player
    # Center serve % around 0.62 (ATP avg), with realistic spread
    serve_skill = {}  # player_id -> {surface: spw}
    return_skill = {}
    player_names = {}
    for pid in range(n_players):
        # Each player has a "talent" baseline and surface-specific shifts
        baseline_serve = rng.normal(0.62, 0.04)  # 0.54-0.70 typical range
        baseline_return = rng.normal(0.38, 0.03)
        serve_skill[pid] = {}
        return_skill[pid] = {}
        for s in surfaces:
            surf_offset_serve = rng.normal(0, 0.02)
            surf_offset_return = rng.normal(0, 0.015)
            serve_skill[pid][s] = np.clip(baseline_serve + surf_offset_serve, 0.45, 0.78)
            return_skill[pid][s] = np.clip(baseline_return + surf_offset_return, 0.25, 0.50)
        player_names[pid] = f"Player_{pid:03d}"

    # Generate matches over a 3-year window
    rows = []
    start_date = pd.Timestamp("2022-01-01")
    for i in range(n_matches):
        a, b = rng.choice(n_players, 2, replace=False)
        surface = rng.choice(surfaces, p=surface_weights)
        date = start_date + pd.Timedelta(days=int(rng.integers(0, 3 * 365)))
        best_of = 5 if rng.random() < 0.05 else 3   # ~5% Slams in Bo5

        # Compute point-win probs using Barnett-Clarke style:
        # spw_a = avg + (serve_a - avg_serve) - (return_b - avg_return)
        AVG_SERVE = 0.62
        AVG_RET = 0.38
        spw_a = serve_skill[a][surface] + (AVG_RET - return_skill[b][surface])
        spw_b = serve_skill[b][surface] + (AVG_RET - return_skill[a][surface])
        # Clip to reasonable range
        spw_a = float(np.clip(spw_a, 0.40, 0.85))
        spw_b = float(np.clip(spw_b, 0.40, 0.85))

        # Simulate match: who wins? Use the markov module's match prob
        from markov import p_match
        p_a_wins = p_match(spw_a, spw_b, best_of=best_of)
        a_won = rng.random() < p_a_wins

        # Simulate aggregate svpt counts
        # Typical match: ~70-90 svpt per player in Bo3, ~110-140 in Bo5
        if best_of == 3:
            svpt_a = int(rng.normal(80, 12))
            svpt_b = int(rng.normal(80, 12))
        else:
            svpt_a = int(rng.normal(125, 20))
            svpt_b = int(rng.normal(125, 20))
        svpt_a = max(40, svpt_a)
        svpt_b = max(40, svpt_b)
        sv_won_a = rng.binomial(svpt_a, spw_a)
        sv_won_b = rng.binomial(svpt_b, spw_b)

        winner_id, loser_id = (a, b) if a_won else (b, a)
        if a_won:
            w_svpt, w_sv_won = svpt_a, sv_won_a
            l_svpt, l_sv_won = svpt_b, sv_won_b
        else:
            w_svpt, w_sv_won = svpt_b, sv_won_b
            l_svpt, l_sv_won = svpt_a, sv_won_a

        rows.append({
            "date": date,
            "surface": surface,
            "tourney_level": "M" if rng.random() < 0.3 else "A",
            "best_of": best_of,
            "winner_id": int(winner_id),
            "winner_name": player_names[winner_id],
            "loser_id": int(loser_id),
            "loser_name": player_names[loser_id],
            "w_svpt": w_svpt,
            "w_sv_won": w_sv_won,
            "l_svpt": l_svpt,
            "l_sv_won": l_sv_won,
        })

    df = pd.DataFrame(rows)

    # Convert to long format (two rows per match, one per player perspective)
    w = pd.DataFrame({
        "date": df["date"],
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "best_of": df["best_of"],
        "player_id": df["winner_id"],
        "player_name": df["winner_name"],
        "opp_id": df["loser_id"],
        "opp_name": df["loser_name"],
        "won": True,
        "svpt": df["w_svpt"],
        "sv_won": df["w_sv_won"],
        "rpt": df["l_svpt"],
        "rpt_won": df["l_svpt"] - df["l_sv_won"],
    })
    l = pd.DataFrame({
        "date": df["date"],
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "best_of": df["best_of"],
        "player_id": df["loser_id"],
        "player_name": df["loser_name"],
        "opp_id": df["winner_id"],
        "opp_name": df["winner_name"],
        "won": False,
        "svpt": df["l_svpt"],
        "sv_won": df["l_sv_won"],
        "rpt": df["w_svpt"],
        "rpt_won": df["w_svpt"] - df["w_sv_won"],
    })
    long = pd.concat([w, l], ignore_index=True).sort_values("date").reset_index(drop=True)
    long["spw"] = long["sv_won"] / long["svpt"]
    long["rpw"] = long["rpt_won"] / long["rpt"]

    # Save the ground-truth skills so we can evaluate
    truth = []
    for pid in range(n_players):
        for s in surfaces:
            truth.append({
                "player_id": pid,
                "player_name": player_names[pid],
                "surface": s,
                "true_serve_skill": serve_skill[pid][s],
                "true_return_skill": return_skill[pid][s],
            })
    truth_df = pd.DataFrame(truth)

    return long, truth_df


if __name__ == "__main__":
    long, truth = generate_synthetic_matches(n_players=100, n_matches=5000)
    print(f"Generated {len(long)} player-match rows ({len(long)//2} matches)")
    print(long.head())
    print(f"\nMean spw: {long['spw'].mean():.4f} (target ~0.62)")
    print(f"Mean rpw: {long['rpw'].mean():.4f} (target ~0.38)")
    print(f"\nSurface breakdown:")
    print(long['surface'].value_counts())
    print(f"\nGround-truth skill table sample:")
    print(truth.head())
