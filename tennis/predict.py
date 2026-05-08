"""
Match prediction: bring it all together.

Given a fitted SkillTable and two players, produce P(A wins match).
This wraps: skill lookup -> point-win probs -> Markov match prob.
"""

import numpy as np
from skill_estimation import SkillTable, _sigmoid
from markov import p_match, p_set, p_game


def predict_match(
    skill_table: SkillTable,
    player_a_id: int,
    player_b_id: int,
    best_of: int = 3,
) -> dict:
    """
    Returns dict with point/game/set/match probabilities for player A.

    Note: skill_table must be for the correct surface.
    """
    # Get logit-scale skills
    s_a = skill_table.get_serve(player_a_id)
    r_a = skill_table.get_return(player_a_id)
    s_b = skill_table.get_serve(player_b_id)
    r_b = skill_table.get_return(player_b_id)
    intercept = skill_table.intercept

    # Point-win probs on each player's serve
    spw_a = _sigmoid(intercept + s_a - r_b)
    spw_b = _sigmoid(intercept + s_b - r_a)

    return {
        "spw_a": spw_a,
        "spw_b": spw_b,
        "p_hold_a": p_game(spw_a),
        "p_hold_b": p_game(spw_b),
        "p_set": p_set(spw_a, spw_b),
        "p_match": p_match(spw_a, spw_b, best_of=best_of),
    }


def odds_to_prob(odds: float, vig_method: str = "multiplicative") -> float:
    """
    Convert decimal odds to implied probability. For pair-of-odds (the book's
    margin), use remove_vig() instead.
    """
    return 1.0 / odds


def remove_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """
    Convert two-way market odds to fair probabilities by removing the
    bookmaker's margin. Uses the simple multiplicative method.
    """
    p_a = 1.0 / odds_a
    p_b = 1.0 / odds_b
    total = p_a + p_b
    return p_a / total, p_b / total


def find_value(model_prob: float, decimal_odds: float, threshold: float = 0.05) -> dict:
    """
    EV per unit stake: model_prob * (odds - 1) - (1 - model_prob) * 1
                     = model_prob * odds - 1
    Edge: model_prob - implied_prob (no vig adjustment, since you bet at
    posted odds).

    Returns dict with edge, EV, and recommendation flag.
    """
    implied = 1.0 / decimal_odds
    edge = model_prob - implied
    ev = model_prob * decimal_odds - 1.0
    return {
        "model_prob": model_prob,
        "implied_prob": implied,
        "edge": edge,
        "ev": ev,
        "is_value": ev > threshold,
    }


def kelly_fraction(model_prob: float, decimal_odds: float, kelly_mult: float = 0.25) -> float:
    """
    Fractional Kelly stake as fraction of bankroll.

    Full Kelly: f = (model_prob * (odds - 1) - (1 - model_prob)) / (odds - 1)
              = (model_prob * odds - 1) / (odds - 1)

    Returns 0 if no edge. kelly_mult is the fraction of full Kelly to use
    (1/4 Kelly is standard for surviving model error and variance).
    """
    if decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1
    f_full = (model_prob * decimal_odds - 1) / b
    if f_full <= 0:
        return 0.0
    return kelly_mult * f_full


if __name__ == "__main__":
    import pandas as pd
    from synthetic_data import generate_synthetic_matches
    from skill_estimation import fit_skills

    long, truth = generate_synthetic_matches(n_players=100, n_matches=5000)
    st = fit_skills(long, surface="Hard", as_of=pd.Timestamp("2024-06-01"))

    # Pick the strongest and weakest player on Hard
    truth_hard = truth[truth["surface"] == "Hard"].sort_values(
        "true_serve_skill", ascending=False
    )
    strong = int(truth_hard.iloc[0]["player_id"])
    weak = int(truth_hard.iloc[-1]["player_id"])
    print(f"Strong server: Player {strong} (true serve skill = {truth_hard.iloc[0]['true_serve_skill']:.3f})")
    print(f"Weak server:   Player {weak} (true serve skill = {truth_hard.iloc[-1]['true_serve_skill']:.3f})")

    pred = predict_match(st, strong, weak, best_of=3)
    print(f"\nMatch prediction (Bo3):")
    for k, v in pred.items():
        print(f"  {k}: {v:.4f}")

    # Value bet example: book offers odds, model disagrees
    print(f"\nValue check examples:")
    fair_odds_for_strong = 1.0 / pred["p_match"]
    print(f"  Fair odds for strong player: {fair_odds_for_strong:.2f}")
    print(f"  If book offers {fair_odds_for_strong * 1.10:.2f} (10% above fair):")
    val = find_value(pred["p_match"], fair_odds_for_strong * 1.10)
    print(f"    edge = {val['edge']:.4f}, EV = {val['ev']:.4f}, value? {val['is_value']}")
    print(f"    Kelly stake (1/4 Kelly): {kelly_fraction(pred['p_match'], fair_odds_for_strong * 1.10):.4f}")

    print(f"\n  If book offers {fair_odds_for_strong * 0.95:.2f} (5% below fair):")
    val = find_value(pred["p_match"], fair_odds_for_strong * 0.95)
    print(f"    edge = {val['edge']:.4f}, EV = {val['ev']:.4f}, value? {val['is_value']}")
