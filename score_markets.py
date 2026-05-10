"""
Score-market probabilities derived from point-win probabilities.

Given p_a, p_b (per-point win probs on each player's serve), compute:
  - P(A wins by exact set count): A wins 2-0, 2-1, etc.
  - P(B wins ≥1 set)             — A doesn't sweep
  - P(over/under N total games)
  - P(A wins set 1)              — independent of match (first set)
  - Set-handicap markets         — A wins by ≥1 set, by ≥2 sets, etc.

The straightforward way: simulate a large set distribution, then derive each
market from it. We use exact probabilities by enumerating all set sequences
since Bo3 has 6 possible outcomes (2-0, 2-1 in either direction, plus B
sweeping) and Bo5 has 20.

For total games we need the full distribution over (games_won_by_A,
games_won_by_B) per set, which requires more work but is still tractable.
"""

from functools import lru_cache
from markov import p_set, p_game, p_tiebreak


# ---------- SET-LEVEL OUTCOMES ----------

def set_outcome_probs(p_a: float, p_b: float) -> dict:
    """
    Returns probabilities of each possible set score, from A's perspective.
    Keys: (a_games, b_games), values: probability.
    
    Possible set outcomes: 6-0, 6-1, 6-2, 6-3, 6-4, 7-5, 7-6, and the
    same scores with A and B swapped.
    
    Computed via DP over (a_games, b_games, n_games_played).
    """
    pg_a = p_game(p_a)
    pg_b = p_game(p_b)

    # Probability of each (ga, gb) end state. We do forward DP.
    # State: (ga, gb, n_played). Each (ga, gb) terminal is a set outcome.
    out = {}

    def add(state_prob, key):
        out[key] = out.get(key, 0.0) + state_prob

    # We'll do BFS-style: at each game, branch.
    # Use a dict of (ga, gb, n_played) -> probability.
    states = {(0, 0, 0): 1.0}
    while states:
        new_states = {}
        for (ga, gb, n), prob in states.items():
            # Terminal conditions
            if ga == 6 and gb <= 4:
                add(prob, (6, gb))
                continue
            if gb == 6 and ga <= 4:
                add(prob, (ga, 6))
                continue
            if ga == 7 and gb == 5:
                add(prob, (7, 5))
                continue
            if gb == 7 and ga == 5:
                add(prob, (5, 7))
                continue
            if ga == 6 and gb == 6:
                # Tiebreak: A wins 7-6, B wins 6-7
                pt = p_tiebreak(p_a, p_b)
                add(prob * pt, (7, 6))
                add(prob * (1 - pt), (6, 7))
                continue
            # Non-terminal: who serves this game?
            a_serving = (n % 2 == 0)
            if a_serving:
                p_a_wins_game = pg_a
            else:
                p_a_wins_game = 1 - pg_b
            new_states[(ga + 1, gb, n + 1)] = new_states.get((ga + 1, gb, n + 1), 0.0) + prob * p_a_wins_game
            new_states[(ga, gb + 1, n + 1)] = new_states.get((ga, gb + 1, n + 1), 0.0) + prob * (1 - p_a_wins_game)
        states = new_states

    return out


# ---------- MATCH-LEVEL DERIVED MARKETS ----------

def match_market_probs(p_a: float, p_b: float, best_of: int = 3) -> dict:
    """
    Returns a dict of derived market probabilities.

    Note on independence assumption: we treat sets as i.i.d. — same per-point
    probs every set, no momentum. This means a "won 1st set" doesn't shift
    point probs in subsequent sets. Real tennis has within-match correlation
    that this misses; we'll quantify it empirically later.
    """
    p_set_a = p_set(p_a, p_b)
    p_set_b = 1 - p_set_a

    sets_to_win = best_of // 2 + 1

    # Enumerate all sequences. For Bo3 with sets_to_win=2:
    #   AA, BAA, ABA  -> A wins 2-0 (1 way), 2-1 (2 ways)
    #   BB, ABB, BAB  -> B wins
    # Use direct combinatorics.
    from math import comb
    # P(A wins in exactly k sets, where k in {sets_to_win, ..., best_of})
    # = C(k-1, sets_to_win - 1) * p_set_a^sets_to_win * p_set_b^(k - sets_to_win)
    p_a_2_0 = None  # set counts depend on best_of
    a_set_count_probs = {}  # (a_sets, b_sets) -> prob
    b_set_count_probs = {}

    for k in range(sets_to_win, best_of + 1):
        # A wins in k sets: A wins last set, won (sets_to_win - 1) of first (k-1)
        p = comb(k - 1, sets_to_win - 1) * (p_set_a ** sets_to_win) * (p_set_b ** (k - sets_to_win))
        a_set_count_probs[(sets_to_win, k - sets_to_win)] = p
        # B wins in k sets
        p2 = comb(k - 1, sets_to_win - 1) * (p_set_b ** sets_to_win) * (p_set_a ** (k - sets_to_win))
        b_set_count_probs[(k - sets_to_win, sets_to_win)] = p2

    p_a_wins = sum(a_set_count_probs.values())
    p_b_wins = sum(b_set_count_probs.values())

    out = {
        "p_match_a": p_a_wins,
        "p_set": p_set_a,
        "set_count_probs_a": a_set_count_probs,  # (a_sets, b_sets) -> prob
        "set_count_probs_b": b_set_count_probs,
    }

    # Common derived markets
    if best_of == 3:
        out["p_a_2_0"] = a_set_count_probs.get((2, 0), 0.0)
        out["p_a_2_1"] = a_set_count_probs.get((2, 1), 0.0)
        out["p_b_2_0"] = b_set_count_probs.get((0, 2), 0.0)
        out["p_b_2_1"] = b_set_count_probs.get((1, 2), 0.0)
        out["p_a_wins_at_least_one_set"] = (
            a_set_count_probs.get((2, 0), 0.0)
            + a_set_count_probs.get((2, 1), 0.0)
            + b_set_count_probs.get((1, 2), 0.0)
        )
        out["p_b_wins_at_least_one_set"] = (
            b_set_count_probs.get((0, 2), 0.0)
            + b_set_count_probs.get((1, 2), 0.0)
            + a_set_count_probs.get((2, 1), 0.0)
        )
        out["p_match_total_sets_2"] = (
            a_set_count_probs.get((2, 0), 0.0)
            + b_set_count_probs.get((0, 2), 0.0)
        )
        out["p_match_total_sets_3"] = (
            a_set_count_probs.get((2, 1), 0.0)
            + b_set_count_probs.get((1, 2), 0.0)
        )
    elif best_of == 5:
        # A wins 3-0, 3-1, 3-2; B wins 0-3, 1-3, 2-3
        out["p_a_3_0"] = a_set_count_probs.get((3, 0), 0.0)
        out["p_a_3_1"] = a_set_count_probs.get((3, 1), 0.0)
        out["p_a_3_2"] = a_set_count_probs.get((3, 2), 0.0)
        out["p_b_3_0"] = b_set_count_probs.get((0, 3), 0.0)
        out["p_b_3_1"] = b_set_count_probs.get((1, 3), 0.0)
        out["p_b_3_2"] = b_set_count_probs.get((2, 3), 0.0)
        out["p_a_wins_at_least_one_set"] = (
            sum(p for (a, b), p in a_set_count_probs.items() if a >= 1)
            + sum(p for (a, b), p in b_set_count_probs.items() if a >= 1)
        )
        out["p_b_wins_at_least_one_set"] = (
            sum(p for (a, b), p in a_set_count_probs.items() if b >= 1)
            + sum(p for (a, b), p in b_set_count_probs.items() if b >= 1)
        )

    return out


# ---------- TOTAL GAMES (heavier compute) ----------

def total_games_distribution(p_a: float, p_b: float, best_of: int = 3) -> dict:
    """
    Returns dict: total_games -> probability.

    Computed by combining set-game distributions across the match.
    Approximation: we use the marginal set-outcome distribution and
    treat sets as i.i.d. in their game count. This is exact for Bo3
    when both players have stationary point probs.
    """
    set_outcomes = set_outcome_probs(p_a, p_b)
    # set_outcomes[(a, b)] = P(set ends a-b from A's perspective)

    # For each set outcome, total games in that set = a + b. But we also need
    # to track who won the set, because match length depends on it.

    sets_to_win = best_of // 2 + 1
    p_set_a = sum(p for (a, b), p in set_outcomes.items() if a > b)

    # Per-set distributions, conditional on who won
    a_won_set_dist = {}  # games -> prob | A won this set
    b_won_set_dist = {}
    for (a, b), p in set_outcomes.items():
        games = a + b
        if a > b:
            a_won_set_dist[games] = a_won_set_dist.get(games, 0.0) + p / p_set_a
        else:
            b_won_set_dist[games] = b_won_set_dist.get(games, 0.0) + p / (1 - p_set_a)

    # Enumerate set sequences by winner
    from math import comb
    total_dist = {}
    for k in range(sets_to_win, best_of + 1):
        # A wins in k sets
        seq_prob_a = comb(k - 1, sets_to_win - 1) * (p_set_a ** sets_to_win) * ((1 - p_set_a) ** (k - sets_to_win))
        # In this case, A wins `sets_to_win` sets, B wins `k - sets_to_win`
        # We need distribution of total_games given this set count.
        # Convolve: sum of `sets_to_win` draws from a_won_set_dist + (k - sets_to_win) from b_won_set_dist.
        n_a_sets = sets_to_win
        n_b_sets = k - sets_to_win
        seq_total_dist = _convolve_set_dists(a_won_set_dist, n_a_sets, b_won_set_dist, n_b_sets)
        for total, p in seq_total_dist.items():
            total_dist[total] = total_dist.get(total, 0.0) + seq_prob_a * p

        # B wins in k sets
        seq_prob_b = comb(k - 1, sets_to_win - 1) * ((1 - p_set_a) ** sets_to_win) * (p_set_a ** (k - sets_to_win))
        n_a_sets = k - sets_to_win
        n_b_sets = sets_to_win
        seq_total_dist = _convolve_set_dists(a_won_set_dist, n_a_sets, b_won_set_dist, n_b_sets)
        for total, p in seq_total_dist.items():
            total_dist[total] = total_dist.get(total, 0.0) + seq_prob_b * p

    return total_dist


def _convolve_set_dists(a_dist, n_a, b_dist, n_b):
    """Distribution of sum of n_a draws from a_dist + n_b draws from b_dist."""
    # Start with degenerate 0
    result = {0: 1.0}
    for _ in range(n_a):
        new = {}
        for total, p1 in result.items():
            for g, p2 in a_dist.items():
                new[total + g] = new.get(total + g, 0.0) + p1 * p2
        result = new
    for _ in range(n_b):
        new = {}
        for total, p1 in result.items():
            for g, p2 in b_dist.items():
                new[total + g] = new.get(total + g, 0.0) + p1 * p2
        result = new
    return result


def p_total_games_over(p_a: float, p_b: float, line: float, best_of: int = 3) -> float:
    """P(total games > line)."""
    dist = total_games_distribution(p_a, p_b, best_of)
    return sum(p for g, p in dist.items() if g > line)


if __name__ == "__main__":
    # Sanity: equal players
    p_a = p_b = 0.62
    out = match_market_probs(p_a, p_b, best_of=3)
    print(f"Equal players (p_a=p_b=0.62), Bo3:")
    for k, v in out.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    # Should sum to ~1
    s = (out["p_a_2_0"] + out["p_a_2_1"] + out["p_b_2_0"] + out["p_b_2_1"])
    print(f"  Sum of set-counts: {s:.4f} (should be 1.0)")
    print(f"  P(A wins) + P(B wins) = {out['p_match_a'] + (1 - out['p_match_a']):.4f}")

    # Asymmetric
    print(f"\nAsymmetric (p_a=0.68, p_b=0.58), Bo3:")
    out = match_market_probs(0.68, 0.58, best_of=3)
    for k, v in out.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    # Total games
    print(f"\nTotal games distribution (p_a=0.65, p_b=0.60, Bo3):")
    dist = total_games_distribution(0.65, 0.60, best_of=3)
    cum = 0.0
    for g in sorted(dist.keys()):
        cum += dist[g]
        if 12 <= g <= 35:
            print(f"  {g}: {dist[g]:.4f}  (cum: {cum:.4f})")
    print(f"  P(over 21.5) = {p_total_games_over(0.65, 0.60, 21.5, 3):.4f}")
    print(f"  P(over 22.5) = {p_total_games_over(0.65, 0.60, 22.5, 3):.4f}")
    print(f"  Total prob mass: {sum(dist.values()):.4f}")
