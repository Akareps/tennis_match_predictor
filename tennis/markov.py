"""
Markov chain probabilities for tennis: point -> game -> set -> match.

Inputs: p_a = P(A wins point on A's serve), p_b = P(B wins point on B's serve).
All functions return P(A wins ...). Assumes points are i.i.d.
"""

from functools import lru_cache
from math import comb


# ---------- GAME ----------

@lru_cache(maxsize=None)
def p_game(p: float) -> float:
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    q = 1 - p
    win_no_deuce = p**4 + 4 * p**4 * q + 10 * p**4 * q**2
    p_deuce = 20 * p**3 * q**3
    p_win_from_deuce = p**2 / (p**2 + q**2)
    return win_no_deuce + p_deuce * p_win_from_deuce


# ---------- TIEBREAK ----------

def p_tiebreak(p_a: float, p_b: float, target: int = 7) -> float:
    """P(A wins tiebreak). A serves point 0, then alternating pairs."""
    q_a = 1 - p_a
    q_b = 1 - p_b

    def srv(n: int) -> int:
        if n == 0:
            return 0
        return ((n - 1) // 2 + 1) % 2

    cache = {}

    def f(a, b, n):
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        # When tied at >= target-1, collapse to closed-form "win by 2".
        if a == b and a >= target - 1:
            return _win_from_tied_extension(p_a, p_b, n)
        if (a, b, n) in cache:
            return cache[(a, b, n)]
        s = srv(n)
        if s == 0:
            res = p_a * f(a + 1, b, n + 1) + q_a * f(a, b + 1, n + 1)
        else:
            res = q_b * f(a + 1, b, n + 1) + p_b * f(a, b + 1, n + 1)
        cache[(a, b, n)] = res
        return res

    return f(0, 0, 0)


def _win_from_tied_extension(p_a: float, p_b: float, n: int) -> float:
    """Closed-form P(A wins) from a tied score in tiebreak win-by-2 phase."""
    def srv(k):
        if k == 0:
            return 0
        return ((k - 1) // 2 + 1) % 2
    s1, s2 = srv(n), srv(n + 1)
    pa1 = p_a if s1 == 0 else (1 - p_b)
    pa2 = p_a if s2 == 0 else (1 - p_b)
    p_plus2 = pa1 * pa2
    p_minus2 = (1 - pa1) * (1 - pa2)
    if p_plus2 + p_minus2 == 0:
        return 0.5
    return p_plus2 / (p_plus2 + p_minus2)


# ---------- SET ----------

def p_set(p_a: float, p_b: float) -> float:
    """P(A wins a 6-game set with tiebreak at 6-6. A serves first game."""
    pg_a = p_game(p_a)
    p_a_break = 1 - p_game(p_b)

    cache = {}

    def f(ga, gb, n_games):
        if ga == 6 and gb <= 4:
            return 1.0
        if gb == 6 and ga <= 4:
            return 0.0
        if ga == 7:
            return 1.0
        if gb == 7:
            return 0.0
        if ga == 6 and gb == 6:
            return p_tiebreak(p_a, p_b, target=7)
        if (ga, gb) in cache:
            return cache[(ga, gb)]
        a_serving = (n_games % 2 == 0)
        if a_serving:
            res = pg_a * f(ga + 1, gb, n_games + 1) + (1 - pg_a) * f(ga, gb + 1, n_games + 1)
        else:
            res = p_a_break * f(ga + 1, gb, n_games + 1) + (1 - p_a_break) * f(ga, gb + 1, n_games + 1)
        cache[(ga, gb)] = res
        return res

    return f(0, 0, 0)


# ---------- MATCH ----------

def p_match(p_a: float, p_b: float, best_of: int = 3) -> float:
    """P(A wins match). Sets treated as i.i.d."""
    ps = p_set(p_a, p_b)
    sets_to_win = best_of // 2 + 1
    total = 0.0
    for k in range(sets_to_win):
        total += comb(sets_to_win - 1 + k, k) * (ps ** sets_to_win) * ((1 - ps) ** k)
    return total


if __name__ == "__main__":
    assert abs(p_game(0.65) - 0.830) < 0.01, p_game(0.65)
    assert abs(p_set(0.65, 0.65) - 0.5) < 1e-3, p_set(0.65, 0.65)
    assert abs(p_match(0.65, 0.65) - 0.5) < 1e-3
    assert abs(p_tiebreak(0.65, 0.65) - 0.5) < 1e-3

    print(f"p_game(0.60) = {p_game(0.60):.4f}  (avg server holds)")
    print(f"p_game(0.65) = {p_game(0.65):.4f}")
    print(f"p_game(0.70) = {p_game(0.70):.4f}  (big server)")
    print(f"p_tiebreak(0.70, 0.60) = {p_tiebreak(0.70, 0.60):.4f}")
    print(f"p_set(0.68, 0.62) = {p_set(0.68, 0.62):.4f}")
    print(f"p_match(0.68, 0.62, Bo3) = {p_match(0.68, 0.62, 3):.4f}")
    print(f"p_match(0.68, 0.62, Bo5) = {p_match(0.68, 0.62, 5):.4f}")
    print(f"p_match(0.70, 0.60) = {p_match(0.70, 0.60):.4f}")
    print(f"p_match(0.75, 0.55) = {p_match(0.75, 0.55):.4f}")
    print("All sanity checks passed.")
