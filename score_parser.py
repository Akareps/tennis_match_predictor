"""
Parse Sackmann's match `score` field into structured set/game data.

Sackmann score format examples:
  "6-4 6-3"               — straight sets, A won
  "6-4 3-6 7-5"           — three sets
  "6-3 6-7(4) 7-6(2)"     — with tiebreak score in parens
  "6-2 6-7(5) 7-6(3) 6-1" — Bo5
  "W/O", "RET", "DEF"     — walkover/retirement (skip)

This module returns lists of (winner_games, loser_games) per set, from the
WINNER's perspective. To convert to player_a perspective, swap if A != winner.
"""

import re


def parse_score(score_str: str) -> list[tuple[int, int]] | None:
    """
    Returns list of (winner_games, loser_games) tuples, one per set.
    Returns None for walkovers, retirements, or unparseable strings.
    """
    if not isinstance(score_str, str) or not score_str.strip():
        return None
    s = score_str.strip()
    # Skip incomplete matches
    if any(tok in s.upper() for tok in ["W/O", "WO", "RET", "DEF", "ABN"]):
        return None

    sets = []
    # Tokenize on whitespace
    for token in s.split():
        # Strip tiebreak score in parens, e.g. "7-6(4)" -> "7-6"
        m = re.match(r"^(\d+)-(\d+)(?:\(\d+\))?$", token)
        if not m:
            return None
        w_games, l_games = int(m.group(1)), int(m.group(2))
        # Sanity: scores should be 0-7 typically
        if w_games > 7 or l_games > 7:
            # Could be a "10-8" super-tiebreak in doubles or a final-set
            # extension. For our purposes (Bo3 / Bo5 singles) we'll accept
            # up to 13 to be safe but flag oddities.
            if w_games > 13 or l_games > 13:
                return None
        sets.append((w_games, l_games))

    if not sets:
        return None
    return sets


def n_sets_played(parsed: list[tuple[int, int]]) -> int:
    return len(parsed)


def winner_won_in_n_sets(parsed: list[tuple[int, int]]) -> int:
    """How many sets did the winner win?"""
    return sum(1 for w, l in parsed if w > l)


def loser_won_at_least_one_set(parsed: list[tuple[int, int]]) -> bool:
    return any(l > w for w, l in parsed)


def total_games(parsed: list[tuple[int, int]]) -> int:
    return sum(w + l for w, l in parsed)


if __name__ == "__main__":
    # Smoke tests
    cases = [
        ("6-4 6-3", [(6, 4), (6, 3)]),
        ("6-4 3-6 7-5", [(6, 4), (3, 6), (7, 5)]),
        ("6-3 6-7(4) 7-6(2)", [(6, 3), (6, 7), (7, 6)]),
        ("6-2 6-7(5) 7-6(3) 6-1", [(6, 2), (6, 7), (7, 6), (6, 1)]),
        ("W/O", None),
        ("3-6 RET", None),
        ("", None),
    ]
    for s, expected in cases:
        got = parse_score(s)
        ok = got == expected
        print(f"  {'OK' if ok else 'FAIL'}: {s!r:30} -> {got}")
