"""Outcome-neutral linkage between Sackmann matches and tennis-data odds.

Jeff Sackmann's ``tourney_date`` is the start of an event, not the day on
which every match was played.  This module builds a crosswalk to tennis-data
rows and exposes their actual match dates.  Candidate selection deliberately
uses only player identities, tournament, round, surface, and dates.  Winner
orientation, scores, rankings, and prices are never used to choose a link.

Both raw Sackmann frames (``winner_name``/``loser_name`` and
``tourney_date``) and the outcome-neutral frames produced by
``tennis_experiments.data.canonicalize_matches`` are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping

import pandas as pd


REASON_DESCRIPTIONS: Mapping[str, str] = {
    "matched": "A unique odds row was assigned.",
    "invalid_sackmann_date": "The tournament start date could not be parsed.",
    "invalid_player_name": "One or both Sackmann player names could not be normalized.",
    "same_player_pair": "Both normalized player identities are the same.",
    "no_pair_candidate": "No valid odds row contains the same unordered player pair.",
    "outside_date_window": "Pair candidates exist, but none fall in the allowed event window.",
    "below_min_score": "Date-valid candidates exist, but contextual agreement is too weak.",
    "lost_one_to_one_assignment": "An eligible candidate was assigned to a better competing row.",
}


@dataclass(frozen=True)
class LinkageConfig:
    """Controls the conservative candidate window and contextual score."""

    max_days_before_start: int = 3
    max_days_after_start: int = 21
    min_score: float = 40.0
    tournament_weight: float = 55.0
    round_weight: float = 30.0
    surface_weight: float = 10.0
    date_weight: float = 5.0
    missing_similarity: float = 0.5

    def __post_init__(self) -> None:
        if self.max_days_before_start < 0 or self.max_days_after_start < 0:
            raise ValueError("date-window sizes must be non-negative")
        if not 0.0 <= self.missing_similarity <= 1.0:
            raise ValueError("missing_similarity must be between zero and one")
        weights = (
            self.tournament_weight,
            self.round_weight,
            self.surface_weight,
            self.date_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("linkage weights must be non-negative")
        if self.min_score < 0 or self.min_score > sum(weights):
            raise ValueError("min_score must be within the attainable score range")


@dataclass
class LinkageResult:
    """Crosswalk, per-match diagnostics, and aggregate linkage counts."""

    links: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: dict[str, Any]


_PARTICLES = {
    "al",
    "da",
    "das",
    "de",
    "del",
    "della",
    "den",
    "der",
    "di",
    "do",
    "dos",
    "du",
    "la",
    "le",
    "van",
    "von",
}

_TOURNAMENT_STOPWORDS = {
    "atp",
    "wta",
    "open",
    "international",
    "internationals",
    "championship",
    "championships",
    "championships",
    "tennis",
    "tournament",
    "masters",
    "presented",
    "sponsored",
    "by",
}

_TOURNAMENT_ALIASES = {
    "french": "roland garros",
    "french france": "roland garros",
    "roland garros": "roland garros",
    "italian": "rome",
    "italy": "rome",
    "internazionali bnl d italia": "rome",
    "monte carlo rolex": "monte carlo",
    "western southern cincinnati": "cincinnati",
    "western southern": "cincinnati",
    "citi washington": "washington",
}

_DIRECT_ROUNDS = {
    "f": "F",
    "final": "F",
    "the final": "F",
    "finals": "F",
    "sf": "SF",
    "semi final": "SF",
    "semi finals": "SF",
    "semifinal": "SF",
    "semifinals": "SF",
    "qf": "QF",
    "quarter final": "QF",
    "quarter finals": "QF",
    "quarterfinal": "QF",
    "quarterfinals": "QF",
    "rr": "RR",
    "round robin": "RR",
    "bronze": "BR",
    "bronze medal": "BR",
}

_ROUND_DISTANCE = {
    "F": 0,
    "SF": 1,
    "QF": 2,
    "R16": 3,
    "R32": 4,
    "R64": 5,
    "R128": 6,
    "R256": 7,
}


def _ascii_words(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", text.lower())


def _name_aliases(name: Any, source: str) -> set[str]:
    words = _ascii_words(name)
    if len(words) < 2:
        return set()

    if source == "auto":
        raw = str(name).strip()
        source = "odds" if re.search(r"(?:^|\s)[A-Za-z](?:\.[A-Za-z])*\.?$", raw) else "sackmann"
    if source not in {"sackmann", "odds"}:
        raise ValueError("source must be 'sackmann', 'odds', or 'auto'")

    aliases: set[str] = set()
    if source == "odds":
        # The final one or more one-letter tokens are given-name initials.
        initial_tokens_start = len(words)
        while initial_tokens_start > 0 and len(words[initial_tokens_start - 1]) == 1:
            initial_tokens_start -= 1
        if initial_tokens_start == len(words) or initial_tokens_start == 0:
            return set()
        surname_words = words[:initial_tokens_start]
        first_initial = words[initial_tokens_start][0]
        aliases.add(f"{' '.join(surname_words)}|{first_initial}")
        # A suffix alias handles occasional disagreement over surname particles.
        if len(surname_words) > 1 and surname_words[-2] not in _PARTICLES:
            aliases.add(f"{surname_words[-1]}|{first_initial}")
        return aliases

    first_initial = words[0][0]
    given_end = 1
    while given_end < len(words) - 1 and len(words[given_end]) == 1:
        given_end += 1
    surname_words = words[given_end:]
    if not surname_words:
        return set()

    particle_positions = [
        position for position in range(1, len(words) - 1) if words[position] in _PARTICLES
    ]
    if particle_positions:
        surname_words = words[particle_positions[0] :]

    aliases.add(f"{' '.join(surname_words)}|{first_initial}")
    aliases.add(f"{words[-1]}|{first_initial}")
    return aliases


def normalize_player_name(name: Any, source: str = "auto") -> str:
    """Return a stable ``surname|first-initial`` key.

    ``source='sackmann'`` expects a full given-name-first name;
    ``source='odds'`` expects tennis-data's surname-first initials format.
    The internal linker considers a small, conservative alias set for compound
    surnames; this function returns its most descriptive member.
    """

    aliases = _name_aliases(name, source)
    if not aliases:
        return ""
    return sorted(aliases, key=lambda item: (-len(item.split("|", 1)[0]), item))[0]


def unordered_player_pair(
    first_name: Any,
    second_name: Any,
    *,
    source: str = "auto",
) -> tuple[str, str] | None:
    """Return an outcome-neutral pair key, or ``None`` for invalid names."""

    first = normalize_player_name(first_name, source)
    second = normalize_player_name(second_name, source)
    if not first or not second or first == second:
        return None
    return tuple(sorted((first, second)))


def _pair_aliases(first_name: Any, second_name: Any, source: str) -> set[tuple[str, str]]:
    first_aliases = _name_aliases(first_name, source)
    second_aliases = _name_aliases(second_name, source)
    return {
        tuple(sorted((first, second)))
        for first in first_aliases
        for second in second_aliases
        if first and second and first != second
    }


def normalize_tournament(value: Any) -> str:
    """Normalize common event naming and sponsor differences."""

    words = [word for word in _ascii_words(value) if not word.isdigit()]
    words = [word for word in words if word not in _TOURNAMENT_STOPWORDS]
    normalized = " ".join(words)
    return _TOURNAMENT_ALIASES.get(normalized, normalized)


def _numbered_round(value: Any) -> int | None:
    words = " ".join(_ascii_words(value))
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\s+round\b", words)
    if match:
        return int(match.group(1))
    return None


def normalize_round(
    value: Any,
    *,
    max_numbered_round: int | None = None,
    draw_size: Any = None,
) -> str:
    """Normalize Sackmann and tennis-data round labels.

    A tennis-data label such as ``1st Round`` is draw-dependent.  When a draw
    size or the event's largest numbered round is known, it is converted to the
    corresponding Sackmann label (for example, ``R32``).  Otherwise it remains
    ``ROUND_1`` rather than pretending to know the draw.
    """

    words = " ".join(_ascii_words(value))
    if not words:
        return ""
    if words in _DIRECT_ROUNDS:
        return _DIRECT_ROUNDS[words]
    compact = words.replace(" ", "").upper()
    if re.fullmatch(r"R(?:16|32|64|128|256)", compact):
        return compact
    if re.fullmatch(r"Q[1-4]", compact):
        return compact

    ordinal = _numbered_round(value)
    if ordinal is None:
        return words.upper().replace(" ", "_")

    inferred_max = max_numbered_round
    try:
        numeric_draw = int(float(draw_size))
        if numeric_draw >= 16:
            draw_rounds = int(math.ceil(math.log2(numeric_draw)))
            inferred_max = max(1, draw_rounds - 3)
    except (TypeError, ValueError, OverflowError):
        pass
    if inferred_max is None or ordinal > inferred_max:
        return f"ROUND_{ordinal}"
    distance_from_final = 3 + inferred_max - ordinal
    size = 2 ** (distance_from_final + 1)
    return f"R{size}"


def _tournament_similarity(first: str, second: str, missing: float) -> float:
    if not first or not second:
        return missing
    if first == second:
        return 1.0
    first_tokens = set(first.split())
    second_tokens = set(second.split())
    union = first_tokens | second_tokens
    jaccard = len(first_tokens & second_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, first, second).ratio()
    # Sequence similarity is discounted: shared event/location tokens are more
    # trustworthy than sponsor-driven character resemblance.
    return max(jaccard, 0.8 * sequence)


def _round_similarity(first: str, second: str, missing: float) -> float:
    if not first or not second:
        return missing
    if first == second:
        return 1.0
    if first.startswith("ROUND_") or second.startswith("ROUND_"):
        return missing
    if first in _ROUND_DISTANCE and second in _ROUND_DISTANCE:
        gap = abs(_ROUND_DISTANCE[first] - _ROUND_DISTANCE[second])
        return max(0.0, 1.0 - 0.4 * gap)
    return 0.0


def _surface_similarity(first: Any, second: Any, missing: float) -> float:
    first_words = _ascii_words(first)
    second_words = _ascii_words(second)
    if not first_words or not second_words or first_words == ["unknown"] or second_words == ["unknown"]:
        return missing
    return float(first_words == second_words)


def _parse_sackmann_dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return pd.to_datetime(values, errors="coerce").dt.normalize()
    strings = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    looks_compact = strings.str.fullmatch(r"\d{8}").fillna(False)
    parsed = pd.to_datetime(strings.where(looks_compact), format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(strings.where(~looks_compact), errors="coerce")
    return parsed.fillna(fallback).dt.normalize()


def _column(frame: pd.DataFrame, *names: str, required: bool = False) -> str | None:
    exact = {str(name): name for name in frame.columns}
    lowered = {str(name).lower(): name for name in frame.columns}
    for name in names:
        if name in exact:
            return exact[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    if required:
        raise ValueError(f"Missing required column; expected one of: {', '.join(names)}")
    return None


def _safe_column_name(value: Any) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return safe or "unnamed"


def _build_components(edges: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_s: dict[int, list[dict[str, Any]]] = {}
    by_o: dict[int, list[dict[str, Any]]] = {}
    for edge in edges:
        by_s.setdefault(edge["s_pos"], []).append(edge)
        by_o.setdefault(edge["o_pos"], []).append(edge)

    components: list[list[dict[str, Any]]] = []
    unseen_s = set(by_s)
    while unseen_s:
        stack_s = [min(unseen_s)]
        component_s: set[int] = set()
        component_o: set[int] = set()
        while stack_s:
            s_pos = stack_s.pop()
            if s_pos in component_s:
                continue
            component_s.add(s_pos)
            unseen_s.discard(s_pos)
            for edge in by_s.get(s_pos, []):
                o_pos = edge["o_pos"]
                if o_pos in component_o:
                    continue
                component_o.add(o_pos)
                for neighboring in by_o.get(o_pos, []):
                    if neighboring["s_pos"] not in component_s:
                        stack_s.append(neighboring["s_pos"])
        components.append(
            [edge for s_pos in component_s for edge in by_s[s_pos] if edge["o_pos"] in component_o]
        )
    return components


def _exact_component_assignment(edges: list[dict[str, Any]]) -> list[tuple[int, int]]:
    left = sorted({edge["s_pos"] for edge in edges})
    right = sorted({edge["o_pos"] for edge in edges})
    right_position = {node: position for position, node in enumerate(right)}
    choices: dict[int, list[tuple[int, float]]] = {node: [] for node in left}
    for edge in edges:
        choices[edge["s_pos"]].append((edge["o_pos"], float(edge["link_score"])))
    for node in choices:
        choices[node].sort(key=lambda item: (-item[1], item[0]))

    @lru_cache(maxsize=None)
    def solve(position: int, used: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if position == len(left):
            return 0, 0.0, ()
        s_pos = left[position]
        best = solve(position + 1, used)
        for o_pos, score in choices[s_pos]:
            bit = 1 << right_position[o_pos]
            if used & bit:
                continue
            count, total, pairs = solve(position + 1, used | bit)
            candidate = count + 1, total + score, ((s_pos, o_pos),) + pairs
            if candidate[0] > best[0]:
                best = candidate
            elif candidate[0] == best[0] and candidate[1] > best[1] + 1e-12:
                best = candidate
            elif (
                candidate[0] == best[0]
                and abs(candidate[1] - best[1]) <= 1e-12
                and candidate[2] < best[2]
            ):
                best = candidate
        return best

    return list(solve(0, 0)[2])


def _large_component_assignment(edges: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Maximum-cardinality deterministic fallback for unusually large components."""

    choices: dict[int, list[tuple[int, float]]] = {}
    for edge in edges:
        choices.setdefault(edge["s_pos"], []).append((edge["o_pos"], edge["link_score"]))
    for s_pos in choices:
        choices[s_pos].sort(key=lambda item: (-item[1], item[0]))
    order = sorted(choices, key=lambda node: (len(choices[node]), -choices[node][0][1], node))
    matched_odds: dict[int, int] = {}

    def augment(s_pos: int, visited: set[int]) -> bool:
        for o_pos, _ in choices[s_pos]:
            if o_pos in visited:
                continue
            visited.add(o_pos)
            incumbent = matched_odds.get(o_pos)
            if incumbent is None or augment(incumbent, visited):
                matched_odds[o_pos] = s_pos
                return True
        return False

    for s_pos in order:
        augment(s_pos, set())
    return sorted((s_pos, o_pos) for o_pos, s_pos in matched_odds.items())


def _assign_one_to_one(edges: list[dict[str, Any]]) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    for component in _build_components(edges):
        left_count = len({edge["s_pos"] for edge in component})
        right_count = len({edge["o_pos"] for edge in component})
        if left_count <= 18 and right_count <= 18:
            pairs = _exact_component_assignment(component)
        else:
            pairs = _large_component_assignment(component)
        selected.update(pairs)
    return selected


def _empty_links(odds_columns: Iterable[Any]) -> pd.DataFrame:
    columns = [
        "sackmann_index",
        "sackmann_position",
        "odds_index",
        "odds_position",
        "tournament_start_date",
        "actual_match_date",
        "date_offset_days",
        "player_pair_key",
        "sackmann_tournament",
        "odds_tournament",
        "sackmann_round",
        "odds_round",
        "tournament_similarity",
        "round_similarity",
        "surface_similarity",
        "date_similarity",
        "link_score",
        "chronology_order",
    ]
    for column in odds_columns:
        output_column = f"odds_{_safe_column_name(column)}"
        if output_column in columns:
            output_column = f"odds_source_{_safe_column_name(column)}"
        columns.append(output_column)
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def link_matches(
    sackmann: pd.DataFrame,
    odds: pd.DataFrame,
    config: LinkageConfig | None = None,
) -> LinkageResult:
    """Link matches without using outcomes or prices as matching evidence.

    Assignment first maximizes the number of unique links, then maximizes total
    contextual score for ordinary (small) ambiguity components.  The returned
    crosswalk is sorted by ``actual_match_date`` and includes a zero-based
    ``chronology_order`` suitable for a walk-forward runner.
    """

    config = config or LinkageConfig()
    if not isinstance(sackmann, pd.DataFrame) or not isinstance(odds, pd.DataFrame):
        raise TypeError("sackmann and odds must both be pandas DataFrames")

    if {"player_a_name", "player_b_name"}.issubset(sackmann.columns):
        s_first = "player_a_name"
        s_second = "player_b_name"
    else:
        s_first = _column(sackmann, "winner_name", required=True)
        s_second = _column(sackmann, "loser_name", required=True)
    s_date = _column(sackmann, "tournament_date", "tourney_date", required=True)
    s_tournament = _column(sackmann, "tourney_name", "tournament")
    s_round = _column(sackmann, "round")
    s_surface = _column(sackmann, "surface")
    s_draw_size = _column(sackmann, "draw_size")

    o_first = _column(odds, "Winner", required=True)
    o_second = _column(odds, "Loser", required=True)
    o_date = _column(odds, "Date", required=True)
    o_tournament = _column(odds, "Tournament")
    o_round = _column(odds, "Round")
    o_surface = _column(odds, "Surface")

    s_work = pd.DataFrame(index=range(len(sackmann)))
    s_work["source_index"] = list(sackmann.index)
    s_work["date"] = list(_parse_sackmann_dates(sackmann[s_date]))
    s_work["first_name"] = list(sackmann[s_first])
    s_work["second_name"] = list(sackmann[s_second])
    s_work["pair_aliases"] = [
        _pair_aliases(first, second, "sackmann")
        for first, second in zip(s_work["first_name"], s_work["second_name"])
    ]
    s_work["tournament_raw"] = list(sackmann[s_tournament]) if s_tournament else ""
    s_work["tournament_norm"] = s_work["tournament_raw"].map(normalize_tournament)
    s_work["round_raw"] = list(sackmann[s_round]) if s_round else ""
    draws = list(sackmann[s_draw_size]) if s_draw_size else [None] * len(sackmann)
    s_work["round_norm"] = [
        normalize_round(raw_round, draw_size=draw)
        for raw_round, draw in zip(s_work["round_raw"], draws)
    ]
    s_work["surface_raw"] = list(sackmann[s_surface]) if s_surface else ""

    o_work = pd.DataFrame(index=range(len(odds)))
    o_work["source_index"] = list(odds.index)
    o_work["date"] = list(pd.to_datetime(odds[o_date], errors="coerce").dt.normalize())
    o_work["first_name"] = list(odds[o_first])
    o_work["second_name"] = list(odds[o_second])
    o_work["pair_aliases"] = [
        _pair_aliases(first, second, "odds")
        for first, second in zip(o_work["first_name"], o_work["second_name"])
    ]
    o_work["tournament_raw"] = list(odds[o_tournament]) if o_tournament else ""
    o_work["tournament_norm"] = o_work["tournament_raw"].map(normalize_tournament)
    o_work["round_raw"] = list(odds[o_round]) if o_round else ""
    o_work["surface_raw"] = list(odds[o_surface]) if o_surface else ""

    numbered = o_work["round_raw"].map(_numbered_round)
    shifted_year = (o_work["date"] + pd.Timedelta(days=7)).dt.year
    edition_key = list(zip(o_work["tournament_norm"], shifted_year))
    maxima: dict[tuple[str, Any], int] = {}
    for key, ordinal in zip(edition_key, numbered):
        if pd.notna(ordinal):
            maxima[key] = max(maxima.get(key, 0), int(ordinal))
    o_work["round_norm"] = [
        normalize_round(raw_round, max_numbered_round=maxima.get(key))
        for raw_round, key in zip(o_work["round_raw"], edition_key)
    ]

    valid_odds = o_work["date"].notna() & o_work["pair_aliases"].map(bool)
    pair_index: dict[tuple[str, str], set[int]] = {}
    for o_pos, aliases in o_work.loc[valid_odds, "pair_aliases"].items():
        for alias in aliases:
            pair_index.setdefault(alias, set()).add(int(o_pos))

    diagnostics: list[dict[str, Any]] = []
    diagnostic_by_position: dict[int, dict[str, Any]] = {}
    candidate_edges: list[dict[str, Any]] = []

    for s_pos, row in s_work.iterrows():
        diagnostic = {
            "sackmann_index": row["source_index"],
            "sackmann_position": int(s_pos),
            "reason_code": "",
            "reason": "",
            "pair_candidate_count": 0,
            "date_candidate_count": 0,
            "eligible_candidate_count": 0,
            "best_score": float("nan"),
            "selected_score": float("nan"),
            "score_margin_to_next_candidate": float("nan"),
            "matched_odds_index": None,
            "matched_odds_position": None,
        }
        diagnostic_by_position[int(s_pos)] = diagnostic
        diagnostics.append(diagnostic)

        if pd.isna(row["date"]):
            diagnostic["reason_code"] = "invalid_sackmann_date"
            continue
        first_primary = normalize_player_name(row["first_name"], source="sackmann")
        second_primary = normalize_player_name(row["second_name"], source="sackmann")
        if not first_primary or not second_primary:
            diagnostic["reason_code"] = "invalid_player_name"
            continue
        if first_primary == second_primary:
            diagnostic["reason_code"] = "same_player_pair"
            continue
        if not row["pair_aliases"]:
            diagnostic["reason_code"] = "invalid_player_name"
            continue

        possible_odds: set[int] = set()
        for pair_alias in row["pair_aliases"]:
            possible_odds.update(pair_index.get(pair_alias, set()))
        diagnostic["pair_candidate_count"] = len(possible_odds)
        if not possible_odds:
            diagnostic["reason_code"] = "no_pair_candidate"
            continue

        within_window: list[int] = []
        for o_pos in sorted(possible_odds):
            date_offset = int((o_work.at[o_pos, "date"] - row["date"]).days)
            if -config.max_days_before_start <= date_offset <= config.max_days_after_start:
                within_window.append(o_pos)
        diagnostic["date_candidate_count"] = len(within_window)
        if not within_window:
            diagnostic["reason_code"] = "outside_date_window"
            continue

        for o_pos in within_window:
            odds_row = o_work.loc[o_pos]
            common_aliases = row["pair_aliases"] & odds_row["pair_aliases"]
            tournament_similarity = _tournament_similarity(
                row["tournament_norm"], odds_row["tournament_norm"], config.missing_similarity
            )
            round_similarity = _round_similarity(
                row["round_norm"], odds_row["round_norm"], config.missing_similarity
            )
            surface_similarity = _surface_similarity(
                row["surface_raw"], odds_row["surface_raw"], config.missing_similarity
            )
            date_offset = int((odds_row["date"] - row["date"]).days)
            window_size = (
                config.max_days_after_start if date_offset >= 0 else config.max_days_before_start
            )
            date_similarity = 1.0 if window_size == 0 else 1.0 - abs(date_offset) / window_size
            link_score = (
                config.tournament_weight * tournament_similarity
                + config.round_weight * round_similarity
                + config.surface_weight * surface_similarity
                + config.date_weight * date_similarity
            )
            diagnostic["best_score"] = max(
                link_score,
                diagnostic["best_score"] if pd.notna(diagnostic["best_score"]) else -math.inf,
            )
            if link_score + 1e-12 < config.min_score:
                continue
            candidate_edges.append(
                {
                    "s_pos": int(s_pos),
                    "o_pos": int(o_pos),
                    "player_pair_key": " <> ".join(sorted(" / ".join(pair) for pair in common_aliases)),
                    "date_offset_days": date_offset,
                    "tournament_similarity": tournament_similarity,
                    "round_similarity": round_similarity,
                    "surface_similarity": surface_similarity,
                    "date_similarity": date_similarity,
                    "link_score": link_score,
                }
            )
            diagnostic["eligible_candidate_count"] += 1

        if not diagnostic["eligible_candidate_count"]:
            diagnostic["reason_code"] = "below_min_score"

    selected_pairs = _assign_one_to_one(candidate_edges) if candidate_edges else set()
    edge_lookup = {(edge["s_pos"], edge["o_pos"]): edge for edge in candidate_edges}
    link_records: list[dict[str, Any]] = []
    used_odds: set[int] = set()

    for s_pos, o_pos in sorted(selected_pairs):
        edge = edge_lookup[(s_pos, o_pos)]
        s_row = s_work.loc[s_pos]
        o_row = o_work.loc[o_pos]
        diagnostic = diagnostic_by_position[s_pos]
        diagnostic["reason_code"] = "matched"
        diagnostic["matched_odds_index"] = o_row["source_index"]
        diagnostic["matched_odds_position"] = o_pos
        diagnostic["selected_score"] = edge["link_score"]
        alternatives = [
            candidate["link_score"]
            for candidate in candidate_edges
            if candidate["s_pos"] == s_pos and candidate["o_pos"] != o_pos
        ]
        if alternatives:
            diagnostic["score_margin_to_next_candidate"] = edge["link_score"] - max(alternatives)
        used_odds.add(o_pos)
        record = {
            "sackmann_index": s_row["source_index"],
            "sackmann_position": s_pos,
            "odds_index": o_row["source_index"],
            "odds_position": o_pos,
            "tournament_start_date": s_row["date"],
            "actual_match_date": o_row["date"],
            "date_offset_days": edge["date_offset_days"],
            "player_pair_key": edge["player_pair_key"],
            "sackmann_tournament": s_row["tournament_raw"],
            "odds_tournament": o_row["tournament_raw"],
            "sackmann_round": s_row["round_norm"],
            "odds_round": o_row["round_norm"],
            "tournament_similarity": edge["tournament_similarity"],
            "round_similarity": edge["round_similarity"],
            "surface_similarity": edge["surface_similarity"],
            "date_similarity": edge["date_similarity"],
            "link_score": edge["link_score"],
        }
        original_odds_row = odds.iloc[o_pos]
        for column, value in original_odds_row.items():
            output_column = f"odds_{_safe_column_name(column)}"
            if output_column in record:
                output_column = f"odds_source_{_safe_column_name(column)}"
            record[output_column] = value
        if "source_row" in sackmann.columns:
            record["sackmann_source_row"] = sackmann.iloc[s_pos]["source_row"]
        link_records.append(record)

    for s_pos, diagnostic in diagnostic_by_position.items():
        if not diagnostic["reason_code"]:
            diagnostic["reason_code"] = "lost_one_to_one_assignment"
        diagnostic["reason"] = REASON_DESCRIPTIONS[diagnostic["reason_code"]]

    if link_records:
        links = pd.DataFrame(link_records).sort_values(
            ["actual_match_date", "sackmann_position", "odds_position"], kind="stable"
        ).reset_index(drop=True)
        links["chronology_order"] = range(len(links))
    else:
        links = _empty_links(odds.columns)

    diagnostic_frame = pd.DataFrame(diagnostics).sort_values("sackmann_position").reset_index(drop=True)
    reason_counts = diagnostic_frame["reason_code"].value_counts().sort_index().to_dict()
    summary: dict[str, Any] = {
        "sackmann_rows": int(len(sackmann)),
        "odds_rows": int(len(odds)),
        "valid_odds_rows": int(valid_odds.sum()),
        "invalid_odds_rows": int((~valid_odds).sum()),
        "matched_rows": int(len(links)),
        "unmatched_sackmann_rows": int(len(sackmann) - len(links)),
        "unused_valid_odds_rows": int(valid_odds.sum() - len(used_odds)),
        "reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
    }
    return LinkageResult(links=links, diagnostics=diagnostic_frame, summary=summary)


__all__ = [
    "LinkageConfig",
    "LinkageResult",
    "REASON_DESCRIPTIONS",
    "link_matches",
    "normalize_player_name",
    "normalize_round",
    "normalize_tournament",
    "unordered_player_pair",
]
