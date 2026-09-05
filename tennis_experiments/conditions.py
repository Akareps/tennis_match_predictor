"""Dynamic tournament-condition experiments from already-completed matches.

This module tests a narrow hypothesis: do serve and score residuals from
earlier rounds of the same tournament edition improve forecasts for later
matches?  It uses only frozen Sackmann main-tour files.  Tournament start dates
are coarse, so every same-date round is predicted simultaneously and event
state is revealed only after the complete batch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import canonicalize_matches, deduplicate_canonical_matches, load_cached_matches
from .experiment import (
    ProjectConfig,
    _digest_record,
    _manifest,
    _prepare_directory,
    _read_json,
    _write_json,
)
from .fitness import OffsetLogisticModel, fit_offset_logistic
from .metrics import (
    accuracy,
    binary_log_loss,
    brier_score,
    calibration_intercept_slope,
    paired_block_bootstrap_gap,
)
from .validation import select_by_validation_loss


STAGE_CONDITIONS_VALIDATION = "09_event_conditions_validation"
STAGE_CONDITIONS_HOLDOUT = "10_event_conditions_holdout"

RATE_METRICS = ("ace", "serve_win", "hold")
SCORE_METRICS = ("deciding_set", "over_22_5")
ALL_CONDITION_METRICS = RATE_METRICS + SCORE_METRICS
TARGETS = {
    "over_22_5": "target_over_22_5",
    "deciding_set": "target_deciding_set",
}
EVENT_PRIOR_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
L2_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
ESTABLISHED_EVENT_MATCHES = 8

# One pseudo-match has a metric-specific exposure.  The validation grid scales
# these quantities together, which keeps the search small while respecting the
# very different denominators of points, service games, and matches.
PSEUDO_EXPOSURE_PER_MATCH = {
    "ace": 120.0,
    "serve_win": 120.0,
    "hold": 20.0,
    "deciding_set": 1.0,
    "over_22_5": 1.0,
}

BASE_FEATURES = (
    "expected_ace_rate",
    "expected_serve_win_rate",
    "expected_hold_rate",
    "expected_deciding_set_rate",
    "expected_over_22_5_rate",
    "surface_clay",
    "surface_grass",
    "surface_hard",
    "surface_carpet",
    "level_grand_slam",
    "level_masters",
    "round_order_scaled",
    "rank_mean_log",
    "rank_gap_log",
    "rank_missing_count",
    "log_point_evidence",
    "log_score_evidence",
    "log_event_prior_matches",
)
SERVE_CONDITION_FEATURES = (
    "event_ace_condition",
    "event_serve_win_condition",
    "event_hold_condition",
)
SCORE_CONDITION_FEATURES = (
    "event_deciding_set_condition",
    "event_over_22_5_condition",
)
CONDITION_VARIANTS = {
    "player_context_baseline": BASE_FEATURES,
    "plus_serve_conditions": BASE_FEATURES + SERVE_CONDITION_FEATURES,
    "plus_score_conditions": BASE_FEATURES
    + SERVE_CONDITION_FEATURES
    + SCORE_CONDITION_FEATURES,
}
CONDITION_ONLY_FEATURES = {
    "plus_serve_conditions": SERVE_CONDITION_FEATURES,
    "plus_score_conditions": SERVE_CONDITION_FEATURES + SCORE_CONDITION_FEATURES,
}
CONDITION_VARIANT_ORDER = {
    name: index for index, name in enumerate(CONDITION_VARIANTS)
}

_PRIOR_RATES = {
    "atp": {
        "ace": 0.08,
        "serve_win": 0.63,
        "hold": 0.78,
        "deciding_set": 0.36,
        "over_22_5": 0.42,
    },
    "wta": {
        "ace": 0.045,
        "serve_win": 0.58,
        "hold": 0.66,
        "deciding_set": 0.34,
        "over_22_5": 0.39,
    },
}
_OVERALL_PRIOR_TRIALS = {
    "ace": 200.0,
    "serve_win": 200.0,
    "hold": 30.0,
    "deciding_set": 20.0,
    "over_22_5": 20.0,
}
_SURFACE_PRIOR_TRIALS = {
    "ace": 100.0,
    "serve_win": 100.0,
    "hold": 15.0,
    "deciding_set": 10.0,
    "over_22_5": 10.0,
}


@dataclass
class _RateState:
    successes: float = 0.0
    trials: float = 0.0

    def update(self, successes: float, trials: float) -> None:
        if not np.isfinite(successes) or not np.isfinite(trials):
            raise ValueError("Rate updates must be finite")
        if trials <= 0 or successes < 0 or successes > trials:
            raise ValueError("Rate updates must satisfy 0 <= successes <= trials")
        self.successes += float(successes)
        self.trials += float(trials)


class _PlayerRates:
    def __init__(self, tour: str) -> None:
        self.tour = tour
        self.states: dict[tuple[int, str, str, str], _RateState] = defaultdict(_RateState)

    def _state(self, player: int, surface: str, role: str, metric: str) -> _RateState:
        return self.states[(int(player), str(surface), str(role), str(metric))]

    def rate(self, player: int, surface: str, role: str, metric: str) -> float:
        prior_rate = _PRIOR_RATES[self.tour][metric]
        overall = self._state(player, "*", role, metric)
        overall_weight = _OVERALL_PRIOR_TRIALS[metric]
        overall_rate = (
            overall.successes + prior_rate * overall_weight
        ) / (overall.trials + overall_weight)
        surface_state = self._state(player, surface, role, metric)
        surface_weight = _SURFACE_PRIOR_TRIALS[metric]
        return float(
            (surface_state.successes + overall_rate * surface_weight)
            / (surface_state.trials + surface_weight)
        )

    def evidence(self, player: int, role: str, metric: str) -> float:
        return float(self._state(player, "*", role, metric).trials)

    def update(
        self,
        player: int,
        surface: str,
        role: str,
        metric: str,
        successes: float,
        trials: float,
    ) -> None:
        self._state(player, "*", role, metric).update(successes, trials)
        self._state(player, surface, role, metric).update(successes, trials)


@dataclass(frozen=True)
class ConditionResidualModel:
    """A no-intercept residual that is exactly zero when conditions are zero."""

    feature_names: tuple[str, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    l2: float
    iterations: int
    converged: bool

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        base_column: str = "p_baseline",
    ) -> np.ndarray:
        base = np.clip(frame[base_column].to_numpy(dtype=float), 1e-9, 1.0 - 1e-9)
        offset = np.log(base / (1.0 - base))
        values = frame[list(self.feature_names)].to_numpy(dtype=float)
        design = values / np.asarray(self.scales)
        eta = np.clip(offset + design @ np.asarray(self.coefficients), -30.0, 30.0)
        result = 1.0 / (1.0 + np.exp(-eta))
        zero_rows = np.all(values == 0.0, axis=1)
        result[zero_rows] = base[zero_rows]
        return result

    def coefficient_rows(self, variant: str) -> list[dict[str, Any]]:
        return [
            {
                "variant": variant,
                "feature": feature,
                "coefficient_standardized": coefficient,
                "coefficient_raw": coefficient / scale,
                "training_mean": 0.0,
                "training_scale": scale,
                "l2": self.l2,
                "iterations": self.iterations,
                "converged": self.converged,
            }
            for feature, scale, coefficient in zip(
                self.feature_names, self.scales, self.coefficients
            )
        ]


def fit_condition_residual(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    l2: float,
    base_column: str = "p_baseline",
    outcome_column: str = "actual_a_won",
    max_iterations: int = 100,
) -> ConditionResidualModel:
    """Fit condition-only log-odds adjustments with no intercept or centering."""

    if l2 < 0 or not np.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    columns = [base_column, outcome_column, *feature_names]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("Missing condition-residual columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Cannot fit a condition residual on an empty frame")
    base = np.clip(frame[base_column].to_numpy(dtype=float), 1e-9, 1.0 - 1e-9)
    outcomes = frame[outcome_column].to_numpy(dtype=float)
    if not np.isin(outcomes, [0.0, 1.0]).all() or len(np.unique(outcomes)) < 2:
        raise ValueError("Condition-residual outcomes must contain both binary classes")
    values = frame[list(feature_names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Condition features must be finite")
    scales = np.sqrt(np.mean(values**2, axis=0))
    scales = np.where(scales < 1e-9, 1.0, scales)
    design = values / scales
    offset = np.log(base / (1.0 - base))
    beta = np.zeros(len(feature_names), dtype=float)
    penalty = np.eye(len(feature_names), dtype=float) * float(l2)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        eta = np.clip(offset + design @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        information = design.T @ (weights[:, None] * design) + penalty
        score = design.T @ (outcomes - fitted) - penalty @ beta
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as error:
            raise ValueError("Condition-residual information matrix is singular") from error
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    if not np.isfinite(beta).all():
        raise ValueError("Condition-residual fit produced non-finite coefficients")
    return ConditionResidualModel(
        feature_names=tuple(feature_names),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in beta),
        l2=float(l2),
        iterations=iterations,
        converged=converged,
    )


def _number(row: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(row.get(name))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _side_point_observation(row: Mapping[str, Any], side: str) -> dict[str, tuple[float, float]] | None:
    aces = _number(row, f"{side}_aces")
    double_faults = _number(row, f"{side}_double_faults")
    service_points = _number(row, f"{side}_service_points")
    first_serves_in = _number(row, f"{side}_first_serves_in")
    first_won = _number(row, f"{side}_first_serve_points_won")
    second_won = _number(row, f"{side}_second_serve_points_won")
    service_games = _number(row, f"{side}_service_games")
    break_points_saved = _number(row, f"{side}_break_points_saved")
    break_points_faced = _number(row, f"{side}_break_points_faced")
    values = (
        aces,
        double_faults,
        service_points,
        first_serves_in,
        first_won,
        second_won,
        service_games,
        break_points_saved,
        break_points_faced,
    )
    if any(value is None for value in values):
        return None
    assert all(value is not None for value in values)
    serve_wins = float(first_won + second_won)
    breaks = float(break_points_faced - break_points_saved)
    holds = float(service_games - breaks)
    plausible = (
        service_points > 0
        and service_games > 0
        and 0 <= aces <= service_points
        and 0 <= double_faults <= service_points
        and 0 <= first_serves_in <= service_points
        and 0 <= first_won <= first_serves_in
        and 0 <= second_won <= service_points - first_serves_in
        and 0 <= serve_wins <= service_points
        and 0 <= break_points_saved <= break_points_faced
        and 0 <= breaks <= service_games
    )
    if not plausible:
        return None
    return {
        "ace": (float(aces), float(service_points)),
        "serve_win": (serve_wins, float(service_points)),
        "hold": (holds, float(service_games)),
    }


_SET_TOKEN = re.compile(r"^(\d+)-(\d+)(?:\((\d+)\))?$")
_NONSTANDARD_EVENT_NAMES = ("NEXT GEN", "NEXTGEN", "LAVER CUP", "OLYMPICS")


def parse_standard_bo3_score(
    score: Any,
    *,
    best_of: int = 3,
    is_retirement: bool = False,
) -> dict[str, int] | None:
    """Parse a completed conventional best-of-three winner-oriented score.

    Sackmann scores put the recorded winner first in every set.  Match
    tiebreaks, partial scores, illegal set scores, walkovers, and every explicit
    non-completion are rejected rather than repaired heuristically.
    """

    if int(best_of) != 3 or bool(is_retirement):
        return None
    rendered = "" if score is None else str(score).strip()
    upper = rendered.upper()
    if not rendered or "[" in rendered or "]" in rendered:
        return None
    if upper.replace(" ", "") in {"WO", "W/O"}:
        return None
    if any(token in upper for token in ("W/O", " WO", "RET", "DEF", "ABN")):
        return None
    tokens = rendered.split()
    if len(tokens) not in {2, 3}:
        return None

    games = 0
    winner_sets = 0
    loser_sets = 0
    tiebreak_sets = 0
    set_winners: list[int] = []
    for token_index, token in enumerate(tokens):
        match = _SET_TOKEN.fullmatch(token)
        if match is None:
            return None
        winner_games = int(match.group(1))
        loser_games = int(match.group(2))
        tiebreak_points = match.group(3)
        ordered = sorted((winner_games, loser_games), reverse=True)
        legal = (
            (ordered[0] == 6 and ordered[1] <= 4)
            or ordered == [7, 5]
            or ordered == [7, 6]
            or (
                token_index == len(tokens) - 1
                and len(tokens) == 3
                and ordered[0] >= 8
                and ordered[0] - ordered[1] == 2
            )
        )
        if not legal or winner_games == loser_games:
            return None
        is_tiebreak = {winner_games, loser_games} == {6, 7}
        if tiebreak_points is not None and not is_tiebreak:
            return None
        games += winner_games + loser_games
        tiebreak_sets += int(is_tiebreak)
        won_by_recorded_winner = int(winner_games > loser_games)
        set_winners.append(won_by_recorded_winner)
        winner_sets += won_by_recorded_winner
        loser_sets += 1 - won_by_recorded_winner

    # The recorded match winner must win the final set and reach two sets only
    # at that point.  This also rejects extra sets after a completed result.
    if winner_sets != 2 or loser_sets != len(tokens) - 2:
        return None
    if set_winners[-1] != 1 or sum(set_winners[:-1]) >= 2:
        return None
    return {
        "games": int(games),
        "sets": int(len(tokens)),
        "deciding_set": int(len(tokens) == 3),
        "has_tiebreak": int(tiebreak_sets > 0),
        "tiebreak_sets": int(tiebreak_sets),
        "over_22_5": int(games > 22.5),
    }


def _event_is_standard(group: pd.DataFrame) -> bool:
    levels = group["tourney_level"].fillna("").astype(str).str.upper().str.strip()
    names = group["tourney_name"].fillna("").astype(str).str.upper()
    scores = group["score"].fillna("").astype(str)
    if levels.isin({"D", "O"}).any():
        return False
    if any(names.str.contains(token, regex=False).any() for token in _NONSTANDARD_EVENT_NAMES):
        return False
    # Exclude the entire event: dropping only match-tiebreak rows would
    # selectively remove matches that reached one set all.
    return not scores.str.contains("[", regex=False).any()


def _match_point_observations(
    row: Mapping[str, Any],
    parsed_score: Mapping[str, int] | None,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]] | None:
    if parsed_score is None:
        return None
    point_a = _side_point_observation(row, "a")
    point_b = _side_point_observation(row, "b")
    if point_a is None or point_b is None:
        return None
    service_games = point_a["hold"][1] + point_b["hold"][1]
    conventional_games = float(parsed_score["games"])
    games_without_breakers = conventional_games - float(parsed_score["tiebreak_sets"])
    if service_games not in {conventional_games, games_without_breakers}:
        return None
    return point_a, point_b


def _rank_features(a_rank: Any, b_rank: Any) -> tuple[float, float, float]:
    values: list[float] = []
    missing = 0
    for raw in (a_rank, b_rank):
        value = _number({"value": raw}, "value")
        if value is None or value < 1:
            value = 2000.0
            missing += 1
        values.append(min(2000.0, value))
    logs = [log(value) for value in values]
    return float(sum(logs) / 2.0), float(abs(logs[0] - logs[1])), float(missing)


def _mean_probability(left: float, right: float) -> float:
    clipped = np.clip([left, right], 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + np.exp(-float(np.mean(logits)))))


def _side_expected_rates(
    rates: _PlayerRates,
    *,
    player_a: int,
    player_b: int,
    surface: str,
    metric: str,
) -> tuple[float, float]:
    if metric in RATE_METRICS:
        a_serve = rates.rate(player_a, surface, "serve", metric)
        b_serve = rates.rate(player_b, surface, "serve", metric)
        a_allowed = rates.rate(player_a, surface, "return_allowed", metric)
        b_allowed = rates.rate(player_b, surface, "return_allowed", metric)
        return (
            _mean_probability(a_serve, b_allowed),
            _mean_probability(b_serve, a_allowed),
        )
    a_rate = rates.rate(player_a, surface, "match", metric)
    b_rate = rates.rate(player_b, surface, "match", metric)
    combined = _mean_probability(a_rate, b_rate)
    return combined, combined


def _pair_rate(
    rates: _PlayerRates,
    *,
    player_a: int,
    player_b: int,
    surface: str,
    metric: str,
) -> float:
    left, right = _side_expected_rates(
        rates,
        player_a=player_a,
        player_b=player_b,
        surface=surface,
        metric=metric,
    )
    return float((left + right) / 2.0)


def _surface_flags(surface: str) -> dict[str, float]:
    normalized = str(surface).strip().lower()
    return {
        "surface_clay": float(normalized == "clay"),
        "surface_grass": float(normalized == "grass"),
        "surface_hard": float(normalized == "hard"),
        "surface_carpet": float(normalized == "carpet"),
    }


def build_event_condition_panel(matches: pd.DataFrame, tour: str) -> pd.DataFrame:
    """Return causal baselines, leave-player-out event state, and score targets.

    Player profiles are frozen for every tournament start date.  Within an
    event, a whole round is emitted before its observations become visible.
    Event state also excludes any earlier match involving either target player,
    separating venue conditions from the target players' own tournament form.
    """

    tour = tour.lower()
    if tour not in _PRIOR_RATES:
        raise ValueError("tour must be 'atp' or 'wta'")
    required = {
        "source_row",
        "source_group",
        "match_key",
        "event_key",
        "tournament_date",
        "tourney_name",
        "tourney_level",
        "surface",
        "round",
        "round_order",
        "best_of",
        "player_a_id",
        "player_b_id",
        "score",
        "is_retirement",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError("Missing event-condition columns: " + ", ".join(missing))
    if not matches["source_group"].eq("main").all():
        raise ValueError("Event conditions use main-tour matches only")
    if matches["match_key"].duplicated().any():
        raise ValueError("Event-condition inputs contain duplicate match keys")

    working = matches.copy()
    working["tournament_date"] = pd.to_datetime(working["tournament_date"])
    if working["tournament_date"].isna().any():
        raise ValueError("Event-condition inputs contain invalid tournament dates")
    working = working.sort_values(
        ["tournament_date", "round_order", "event_key", "match_key"], kind="stable"
    ).reset_index(drop=True)
    standard_events = {
        str(event_key): _event_is_standard(group)
        for event_key, group in working.groupby("event_key", sort=False)
    }

    rates = _PlayerRates(tour)
    event_contributions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for tournament_date, date_frame in working.groupby("tournament_date", sort=False):
        date_records = date_frame.replace({np.nan: None}).to_dict("records")
        for round_order, round_frame in date_frame.groupby("round_order", sort=True):
            batch_id = f"{pd.Timestamp(tournament_date):%Y-%m-%d}|{int(round_order)}"
            records = round_frame.replace({np.nan: None}).to_dict("records")
            batch_outputs: dict[str, dict[str, Any]] = {}

            for record in records:
                player_a = int(record["player_a_id"])
                player_b = int(record["player_b_id"])
                target_players = {player_a, player_b}
                surface = str(record.get("surface", "Unknown"))
                event_key = str(record["event_key"])
                standard_event = bool(standard_events[event_key])
                parsed = (
                    parse_standard_bo3_score(
                        record.get("score"),
                        best_of=int(record.get("best_of", 3)),
                        is_retirement=bool(record.get("is_retirement", False)),
                    )
                    if standard_event
                    else None
                )
                point_pair = _match_point_observations(record, parsed)

                expected_sides = {
                    metric: _side_expected_rates(
                        rates,
                        player_a=player_a,
                        player_b=player_b,
                        surface=surface,
                        metric=metric,
                    )
                    for metric in ALL_CONDITION_METRICS
                }
                expected = {
                    metric: float(sum(expected_sides[metric]) / 2.0)
                    for metric in ALL_CONDITION_METRICS
                }
                eligible_prior = [
                    contribution
                    for contribution in event_contributions[event_key]
                    if target_players.isdisjoint(contribution["players"])
                ]
                residual_sums = {metric: 0.0 for metric in ALL_CONDITION_METRICS}
                residual_exposures = {metric: 0.0 for metric in ALL_CONDITION_METRICS}
                residual_counts = {metric: 0 for metric in ALL_CONDITION_METRICS}
                for contribution in eligible_prior:
                    for metric, (residual, exposure) in contribution["metrics"].items():
                        residual_sums[metric] += float(residual)
                        residual_exposures[metric] += float(exposure)
                        residual_counts[metric] += 1

                prior_point_matches = residual_counts["serve_win"]
                prior_score_matches = residual_counts["over_22_5"]
                point_evidence = min(
                    rates.evidence(player_a, "serve", "serve_win"),
                    rates.evidence(player_b, "serve", "serve_win"),
                )
                score_evidence = min(
                    rates.evidence(player_a, "match", "deciding_set"),
                    rates.evidence(player_b, "match", "deciding_set"),
                )
                rank_mean, rank_gap, rank_missing = _rank_features(
                    record.get("a_rank"), record.get("b_rank")
                )
                output: dict[str, Any] = {
                    "source_row": int(record["source_row"]),
                    "match_key": record["match_key"],
                    "event_key": event_key,
                    "tournament_date": pd.Timestamp(tournament_date),
                    "chronology_batch": batch_id,
                    "round": record.get("round"),
                    "round_order": int(record.get("round_order", 0)),
                    "surface": surface,
                    "tourney_level": str(record.get("tourney_level", "?")),
                    "best_of": int(record.get("best_of", 3)),
                    "player_a_id": player_a,
                    "player_b_id": player_b,
                    "standard_event": standard_event,
                    "score_target_valid": parsed is not None,
                    "point_stats_valid": point_pair is not None,
                    "rank_mean_log": rank_mean,
                    "rank_gap_log": rank_gap,
                    "rank_missing_count": rank_missing,
                    "log_point_evidence": float(np.log1p(point_evidence)),
                    "log_score_evidence": float(np.log1p(score_evidence)),
                    "event_prior_matches": int(prior_point_matches),
                    "event_prior_point_matches": int(prior_point_matches),
                    "event_prior_score_matches": int(prior_score_matches),
                    "event_prior_service_points": float(
                        residual_exposures["serve_win"]
                    ),
                    "event_condition_available": bool(
                        prior_point_matches >= ESTABLISHED_EVENT_MATCHES
                    ),
                    "log_event_prior_matches": float(np.log1p(prior_point_matches)),
                    "level_grand_slam": float(
                        str(record.get("tourney_level", "")) == "G"
                    ),
                    "level_masters": float(
                        str(record.get("tourney_level", "")) in {"M", "P", "PM", "W"}
                    ),
                    "round_order_scaled": float(record.get("round_order", 0)) / 8.0,
                }
                output.update(_surface_flags(surface))
                for metric in ALL_CONDITION_METRICS:
                    output[f"expected_{metric}_rate"] = expected[metric]
                    output[f"expected_a_{metric}_rate"] = expected_sides[metric][0]
                    output[f"expected_b_{metric}_rate"] = expected_sides[metric][1]
                    output[f"event_{metric}_residual_sum"] = residual_sums[metric]
                    output[f"event_{metric}_residual_exposure"] = residual_exposures[metric]
                    output[f"event_{metric}_residual_count"] = residual_counts[metric]

                output["target_over_22_5"] = (
                    float(parsed["over_22_5"]) if parsed is not None else np.nan
                )
                output["target_deciding_set"] = (
                    float(parsed["deciding_set"]) if parsed is not None else np.nan
                )
                for metric in RATE_METRICS:
                    if point_pair is not None:
                        point_a, point_b = point_pair
                        successes = point_a[metric][0] + point_b[metric][0]
                        trials = point_a[metric][1] + point_b[metric][1]
                        output[f"actual_{metric}_successes"] = float(successes)
                        output[f"actual_{metric}_trials"] = float(trials)
                        output[f"actual_{metric}_rate"] = float(successes / trials)
                    else:
                        output[f"actual_{metric}_successes"] = np.nan
                        output[f"actual_{metric}_trials"] = np.nan
                        output[f"actual_{metric}_rate"] = np.nan

                rows.append(output)
                batch_outputs[str(record["match_key"])] = output

            # Reveal a complete round together.  No row within the round can
            # observe another row's score or point statistics.
            for record in records:
                output = batch_outputs[str(record["match_key"])]
                if not bool(output["standard_event"]):
                    continue
                parsed = parse_standard_bo3_score(
                    record.get("score"),
                    best_of=int(record.get("best_of", 3)),
                    is_retirement=bool(record.get("is_retirement", False)),
                )
                point_pair = _match_point_observations(record, parsed)
                metric_values: dict[str, tuple[float, float]] = {}
                if point_pair is not None:
                    point_a, point_b = point_pair
                    for metric in RATE_METRICS:
                        a_successes, a_trials = point_a[metric]
                        b_successes, b_trials = point_b[metric]
                        expected_successes = (
                            float(output[f"expected_a_{metric}_rate"]) * a_trials
                            + float(output[f"expected_b_{metric}_rate"]) * b_trials
                        )
                        metric_values[metric] = (
                            float(a_successes + b_successes - expected_successes),
                            float(a_trials + b_trials),
                        )
                if parsed is not None:
                    for metric, target_column in (
                        ("deciding_set", "target_deciding_set"),
                        ("over_22_5", "target_over_22_5"),
                    ):
                        metric_values[metric] = (
                            float(output[target_column])
                            - float(output[f"expected_{metric}_rate"]),
                            1.0,
                        )
                if metric_values:
                    event_contributions[str(record["event_key"])].append(
                        {
                            "players": frozenset(
                                (int(record["player_a_id"]), int(record["player_b_id"]))
                            ),
                            "metrics": metric_values,
                        }
                    )

        # Profiles for all events sharing a published start date are updated
        # only after that date is fully emitted.  Thus a numerically earlier
        # round in another event cannot contaminate a player's frozen baseline.
        for record in date_records:
            if not standard_events[str(record["event_key"])]:
                continue
            parsed = parse_standard_bo3_score(
                record.get("score"),
                best_of=int(record.get("best_of", 3)),
                is_retirement=bool(record.get("is_retirement", False)),
            )
            point_pair = _match_point_observations(record, parsed)
            player_a = int(record["player_a_id"])
            player_b = int(record["player_b_id"])
            surface = str(record.get("surface", "Unknown"))
            if point_pair is not None:
                point_a, point_b = point_pair
                for metric in RATE_METRICS:
                    a_successes, a_trials = point_a[metric]
                    b_successes, b_trials = point_b[metric]
                    rates.update(player_a, surface, "serve", metric, a_successes, a_trials)
                    rates.update(
                        player_b, surface, "return_allowed", metric, a_successes, a_trials
                    )
                    rates.update(player_b, surface, "serve", metric, b_successes, b_trials)
                    rates.update(
                        player_a, surface, "return_allowed", metric, b_successes, b_trials
                    )
            if parsed is not None:
                for metric in SCORE_METRICS:
                    target = float(parsed[metric])
                    for player in (player_a, player_b):
                        rates.update(player, surface, "match", metric, target, 1.0)

    panel = pd.DataFrame(rows)
    if panel["match_key"].duplicated().any():
        raise ValueError("Event-condition panel contains duplicate match keys")
    return panel


def materialize_event_conditions(
    panel: pd.DataFrame,
    prior_matches: float,
) -> pd.DataFrame:
    """Shrink event residuals by metric-appropriate pseudo exposures."""

    if not np.isfinite(prior_matches) or prior_matches < 0:
        raise ValueError("prior_matches must be finite and non-negative")
    output = panel.copy()
    for metric in ALL_CONDITION_METRICS:
        numerator = output[f"event_{metric}_residual_sum"].to_numpy(dtype=float)
        exposure = output[f"event_{metric}_residual_exposure"].to_numpy(dtype=float)
        pseudo_exposure = PSEUDO_EXPOSURE_PER_MATCH[metric] * float(prior_matches)
        denominator = exposure + pseudo_exposure
        output[f"event_{metric}_condition"] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        established = output["event_prior_point_matches"].to_numpy(dtype=float) >= (
            ESTABLISHED_EVENT_MATCHES
        )
        output[f"event_{metric}_condition"] = np.where(
            established, output[f"event_{metric}_condition"], 0.0
        )
    return output


def _load_condition_matches(
    config: ProjectConfig,
    years: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_cached_matches(
        config.project_root / "data",
        config.tour,
        years,
        include_lower_tiers=False,
    )
    matches, audit = deduplicate_canonical_matches(
        canonicalize_matches(raw, config.tour)
    )
    if not matches["source_group"].eq("main").all():
        raise AssertionError("Condition loader returned a non-main row")
    return matches, audit


def _target_frame(
    panel: pd.DataFrame,
    target: str,
    *,
    prior_matches: float,
) -> pd.DataFrame:
    target_column = TARGETS[target]
    frame = materialize_event_conditions(panel, prior_matches)
    frame = frame.loc[frame[target_column].notna()].copy()
    frame["actual_a_won"] = frame[target_column].astype(int)
    frame["p_model"] = 0.5
    return frame


def _candidate_id(variant: str, l2: float, prior_matches: float | None) -> str:
    prior = "none" if prior_matches is None else f"{prior_matches:g}"
    return f"{variant}__l2_{l2:g}__event_prior_{prior}"


def _evaluate_binary(frame: pd.DataFrame) -> dict[str, float | int]:
    probabilities = frame["p_model"].to_numpy(dtype=float)
    outcomes = frame["actual_a_won"].to_numpy(dtype=int)
    intercept, slope = calibration_intercept_slope(probabilities, outcomes)
    return {
        "n": int(len(frame)),
        "event_observed_n": int(frame["event_prior_matches"].gt(0).sum()),
        "event_established_n": int(
            frame["event_prior_matches"].ge(ESTABLISHED_EVENT_MATCHES).sum()
        ),
        "log_loss": binary_log_loss(probabilities, outcomes),
        "brier": brier_score(probabilities, outcomes),
        "accuracy": accuracy(probabilities, outcomes),
        "outcome_rate": float(np.mean(outcomes)),
        "mean_probability": float(np.mean(probabilities)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _binary_slices(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", frame
    yield "event_unavailable", frame.loc[
        frame["event_prior_matches"].lt(ESTABLISHED_EVENT_MATCHES)
    ]
    yield "event_observed", frame.loc[frame["event_prior_matches"].gt(0)]
    yield "event_established", frame.loc[
        frame["event_prior_matches"].ge(ESTABLISHED_EVENT_MATCHES)
    ]


def _selected_model_payload(
    model: OffsetLogisticModel | ConditionResidualModel,
    *,
    candidate: str,
    l2: float,
    prior_matches: float | None,
    validation_log_loss: float,
    validation_n: int,
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "l2": float(l2),
        "event_prior_matches": prior_matches,
        "features": list(model.feature_names),
        "validation_log_loss": float(validation_log_loss),
        "validation_n": int(validation_n),
    }


def _fit_baseline_candidate(
    training_panel: pd.DataFrame,
    evaluation_panel: pd.DataFrame,
    *,
    target: str,
    l2: float,
) -> tuple[OffsetLogisticModel, pd.DataFrame]:
    training = _target_frame(training_panel, target, prior_matches=4.0)
    evaluation = _target_frame(evaluation_panel, target, prior_matches=4.0)
    model = fit_offset_logistic(training, BASE_FEATURES, l2=l2)
    evaluation["p_model"] = model.predict(evaluation)
    evaluation["p_baseline"] = evaluation["p_model"]
    evaluation["target"] = target
    evaluation["variant"] = "player_context_baseline"
    return model, evaluation


def _fit_condition_candidate(
    training_panel: pd.DataFrame,
    evaluation_panel: pd.DataFrame,
    *,
    target: str,
    variant: str,
    l2: float,
    prior_matches: float,
    baseline_model: OffsetLogisticModel,
) -> tuple[ConditionResidualModel, pd.DataFrame]:
    if variant not in CONDITION_ONLY_FEATURES:
        raise ValueError(f"Not a condition-only variant: {variant}")
    training = _target_frame(training_panel, target, prior_matches=prior_matches)
    evaluation = _target_frame(evaluation_panel, target, prior_matches=prior_matches)
    training["p_baseline"] = baseline_model.predict(training)
    evaluation["p_baseline"] = baseline_model.predict(evaluation)
    features = CONDITION_ONLY_FEATURES[variant]
    model = fit_condition_residual(training, features, l2=l2)
    evaluation["p_model"] = model.predict(evaluation)
    evaluation["target"] = target
    evaluation["variant"] = variant
    unavailable = evaluation["event_prior_matches"].lt(ESTABLISHED_EVENT_MATCHES)
    if not np.array_equal(
        evaluation.loc[unavailable, "p_model"].to_numpy(),
        evaluation.loc[unavailable, "p_baseline"].to_numpy(),
    ):
        raise AssertionError("Condition model changed a row without established evidence")
    return model, evaluation


def _weighted_binomial_loss(
    successes: np.ndarray,
    trials: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    return float(np.sum(-(s * np.log(p) + (n - s) * np.log(1 - p))) / np.sum(n))


def _persistence_metrics(
    frame: pd.DataFrame,
    metric: str,
    prior_matches: float,
) -> dict[str, float | int]:
    materialized = materialize_event_conditions(frame, prior_matches)
    mask = (
        materialized[f"actual_{metric}_trials"].notna()
        & materialized[f"event_{metric}_residual_count"].gt(0)
        & materialized["event_prior_point_matches"].ge(ESTABLISHED_EVENT_MATCHES)
    )
    work = materialized.loc[mask].copy()
    if work.empty:
        return {
            "n": 0,
            "trials": 0,
            "base_log_loss": float("nan"),
            "adjusted_log_loss": float("nan"),
            "delta_log_loss": float("nan"),
            "base_rmse": float("nan"),
            "adjusted_rmse": float("nan"),
        }
    successes = work[f"actual_{metric}_successes"].to_numpy(dtype=float)
    trials = work[f"actual_{metric}_trials"].to_numpy(dtype=float)
    actual_rate = successes / trials
    base = work[f"expected_{metric}_rate"].to_numpy(dtype=float)
    adjusted = np.clip(
        base + work[f"event_{metric}_condition"].to_numpy(dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    base_loss = _weighted_binomial_loss(successes, trials, base)
    adjusted_loss = _weighted_binomial_loss(successes, trials, adjusted)
    return {
        "n": int(len(work)),
        "trials": int(np.sum(trials)),
        "base_log_loss": base_loss,
        "adjusted_log_loss": adjusted_loss,
        "delta_log_loss": adjusted_loss - base_loss,
        "base_rmse": float(np.sqrt(np.mean((actual_rate - base) ** 2))),
        "adjusted_rmse": float(np.sqrt(np.mean((actual_rate - adjusted) ** 2))),
    }


def run_event_conditions_validation_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stage 09: tune event shrinkage and target models before 2024."""

    destination = config.stage_directory(STAGE_CONDITIONS_VALIDATION)
    _prepare_directory(destination, overwrite)
    matches, dedupe_audit = _load_condition_matches(
        config, range(config.history_start, config.test_year)
    )
    panel = build_event_condition_panel(matches, config.tour)
    if panel["tournament_date"].ge(config.test_start).any():
        raise AssertionError("Stage 09 loaded a row from the final benchmark period")
    validation_year = config.test_year - 1
    training_year = validation_year - 1
    training = panel.loc[panel["tournament_date"].dt.year.eq(training_year)].copy()
    validation = panel.loc[panel["tournament_date"].dt.year.eq(validation_year)].copy()
    if training.empty or validation.empty:
        raise ValueError("Event-condition validation needs training and validation rows")

    # Lock the context baseline first.  Every condition candidate is then a
    # zero-intercept residual on this exact same baseline, so differences can
    # be attributed only to the event-condition columns.
    baseline_models: dict[str, OffsetLogisticModel] = {}
    for target in TARGETS:
        baseline_candidates: dict[str, OffsetLogisticModel] = {}
        baseline_rows: list[dict[str, Any]] = []
        for l2 in L2_GRID:
            model, evaluated = _fit_baseline_candidate(
                training, validation, target=target, l2=l2
            )
            candidate = _candidate_id("player_context_baseline", l2, None)
            baseline_candidates[candidate] = model
            baseline_rows.append(
                {
                    "candidate": candidate,
                    "log_loss": binary_log_loss(
                        evaluated["p_model"], evaluated["actual_a_won"]
                    ),
                    "n": len(evaluated),
                }
            )
        selected_baseline, _ = select_by_validation_loss(
            pd.DataFrame(baseline_rows)
        )
        baseline_models[target] = baseline_candidates[str(selected_baseline)]

    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    candidate_models: dict[
        tuple[str, str, str], OffsetLogisticModel | ConditionResidualModel
    ] = {}
    for target in TARGETS:
        for variant in CONDITION_VARIANTS:
            priors: Sequence[float] = (
                (4.0,) if variant == "player_context_baseline" else EVENT_PRIOR_GRID
            )
            for prior_matches in priors:
                for l2 in L2_GRID:
                    if variant == "player_context_baseline":
                        model, evaluated = _fit_baseline_candidate(
                            training, validation, target=target, l2=l2
                        )
                    else:
                        model, evaluated = _fit_condition_candidate(
                            training,
                            validation,
                            target=target,
                            variant=variant,
                            l2=l2,
                            prior_matches=prior_matches,
                            baseline_model=baseline_models[target],
                        )
                    candidate = _candidate_id(
                        variant,
                        l2,
                        None if variant == "player_context_baseline" else prior_matches,
                    )
                    key = (target, variant, candidate)
                    candidate_models[key] = model
                    coefficient_rows.extend(
                        {
                            **row,
                            "target": target,
                            "candidate": candidate,
                            "event_prior_matches": (
                                np.nan
                                if variant == "player_context_baseline"
                                else prior_matches
                            ),
                        }
                        for row in model.coefficient_rows(variant)
                    )
                    for slice_name, sliced in _binary_slices(evaluated):
                        if sliced.empty:
                            continue
                        metrics = _evaluate_binary(sliced)
                        fold_rows.append(
                            {
                                "target": target,
                                "variant": variant,
                                "variant_order": CONDITION_VARIANT_ORDER[variant],
                                "candidate": candidate,
                                "l2": l2,
                                "event_prior_matches": (
                                    np.nan
                                    if variant == "player_context_baseline"
                                    else prior_matches
                                ),
                                "slice": slice_name,
                                "training_end": f"{validation_year}-01-01",
                                "validation_year": validation_year,
                                **metrics,
                            }
                        )

    folds = pd.DataFrame(fold_rows)
    selected: dict[str, dict[str, Any]] = {}
    summaries: list[pd.DataFrame] = []
    selected_predictions: list[pd.DataFrame] = []
    for target in TARGETS:
        selected[target] = {}
        for variant in CONDITION_VARIANTS:
            rows = folds.loc[
                folds["target"].eq(target)
                & folds["variant"].eq(variant)
                & folds["slice"].eq("all")
            ]
            candidate, candidate_summary = select_by_validation_loss(rows)
            chosen = rows.loc[rows["candidate"].eq(candidate)].iloc[0]
            key = (target, variant, str(candidate))
            model = candidate_models[key]
            selected[target][variant] = _selected_model_payload(
                model,
                candidate=str(candidate),
                l2=float(chosen["l2"]),
                prior_matches=(
                    None
                    if pd.isna(chosen["event_prior_matches"])
                    else float(chosen["event_prior_matches"])
                ),
                validation_log_loss=float(chosen["log_loss"]),
                validation_n=int(chosen["n"]),
            )
            candidate_summary.insert(0, "target", target)
            candidate_summary.insert(1, "variant", variant)
            candidate_summary.insert(2, "variant_order", CONDITION_VARIANT_ORDER[variant])
            summaries.append(candidate_summary)
            prior_matches = (
                4.0
                if pd.isna(chosen["event_prior_matches"])
                else float(chosen["event_prior_matches"])
            )
            chosen_frame = _target_frame(
                validation, target, prior_matches=prior_matches
            )
            if variant == "player_context_baseline":
                chosen_frame["p_model"] = model.predict(chosen_frame)
                chosen_frame["p_baseline"] = chosen_frame["p_model"]
            else:
                chosen_frame["p_baseline"] = baseline_models[target].predict(
                    chosen_frame
                )
                chosen_frame["p_model"] = model.predict(chosen_frame)
            chosen_frame["target"] = target
            chosen_frame["variant"] = variant
            chosen_frame["candidate"] = str(candidate)
            selected_predictions.append(chosen_frame)

    locked_best = {
        target: min(
            CONDITION_VARIANTS,
            key=lambda variant: (
                selected[target][variant]["validation_log_loss"],
                CONDITION_VARIANT_ORDER[variant],
            ),
        )
        for target in TARGETS
    }

    persistence_rows: list[dict[str, Any]] = []
    persistence_selected: dict[str, float] = {}
    for metric in RATE_METRICS:
        for prior_matches in EVENT_PRIOR_GRID:
            metrics = _persistence_metrics(validation, metric, prior_matches)
            persistence_rows.append(
                {
                    "metric": metric,
                    "event_prior_matches": prior_matches,
                    **metrics,
                }
            )
        eligible = [
            row
            for row in persistence_rows
            if row["metric"] == metric and np.isfinite(row["adjusted_log_loss"])
        ]
        persistence_selected[metric] = float(
            min(eligible, key=lambda row: (row["adjusted_log_loss"], row["event_prior_matches"]))[
                "event_prior_matches"
            ]
        )

    selection_payload = {
        "warmup_years": list(range(config.history_start, training_year)),
        "training_years": [training_year],
        "validation_year": validation_year,
        "selection_metric": "all-match binary log loss",
        "locked_best_variant": locked_best,
        "persistence_event_priors": persistence_selected,
        "selected": selected,
    }
    folds.to_csv(destination / "fold_metrics.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(
        destination / "candidate_summary.csv", index=False
    )
    pd.concat(selected_predictions, ignore_index=True).to_csv(
        destination / "validation_predictions.csv", index=False
    )
    pd.DataFrame(coefficient_rows).to_csv(
        destination / "training_coefficients.csv", index=False
    )
    pd.DataFrame(persistence_rows).to_csv(
        destination / "condition_persistence.csv", index=False
    )
    coverage = (
        panel.assign(year=panel["tournament_date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            panel_rows=("match_key", "size"),
            clean_bo3_targets=("score_target_valid", "sum"),
            valid_point_stats=("point_stats_valid", "sum"),
            event_observed=("event_prior_matches", lambda values: int((values > 0).sum())),
            event_established=(
                "event_prior_matches",
                lambda values: int((values >= ESTABLISHED_EVENT_MATCHES).sum()),
            ),
        )
    )
    coverage.to_csv(destination / "feature_coverage.csv", index=False)
    dedupe_audit.to_csv(destination / "deduplication_audit.csv", index=False)
    _write_json(destination / "selected_config.json", selection_payload)
    summary = {
        "tour": config.tour,
        "forecast_horizon": "round",
        "target_definitions": {
            "over_22_5": "at least 23 games in a completed conventional best-of-three match",
            "deciding_set": "a completed conventional best-of-three match reached a third set",
        },
        "final_holdout_loaded": False,
        "benchmark_status": "pre-2024 validation only",
        "panel_rows": int(len(panel)),
        **selection_payload,
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_CONDITIONS_VALIDATION,
            {
                "forecast_horizon": "round",
                "final_holdout_loaded": False,
                "event_prior_grid": list(EVENT_PRIOR_GRID),
                "l2_grid": list(L2_GRID),
                "condition_variants": {
                    key: list(value) for key, value in CONDITION_VARIANTS.items()
                },
            },
            input_paths=[
                config.project_root / "data" / f"{config.tour}_matches_{year}.csv"
                for year in range(config.history_start, config.test_year)
            ],
        ),
    )
    return summary


def _read_condition_selection(config: ProjectConfig) -> dict[str, Any]:
    path = config.stage_directory(STAGE_CONDITIONS_VALIDATION) / "selected_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run event-conditions-validation first; missing: {path}")
    payload = _read_json(path)
    if set(payload.get("selected", {})) != set(TARGETS):
        raise ValueError("Condition selected_config has unexpected targets")
    if int(payload.get("validation_year", -1)) != config.test_year - 1:
        raise ValueError("Condition selected_config belongs to a different test year")
    if payload.get("training_years") != [config.test_year - 2]:
        raise ValueError("Condition selected_config has unexpected training years")
    if payload.get("warmup_years") != list(
        range(config.history_start, config.test_year - 2)
    ):
        raise ValueError("Condition selected_config has unexpected warmup years")
    return payload


def _paired_binary_gap(
    variant: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    probabilities = reference[["match_key", "p_model"]].rename(
        columns={"p_model": "reference_probability"}
    )
    paired = variant.merge(probabilities, on="match_key", how="inner", validate="one_to_one")
    if len(paired) != len(variant) or len(paired) != len(reference):
        raise ValueError("Condition comparison cohorts differ")
    return paired_block_bootstrap_gap(
        paired,
        model_probability_column="p_model",
        market_probability_column="reference_probability",
        samples=samples,
        seed=seed,
    )


def _persistence_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    prior_matches: float,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    materialized = materialize_event_conditions(frame, prior_matches)
    mask = (
        materialized[f"actual_{metric}_trials"].notna()
        & materialized[f"event_{metric}_residual_count"].gt(0)
        & materialized["event_prior_point_matches"].ge(ESTABLISHED_EVENT_MATCHES)
    )
    work = materialized.loc[mask].copy()
    if work.empty:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    successes = work[f"actual_{metric}_successes"].to_numpy(dtype=float)
    trials = work[f"actual_{metric}_trials"].to_numpy(dtype=float)
    base = np.clip(work[f"expected_{metric}_rate"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    adjusted = np.clip(
        base + work[f"event_{metric}_condition"].to_numpy(dtype=float),
        1e-9,
        1 - 1e-9,
    )
    base_losses = -(successes * np.log(base) + (trials - successes) * np.log(1 - base))
    adjusted_losses = -(
        successes * np.log(adjusted) + (trials - successes) * np.log(1 - adjusted)
    )
    blocks = pd.DataFrame(
        {
            "event_key": work["event_key"].astype(str).to_numpy(),
            "delta": adjusted_losses - base_losses,
            "trials": trials,
        }
    )
    groups = [group for _, group in blocks.groupby("event_key", sort=False)]
    if not groups:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        picked = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[position] for position in picked], ignore_index=True)
        draws[index] = float(sample["delta"].sum() / sample["trials"].sum())
    observed = float((adjusted_losses - base_losses).sum() / trials.sum())
    return {
        "mean": observed,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def run_event_conditions_holdout_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Stage 10: refit locked event models on 2022-23 and benchmark 2024."""

    destination = config.stage_directory(STAGE_CONDITIONS_HOLDOUT)
    _prepare_directory(destination, overwrite)
    selection = _read_condition_selection(config)
    matches, dedupe_audit = _load_condition_matches(config, config.years)
    panel = build_event_condition_panel(matches, config.tour)
    refit_start_year = config.test_year - 2
    training = panel.loc[
        panel["tournament_date"].dt.year.ge(refit_start_year)
        & panel["tournament_date"].lt(config.test_start)
    ].copy()
    holdout = panel.loc[
        panel["tournament_date"].ge(config.test_start)
        & panel["tournament_date"].lt(config.test_end)
    ].copy()

    variant_frames: dict[str, dict[str, pd.DataFrame]] = {}
    coefficient_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        variant_frames[target] = {}
        baseline_spec = selection["selected"][target]["player_context_baseline"]
        baseline_model, baseline_frame = _fit_baseline_candidate(
            training,
            holdout,
            target=target,
            l2=float(baseline_spec["l2"]),
        )
        baseline_frame["candidate"] = baseline_spec["candidate"]
        baseline_frame["variant_order"] = CONDITION_VARIANT_ORDER[
            "player_context_baseline"
        ]
        variant_frames[target]["player_context_baseline"] = baseline_frame
        coefficient_rows.extend(
            {
                **row,
                "target": target,
                "candidate": baseline_spec["candidate"],
                "event_prior_matches": None,
            }
            for row in baseline_model.coefficient_rows("player_context_baseline")
        )

        for variant, spec in selection["selected"][target].items():
            if variant == "player_context_baseline":
                continue
            prior_matches = (
                float(spec["event_prior_matches"])
            )
            model, evaluated = _fit_condition_candidate(
                training,
                holdout,
                target=target,
                variant=variant,
                l2=float(spec["l2"]),
                prior_matches=prior_matches,
                baseline_model=baseline_model,
            )
            evaluated["candidate"] = spec["candidate"]
            evaluated["variant_order"] = CONDITION_VARIANT_ORDER[variant]
            variant_frames[target][variant] = evaluated
            coefficient_rows.extend(
                {
                    **row,
                    "target": target,
                    "candidate": spec["candidate"],
                    "event_prior_matches": spec["event_prior_matches"],
                }
                for row in model.coefficient_rows(variant)
            )

    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for target, frames in variant_frames.items():
        ordered = sorted(frames, key=lambda value: CONDITION_VARIANT_ORDER[value])
        baseline = frames["player_context_baseline"]
        for position, variant in enumerate(ordered):
            frame = frames[variant]
            prediction_frames.append(frame)
            previous_variant = ordered[max(0, position - 1)]
            previous = frames[previous_variant]
            for slice_name, sliced in _binary_slices(frame):
                if sliced.empty:
                    continue
                keys = set(sliced["match_key"])
                baseline_slice = baseline.loc[baseline["match_key"].isin(keys)]
                previous_slice = previous.loc[previous["match_key"].isin(keys)]
                base_gap = _paired_binary_gap(
                    sliced, baseline_slice, samples=bootstrap_samples, seed=42
                )
                previous_gap = _paired_binary_gap(
                    sliced, previous_slice, samples=bootstrap_samples, seed=84
                )
                metrics = _evaluate_binary(sliced)
                metric_rows.append(
                    {
                        "tour": config.tour,
                        "target": target,
                        "variant": variant,
                        "variant_order": CONDITION_VARIANT_ORDER[variant],
                        "previous_variant": previous_variant,
                        "slice": slice_name,
                        **metrics,
                        "delta_log_loss_vs_baseline": base_gap["mean"],
                        "delta_vs_baseline_ci_low": base_gap["ci_low"],
                        "delta_vs_baseline_ci_high": base_gap["ci_high"],
                        "delta_log_loss_vs_previous": previous_gap["mean"],
                        "delta_vs_previous_ci_low": previous_gap["ci_low"],
                        "delta_vs_previous_ci_high": previous_gap["ci_high"],
                    }
                )
                delta_rows.append(
                    {
                        "tour": config.tour,
                        "target": target,
                        "variant": variant,
                        "slice": slice_name,
                        "n": int(len(sliced)),
                        "delta_log_loss_vs_baseline": base_gap["mean"],
                        "baseline_ci_low": base_gap["ci_low"],
                        "baseline_ci_high": base_gap["ci_high"],
                        "delta_log_loss_vs_previous": previous_gap["mean"],
                        "previous_ci_low": previous_gap["ci_low"],
                        "previous_ci_high": previous_gap["ci_high"],
                    }
                )

    persistence_rows: list[dict[str, Any]] = []
    for metric, prior_matches in selection["persistence_event_priors"].items():
        metrics = _persistence_metrics(holdout, metric, float(prior_matches))
        interval = _persistence_bootstrap(
            holdout,
            metric,
            float(prior_matches),
            samples=bootstrap_samples,
            seed=126,
        )
        persistence_rows.append(
            {
                "tour": config.tour,
                "metric": metric,
                "event_prior_matches": float(prior_matches),
                **metrics,
                "delta_ci_low": interval["ci_low"],
                "delta_ci_high": interval["ci_high"],
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target", "slice", "variant_order"], kind="stable"
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    persistence = pd.DataFrame(persistence_rows)
    predictions.to_csv(destination / "predictions.csv", index=False)
    metrics.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(destination / "paired_deltas.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(destination / "coefficients.csv", index=False)
    persistence.to_csv(destination / "condition_persistence.csv", index=False)
    dedupe_audit.to_csv(destination / "deduplication_audit.csv", index=False)
    holdout_coverage = pd.DataFrame(
        [
            {
                "tour": config.tour,
                "year": config.test_year,
                "panel_rows": int(len(holdout)),
                "clean_bo3_targets": int(holdout["score_target_valid"].sum()),
                "valid_point_stats": int(holdout["point_stats_valid"].sum()),
                "event_observed": int(holdout["event_prior_matches"].gt(0).sum()),
                "event_established": int(
                    holdout["event_prior_matches"].ge(ESTABLISHED_EVENT_MATCHES).sum()
                ),
            }
        ]
    )
    holdout_coverage.to_csv(destination / "feature_coverage.csv", index=False)

    all_metrics = metrics.loc[metrics["slice"].eq("all")].to_dict("records")
    locked_metrics = [
        row
        for row in all_metrics
        if row["variant"] == selection["locked_best_variant"][row["target"]]
    ]
    summary = {
        "tour": config.tour,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "forecast_horizon": "round",
        "market_prices_used": False,
        "profitability_claim_allowed": False,
        "benchmark_status": (
            "reused forward benchmark; out-of-sample for fitting but not a fresh "
            "confirmatory holdout"
        ),
        "refit_years": list(range(refit_start_year, config.test_year)),
        "targets": list(TARGETS),
        "locked_best_variant_from_validation": selection["locked_best_variant"],
        "locked_holdout_metrics": locked_metrics,
        "all_metrics": all_metrics,
        "condition_persistence": persistence_rows,
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_CONDITIONS_HOLDOUT,
            {
                "forecast_horizon": "round",
                "bootstrap_samples": bootstrap_samples,
                "market_prices_used": False,
                "upstream_artifacts": [
                    _digest_record(
                        config.stage_directory(STAGE_CONDITIONS_VALIDATION)
                        / "selected_config.json",
                        config.project_root,
                    )
                ],
            },
            input_paths=[
                config.project_root / "data" / f"{config.tour}_matches_{year}.csv"
                for year in config.years
            ],
        ),
    )
    return summary


__all__ = [
    "ALL_CONDITION_METRICS",
    "BASE_FEATURES",
    "CONDITION_VARIANTS",
    "EVENT_PRIOR_GRID",
    "L2_GRID",
    "STAGE_CONDITIONS_HOLDOUT",
    "STAGE_CONDITIONS_VALIDATION",
    "build_event_condition_panel",
    "materialize_event_conditions",
    "parse_standard_bo3_score",
    "run_event_conditions_holdout_stage",
    "run_event_conditions_validation_stage",
]
