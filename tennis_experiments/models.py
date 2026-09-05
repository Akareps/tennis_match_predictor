"""Small online models with an explicit predict-then-update contract."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log
from typing import Iterable, Mapping, Any

import numpy as np


DEFAULT_K_BY_LEVEL = {
    "G": 32.0,
    "M": 28.0,
    "PM": 28.0,
    "F": 28.0,
    "A": 24.0,
    "P": 24.0,
    "W": 24.0,
    "D": 20.0,
    "O": 20.0,
    "I": 20.0,
    "C": 18.0,
    "S": 14.0,
}


def elo_probability(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    """Return the standard base-10 Elo probability for player A."""

    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def _optional_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


@dataclass(frozen=True)
class RankTransform:
    """Map an official rank onto the Elo rating scale."""

    intercept: float
    log_rank_slope: float
    minimum_rank: int = 1
    maximum_rank: int = 2000

    def rating(self, rank: Any) -> float | None:
        numeric = _optional_float(rank)
        if numeric is None or numeric < self.minimum_rank:
            return None
        clipped = min(float(self.maximum_rank), numeric)
        return self.intercept + self.log_rank_slope * log(clipped)


class EloModel:
    """Overall or surface-blended Elo with order-independent batch updates.

    A caller predicts every match in a chronology batch before calling
    :meth:`batch_update`.  All update residuals are computed from the state at
    the start of the batch, preventing within-round outcome leakage.
    """

    name = "surface_elo"

    def __init__(
        self,
        *,
        initial_rating: float = 1500.0,
        scale: float = 400.0,
        surface_weight: float = 0.5,
        default_k: float = 24.0,
        k_by_level: Mapping[str, float] | None = None,
    ) -> None:
        if not 0.0 <= surface_weight <= 1.0:
            raise ValueError("surface_weight must be between 0 and 1")
        self.initial_rating = float(initial_rating)
        self.scale = float(scale)
        self.surface_weight = float(surface_weight)
        self.default_k = float(default_k)
        self.k_by_level = dict(DEFAULT_K_BY_LEVEL if k_by_level is None else k_by_level)
        self.overall: dict[int, float] = defaultdict(lambda: self.initial_rating)
        self.by_surface: dict[str, dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: self.initial_rating)
        )
        self.observations: dict[int, float] = defaultdict(float)
        self.surface_observations: dict[str, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.main_observations: dict[int, float] = defaultdict(float)

    def _prediction_ratings(
        self,
        player_id: int,
        surface: str,
        rank: Any = None,
    ) -> tuple[float, float]:
        del rank
        return self.overall[int(player_id)], self.by_surface[str(surface)][int(player_id)]

    def predict_details(
        self,
        player_a_id: int,
        player_b_id: int,
        surface: str,
        *,
        a_rank: Any = None,
        b_rank: Any = None,
    ) -> dict[str, float]:
        overall_a, surface_a = self._prediction_ratings(player_a_id, surface, a_rank)
        overall_b, surface_b = self._prediction_ratings(player_b_id, surface, b_rank)
        p_overall = elo_probability(overall_a, overall_b, self.scale)
        p_surface = elo_probability(surface_a, surface_b, self.scale)
        p_model = (1.0 - self.surface_weight) * p_overall + self.surface_weight * p_surface
        return {
            "p_model": float(p_model),
            "p_overall": float(p_overall),
            "p_surface": float(p_surface),
            "rating_a": float(overall_a),
            "rating_b": float(overall_b),
            "surface_rating_a": float(surface_a),
            "surface_rating_b": float(surface_b),
            "a_observations": float(self.observations[int(player_a_id)]),
            "b_observations": float(self.observations[int(player_b_id)]),
            "a_main_observations": float(self.main_observations[int(player_a_id)]),
            "b_main_observations": float(self.main_observations[int(player_b_id)]),
        }

    def predict(self, player_a_id: int, player_b_id: int, surface: str, **kwargs: Any) -> float:
        return self.predict_details(player_a_id, player_b_id, surface, **kwargs)["p_model"]

    def _k(self, level: Any) -> float:
        return float(self.k_by_level.get(str(level), self.default_k))

    def _update_weight(self, match: Mapping[str, Any]) -> float:
        del match
        return 1.0

    def _match_k(self, match: Mapping[str, Any]) -> float:
        return self._k(match.get("tourney_level", "?")) * self._update_weight(match)

    def batch_update(self, matches: Iterable[Mapping[str, Any]]) -> None:
        """Apply labeled matches as one simultaneous update batch."""

        overall_delta: dict[int, float] = defaultdict(float)
        surface_delta: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        observation_delta: dict[int, float] = defaultdict(float)
        surface_observation_delta: dict[str, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        main_observation_delta: dict[int, float] = defaultdict(float)

        for match in matches:
            player_a = int(match["player_a_id"])
            player_b = int(match["player_b_id"])
            surface = str(match.get("surface", "Unknown"))
            outcome = float(match["actual_a_won"])
            if outcome not in {0.0, 1.0}:
                raise ValueError("actual_a_won must be 0 or 1")
            details = self.predict_details(
                player_a,
                player_b,
                surface,
                a_rank=match.get("a_rank"),
                b_rank=match.get("b_rank"),
            )
            update_weight = self._update_weight(match)
            if not np.isfinite(update_weight) or update_weight < 0:
                raise ValueError("update weights must be finite and non-negative")
            k = self._match_k(match)
            overall_change = k * (outcome - details["p_overall"])
            surface_change = k * (outcome - details["p_surface"])
            overall_delta[player_a] += overall_change
            overall_delta[player_b] -= overall_change
            surface_delta[surface][player_a] += surface_change
            surface_delta[surface][player_b] -= surface_change
            observation_delta[player_a] += update_weight
            observation_delta[player_b] += update_weight
            surface_observation_delta[surface][player_a] += update_weight
            surface_observation_delta[surface][player_b] += update_weight
            if str(match.get("source_group", "main")) == "main":
                main_observation_delta[player_a] += 1.0
                main_observation_delta[player_b] += 1.0

        for player_id, delta in overall_delta.items():
            self.overall[player_id] += delta
        for surface, changes in surface_delta.items():
            for player_id, delta in changes.items():
                self.by_surface[surface][player_id] += delta
        for player_id, count in observation_delta.items():
            self.observations[player_id] += count
        for surface, counts in surface_observation_delta.items():
            for player_id, count in counts.items():
                self.surface_observations[surface][player_id] += count
        for player_id, count in main_observation_delta.items():
            self.main_observations[player_id] += count


class OverallEloModel(EloModel):
    name = "overall_elo"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["surface_weight"] = 0.0
        super().__init__(**kwargs)


class RankAugmentedEloModel(EloModel):
    """Elo whose prediction ratings retain a controlled official-rank signal.

    Unlike the archived implementation, the latent Elo state is not replaced
    by a rank/Elo mixture after every match.  Rank affects the prediction and
    update residual, while the learned Elo state remains interpretable.
    """

    name = "rank_augmented_elo"

    def __init__(
        self,
        rank_transform: RankTransform,
        *,
        warmup_matches: int = 30,
        warmup_rank_weight: float = 1.0,
        continuous_rank_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if warmup_matches < 1:
            raise ValueError("warmup_matches must be positive")
        for value, label in (
            (warmup_rank_weight, "warmup_rank_weight"),
            (continuous_rank_weight, "continuous_rank_weight"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")
        self.rank_transform = rank_transform
        self.warmup_matches = int(warmup_matches)
        self.warmup_rank_weight = float(warmup_rank_weight)
        self.continuous_rank_weight = float(continuous_rank_weight)

    def rank_weight(self, player_id: int) -> float:
        n = self.observations[int(player_id)]
        remaining = max(0.0, 1.0 - n / self.warmup_matches)
        return self.continuous_rank_weight + (
            self.warmup_rank_weight - self.continuous_rank_weight
        ) * remaining

    def _prediction_ratings(
        self,
        player_id: int,
        surface: str,
        rank: Any = None,
    ) -> tuple[float, float]:
        elo_overall, elo_surface = super()._prediction_ratings(player_id, surface, rank)
        rank_rating = self.rank_transform.rating(rank)
        if rank_rating is None:
            return elo_overall, elo_surface
        weight = self.rank_weight(player_id)
        return (
            (1.0 - weight) * elo_overall + weight * rank_rating,
            (1.0 - weight) * elo_surface + weight * rank_rating,
        )


class TierWeightedRankAugmentedEloModel(RankAugmentedEloModel):
    """Rank-augmented Elo with lower circuits treated as fractional evidence."""

    name = "tier_weighted_rank_augmented_elo"

    def __init__(
        self,
        rank_transform: RankTransform,
        *,
        source_weights: Mapping[str, float],
        unlisted_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        weights = {"main": 1.0, **{str(key): float(value) for key, value in source_weights.items()}}
        values = [*weights.values(), float(unlisted_weight)]
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("source weights must be finite and non-negative")
        self.source_weights = weights
        self.unlisted_weight = float(unlisted_weight)
        super().__init__(rank_transform, **kwargs)

    def _update_weight(self, match: Mapping[str, Any]) -> float:
        group = str(match.get("source_group", "main"))
        return self.source_weights.get(group, self.unlisted_weight)


def fit_rank_transform(
    batches: Iterable[Iterable[Mapping[str, Any]]],
    *,
    established_threshold: int = 30,
    maximum_rank: int = 2000,
    elo_kwargs: Mapping[str, Any] | None = None,
) -> RankTransform:
    """Fit ``rating = intercept + slope * log(rank)`` without same-batch leakage."""

    model = OverallEloModel(**dict(elo_kwargs or {}))
    rank_values: list[float] = []
    rating_values: list[float] = []
    for batch_iter in batches:
        batch = list(batch_iter)
        for match in batch:
            for side in ("a", "b"):
                player_id = int(match[f"player_{side}_id"])
                rank = _optional_float(match.get(f"{side}_rank"))
                if (
                    rank is not None
                    and 1 <= rank <= maximum_rank
                    and model.observations[player_id] >= established_threshold
                ):
                    rank_values.append(log(rank))
                    rating_values.append(model.overall[player_id])
        model.batch_update(batch)
    if len(rank_values) < 2 or np.var(rank_values) == 0:
        raise ValueError("Not enough established rank observations to fit a rank transform")
    design = np.column_stack([np.ones(len(rank_values)), np.asarray(rank_values)])
    intercept, slope = np.linalg.lstsq(design, np.asarray(rating_values), rcond=None)[0]
    return RankTransform(float(intercept), float(slope), maximum_rank=maximum_rank)
