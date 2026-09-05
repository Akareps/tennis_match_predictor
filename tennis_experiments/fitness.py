"""Causal workload and recent-availability experiments.

The source match files publish tournament start dates rather than exact match
timestamps.  This module therefore uses the same conservative round batches as
the core backtester.  A target can see strictly earlier tournament dates and
completed earlier rounds from its own event; it cannot see another result from
the same chronology batch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, iter_chronology_batches, run_model_factories
from .cross_tier import (
    VARIANT_ORDER as CROSS_TIER_VARIANT_ORDER,
    _load_holdout_cross_tier,
    _load_pretest_cross_tier,
    _market_cohort,
    _model_from_spec,
    _read_cross_tier_selection,
    _require_lower_tier_inputs,
    _validation_fold_data,
)
from .experiment import (
    ProjectConfig,
    _digest_record,
    _fit_rank_before,
    _manifest,
    _prepare_directory,
    _read_json,
    _write_json,
)
from .metrics import (
    binary_log_loss,
    evaluate_frame,
    paired_block_bootstrap_gap,
)
from .models import EloModel
from .validation import select_by_validation_loss


STAGE_WORKLOAD_VALIDATION = "07_workload_validation"
STAGE_WORKLOAD_HOLDOUT = "08_workload_holdout"

RECOVERY_FEATURES = (
    "diff_short_rest",
    "diff_long_layoff",
    "diff_surface_switch",
    "diff_history_missing",
)
RECENT_LOAD_FEATURES = (
    "diff_matches_14d",
    "diff_games_14d",
    "diff_games_missing_14d",
    "diff_minutes_14d",
    "diff_minutes_missing_14d",
    "diff_bo5_sets_14d",
)
SAME_EVENT_FEATURES = (
    "diff_same_event_matches",
    "diff_same_event_games",
    "diff_same_event_games_missing",
    "diff_same_event_minutes",
    "diff_same_event_minutes_missing",
    "diff_same_event_bo5_sets",
)
AVAILABILITY_FEATURES = (
    "diff_retirement_decay",
    "diff_retirements_90d",
)

WORKLOAD_VARIANTS: dict[str, tuple[str, ...] | None] = {
    "base_model": None,
    "recalibrated": (),
    "plus_recovery": RECOVERY_FEATURES,
    "plus_recent_load": RECOVERY_FEATURES + RECENT_LOAD_FEATURES,
    "plus_same_event_load": RECOVERY_FEATURES
    + RECENT_LOAD_FEATURES
    + SAME_EVENT_FEATURES,
    "plus_availability": RECOVERY_FEATURES
    + RECENT_LOAD_FEATURES
    + SAME_EVENT_FEATURES
    + AVAILABILITY_FEATURES,
}
WORKLOAD_VARIANT_ORDER = {name: index for index, name in enumerate(WORKLOAD_VARIANTS)}
L2_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

_SET_SCORE = re.compile(r"^(\d+)-(\d+)(?:\(\d+\))?$")
_TERMINATION = re.compile(r"\b(?:RET|DEF|ABN)\b", flags=re.IGNORECASE)


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def score_load(score: Any) -> dict[str, Any]:
    """Extract conservative physical-load quantities from a Sackmann score."""

    rendered = "" if score is None else str(score).strip()
    upper = rendered.upper().replace(" ", "")
    walkover = upper in {"W/O", "WO"}
    if walkover:
        return {"played": False, "games": None, "sets": None, "walkover": True}

    cleaned = _TERMINATION.sub("", rendered).strip()
    games: list[int] = []
    for raw_token in cleaned.split():
        token = raw_token.strip("[]")
        match = _SET_SCORE.fullmatch(token)
        if match is None:
            continue
        left, right = int(match.group(1)), int(match.group(2))
        # A match tiebreak such as 10-8 is not eighteen conventional games.
        games.append(min(left + right, 13))
    return {
        "played": True,
        "games": int(sum(games)) if games else None,
        "sets": int(len(games)) if games else None,
        "walkover": False,
    }


def _side_snapshot(
    observations: Sequence[Mapping[str, Any]],
    *,
    event_date: pd.Timestamp,
    event_key: str,
    surface: str,
) -> dict[str, float]:
    played = [item for item in observations if bool(item["played"])]
    prior_events = [item for item in played if pd.Timestamp(item["date"]) < event_date]
    same_event = [item for item in played if str(item["event_key"]) == event_key]

    if prior_events:
        last_date = max(pd.Timestamp(item["date"]) for item in prior_events)
        last_items = [item for item in prior_events if pd.Timestamp(item["date"]) == last_date]
        last_surface = str(last_items[-1]["surface"])
        days_since = max(0, int((event_date - last_date).days))
        history_missing = 0.0
        short_rest = exp(-days_since / 7.0)
        long_layoff = float(days_since > 45)
        surface_switch = float(
            surface != "Unknown"
            and last_surface != "Unknown"
            and surface != last_surface
        )
    else:
        days_since = 365
        history_missing = 1.0
        short_rest = 0.0
        long_layoff = 0.0
        surface_switch = 0.0

    recent = [
        item
        for item in played
        if 0 < int((event_date - pd.Timestamp(item["date"])).days) <= 14
    ]
    availability = [
        item
        for item in observations
        if bool(item["availability_event"])
        and (
            pd.Timestamp(item["date"]) < event_date
            or str(item["event_key"]) == event_key
        )
        and 0 <= int((event_date - pd.Timestamp(item["date"])).days) <= 90
    ]
    if availability:
        last_availability = max(pd.Timestamp(item["date"]) for item in availability)
        availability_days = max(0, int((event_date - last_availability).days))
        retirement_decay = exp(-availability_days / 30.0)
    else:
        retirement_decay = 0.0

    def total(items: Sequence[Mapping[str, Any]], field: str) -> float:
        return float(sum(float(item[field]) for item in items if item[field] is not None))

    def missing(items: Sequence[Mapping[str, Any]], field: str) -> float:
        return float(sum(item[field] is None for item in items))

    return {
        "days_since_previous_event": float(days_since),
        "history_missing": history_missing,
        "short_rest": float(short_rest),
        "long_layoff": long_layoff,
        "surface_switch": surface_switch,
        "matches_14d": float(len(recent)),
        "games_14d": total(recent, "games"),
        "games_missing_14d": missing(recent, "games"),
        "minutes_14d": total(recent, "minutes"),
        "minutes_missing_14d": missing(recent, "minutes"),
        "bo5_sets_14d": total(recent, "bo5_sets"),
        "same_event_matches": float(len(same_event)),
        "same_event_games": total(same_event, "games"),
        "same_event_games_missing": missing(same_event, "games"),
        "same_event_minutes": total(same_event, "minutes"),
        "same_event_minutes_missing": missing(same_event, "minutes"),
        "same_event_bo5_sets": total(same_event, "bo5_sets"),
        "retirement_decay": float(retirement_decay),
        "retirements_90d": float(len(availability)),
    }


def build_workload_features(main_matches: pd.DataFrame) -> pd.DataFrame:
    """Build one pre-match feature row per main-tour match without batch leakage."""

    required = {
        "match_key",
        "event_key",
        "tournament_date",
        "surface",
        "score",
        "minutes",
        "best_of",
        "player_a_id",
        "player_b_id",
        "actual_a_won",
        "is_retirement",
    }
    missing_columns = sorted(required - set(main_matches.columns))
    if missing_columns:
        raise ValueError("Missing workload columns: " + ", ".join(missing_columns))
    if not main_matches["source_group"].eq("main").all():
        raise ValueError("Workload features must be built from main-tour rows only")

    history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    side_fields: tuple[str, ...] | None = None
    for batch_id, batch in iter_chronology_batches(main_matches, "round"):
        records = batch.replace({np.nan: None}).to_dict("records")
        for record in records:
            event_date = pd.Timestamp(record["tournament_date"])
            snapshots = {}
            for side in ("a", "b"):
                player_id = int(record[f"player_{side}_id"])
                snapshots[side] = _side_snapshot(
                    history[player_id],
                    event_date=event_date,
                    event_key=str(record["event_key"]),
                    surface=str(record.get("surface", "Unknown")),
                )
            side_fields = tuple(snapshots["a"])
            output: dict[str, Any] = {
                "match_key": record["match_key"],
                "event_key": record["event_key"],
                "tournament_date": event_date,
                "round": record.get("round"),
                "chronology_batch": batch_id,
                "player_a_id": int(record["player_a_id"]),
                "player_b_id": int(record["player_b_id"]),
            }
            for field in side_fields:
                output[f"a_{field}"] = snapshots["a"][field]
                output[f"b_{field}"] = snapshots["b"][field]
                output[f"diff_{field}"] = snapshots["a"][field] - snapshots["b"][field]
            rows.append(output)

        # Reveal the entire batch only after all feature snapshots are frozen.
        for record in records:
            load = score_load(record.get("score"))
            minutes = _optional_number(record.get("minutes"))
            if minutes is not None and minutes < 0:
                minutes = None
            sets = load["sets"]
            observation_base = {
                "date": pd.Timestamp(record["tournament_date"]),
                "event_key": str(record["event_key"]),
                "surface": str(record.get("surface", "Unknown")),
                "played": bool(load["played"]),
                "games": load["games"],
                "sets": sets,
                "minutes": minutes if load["played"] else None,
                "bo5_sets": (
                    float(sets)
                    if load["played"] and sets is not None and int(record.get("best_of", 3)) == 5
                    else 0.0
                ),
            }
            losing_side = "b" if int(record["actual_a_won"]) == 1 else "a"
            for side in ("a", "b"):
                observation = dict(observation_base)
                observation["availability_event"] = bool(record.get("is_retirement", False)) and side == losing_side
                history[int(record[f"player_{side}_id"])].append(observation)

    result = pd.DataFrame(rows)
    if result["match_key"].duplicated().any():
        raise ValueError("Workload feature panel contains duplicate match keys")
    expected = {
        feature
        for features in WORKLOAD_VARIANTS.values()
        if features is not None
        for feature in features
    }
    absent = sorted(expected - set(result.columns))
    if absent:
        raise AssertionError("Workload feature construction missed: " + ", ".join(absent))
    return result


@dataclass(frozen=True)
class OffsetLogisticModel:
    """A fitted workload residual on top of fixed Elo log-odds."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    l2: float
    iterations: int
    converged: bool

    def predict(self, frame: pd.DataFrame, *, base_column: str = "p_model") -> np.ndarray:
        base = np.clip(frame[base_column].to_numpy(dtype=float), 1e-9, 1.0 - 1e-9)
        offset = np.log(base / (1.0 - base))
        if self.feature_names:
            values = frame[list(self.feature_names)].to_numpy(dtype=float)
            means = np.asarray(self.means)
            scales = np.asarray(self.scales)
            standardized = (values - means) / scales
            design = np.column_stack([np.ones(len(frame)), standardized])
        else:
            design = np.ones((len(frame), 1), dtype=float)
        eta = np.clip(offset + design @ np.asarray(self.coefficients), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-eta))

    def coefficient_rows(self, variant: str) -> list[dict[str, Any]]:
        rows = [
            {
                "variant": variant,
                "feature": "intercept",
                "coefficient_standardized": self.coefficients[0],
                "coefficient_raw": self.coefficients[0]
                - sum(
                    coefficient * mean / scale
                    for coefficient, mean, scale in zip(
                        self.coefficients[1:], self.means, self.scales
                    )
                ),
                "training_mean": 0.0,
                "training_scale": 1.0,
                "l2": self.l2,
                "iterations": self.iterations,
                "converged": self.converged,
            }
        ]
        for feature, mean, scale, coefficient in zip(
            self.feature_names, self.means, self.scales, self.coefficients[1:]
        ):
            rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "coefficient_standardized": coefficient,
                    "coefficient_raw": coefficient / scale,
                    "training_mean": mean,
                    "training_scale": scale,
                    "l2": self.l2,
                    "iterations": self.iterations,
                    "converged": self.converged,
                }
            )
        return rows


def fit_offset_logistic(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    l2: float,
    base_column: str = "p_model",
    outcome_column: str = "actual_a_won",
    max_iterations: int = 100,
) -> OffsetLogisticModel:
    """Fit ``logit(y) = logit(base) + intercept + workload residual``."""

    if l2 < 0 or not np.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    columns = [base_column, outcome_column, *feature_names]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("Missing offset-model columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Cannot fit an offset model on an empty frame")

    base = np.clip(frame[base_column].to_numpy(dtype=float), 1e-9, 1.0 - 1e-9)
    outcomes = frame[outcome_column].to_numpy(dtype=float)
    if not np.isin(outcomes, [0.0, 1.0]).all() or len(np.unique(outcomes)) < 2:
        raise ValueError("Offset-model outcomes must contain both binary classes")
    offset = np.log(base / (1.0 - base))

    if feature_names:
        values = frame[list(feature_names)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Workload features must be finite")
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        scales = np.where(scales < 1e-9, 1.0, scales)
        design = np.column_stack([np.ones(len(frame)), (values - means) / scales])
    else:
        means = np.asarray([], dtype=float)
        scales = np.asarray([], dtype=float)
        design = np.ones((len(frame), 1), dtype=float)

    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag([0.0, *([float(l2)] * (design.shape[1] - 1))])
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
            raise ValueError("Offset-model information matrix is singular") from error
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break

    if not np.isfinite(beta).all():
        raise ValueError("Offset-model fit produced non-finite coefficients")
    return OffsetLogisticModel(
        feature_names=tuple(feature_names),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in beta),
        l2=float(l2),
        iterations=iterations,
        converged=converged,
    )


def _best_cross_tier_base(config: ProjectConfig) -> tuple[str, dict[str, Any], dict[str, Any]]:
    selection = _read_cross_tier_selection(config)
    summary_path = config.stage_directory("05_cross_tier_validation") / "candidate_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing cross-tier candidate summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for variant, spec in selection["selected"].items():
        matched = summary.loc[
            summary["variant"].eq(variant) & summary["candidate"].eq(spec["candidate"])
        ]
        if len(matched) != 1:
            raise ValueError(f"Cannot identify selected cross-tier row for {variant}")
        rows.append(
            {
                "variant": variant,
                "validation_log_loss": float(matched.iloc[0]["validation_log_loss"]),
                "variant_order": CROSS_TIER_VARIANT_ORDER[variant],
            }
        )
    winner = sorted(
        rows,
        key=lambda row: (row["validation_log_loss"], row["variant_order"]),
    )[0]
    variant = str(winner["variant"])
    return variant, dict(selection["selected"][variant]), dict(selection["base_model"])


def _base_predictions_for_year(
    matches: pd.DataFrame,
    main_matches: pd.DataFrame,
    *,
    year: int,
    base_spec: Mapping[str, Any],
    base_candidate: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    rank_transform = _fit_rank_before(main_matches, start, "round")
    fold_data = _validation_fold_data(matches, start, end)
    factory: Callable[[], EloModel] = lambda: _model_from_spec(
        base_spec, base_candidate, rank_transform
    )
    predictions = run_model_factories(
        fold_data,
        {"base_model": factory},
        BacktestConfig(
            test_start=start,
            test_end=end,
            horizon="round",
            update_retirements=False,
            evaluate_retirements=False,
        ),
    )["base_model"]
    if not predictions["source_group"].eq("main").all():
        raise AssertionError("Workload OOS predictions emitted a lower-tier target")
    predictions["prediction_year"] = int(year)
    return predictions, {
        "year": int(year),
        "intercept": float(rank_transform.intercept),
        "log_rank_slope": float(rank_transform.log_rank_slope),
    }


def _join_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    payload = features.drop(
        columns=[
            column
            for column in (
                "event_key",
                "tournament_date",
                "round",
                "chronology_batch",
                "player_a_id",
                "player_b_id",
            )
            if column in features
        ]
    )
    joined = predictions.merge(payload, on="match_key", how="left", validate="one_to_one")
    required_features = sorted(
        {
            feature
            for values in WORKLOAD_VARIANTS.values()
            if values is not None
            for feature in values
        }
    )
    if joined[required_features].isna().any().any():
        raise ValueError("A base prediction is missing causal workload features")
    return joined


def _feature_coverage(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["year"] = pd.to_datetime(frame["tournament_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        rows.append(
            {
                "year": int(year),
                "matches": int(len(group)),
                "either_missing_prior_history": int(
                    (group["a_history_missing"].gt(0) | group["b_history_missing"].gt(0)).sum()
                ),
                "either_recent_match_14d": int(
                    (group["a_matches_14d"].gt(0) | group["b_matches_14d"].gt(0)).sum()
                ),
                "either_same_event_load": int(
                    (group["a_same_event_matches"].gt(0) | group["b_same_event_matches"].gt(0)).sum()
                ),
                "either_recent_availability_event": int(
                    (group["a_retirement_decay"].gt(0) | group["b_retirement_decay"].gt(0)).sum()
                ),
                "minutes_missing_recent_total": int(
                    group["a_minutes_missing_14d"].sum()
                    + group["b_minutes_missing_14d"].sum()
                ),
                "minutes_missing_same_event_total": int(
                    group["a_same_event_minutes_missing"].sum()
                    + group["b_same_event_minutes_missing"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _slices(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", frame
    yield "cold_start", frame.loc[frame["is_cold_start"]]
    yield "established", frame.loc[~frame["is_cold_start"]]


def _variant_frame(
    base: pd.DataFrame,
    variant: str,
    model: OffsetLogisticModel | None,
) -> pd.DataFrame:
    output = base.copy()
    output["p_base"] = output["p_model"].astype(float)
    if model is not None:
        output["p_model"] = model.predict(output, base_column="p_base")
    output["variant"] = variant
    output["variant_order"] = WORKLOAD_VARIANT_ORDER[variant]
    return output


def run_workload_validation_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stage 07: select residual shrinkage entirely before the final holdout."""

    destination = config.stage_directory(STAGE_WORKLOAD_VALIDATION)
    _prepare_directory(destination, overwrite)
    matches, dedupe_audit = _load_pretest_cross_tier(config)
    main = matches.loc[matches["source_group"].eq("main")].copy()
    features = build_workload_features(main)
    base_variant, base_spec, base_candidate = _best_cross_tier_base(config)

    oos_years = list(range(config.history_start + 1, config.test_year))
    if len(oos_years) < 2:
        raise ValueError("Workload validation needs at least two pre-holdout OOS years")
    validation_year = oos_years[-1]
    residual_training_years = oos_years[:-1]
    panels: list[pd.DataFrame] = []
    rank_rows: list[dict[str, float]] = []
    for year in oos_years:
        predictions, rank_row = _base_predictions_for_year(
            matches,
            main,
            year=year,
            base_spec=base_spec,
            base_candidate=base_candidate,
        )
        panels.append(_join_features(predictions, features))
        rank_rows.append(rank_row)
    oos = pd.concat(panels, ignore_index=True)
    residual_train = oos.loc[oos["prediction_year"].isin(residual_training_years)].copy()
    validation = oos.loc[oos["prediction_year"].eq(validation_year)].copy()

    evaluation_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    candidate_frames: dict[str, pd.DataFrame] = {}
    candidate_models: dict[str, OffsetLogisticModel | None] = {}
    for variant, feature_names in WORKLOAD_VARIANTS.items():
        penalties = (None,) if feature_names is None else ((0.0,) if not feature_names else L2_GRID)
        for penalty in penalties:
            if feature_names is None:
                candidate = "base_model"
                model = None
            else:
                candidate = f"{variant}__l2_{float(penalty):g}"
                model = fit_offset_logistic(
                    residual_train, feature_names, l2=float(penalty)
                )
                coefficient_rows.extend(model.coefficient_rows(candidate))
            evaluated = _variant_frame(validation, variant, model)
            candidate_frames[candidate] = evaluated
            candidate_models[candidate] = model
            for slice_name, sliced in _slices(evaluated):
                if sliced.empty:
                    continue
                evaluation_rows.append(
                    {
                        "variant": variant,
                        "variant_order": WORKLOAD_VARIANT_ORDER[variant],
                        "candidate": candidate,
                        "l2": np.nan if penalty is None else float(penalty),
                        "slice": slice_name,
                        "training_years": ",".join(map(str, residual_training_years)),
                        "validation_year": validation_year,
                        "n": int(len(sliced)),
                        "log_loss": binary_log_loss(
                            sliced["p_model"], sliced["actual_a_won"]
                        ),
                    }
                )

    evaluations = pd.DataFrame(evaluation_rows)
    selected: dict[str, dict[str, Any]] = {}
    summaries: list[pd.DataFrame] = []
    selected_prediction_frames: list[pd.DataFrame] = []
    for variant, feature_names in WORKLOAD_VARIANTS.items():
        all_rows = evaluations.loc[
            evaluations["variant"].eq(variant) & evaluations["slice"].eq("all")
        ]
        candidate, candidate_summary = select_by_validation_loss(all_rows)
        selected_row = all_rows.loc[all_rows["candidate"].eq(candidate)].iloc[0]
        selected[variant] = {
            "candidate": str(candidate),
            "l2": None if pd.isna(selected_row["l2"]) else float(selected_row["l2"]),
            "features": [] if feature_names is None else list(feature_names),
            "validation_log_loss": float(selected_row["log_loss"]),
            "validation_n": int(selected_row["n"]),
        }
        candidate_summary.insert(0, "variant", variant)
        candidate_summary.insert(1, "variant_order", WORKLOAD_VARIANT_ORDER[variant])
        summaries.append(candidate_summary)
        selected_prediction_frames.append(candidate_frames[str(candidate)])

    locked_best_variant = min(
        selected,
        key=lambda variant: (
            selected[variant]["validation_log_loss"],
            WORKLOAD_VARIANT_ORDER[variant],
        ),
    )
    selection_payload = {
        "base_cross_tier_variant": base_variant,
        "base_cross_tier_spec": base_spec,
        "base_model_candidate": base_candidate,
        "residual_training_years": residual_training_years,
        "validation_year": validation_year,
        "selection_metric": "main-tour validation log loss",
        "locked_best_variant": locked_best_variant,
        "selected": selected,
    }

    evaluations.to_csv(destination / "fold_metrics.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(
        destination / "candidate_summary.csv", index=False
    )
    pd.concat(selected_prediction_frames, ignore_index=True).to_csv(
        destination / "validation_predictions.csv", index=False
    )
    pd.DataFrame(coefficient_rows).to_csv(
        destination / "training_coefficients.csv", index=False
    )
    pd.DataFrame(rank_rows).to_csv(destination / "rank_transforms.csv", index=False)
    _feature_coverage(features).to_csv(destination / "feature_coverage.csv", index=False)
    dedupe_audit.to_csv(destination / "deduplication_audit.csv", index=False)
    _write_json(destination / "selected_config.json", selection_payload)
    summary = {
        "tour": config.tour,
        "forecast_horizon": "round",
        "workload_source": "main-tour matches only",
        "exact_match_timestamps_available": False,
        "final_holdout_loaded": False,
        **selection_payload,
    }
    _write_json(destination / "summary.json", summary)
    lower_inputs = _require_lower_tier_inputs(
        config, range(config.history_start, config.test_year)
    )
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_WORKLOAD_VALIDATION,
            {
                "forecast_horizon": "round",
                "final_holdout_loaded": False,
                "l2_grid": list(L2_GRID),
                "workload_variants": {
                    key: None if value is None else list(value)
                    for key, value in WORKLOAD_VARIANTS.items()
                },
                "lower_tier_inputs": [
                    _digest_record(path, config.project_root) for path in lower_inputs
                ],
                "upstream_artifacts": [
                    _digest_record(
                        config.stage_directory("05_cross_tier_validation")
                        / "selected_config.json",
                        config.project_root,
                    ),
                    _digest_record(
                        config.stage_directory("05_cross_tier_validation")
                        / "candidate_summary.csv",
                        config.project_root,
                    ),
                ],
            },
        ),
    )
    return summary


def _read_workload_selection(config: ProjectConfig) -> dict[str, Any]:
    path = config.stage_directory(STAGE_WORKLOAD_VALIDATION) / "selected_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run workload-validation first; missing: {path}")
    payload = _read_json(path)
    if set(payload.get("selected", {})) != set(WORKLOAD_VARIANTS):
        raise ValueError("Workload selected_config has unexpected variants")
    return payload


def _paired_probability_gap(
    variant: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    reference_probabilities = reference[["match_key", "p_model"]].rename(
        columns={"p_model": "reference_probability"}
    )
    paired = variant.merge(
        reference_probabilities, on="match_key", how="inner", validate="one_to_one"
    )
    if len(paired) != len(variant) or len(paired) != len(reference):
        raise ValueError("Workload comparison cohorts differ")
    return paired_block_bootstrap_gap(
        paired,
        model_probability_column="p_model",
        market_probability_column="reference_probability",
        samples=samples,
        seed=seed,
    )


def run_workload_holdout_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Stage 08: refit locked residuals and evaluate the untouched final year."""

    destination = config.stage_directory(STAGE_WORKLOAD_HOLDOUT)
    _prepare_directory(destination, overwrite)
    selection = _read_workload_selection(config)
    matches, dedupe_audit = _load_holdout_cross_tier(config)
    market = _market_cohort(config)
    market_dates = market.set_index("match_key")["actual_match_date"]
    dated = matches["match_key"].isin(market_dates.index)
    matches.loc[dated, "actual_match_date"] = pd.to_datetime(
        matches.loc[dated, "match_key"].map(market_dates), errors="coerce"
    ).to_numpy()
    main = matches.loc[matches["source_group"].eq("main")].copy()
    features = build_workload_features(main)

    base_spec = selection["base_cross_tier_spec"]
    base_candidate = selection["base_model_candidate"]
    fit_years = [
        *selection["residual_training_years"],
        int(selection["validation_year"]),
    ]
    training_panels: list[pd.DataFrame] = []
    rank_rows: list[dict[str, float]] = []
    for year in fit_years:
        predictions, rank_row = _base_predictions_for_year(
            matches,
            main,
            year=int(year),
            base_spec=base_spec,
            base_candidate=base_candidate,
        )
        training_panels.append(_join_features(predictions, features))
        rank_rows.append(rank_row)
    residual_train = pd.concat(training_panels, ignore_index=True)

    rank_transform = _fit_rank_before(main, config.test_start, "round")
    factory: Callable[[], EloModel] = lambda: _model_from_spec(
        base_spec, base_candidate, rank_transform
    )
    holdout_predictions = run_model_factories(
        matches,
        {"base_model": factory},
        BacktestConfig(
            test_start=config.test_start,
            test_end=config.test_end,
            horizon="round",
            update_retirements=False,
            evaluate_retirements=False,
        ),
    )["base_model"]
    if not holdout_predictions["source_group"].eq("main").all():
        raise AssertionError("Workload holdout emitted a lower-tier target")
    holdout = _join_features(holdout_predictions, features).merge(
        market, on="match_key", how="inner", validate="one_to_one"
    )
    if holdout.empty:
        raise ValueError("No complete workload market cohort")

    variant_frames: dict[str, pd.DataFrame] = {}
    coefficient_rows: list[dict[str, Any]] = []
    for variant, spec in selection["selected"].items():
        feature_names = WORKLOAD_VARIANTS[variant]
        if feature_names is None:
            model = None
        else:
            model = fit_offset_logistic(
                residual_train,
                feature_names,
                l2=float(spec["l2"]),
            )
            coefficient_rows.extend(model.coefficient_rows(variant))
        variant_frames[variant] = _variant_frame(holdout, variant, model)

    baseline = variant_frames["base_model"]
    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    ordered_variants = sorted(variant_frames, key=lambda value: WORKLOAD_VARIANT_ORDER[value])
    for position, variant in enumerate(ordered_variants):
        frame = variant_frames[variant]
        previous_variant = ordered_variants[max(0, position - 1)]
        previous = variant_frames[previous_variant]
        for slice_name, sliced in _slices(frame):
            if sliced.empty:
                continue
            keys = set(sliced["match_key"])
            baseline_slice = baseline.loc[baseline["match_key"].isin(keys)].copy()
            previous_slice = previous.loc[previous["match_key"].isin(keys)].copy()
            metrics = evaluate_frame(sliced)
            market_gap = paired_block_bootstrap_gap(
                sliced, samples=bootstrap_samples, seed=42
            )
            base_gap = _paired_probability_gap(
                sliced, baseline_slice, samples=bootstrap_samples, seed=84
            )
            previous_gap = _paired_probability_gap(
                sliced, previous_slice, samples=bootstrap_samples, seed=126
            )
            metric_rows.append(
                {
                    "tour": config.tour,
                    "variant": variant,
                    "variant_order": WORKLOAD_VARIANT_ORDER[variant],
                    "previous_variant": previous_variant,
                    "slice": slice_name,
                    **metrics,
                    "market_gap_ci_low": market_gap["ci_low"],
                    "market_gap_ci_high": market_gap["ci_high"],
                    "delta_log_loss_vs_base": base_gap["mean"],
                    "delta_vs_base_ci_low": base_gap["ci_low"],
                    "delta_vs_base_ci_high": base_gap["ci_high"],
                    "delta_log_loss_vs_previous": previous_gap["mean"],
                    "delta_vs_previous_ci_low": previous_gap["ci_low"],
                    "delta_vs_previous_ci_high": previous_gap["ci_high"],
                }
            )
            delta_rows.append(
                {
                    "tour": config.tour,
                    "variant": variant,
                    "previous_variant": previous_variant,
                    "slice": slice_name,
                    "n": int(len(sliced)),
                    "delta_log_loss_vs_base": base_gap["mean"],
                    "base_ci_low": base_gap["ci_low"],
                    "base_ci_high": base_gap["ci_high"],
                    "delta_log_loss_vs_previous": previous_gap["mean"],
                    "previous_ci_low": previous_gap["ci_low"],
                    "previous_ci_high": previous_gap["ci_high"],
                }
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["slice", "variant_order"], kind="stable"
    )
    predictions = pd.concat(
        [variant_frames[variant] for variant in ordered_variants], ignore_index=True
    )
    predictions.to_csv(destination / "common_cohort_predictions.csv", index=False)
    metrics.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(destination / "paired_deltas.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(destination / "coefficients.csv", index=False)
    pd.DataFrame(
        [
            *rank_rows,
            {
                "year": config.test_year,
                "intercept": rank_transform.intercept,
                "log_rank_slope": rank_transform.log_rank_slope,
            },
        ]
    ).to_csv(destination / "rank_transforms.csv", index=False)
    _feature_coverage(features).to_csv(destination / "feature_coverage.csv", index=False)
    dedupe_audit.to_csv(destination / "deduplication_audit.csv", index=False)

    all_metrics = metrics.loc[metrics["slice"].eq("all")].to_dict("records")
    locked = metrics.loc[
        metrics["slice"].eq("all")
        & metrics["variant"].eq(selection["locked_best_variant"])
    ]
    summary = {
        "tour": config.tour,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "forecast_horizon": "round",
        "workload_source": "main-tour matches only",
        "date_resolution": "tournament start plus conservative round order",
        "lower_tier_target_year_loaded": False,
        "market_cohort": "complete linked Pinnacle pairs",
        "common_matches_per_variant": int(len(holdout)),
        "locked_best_variant_from_validation": selection["locked_best_variant"],
        "locked_best_holdout_metrics": locked.to_dict("records"),
        "all_metrics": all_metrics,
        "selected": selection["selected"],
    }
    _write_json(destination / "summary.json", summary)
    lower_inputs = _require_lower_tier_inputs(
        config, range(config.history_start, config.test_year)
    )
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_WORKLOAD_HOLDOUT,
            {
                "forecast_horizon": "round",
                "bootstrap_samples": bootstrap_samples,
                "lower_tier_policy": "strictly pre-holdout history only",
                "workload_policy": "main-tour prior events and earlier event rounds only",
                "lower_tier_inputs": [
                    _digest_record(path, config.project_root) for path in lower_inputs
                ],
                "upstream_artifacts": [
                    _digest_record(
                        config.stage_directory(STAGE_WORKLOAD_VALIDATION)
                        / "selected_config.json",
                        config.project_root,
                    ),
                    _digest_record(
                        config.stage_directory("00_linkage") / "linked_market.csv",
                        config.project_root,
                    ),
                ],
            },
        ),
    )
    return summary


__all__ = [
    "L2_GRID",
    "STAGE_WORKLOAD_HOLDOUT",
    "STAGE_WORKLOAD_VALIDATION",
    "WORKLOAD_VARIANTS",
    "OffsetLogisticModel",
    "build_workload_features",
    "fit_offset_logistic",
    "run_workload_holdout_stage",
    "run_workload_validation_stage",
    "score_load",
]
