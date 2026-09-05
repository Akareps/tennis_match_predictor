"""Leakage-aware chronology and walk-forward execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .models import EloModel


ForecastHorizon = Literal["draw", "round", "match_date"]


@dataclass(frozen=True)
class BacktestConfig:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    horizon: ForecastHorizon = "round"
    update_retirements: bool = False
    evaluate_retirements: bool = False

    def __post_init__(self) -> None:
        if pd.Timestamp(self.test_end) <= pd.Timestamp(self.test_start):
            raise ValueError("test_end must be after test_start")
        if self.horizon not in {"draw", "round", "match_date"}:
            raise ValueError(f"Unsupported horizon: {self.horizon}")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.replace({np.nan: None}).to_dict("records")


def iter_chronology_batches(
    matches: pd.DataFrame,
    horizon: ForecastHorizon,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield deterministic batches whose outcomes are simultaneous to the model.

    ``draw`` freezes ratings for every event sharing the same published start
    date. ``round`` allows earlier rounds to inform later rounds but predicts a
    whole same-date round across all events before updating. Synchronizing
    events is conservative: without exact timestamps, an alphabetically earlier
    event must not reveal outcomes to another event that started on the same
    day. ``match_date`` requires an actual linked match date and predicts all
    matches on that date before updating.
    """

    required = {"event_key", "tournament_date", "round_order", "source_row"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Missing chronology columns: {', '.join(missing)}")

    frame = matches.copy()
    frame["tournament_date"] = pd.to_datetime(frame["tournament_date"])
    if horizon == "match_date":
        if "actual_match_date" not in frame:
            raise ValueError("match_date horizon requires actual_match_date")
        frame["actual_match_date"] = pd.to_datetime(frame["actual_match_date"], errors="coerce")
        if frame["actual_match_date"].isna().any():
            missing_count = int(frame["actual_match_date"].isna().sum())
            raise ValueError(f"match_date horizon has {missing_count} rows without an actual date")
        keys = ["actual_match_date"]
        sort_columns = ["actual_match_date", "event_key", "round_order", "source_row"]
    elif horizon == "draw":
        keys = ["tournament_date"]
        sort_columns = ["tournament_date", "event_key", "round_order", "source_row"]
    else:
        keys = ["tournament_date", "round_order"]
        sort_columns = ["tournament_date", "round_order", "event_key", "source_row"]

    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    grouper: str | list[str] = keys[0] if len(keys) == 1 else keys
    for key, batch in frame.groupby(grouper, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rendered = "|".join(str(value) for value in key)
        yield rendered, batch.copy()


def _evaluation_date(row: Mapping[str, Any]) -> pd.Timestamp:
    actual = row.get("actual_match_date")
    if actual is not None and not pd.isna(actual):
        return pd.Timestamp(actual)
    return pd.Timestamp(row["tournament_date"])


def _prediction_record(
    row: Mapping[str, Any],
    model: EloModel,
    config: BacktestConfig,
    batch_id: str,
    model_name: str,
) -> dict[str, Any]:
    evaluation_date = _evaluation_date(row)
    details = model.predict_details(
        int(row["player_a_id"]),
        int(row["player_b_id"]),
        str(row.get("surface", "Unknown")),
        a_rank=row.get("a_rank"),
        b_rank=row.get("b_rank"),
    )
    output = {
        "model": model_name,
        "chronology_horizon": config.horizon,
        "chronology_batch": batch_id,
        "prediction_date": evaluation_date,
        "match_key": row.get("match_key"),
        "source_row": row.get("source_row"),
        "source_file": row.get("source_file"),
        "source_tier": row.get("source_tier"),
        "source_group": row.get("source_group"),
        "event_key": row.get("event_key"),
        "tournament_date": row.get("tournament_date"),
        "actual_match_date": row.get("actual_match_date"),
        "tourney_name": row.get("tourney_name"),
        "tourney_level": row.get("tourney_level"),
        "surface": row.get("surface"),
        "round": row.get("round"),
        "player_a_id": int(row["player_a_id"]),
        "player_b_id": int(row["player_b_id"]),
        "a_rank": row.get("a_rank"),
        "b_rank": row.get("b_rank"),
        "actual_a_won": int(row["actual_a_won"]),
        "is_retirement": bool(row.get("is_retirement", False)),
        "is_cold_start": (
            details["a_main_observations"] < 30 or details["b_main_observations"] < 30
        ),
    }
    output.update(details)
    return output


def run_walk_forward(
    matches: pd.DataFrame,
    model: EloModel,
    config: BacktestConfig,
    *,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Predict every batch before revealing any outcome from that batch."""

    predictions: list[dict[str, Any]] = []
    for batch_id, batch_frame in iter_chronology_batches(matches, config.horizon):
        batch_records = _records(batch_frame)
        for row in batch_records:
            evaluation_date = _evaluation_date(row)
            in_test = pd.Timestamp(config.test_start) <= evaluation_date < pd.Timestamp(config.test_end)
            is_retirement = bool(row.get("is_retirement", False))
            if not in_test or (is_retirement and not config.evaluate_retirements):
                continue
            predictions.append(
                _prediction_record(row, model, config, batch_id, model_name or model.name)
            )

        updates = [
            row
            for row in batch_records
            if config.update_retirements or not bool(row.get("is_retirement", False))
        ]
        model.batch_update(updates)
    return pd.DataFrame(predictions)


def run_model_factories(
    matches: pd.DataFrame,
    factories: Mapping[str, Callable[[], EloModel]],
    config: BacktestConfig,
) -> dict[str, pd.DataFrame]:
    """Run models from identical empty state while sharing chronology work."""

    models = {name: factory() for name, factory in factories.items()}
    predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    for batch_id, batch_frame in iter_chronology_batches(matches, config.horizon):
        batch_records = _records(batch_frame)
        eligible = []
        for row in batch_records:
            evaluation_date = _evaluation_date(row)
            in_test = pd.Timestamp(config.test_start) <= evaluation_date < pd.Timestamp(config.test_end)
            is_retirement = bool(row.get("is_retirement", False))
            if in_test and (config.evaluate_retirements or not is_retirement):
                eligible.append(row)
        for name, model in models.items():
            predictions[name].extend(
                _prediction_record(row, model, config, batch_id, name) for row in eligible
            )
        updates = [
            row
            for row in batch_records
            if config.update_retirements or not bool(row.get("is_retirement", False))
        ]
        for model in models.values():
            model.batch_update(updates)
    return {name: pd.DataFrame(rows) for name, rows in predictions.items()}
