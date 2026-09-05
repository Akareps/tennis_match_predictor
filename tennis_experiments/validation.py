"""Temporal folds and deterministic model-selection helpers.

The final test period is a hard boundary: rows on or after ``test_start`` are
never returned by the fold builder.  This keeps tuning separate from the
single, untouched benchmark reported by an experiment stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    """One expanding-window split, expressed as half-open date intervals."""

    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_indices: tuple[Any, ...]
    validation_indices: tuple[Any, ...]


def expanding_year_folds(
    frame: pd.DataFrame,
    *,
    test_start: str | pd.Timestamp,
    date_column: str = "tournament_date",
    minimum_training_years: int = 1,
    validation_years: int = 1,
    step_years: int = 1,
) -> list[TemporalFold]:
    """Build annual expanding folds entirely before an untouched test period.

    The first training interval begins on January 1 of the earliest observed
    year.  Each validation interval is half-open and every earlier row is part
    of its training set.  Partial final validation windows are omitted.
    """

    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    if minimum_training_years < 1 or validation_years < 1 or step_years < 1:
        raise ValueError("year counts must all be positive")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_column} contains {int(dates.isna().sum())} invalid dates")
    if frame.empty:
        return []

    boundary = pd.Timestamp(test_start).normalize()
    earliest = pd.Timestamp(year=int(dates.min().year), month=1, day=1)
    validation_start = earliest + pd.DateOffset(years=minimum_training_years)
    folds: list[TemporalFold] = []

    while True:
        validation_end = validation_start + pd.DateOffset(years=validation_years)
        if validation_end > boundary:
            break
        train_mask = dates.lt(validation_start)
        validation_mask = dates.ge(validation_start) & dates.lt(validation_end)
        if train_mask.any() and validation_mask.any():
            folds.append(
                TemporalFold(
                    fold_id=f"{validation_start:%Y%m%d}_{validation_end:%Y%m%d}",
                    train_start=earliest,
                    train_end=validation_start,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    train_indices=tuple(frame.index[train_mask]),
                    validation_indices=tuple(frame.index[validation_mask]),
                )
            )
        validation_start += pd.DateOffset(years=step_years)
    return folds


def select_by_validation_loss(
    evaluations: pd.DataFrame,
    *,
    candidate_column: str = "candidate",
    loss_column: str = "log_loss",
    weight_column: str = "n",
) -> tuple[Any, pd.DataFrame]:
    """Select the lowest weighted validation loss with stable tie-breaking.

    Returns the original candidate value and a one-row-per-candidate summary.
    Candidate ordering is based on its string representation only when losses
    tie, so selection is deterministic regardless of input row order.
    """

    required = {candidate_column, loss_column, weight_column}
    missing = sorted(required - set(evaluations.columns))
    if missing:
        raise ValueError(f"Missing evaluation columns: {', '.join(missing)}")
    if evaluations.empty:
        raise ValueError("No validation evaluations supplied")

    work = evaluations[[candidate_column, loss_column, weight_column]].copy()
    work[loss_column] = pd.to_numeric(work[loss_column], errors="coerce")
    work[weight_column] = pd.to_numeric(work[weight_column], errors="coerce")
    invalid = (
        ~np.isfinite(work[loss_column])
        | ~np.isfinite(work[weight_column])
        | work[weight_column].le(0)
    )
    if invalid.any():
        raise ValueError("Validation losses must be finite and weights must be positive")
    work["weighted_loss"] = work[loss_column] * work[weight_column]

    rows: list[dict[str, Any]] = []
    for candidate, group in work.groupby(candidate_column, sort=False, dropna=False):
        total_weight = float(group[weight_column].sum())
        rows.append(
            {
                candidate_column: candidate,
                "validation_log_loss": float(group["weighted_loss"].sum() / total_weight),
                "validation_n": int(total_weight),
                "folds": int(len(group)),
                "_tie_key": repr(candidate),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "_tie_key"], kind="stable"
    ).reset_index(drop=True)
    selected = summary.iloc[0][candidate_column]
    return selected, summary.drop(columns="_tie_key")


__all__ = ["TemporalFold", "expanding_year_folds", "select_by_validation_loss"]
