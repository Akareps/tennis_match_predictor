"""Validation and descriptive summaries for timestamped market snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .linkage import normalize_player_name


ODDS_GAP_REQUIRED_COLUMNS = {
    "snapshot_ts",
    "sport",
    "away",
    "home",
    "commence_time",
    "market",
    "side",
    "book",
    "american_odds",
    "devig_fair_prob",
    "best_price_flag",
}


def american_to_decimal(values: pd.Series) -> pd.Series:
    """Convert finite non-zero American prices to decimal odds."""

    odds = pd.to_numeric(values, errors="coerce")
    positive = odds.gt(0)
    result = pd.Series(np.nan, index=odds.index, dtype=float)
    result.loc[positive] = 1.0 + odds.loc[positive] / 100.0
    negative = odds.lt(0)
    result.loc[negative] = 1.0 + 100.0 / odds.loc[negative].abs()
    return result


def _player_key(values: pd.Series) -> pd.Series:
    return values.map(lambda value: normalize_player_name(value, source="sackmann"))


def audit_odds_gap_snapshot(
    source: str | Path | pd.DataFrame,
    *,
    tour: str | None = None,
    horizons_hours: Iterable[float] = (24.0, 6.0, 1.0),
    tolerance_minutes: float = 90.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return match summaries, horizon availability, and a strict input audit.

    This function intentionally does not attach outcomes.  Its role is to
    validate that a raw snapshot can support fixed pre-match horizons and to
    preserve schedule revisions rather than treating scheduled start time as a
    stable match identifier.
    """

    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        raw = pd.read_csv(Path(source), low_memory=False)
    missing = sorted(ODDS_GAP_REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"Missing Odds Gap columns: {', '.join(missing)}")
    if tour not in {None, "atp", "wta"}:
        raise ValueError("tour must be None, 'atp', or 'wta'")

    frame = raw.copy()
    if tour is not None:
        frame = frame.loc[frame["sport"].astype(str).str.startswith(f"tennis_{tour}_")].copy()
    if frame.empty:
        raise ValueError("No rows remain after the tour filter")

    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
    frame["commence_time"] = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce")
    frame["american_odds"] = pd.to_numeric(frame["american_odds"], errors="coerce")
    frame["devig_fair_prob"] = pd.to_numeric(frame["devig_fair_prob"], errors="coerce")
    invalid_time = frame[["snapshot_ts", "commence_time"]].isna().any(axis=1)
    invalid_price = (
        frame["american_odds"].isna()
        | frame["american_odds"].eq(0)
        | ~frame["devig_fair_prob"].between(0.0, 1.0, inclusive="both")
    )
    invalid_side = ~frame["side"].isin(["home", "away"])
    if invalid_time.any() or invalid_price.any() or invalid_side.any():
        raise ValueError(
            "Invalid Odds Gap rows: "
            f"time={int(invalid_time.sum())}, price={int(invalid_price.sum())}, "
            f"side={int(invalid_side.sum())}"
        )

    frame["away_key"] = _player_key(frame["away"])
    frame["home_key"] = _player_key(frame["home"])
    invalid_name = frame["away_key"].eq("") | frame["home_key"].eq("") | frame["away_key"].eq(frame["home_key"])
    if invalid_name.any():
        raise ValueError(f"Invalid player identities in {int(invalid_name.sum())} rows")
    frame["player_a_key"] = frame[["away_key", "home_key"]].min(axis=1)
    frame["player_b_key"] = frame[["away_key", "home_key"]].max(axis=1)
    frame["market_match_key"] = (
        frame["sport"].astype(str)
        + "|"
        + frame["player_a_key"]
        + "|"
        + frame["player_b_key"]
    )
    frame["offered_player_key"] = np.where(
        frame["side"].eq("home"), frame["home_key"], frame["away_key"]
    )
    frame["decimal_odds"] = american_to_decimal(frame["american_odds"])
    frame["is_pregame"] = frame["snapshot_ts"].lt(frame["commence_time"])
    pregame = frame.loc[frame["is_pregame"]].copy()
    if pregame.empty:
        raise ValueError("The snapshot contains no pre-match rows")

    schedule_consistency = pregame.groupby(
        ["market_match_key", "snapshot_ts"], sort=False
    )["commence_time"].nunique()
    inconsistent_schedule_snapshots = int(schedule_consistency.gt(1).sum())

    snapshots = (
        pregame.groupby(["market_match_key", "snapshot_ts"], as_index=False, sort=True)
        .agg(
            sport=("sport", "first"),
            player_a_key=("player_a_key", "first"),
            player_b_key=("player_b_key", "first"),
            commence_time=("commence_time", "max"),
            books=("book", "nunique"),
            rows=("book", "size"),
        )
    )
    probability = (
        pregame.loc[pregame["offered_player_key"].eq(pregame["player_a_key"])]
        .groupby(["market_match_key", "snapshot_ts"], sort=True)["devig_fair_prob"]
        .median()
        .rename("consensus_prob_a")
        .reset_index()
    )
    snapshots = snapshots.merge(
        probability,
        on=["market_match_key", "snapshot_ts"],
        how="left",
        validate="one_to_one",
    )
    snapshots["lead_minutes"] = (
        snapshots["commence_time"] - snapshots["snapshot_ts"]
    ).dt.total_seconds() / 60.0

    ordered = snapshots.sort_values(["market_match_key", "snapshot_ts"], kind="stable")
    first = ordered.groupby("market_match_key", sort=True).first()
    last = ordered.groupby("market_match_key", sort=True).last()
    aggregate = ordered.groupby("market_match_key", sort=True).agg(
        sport=("sport", "first"),
        player_a_key=("player_a_key", "first"),
        player_b_key=("player_b_key", "first"),
        snapshot_count=("snapshot_ts", "nunique"),
        schedule_versions=("commence_time", "nunique"),
        book_count_max=("books", "max"),
    )
    match_summary = aggregate.reset_index()
    match_summary["first_snapshot_ts"] = match_summary["market_match_key"].map(first["snapshot_ts"])
    match_summary["last_snapshot_ts"] = match_summary["market_match_key"].map(last["snapshot_ts"])
    match_summary["latest_commence_time"] = match_summary["market_match_key"].map(last["commence_time"])
    match_summary["first_lead_minutes"] = match_summary["market_match_key"].map(first["lead_minutes"])
    match_summary["last_lead_minutes"] = match_summary["market_match_key"].map(last["lead_minutes"])
    match_summary["opening_consensus_prob_a"] = match_summary["market_match_key"].map(
        first["consensus_prob_a"]
    )
    match_summary["latest_consensus_prob_a"] = match_summary["market_match_key"].map(
        last["consensus_prob_a"]
    )
    match_summary["probability_change_a"] = (
        match_summary["latest_consensus_prob_a"] - match_summary["opening_consensus_prob_a"]
    )
    match_summary["absolute_probability_change"] = match_summary["probability_change_a"].abs()

    horizon_rows: list[dict[str, Any]] = []
    tolerance = float(tolerance_minutes)
    if tolerance < 0:
        raise ValueError("tolerance_minutes must be non-negative")
    for hours in horizons_hours:
        target_minutes = float(hours) * 60.0
        if target_minutes < 0:
            raise ValueError("horizons_hours must be non-negative")
        candidates = snapshots.loc[
            snapshots["lead_minutes"].between(
                target_minutes, target_minutes + tolerance, inclusive="both"
            )
        ].copy()
        if not candidates.empty:
            candidates["distance_from_horizon"] = candidates["lead_minutes"] - target_minutes
            candidates = candidates.sort_values(
                ["market_match_key", "distance_from_horizon", "snapshot_ts"], kind="stable"
            ).drop_duplicates("market_match_key", keep="first")
        chosen = candidates.set_index("market_match_key") if not candidates.empty else pd.DataFrame()
        for key in match_summary["market_match_key"]:
            available = not chosen.empty and key in chosen.index
            row: dict[str, Any] = {
                "market_match_key": key,
                "horizon_hours": float(hours),
                "available": bool(available),
                "tolerance_minutes": tolerance,
                "snapshot_ts": pd.NaT,
                "contemporaneous_commence_time": pd.NaT,
                "lead_minutes": np.nan,
                "consensus_prob_a": np.nan,
            }
            if available:
                selected = chosen.loc[key]
                row.update(
                    {
                        "snapshot_ts": selected["snapshot_ts"],
                        "contemporaneous_commence_time": selected["commence_time"],
                        "lead_minutes": float(selected["lead_minutes"]),
                        "consensus_prob_a": float(selected["consensus_prob_a"]),
                    }
                )
            horizon_rows.append(row)
    horizon_frame = pd.DataFrame(horizon_rows)

    cadence = ordered.groupby("market_match_key")["snapshot_ts"].apply(
        lambda values: values.sort_values().diff().dt.total_seconds().div(60).dropna().median()
    )
    audit = {
        "rows": int(len(frame)),
        "pregame_rows": int(len(pregame)),
        "post_start_rows": int((~frame["is_pregame"]).sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "matches": int(match_summary["market_match_key"].nunique()),
        "snapshot_timestamps": int(frame["snapshot_ts"].nunique()),
        "snapshot_min": frame["snapshot_ts"].min(),
        "snapshot_max": frame["snapshot_ts"].max(),
        "books": int(frame["book"].nunique()),
        "sports": frame["sport"].value_counts().sort_index().to_dict(),
        "matches_with_schedule_revisions": int(match_summary["schedule_versions"].gt(1).sum()),
        "inconsistent_schedule_snapshots": inconsistent_schedule_snapshots,
        "median_snapshots_per_match": float(match_summary["snapshot_count"].median()),
        "median_scan_cadence_minutes": float(cadence.dropna().median()),
        "horizon_coverage": {
            str(float(hours)): int(
                horizon_frame.loc[horizon_frame["horizon_hours"].eq(float(hours)), "available"].sum()
            )
            for hours in horizons_hours
        },
        "contains_outcomes": False,
        "performance_claim_allowed": False,
    }
    return match_summary, horizon_frame, audit


__all__ = ["audit_odds_gap_snapshot", "american_to_decimal", "ODDS_GAP_REQUIRED_COLUMNS"]
