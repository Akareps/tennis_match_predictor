"""Data loading and canonical match representation.

This module intentionally does not download data.  Downloading and snapshotting
are separate operations so an experiment can record exactly which inputs it
used.  The canonical representation orders players by stable player id; the
winner is stored only in ``actual_a_won`` and is never encoded by row order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROUND_ORDER = {
    "Q1": -3,
    "Q2": -2,
    "Q3": -1,
    "Q4": 0,
    "RR": 1,
    "R128": 2,
    "R64": 3,
    "R32": 4,
    "R16": 5,
    "QF": 6,
    "SF": 7,
    "F": 8,
    "BR": 8,
}

NON_COMPLETION_TOKENS = ("W/O", " WO", "RET", "DEF", "ABN")


def _candidate_files(data_dir: Path, tour: str, year: int, include_lower_tiers: bool) -> list[Path]:
    files = [data_dir / f"{tour}_matches_{year}.csv"]
    if include_lower_tiers:
        if tour == "atp":
            files.extend(
                [
                    data_dir / f"atp_matches_qual_chall_{year}.csv",
                    data_dir / f"atp_matches_futures_{year}.csv",
                ]
            )
        else:
            files.append(data_dir / f"wta_matches_qual_itf_{year}.csv")
    return files


def _source_groups(frame: pd.DataFrame, tour: str, filename: str) -> pd.Series:
    """Classify rows into cumulative cross-tier experiment groups.

    Sackmann's ATP qualifying/Challenger file and WTA qualifying/ITF file each
    combine more than one circuit.  The event level disambiguates those rows:
    ``C`` is Challenger/WTA125, numeric levels are Futures/ITF, and the other
    rows in the mixed files are tour qualifying.
    """

    groups = pd.Series("main", index=frame.index, dtype="object")
    levels = frame.get("tourney_level", pd.Series("", index=frame.index)).astype(str).str.strip()
    if tour == "atp" and "_qual_chall_" in filename:
        groups = pd.Series(np.where(levels.eq("C"), "challenger", "qualifying"), index=frame.index)
    elif tour == "atp" and "_futures_" in filename:
        groups = pd.Series("developmental", index=frame.index, dtype="object")
    elif tour == "wta" and "_qual_itf_" in filename:
        numeric_level = pd.to_numeric(levels, errors="coerce").notna()
        groups = pd.Series(
            np.where(numeric_level, "developmental", np.where(levels.eq("C"), "challenger", "qualifying")),
            index=frame.index,
        )
    return groups.astype(str)


def load_cached_matches(
    data_dir: str | Path,
    tour: str,
    years: Iterable[int],
    *,
    include_lower_tiers: bool = False,
    require_all: bool = True,
) -> pd.DataFrame:
    """Load a versionable local snapshot of Sackmann-format CSV files.

    ``source_file`` and ``source_tier`` are retained for ablation reporting.
    When lower-tier files are requested, absent optional files are tolerated;
    absent main-tour files are not.
    """

    tour = tour.lower()
    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")
    root = Path(data_dir)
    frames: list[pd.DataFrame] = []
    missing_main: list[Path] = []
    for year in years:
        candidates = _candidate_files(root, tour, int(year), include_lower_tiers)
        for position, path in enumerate(candidates):
            if not path.exists():
                if position == 0:
                    missing_main.append(path)
                continue
            frame = pd.read_csv(path, low_memory=False)
            frame["source_file"] = path.name
            frame["source_tier"] = "main" if position == 0 else "lower"
            frame["source_group"] = _source_groups(frame, tour, path.name)
            frame["source_file_row"] = np.arange(len(frame), dtype=int)
            frame["source_year"] = int(year)
            frames.append(frame)
    if missing_main and require_all:
        rendered = ", ".join(str(path) for path in missing_main)
        raise FileNotFoundError(f"Missing required match snapshots: {rendered}")
    if not frames:
        raise FileNotFoundError(f"No {tour.upper()} match snapshots found in {root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _parse_tournament_date(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("Int64")
    return pd.to_datetime(numeric.astype("string"), format="%Y%m%d", errors="coerce")


def _column(frame: pd.DataFrame, name: str, default: object = np.nan) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _value_for_a(frame: pd.DataFrame, winner_col: str, loser_col: str) -> pd.Series:
    winner_values = _column(frame, winner_col)
    loser_values = _column(frame, loser_col)
    return winner_values.where(frame["actual_a_won"].eq(1), loser_values)


def _value_for_b(frame: pd.DataFrame, winner_col: str, loser_col: str) -> pd.Series:
    winner_values = _column(frame, winner_col)
    loser_values = _column(frame, loser_col)
    return loser_values.where(frame["actual_a_won"].eq(1), winner_values)


def canonicalize_matches(raw: pd.DataFrame, tour: str) -> pd.DataFrame:
    """Return one row per match with outcome-neutral player ordering.

    The returned ``tournament_date`` remains an event-start date.  It must not
    be treated as an exact match timestamp; the backtest module uses explicit
    draw/round batches unless linkage supplies ``actual_match_date``.
    """

    required = {
        "tourney_date",
        "tourney_name",
        "surface",
        "tourney_level",
        "winner_id",
        "loser_id",
        "winner_name",
        "loser_name",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing Sackmann columns: {', '.join(missing)}")

    frame = raw.copy()
    frame["source_row"] = np.arange(len(frame), dtype=int)
    frame["tournament_date"] = _parse_tournament_date(frame["tourney_date"])
    frame["winner_id"] = pd.to_numeric(frame["winner_id"], errors="coerce").astype("Int64")
    frame["loser_id"] = pd.to_numeric(frame["loser_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["tournament_date", "winner_id", "loser_id"]).copy()

    winner_is_a = frame["winner_id"] < frame["loser_id"]
    frame["player_a_id"] = frame["winner_id"].where(winner_is_a, frame["loser_id"]).astype("int64")
    frame["player_b_id"] = frame["loser_id"].where(winner_is_a, frame["winner_id"]).astype("int64")
    frame["actual_a_won"] = winner_is_a.astype("int8")
    frame["player_a_name"] = _value_for_a(frame, "winner_name", "loser_name")
    frame["player_b_name"] = _value_for_b(frame, "winner_name", "loser_name")
    frame["a_rank"] = pd.to_numeric(_value_for_a(frame, "winner_rank", "loser_rank"), errors="coerce")
    frame["b_rank"] = pd.to_numeric(_value_for_b(frame, "winner_rank", "loser_rank"), errors="coerce")
    frame["a_rank_points"] = pd.to_numeric(
        _value_for_a(frame, "winner_rank_points", "loser_rank_points"), errors="coerce"
    )
    frame["b_rank_points"] = pd.to_numeric(
        _value_for_b(frame, "winner_rank_points", "loser_rank_points"), errors="coerce"
    )
    stat_columns = {
        "aces": ("w_ace", "l_ace"),
        "double_faults": ("w_df", "l_df"),
        "service_points": ("w_svpt", "l_svpt"),
        "first_serves_in": ("w_1stIn", "l_1stIn"),
        "first_serve_points_won": ("w_1stWon", "l_1stWon"),
        "second_serve_points_won": ("w_2ndWon", "l_2ndWon"),
        "service_games": ("w_SvGms", "l_SvGms"),
        "break_points_saved": ("w_bpSaved", "l_bpSaved"),
        "break_points_faced": ("w_bpFaced", "l_bpFaced"),
    }
    for canonical_name, (winner_column, loser_column) in stat_columns.items():
        frame[f"a_{canonical_name}"] = pd.to_numeric(
            _value_for_a(frame, winner_column, loser_column), errors="coerce"
        )
        frame[f"b_{canonical_name}"] = pd.to_numeric(
            _value_for_b(frame, winner_column, loser_column), errors="coerce"
        )

    frame["tour"] = tour.lower()
    frame["round"] = frame.get("round", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    frame["round_order"] = frame["round"].map(ROUND_ORDER).fillna(0).astype(int)
    frame["match_num"] = pd.to_numeric(_column(frame, "match_num"), errors="coerce")
    frame["draw_size"] = pd.to_numeric(_column(frame, "draw_size"), errors="coerce")
    frame["surface"] = frame["surface"].fillna("Unknown").astype(str).str.strip()
    frame["tourney_level"] = frame["tourney_level"].fillna("?").astype(str).str.strip()
    frame["best_of"] = pd.to_numeric(_column(frame, "best_of", 3), errors="coerce").fillna(3).astype(int)
    frame["minutes"] = pd.to_numeric(_column(frame, "minutes"), errors="coerce")
    frame["score"] = _column(frame, "score", "").fillna("").astype(str)
    upper_score = " " + frame["score"].str.upper()
    frame["is_retirement"] = upper_score.apply(lambda score: any(token in score for token in NON_COMPLETION_TOKENS))
    frame["has_point_stats"] = (
        pd.to_numeric(_column(frame, "w_svpt"), errors="coerce").gt(0)
        & pd.to_numeric(_column(frame, "l_svpt"), errors="coerce").gt(0)
    )

    if "tourney_id" in frame:
        tournament_id = frame["tourney_id"].fillna("").astype(str)
    else:
        tournament_id = frame["tourney_name"].fillna("").astype(str)
    frame["event_key"] = frame["tour"] + "|" + tournament_id
    match_number_token = frame["match_num"].map(
        lambda value: str(int(value)) if pd.notna(value) and float(value).is_integer() else str(value)
    )
    match_number_token = match_number_token.where(frame["match_num"].notna(), "row" + frame["source_row"].astype(str))
    frame["match_key"] = (
        frame["event_key"]
        + "|"
        + frame["round"]
        + "|"
        + frame["player_a_id"].astype(str)
        + "|"
        + frame["player_b_id"].astype(str)
        + "|"
        + match_number_token
    )
    frame["actual_match_date"] = pd.NaT

    canonical_columns = [
        "source_row",
        "source_file_row",
        "source_file",
        "source_tier",
        "source_group",
        "tour",
        "event_key",
        "match_key",
        "tournament_date",
        "actual_match_date",
        "tourney_name",
        "tourney_level",
        "surface",
        "round",
        "round_order",
        "match_num",
        "draw_size",
        "best_of",
        "minutes",
        "player_a_id",
        "player_b_id",
        "player_a_name",
        "player_b_name",
        "a_rank",
        "b_rank",
        "a_rank_points",
        "b_rank_points",
        "a_aces",
        "b_aces",
        "a_double_faults",
        "b_double_faults",
        "a_service_points",
        "b_service_points",
        "a_first_serves_in",
        "b_first_serves_in",
        "a_first_serve_points_won",
        "b_first_serve_points_won",
        "a_second_serve_points_won",
        "b_second_serve_points_won",
        "a_service_games",
        "b_service_games",
        "a_break_points_saved",
        "b_break_points_saved",
        "a_break_points_faced",
        "b_break_points_faced",
        "actual_a_won",
        "score",
        "is_retirement",
        "has_point_stats",
    ]
    defaults = {
        "source_file_row": frame["source_row"],
        "source_file": "unknown",
        "source_tier": "unknown",
        "source_group": "unknown",
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    return frame[canonical_columns].reset_index(drop=True)


def deduplicate_canonical_matches(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove repeated source records while retaining a row-level audit.

    ``match_num`` is deliberately excluded.  The source occasionally assigns
    two match numbers to the same event/round/player/score record.  Including
    normalized score avoids collapsing legitimate rematches between the same
    players at team or round-robin events.
    """

    required = {
        "tour",
        "event_key",
        "round",
        "player_a_id",
        "player_b_id",
        "score",
        "source_row",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Missing deduplication columns: {', '.join(missing)}")

    frame = matches.copy()
    normalized_score = (
        frame["score"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    identity_columns = ["tour", "event_key", "round", "player_a_id", "player_b_id"]
    identity = frame[identity_columns].astype(str).agg("|".join, axis=1) + "|" + normalized_score
    frame["_dedupe_identity"] = identity
    self_match_mask = frame["player_a_id"].eq(frame["player_b_id"])
    duplicate_mask = frame.duplicated("_dedupe_identity", keep="first") & ~self_match_mask

    first_rows = frame.loc[~duplicate_mask, ["_dedupe_identity", "source_row", "match_key"]].rename(
        columns={"source_row": "retained_source_row", "match_key": "retained_match_key"}
    )
    audit = frame.loc[duplicate_mask].merge(
        first_rows,
        on="_dedupe_identity",
        how="left",
        validate="many_to_one",
        suffixes=("", "_retained"),
    )
    if not audit.empty:
        audit = audit.rename(
            columns={"source_row": "removed_source_row", "match_key": "removed_match_key"}
        )
        audit["reason_code"] = "duplicate_event_round_players_score"
        audit = audit[
            [
                "reason_code",
                "_dedupe_identity",
                "source_file",
                "source_file_row",
                "removed_source_row",
                "removed_match_key",
                "retained_source_row",
                "retained_match_key",
            ]
        ].rename(columns={"_dedupe_identity": "dedupe_identity"})
    audit_columns = [
        "reason_code",
        "dedupe_identity",
        "source_file",
        "source_file_row",
        "removed_source_row",
        "removed_match_key",
        "retained_source_row",
        "retained_match_key",
    ]
    if audit.empty:
        audit = pd.DataFrame(columns=audit_columns)

    self_audit = frame.loc[self_match_mask].copy()
    if not self_audit.empty:
        self_audit = self_audit.assign(
            reason_code="invalid_self_match",
            dedupe_identity=self_audit["_dedupe_identity"],
            removed_source_row=self_audit["source_row"],
            removed_match_key=self_audit["match_key"],
            retained_source_row=np.nan,
            retained_match_key=None,
        )[audit_columns]
        audit = pd.concat([audit, self_audit], ignore_index=True)

    removal_mask = duplicate_mask | self_match_mask
    cleaned = frame.loc[~removal_mask].drop(columns="_dedupe_identity").reset_index(drop=True)
    return cleaned, audit.reset_index(drop=True)
