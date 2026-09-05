"""Prepare linked bookmaker odds for outcome-neutral evaluation.

Link selection belongs in :mod:`tennis_experiments.linkage`.  This module is
deliberately downstream of that decision: it verifies the selected player pair,
maps winner/loser-labelled prices onto canonical player A/player B columns, and
removes the bookmaker margin.  Prices and outcomes are therefore never inputs
to candidate selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .linkage import LinkageResult, _name_aliases


_CANONICAL_REQUIRED = {
    "source_row",
    "player_a_name",
    "player_b_name",
    "actual_a_won",
    "actual_match_date",
}

_LINK_DIAGNOSTIC_COLUMNS = (
    "reason_code",
    "reason",
    "pair_candidate_count",
    "date_candidate_count",
    "eligible_candidate_count",
    "best_score",
    "selected_score",
    "score_margin_to_next_candidate",
    "matched_odds_index",
    "matched_odds_position",
)


def _require_unique_key(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame:
        raise ValueError(f"{label} is missing required column {column!r}")
    if frame[column].isna().any():
        raise ValueError(f"{label}.{column} contains missing values")
    duplicates = frame.loc[frame[column].duplicated(keep=False), column].unique().tolist()
    if duplicates:
        preview = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"{label}.{column} must be unique; duplicate values: {preview}")


def _valid_decimal_odd(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 1.0:
        return None
    return numeric


def _complete_pair(row: pd.Series, winner_column: str, loser_column: str) -> tuple[float, float] | None:
    if winner_column not in row.index or loser_column not in row.index:
        return None
    winner = _valid_decimal_odd(row[winner_column])
    loser = _valid_decimal_odd(row[loser_column])
    if winner is None or loser is None:
        return None
    return winner, loser


def _orientation(row: pd.Series) -> bool:
    """Return whether the odds-file winner is canonical player A."""

    a_aliases = _name_aliases(row["player_a_name"], "sackmann")
    b_aliases = _name_aliases(row["player_b_name"], "sackmann")
    winner_aliases = _name_aliases(row["odds_winner"], "odds")
    loser_aliases = _name_aliases(row["odds_loser"], "odds")

    winner_is_a = bool(a_aliases & winner_aliases) and bool(b_aliases & loser_aliases)
    winner_is_b = bool(b_aliases & winner_aliases) and bool(a_aliases & loser_aliases)
    if winner_is_a == winner_is_b:
        raise ValueError(
            "linked odds player pair does not align unambiguously with canonical players"
        )
    return winner_is_a


def _diagnostics_with_source_rows(
    matches: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Map linker diagnostics back to the canonical stable source key."""

    if diagnostics.empty:
        return pd.DataFrame(columns=["source_row"])
    diagnostic = diagnostics.copy()
    if "sackmann_source_row" in diagnostic:
        diagnostic["source_row"] = diagnostic["sackmann_source_row"]
    else:
        if "sackmann_position" not in diagnostic:
            raise ValueError(
                "linkage diagnostics need sackmann_source_row or sackmann_position"
            )
        positions = pd.to_numeric(diagnostic["sackmann_position"], errors="coerce")
        if positions.isna().any() or not positions.eq(positions.astype(int)).all():
            raise ValueError("linkage diagnostics contain invalid sackmann_position values")
        integer_positions = positions.astype(int)
        if integer_positions.lt(0).any() or integer_positions.ge(len(matches)).any():
            raise ValueError("linkage diagnostics contain out-of-range sackmann_position values")
        diagnostic["source_row"] = matches.iloc[integer_positions]["source_row"].to_numpy()

    _require_unique_key(diagnostic, "source_row", "linkage diagnostics")
    available = [column for column in _LINK_DIAGNOSTIC_COLUMNS if column in diagnostic]
    renamed = diagnostic[["source_row", *available]].rename(
        columns={column: f"linkage_{column}" for column in available}
    )
    return renamed


def _merge_link_payload(matches: pd.DataFrame, result: LinkageResult) -> pd.DataFrame:
    links = result.links.copy()
    if links.empty and "sackmann_source_row" not in links:
        links["sackmann_source_row"] = pd.Series(dtype=matches["source_row"].dtype)
    _require_unique_key(links, "sackmann_source_row", "linkage links")

    unknown = set(links["sackmann_source_row"]) - set(matches["source_row"])
    if unknown:
        preview = ", ".join(map(str, list(unknown)[:5]))
        raise ValueError(f"linkage links reference unknown source_row values: {preview}")

    payload = links.rename(columns={"sackmann_source_row": "source_row"})
    if "actual_match_date" in payload:
        payload = payload.rename(columns={"actual_match_date": "linked_actual_match_date"})
    overlap = (set(payload.columns) & set(matches.columns)) - {"source_row"}
    if overlap:
        # Link metadata should remain recognizable.  Prefix only genuine name
        # collisions, rather than every field supplied by the linker.
        payload = payload.rename(columns={column: f"linkage_{column}" for column in overlap})

    merged = matches.merge(payload, on="source_row", how="left", validate="one_to_one", indicator="_link_merge")
    merged["has_odds_link"] = merged["_link_merge"].eq("both")
    merged = merged.drop(columns="_link_merge")

    if "linked_actual_match_date" in merged:
        original = pd.to_datetime(merged["actual_match_date"], errors="coerce")
        linked = pd.to_datetime(merged["linked_actual_match_date"], errors="coerce")
        conflicts = original.notna() & linked.notna() & original.ne(linked)
        if conflicts.any():
            bad = merged.loc[conflicts, "source_row"].tolist()[:5]
            raise ValueError(f"actual_match_date conflicts for source_row values: {bad}")
        merged["actual_match_date"] = linked.combine_first(original)
        merged = merged.drop(columns="linked_actual_match_date")
    return merged


def _attach_diagnostics(
    merged: pd.DataFrame,
    matches: pd.DataFrame,
    result: LinkageResult,
) -> pd.DataFrame:
    diagnostic = _diagnostics_with_source_rows(matches, result.diagnostics)
    if diagnostic.empty and len(diagnostic.columns) == 1:
        return merged
    overlap = (set(diagnostic.columns) & set(merged.columns)) - {"source_row"}
    if overlap:
        raise ValueError(f"link diagnostics collide with link payload columns: {sorted(overlap)}")
    return merged.merge(diagnostic, on="source_row", how="left", validate="one_to_one")


def attach_linked_odds(
    matches: pd.DataFrame,
    linkage: LinkageResult,
    *,
    allow_average_fallback: bool = True,
    require_outcome_consistency: bool = False,
    linked_only: bool = False,
) -> pd.DataFrame:
    """Attach, orient, and de-vig already-linked historical odds.

    A complete Pinnacle ``PSW``/``PSL`` pair is preferred.  If requested, a
    complete ``AvgW``/``AvgL`` pair is used only when the Pinnacle pair is not
    available.  A partial pair is never mixed with another bookmaker source.
    ``p_market_a`` (also exposed as ``fair_prob_a`` for the metrics module) is
    the multiplicatively de-vigged probability for canonical player A.

    ``link_outcome_consistent`` audits whether the odds file's historical
    Winner label agrees with the canonical result.  It is diagnostic by
    default because source files occasionally contain a bad Winner label even
    when the player pair and associated prices are otherwise identifiable.
    Set ``require_outcome_consistency=True`` for a strict source-quality gate.

    The returned frame contains every canonical row by default.  Set
    ``linked_only=True`` to retain only rows with a selected odds link.
    """

    if not isinstance(matches, pd.DataFrame):
        raise TypeError("matches must be a pandas DataFrame")
    if not isinstance(linkage, LinkageResult):
        raise TypeError("linkage must be a LinkageResult")
    missing = sorted(_CANONICAL_REQUIRED - set(matches.columns))
    if missing:
        raise ValueError(f"canonical matches are missing columns: {', '.join(missing)}")
    _require_unique_key(matches, "source_row", "canonical matches")

    merged = _merge_link_payload(matches.copy(), linkage)
    merged = _attach_diagnostics(merged, matches, linkage)

    output_float_columns = (
        "odds_a",
        "odds_b",
        "p_market_a",
        "fair_prob_a",
        "market_overround",
        "max_odds_a",
        "max_odds_b",
        "b365_odds_a",
        "b365_odds_b",
    )
    for column in output_float_columns:
        merged[column] = np.nan
    merged["market_odds_source"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged["link_pair_validated"] = False
    merged["link_outcome_consistent"] = pd.Series(pd.NA, index=merged.index, dtype="boolean")

    bad_pairs: list[Any] = []
    bad_outcomes: list[Any] = []
    for index in merged.index[merged["has_odds_link"]]:
        row = merged.loc[index]
        if "odds_winner" not in merged or "odds_loser" not in merged:
            raise ValueError("linkage links are missing odds_winner or odds_loser")
        try:
            winner_is_a = _orientation(row)
        except ValueError:
            bad_pairs.append(row["source_row"])
            continue

        merged.at[index, "link_pair_validated"] = True
        outcome_consistent = bool(int(row["actual_a_won"])) == winner_is_a
        merged.at[index, "link_outcome_consistent"] = outcome_consistent
        if not outcome_consistent:
            bad_outcomes.append(row["source_row"])

        selected: tuple[float, float] | None = _complete_pair(row, "odds_psw", "odds_psl")
        source = "pinnacle"
        if selected is None and allow_average_fallback:
            selected = _complete_pair(row, "odds_avgw", "odds_avgl")
            source = "average"
        if selected is not None:
            winner_price, loser_price = selected
            odds_a, odds_b = (
                (winner_price, loser_price) if winner_is_a else (loser_price, winner_price)
            )
            implied_a = 1.0 / odds_a
            implied_b = 1.0 / odds_b
            overround = implied_a + implied_b
            p_market_a = implied_a / overround
            merged.at[index, "odds_a"] = odds_a
            merged.at[index, "odds_b"] = odds_b
            merged.at[index, "p_market_a"] = p_market_a
            merged.at[index, "fair_prob_a"] = p_market_a
            merged.at[index, "market_overround"] = overround
            merged.at[index, "market_odds_source"] = source

        for winner_column, loser_column, a_column, b_column in (
            ("odds_maxw", "odds_maxl", "max_odds_a", "max_odds_b"),
            ("odds_b365w", "odds_b365l", "b365_odds_a", "b365_odds_b"),
        ):
            pair = _complete_pair(row, winner_column, loser_column)
            if pair is None:
                continue
            winner_price, loser_price = pair
            merged.at[index, a_column] = winner_price if winner_is_a else loser_price
            merged.at[index, b_column] = loser_price if winner_is_a else winner_price

    if bad_pairs:
        raise ValueError(
            "linked odds player pair mismatch for source_row values: "
            + ", ".join(map(str, bad_pairs[:10]))
        )
    if bad_outcomes and require_outcome_consistency:
        raise ValueError(
            "linked odds winner conflicts with canonical outcome for source_row values: "
            + ", ".join(map(str, bad_outcomes[:10]))
        )

    merged.attrs["linkage_summary"] = dict(linkage.summary)
    if linked_only:
        merged = merged.loc[merged["has_odds_link"]].reset_index(drop=True)
        merged.attrs["linkage_summary"] = dict(linkage.summary)
    return merged


__all__ = ["attach_linked_odds"]
