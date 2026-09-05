"""Conservative cross-tier validation and untouched main-tour holdout stages."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_model_factories
from .data import canonicalize_matches, deduplicate_canonical_matches, load_cached_matches
from .experiment import (
    ProjectConfig,
    _digest_record,
    _fit_rank_before,
    _manifest,
    _prepare_directory,
    _read_json,
    _read_selected_candidates,
    _scaled_k,
    _write_json,
)
from .metrics import (
    align_common_cohort,
    binary_log_loss,
    evaluate_frame,
    paired_block_bootstrap_gap,
)
from .models import EloModel, TierWeightedRankAugmentedEloModel
from .validation import expanding_year_folds, select_by_validation_loss


STAGE_CROSS_TIER_VALIDATION = "05_cross_tier_validation"
STAGE_CROSS_TIER_HOLDOUT = "06_cross_tier_holdout"
SOURCE_GROUPS = ("main", "qualifying", "challenger", "developmental")
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0)
VARIANT_ORDER = {
    "main_only": 0,
    "plus_qualifying": 1,
    "plus_challenger": 2,
    "plus_developmental": 3,
}


def _lower_tier_paths(config: ProjectConfig, years: Iterable[int] | None = None) -> list[Path]:
    selected_years = list(config.years if years is None else years)
    root = config.project_root / "data"
    if config.tour == "atp":
        return [
            root / filename
            for year in selected_years
            for filename in (
                f"atp_matches_qual_chall_{year}.csv",
                f"atp_matches_futures_{year}.csv",
            )
        ]
    return [root / f"wta_matches_qual_itf_{year}.csv" for year in selected_years]


def _require_lower_tier_inputs(config: ProjectConfig, years: Iterable[int]) -> list[Path]:
    paths = _lower_tier_paths(config, years)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing lower-tier snapshots: " + ", ".join(missing))
    source_manifest = config.project_root / "data" / "lower_tier_sources.json"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing lower-tier source manifest: {source_manifest}")
    return [*paths, source_manifest]


def _load_pretest_cross_tier(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only years strictly before the final holdout."""

    years = range(config.history_start, config.test_year)
    _require_lower_tier_inputs(config, years)
    raw = load_cached_matches(
        config.project_root / "data",
        config.tour,
        years,
        include_lower_tiers=True,
    )
    return deduplicate_canonical_matches(canonicalize_matches(raw, config.tour))


def _load_holdout_cross_tier(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pretest lower tiers plus main-tour rows through the holdout year."""

    pretest_years = range(config.history_start, config.test_year)
    _require_lower_tier_inputs(config, pretest_years)
    history = load_cached_matches(
        config.project_root / "data",
        config.tour,
        pretest_years,
        include_lower_tiers=True,
    )
    target = load_cached_matches(
        config.project_root / "data",
        config.tour,
        [config.test_year],
        include_lower_tiers=False,
    )
    raw = pd.concat([history, target], ignore_index=True, sort=False)
    matches, audit = deduplicate_canonical_matches(canonicalize_matches(raw, config.tour))
    lower_in_holdout = matches["source_group"].ne("main") & matches["tournament_date"].ge(
        config.test_start
    )
    if lower_in_holdout.any():
        raise AssertionError("Final holdout unexpectedly contains lower-tier target-year rows")
    return matches, audit


def _candidate_name(q: float, c: float, d: float) -> str:
    tags = [int(round(value * 100)) for value in (q, c, d)]
    return f"q{tags[0]:03d}_c{tags[1]:03d}_d{tags[2]:03d}"


def cross_tier_candidate_grid() -> list[dict[str, Any]]:
    """Return nested cumulative ablations, including zero-weight nulls."""

    rows: list[dict[str, Any]] = []
    definitions = (
        ("main_only", [(0.0, 0.0, 0.0)]),
        ("plus_qualifying", [(q, 0.0, 0.0) for q in WEIGHT_GRID]),
        ("plus_challenger", [(q, c, 0.0) for q, c in product(WEIGHT_GRID, repeat=2)]),
        (
            "plus_developmental",
            [(q, c, d) for q, c, d in product(WEIGHT_GRID, repeat=3)],
        ),
    )
    for variant, weights in definitions:
        for qualifying, challenger, developmental in weights:
            rows.append(
                {
                    "variant": variant,
                    "candidate": f"{variant}__{_candidate_name(qualifying, challenger, developmental)}",
                    "qualifying_weight": qualifying,
                    "challenger_weight": challenger,
                    "developmental_weight": developmental,
                }
            )
    return rows


def _model_from_spec(
    spec: Mapping[str, Any],
    base_candidate: Mapping[str, Any],
    rank_transform: Any,
) -> EloModel:
    k_by_level, default_k = _scaled_k(float(base_candidate["k_multiplier"]))
    return TierWeightedRankAugmentedEloModel(
        rank_transform,
        source_weights={
            "main": 1.0,
            "qualifying": float(spec["qualifying_weight"]),
            "challenger": float(spec["challenger_weight"]),
            "developmental": float(spec["developmental_weight"]),
        },
        surface_weight=float(base_candidate["surface_weight"]),
        warmup_matches=int(base_candidate["warmup_matches"]),
        warmup_rank_weight=1.0,
        continuous_rank_weight=0.0,
        k_by_level=k_by_level,
        default_k=default_k,
    )


def _group_counts(matches: pd.DataFrame) -> pd.DataFrame:
    return (
        matches.assign(year=matches["tournament_date"].dt.year)
        .groupby(["year", "source_group"], as_index=False, sort=True)
        .agg(
            rows=("match_key", "size"),
            retirements=("is_retirement", "sum"),
            players_a=("player_a_id", "nunique"),
            players_b=("player_b_id", "nunique"),
        )
    )


def _validation_fold_data(
    matches: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Allow lower tiers as history only, never as same-year validation updates."""

    before_end = matches["tournament_date"].lt(validation_end)
    main_or_prior_lower = matches["source_group"].eq("main") | matches["tournament_date"].lt(
        validation_start
    )
    return matches.loc[before_end & main_or_prior_lower].copy()


def _slice_rows(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", frame
    yield "cold_start", frame.loc[frame["is_cold_start"]]
    yield "established", frame.loc[~frame["is_cold_start"]]


def run_cross_tier_validation_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stage 05: select lower-tier evidence weights entirely before 2024."""

    destination = config.stage_directory(STAGE_CROSS_TIER_VALIDATION)
    _prepare_directory(destination, overwrite)
    matches, dedupe_audit = _load_pretest_cross_tier(config)
    main = matches.loc[matches["source_group"].eq("main")].copy()
    folds = expanding_year_folds(
        main,
        test_start=config.test_start,
        date_column="tournament_date",
        minimum_training_years=1,
    )
    if not folds:
        raise ValueError("No complete pre-holdout validation folds are available")

    base_candidate = _read_selected_candidates(config)["rank_warmstart"]
    candidates = cross_tier_candidate_grid()
    evaluation_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_data = _validation_fold_data(
            matches, fold.validation_start, fold.validation_end
        )
        rank_transform = _fit_rank_before(main, fold.validation_start, "round")
        rank_rows.append(
            {
                "fold_id": fold.fold_id,
                "intercept": rank_transform.intercept,
                "log_rank_slope": rank_transform.log_rank_slope,
            }
        )
        factories: dict[str, Callable[[], EloModel]] = {
            candidate["candidate"]: (
                lambda candidate=candidate, rank_transform=rank_transform: _model_from_spec(
                    candidate, base_candidate, rank_transform
                )
            )
            for candidate in candidates
        }
        predictions = run_model_factories(
            fold_data,
            factories,
            BacktestConfig(
                test_start=fold.validation_start,
                test_end=fold.validation_end,
                horizon="round",
                update_retirements=False,
                evaluate_retirements=False,
            ),
        )
        for candidate in candidates:
            frame = predictions[candidate["candidate"]]
            if not frame["source_group"].eq("main").all():
                raise AssertionError("Validation emitted a lower-tier target")
            for slice_name, sliced in _slice_rows(frame):
                if sliced.empty:
                    continue
                evaluation_rows.append(
                    {
                        **candidate,
                        "fold_id": fold.fold_id,
                        "validation_start": fold.validation_start,
                        "validation_end": fold.validation_end,
                        "slice": slice_name,
                        "n": int(len(sliced)),
                        "log_loss": binary_log_loss(
                            sliced["p_model"], sliced["actual_a_won"]
                        ),
                    }
                )

    evaluations = pd.DataFrame(evaluation_rows)
    cold = evaluations.loc[evaluations["slice"].eq("cold_start")]
    selected: dict[str, dict[str, Any]] = {}
    summaries: list[pd.DataFrame] = []
    for variant in VARIANT_ORDER:
        candidates_for_variant = cold.loc[cold["variant"].eq(variant)]
        selected_id, candidate_summary = select_by_validation_loss(candidates_for_variant)
        selected[variant] = next(
            candidate for candidate in candidates if candidate["candidate"] == selected_id
        )
        candidate_summary.insert(0, "variant", variant)
        summaries.append(candidate_summary)

    evaluations.to_csv(destination / "fold_metrics.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(
        destination / "candidate_summary.csv", index=False
    )
    pd.DataFrame(rank_rows).to_csv(destination / "rank_transforms.csv", index=False)
    dedupe_audit.to_csv(destination / "deduplication_audit.csv", index=False)
    _group_counts(matches).to_csv(destination / "source_group_counts.csv", index=False)
    selection_payload = {
        "selection_metric": "pooled cold-start main-tour log loss",
        "cold_start_definition": "either player has fewer than 30 prior main-tour matches",
        "lower_tier_policy": "history strictly before each validation year",
        "base_model": base_candidate,
        "selected": selected,
    }
    _write_json(destination / "selected_config.json", selection_payload)
    summary = {
        "tour": config.tour,
        "folds": [asdict(fold) | {"train_indices": len(fold.train_indices), "validation_indices": len(fold.validation_indices)} for fold in folds],
        "candidate_count": len(candidates),
        "deduplicated_or_invalid_rows": int(len(dedupe_audit)),
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
            STAGE_CROSS_TIER_VALIDATION,
            {
                "forecast_horizon": "round",
                "final_holdout_loaded": False,
                "weight_grid": list(WEIGHT_GRID),
                "candidate_grid": candidates,
                "lower_tier_inputs": [
                    _digest_record(path, config.project_root) for path in lower_inputs
                ],
                "upstream_artifacts": [
                    _digest_record(
                        config.stage_directory("01_validation") / "selected_config.json",
                        config.project_root,
                    )
                ],
            },
        ),
    )
    return summary


def _read_cross_tier_selection(config: ProjectConfig) -> dict[str, Any]:
    path = config.stage_directory(STAGE_CROSS_TIER_VALIDATION) / "selected_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run cross-tier-validation first; missing: {path}")
    payload = _read_json(path)
    if set(payload.get("selected", {})) != set(VARIANT_ORDER):
        raise ValueError("Cross-tier selected_config has unexpected variants")
    return payload


def _market_cohort(config: ProjectConfig) -> pd.DataFrame:
    path = config.stage_directory("00_linkage") / "linked_market.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run linkage first; missing: {path}")
    market = pd.read_csv(path, low_memory=False)
    eligible = (
        market["market_odds_source"].eq("pinnacle")
        & market["odds_a"].notna()
        & market["odds_b"].notna()
        & market["fair_prob_a"].notna()
        & market["link_pair_validated"].eq(True)
    )
    columns = [
        "match_key",
        "actual_match_date",
        "odds_a",
        "odds_b",
        "fair_prob_a",
        "p_market_a",
        "market_overround",
        "max_odds_a",
        "max_odds_b",
        "b365_odds_a",
        "b365_odds_b",
        "link_outcome_consistent",
    ]
    result = market.loc[eligible, columns].copy()
    if result["match_key"].duplicated().any():
        raise ValueError("Market cohort contains duplicate match keys")
    return result


def _paired_delta_frame(
    variant: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    reference = baseline[["match_key", "p_model"]].rename(
        columns={"p_model": "baseline_probability"}
    )
    paired = variant.merge(reference, on="match_key", how="inner", validate="one_to_one")
    if len(paired) != len(variant) or len(paired) != len(baseline):
        raise ValueError("Variant and baseline cohorts differ")
    return paired_block_bootstrap_gap(
        paired,
        model_probability_column="p_model",
        market_probability_column="baseline_probability",
        samples=samples,
        seed=seed,
    )


def run_cross_tier_holdout_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Stage 06: compare locked lower-tier variants on the common 2024 cohort."""

    destination = config.stage_directory(STAGE_CROSS_TIER_HOLDOUT)
    _prepare_directory(destination, overwrite)
    selection = _read_cross_tier_selection(config)
    matches, dedupe_audit = _load_holdout_cross_tier(config)
    market = _market_cohort(config)
    market_dates = market.set_index("match_key")["actual_match_date"]
    dated = matches["match_key"].isin(market_dates.index)
    matches.loc[dated, "actual_match_date"] = pd.to_datetime(
        matches.loc[dated, "match_key"].map(market_dates), errors="coerce"
    ).to_numpy()
    main = matches.loc[matches["source_group"].eq("main")].copy()
    rank_transform = _fit_rank_before(main, config.test_start, "round")
    selected = selection["selected"]
    base_candidate = selection["base_model"]
    factories: dict[str, Callable[[], EloModel]] = {
        variant: (
            lambda spec=spec: _model_from_spec(spec, base_candidate, rank_transform)
        )
        for variant, spec in selected.items()
    }
    predictions = run_model_factories(
        matches,
        factories,
        BacktestConfig(
            test_start=config.test_start,
            test_end=config.test_end,
            horizon="round",
            update_retirements=False,
            evaluate_retirements=False,
        ),
    )
    if any(not frame["source_group"].eq("main").all() for frame in predictions.values()):
        raise AssertionError("Holdout emitted a lower-tier target")

    joined = {
        variant: frame.merge(market, on="match_key", how="inner", validate="one_to_one")
        for variant, frame in predictions.items()
    }
    common = align_common_cohort(joined)
    if not common or any(frame.empty for frame in common.values()):
        raise ValueError("No complete common cross-tier market cohort")
    baseline = common["main_only"]

    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for variant in sorted(common, key=lambda value: VARIANT_ORDER[value]):
        frame = common[variant]
        for slice_name, sliced in _slice_rows(frame):
            if sliced.empty:
                continue
            baseline_slice = baseline.loc[
                baseline["match_key"].isin(sliced["match_key"])
            ].copy()
            metrics = evaluate_frame(sliced)
            market_interval = paired_block_bootstrap_gap(
                sliced, samples=bootstrap_samples, seed=42
            )
            model_delta = _paired_delta_frame(
                sliced,
                baseline_slice,
                samples=bootstrap_samples,
                seed=84,
            )
            metric_rows.append(
                {
                    "tour": config.tour,
                    "variant": variant,
                    "variant_order": VARIANT_ORDER[variant],
                    "slice": slice_name,
                    **metrics,
                    "market_gap_ci_low": market_interval["ci_low"],
                    "market_gap_ci_high": market_interval["ci_high"],
                    "delta_log_loss_vs_main_only": model_delta["mean"],
                    "delta_ci_low": model_delta["ci_low"],
                    "delta_ci_high": model_delta["ci_high"],
                }
            )
            delta_rows.append(
                {
                    "tour": config.tour,
                    "variant": variant,
                    "slice": slice_name,
                    "n": int(len(sliced)),
                    "delta_log_loss_vs_main_only": model_delta["mean"],
                    "ci_low": model_delta["ci_low"],
                    "ci_high": model_delta["ci_high"],
                }
            )

    all_predictions = pd.concat(
        [frame.assign(variant=variant) for variant, frame in predictions.items()],
        ignore_index=True,
    )
    common_predictions = pd.concat(
        [frame.assign(variant=variant) for variant, frame in common.items()],
        ignore_index=True,
    )
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["slice", "variant_order"], kind="stable"
    )
    all_predictions.to_csv(destination / "predictions.csv", index=False)
    common_predictions.to_csv(destination / "common_cohort_predictions.csv", index=False)
    metrics_frame.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(destination / "paired_deltas.csv", index=False)
    dedupe_audit.to_csv(destination / "deduplication_audit.csv", index=False)
    _group_counts(matches).to_csv(destination / "source_group_counts.csv", index=False)
    _write_json(
        destination / "rank_transform.json",
        {
            "intercept": rank_transform.intercept,
            "log_rank_slope": rank_transform.log_rank_slope,
        },
    )
    all_slice = metrics_frame.loc[metrics_frame["slice"].eq("all")]
    cold_slice = metrics_frame.loc[metrics_frame["slice"].eq("cold_start")]
    summary = {
        "tour": config.tour,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "forecast_horizon": "round",
        "lower_tier_cutoff": config.test_start,
        "lower_tier_target_year_loaded": False,
        "market_cohort": "complete linked Pinnacle pairs",
        "common_matches_per_variant": int(len(baseline)),
        "cold_start_matches_per_variant": int(len(cold_slice.iloc[0:1]) and cold_slice.iloc[0]["n"]),
        "selected": selected,
        "all_metrics": all_slice.to_dict("records"),
        "cold_start_metrics": cold_slice.to_dict("records"),
    }
    _write_json(destination / "summary.json", summary)
    upstream = config.stage_directory(STAGE_CROSS_TIER_VALIDATION) / "selected_config.json"
    market_source = config.stage_directory("00_linkage") / "linked_market.csv"
    lower_inputs = _require_lower_tier_inputs(
        config, range(config.history_start, config.test_year)
    )
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_CROSS_TIER_HOLDOUT,
            {
                "forecast_horizon": "round",
                "bootstrap_samples": bootstrap_samples,
                "selection_metric": selection["selection_metric"],
                "lower_tier_policy": "strictly pre-holdout history only",
                "lower_tier_inputs": [
                    _digest_record(path, config.project_root) for path in lower_inputs
                ],
                "upstream_artifacts": [
                    _digest_record(upstream, config.project_root),
                    _digest_record(market_source, config.project_root),
                ],
            },
        ),
    )
    return summary


__all__ = [
    "STAGE_CROSS_TIER_HOLDOUT",
    "STAGE_CROSS_TIER_VALIDATION",
    "cross_tier_candidate_grid",
    "run_cross_tier_holdout_stage",
    "run_cross_tier_validation_stage",
]
