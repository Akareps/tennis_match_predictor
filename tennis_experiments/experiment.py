"""Discrete, reproducible experiment stages for the tennis project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from . import __version__
from .backtest import BacktestConfig, iter_chronology_batches, run_model_factories
from .data import canonicalize_matches, load_cached_matches
from .linkage import LinkageConfig, LinkageResult, link_matches
from .market_data import audit_odds_gap_snapshot
from .metrics import (
    align_common_cohort,
    binary_log_loss,
    evaluate_frame,
    paired_block_bootstrap_gap,
)
from .models import DEFAULT_K_BY_LEVEL, EloModel, OverallEloModel, RankAugmentedEloModel, fit_rank_transform
from .odds import attach_linked_odds
from .validation import expanding_year_folds, select_by_validation_loss


STAGE_LINKAGE = "00_linkage"
STAGE_VALIDATION = "01_validation"
STAGE_HOLDOUT = "02_holdout"
STAGE_PRICE_BOUND = "03_price_bound"
STAGE_MARKET_SNAPSHOT = "04_market_snapshot"


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    tour: str
    history_start: int = 2021
    test_year: int = 2024
    output_root: Path | None = None

    def __post_init__(self) -> None:
        tour = self.tour.lower()
        if tour not in {"atp", "wta"}:
            raise ValueError("tour must be 'atp' or 'wta'")
        if self.history_start >= self.test_year:
            raise ValueError("history_start must be before test_year")
        object.__setattr__(self, "tour", tour)
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        if self.output_root is None:
            object.__setattr__(self, "output_root", Path(self.project_root).resolve() / "results")
        else:
            object.__setattr__(self, "output_root", Path(self.output_root).resolve())

    @property
    def years(self) -> list[int]:
        return list(range(self.history_start, self.test_year + 1))

    @property
    def test_start(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.test_year, month=1, day=1)

    @property
    def test_end(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.test_year + 1, month=1, day=1)

    def stage_directory(self, stage: str) -> Path:
        assert self.output_root is not None
        return self.output_root / stage / self.tour


def _prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Stage output already exists at {path}. Use --overwrite only when intentionally rerunning it."
        )
    path.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": _file_digest(path),
    }


def _input_paths(config: ProjectConfig) -> list[Path]:
    data_paths = [config.project_root / "data" / f"{config.tour}_matches_{year}.csv" for year in config.years]
    return [*data_paths, config.project_root / "odds_data" / f"{config.tour}_{config.test_year}.xlsx"]


def _manifest(
    config: ProjectConfig,
    stage: str,
    extra: dict[str, Any],
    *,
    input_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    inputs = []
    paths = _input_paths(config) if input_paths is None else list(input_paths)
    for path in paths:
        if path.exists():
            inputs.append(_digest_record(path, config.project_root))
    code_paths = sorted((config.project_root / "tennis_experiments").glob("*.py"))
    code_paths.append(config.project_root / "run_experiments.py")
    return {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "tour": config.tour,
        "history_start": config.history_start,
        "test_year": config.test_year,
        "inputs": inputs,
        "implementation": [_digest_record(path, config.project_root) for path in code_paths],
        **extra,
    }


def _load_matches(config: ProjectConfig) -> pd.DataFrame:
    raw = load_cached_matches(config.project_root / "data", config.tour, config.years)
    matches = canonicalize_matches(raw, config.tour)
    if not matches["source_row"].is_unique:
        raise ValueError("canonical source_row must be unique")
    if not matches["match_key"].is_unique:
        raise ValueError("canonical match_key must be unique")
    return matches


def _target_file_matches(matches: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    source_name = f"{config.tour}_matches_{config.test_year}.csv"
    target = matches.loc[matches["source_file"].eq(source_name)].copy()
    if target.empty:
        raise ValueError(f"No canonical rows came from {source_name}")
    return target


def _load_odds(config: ProjectConfig) -> pd.DataFrame:
    path = config.project_root / "odds_data" / f"{config.tour}_{config.test_year}.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached odds snapshot: {path}")
    odds = pd.read_excel(path)
    odds.columns = [str(column).strip() for column in odds.columns]
    return odds


def run_linkage_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
    linkage_config: LinkageConfig | None = None,
) -> dict[str, Any]:
    """Stage 00: link target-year matches to dated odds rows and audit it."""

    destination = config.stage_directory(STAGE_LINKAGE)
    _prepare_directory(destination, overwrite)
    matches = _load_matches(config)
    target = _target_file_matches(matches, config)
    odds = _load_odds(config)
    effective_linkage_config = linkage_config or LinkageConfig()
    result = link_matches(target, odds, effective_linkage_config)
    attached = attach_linked_odds(target, result)

    target_audit = target.reset_index(drop=True).reset_index(names="sackmann_position")
    audit = result.diagnostics.merge(
        target_audit[["sackmann_position", "round", "tourney_level", "surface"]],
        on="sackmann_position",
        how="left",
        validate="one_to_one",
    )
    audit["matched"] = audit["reason_code"].eq("matched")
    coverage_rows: list[dict[str, Any]] = []
    for dimension in ("round", "tourney_level", "surface"):
        for value, group in audit.groupby(dimension, dropna=False, sort=True):
            matched_count = int(group["matched"].sum())
            coverage_rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "rows": int(len(group)),
                    "matched_rows": matched_count,
                    "link_rate": matched_count / len(group),
                }
            )
    coverage = pd.DataFrame(coverage_rows)

    priced = attached.loc[attached["odds_a"].notna() & attached["odds_b"].notna()]
    date_values = pd.to_datetime(result.links.get("actual_match_date"), errors="coerce")
    summary = {
        **result.summary,
        "tour": config.tour,
        "link_rate": len(result.links) / len(target),
        "priced_rows": int(len(priced)),
        "pinnacle_rows": int(priced["market_odds_source"].eq("pinnacle").sum()),
        "average_fallback_rows": int(priced["market_odds_source"].eq("average").sum()),
        "outcome_label_disagreements": int(attached["link_outcome_consistent"].eq(False).sum()),
        "actual_match_date_min": date_values.min(),
        "actual_match_date_max": date_values.max(),
    }
    result.links.to_csv(destination / "links.csv", index=False)
    result.diagnostics.to_csv(destination / "diagnostics.csv", index=False)
    coverage.to_csv(destination / "coverage.csv", index=False)
    attached.loc[attached["has_odds_link"]].to_csv(destination / "linked_market.csv", index=False)
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_LINKAGE,
            {
                "linkage_config": asdict(effective_linkage_config),
                "outputs": ["coverage.csv", "diagnostics.csv", "linked_market.csv", "links.csv", "summary.json"],
            },
        ),
    )
    return summary


def _read_linkage_stage(config: ProjectConfig) -> LinkageResult:
    source = config.stage_directory(STAGE_LINKAGE)
    required = [source / "links.csv", source / "diagnostics.csv", source / "summary.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run the linkage stage first; missing: " + ", ".join(missing))
    links = pd.read_csv(source / "links.csv")
    diagnostics = pd.read_csv(source / "diagnostics.csv")
    for column in ("actual_match_date", "tournament_start_date", "odds_date"):
        if column in links:
            links[column] = pd.to_datetime(links[column], errors="coerce")
    return LinkageResult(links=links, diagnostics=diagnostics, summary=_read_json(source / "summary.json"))


def _scaled_k(multiplier: float) -> tuple[dict[str, float], float]:
    return ({level: value * multiplier for level, value in DEFAULT_K_BY_LEVEL.items()}, 24.0 * multiplier)


def _candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for multiplier in (0.75, 1.0, 1.25):
        tag = int(round(multiplier * 100))
        candidates.append(
            {"candidate": f"overall_k{tag}", "family": "overall_elo", "k_multiplier": multiplier}
        )
        for surface_weight in (0.25, 0.5, 0.75):
            surface_tag = int(round(surface_weight * 100))
            candidates.append(
                {
                    "candidate": f"surface_s{surface_tag}_k{tag}",
                    "family": "surface_elo",
                    "surface_weight": surface_weight,
                    "k_multiplier": multiplier,
                }
            )
            for warmup_matches in (15, 30, 60):
                candidates.append(
                    {
                        "candidate": f"rank_s{surface_tag}_w{warmup_matches}_k{tag}",
                        "family": "rank_warmstart",
                        "surface_weight": surface_weight,
                        "warmup_matches": warmup_matches,
                        "k_multiplier": multiplier,
                    }
                )
    return candidates


def _model_from_candidate(candidate: dict[str, Any], rank_transform: Any = None) -> EloModel:
    k_by_level, default_k = _scaled_k(float(candidate["k_multiplier"]))
    common = {"k_by_level": k_by_level, "default_k": default_k}
    family = candidate["family"]
    if family == "overall_elo":
        return OverallEloModel(**common)
    if family == "surface_elo":
        return EloModel(surface_weight=float(candidate["surface_weight"]), **common)
    if family == "rank_warmstart":
        if rank_transform is None:
            raise ValueError("rank_warmstart requires a fitted rank transform")
        return RankAugmentedEloModel(
            rank_transform,
            surface_weight=float(candidate["surface_weight"]),
            warmup_matches=int(candidate["warmup_matches"]),
            warmup_rank_weight=1.0,
            continuous_rank_weight=0.0,
            **common,
        )
    raise ValueError(f"Unknown model family: {family}")


def _records_by_batch(matches: pd.DataFrame, horizon: str) -> Iterable[list[dict[str, Any]]]:
    for _, batch in iter_chronology_batches(matches, horizon):
        yield batch.replace({np.nan: None}).to_dict("records")


def _fit_rank_before(matches: pd.DataFrame, boundary: pd.Timestamp, horizon: str) -> Any:
    training = matches.loc[
        matches["tournament_date"].lt(boundary) & ~matches["is_retirement"]
    ].copy()
    return fit_rank_transform(_records_by_batch(training, horizon))


def run_validation_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stage 01: select each model family's settings using pre-test folds only."""

    destination = config.stage_directory(STAGE_VALIDATION)
    _prepare_directory(destination, overwrite)
    matches = _load_matches(config)
    pretest = matches.loc[matches["tournament_date"].lt(config.test_start)].copy()
    folds = expanding_year_folds(
        pretest,
        test_start=config.test_start,
        date_column="tournament_date",
        minimum_training_years=1,
    )
    if not folds:
        raise ValueError("No complete pre-test validation folds are available")

    candidates = _candidate_grid()
    evaluation_rows: list[dict[str, Any]] = []
    rank_fit_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_data = pretest.loc[pretest["tournament_date"].lt(fold.validation_end)].copy()
        rank_transform = _fit_rank_before(pretest, fold.validation_start, "round")
        rank_fit_rows.append(
            {
                "fold_id": fold.fold_id,
                "intercept": rank_transform.intercept,
                "log_rank_slope": rank_transform.log_rank_slope,
            }
        )
        backtest_config = BacktestConfig(
            test_start=fold.validation_start,
            test_end=fold.validation_end,
            horizon="round",
            update_retirements=False,
            evaluate_retirements=False,
        )
        factories: dict[str, Callable[[], EloModel]] = {
            candidate["candidate"]: (
                lambda candidate=candidate, rank_transform=rank_transform: _model_from_candidate(
                    candidate, rank_transform
                )
            )
            for candidate in candidates
        }
        predictions = run_model_factories(fold_data, factories, backtest_config)
        for candidate in candidates:
            frame = predictions[candidate["candidate"]]
            evaluation_rows.append(
                {
                    **candidate,
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "validation_start": fold.validation_start,
                    "validation_end": fold.validation_end,
                    "n": int(len(frame)),
                    "log_loss": binary_log_loss(frame["p_model"], frame["actual_a_won"]),
                }
            )

    evaluations = pd.DataFrame(evaluation_rows)
    selected: dict[str, dict[str, Any]] = {}
    summaries: list[pd.DataFrame] = []
    for family, family_rows in evaluations.groupby("family", sort=True):
        candidate_id, summary = select_by_validation_loss(family_rows)
        winner = next(candidate for candidate in candidates if candidate["candidate"] == candidate_id)
        selected[str(family)] = winner
        summary.insert(0, "family", family)
        summaries.append(summary)

    selection_summary = pd.concat(summaries, ignore_index=True)
    evaluations.to_csv(destination / "fold_metrics.csv", index=False)
    selection_summary.to_csv(destination / "candidate_summary.csv", index=False)
    pd.DataFrame(rank_fit_rows).to_csv(destination / "rank_transforms.csv", index=False)
    _write_json(destination / "selected_config.json", selected)
    summary = {
        "tour": config.tour,
        "folds": [asdict(fold) | {"train_indices": len(fold.train_indices), "validation_indices": len(fold.validation_indices)} for fold in folds],
        "candidate_count": len(candidates),
        "selected": selected,
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_VALIDATION,
            {
                "forecast_horizon": "round",
                "final_test_excluded": [str(config.test_start), str(config.test_end)],
                "candidate_grid": candidates,
            },
        ),
    )
    return summary


def _read_selected_candidates(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    path = config.stage_directory(STAGE_VALIDATION) / "selected_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run the validation stage first; missing: {path}")
    selected = _read_json(path)
    expected = {"overall_elo", "surface_elo", "rank_warmstart"}
    if set(selected) != expected:
        raise ValueError(f"selected_config must contain exactly {sorted(expected)}")
    return selected


def _market_payload(attached: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "source_row",
        "has_odds_link",
        "link_pair_validated",
        "link_outcome_consistent",
        "link_score",
        "date_offset_days",
        "market_odds_source",
        "odds_a",
        "odds_b",
        "fair_prob_a",
        "p_market_a",
        "market_overround",
        "max_odds_a",
        "max_odds_b",
        "b365_odds_a",
        "b365_odds_b",
    ]
    return attached[[column for column in desired if column in attached]].copy()


def _score_slices(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", frame
    yield "cold_start", frame.loc[frame["is_cold_start"]]
    yield "established", frame.loc[~frame["is_cold_start"]]


def run_holdout_stage(
    config: ProjectConfig,
    *,
    overwrite: bool = False,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Stage 02: evaluate locked pre-test choices once on the final test year."""

    destination = config.stage_directory(STAGE_HOLDOUT)
    _prepare_directory(destination, overwrite)
    selected = _read_selected_candidates(config)
    linkage = _read_linkage_stage(config)
    matches = _load_matches(config)
    target = _target_file_matches(matches, config)
    attached_target = attach_linked_odds(target, linkage)

    date_map = attached_target.set_index("source_row")["actual_match_date"]
    target_mask = matches["source_row"].isin(date_map.index)
    matches.loc[target_mask, "actual_match_date"] = matches.loc[target_mask, "source_row"].map(date_map)
    market = _market_payload(attached_target)

    all_predictions: list[pd.DataFrame] = []
    rank_fits: dict[str, dict[str, float]] = {}
    for horizon in ("draw", "round"):
        rank_transform = _fit_rank_before(matches, config.test_start, horizon)
        rank_fits[horizon] = {
            "intercept": rank_transform.intercept,
            "log_rank_slope": rank_transform.log_rank_slope,
        }
        factories = {
            family: (
                lambda candidate=candidate, rank_transform=rank_transform: _model_from_candidate(
                    candidate, rank_transform
                )
            )
            for family, candidate in selected.items()
        }
        predictions = run_model_factories(
            matches,
            factories,
            BacktestConfig(
                test_start=config.test_start,
                test_end=config.test_end,
                horizon=horizon,
                update_retirements=False,
                evaluate_retirements=False,
            ),
        )
        for family, frame in predictions.items():
            joined = frame.merge(market, on="source_row", how="left", validate="one_to_one")
            joined["family"] = family
            joined["selected_candidate"] = selected[family]["candidate"]
            all_predictions.append(joined)

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    by_model = {
        f"{horizon}__{family}": group.loc[
            group["market_odds_source"].eq("pinnacle")
            & group["odds_a"].notna()
            & group["odds_b"].notna()
            & group["fair_prob_a"].notna()
        ].copy()
        for (horizon, family), group in prediction_frame.groupby(
            ["chronology_horizon", "family"], sort=True
        )
    }
    common = align_common_cohort(by_model)
    common_frame = pd.concat(common.values(), ignore_index=True) if common else pd.DataFrame()
    representative = next(iter(common.values())) if common else pd.DataFrame()
    cohort_dates = (
        pd.to_datetime(representative["prediction_date"], errors="coerce")
        if "prediction_date" in representative
        else pd.Series(dtype="datetime64[ns]")
    )

    metric_rows: list[dict[str, Any]] = []
    for model_key, frame in common.items():
        horizon, family = model_key.split("__", 1)
        for slice_name, sliced in _score_slices(frame):
            if sliced.empty:
                continue
            metrics = evaluate_frame(sliced)
            interval = paired_block_bootstrap_gap(sliced, samples=bootstrap_samples)
            metric_rows.append(
                {
                    "tour": config.tour,
                    "horizon": horizon,
                    "family": family,
                    "selected_candidate": selected[family]["candidate"],
                    "slice": slice_name,
                    **metrics,
                    "gap_ci_low": interval["ci_low"],
                    "gap_ci_high": interval["ci_high"],
                }
            )
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["slice", "horizon", "log_loss", "family"], kind="stable"
    )
    prediction_frame.to_csv(destination / "predictions.csv", index=False)
    common_frame.to_csv(destination / "common_cohort_predictions.csv", index=False)
    metrics_frame.to_csv(destination / "metrics.csv", index=False)
    _write_json(destination / "rank_transforms.json", rank_fits)
    summary = {
        "tour": config.tour,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "retirements_updated": False,
        "retirements_evaluated": False,
        "market_cohort": "complete Pinnacle pair only",
        "common_matches_per_model": int(len(representative)),
        "cohort_date_min": cohort_dates.min() if len(cohort_dates) else None,
        "cohort_date_max": cohort_dates.max() if len(cohort_dates) else None,
        "models": sorted(common),
        "selected": selected,
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_HOLDOUT,
            {
                "forecast_horizons": ["draw", "round"],
                "bootstrap_samples": bootstrap_samples,
                "selected_candidates": selected,
                "upstream_artifacts": [
                    _digest_record(
                        config.stage_directory(STAGE_LINKAGE) / "links.csv", config.project_root
                    ),
                    _digest_record(
                        config.stage_directory(STAGE_LINKAGE) / "diagnostics.csv", config.project_root
                    ),
                    _digest_record(
                        config.stage_directory(STAGE_VALIDATION) / "selected_config.json",
                        config.project_root,
                    ),
                ],
            },
        ),
    )
    return summary


def _settled_return(frame: pd.DataFrame, a_column: str, b_column: str) -> dict[str, Any]:
    available = frame[a_column].notna() & frame[b_column].notna()
    priced = frame.loc[available]
    if priced.empty:
        return {"bets": 0, "profit": 0.0, "roi": float("nan"), "mean_price": float("nan")}
    bet_a = priced["bet_side"].eq("a")
    prices = priced[a_column].where(bet_a, priced[b_column]).astype(float)
    won = priced["actual_a_won"].astype(bool).eq(bet_a)
    returns = np.where(won, prices - 1.0, -1.0)
    return {
        "bets": int(len(priced)),
        "profit": float(np.sum(returns)),
        "roi": float(np.mean(returns)),
        "mean_price": float(prices.mean()),
    }


def run_price_bound_stage(config: ProjectConfig, *, overwrite: bool = False) -> dict[str, Any]:
    """Stage 03: replay identical positive-EV Pinnacle selections at other prices."""

    destination = config.stage_directory(STAGE_PRICE_BOUND)
    _prepare_directory(destination, overwrite)
    source = config.stage_directory(STAGE_HOLDOUT) / "common_cohort_predictions.csv"
    if not source.exists():
        raise FileNotFoundError(f"Run the holdout stage first; missing: {source}")
    predictions = pd.read_csv(source)

    output_rows: list[dict[str, Any]] = []
    bet_frames: list[pd.DataFrame] = []
    for (horizon, family), group in predictions.groupby(["chronology_horizon", "family"], sort=True):
        frame = group.copy()
        frame["ev_a_at_pinnacle"] = frame["p_model"] * frame["odds_a"] - 1.0
        frame["ev_b_at_pinnacle"] = (1.0 - frame["p_model"]) * frame["odds_b"] - 1.0
        frame["bet_side"] = np.where(
            frame["ev_a_at_pinnacle"].ge(frame["ev_b_at_pinnacle"]), "a", "b"
        )
        frame["selected_ev"] = frame[["ev_a_at_pinnacle", "ev_b_at_pinnacle"]].max(axis=1)
        bets = frame.loc[frame["selected_ev"].gt(0)].copy()
        bets["family"] = family
        bet_frames.append(bets)
        for price_source, a_column, b_column in (
            ("pinnacle", "odds_a", "odds_b"),
            ("bet365", "b365_odds_a", "b365_odds_b"),
            ("ex_post_maximum", "max_odds_a", "max_odds_b"),
        ):
            result = _settled_return(bets, a_column, b_column)
            output_rows.append(
                {
                    "tour": config.tour,
                    "horizon": horizon,
                    "family": family,
                    "selection_rule": "positive expected value at Pinnacle close",
                    "price_source": price_source,
                    **result,
                }
            )

    output = pd.DataFrame(output_rows)
    output.to_csv(destination / "price_bound.csv", index=False)
    pd.concat(bet_frames, ignore_index=True).to_csv(destination / "selected_bets.csv", index=False)
    summary = {
        "tour": config.tour,
        "selection_rule": "positive model expected value using the Pinnacle closing price",
        "same_selections_across_price_sources": True,
        "warning": "MaxW/MaxL is an ex-post best quoted price, not proof that a bet was executable.",
        "rows": output.to_dict("records"),
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_PRICE_BOUND,
            {
                **summary,
                "upstream_artifacts": [_digest_record(source, config.project_root)],
            },
        ),
    )
    return summary


def run_market_snapshot_stage(config: ProjectConfig, *, overwrite: bool = False) -> dict[str, Any]:
    """Stage 04: freeze and validate free timestamped odds without outcome claims."""

    destination = config.stage_directory(STAGE_MARKET_SNAPSHOT)
    _prepare_directory(destination, overwrite)
    basename = "odds_gap_tennis_ml_20260904"
    raw_path = config.project_root / "odds_data" / f"{basename}.csv"
    headers_path = config.project_root / "odds_data" / f"{basename}.headers.txt"
    source_manifest = config.project_root / "odds_data" / f"{basename}.manifest.json"
    required = [raw_path, headers_path, source_manifest]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen market snapshot inputs: " + ", ".join(missing))

    matches, horizons, audit = audit_odds_gap_snapshot(raw_path, tour=config.tour)
    matches.to_csv(destination / "match_snapshot_summary.csv", index=False)
    horizons.to_csv(destination / "forecast_horizon_availability.csv", index=False)
    summary = {
        "tour": config.tour,
        **audit,
        "status": "intake validated; outcomes unavailable",
        "claim_boundary": (
            "This stage validates timestamp, price, schedule-revision, and fixed-horizon handling only. "
            "It does not estimate accuracy, edge, execution, or profitability."
        ),
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "manifest.json",
        _manifest(
            config,
            STAGE_MARKET_SNAPSHOT,
            {
                "raw_snapshot_inputs": [
                    _digest_record(path, config.project_root) for path in required
                ],
                "forecast_horizons_hours": [24.0, 6.0, 1.0],
                "horizon_tolerance_minutes": 90.0,
                "outcome_data_used": False,
                "outputs": [
                    "forecast_horizon_availability.csv",
                    "match_snapshot_summary.csv",
                    "summary.json",
                ],
            },
        ),
    )
    return summary


__all__ = [
    "ProjectConfig",
    "run_linkage_stage",
    "run_validation_stage",
    "run_holdout_stage",
    "run_price_bound_stage",
    "run_market_snapshot_stage",
]
