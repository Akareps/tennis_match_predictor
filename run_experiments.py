"""Command-line entry point for the rebuilt experiment sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from tennis_experiments.cross_tier import (
    run_cross_tier_holdout_stage,
    run_cross_tier_validation_stage,
)
from tennis_experiments.conditions import (
    run_event_conditions_holdout_stage,
    run_event_conditions_validation_stage,
)
from tennis_experiments.experiment import (
    ProjectConfig,
    run_holdout_stage,
    run_linkage_stage,
    run_market_snapshot_stage,
    run_price_bound_stage,
    run_validation_stage,
)
from tennis_experiments.fitness import (
    run_workload_holdout_stage,
    run_workload_validation_stage,
)


STAGES: dict[str, Callable[..., dict[str, Any]]] = {
    "linkage": run_linkage_stage,
    "validation": run_validation_stage,
    "holdout": run_holdout_stage,
    "price-bound": run_price_bound_stage,
    "market-snapshot": run_market_snapshot_stage,
    "cross-tier-validation": run_cross_tier_validation_stage,
    "cross-tier-holdout": run_cross_tier_holdout_stage,
    "workload-validation": run_workload_validation_stage,
    "workload-holdout": run_workload_holdout_stage,
    "event-conditions-validation": run_event_conditions_validation_stage,
    "event-conditions-holdout": run_event_conditions_holdout_stage,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run one immutable tennis experiment stage at a time."
    )
    result.add_argument("stage", choices=STAGES)
    result.add_argument("--tour", choices=("atp", "wta", "both"), default="both")
    result.add_argument("--history-start", type=int, default=2021)
    result.add_argument("--test-year", type=int, default=2024)
    result.add_argument("--output-root", type=Path, default=None)
    result.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files in this stage's existing output directory",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    root = Path(__file__).resolve().parent
    tours = ("atp", "wta") if args.tour == "both" else (args.tour,)
    for tour in tours:
        config = ProjectConfig(
            project_root=root,
            tour=tour,
            history_start=args.history_start,
            test_year=args.test_year,
            output_root=args.output_root,
        )
        summary = STAGES[args.stage](config, overwrite=args.overwrite)
        print(f"{tour.upper()} {args.stage}: {summary}")


if __name__ == "__main__":
    main()
