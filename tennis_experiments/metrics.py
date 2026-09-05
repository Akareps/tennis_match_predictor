"""Common-cohort scoring for every experiment stage."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EPSILON = 1e-9


def _probabilities(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1.0 - EPSILON)


def binary_log_loss(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p = _probabilities(probabilities)
    y = np.asarray(outcomes, dtype=float)
    if len(p) != len(y):
        raise ValueError("probabilities and outcomes must have equal length")
    if not len(p):
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def accuracy(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    return float(np.mean((p >= 0.5) == y)) if len(p) else float("nan")


def calibration_intercept_slope(
    probabilities: Iterable[float],
    outcomes: Iterable[int],
    *,
    max_iterations: int = 100,
) -> tuple[float, float]:
    """Fit ``outcome ~ intercept + slope * logit(probability)`` via Newton steps."""

    p = _probabilities(probabilities)
    y = np.asarray(outcomes, dtype=float)
    if len(p) < 3 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit_p = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(p)), logit_p])
    beta = np.array([0.0, 1.0])
    for _ in range(max_iterations):
        eta = np.clip(design @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (y - fitted)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def flat_stake_roi(
    frame: pd.DataFrame,
    *,
    probability_column: str = "p_model",
    threshold: float = 0.0,
) -> dict[str, float | int]:
    """Bet the side with greatest positive model EV at recorded decimal odds."""

    required = {probability_column, "odds_a", "odds_b", "actual_a_won"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"bets": 0, "profit": 0.0, "roi": float("nan")}
    profits: list[float] = []
    for row in frame.itertuples(index=False):
        p_a = float(getattr(row, probability_column))
        odds_a = float(row.odds_a)
        odds_b = float(row.odds_b)
        ev_a = p_a * odds_a - 1.0
        ev_b = (1.0 - p_a) * odds_b - 1.0
        if max(ev_a, ev_b) <= threshold:
            continue
        bet_a = ev_a >= ev_b
        won = bool(row.actual_a_won) if bet_a else not bool(row.actual_a_won)
        odds = odds_a if bet_a else odds_b
        profits.append(odds - 1.0 if won else -1.0)
    profit = float(sum(profits))
    return {
        "bets": len(profits),
        "profit": profit,
        "roi": profit / len(profits) if profits else float("nan"),
    }


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    probability_column: str = "p_model",
    market_probability_column: str = "fair_prob_a",
) -> dict[str, float | int]:
    y = frame["actual_a_won"].to_numpy(dtype=float)
    p = frame[probability_column].to_numpy(dtype=float)
    intercept, slope = calibration_intercept_slope(p, y)
    metrics: dict[str, float | int] = {
        "n": len(frame),
        "log_loss": binary_log_loss(p, y),
        "brier": brier_score(p, y),
        "accuracy": accuracy(p, y),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    if market_probability_column in frame and frame[market_probability_column].notna().all():
        market = frame[market_probability_column].to_numpy(dtype=float)
        market_loss = binary_log_loss(market, y)
        metrics.update(
            {
                "market_log_loss": market_loss,
                "log_loss_gap": float(metrics["log_loss"]) - market_loss,
                "market_brier": brier_score(market, y),
                "market_accuracy": accuracy(market, y),
            }
        )
    roi = flat_stake_roi(frame, probability_column=probability_column)
    metrics.update({f"bet_{key}": value for key, value in roi.items()})
    return metrics


def align_common_cohort(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every model to identical, unique match keys."""

    if not frames:
        return {}
    common: set[str] | None = None
    for name, frame in frames.items():
        if frame["match_key"].duplicated().any():
            raise ValueError(f"Model {name} contains duplicate match_key rows")
        keys = set(frame["match_key"].dropna().astype(str))
        common = keys if common is None else common & keys
    common = common or set()
    return {
        name: frame[frame["match_key"].astype(str).isin(common)].sort_values("match_key").reset_index(drop=True)
        for name, frame in frames.items()
    }


def paired_block_bootstrap_gap(
    frame: pd.DataFrame,
    *,
    model_probability_column: str = "p_model",
    market_probability_column: str = "fair_prob_a",
    block_column: str = "event_key",
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired model-minus-market log-loss by tournament/week block."""

    if not len(frame):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    p_model = _probabilities(frame[model_probability_column])
    p_market = _probabilities(frame[market_probability_column])
    y = frame["actual_a_won"].to_numpy(dtype=float)
    per_match = -(
        y * np.log(p_model) + (1 - y) * np.log(1 - p_model)
    ) + (y * np.log(p_market) + (1 - y) * np.log(1 - p_market))
    working = pd.DataFrame({"block": frame[block_column].astype(str), "difference": per_match})
    groups = [group["difference"].to_numpy() for _, group in working.groupby("block", sort=False)]
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        picked = rng.integers(0, len(groups), len(groups))
        values = np.concatenate([groups[group_index] for group_index in picked])
        draws[index] = values.mean()
    return {
        "mean": float(per_match.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }
