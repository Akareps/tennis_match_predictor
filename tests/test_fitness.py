import unittest

import numpy as np
import pandas as pd

from tennis_experiments.fitness import (
    build_workload_features,
    fit_offset_logistic,
    score_load,
)
from tennis_experiments.metrics import binary_log_loss


def _match(
    source_row: int,
    *,
    date: str,
    event: str,
    round_name: str,
    round_order: int,
    player_a: int,
    player_b: int,
    score: str = "6-4 6-4",
    minutes: float | None = 90.0,
    actual_a_won: int = 1,
    retirement: bool = False,
) -> dict:
    return {
        "source_row": source_row,
        "source_group": "main",
        "match_key": f"m{source_row}",
        "event_key": event,
        "tournament_date": pd.Timestamp(date),
        "actual_match_date": pd.NaT,
        "surface": "Hard",
        "round": round_name,
        "round_order": round_order,
        "score": score,
        "minutes": minutes,
        "best_of": 3,
        "player_a_id": player_a,
        "player_b_id": player_b,
        "actual_a_won": actual_a_won,
        "is_retirement": retirement,
    }


class ScoreLoadTests(unittest.TestCase):
    def test_completed_retired_and_walkover_scores(self) -> None:
        self.assertEqual(score_load("6-4 7-6(5)")["games"], 23)
        self.assertEqual(score_load("3-2 RET")["games"], 5)
        self.assertFalse(score_load("W/O")["played"])

    def test_match_tiebreak_is_capped(self) -> None:
        self.assertEqual(score_load("6-4 3-6 10-8")["games"], 32)


class WorkloadFeatureTests(unittest.TestCase):
    def test_same_batch_is_frozen_and_prior_round_is_visible(self) -> None:
        matches = pd.DataFrame(
            [
                _match(0, date="2023-01-01", event="e1", round_name="R32", round_order=4, player_a=1, player_b=2),
                _match(1, date="2023-01-01", event="e1", round_name="R32", round_order=4, player_a=1, player_b=3),
                _match(2, date="2023-01-01", event="e1", round_name="R16", round_order=5, player_a=1, player_b=4),
                _match(3, date="2023-01-08", event="e2", round_name="R32", round_order=4, player_a=1, player_b=5),
            ]
        )
        features = build_workload_features(matches).set_index("match_key")
        self.assertEqual(features.loc["m0", "a_same_event_matches"], 0)
        self.assertEqual(features.loc["m1", "a_same_event_matches"], 0)
        self.assertEqual(features.loc["m2", "a_same_event_matches"], 2)
        self.assertEqual(features.loc["m3", "a_matches_14d"], 3)
        self.assertEqual(features.loc["m3", "a_days_since_previous_event"], 7)

    def test_only_loser_receives_availability_event(self) -> None:
        matches = pd.DataFrame(
            [
                _match(
                    0,
                    date="2023-01-01",
                    event="e1",
                    round_name="R32",
                    round_order=4,
                    player_a=1,
                    player_b=2,
                    score="6-3 2-1 RET",
                    retirement=True,
                ),
                _match(1, date="2023-01-08", event="e2", round_name="R32", round_order=4, player_a=1, player_b=3),
                _match(2, date="2023-01-08", event="e3", round_name="R32", round_order=4, player_a=2, player_b=4),
            ]
        )
        features = build_workload_features(matches).set_index("match_key")
        self.assertEqual(features.loc["m1", "a_retirement_decay"], 0)
        self.assertGreater(features.loc["m2", "a_retirement_decay"], 0)

    def test_same_date_other_event_is_not_treated_as_known_availability(self) -> None:
        matches = pd.DataFrame(
            [
                _match(
                    0,
                    date="2023-01-01",
                    event="e1",
                    round_name="R32",
                    round_order=4,
                    player_a=1,
                    player_b=2,
                    score="6-3 2-1 RET",
                    retirement=True,
                ),
                _match(
                    1,
                    date="2023-01-01",
                    event="e2",
                    round_name="R16",
                    round_order=5,
                    player_a=2,
                    player_b=3,
                ),
            ]
        )
        features = build_workload_features(matches).set_index("match_key")
        self.assertEqual(features.loc["m1", "a_retirement_decay"], 0)


class OffsetModelTests(unittest.TestCase):
    def test_residual_can_learn_signal_without_changing_base_coefficient(self) -> None:
        feature = np.linspace(-2.0, 2.0, 200)
        outcome = (feature > 0).astype(int)
        frame = pd.DataFrame(
            {"p_model": np.full(len(feature), 0.5), "actual_a_won": outcome, "signal": feature}
        )
        model = fit_offset_logistic(frame, ["signal"], l2=1.0)
        fitted = model.predict(frame)
        self.assertTrue(model.converged)
        self.assertLess(binary_log_loss(fitted, outcome), binary_log_loss(frame["p_model"], outcome))
        self.assertGreater(fitted[-1], fitted[0])

    def test_nonfinite_features_fail_closed(self) -> None:
        frame = pd.DataFrame(
            {"p_model": [0.4, 0.6, 0.5], "actual_a_won": [0, 1, 0], "signal": [0.0, np.nan, 1.0]}
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_offset_logistic(frame, ["signal"], l2=1.0)


if __name__ == "__main__":
    unittest.main()
