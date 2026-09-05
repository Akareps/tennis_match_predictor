import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from tennis_experiments.conditions import (
    ALL_CONDITION_METRICS,
    PSEUDO_EXPOSURE_PER_MATCH,
    build_event_condition_panel,
    fit_condition_residual,
    materialize_event_conditions,
    parse_standard_bo3_score,
)


def _match(
    key: str,
    *,
    event: str = "atp|2024-X",
    date: str = "2024-01-01",
    round_name: str = "R32",
    round_order: int = 4,
    players: tuple[int, int] = (1, 2),
    score: str = "6-4 6-4",
    retired: bool = False,
    aces: tuple[float, float] = (4.0, 4.0),
) -> dict[str, object]:
    parsed = parse_standard_bo3_score(score, is_retirement=retired)
    service_games = 20 if parsed is None else parsed["games"] - parsed["tiebreak_sets"]
    a_games = service_games // 2
    b_games = service_games - a_games
    return {
        "source_row": int(key.removeprefix("m")),
        "source_group": "main",
        "match_key": key,
        "event_key": event,
        "tournament_date": pd.Timestamp(date),
        "tourney_name": "Example Open",
        "tourney_level": "A",
        "surface": "Hard",
        "round": round_name,
        "round_order": round_order,
        "best_of": 3,
        "player_a_id": players[0],
        "player_b_id": players[1],
        "a_rank": 25,
        "b_rank": 40,
        "score": score,
        "is_retirement": retired,
        "a_aces": aces[0],
        "b_aces": aces[1],
        "a_double_faults": 3.0,
        "b_double_faults": 3.0,
        "a_service_points": 60.0,
        "b_service_points": 60.0,
        "a_first_serves_in": 36.0,
        "b_first_serves_in": 36.0,
        "a_first_serve_points_won": 25.0,
        "b_first_serve_points_won": 25.0,
        "a_second_serve_points_won": 12.0,
        "b_second_serve_points_won": 12.0,
        "a_service_games": float(a_games),
        "b_service_games": float(b_games),
        "a_break_points_saved": 2.0,
        "b_break_points_saved": 2.0,
        "a_break_points_faced": 3.0,
        "b_break_points_faced": 3.0,
    }


class StrictScoreTests(unittest.TestCase):
    def test_completed_standard_bo3_targets(self):
        straight = parse_standard_bo3_score("7-6(5) 6-4")
        self.assertEqual(straight["games"], 23)
        self.assertEqual(straight["over_22_5"], 1)
        self.assertEqual(straight["deciding_set"], 0)
        deciding = parse_standard_bo3_score("6-4 3-6 6-2")
        self.assertEqual(deciding["games"], 27)
        self.assertEqual(deciding["deciding_set"], 1)
        advantage = parse_standard_bo3_score("6-7(1) 7-6(2) 9-7")
        self.assertEqual(advantage["games"], 42)
        self.assertEqual(advantage["deciding_set"], 1)

    def test_partial_malformed_and_nonstandard_scores_are_rejected(self):
        invalid = (
            "",
            "W/O",
            "WO",
            "6-3 RET",
            "6-4",
            "6-4 6-4 junk",
            "4-2 4-2",
            "6-4 3-6 [10-8]",
            "6-4 6-4 6-3",
        )
        for score in invalid:
            with self.subTest(score=score):
                self.assertIsNone(parse_standard_bo3_score(score))
        self.assertIsNone(parse_standard_bo3_score("6-4 6-4", best_of=5))


class EventConditionFeatureTests(unittest.TestCase):
    def test_condition_residual_is_exactly_neutral_when_features_are_zero(self):
        frame = pd.DataFrame(
            {
                "p_baseline": [0.35, 0.45, 0.55, 0.65, 0.40, 0.60],
                "actual_a_won": [0, 0, 1, 1, 1, 0],
                "condition": [0.0, -0.2, 0.1, 0.0, 0.3, -0.1],
            }
        )
        model = fit_condition_residual(frame, ["condition"], l2=1.0)
        prediction = model.predict(frame)
        self.assertEqual(prediction[0], frame.loc[0, "p_baseline"])
        self.assertEqual(prediction[3], frame.loc[3, "p_baseline"])

    def test_same_round_is_frozen_and_next_round_sees_both(self):
        matches = pd.DataFrame(
            [
                _match("m1", players=(1, 2)),
                _match("m2", players=(3, 4)),
                _match(
                    "m3",
                    round_name="R16",
                    round_order=5,
                    players=(5, 6),
                ),
            ]
        )
        panel = build_event_condition_panel(matches, "atp").set_index("match_key")
        self.assertEqual(panel.loc["m1", "event_prior_point_matches"], 0)
        self.assertEqual(panel.loc["m2", "event_prior_point_matches"], 0)
        self.assertEqual(panel.loc["m3", "event_prior_point_matches"], 2)
        self.assertEqual(panel.loc["m3", "event_serve_win_residual_count"], 2)
        self.assertEqual(panel.loc["m3", "event_over_22_5_residual_count"], 2)

    def test_same_date_other_event_is_isolated_and_profiles_are_frozen(self):
        matches = pd.DataFrame(
            [
                _match("m1", event="atp|A", players=(1, 2), aces=(12, 12)),
                _match(
                    "m2",
                    event="atp|B",
                    round_name="R16",
                    round_order=5,
                    players=(1, 3),
                ),
            ]
        )
        panel = build_event_condition_panel(matches, "atp").set_index("match_key")
        self.assertEqual(panel.loc["m2", "event_prior_point_matches"], 0)
        self.assertAlmostEqual(panel.loc["m2", "expected_ace_rate"], 0.08)

    def test_prior_matches_involving_target_players_are_left_out(self):
        matches = pd.DataFrame(
            [
                _match("m1", players=(1, 2), aces=(0, 0)),
                _match("m2", players=(4, 5), aces=(10, 10)),
                _match(
                    "m3",
                    round_name="R16",
                    round_order=5,
                    players=(1, 3),
                ),
            ]
        )
        target = build_event_condition_panel(matches, "atp").set_index("match_key").loc["m3"]
        self.assertEqual(target["event_ace_residual_count"], 1)
        self.assertEqual(target["event_over_22_5_residual_count"], 1)
        self.assertAlmostEqual(target["event_ace_residual_sum"], 20.0 - 0.08 * 120.0)

    def test_current_and_future_stats_cannot_change_target_features(self):
        history = _match("m1", players=(1, 2))
        target = _match(
            "m2", round_name="R16", round_order=5, players=(3, 4)
        )
        future = _match("m3", round_name="QF", round_order=6, players=(5, 6))
        base = build_event_condition_panel(pd.DataFrame([history, target, future]), "atp")

        changed_target = dict(target)
        changed_target.update(
            {"score": "6-4 3-6 6-2", "a_aces": 15.0, "b_aces": 15.0,
             "a_service_games": 13.0, "b_service_games": 14.0}
        )
        current_mutation = build_event_condition_panel(
            pd.DataFrame([history, changed_target, future]), "atp"
        )
        feature_columns = [
            column
            for column in base.columns
            if column.startswith(("expected_", "event_", "log_", "rank_", "surface_", "level_"))
            or column in {"chronology_batch", "round_order_scaled"}
        ]
        assert_series_equal(
            base.set_index("match_key").loc["m2", feature_columns],
            current_mutation.set_index("match_key").loc["m2", feature_columns],
        )

        changed_future = dict(future)
        changed_future.update({"score": "7-6 7-6", "a_aces": 18.0, "b_aces": 18.0,
                               "a_service_games": 12.0, "b_service_games": 12.0})
        future_mutation = build_event_condition_panel(
            pd.DataFrame([history, target, changed_future]), "atp"
        )
        assert_series_equal(
            base.set_index("match_key").loc["m2"],
            future_mutation.set_index("match_key").loc["m2"],
        )

    def test_invalid_retired_and_malformed_rows_fail_closed(self):
        invalid_stats = _match("m1", players=(1, 2))
        invalid_stats["a_aces"] = 100.0
        retired = _match("m2", players=(3, 4), score="6-3 RET", retired=True)
        malformed = _match("m3", players=(5, 6), score="6-4 6-4 junk")
        target = _match(
            "m4", round_name="R16", round_order=5, players=(7, 8)
        )
        row = build_event_condition_panel(
            pd.DataFrame([invalid_stats, retired, malformed, target]), "atp"
        ).set_index("match_key").loc["m4"]
        self.assertEqual(row["event_prior_point_matches"], 0)
        self.assertEqual(row["event_prior_score_matches"], 1)
        self.assertEqual(row["event_over_22_5_residual_count"], 1)

    def test_alternative_format_excludes_the_entire_event(self):
        bracket = _match("m1", score="6-4 3-6 [10-8]")
        ordinary = _match("m2", players=(3, 4))
        panel = build_event_condition_panel(pd.DataFrame([bracket, ordinary]), "atp")
        self.assertFalse(panel["standard_event"].any())
        self.assertTrue(panel["target_over_22_5"].isna().all())

    def test_input_row_order_does_not_change_panel(self):
        matches = pd.DataFrame(
            [
                _match("m1", event="atp|A", players=(1, 2)),
                _match("m2", event="atp|A", players=(3, 4)),
                _match("m3", event="atp|A", round_name="R16", round_order=5, players=(1, 5)),
                _match("m4", event="atp|B", round_name="R16", round_order=5, players=(6, 7)),
            ]
        )
        original = build_event_condition_panel(matches, "atp").sort_values("match_key").reset_index(drop=True)
        shuffled = build_event_condition_panel(
            matches.sample(frac=1.0, random_state=7), "atp"
        ).sort_values("match_key").reset_index(drop=True)
        assert_frame_equal(original, shuffled, check_exact=False, atol=1e-12, rtol=1e-12)

    def test_materialization_uses_exposure_and_preserves_input(self):
        values = {"event_prior_point_matches": [8]}
        for metric in ALL_CONDITION_METRICS:
            values[f"event_{metric}_residual_sum"] = [0.4]
            values[f"event_{metric}_residual_exposure"] = [2.0]
        panel = pd.DataFrame(values)
        original = panel.copy(deep=True)
        result = materialize_event_conditions(panel, 2.0)
        for metric in ALL_CONDITION_METRICS:
            expected = 0.4 / (2.0 + 2.0 * PSEUDO_EXPOSURE_PER_MATCH[metric])
            self.assertAlmostEqual(result.loc[0, f"event_{metric}_condition"], expected)
        unshrunk = materialize_event_conditions(panel, 0.0)
        self.assertAlmostEqual(unshrunk.loc[0, "event_serve_win_condition"], 0.2)
        unavailable = panel.assign(event_prior_point_matches=7)
        self.assertEqual(
            materialize_event_conditions(unavailable, 0.0).loc[
                0, "event_serve_win_condition"
            ],
            0.0,
        )
        assert_frame_equal(panel, original)
        for invalid in (-1.0, np.nan, np.inf):
            with self.subTest(prior=invalid):
                with self.assertRaises(ValueError):
                    materialize_event_conditions(panel, invalid)


if __name__ == "__main__":
    unittest.main()
