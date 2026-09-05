import unittest

import pandas as pd

from tennis_experiments.backtest import (
    BacktestConfig,
    iter_chronology_batches,
    run_model_factories,
    run_walk_forward,
)
from tennis_experiments.models import EloModel, OverallEloModel


def matches_frame():
    # Deliberately final-first, like the Sackmann source files.
    return pd.DataFrame(
        [
            {
                "source_row": 0,
                "event_key": "atp|x",
                "match_key": "final",
                "tournament_date": pd.Timestamp("2024-01-01"),
                "actual_match_date": pd.NaT,
                "tourney_name": "X",
                "tourney_level": "A",
                "surface": "Hard",
                "round": "F",
                "round_order": 8,
                "player_a_id": 1,
                "player_b_id": 3,
                "a_rank": None,
                "b_rank": None,
                "actual_a_won": 0,
                "is_retirement": False,
            },
            {
                "source_row": 1,
                "event_key": "atp|x",
                "match_key": "first",
                "tournament_date": pd.Timestamp("2024-01-01"),
                "actual_match_date": pd.NaT,
                "tourney_name": "X",
                "tourney_level": "A",
                "surface": "Hard",
                "round": "R32",
                "round_order": 4,
                "player_a_id": 1,
                "player_b_id": 2,
                "a_rank": None,
                "b_rank": None,
                "actual_a_won": 1,
                "is_retirement": False,
            },
        ]
    )


class BacktestTests(unittest.TestCase):
    def test_round_batches_sort_early_round_first(self):
        batches = list(iter_chronology_batches(matches_frame(), "round"))
        self.assertEqual(batches[0][1].iloc[0].match_key, "first")
        self.assertEqual(batches[1][1].iloc[0].match_key, "final")

    def test_round_horizon_uses_prior_round_but_draw_horizon_does_not(self):
        config = dict(test_start=pd.Timestamp("2024-01-01"), test_end=pd.Timestamp("2024-02-01"))
        round_predictions = run_walk_forward(
            matches_frame(), OverallEloModel(), BacktestConfig(horizon="round", **config)
        ).set_index("match_key")
        draw_predictions = run_walk_forward(
            matches_frame(), OverallEloModel(), BacktestConfig(horizon="draw", **config)
        ).set_index("match_key")
        self.assertEqual(round_predictions.loc["first", "p_model"], 0.5)
        self.assertGreater(round_predictions.loc["final", "p_model"], 0.5)
        self.assertEqual(draw_predictions.loc["first", "p_model"], 0.5)
        self.assertEqual(draw_predictions.loc["final", "p_model"], 0.5)

    def test_match_date_requires_complete_dates(self):
        with self.assertRaises(ValueError):
            list(iter_chronology_batches(matches_frame(), "match_date"))

    def test_same_date_events_are_predicted_before_cross_event_updates(self):
        rows = matches_frame()
        other = rows.iloc[[1]].copy()
        other["source_row"] = 2
        other["event_key"] = "atp|a-earlier-alphabetically"
        other["match_key"] = "other-first"
        combined = pd.concat([rows, other], ignore_index=True)
        round_batches = list(iter_chronology_batches(combined, "round"))
        first_round = round_batches[0][1]
        self.assertEqual(set(first_round["match_key"]), {"first", "other-first"})
        draw_batches = list(iter_chronology_batches(combined, "draw"))
        self.assertEqual(len(draw_batches), 1)

    def test_shared_runner_matches_independent_runs(self):
        config = BacktestConfig(
            test_start=pd.Timestamp("2024-01-01"),
            test_end=pd.Timestamp("2024-02-01"),
            horizon="round",
        )
        factories = {
            "overall": lambda: OverallEloModel(),
            "surface": lambda: EloModel(surface_weight=0.5),
        }
        shared = run_model_factories(matches_frame(), factories, config)
        for name, factory in factories.items():
            independent = run_walk_forward(matches_frame(), factory(), config, model_name=name)
            pd.testing.assert_frame_equal(shared[name], independent)


if __name__ == "__main__":
    unittest.main()
