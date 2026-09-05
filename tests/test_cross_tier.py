import unittest

import pandas as pd

from tennis_experiments.cross_tier import (
    _paired_delta_frame,
    _validation_fold_data,
    cross_tier_candidate_grid,
)


class CrossTierTests(unittest.TestCase):
    def test_candidate_grid_is_nested_unique_and_includes_null_weights(self):
        grid = cross_tier_candidate_grid()
        self.assertEqual(len(grid), 85)
        self.assertEqual(len({row["candidate"] for row in grid}), 85)
        main = [row for row in grid if row["variant"] == "main_only"]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0]["developmental_weight"], 0.0)

    def test_validation_lower_tiers_stop_at_the_validation_boundary(self):
        frame = pd.DataFrame(
            {
                "tournament_date": pd.to_datetime(
                    ["2021-01-01", "2022-01-01", "2022-01-01"]
                ),
                "source_group": ["developmental", "developmental", "main"],
            }
        )
        selected = _validation_fold_data(
            frame, pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")
        )
        self.assertEqual(selected["source_group"].tolist(), ["developmental", "main"])

    def test_paired_delta_is_zero_for_identical_predictions(self):
        frame = pd.DataFrame(
            {
                "match_key": ["a", "b"],
                "event_key": ["x", "y"],
                "p_model": [0.6, 0.4],
                "actual_a_won": [1, 0],
            }
        )
        result = _paired_delta_frame(frame, frame, samples=20, seed=1)
        self.assertAlmostEqual(result["mean"], 0.0)
        self.assertAlmostEqual(result["ci_low"], 0.0)
        self.assertAlmostEqual(result["ci_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
