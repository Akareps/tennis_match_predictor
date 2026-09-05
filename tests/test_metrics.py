import unittest

import pandas as pd

from tennis_experiments.metrics import (
    align_common_cohort,
    binary_log_loss,
    evaluate_frame,
    flat_stake_roi,
)


class MetricsTests(unittest.TestCase):
    def test_better_probabilities_have_lower_loss(self):
        outcomes = [1, 0]
        self.assertLess(binary_log_loss([0.9, 0.1], outcomes), binary_log_loss([0.6, 0.4], outcomes))

    def test_flat_stake_roi(self):
        frame = pd.DataFrame(
            {
                "p_model": [0.8, 0.8],
                "odds_a": [2.0, 2.0],
                "odds_b": [2.0, 2.0],
                "actual_a_won": [1, 0],
            }
        )
        result = flat_stake_roi(frame)
        self.assertEqual(result["bets"], 2)
        self.assertEqual(result["roi"], 0.0)

    def test_common_cohort(self):
        frames = {
            "a": pd.DataFrame({"match_key": ["1", "2"], "x": [1, 2]}),
            "b": pd.DataFrame({"match_key": ["2", "3"], "x": [2, 3]}),
        }
        aligned = align_common_cohort(frames)
        self.assertEqual(aligned["a"].match_key.tolist(), ["2"])
        self.assertEqual(aligned["b"].match_key.tolist(), ["2"])

    def test_evaluate_frame_includes_market_gap(self):
        frame = pd.DataFrame(
            {
                "p_model": [0.8, 0.2, 0.7, 0.3],
                "fair_prob_a": [0.7, 0.3, 0.6, 0.4],
                "actual_a_won": [1, 0, 1, 0],
                "odds_a": [2.0] * 4,
                "odds_b": [2.0] * 4,
            }
        )
        metrics = evaluate_frame(frame)
        self.assertLess(metrics["log_loss_gap"], 0)


if __name__ == "__main__":
    unittest.main()
