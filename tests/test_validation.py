import unittest

import pandas as pd

from tennis_experiments.validation import expanding_year_folds, select_by_validation_loss


class TemporalValidationTests(unittest.TestCase):
    def test_folds_expand_and_never_touch_final_test(self):
        frame = pd.DataFrame(
            {"date": pd.to_datetime(["2021-06-01", "2022-06-01", "2023-06-01", "2024-06-01"])}
        )
        folds = expanding_year_folds(frame, date_column="date", test_start="2024-01-01")
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0].validation_indices, (1,))
        self.assertEqual(folds[1].train_indices, (0, 1))
        self.assertEqual(folds[1].validation_indices, (2,))
        for fold in folds:
            self.assertLessEqual(fold.validation_end, pd.Timestamp("2024-01-01"))
            self.assertTrue(set(fold.train_indices).isdisjoint(fold.validation_indices))

    def test_partial_validation_window_is_not_used(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2021-01-01", "2022-01-01"])})
        folds = expanding_year_folds(
            frame, date_column="date", test_start="2022-07-01", validation_years=1
        )
        self.assertEqual(folds, [])

    def test_invalid_dates_fail_closed(self):
        frame = pd.DataFrame({"date": ["2021-01-01", "not-a-date"]})
        with self.assertRaises(ValueError):
            expanding_year_folds(frame, date_column="date", test_start="2024-01-01")

    def test_selection_uses_weighted_loss_and_deterministic_ties(self):
        evaluations = pd.DataFrame(
            [
                {"candidate": "b", "log_loss": 0.4, "n": 10},
                {"candidate": "b", "log_loss": 0.6, "n": 10},
                {"candidate": "a", "log_loss": 0.5, "n": 20},
            ]
        )
        selected, summary = select_by_validation_loss(evaluations.sample(frac=1, random_state=3))
        self.assertEqual(selected, "a")
        self.assertEqual(list(summary["candidate"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
