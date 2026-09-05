import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tennis_experiments.experiment import (
    ProjectConfig,
    _candidate_grid,
    _prepare_directory,
    _settled_return,
)


class ExperimentStageTests(unittest.TestCase):
    def test_candidate_grid_is_unique_and_contains_three_families(self):
        candidates = _candidate_grid()
        self.assertEqual(len(candidates), 39)
        self.assertEqual(len({row["candidate"] for row in candidates}), 39)
        self.assertEqual(
            {row["family"] for row in candidates},
            {"overall_elo", "surface_elo", "rank_warmstart"},
        )

    def test_existing_stage_is_not_overwritten_implicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "stage"
            destination.mkdir()
            (destination / "result.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_directory(destination, overwrite=False)
            _prepare_directory(destination, overwrite=True)
            self.assertEqual((destination / "result.txt").read_text(encoding="utf-8"), "keep")

    def test_price_replay_keeps_the_selected_side(self):
        frame = pd.DataFrame(
            {
                "bet_side": ["a", "b"],
                "actual_a_won": [1, 1],
                "price_a": [2.0, 1.5],
                "price_b": [2.2, 3.0],
            }
        )
        result = _settled_return(frame, "price_a", "price_b")
        self.assertEqual(result["bets"], 2)
        self.assertAlmostEqual(result["profit"], 0.0)

    def test_project_config_defines_a_half_open_test_year(self):
        config = ProjectConfig(Path("."), "ATP", history_start=2021, test_year=2024)
        self.assertEqual(config.tour, "atp")
        self.assertEqual(config.test_start, pd.Timestamp("2024-01-01"))
        self.assertEqual(config.test_end, pd.Timestamp("2025-01-01"))


if __name__ == "__main__":
    unittest.main()
