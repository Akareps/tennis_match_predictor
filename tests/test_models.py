import unittest

from tennis_experiments.models import (
    EloModel,
    OverallEloModel,
    RankAugmentedEloModel,
    RankTransform,
    TierWeightedRankAugmentedEloModel,
    elo_probability,
)


def match(a, b, outcome, surface="Hard", level="A", a_rank=None, b_rank=None):
    return {
        "player_a_id": a,
        "player_b_id": b,
        "actual_a_won": outcome,
        "surface": surface,
        "tourney_level": level,
        "a_rank": a_rank,
        "b_rank": b_rank,
    }


class EloTests(unittest.TestCase):
    def test_probability_is_symmetric(self):
        self.assertAlmostEqual(elo_probability(1600, 1500), 1 - elo_probability(1500, 1600))

    def test_update_conserves_total_rating(self):
        model = EloModel()
        before = model.overall[1] + model.overall[2]
        model.batch_update([match(1, 2, 1)])
        self.assertAlmostEqual(model.overall[1] + model.overall[2], before)
        self.assertGreater(model.overall[1], model.overall[2])

    def test_batch_update_is_order_independent(self):
        first = OverallEloModel()
        second = OverallEloModel()
        rows = [match(1, 2, 1), match(1, 3, 0), match(2, 3, 1)]
        first.batch_update(rows)
        second.batch_update(list(reversed(rows)))
        for player in (1, 2, 3):
            self.assertAlmostEqual(first.overall[player], second.overall[player])

    def test_rank_signal_is_used_without_changing_latent_state(self):
        transform = RankTransform(intercept=1900.0, log_rank_slope=-100.0)
        model = RankAugmentedEloModel(transform, surface_weight=0.0)
        probability = model.predict(1, 2, "Hard", a_rank=1, b_rank=100)
        self.assertGreater(probability, 0.5)
        self.assertEqual(model.overall[1], 1500.0)
        self.assertEqual(model.overall[2], 1500.0)

    def test_tier_weight_scales_rating_and_effective_evidence(self):
        transform = RankTransform(intercept=1900.0, log_rank_slope=-100.0)
        full = TierWeightedRankAugmentedEloModel(
            transform,
            source_weights={"developmental": 1.0},
            surface_weight=0.0,
        )
        quarter = TierWeightedRankAugmentedEloModel(
            transform,
            source_weights={"developmental": 0.25},
            surface_weight=0.0,
        )
        row = {**match(1, 2, 1), "source_group": "developmental"}
        full.batch_update([row])
        quarter.batch_update([row])
        self.assertAlmostEqual(
            quarter.overall[1] - quarter.initial_rating,
            0.25 * (full.overall[1] - full.initial_rating),
        )
        self.assertEqual(full.observations[1], 1.0)
        self.assertEqual(quarter.observations[1], 0.25)
        self.assertEqual(quarter.main_observations[1], 0.0)

    def test_main_observations_are_tracked_separately(self):
        model = EloModel()
        model.batch_update([{**match(1, 2, 1), "source_group": "main"}])
        details = model.predict_details(1, 2, "Hard")
        self.assertEqual(details["a_observations"], 1.0)
        self.assertEqual(details["a_main_observations"], 1.0)


if __name__ == "__main__":
    unittest.main()
