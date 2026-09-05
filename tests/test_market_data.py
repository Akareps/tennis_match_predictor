import unittest

import pandas as pd

from tennis_experiments.market_data import american_to_decimal, audit_odds_gap_snapshot


class MarketDataTests(unittest.TestCase):
    def test_american_prices_convert_in_both_directions(self):
        converted = american_to_decimal(pd.Series([150, -200]))
        self.assertAlmostEqual(converted.iloc[0], 2.5)
        self.assertAlmostEqual(converted.iloc[1], 1.5)

    def test_pair_identity_survives_orientation_and_schedule_changes(self):
        rows = []
        for snapshot, start, home, away, home_p in (
            ("2026-09-01T00:00:00Z", "2026-09-02T02:00:00Z", "Ada Alpha", "Bea Beta", 0.60),
            ("2026-09-02T00:30:00Z", "2026-09-02T03:00:00Z", "Bea Beta", "Ada Alpha", 0.40),
        ):
            for side, probability in (("home", home_p), ("away", 1.0 - home_p)):
                rows.append(
                    {
                        "snapshot_ts": snapshot,
                        "sport": "tennis_wta_example",
                        "away": away,
                        "home": home,
                        "commence_time": start,
                        "market": "ml",
                        "side": side,
                        "line": None,
                        "book": "example",
                        "american_odds": -150 if probability > 0.5 else 130,
                        "devig_fair_prob": probability,
                        "best_price_flag": 1,
                    }
                )
        summary, horizons, audit = audit_odds_gap_snapshot(
            pd.DataFrame(rows), tour="wta", horizons_hours=(24, 1), tolerance_minutes=120
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0].schedule_versions, 2)
        self.assertEqual(audit["matches_with_schedule_revisions"], 1)
        self.assertTrue(horizons.loc[horizons.horizon_hours.eq(24), "available"].iloc[0])
        self.assertTrue(horizons.loc[horizons.horizon_hours.eq(1), "available"].iloc[0])

    def test_invalid_side_fails_closed(self):
        row = {
            "snapshot_ts": "2026-09-01T00:00:00Z",
            "sport": "tennis_atp_example",
            "away": "Ada Alpha",
            "home": "Bea Beta",
            "commence_time": "2026-09-02T00:00:00Z",
            "market": "ml",
            "side": "draw",
            "book": "example",
            "american_odds": 100,
            "devig_fair_prob": 0.5,
            "best_price_flag": 1,
        }
        with self.assertRaises(ValueError):
            audit_odds_gap_snapshot(pd.DataFrame([row]))


if __name__ == "__main__":
    unittest.main()
