import unittest

import pandas as pd

from tennis_experiments.data import _source_groups, canonicalize_matches, deduplicate_canonical_matches


class CanonicalDataTests(unittest.TestCase):
    def test_mixed_lower_tier_files_are_taxonomized(self):
        atp = pd.DataFrame({"tourney_level": ["A", "C"]})
        self.assertEqual(
            _source_groups(atp, "atp", "atp_matches_qual_chall_2023.csv").tolist(),
            ["qualifying", "challenger"],
        )
        wta = pd.DataFrame({"tourney_level": ["P", "C", "25"]})
        self.assertEqual(
            _source_groups(wta, "wta", "wta_matches_qual_itf_2023.csv").tolist(),
            ["qualifying", "challenger", "developmental"],
        )

    def test_player_order_is_independent_of_winner(self):
        raw = pd.DataFrame(
            [
                {
                    "tourney_date": 20240101,
                    "tourney_id": "2024-X",
                    "tourney_name": "Example",
                    "tourney_level": "A",
                    "surface": "Hard",
                    "round": "R32",
                    "match_num": 1,
                    "best_of": 3,
                    "winner_id": 20,
                    "winner_name": "Higher Id",
                    "winner_rank": 10,
                    "winner_rank_points": 2000,
                    "loser_id": 10,
                    "loser_name": "Lower Id",
                    "loser_rank": 50,
                    "loser_rank_points": 900,
                    "score": "6-4 6-4",
                    "w_svpt": 60,
                    "l_svpt": 65,
                    "w_ace": 8,
                    "l_ace": 2,
                    "w_df": 3,
                    "l_df": 5,
                    "w_1stIn": 38,
                    "l_1stIn": 40,
                    "w_1stWon": 30,
                    "l_1stWon": 25,
                    "w_2ndWon": 12,
                    "l_2ndWon": 10,
                    "w_SvGms": 10,
                    "l_SvGms": 10,
                    "w_bpSaved": 4,
                    "l_bpSaved": 2,
                    "w_bpFaced": 5,
                    "l_bpFaced": 5,
                }
            ]
        )
        match = canonicalize_matches(raw, "atp").iloc[0]
        self.assertEqual(match.player_a_id, 10)
        self.assertEqual(match.player_b_id, 20)
        self.assertEqual(match.actual_a_won, 0)
        self.assertEqual(match.player_a_name, "Lower Id")
        self.assertEqual(match.a_rank, 50)
        self.assertTrue(match.has_point_stats)
        self.assertEqual(match.a_aces, 2)
        self.assertEqual(match.b_aces, 8)
        self.assertEqual(match.a_service_points, 65)
        self.assertEqual(match.b_service_points, 60)
        self.assertEqual(match.a_first_serve_points_won, 25)
        self.assertEqual(match.b_first_serve_points_won, 30)

    def test_retirement_is_explicit(self):
        raw = pd.DataFrame(
            [
                {
                    "tourney_date": 20240101,
                    "tourney_name": "Example",
                    "tourney_level": "A",
                    "surface": "Clay",
                    "winner_id": 1,
                    "winner_name": "One A.",
                    "loser_id": 2,
                    "loser_name": "Two B.",
                    "score": "6-3 RET",
                    "best_of": 3,
                }
            ]
        )
        self.assertTrue(canonicalize_matches(raw, "atp").iloc[0].is_retirement)

    def test_match_key_distinguishes_repeated_pair_in_same_round(self):
        shared = {
            "tourney_date": 20240101,
            "tourney_id": "2024-X",
            "tourney_name": "Example",
            "tourney_level": "D",
            "surface": "Hard",
            "round": "RR",
            "winner_id": 1,
            "winner_name": "One A.",
            "loser_id": 2,
            "loser_name": "Two B.",
            "score": "6-4 6-4",
        }
        raw = pd.DataFrame([{**shared, "match_num": 2}, {**shared, "match_num": 3}])
        canonical = canonicalize_matches(raw, "atp")
        self.assertTrue(canonical["match_key"].is_unique)

    def test_deduplication_ignores_changed_match_number_but_not_changed_score(self):
        shared = {
            "tourney_date": 20240101,
            "tourney_id": "2024-X",
            "tourney_name": "Example",
            "tourney_level": "C",
            "surface": "Hard",
            "round": "F",
            "winner_id": 1,
            "winner_name": "One A.",
            "loser_id": 2,
            "loser_name": "Two B.",
        }
        raw = pd.DataFrame(
            [
                {**shared, "match_num": 10, "score": "6-4 6-4"},
                {**shared, "match_num": 11, "score": " 6-4   6-4 "},
                {**shared, "match_num": 12, "score": "7-6 7-6"},
            ]
        )
        canonical = canonicalize_matches(raw, "atp")
        cleaned, audit = deduplicate_canonical_matches(canonical)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit.iloc[0].reason_code, "duplicate_event_round_players_score")

    def test_deduplication_removes_impossible_self_match(self):
        raw = pd.DataFrame(
            [
                {
                    "tourney_date": 20240101,
                    "tourney_id": "2024-X",
                    "tourney_name": "Example",
                    "tourney_level": "15",
                    "surface": "Clay",
                    "round": "Q1",
                    "match_num": 1,
                    "winner_id": 9,
                    "winner_name": "Same Player",
                    "loser_id": 9,
                    "loser_name": "Same Player",
                    "score": "6-4 6-4",
                }
            ]
        )
        canonical = canonicalize_matches(raw, "wta")
        cleaned, audit = deduplicate_canonical_matches(canonical)
        self.assertTrue(cleaned.empty)
        self.assertEqual(audit.iloc[0].reason_code, "invalid_self_match")


if __name__ == "__main__":
    unittest.main()
