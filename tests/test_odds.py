import unittest

import numpy as np
import pandas as pd

from tennis_experiments.linkage import LinkageResult
from tennis_experiments.odds import attach_linked_odds


def canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row": [10, 11, 12],
            "match_key": ["m10", "m11", "m12"],
            "player_a_name": ["Alice Alpha", "Alice Alpha", "Carol Gamma"],
            "player_b_name": ["Zoe Zulu", "Zoe Zulu", "Dora Delta"],
            "actual_a_won": [1, 0, 1],
            "actual_match_date": pd.to_datetime([None, None, None]),
        }
    )


def result(links: pd.DataFrame, diagnostics: pd.DataFrame | None = None) -> LinkageResult:
    if diagnostics is None:
        diagnostics = pd.DataFrame(
            {
                "sackmann_position": [0, 1, 2],
                "reason_code": ["matched", "matched", "no_pair_candidate"],
                "reason": ["linked", "linked", "none"],
                "pair_candidate_count": [1, 1, 0],
                "selected_score": [98.0, 97.0, np.nan],
            }
        )
    return LinkageResult(links=links, diagnostics=diagnostics, summary={"matched_rows": len(links)})


class OddsAttachmentTests(unittest.TestCase):
    def test_orients_prices_by_identity_and_preserves_link_data(self):
        links = pd.DataFrame(
            {
                "sackmann_source_row": [10, 11],
                "actual_match_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "link_score": [98.0, 97.0],
                "date_offset_days": [1, 2],
                "odds_winner": ["Alpha A.", "Zulu Z."],
                "odds_loser": ["Zulu Z.", "Alpha A."],
                "odds_psw": [1.50, 1.70],
                "odds_psl": [2.80, 2.20],
                "odds_avgw": [1.60, 1.80],
                "odds_avgl": [2.60, 2.10],
                "odds_maxw": [1.55, 1.75],
                "odds_maxl": [2.90, 2.30],
                "odds_b365w": [1.48, 1.68],
                "odds_b365l": [2.75, 2.15],
            }
        )
        attached = attach_linked_odds(canonical(), result(links))

        first = attached.loc[attached["source_row"].eq(10)].iloc[0]
        self.assertEqual(first["odds_a"], 1.50)
        self.assertEqual(first["odds_b"], 2.80)
        self.assertEqual(first["max_odds_a"], 1.55)
        self.assertEqual(first["b365_odds_b"], 2.75)
        expected = (1 / 1.50) / ((1 / 1.50) + (1 / 2.80))
        self.assertAlmostEqual(first["p_market_a"], expected)
        self.assertAlmostEqual(first["fair_prob_a"], expected)
        self.assertEqual(first["market_odds_source"], "pinnacle")
        self.assertEqual(first["actual_match_date"], pd.Timestamp("2024-01-02"))
        self.assertEqual(first["link_score"], 98.0)
        self.assertEqual(first["linkage_reason_code"], "matched")

        second = attached.loc[attached["source_row"].eq(11)].iloc[0]
        self.assertEqual(second["odds_a"], 2.20)
        self.assertEqual(second["odds_b"], 1.70)
        self.assertEqual(second["max_odds_a"], 2.30)
        self.assertEqual(second["b365_odds_b"], 1.68)
        self.assertTrue(bool(second["link_pair_validated"]))
        self.assertTrue(bool(second["link_outcome_consistent"]))

        unmatched = attached.loc[attached["source_row"].eq(12)].iloc[0]
        self.assertFalse(bool(unmatched["has_odds_link"]))
        self.assertTrue(pd.isna(unmatched["odds_a"]))
        self.assertEqual(unmatched["linkage_reason_code"], "no_pair_candidate")
        self.assertEqual(attached.attrs["linkage_summary"], {"matched_rows": 2})

    def test_complete_average_pair_is_optional_fallback(self):
        links = pd.DataFrame(
            {
                "sackmann_source_row": [10],
                "actual_match_date": pd.to_datetime(["2024-01-02"]),
                "odds_winner": ["Alpha A."],
                "odds_loser": ["Zulu Z."],
                "odds_psw": [1.50],
                "odds_psl": [np.nan],
                "odds_avgw": [1.60],
                "odds_avgl": [2.60],
            }
        )
        with_fallback = attach_linked_odds(canonical(), result(links))
        row = with_fallback.loc[with_fallback["source_row"].eq(10)].iloc[0]
        self.assertEqual(row["market_odds_source"], "average")
        self.assertEqual(row["odds_a"], 1.60)
        self.assertEqual(row["odds_b"], 2.60)

        without_fallback = attach_linked_odds(
            canonical(), result(links), allow_average_fallback=False
        )
        row = without_fallback.loc[without_fallback["source_row"].eq(10)].iloc[0]
        self.assertTrue(pd.isna(row["market_odds_source"]))
        self.assertTrue(pd.isna(row["odds_a"]))

    def test_never_mixes_partial_pinnacle_and_average_pairs(self):
        links = pd.DataFrame(
            {
                "sackmann_source_row": [10],
                "actual_match_date": pd.to_datetime(["2024-01-02"]),
                "odds_winner": ["Alpha A."],
                "odds_loser": ["Zulu Z."],
                "odds_psw": [1.50],
                "odds_psl": [np.nan],
                "odds_avgw": [1.60],
                "odds_avgl": [np.nan],
            }
        )
        attached = attach_linked_odds(canonical(), result(links))
        row = attached.loc[attached["source_row"].eq(10)].iloc[0]
        self.assertTrue(pd.isna(row["odds_a"]))
        self.assertTrue(pd.isna(row["odds_b"]))
        self.assertTrue(pd.isna(row["p_market_a"]))

    def test_pair_mismatch_and_outcome_mismatch_fail_clearly(self):
        pair_mismatch = pd.DataFrame(
            {
                "sackmann_source_row": [10],
                "odds_winner": ["Wrong W."],
                "odds_loser": ["Zulu Z."],
            }
        )
        with self.assertRaisesRegex(ValueError, "player pair mismatch.*10"):
            attach_linked_odds(canonical(), result(pair_mismatch))

        outcome_mismatch = pd.DataFrame(
            {
                "sackmann_source_row": [10],
                "odds_winner": ["Zulu Z."],
                "odds_loser": ["Alpha A."],
            }
        )
        audited = attach_linked_odds(canonical(), result(outcome_mismatch))
        row = audited.loc[audited["source_row"].eq(10)].iloc[0]
        self.assertFalse(bool(row["link_outcome_consistent"]))
        with self.assertRaisesRegex(ValueError, "winner conflicts.*10"):
            attach_linked_odds(
                canonical(), result(outcome_mismatch), require_outcome_consistency=True
            )

    def test_duplicate_or_unknown_source_rows_fail(self):
        duplicate_matches = pd.concat([canonical(), canonical().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "canonical matches.source_row must be unique"):
            attach_linked_odds(duplicate_matches, result(pd.DataFrame()))

        duplicate_links = pd.DataFrame(
            {
                "sackmann_source_row": [10, 10],
                "odds_winner": ["Alpha A.", "Alpha A."],
                "odds_loser": ["Zulu Z.", "Zulu Z."],
            }
        )
        with self.assertRaisesRegex(ValueError, "linkage links.sackmann_source_row must be unique"):
            attach_linked_odds(canonical(), result(duplicate_links))

        unknown = pd.DataFrame(
            {
                "sackmann_source_row": [999],
                "odds_winner": ["Alpha A."],
                "odds_loser": ["Zulu Z."],
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown source_row values: 999"):
            attach_linked_odds(canonical(), result(unknown))

    def test_linked_only_and_empty_linkage(self):
        empty = LinkageResult(
            links=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            summary={"matched_rows": 0},
        )
        attached = attach_linked_odds(canonical(), empty)
        self.assertEqual(len(attached), 3)
        self.assertFalse(attached["has_odds_link"].any())

        one = pd.DataFrame(
            {
                "sackmann_source_row": [10],
                "odds_winner": ["Alpha A."],
                "odds_loser": ["Zulu Z."],
                "odds_psw": [1.5],
                "odds_psl": [2.8],
            }
        )
        attached = attach_linked_odds(canonical(), result(one), linked_only=True)
        self.assertEqual(attached["source_row"].tolist(), [10])


if __name__ == "__main__":
    unittest.main()
