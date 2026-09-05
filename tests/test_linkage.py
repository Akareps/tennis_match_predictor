import unittest

import pandas as pd

from tennis_experiments.linkage import (
    LinkageConfig,
    link_matches,
    normalize_player_name,
    normalize_round,
    normalize_tournament,
    unordered_player_pair,
)


class NormalizationTests(unittest.TestCase):
    def test_name_formats_diacritics_and_compound_surnames(self):
        self.assertEqual(
            normalize_player_name("Novak Djokovic", source="sackmann"),
            normalize_player_name("Djokovic N.", source="odds"),
        )
        self.assertEqual(
            normalize_player_name("Juan Martín del Potro", source="sackmann"),
            normalize_player_name("del Potro J.", source="odds"),
        )
        self.assertEqual(
            normalize_player_name("J.J. Wolf", source="sackmann"),
            normalize_player_name("Wolf J.J.", source="odds"),
        )

    def test_pair_is_unordered(self):
        forward = unordered_player_pair(
            "Novak Djokovic", "Carlos Alcaraz", source="sackmann"
        )
        reverse = unordered_player_pair(
            "Carlos Alcaraz", "Novak Djokovic", source="sackmann"
        )
        self.assertEqual(forward, reverse)

    def test_tournament_and_contextual_round_normalization(self):
        self.assertEqual(normalize_tournament("Brisbane International"), "brisbane")
        self.assertEqual(normalize_tournament("ATP Brisbane Open"), "brisbane")
        self.assertEqual(normalize_round("The Final"), "F")
        self.assertEqual(normalize_round("1st Round", max_numbered_round=2), "R32")
        self.assertEqual(normalize_round("3rd Round", draw_size=128), "R32")
        self.assertEqual(normalize_round("1st Round"), "ROUND_1")


class LinkageTests(unittest.TestCase):
    @staticmethod
    def _sackmann_row(**overrides):
        row = {
            "tourney_date": 20240101,
            "tourney_name": "Paris Masters",
            "surface": "Hard",
            "round": "R32",
            "draw_size": 32,
            "winner_name": "Novak Djokovic",
            "loser_name": "Carlos Alcaraz",
            "actual_a_won": 1,
        }
        row.update(overrides)
        return row

    @staticmethod
    def _odds_row(**overrides):
        row = {
            "Date": "2024-01-03",
            "Tournament": "Paris",
            "Surface": "Hard",
            "Round": "1st Round",
            "Winner": "Alcaraz C.",
            "Loser": "Djokovic N.",
            "PSW": 2.25,
            "PSL": 1.72,
        }
        row.update(overrides)
        return row

    def test_best_contextual_candidate_supplies_actual_date(self):
        sackmann = pd.DataFrame([self._sackmann_row()])
        odds = pd.DataFrame(
            [
                self._odds_row(
                    Date="2024-01-01",
                    Tournament="Adelaide International",
                    Round="1st Round",
                ),
                self._odds_row(Date="2024-01-03", Tournament="Paris Masters"),
            ],
            index=[41, 73],
        )

        result = link_matches(sackmann, odds)

        self.assertEqual(result.summary["matched_rows"], 1)
        self.assertEqual(result.links.loc[0, "odds_index"], 73)
        self.assertEqual(result.links.loc[0, "actual_match_date"], pd.Timestamp("2024-01-03"))
        self.assertEqual(result.links.loc[0, "date_offset_days"], 2)
        self.assertEqual(result.links.loc[0, "chronology_order"], 0)
        self.assertEqual(result.links.loc[0, "odds_psw"], 2.25)
        self.assertEqual(result.diagnostics.loc[0, "reason_code"], "matched")

    def test_winner_orientation_and_price_do_not_choose_candidate(self):
        sackmann = pd.DataFrame([self._sackmann_row()])
        # Candidate 0 has the opposite winner orientation but the right event.
        # Candidate 1 agrees with Sackmann's winner orientation but is a worse
        # contextual match.  Odds are also made deliberately extreme.
        odds = pd.DataFrame(
            [
                self._odds_row(
                    Winner="Alcaraz C.",
                    Loser="Djokovic N.",
                    Tournament="Paris Masters",
                    PSW=99.0,
                    PSL=1.01,
                ),
                self._odds_row(
                    Winner="Djokovic N.",
                    Loser="Alcaraz C.",
                    Tournament="Adelaide International",
                    PSW=1.01,
                    PSL=99.0,
                ),
            ]
        )

        linked = link_matches(sackmann, odds).links
        self.assertEqual(linked.loc[0, "odds_index"], 0)

        swapped = odds.copy()
        swapped[["Winner", "Loser"]] = swapped[["Loser", "Winner"]]
        linked_after_swap = link_matches(sackmann, swapped).links
        self.assertEqual(linked_after_swap.loc[0, "odds_index"], 0)
        self.assertEqual(linked_after_swap.loc[0, "link_score"], linked.loc[0, "link_score"])

    def test_assignment_is_one_to_one_and_reports_competition(self):
        sackmann = pd.DataFrame(
            [
                self._sackmann_row(),
                self._sackmann_row(actual_a_won=0),
            ],
            index=[100, 200],
        )
        odds = pd.DataFrame([self._odds_row()], index=[900])

        result = link_matches(sackmann, odds)

        self.assertEqual(len(result.links), 1)
        self.assertEqual(result.links["odds_index"].nunique(), 1)
        self.assertEqual(result.links.loc[0, "sackmann_index"], 100)
        reasons = dict(zip(result.diagnostics["sackmann_index"], result.diagnostics["reason_code"]))
        self.assertEqual(reasons[100], "matched")
        self.assertEqual(reasons[200], "lost_one_to_one_assignment")

    def test_two_rounds_are_assigned_to_the_corresponding_unique_rows(self):
        sackmann = pd.DataFrame(
            [
                self._sackmann_row(round="R32"),
                self._sackmann_row(round="R16"),
            ]
        )
        odds = pd.DataFrame(
            [
                self._odds_row(Date="2024-01-02", Round="1st Round"),
                self._odds_row(Date="2024-01-04", Round="2nd Round"),
            ]
        )

        result = link_matches(sackmann, odds)

        self.assertEqual(len(result.links), 2)
        assignments = dict(zip(result.links["sackmann_round"], result.links["odds_round"]))
        self.assertEqual(assignments, {"R32": "R32", "R16": "R16"})
        self.assertEqual(result.links["odds_index"].nunique(), 2)

    def test_reason_codes_distinguish_no_pair_and_outside_window(self):
        sackmann = pd.DataFrame(
            [
                self._sackmann_row(),
                self._sackmann_row(
                    winner_name="Jannik Sinner",
                    loser_name="Daniil Medvedev",
                ),
            ]
        )
        odds = pd.DataFrame([self._odds_row(Date="2024-03-01")])

        result = link_matches(sackmann, odds)

        self.assertEqual(
            result.diagnostics["reason_code"].tolist(),
            ["outside_date_window", "no_pair_candidate"],
        )
        self.assertEqual(result.summary["matched_rows"], 0)
        self.assertEqual(result.summary["reason_counts"]["outside_date_window"], 1)

    def test_minimal_optional_schema_and_canonical_player_columns(self):
        canonical = pd.DataFrame(
            {
                "tournament_date": [pd.Timestamp("2024-01-01")],
                "player_a_name": ["Carlos Alcaraz"],
                "player_b_name": ["Novak Djokovic"],
                "actual_a_won": [0],
            }
        )
        odds = pd.DataFrame(
            {
                "Date": ["2024-01-02"],
                "Winner": ["Djokovic N."],
                "Loser": ["Alcaraz C."],
            }
        )

        result = link_matches(canonical, odds)

        self.assertEqual(len(result.links), 1)
        self.assertEqual(result.links.loc[0, "actual_match_date"], pd.Timestamp("2024-01-02"))

    def test_config_can_require_stronger_context(self):
        sackmann = pd.DataFrame([self._sackmann_row()])
        odds = pd.DataFrame(
            [self._odds_row(Tournament="Adelaide", Surface="Clay", Round="The Final")]
        )

        result = link_matches(sackmann, odds, LinkageConfig(min_score=80.0))

        self.assertEqual(len(result.links), 0)
        self.assertEqual(result.diagnostics.loc[0, "reason_code"], "below_min_score")


if __name__ == "__main__":
    unittest.main()
