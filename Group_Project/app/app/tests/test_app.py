import unittest
import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app import FIELDS, EMPTY_DETAILS, clear_profile, details_html, load_model, rank_careers, recommend, select_career, recommend_pages, edit_profile, career_choices, choose_career


class CareerInterfaceTests(unittest.TestCase):
    profile = ["Bachelor's degree", "Python Excel statistics", "data analysis", "2 years reporting", "analytics", ""]

    def test_ranking_matches_saved_model(self):
        matches = rank_careers(*self.profile)
        data, vectorizer, matrix = load_model()
        text = "\n".join(f"{label}: {value}." for label, value in zip(FIELDS, self.profile) if value)
        scores = cosine_similarity(vectorizer.transform([text]), matrix).ravel()
        indices = np.argsort(-scores, kind="stable")[:5]
        self.assertEqual([r["O*NET-SOC Code"] for r in matches], data.iloc[indices]["O*NET-SOC Code"].tolist())
        np.testing.assert_allclose([r["Similarity score"] for r in matches], scores[indices])
        self.assertEqual(len(matches), 5)

    def test_invalid_search_clears_results(self):
        for profile in [["Bachelor's degree", "", "", "", "", ""], ["", "zzzxxyyqqvv", "", "", "", ""], ["", "x" * 4001, "", "", "", ""]]:
            state, rows, details, status = recommend(*profile)
            self.assertEqual(state, [])
            self.assertEqual(rows, [])
            self.assertEqual(details, EMPTY_DETAILS)
            self.assertTrue(status)

    def test_details_escape_user_visible_data_and_missing_values(self):
        markup = details_html({"Title": "<script>bad</script>", "O*NET-SOC Code": "15-2051.00"})
        self.assertNotIn("<script>", markup)
        self.assertIn("Not available in the dataset", markup)

    def test_profiles_produce_different_results(self):
        other = ["", "nursing patient care medicine", "healthcare", "hospital", "nursing", ""]
        self.assertNotEqual(rank_careers(*self.profile)[0]["Title"], rank_careers(*other)[0]["Title"])

    def test_clear_removes_profile_and_results(self):
        result = clear_profile()
        self.assertEqual(result[1:6], ("", "", "", "", ""))
        self.assertEqual(result[6:8], ([], []))

    def test_each_result_row_selects_its_own_details(self):
        matches = rank_careers(*self.profile)
        for index, match in enumerate(matches):
            event = gr.SelectData(None, {"index": [index, 1], "value": match["Title"], "selected": True})
            self.assertIn(match["O*NET-SOC Code"], select_career(matches, event))

    def test_navigation_requires_successful_search(self):
        valid = recommend_pages(*self.profile)
        self.assertFalse(valid[4]["visible"])
        self.assertTrue(valid[5]["visible"])
        invalid = recommend_pages(None, "", "", "", "", "")
        self.assertTrue(invalid[4]["visible"])
        self.assertFalse(invalid[5]["visible"])
        self.assertTrue(invalid[6]["visible"])
        back = edit_profile()
        self.assertTrue(back[0]["visible"])
        self.assertFalse(back[1]["visible"])

    def test_accessible_picker_matches_each_recommendation(self):
        matches = rank_careers(*self.profile)
        choices = career_choices(matches)
        self.assertEqual(len(choices["choices"]), 5)
        for index, match in enumerate(matches):
            self.assertIn(match["O*NET-SOC Code"], choose_career(str(index), matches))
        self.assertEqual(career_choices([])["choices"], [])
        self.assertEqual(choose_career(None, []), EMPTY_DETAILS)


if __name__ == "__main__":
    unittest.main()
