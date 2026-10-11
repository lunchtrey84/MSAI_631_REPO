"""Tests for the integrated career-recommendation interface."""

import gradio as gr

from app import (
    EMPTY_DETAILS,
    career_choices,
    choose_career,
    clear_profile,
    details_html,
    edit_profile,
    rank_careers,
    recommend,
    recommend_pages,
    select_career,
)


TFIDF_PROFILE = [
    "Bachelor's degree",
    "Python, Excel, statistics",
    "Data analysis and problem solving",
    "2 years in business reporting",
    "Explore analytics roles",
    "",
    "tfidf",
]

MINILM_PROFILE = [
    "Master's degree",
    "Leadership, communication, planning, training, critical thinking",
    "Technology, education, and management",
    "More than ten years of professional experience",
    "Leadership, management, training, and technology",
    "",
    "minilm",
]


def test_tfidf_returns_five_recommendations():
    matches = rank_careers(*TFIDF_PROFILE)

    assert len(matches) == 5

    required_columns = {
        "O*NET-SOC Code",
        "Title",
        "Description",
        "Education",
        "Experience",
        "Similarity score",
    }

    assert required_columns.issubset(matches[0].keys())


def test_minilm_returns_five_recommendations():
    matches = rank_careers(*MINILM_PROFILE)

    assert len(matches) == 5
    assert all(match["Similarity score"] >= 0 for match in matches)


def test_different_profiles_produce_different_results():
    analytics_matches = rank_careers(*TFIDF_PROFILE)

    healthcare_profile = [
        "Bachelor's degree",
        "Nursing, patient care, medicine, communication",
        "Healthcare and helping patients",
        "Experience working in a hospital",
        "Explore nursing and healthcare careers",
        "",
        "tfidf",
    ]

    healthcare_matches = rank_careers(*healthcare_profile)

    assert analytics_matches[0]["Title"] != healthcare_matches[0]["Title"]


def test_invalid_search_clears_previous_results():
    invalid_profiles = [
        [None, "", "", "", "", "", "tfidf"],
        ["Bachelor's degree", "", "", "", "", "", "minilm"],
        [None, "x" * 4001, "", "", "", "", "tfidf"],
    ]

    for profile in invalid_profiles:
        state, rows, details, status = recommend(*profile)

        assert state == []
        assert rows == []
        assert details == EMPTY_DETAILS
        assert status


def test_details_escape_html_and_handle_missing_values():
    markup = details_html(
        {
            "Title": "<script>bad</script>",
            "O*NET-SOC Code": "15-2051.00",
        }
    )

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "Not available in the dataset" in markup


def test_clear_profile_removes_inputs_and_results():
    result = clear_profile()

    assert result[0] is None
    assert result[1:6] == ("", "", "", "", "")
    assert result[6] == "minilm"
    assert result[7:9] == ([], [])
    assert result[9] == EMPTY_DETAILS


def test_each_result_can_display_its_details():
    matches = rank_careers(*TFIDF_PROFILE)

    for index, match in enumerate(matches):
        event = gr.SelectData(
            None,
            {
                "index": [index, 1],
                "value": match["Title"],
                "selected": True,
            },
        )

        selected_details = select_career(matches, event)

        assert match["O*NET-SOC Code"] in selected_details


def test_navigation_requires_successful_search():
    valid = recommend_pages(*TFIDF_PROFILE)

    assert valid[4]["visible"] is False
    assert valid[5]["visible"] is True

    invalid = recommend_pages(
        None,
        "",
        "",
        "",
        "",
        "",
        "tfidf",
    )

    assert invalid[4]["visible"] is True
    assert invalid[5]["visible"] is False
    assert invalid[6]["visible"] is True

    back = edit_profile()

    assert back[0]["visible"] is True
    assert back[1]["visible"] is False


def test_career_picker_matches_recommendations():
    matches = rank_careers(*TFIDF_PROFILE)
    choices = career_choices(matches)

    assert len(choices["choices"]) == 5

    for index, match in enumerate(matches):
        details = choose_career(str(index), matches)
        assert match["O*NET-SOC Code"] in details

    assert career_choices([])["choices"] == []
    assert choose_career(None, []) == EMPTY_DETAILS
