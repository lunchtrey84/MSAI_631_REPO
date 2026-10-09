import pytest

from data_processing import (
    ProfileValidationError,
    normalize_text,
    prepare_profile_from_mapping,
    prepare_user_profile,
)


def test_normalize_text_removes_markup_and_extra_whitespace():
    assert normalize_text("  leadership  <b>and</b>  training\n") == (
        "leadership and training"
    )


def test_prepare_user_profile_labels_each_value():
    result = prepare_user_profile(
        education="Master's degree",
        skills=["leadership", "Python"],
        interests="technology and education",
        work_experience="Five years",
        career_goals="technical leadership",
    )

    assert "Education: Master's degree." in result
    assert "Skills: leadership, Python." in result
    assert "Interests: technology and education." in result
    assert "Experience: Five years." in result
    assert "Career goals: technical leadership." in result


def test_profile_requires_meaningful_matching_information():
    with pytest.raises(ProfileValidationError):
        prepare_user_profile(education="Bachelor's degree")


def test_mapping_ignores_unexpected_keys():
    result = prepare_profile_from_mapping(
        {
            "skills": "communication and analysis",
            "career_goals": "data analyst",
            "unexpected": "ignored",
        }
    )
    assert "unexpected" not in result
    assert "Skills: communication and analysis." in result
