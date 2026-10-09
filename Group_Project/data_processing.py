"""Input validation and profile construction for the career recommender.

This module is intentionally independent of Gradio and the recommendation
model.  The user interface can call ``prepare_user_profile`` and pass the
returned text to either the TF-IDF model or a future sentence-transformer
model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields


MAX_FIELD_LENGTH = 1_000
MINIMUM_PROFILE_CHARACTERS = 15


class ProfileValidationError(ValueError):
    """Raised when user profile data is missing or cannot be processed."""


@dataclass(frozen=True)
class UserProfile:
    """Normalized information supplied by a user of the application."""

    education: str = ""
    skills: str = ""
    interests: str = ""
    work_experience: str = ""
    career_goals: str = ""
    additional_information: str = ""

    def as_model_text(self) -> str:
        """Return a labeled text profile suitable for semantic comparison."""

        labels = {
            "education": "Education",
            "skills": "Skills",
            "interests": "Interests",
            "work_experience": "Work experience",
            "career_goals": "Career goals",
            "additional_information": "Additional information",
        }
        parts = [
            f"{labels[item.name]}: {getattr(self, item.name)}."
            for item in fields(self)
            if getattr(self, item.name)
        ]
        return " ".join(parts)


def normalize_text(value: object, *, max_length: int = MAX_FIELD_LENGTH) -> str:
    """Convert a form value to clean, single-spaced text.

    HTML tags and control characters are removed because profile values may
    originate from browser form inputs. Text is truncated to keep requests
    predictable and protect the interface from accidentally huge inputs.
    """

    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length]


def _normalize_list_like(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        cleaned = [normalize_text(item) for item in value]
        return ", ".join(item for item in cleaned if item)
    return normalize_text(value)


def validate_profile(profile: UserProfile) -> None:
    """Validate that a profile contains enough information to rank careers."""

    if not profile.skills and not profile.interests and not profile.career_goals:
        raise ProfileValidationError(
            "Enter at least one skill, interest, or career goal."
        )

    if len(profile.as_model_text()) < MINIMUM_PROFILE_CHARACTERS:
        raise ProfileValidationError(
            "Please provide a little more information before requesting recommendations."
        )


def prepare_user_profile(
    education: object = "",
    skills: object = "",
    interests: object = "",
    work_experience: object = "",
    career_goals: object = "",
    additional_information: object = "",
) -> str:
    """Clean form values, validate them, and return model-ready text.

    This is the primary integration function for ``app.py``.
    """

    profile = UserProfile(
        education=_normalize_list_like(education),
        skills=_normalize_list_like(skills),
        interests=_normalize_list_like(interests),
        work_experience=_normalize_list_like(work_experience),
        career_goals=_normalize_list_like(career_goals),
        additional_information=_normalize_list_like(additional_information),
    )
    validate_profile(profile)
    return profile.as_model_text()


def prepare_profile_from_mapping(values: Mapping[str, object]) -> str:
    """Build model-ready text from a dictionary-like form submission."""

    allowed = {item.name for item in fields(UserProfile)}
    filtered = {key: value for key, value in values.items() if key in allowed}
    return prepare_user_profile(**filtered)

