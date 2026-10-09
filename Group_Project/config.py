"""Central configuration for the career recommendation application."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("CAREER_MODEL_DIR", PROJECT_ROOT / "Model"))

CAREER_DATA_PATH = MODEL_DIR / "onet_career_master.csv"
TFIDF_MODEL_PATH = MODEL_DIR / "career_tfidf_model.pkl"
TFIDF_MATRIX_PATH = MODEL_DIR / "career_tfidf_matrix.pkl"

APP_TITLE = "AI Career and Education Recommendation Assistant"
DEFAULT_TOP_K = int(os.getenv("CAREER_TOP_K", "5"))
HOST = os.getenv("CAREER_APP_HOST", "127.0.0.1")
PORT = int(os.getenv("CAREER_APP_PORT", "7860"))
SHARE_APP = os.getenv("CAREER_APP_SHARE", "false").lower() == "true"


def validate_paths() -> None:
    """Raise a helpful error when a required TF-IDF artifact is missing."""

    required = (CAREER_DATA_PATH, TFIDF_MODEL_PATH, TFIDF_MATRIX_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required model artifacts were not found: " + ", ".join(missing)
        )

