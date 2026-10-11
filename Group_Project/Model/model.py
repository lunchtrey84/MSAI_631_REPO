
import os
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAREER_DATA_PATH = os.path.join(
    BASE_DIR,
    "onet_career_master.csv"
)

TFIDF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "career_tfidf_model.pkl"
)

TFIDF_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "career_tfidf_matrix.pkl"
)

MINILM_EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "career_minilm_embeddings.npy"
)

MINILM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "career_minilm_model"
)


# ------------------------------------------------------------
# LOAD SAVED DATA AND MODELS
# ------------------------------------------------------------

career_df = pd.read_csv(CAREER_DATA_PATH)

career_df["combined_text"] = (
    career_df["combined_text"]
    .fillna("")
    .astype(str)
)

tfidf_model = joblib.load(TFIDF_MODEL_PATH)
tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

if os.path.isdir(MINILM_MODEL_PATH):
    minilm_model = SentenceTransformer(
        MINILM_MODEL_PATH
    )
else:
    minilm_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

career_embeddings = np.load(
    MINILM_EMBEDDINGS_PATH
)


# ------------------------------------------------------------
# RECOMMENDATION FUNCTION
# ------------------------------------------------------------

def recommend_careers(
    user_profile,
    model_type="minilm",
    top_n=5
):
    """
    Return ranked career recommendations.

    Parameters
    ----------
    user_profile : str
        Combined user profile text.

    model_type : str
        "minilm" or "tfidf"

    top_n : int
        Number of recommendations to return.

    Returns
    -------
    pandas.DataFrame
    """

    if not isinstance(user_profile, str):
        raise TypeError(
            "user_profile must be a string."
        )

    user_profile = user_profile.strip()

    if not user_profile:
        raise ValueError(
            "user_profile cannot be empty."
        )

    # MiniLM semantic model
    if model_type.lower() == "minilm":

        user_embedding = minilm_model.encode(
            [user_profile],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            user_embedding,
            career_embeddings
        ).flatten()

    # TF-IDF baseline model
    elif model_type.lower() == "tfidf":

        user_vector = tfidf_model.transform(
            [user_profile]
        )

        scores = cosine_similarity(
            user_vector,
            tfidf_matrix
        ).flatten()

    else:
        raise ValueError(
            "model_type must be 'minilm' or 'tfidf'."
        )

    results = career_df[
        [
            "O*NET-SOC Code",
            "Title",
            "Description",
            "Education",
            "Experience"
        ]
    ].copy()

    results["Similarity Score"] = scores

    results = (
        results
        .sort_values(
            "Similarity Score",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return results
