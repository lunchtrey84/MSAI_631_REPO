# AI Career and Education Recommendation Assistant

This group project is a human-centered career recommendation application for
MSAI-631-A01. Users describe their education, skills, interests, professional
experience, and career goals. The application converts the information into a
structured text profile, compares that profile with O*NET occupational records,
and displays the five closest career matches with supporting information.

The system is intended to support career exploration and decision-making. It
does not predict whether a person will succeed in a particular occupation and
should not replace professional career counseling.

## Current status

The application supports two recommendation approaches. TF-IDF with cosine
similarity remains available as the baseline model, while
`all-MiniLM-L6-v2` is the primary semantic recommendation model.

Both models receive the same normalized user-profile text and return the five
occupations with the highest similarity scores. The project includes the
processed O*NET career dataset, TF-IDF artifacts, MiniLM embeddings, model
logic, data-processing utilities, configuration settings, dependencies, and
automated data-processing tests.

The MiniLM model captures semantic relationships between the user profile and
career descriptions instead of relying only on exact word overlap. If the
saved local MiniLM model directory is unavailable, the application loads
`sentence-transformers/all-MiniLM-L6-v2` through the Sentence Transformers
library.

## Application flow

1. The user enters education, skills, interests, experience, and career goals.
2. `data_processing.py` cleans, normalizes, and validates the input.
3. The application combines the responses into one labeled profile string.
4. The user selects MiniLM or TF-IDF as the recommendation model.
5. `recommend_careers()` transforms the profile using the selected model.
6. Cosine similarity compares the profile with the O*NET career records.
7. The occupations are ranked according to their similarity scores.
8. The interface displays the five highest-ranked career recommendations.

## Expected output

Each recommendation contains:

- O*NET-SOC code
- Career title
- Career description
- Typical education level
- Typical experience level
- Similarity score

Similarity scores represent the relationship between the user’s profile and
the occupational text. They are matching scores, not probabilities of career
success.

## Repository structure

```text
Group_Project/
├── Model/
│   ├── app.py                         # Integrated Gradio interface
│   ├── model.py                       # Recommendation and ranking logic
│   ├── model.ipynb                    # Model-development notebook
│   ├── onet_career_master.csv         # Processed O*NET career data
│   ├── career_tfidf_model.pkl         # Saved TF-IDF vectorizer
│   ├── career_tfidf_matrix.pkl        # Saved TF-IDF career matrix
│   ├── career_minilm_embeddings.npy   # Saved MiniLM career embeddings
│   └── career_minilm_model/           # Optional locally saved MiniLM model
├── data_processing.py                 # Input cleaning and profile creation
├── config.py                          # Shared paths and application settings
├── requirements.txt                   # Python dependencies
├── .env.example                       # Optional environment configuration
├── .gitignore
└── tests/
    └── test_data_processing.py
```

The repository may temporarily contain earlier interface copies while the
team completes integration. `Group_Project/Model/app.py` is the currently
integrated interface connected to `data_processing.py` and `model.py`. The
interface-development team will confirm the final application location before
submission.

## Setup

### Local Windows setup

Clone the repository and open the project directory:

```powershell
git clone https://github.com/lunchtrey84/MSAI_631_REPO.git
cd MSAI_631_REPO\Group_Project
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If environment overrides are needed, copy `.env.example` to `.env` and change
only the local values. Never commit credentials, access tokens, or API keys.

### Google Colab setup

```python
!git clone https://github.com/lunchtrey84/MSAI_631_REPO.git
%cd MSAI_631_REPO/Group_Project
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

The repository must be public, or Colab must be authenticated with GitHub
before cloning a private repository.

The final Colab launch command will be documented after the interface team
confirms the final location and Gradio sharing configuration for `app.py`.

## Recommendation-function contract

The model provides the following recommendation function:

```python
recommend_careers(
    user_profile,
    model_type="minilm",
    top_n=5,
)
```

The parameters are:

- `user_profile`: One normalized text string containing the user’s information.
- `model_type`: `"minilm"` for semantic matching or `"tfidf"` for the baseline.
- `top_n`: Number of recommendations to return.

The function returns a pandas DataFrame with these columns:

```text
O*NET-SOC Code
Title
Description
Education
Experience
Similarity Score
```

Example:

```python
from data_processing import prepare_user_profile
from Model.model import recommend_careers

profile_text = prepare_user_profile(
    education="Master's degree",
    skills="leadership, communication, planning, and critical thinking",
    interests="technology, education, and management",
    work_experience="More than ten years of professional experience",
    career_goals="Leadership, management, training, and technology",
)

results = recommend_careers(
    profile_text,
    model_type="minilm",
    top_n=5,
)

print(results)
```

## Interface integration

The interface uses `prepare_user_profile()` before calling the recommendation
model:

```python
from data_processing import ProfileValidationError, prepare_user_profile
from Model.model import recommend_careers

try:
    profile_text = prepare_user_profile(
        education=education,
        skills=skills,
        interests=interests,
        work_experience=work_experience,
        career_goals=career_goals,
        additional_information=additional_information,
    )

    results = recommend_careers(
        profile_text,
        model_type=model_type,
        top_n=5,
    )

except ProfileValidationError as error:
    # Display the validation message in the interface.
    print(str(error))
```

The same normalized profile is used for TF-IDF and MiniLM, allowing the user
to compare the two recommendation approaches.

## Testing

Run the automated data-processing tests from the `Group_Project` directory:

```powershell
python -m pytest -q
```

Suggested end-to-end test profiles include:

1. A complete profile with education, skills, interests, experience, and goals.
2. A profile emphasizing leadership, management, training, and technology.
3. A profile from a different career area to verify that rankings change.
4. Empty input, which should produce an understandable validation message.
5. Very long input, which should be normalized and limited safely.
6. The same profile tested with MiniLM and TF-IDF for comparison.

Before submission, the team should also verify:

- Both model selections return five recommendations.
- The returned columns match the application contract.
- Empty and unsupported inputs produce understandable feedback.
- The application displays similarity scores as matching scores.
- The Gradio interface launches without import or path errors.

## Deployment

Gradio supports local demonstrations and can also be configured for Google
Colab or Hugging Face Spaces. A deployment must include the Python modules,
`requirements.txt`, the processed O*NET dataset, and the required model
artifacts.

Future credentials or access tokens must be stored using environment variables
or repository secrets. Credentials must never be written directly into source
code or committed to GitHub.

## Data and model limitations

- TF-IDF depends heavily on word overlap and may miss related concepts expressed
  with different vocabulary.
- MiniLM improves semantic matching, but its recommendations still depend on the
  quality and completeness of the user profile.
- O*NET descriptions represent occupations broadly and cannot capture every
  personal preference, local opportunity, salary expectation, or employer
  requirement.
- Missing or incomplete fields in the source data can reduce the quality of
  individual recommendations.
- Similarity scores indicate model-based textual similarity, not certainty,
  qualification, employability, or probability of success.
- The current evaluation is based on sample profiles and model comparison, not
  a large formal user study.
- Career recommendations can affect important decisions, so the interface
  should explain results, protect user information, avoid discriminatory
  attributes, and allow users to revise their inputs.
- Users should remain responsible for their final education and career
  decisions.

## Team responsibilities

- **Trey:** Model training, recommendation and ranking logic, evaluation, and
  model documentation.
- **Avinash:** Gradio/Streamlit interface, validation display, output logic,
  usability testing, and interface-development lessons.
- **Surya:** Data processing, configuration and environment files,
  dependencies, repository structure, README and setup documentation, and
  integration support.

## Technology

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- Cosine similarity
- joblib
- Gradio
- Sentence Transformers
- `all-MiniLM-L6-v2`
- O*NET 31.0 occupational data
- Git and GitHub version control
- pytest

## Contribution workflow

Create a branch before changing shared files:

```powershell
git switch -c surya-data-processing
git add Group_Project
git commit -m "Update project documentation for MiniLM integration"
git push -u origin surya-data-processing
```

Open a pull request for team review. Do not modify another team member’s
assigned source files without first coordinating with that team member.
