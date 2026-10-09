# AI Career and Education Recommendation Assistant

This group project is a human-centered career recommendation application for
MSAI-631-A01. A user describes their education, skills, interests, work
experience, and goals. The application converts that information into a
profile, compares it with O*NET career records, and presents the five closest
career matches with understandable supporting information.

## Current status

The current baseline uses TF-IDF features and cosine similarity. Trey trained
the model on 1,016 O*NET 31.0 career records and generated the three artifacts
stored in `Model/`. An all-MiniLM-L6-v2 semantic model is planned and can be
added without changing the input-processing interface.

## Application flow

1. The user enters profile information in the interface.
2. `data_processing.py` cleans and validates the values.
3. The application creates one labeled text profile.
4. The selected model transforms the profile.
5. Similarity scores rank the O*NET occupations.
6. The interface displays the top five careers.

## Expected output

Each recommendation should display:

- Career title
- O*NET-SOC code
- Career description
- Recommended education level
- Recommended experience level
- Similarity score

The results are decision-support information, not a guarantee of employment or
a substitute for professional career counseling. Users should remain in
control and review multiple options before making education or career choices.

## Repository structure

```text
Group_Project/
├── app.py                         # Interface (Avinash)
├── model.py                       # Ranking/model functions (Trey)
├── data_processing.py             # Input cleaning/profile creation (Surya)
├── config.py                      # Shared paths and application settings
├── requirements.txt               # Reproducible Python dependencies
├── .env.example                   # Optional local configuration template
├── .gitignore
├── Model/
│   ├── model.ipynb
│   ├── onet_career_master.csv
│   ├── career_tfidf_model.pkl
│   └── career_tfidf_matrix.pkl
└── tests/
    └── test_data_processing.py
```

`app.py` and the final reusable `model.py` will be added by the assigned team
members during integration.

## Setup

### Local Windows setup

```powershell
git clone https://github.com/lunchtrey84/MSAI_631_REPO.git
cd MSAI_631_REPO\Group_Project
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If environment overrides are needed, copy `.env.example` to `.env` and change
only the local values. Never commit credentials or API keys.

### Google Colab setup

```python
!git clone https://github.com/lunchtrey84/MSAI_631_REPO.git
%cd MSAI_631_REPO/Group_Project
!pip install -r requirements.txt
```

The repository must be public or Colab must be authenticated before a private
repository can be cloned.

## Interface integration contract

`app.py` should use `prepare_user_profile` before calling the model:

```python
from data_processing import ProfileValidationError, prepare_user_profile

try:
    profile_text = prepare_user_profile(
        education=education,
        skills=skills,
        interests=interests,
        work_experience=work_experience,
        career_goals=career_goals,
        additional_information=additional_information,
    )
except ProfileValidationError as error:
    # Display str(error) in the interface.
    ...

# Pass profile_text to the recommendation function supplied by model.py.
```

This contract works for TF-IDF and MiniLM because both models receive the same
normalized text. The final recommendation-function name and returned columns
will be confirmed when `model.py` is added.

## Running and testing

After `app.py` is added, run:

```powershell
python app.py
```

Open the local Gradio URL printed in the terminal, normally
`http://127.0.0.1:7860`.

Run the automated input-processing tests with:

```powershell
python -m pytest -q
```

Suggested end-to-end test profiles:

1. A complete profile with education, skills, interests, experience, and goals.
2. A short but valid profile containing skills and a career goal.
3. Empty input, which should produce a clear validation message.
4. Very long or pasted input, which should be normalized and limited safely.
5. Different career areas to verify that rankings change with user preferences.

## Deployment

Gradio is suitable for local execution, Google Colab demonstrations, and a
future Hugging Face Spaces deployment. For a Space, place `app.py`, the Python
modules, `requirements.txt`, and required model artifacts in the repository.
Use repository secrets for any future API credentials rather than writing them
in source files.

## Data and model limitations

- TF-IDF relies on word overlap and may miss meaning expressed with different
  vocabulary.
- MiniLM should improve semantic matching, but it must be evaluated against the
  TF-IDF baseline before replacing it.
- O*NET descriptions represent occupations broadly and cannot capture every
  individual preference, local opportunity, salary, or hiring requirement.
- Missing fields in the source data can make some recommendations less complete.
- Similarity scores indicate textual/model similarity, not certainty or
  suitability.
- Career recommendations can influence significant decisions, so the interface
  should explain results, avoid discriminatory attributes, protect user data,
  and allow users to revise their inputs.

## Team responsibilities

- **Trey:** model training, recommendation/ranking logic, evaluation, and model
  documentation.
- **Avinash:** Gradio/Streamlit interface, validation display, and output logic.
- **Surya:** data processing, configuration/environment files, requirements,
  repository structure, README/setup, and integration support.

## Technology

- Python
- pandas and NumPy
- scikit-learn TF-IDF and cosine similarity
- joblib model persistence
- Gradio user interface
- O*NET 31.0 occupational data
- Git and GitHub version control
- Planned: sentence-transformers with `all-MiniLM-L6-v2`

## Contribution workflow

Create a branch before changing shared files:

```powershell
git switch -c surya-data-processing
git add Group_Project
git commit -m "Add data processing setup and project documentation"
git push -u origin surya-data-processing
```

Open a pull request for team review. Do not push directly to `main` unless the
repository owner requests it.

