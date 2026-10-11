# AI Career and Education Recommendation Assistant

This MSAI-631-A01 group project is a human-centered application that helps a
user explore occupations matching their education, skills, interests, work
experience, and career goals. It compares a normalized user profile with
1,016 O*NET 31.0 occupational profiles and returns five ranked matches.

The recommendations support exploration and decision-making. They are not
employment guarantees, qualification assessments, or substitutes for
professional career counseling.

## Models

The interface allows the user to select either model:

- **MiniLM (default):** `sentence-transformers/all-MiniLM-L6-v2` creates a
  semantic representation of the user profile. Cosine similarity compares it
  with the saved 384-dimensional career embeddings.
- **TF-IDF (baseline):** a saved scikit-learn TF-IDF vectorizer transforms the
  profile. Cosine similarity compares it with the saved career-term matrix.

The shared function contract is:

```python
recommend_careers(user_profile, model_type="minilm", top_n=5)
```

It returns a Pandas DataFrame containing `O*NET-SOC Code`, `Title`,
`Description`, `Education`, `Experience`, and `Similarity Score`.

## Application flow

1. The user enters education, skills, interests, experience, and goals.
2. `data_processing.py` removes markup/control characters, normalizes spacing,
   validates the profile, and creates one labeled text string.
3. `Model/model.py` applies the selected model.
4. Cosine similarity ranks the O*NET occupational profiles.
5. `app.py` displays the top five matches and supporting details.
6. The user can compare results, revise the profile, clear it, or switch models.

## Repository structure

```text
Group_Project/
├── app.py
├── data_processing.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── COLAB_SETUP.md
├── Model/
│   ├── model.py
│   ├── model.ipynb
│   ├── onet_career_master.csv
│   ├── career_tfidf_model.pkl
│   ├── career_tfidf_matrix.pkl
│   └── career_minilm_embeddings.npy
└── tests/
    └── test_data_processing.py
```

## Local setup

Python 3.11 or newer is recommended.

```powershell
git clone https://github.com/lunchtrey84/MSAI_631_REPO.git
cd MSAI_631_REPO\Group_Project
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860` after the application starts.

On the first MiniLM run, Sentence Transformers downloads
`sentence-transformers/all-MiniLM-L6-v2` if the optional local
`Model/career_minilm_model/` directory is absent. TF-IDF can run without that
download after the dependencies and saved artifacts are installed.

## Google Colab

See `COLAB_SETUP.md`. A Colab demonstration normally requires a public Gradio
link, so set:

```python
%env CAREER_APP_SHARE=true
!python app.py
```

Do not put passwords, tokens, or personal identifiers into the repository.

## Testing

Run all automated tests from the project directory:

```powershell
python -m pytest -q
```

Then run the interface and manually verify:

1. A complete profile using MiniLM.
2. The same profile using TF-IDF.
3. Empty or incomplete input produces an understandable correction message.
4. Different profiles change the recommendations.
5. Every result can be selected and displays the correct O*NET details.
6. Edit profile, clear profile, light mode, dark mode, and keyboard navigation.

## Configuration

`.env.example` documents the optional settings:

- `CAREER_MODEL_DIR`
- `CAREER_TOP_K`
- `CAREER_APP_HOST`
- `CAREER_APP_PORT`
- `CAREER_APP_SHARE`

The application does not require an API key. User profiles are processed in
memory and are not written to disk by this project.

## Limitations and ethics

- TF-IDF depends heavily on shared vocabulary and may miss related meanings.
- MiniLM improves semantic matching but does not prove career suitability.
- O*NET represents occupations broadly and cannot capture every employer,
  location, salary, accessibility need, or personal circumstance.
- Similarity scores are matching values, not probabilities of success.
- Saved joblib files should be loaded only from trusted sources.
- The interface should avoid unnecessary personal or protected information,
  explain its limitations, and keep the user in control of decisions.

## Team responsibilities

- **Trey Williams:** model training, recommendation/ranking logic, model
  comparison, evaluation, and model documentation.
- **Avinash Sai Krishna Kanth Penta:** Gradio interface, display logic,
  interface validation, accessibility, and usability testing.
- **Surya Teja Gogineni:** data processing, configuration/environment files,
  dependencies, repository structure, setup documentation, and integration
  support.

## Credits

Occupational information is derived from the O*NET 31.0 database. The semantic
model is `sentence-transformers/all-MiniLM-L6-v2`. The team should cite O*NET,
Sentence Transformers, scikit-learn, Gradio, and all other sources in the final
design document.
