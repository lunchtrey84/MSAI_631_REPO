# Career Recommendation Assistant — Avinash's interface

Gradio interface for the existing O*NET TF-IDF career recommender. It collects education, skills, interests, experience, goals, and optional information; displays the five highest cosine-similarity matches; and shows career details when a result row is selected.

The app shows one screen at a time. A valid search moves from the profile screen to the results screen. Edit profile returns to the form and preserves the entered values. Invalid searches keep the form visible and show a correction message. Both screens share the same local URL.

## Run locally

Use Python 3.11 or newer. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. On Windows, activate with `.venv\Scripts\activate`. Set `PORT` to change the port. Model files load relative to `app.py`, so launching from another directory also works. No OpenAI API key or paid service is required. Profiles are not written to disk by this application. Clear profile clears the input fields and current results.

`requirements-lock.txt` records the complete tested Python 3.13 environment. The main requirements pin scikit-learn to 1.6.1, matching the version used to save Trey's vectorizer.

## Model provenance and integration

The files in `Model/` are unchanged artifacts supplied by Trey Williams at commit `f35b2abdb4d6581814f69426554ff855174036e5`:

https://github.com/lunchtrey84/MSAI_631_REPO/tree/f35b2abdb4d6581814f69426554ff855174036e5/Group_Project/Model

- `onet_career_master.csv`: processed O*NET career records.
- `career_tfidf_model.pkl`: fitted TF-IDF vectorizer.
- `career_tfidf_matrix.pkl`: corresponding career matrix.

The integration follows the transformation and cosine-similarity ranking in Trey's `model.ipynb`. Avinash's contribution is the interface, validation, model-loading adapter, and display logic. No training or raw Colab data uploads are needed. Keep the CSV and matrix in their original corresponding order. Only load trusted joblib files.

## Limitations

This supplied model uses TF-IDF only; MiniLM and FLAN-T5 are not included. Interests and goals are matched as text, rather than scored through dedicated interest attributes. Similarity is not a probability, qualification assessment, or career guarantee. Missing education/experience values are explicitly labelled unavailable. Education and experience are typical categories, not hard prerequisites. The form requires at least some descriptive background and rejects inputs with no vocabulary overlap. Generic descriptions can still produce weak matches.

## Interface checks

Run `python -m unittest discover -s tests -v` for actual-model ranking and interface callback checks. Use the example profiles, select each result row, revise the profile, and clear it to review the user flow. Participant usability testing has not been performed.

## Group handoff

Copy this folder's tracked contents into the group's `Group_Project` folder, preserving the `Model/` subfolder. Share the instructor-accessible repository and submit its source archive according to Week 7 instructions. Exclude `.venv` and cache directories from the archive. See `DESIGN_NOTES.md` for Avinash's design contribution; it needs to be combined with the other members' sections for the full group design document.
