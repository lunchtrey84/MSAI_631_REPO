"""Avinash's Gradio interface for Trey Williams's O*NET TF-IDF model."""
from functools import lru_cache
from html import escape
from pathlib import Path
import logging
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
FIELDS = ["Education", "Skills", "Interests", "Experience", "Career goals", "Additional information"]
HEADERS = ["Rank", "Career", "Similarity score"]
EMPTY_DETAILS = '<div class="empty-details">Your career details will appear here.<br>Select a match to explore its description, education, and experience.</div>'


@lru_cache(maxsize=1)
def load_model():
    folder = ROOT / "Model"
    names = ["onet_career_master.csv", "career_tfidf_model.pkl", "career_tfidf_matrix.pkl"]
    if any(not (folder / name).is_file() for name in names):
        raise RuntimeError("The career model files are missing from the Model folder.")
    careers = pd.read_csv(folder / names[0]).fillna("")
    required = {"O*NET-SOC Code", "Title", "Description", "Education", "Experience"}
    if not required.issubset(careers.columns):
        raise RuntimeError("The career dataset does not contain the expected columns.")
    vectorizer = joblib.load(folder / names[1])
    matrix = joblib.load(folder / names[2])
    if matrix.shape[0] != len(careers) or matrix.shape[1] != len(vectorizer.get_feature_names_out()):
        raise RuntimeError("The saved model and career dataset do not match.")
    return careers, vectorizer, matrix


def rank_careers(*values):
    values = [str(value or "").strip() for value in values]
    if len(values) != len(FIELDS):
        raise ValueError("The profile needs all six input fields.")
    if not any(values[1:]):
        raise ValueError("Add your skills, interests, experience, or career goals to find matches.")
    if any(len(value) > 4000 for value in values):
        raise ValueError("Keep each field under 4,000 characters.")
    careers, vectorizer, matrix = load_model()
    # Check the user's own words, before adding field labels which could create false matches.
    if vectorizer.transform([" ".join(values)]).nnz == 0:
        raise ValueError("No matching career terms were found. Try describing your skills or goals in English with more detail.")
    profile = "\n".join(f"{label}: {value}." for label, value in zip(FIELDS, values) if value)
    scores = cosine_similarity(vectorizer.transform([profile]), matrix).ravel()
    if not np.isfinite(scores).all() or scores.max() <= 0:
        raise ValueError("No meaningful matches were found. Add more detail and try again.")
    indices = np.argsort(-scores, kind="stable")[:5]
    matches = careers.iloc[indices][["O*NET-SOC Code", "Title", "Description", "Education", "Experience"]].copy()
    matches["Similarity score"] = scores[indices]
    return matches.to_dict("records")


def details_html(match):
    def text(key):
        return escape(str(match.get(key) or "Not available in the dataset"))
    return f'''<article class="career-detail" role="region" aria-label="Selected career details" aria-live="polite">
      <h3>{text("Title")}</h3>
      <div class="soc-code">O*NET-SOC: {text("O*NET-SOC Code")}</div>
      <p>{text("Description")}</p>
      <div class="detail-grid"><div><span>Typical education</span><p>{text("Education")}</p></div>
      <div><span>Typical related experience</span><p>{text("Experience")}</p></div></div>
      <a href="https://www.onetonline.org/link/summary/{text("O*NET-SOC Code")}" target="_blank" rel="noopener noreferrer">Explore this career on O*NET ↗</a>
    </article>'''


def recommend(*values):
    try:
        matches = rank_careers(*values)
    except ValueError as error:
        return [], [], EMPTY_DETAILS, str(error)
    except Exception:
        logging.exception("Career model could not be loaded")
        return [], [], EMPTY_DETAILS, "The career model could not be loaded. Check the model files and setup instructions, then restart the app."
    rows = [[i + 1, item["Title"], round(item["Similarity score"], 4)] for i, item in enumerate(matches)]
    return matches, rows, details_html(matches[0]), "Found five matches. Select a career row to view its details."


def select_career(matches, event: gr.SelectData):
    if not matches:
        return EMPTY_DETAILS
    index = event.index[0] if isinstance(event.index, (tuple, list)) else event.index
    return details_html(matches[index]) if isinstance(index, int) and 0 <= index < len(matches) else EMPTY_DETAILS


def clear_profile():
    return None, "", "", "", "", "", [], [], EMPTY_DETAILS, "Profile cleared. Add your background to start again."


def recommend_pages(*values):
    result = recommend(*values)
    success = bool(result[0])
    return (*result, gr.update(visible=not success), gr.update(visible=success),
            gr.update(value="" if success else result[3], visible=not success))


def edit_profile():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


def clear_pages():
    return (*clear_profile(), *edit_profile())


def career_choices(matches):
    return gr.update(choices=[(f"{i + 1}. {match['Title']}", str(i)) for i, match in enumerate(matches)], value="0" if matches else None)


def choose_career(choice, matches):
    if choice is None or not matches:
        return EMPTY_DETAILS
    return details_html(matches[int(choice)])


CSS = '''
.gradio-container {max-width: 1000px !important; margin: auto;}
#header-row {background: #19334d; border-radius: 14px; padding: 24px; margin-bottom: 18px; align-items: center; gap: 20px;}
#app-header {color: #fff; background: transparent; padding: 0;}
#header-settings {background: transparent; border: 0; min-width: 0;}
#header-settings #color-mode {background: transparent; border: 0;}
#color-mode > label, #color-mode > span, #color-mode legend {color: #fff !important;}
#app-header h1 {overflow-wrap: anywhere;}
#app-header h1 {color: #fff; margin: 0 0 6px; font-size: 30px;}
#app-header p {color: #dce7ef; margin: 0; font-size: 16px;}
.screen-panel {background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 24px !important;}
.screen-heading h2 {font-size: 25px; margin-top: 0;}
.screen-heading strong {color: #00858b;}
#search-button {background: #006b70; color: #fff; border-color: #006b70;}
#search-button:hover {background: #006b70;}
.career-detail {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 20px; overflow-wrap: anywhere;}
.career-detail h3 {margin: 0 0 6px; font-size: 22px;}
.soc-code {font-size: 14px; color: var(--body-text-color-subdued);}
.detail-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 20px 0;}
.detail-grid span {font-weight: 600;}
.detail-grid p {margin-top: 5px;}
.career-detail a {color: var(--link-text-color); text-decoration: underline;}
.empty-details {padding: 30px 15px; color: var(--body-text-color-subdued); text-align: center;}
.gradio-container :is(button, input, textarea, select, a, h2):focus-visible {outline: 3px solid #005fcc !important; outline-offset: 3px;}
.dark .gradio-container :is(button, input, textarea, select, a, h2):focus-visible {outline-color: #ffd166 !important;}
.gradio-container button {min-height: 44px;}
.gradio-container input, .gradio-container textarea {font-size: 16px !important;}
.sr-only {position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap;}
.skip-link {position: absolute; left: 12px; top: -100px; z-index: 100; background: var(--block-background-fill); color: var(--body-text-color); padding: 12px;}
.skip-link:focus {top: 8px;}
@media (prefers-reduced-motion: reduce) {.gradio-container *, .gradio-container *::before, .gradio-container *::after {animation: none !important; transition: none !important; scroll-behavior: auto !important;}}
@media (forced-colors: active) {.gradio-container :focus-visible {outline: 3px solid Highlight !important;} #search-button {border: 2px solid ButtonText;}}
@media (max-width: 700px) {#app-header {padding: 20px;} #app-header h1 {font-size: 24px;} .screen-panel {padding: 16px !important;} .detail-grid {grid-template-columns: 1fr; gap: 8px;}}
'''


def app_theme():
    return gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["Arial", "sans-serif"]).set(
        body_text_color="#172b3a", body_text_color_dark="#f1f5f9",
        body_text_color_subdued="#475569", body_text_color_subdued_dark="#cbd5e1",
        block_label_text_color="#172b3a", block_label_text_color_dark="#f1f5f9",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        input_border_color="#64748b", input_border_color_dark="#94a3b8",
        button_primary_background_fill="#006b70", button_primary_background_fill_dark="#006b70",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    )


UI_JS = '''() => {
    window.careerTheme = (mode) => {
        const dark = mode === 'Dark' || (mode === 'System' && matchMedia('(prefers-color-scheme: dark)').matches);
        document.documentElement.classList.toggle('dark', dark);
        document.body.classList.toggle('dark', dark);
        document.querySelectorAll('gradio-app').forEach(el => el.classList.toggle('dark', dark));
        try { localStorage.setItem('career-color-mode', mode); } catch (_) {}
    };
    let saved = 'System';
    try { saved = localStorage.getItem('career-color-mode') || 'System'; } catch (_) {}
    window.careerTheme(saved);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        let mode = 'System';
        try { mode = localStorage.getItem('career-color-mode') || 'System'; } catch (_) {}
        if (mode === 'System') window.careerTheme(mode);
    });
    window.careerFocus = (page) => {
        requestAnimationFrame(() => {
            const panel = document.getElementById(page);
            const heading = panel?.querySelector('h2');
            if (heading) { heading.setAttribute('tabindex', '-1'); heading.focus(); }
            const error = document.querySelector('#profile-error textarea, #profile-error input');
            const status = document.querySelector('#recommendation-status textarea, #recommendation-status input');
            const live = document.getElementById('career-live');
            if (live) live.textContent = page === 'results-page' ? (status?.value || 'Career matches ready') : (error?.value || 'Your profile');
        });
    };
}'''


def build_app():
    with gr.Blocks(title="Career Recommendation Assistant", analytics_enabled=False) as demo:
        gr.HTML('<a class="skip-link" href="#main-content">Skip to main content</a><div id="career-live" class="sr-only" role="status" aria-live="polite" aria-atomic="true"></div>')
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=2, min_width=280):
                gr.HTML('<header id="app-header"><h1>Career Recommendation Assistant</h1><p>Explore careers that fit your background</p></header>')
            with gr.Column(scale=1, min_width=260, elem_id="header-settings"):
                mode = gr.Radio(["System", "Light", "Dark"], value="System", label="Color mode", elem_id="color-mode")
        demo.load(None, outputs=mode, js="() => { try { const value = localStorage.getItem('career-color-mode'); return ['Light','Dark','System'].includes(value) ? value : 'System'; } catch (_) { return 'System'; } }")
        mode.change(None, inputs=mode, js="(mode) => { window.careerTheme?.(mode); }", queue=False)
        gr.Markdown("Use Tab to move between controls and Shift+Tab to go back. No personal identifiers are needed. Add at least one description of your skills, interests, experience, or goals.")
        matches = gr.State([])
        with gr.Row(equal_height=True, elem_id="main-content"):
            with gr.Column(min_width=280, elem_classes="screen-panel", elem_id="profile-page") as profile_page:
                gr.Markdown("## **01 /** Your profile\nTell us about your background, skills, and goals.", elem_classes="screen-heading")
                with gr.Row():
                    education = gr.Dropdown(["Less than high school", "High school diploma or equivalent", "Some college", "Associate degree", "Bachelor's degree", "Master's degree", "Doctoral degree", "Professional degree", "Other"], label="Education level", info="Choose your highest completed level.")
                    experience = gr.Textbox(label="Work experience", placeholder="2 years in business reporting", lines=2)
                with gr.Row():
                    skills = gr.Textbox(label="Skills", placeholder="Python, Excel, statistics", lines=3)
                    interests = gr.Textbox(label="Interests", placeholder="Data analysis and problem solving", lines=3)
                with gr.Row():
                    goals = gr.Textbox(label="Career goals or preferred area", placeholder="Explore analytics roles", lines=3)
                    additional = gr.Textbox(label="Additional information (optional)", placeholder="Other relevant background", lines=3)
                inputs = [education, skills, interests, experience, goals, additional]
                profile_error = gr.Textbox(label="Please update your profile", value="", visible=False, interactive=False, elem_id="profile-error")
                with gr.Row():
                    search = gr.Button("Find career matches", variant="primary", elem_id="search-button")
                    clear = gr.Button("Clear profile")
                gr.Examples(examples=[["Bachelor's degree", "Python, Excel, statistics", "Data analysis and problem solving", "2 years in business reporting", "Explore analytics roles", ""], ["High school diploma or equivalent", "Customer service, communication, organization", "Helping people and coordinating tasks", "3 years in retail", "Explore office and service roles", ""]], inputs=inputs, label="Try an example profile")
            with gr.Column(min_width=280, elem_classes="screen-panel", elem_id="results-page", visible=False) as results_page:
                gr.Markdown("## **02 /** Career matches\nYour top five matches will appear below.", elem_classes="screen-heading")
                status = gr.Textbox(value="Add your background and select Find career matches.", label="Recommendation status", interactive=False, lines=2, elem_id="recommendation-status")
                table = gr.Dataframe(headers=HEADERS, datatype=["number", "str", "number"], value=[], interactive=False, label="Top 5 career matches", wrap=True, column_widths=[60, "60%", 150], buttons=[])
                picker = gr.Radio(choices=[], label="Choose a career to read its details", value=None)
                details = gr.HTML(EMPTY_DETAILS)
                gr.Markdown("Similarity scores measure text matches, **not career success probabilities**. Education and experience describe typical O*NET responses; they are not eligibility checks. Revise your profile to explore other options.")
                back = gr.Button("← Edit profile")
        outputs = [matches, table, details, status]
        pages = [profile_page, results_page, profile_error]
        searched = search.click(recommend_pages, inputs, outputs + pages, api_name="recommend")
        searched.then(career_choices, matches, picker, api_name=False).then(None, inputs=profile_error, js="(error) => { window.careerFocus?.(error ? 'profile-page' : 'results-page'); }", queue=False)
        picker.change(choose_career, [picker, matches], details, api_name=False)
        table.select(select_career, [matches], details, api_name=False)
        clear.click(clear_pages, outputs=inputs + outputs + pages, api_name="clear_profile").then(career_choices, matches, picker, api_name=False).then(None, js="() => { window.careerFocus?.('profile-page'); }", queue=False)
        back.click(edit_profile, outputs=pages, api_name="edit_profile").then(None, js="() => { window.careerFocus?.('profile-page'); }", queue=False)
    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "7860")), theme=app_theme(), css=CSS, js=UI_JS)
