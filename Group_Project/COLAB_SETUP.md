# Google Colab Setup

## 1. Clone the repository

```python
!git clone https://github.com/lunchtrey84/MSAI_631_REPO.git
%cd MSAI_631_REPO/Group_Project
```

If the repository is private, authenticate Colab with GitHub before cloning.
Never paste a token into a notebook that will be submitted or shared.

## 2. Install dependencies

```python
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

Restart the Colab runtime if package installation requests it, then return to
the project directory.

## 3. Run automated tests

```python
!python -m pytest -q
```

## 4. Launch the interface

```python
%env CAREER_APP_SHARE=true
!python app.py
```

Open the public Gradio link printed by the final command. Keep the cell running
while testing the application.

## 5. Suggested test profile

- Education: Master's degree
- Skills: leadership, communication, planning, training, critical thinking
- Interests: technology, education, and management
- Experience: more than ten years of professional experience
- Career goals: leadership, management, training, and technology

Run the same profile with MiniLM and TF-IDF and record the differences. Do not
interpret the similarity score as a probability of career success.
